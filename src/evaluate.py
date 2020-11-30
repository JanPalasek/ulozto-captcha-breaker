from accuracy.correctly_classified_captcha_accuracy import all_correct_acc
from dataset.captcha_dataset import CaptchaDataset
from dataset.string_encoder import StringEncoder
from model import ResNet


import numpy as np
import random

import argparse
import datetime
import os
import re
import tensorflow as tf

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights_file", default=None, type=str,
                        help="Path to file that contains pre-trained weights.")
    parser.add_argument("--pretrained_model", default=None, type=str)
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
    parser.add_argument("--out_dir", default="../out", type=str, help="Out dir")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--captcha_length", default=4, type=int)
    parser.add_argument("--available_chars", default="abcdefghijklmnopqrstuvwxyz", type=str, help="Labels")
    parser.add_argument("--image_height", default=70, type=int)
    parser.add_argument("--image_width", default=175, type=int)
    parser.add_argument("--l2", default=0.0001, type=float)

    args = parser.parse_args()

    # Fix random seeds and number of threads
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    random.seed(args.seed)

    out_dir = os.path.abspath(args.out_dir)
    data_dir = os.path.join(out_dir, "data")
    test_annotations_path = os.path.join(out_dir, "annotations-test.txt")

    args.logdir = os.path.join(out_dir, "logs", "{}-{}-{}".format(
        os.path.basename(__file__),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
    ))
    args.save_model_path = os.path.join(out_dir, "model")

    classes = len(args.available_chars)

    input_shape = (args.image_height, args.image_width, 1)

    model = ResNet(input_shape=input_shape, output_shape=(args.captcha_length, classes),
                   init_filters=32, l2=args.l2)

    metrics = [tf.keras.metrics.sparse_categorical_accuracy]
    if not args.save_model_path:
        metrics.append(all_correct_acc)
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=metrics)

    model.summary()

    test_dataset = tf.data.Dataset.from_generator(CaptchaDataset(test_annotations_path, args.available_chars).get_data,
                                                 output_types=(tf.float32, tf.int32),
                                                 output_shapes=(input_shape,
                                                                (args.captcha_length,)))

    enc = StringEncoder(available_chars=args.available_chars)
    labels = [enc.encode(y) for _, y in test_dataset]
    pred_labels = model.predict(test_dataset.map(lambda x, y: x).batch(args.batch_size))

    correct = labels == pred_labels

    all_correct = tf.reduce_all(correct, axis=1)
    all_correct = tf.cast(all_correct, tf.dtypes.float32)
    acc = tf.reduce_mean(all_correct)

    dec = enc
    with open(os.path.join(out_dir, "out_test.csv"), "w") as file:
        for i in range(len(pred_labels)):
            decoded_label = dec.decode(labels[i])
            decoded_pred_label = dec.decode(pred_labels[i])
            file.write(f"{all_correct[i]};{decoded_label};{decoded_pred_label}\n")

    print(acc)
