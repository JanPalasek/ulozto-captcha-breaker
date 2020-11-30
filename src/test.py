import sys
sys.path.insert(0, "src")

from dataset.preprocessing.image_preprocessors.convert_to_grayscale_preprocessor import ConvertToGrayscalePreprocessor
from dataset.preprocessing.image_preprocessors.resize_preprocessor import ResizePreprocessor

from dataset.preprocessing.image_preprocessors.image_preprocessor_pipeline import ImagePreprocessorPipeline
from dataset.preprocessing.image_preprocessors.normalize_image_preprocessor import NormalizeImagePreprocessor

from dataset.preprocessing.labels_preprocessors.label_preprocess_pipeline import LabelPreprocessPipeline
from dataset.preprocessing.labels_preprocessors.string_encoder import StringEncoder

import numpy as np
import random

from captcha_network import CaptchaNetwork
from dataset.captcha_dataset import CaptchaDataset

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
    parser.add_argument("--freeze_layers", default=0, type=int,
                        help="How many layers should be frozen for the training."
                             "Counts from the beginning.")
    parser.add_argument("--remove_layers",
                        action="store_true")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
    parser.add_argument("--epochs", default=1500, type=int, help="Number of epochs.")
    parser.add_argument("--out_dir", default="out", type=str, help="Out dir")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--captcha_length", default=4, type=int)
    parser.add_argument("--available_chars", default="abcdefghijklmnopqrstuvwxyz", type=str, help="Labels")
    parser.add_argument("--transformed_img_width", default=None, type=int)
    parser.add_argument("--transformed_img_height", default=None, type=int)
    parser.add_argument("--l2", default=0.0001, type=float)

    args = parser.parse_args()

    # Fix random seeds and number of threads
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    random.seed(args.seed)

    out_dir = os.path.abspath(args.out_dir)
    data_dir = os.path.join(out_dir, "data")
    annotations_path = os.path.join(out_dir, "annotations-test.txt")

    args.logdir = os.path.join(out_dir, "logs", "{}-{}-{}".format(
        os.path.basename(__file__),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
    ))
    args.save_model_path = os.path.join(out_dir, "model")

    dataset = CaptchaDataset(annotations_path, len(args.available_chars))
    inputs, labels = dataset.get_data()

    if args.transformed_img_width is not None and args.transformed_img_height is not None:
        input_shape = (args.transformed_img_height, args.transformed_img_width)
    else:
        image_shape = dataset.get_image_shape()
        input_shape = (image_shape[0], image_shape[1])

    image_preprocess_pipeline = ImagePreprocessorPipeline([
        ConvertToGrayscalePreprocessor(),
        ResizePreprocessor(input_shape[0], input_shape[1]),
        NormalizeImagePreprocessor()
    ])
    label_preprocess_pipeline = LabelPreprocessPipeline(
        StringEncoder(available_chars=args.available_chars)
    )

    network = CaptchaNetwork(image_shape=input_shape,
                             classes=dataset.classes,
                             image_preprocess_pipeline=image_preprocess_pipeline,
                             label_preprocess_pipeline=label_preprocess_pipeline,
                             args=args)

    labels = label_preprocess_pipeline(labels)

    pred_labels = network.predict(inputs, args)

    correct = labels == pred_labels

    all_correct = tf.reduce_all(correct, axis=1)
    all_correct = tf.cast(all_correct, tf.dtypes.float32)
    acc = tf.reduce_mean(all_correct)

    dec = StringEncoder(available_chars=args.available_chars)
    with open(os.path.join(out_dir, "out_test.csv"), "w") as file:
        for i in range(len(pred_labels)):
            decoded_label = dec.decode(labels[i])
            decoded_pred_label = dec.decode(pred_labels[i])
            file.write(f"{all_correct[i]};{decoded_label};{decoded_pred_label}\n")

    print(acc)
