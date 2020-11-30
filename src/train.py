from accuracy.correctly_classified_captcha_accuracy import all_correct_acc

import numpy as np
import random

from dataset.captcha_dataset import CaptchaDataset

import argparse
import datetime
import os
import re
import tensorflow as tf

from model import ResNet

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights_file", default=None, type=str, help="Path to file that contains pre-trained weights.")
    parser.add_argument("--pretrained_model", default=None, type=str)
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
    parser.add_argument("--epochs", default=100, type=int, help="Number of epochs.")
    parser.add_argument("--out_dir", default="../out", type=str, help="Out dir")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--captcha_length", default=4, type=int)
    parser.add_argument("--available_chars", default="abcdefghijklmnopqrstuvwxyz", type=str, help="Labels")
    parser.add_argument("--image_height", default=70, type=int)
    parser.add_argument("--image_width", default=175, type=int)
    parser.add_argument("--l2", default=1e-4, type=float)
    parser.add_argument("--eager_load_data", action="store_true", default=False, help="Must be set true if using tensorflow 2.0. Generator bug.")

    args = parser.parse_args()

    args.save_model_path = None

    assert args.weights_file is None or args.pretrained_model is None, "Cannot load pretrained model and weights file at the same time"

    # Fix random seeds and number of threads
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    random.seed(args.seed)

    if tf.test.gpu_device_name():
        print('Default GPU Device:{}'.format(tf.test.gpu_device_name()))
    else:
        print("No GPU available")

    out_dir = os.path.abspath(args.out_dir)
    data_dir = os.path.join(out_dir, "data")
    train_annotations_path = os.path.join(out_dir, "annotations-train.txt")
    val_annotations_path = os.path.join(out_dir, "annotations-validation.txt")

    args.logdir = os.path.join(out_dir, "logs", "{}-{}-{}".format(
        os.path.basename(__file__),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
    ))

    classes = len(args.available_chars)

    input_shape = (args.image_height, args.image_width, 1)

    train_dataset = tf.data.Dataset.from_generator(CaptchaDataset(train_annotations_path, args.available_chars).get_data, (tf.float32, tf.int32),
                                                   output_shapes=(input_shape, (args.captcha_length,)))
    val_dataset = tf.data.Dataset.from_generator(CaptchaDataset(val_annotations_path, args.available_chars).get_data, (tf.float32, tf.int32),
                                                   output_shapes=(input_shape, (args.captcha_length,)))

    model = ResNet(input_shape=input_shape, output_shape=(args.captcha_length, classes), init_filters=32, l2=args.l2)

    metrics = [tf.keras.metrics.sparse_categorical_accuracy]
    if not args.save_model_path:
        metrics.append(all_correct_acc)
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=metrics)

    model.summary()

    if args.save_model_path:
        model.save_model(args.save_model_path)

    if args.weights_file is not None:
        model.load_weights(args.weights_file)

    if args.eager_load_data:
        train_x, train_y = list(zip(*[(x.numpy(), y.numpy()) for x, y in train_dataset]))
        train_x, train_y = np.array(train_x), np.array(train_y)
        val_x, val_y = list(zip(*[(x.numpy(), y.numpy()) for x, y in val_dataset]))
        val_x, val_y = np.array(val_x), np.array(val_y)

        model.fit(x=train_x, y=train_y, epochs=args.epochs,
                  batch_size=args.batch_size,
                  validation_data=(val_x, val_y),
                  callbacks=[
                      tf.keras.callbacks.TensorBoard(args.logdir),
                      tf.keras.callbacks.ModelCheckpoint(
                          os.path.join(args.logdir, 'cp-{epoch:02d}.h5'), save_weights_only=True)
                  ])
    else:
        model.fit(train_dataset.shuffle(1000).batch(args.batch_size), epochs=args.epochs,
                  validation_data=val_dataset.batch(args.batch_size),
                  callbacks=[
                      tf.keras.callbacks.TensorBoard(args.logdir),
                      tf.keras.callbacks.ModelCheckpoint(
                          os.path.join(args.logdir, 'cp-{epoch:02d}.h5'), save_weights_only=True)
                  ])
