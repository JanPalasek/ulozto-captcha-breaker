import sys
sys.path.insert(0, "src")

import numpy as np
import random

from captcha_detection.captcha_network import CaptchaNetwork
from dataset.captcha_dataset import CaptchaDataset

import argparse
import datetime
import os
import re
import tensorflow as tf

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights_file", default=None, type=str, help="Path to file that contains pre-trained weights.")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
    parser.add_argument("--epochs", default=1500, type=int, help="Number of epochs.")
    parser.add_argument("--out_dir", default="out", type=str, help="Out dir")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--val_split", default=0.1, type=float)
    parser.add_argument("--checkpoint_freq", default=4, type=int, help="How frequently will be model saved."
                                                                           "E.g. if 4, then every fourth epoch will be stored.")

    args = parser.parse_args()

    # Fix random seeds and number of threads
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    random.seed(args.seed)

    out_dir = os.path.abspath(args.out_dir)
    data_dir = os.path.join(out_dir, "data")
    annotations_path = os.path.join(out_dir, "annotations-train.txt")

    args.logdir = os.path.join(out_dir, "logs", "{}-{}-{}".format(
        os.path.basename(__file__),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
    ))

    dataset = CaptchaDataset(annotations_path)
    image_shape = dataset.get_image_shape()
    classes = dataset.get_classes()
    inputs, labels = dataset.get_data()

    network = CaptchaNetwork(image_shape=image_shape,
                             classes=classes,
                             args=args)

    network.train(inputs, labels, args)
