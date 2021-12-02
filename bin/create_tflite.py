import sys

import tensorflow as tf
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="out", type=str, help="Out dir")
    parser.add_argument("--pretrained_model", type=str, required=True)
    args = parser.parse_args()

    converter = tf.lite.TFLiteConverter.from_saved_model(args.pretrained_model)
    tflite_model = converter.convert()

    # Save the model.
    with open(os.path.join(args.out_dir, 'model.tflite'), 'wb') as f:
        f.write(tflite_model)
