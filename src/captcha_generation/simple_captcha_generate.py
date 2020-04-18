import argparse
import os

import sys
sys.path.insert(0, "src")

from captcha.image import ImageCaptcha

from dataset.annotations_generator import AnnotationsGenerator

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--test_split", default=0.1, type=float)
    args = parser.parse_args()

    out_dir = os.path.abspath("out")
    data_dir = os.path.join(out_dir, "data")

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    image = ImageCaptcha()

    for i in range(0, 10000):
        image.write(f'{i:>04}', f'{data_dir}/{i:>04}.png')

    train_annotations_path = os.path.join(out_dir, "annotations-train.txt")
    test_annotations_path = os.path.join(out_dir, "annotations-test.txt")

    generator = AnnotationsGenerator(data_dir, args.test_split)
    generator.save_annotations(train_annotations_path, test_annotations_path)