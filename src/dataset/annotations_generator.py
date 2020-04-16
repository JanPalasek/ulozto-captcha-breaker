import argparse
import datetime
import os
import re

import cv2


class AnnotationsGenerator:
    def __init__(self, dir_path: str):
        self._dir_path = dir_path

    def get_annotations(self):
        for item in os.listdir(self._dir_path):
            item_path = os.path.join(self._dir_path, item)

            yield item_path, os.path.splitext(item)[0]

    def save_annotations(self, path: str):
        with open(path, "w") as file:
            for image_path, label in self.get_annotations():
                file.write(f"{image_path} {label}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--out_dir", default="../out", type=str, help="Out dir")

    args = parser.parse_args()

    out_dir = args.out_dir
    data_dir = os.path.join(out_dir, "data")
    annotations_dir = os.path.join(out_dir, "annotations.txt")

    args.logdir = os.path.join(out_dir, "logs", "{}-{}-{}".format(
        os.path.basename(__file__),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
    ))

    generator = AnnotationsGenerator(data_dir)
    generator.save_annotations(annotations_dir)