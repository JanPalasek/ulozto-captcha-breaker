import argparse
import datetime
import os
import random
import re

import cv2
import numpy as np


class AnnotationsGenerator:
    def __init__(self, dir_path: str, annotations_out_dir: str, validation_ratio: float, test_ratio: float, ignore_case: bool):
        self._dir_path = dir_path
        self._validation_ratio = validation_ratio
        self._test_ratio = test_ratio
        self._ignore_case = ignore_case
        self._annotations_out_dir = annotations_out_dir

    def get_annotations(self):
        for item in os.listdir(self._dir_path):
            item_path = os.path.join(self._dir_path, item)

            item_label = os.path.splitext(item)[0]
            item_label = item_label.split("_")[0]

            yield item_path, item_label

    def save_annotations(self):
        val_annotations_path = os.path.join(self._annotations_out_dir, "annotations-validation.txt")
        test_annotations_path = os.path.join(self._annotations_out_dir, "annotations-test.txt")
        train_annotations_path = os.path.join(self._annotations_out_dir, "annotations-train.txt")
        annotations_path = os.path.join(self._annotations_out_dir, "annotations.txt")

        annotations = np.array(list(self.get_annotations()))
        indices = list(range(len(annotations)))
        random.shuffle(indices)

        test_samples_count = int(len(indices) * self._test_ratio)
        validation_samples_count = int(len(indices) * self._validation_ratio)
        test_indices = indices[:test_samples_count]
        validation_indices = indices[test_samples_count:test_samples_count + validation_samples_count]
        train_indices = indices[test_samples_count + validation_samples_count:]

        test_annotations = annotations[test_indices]
        train_annotations = annotations[train_indices]
        validation_annotations = annotations[validation_indices]

        with open(annotations_path, "w") as annotations_file:
            with open(test_annotations_path, "w") as file:
                for image_path, label in test_annotations:
                    annotation = f"{image_path} {label.lower()}\n"

                    file.write(annotation)
                    annotations_file.write(annotation)

            with open(val_annotations_path, "w") as file:
                for image_path, label in validation_annotations:
                    annotation = f"{image_path} {label.lower()}\n"

                    file.write(annotation)
                    annotations_file.write(annotation)

            with open(train_annotations_path, "w") as file:
                for image_path, label in train_annotations:
                    annotation = f"{image_path} {label.lower()}\n"

                    file.write(annotation)
                    annotations_file.write(annotation)