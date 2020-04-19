import argparse
import datetime
import os
import random
import re

import cv2
import numpy as np


class AnnotationsGenerator:
    def __init__(self, dir_path: str, test_ratio: float, ignore_case: bool):
        self._dir_path = dir_path
        self._test_ratio = test_ratio
        self._ignore_case = ignore_case

    def get_annotations(self):
        for item in os.listdir(self._dir_path):
            item_path = os.path.join(self._dir_path, item)

            item_label = os.path.splitext(item)[0]
            item_label = item_label.split("_")[0]

            yield item_path, item_label

    def save_annotations(self, train_path: str, test_path: str):
        annotations = np.array(list(self.get_annotations()))
        indices = list(range(len(annotations)))
        random.shuffle(indices)

        test_samples_count = int(len(indices) * self._test_ratio)
        test_indices = indices[:test_samples_count]
        train_indices = indices[test_samples_count:]

        test_annotations = annotations[test_indices]
        train_annotations = annotations[train_indices]

        with open(test_path, "w") as file:
            for image_path, label in test_annotations:
                file.write(f"{image_path} {label.lower()}\n")

        with open(train_path, "w") as file:
            for image_path, label in train_annotations:
                file.write(f"{image_path} {label.lower()}\n")