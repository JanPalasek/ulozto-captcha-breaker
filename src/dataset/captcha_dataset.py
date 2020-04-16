import cv2
from typing import List
import random

TRAIN = 0
TEST = 1


class CaptchaDataset:
    def __init__(self, annotations_path: str, train_perc: float, dev_perc: float):
        self._annotations_path = annotations_path

        self._data = self._get_items()
        self._train_perc = train_perc
        self._dev_perc = dev_perc

    def get_image_shape(self):
        return self._data[0][0].shape

    def get_classes(self):
        # TODO
        return 10

    def _get_items(self):
        result = []
        with open(self._annotations_path, "r") as file:
            for line in file:
                image_path, image_label = line.split()
                image_label = list(image_label)

                image = cv2.imread(image_path, 0)
                result.append((image, image_label))

        return result

    def get_data(self):
        data = {}

        train_data_len = int(self._train_perc * len(self._data))
        dev_data_len = int(self._dev_perc * len(self._data))
        random.shuffle(self._data)

        train_data, train_labels = list(zip(*self._data[:train_data_len]))
        dev_data, dev_labels = list(zip(*self._data[train_data_len:train_data_len + dev_data_len]))
        test_data, test_labels = list(zip(*self._data[train_data_len + dev_data_len:]))

        data["train"] = {"data": train_data, "labels": train_labels}
        data["dev"] = {"data": dev_data, "labels": dev_labels}
        data["test"] = {"data": test_data, "labels": test_labels}

        return data
