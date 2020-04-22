import os
import datetime
import cv2

import logging

import numpy as np


class FileWriter:
    def __init__(self, path):
        self._path = path
        self._internal_counter = 0

        if not os.path.exists(path):
            os.makedirs(path)

    def save_image(self, img: np.ndarray, name=None, category="debug"):
        img = np.copy(img)
        if name is None:
            # timestamp
            name = datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S.%f') + str(self._internal_counter)

            self._internal_counter += 1

        dest = self._path + os.sep + "{}_{}.png".format(category, name)

        if img.dtype in [np.float32, np.float64]:
            img = img * 255
            img = img.astype(np.uint8)
        cv2.imwrite(dest, img)
