import cv2
import numpy as np


class ResizePreprocessor:
    def __init__(self, target_height, target_width):
        self._target_height = target_height
        self._target_width = target_width

    def __call__(self, img: np.ndarray):
        return cv2.resize(img, (self._target_width, self._target_height))
