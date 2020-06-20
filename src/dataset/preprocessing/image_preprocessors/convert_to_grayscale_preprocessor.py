import cv2
import numpy as np


class ConvertToGrayscalePreprocessor:
    """
    Converts image to grayscale.
    """
    def __call__(self, img: np.ndarray):
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)