import numpy as np


class NormalizeImagePreprocessor:
    """
    Converts image from byte format (values are integers in {0, ..., 255} to normalized float format (values are
    floats in the interval [0, 1].
    """
    def __init__(self):
        pass

    def __call__(self, image):
        image = image.astype(np.float32) / 255
        image = np.expand_dims(image, axis=len(image.shape))
        return image