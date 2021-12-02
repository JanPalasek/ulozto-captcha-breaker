import numpy as np
from PIL import Image


class ConvertToGrayscalePreprocessor:
    """
    Converts image to grayscale.
    """
    def __call__(self, img: np.ndarray):
        r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        output = 0.299 * r + 0.587 * g + 0.114 * b
        return output


class ImageCutPreprocessor:
    def __init__(self, pieces_count: int):
        self._pieces_count = pieces_count

    def __call__(self, image: np.ndarray):
        images = np.split(image, self._pieces_count, axis=1)

        return np.array(images)


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


class ResizePreprocessor:
    """
    Resizes image to target width and height.
    """
    def __init__(self, target_height, target_width):
        self._target_height = target_height
        self._target_width = target_width

    def __call__(self, img: np.ndarray):
        return img.resize(img, (self._target_width, self._target_height))
