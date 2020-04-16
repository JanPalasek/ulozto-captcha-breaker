import numpy as np


class ImageCutPreprocessor:
    def __init__(self, pieces_count: int):
        self._pieces_count = pieces_count

    def preprocess(self, image: np.ndarray):
        images = np.split(image, self._pieces_count, axis=1)

        return np.array(images)