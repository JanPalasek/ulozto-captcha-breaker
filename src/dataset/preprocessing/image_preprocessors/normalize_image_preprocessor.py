import numpy as np


class NormalizeImagePreprocessor:
    def __init__(self):
        pass

    def preprocess(self, image):
        image = image.astype(np.float32) / 255
        image = np.expand_dims(image, axis=len(image.shape))
        return image