import numpy as np

from ulozto_captcha_breaker.dataset.preprocessing.label_preprocessors import StringEncoder


class LabelPreprocessPipeline:
    def __init__(self, encoder):
        self._encoder = encoder

    def __call__(self, labels):
        result = []
        for label in labels:
            result.append(self._encoder.encode(label))

        return np.array(result)