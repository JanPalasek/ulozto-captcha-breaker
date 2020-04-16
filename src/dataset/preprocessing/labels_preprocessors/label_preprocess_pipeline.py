import numpy as np

from dataset.preprocessing.labels_preprocessors.string_encoder import StringEncoder


class LabelPreprocessPipeline:
    def __init__(self, encoder: StringEncoder):
        self._encoder = encoder

    def __call__(self, labels):
        result = []
        for label in labels:
            result.append(self._encoder.encode(label))

        return np.array(result)