import numpy as np

from dataset.preprocessing.image_preprocessors.image_cut_preprocessor import ImageCutPreprocessor


class ImagePreprocessorPipeline:
    def __init__(self,  preprocessors=[]):
        self._preprocessors = preprocessors

    def __call__(self, images):
        result = []
        for image in images:
            modified_image = np.copy(image)

            for p in self._preprocessors:
                modified_image = p.preprocess(modified_image)

            result.append(modified_image)
        return np.array(result)
