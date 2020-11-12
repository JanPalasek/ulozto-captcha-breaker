import numpy as np

from utils.file_writer import FileWriter


class ImagePreprocessorPipeline:
    """
    ImagePreprocessorPipeline specifies list of operations that are performed on each image in specified order. These
    operations are invoked using *__call__*.
    """
    def __init__(self, preprocessors, out_writer: FileWriter = None, debug_writer: FileWriter = None):
        self._preprocessors = preprocessors

        self._out_writer = out_writer
        self._debug_writer = debug_writer

    def __call__(self, images):
        """
        Performs specified operations on images.
        :param images: List of images.
        :return: Modified list of images.
        """
        result = []
        for image in images:
            modified_image = np.copy(image)

            for p in self._preprocessors:
                modified_image = p(modified_image)
                if self._debug_writer is not None:
                    self._debug_writer.save_image(modified_image, category=str(type(p).__name__).lower())

            result.append(modified_image)

            if self._out_writer is not None:
                self._out_writer.save_image(modified_image, category="final")

        return np.array(result)

