import tensorflow as tf
import numpy as np


class CaptchaDataset:
    def __init__(self, annotations_path: str, available_chars):
        self._annotations_path = annotations_path
        self._available_chars = available_chars

        self._image_shape = next(self.get_data())[0]

    def get_image_shape(self):
        return self._image_shape

    def get_data(self):
        with open(self._annotations_path, "r") as file:
            for line in file:
                image_path, image_label = line.rsplit(maxsplit=1)

                image = tf.keras.preprocessing.image.load_img(image_path, color_mode='rgb')
                image = tf.keras.preprocessing.image.img_to_array(image)
                image = image / 255
                image = 0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]
                image = tf.expand_dims(image, axis=-1)

                indices = []
                for c in image_label:
                    idx = self._available_chars.index(c)
                    indices.append(idx)
                indices = tf.convert_to_tensor(indices)

                yield image, indices
