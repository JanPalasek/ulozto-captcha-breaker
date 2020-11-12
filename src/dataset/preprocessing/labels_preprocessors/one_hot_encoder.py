import tensorflow as tf
from typing import List


class OneHotEncoder:
    def __init__(self, available_chars):
        self._available_chars = available_chars

    def encode_char(self, char: str):
        return tf.one_hot(self._available_chars.index(char), len(self._available_chars))

    def decode_char(self, one_hot_vector):
        index = tf.argmax(one_hot_vector, axis=0)
        return self._available_chars[index]