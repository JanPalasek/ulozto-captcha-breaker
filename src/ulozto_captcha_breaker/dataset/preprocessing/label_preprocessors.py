import numpy as np
import tensorflow as tf
from typing import List


class OneCharEncoder:
    """
    Encodes chars into integers.
    """
    def __init__(self, available_chars):
        self._available_chars = available_chars

    def encode_char(self, char: str):
        return self._available_chars.index(char)

    def encode(self, string):
        result = []
        result.append(self.encode_char(string[1]))
        return np.array(result)

    # def encode(self, input):
    #     result = []
    #     for x in input:
    #         result.append(self.encode_str(x))
    #     return np.array(result)



class OneHotEncoder:
    def __init__(self, available_chars):
        self._available_chars = available_chars

    def encode_char(self, char: str):
        return tf.one_hot(self._available_chars.index(char), len(self._available_chars))

    def decode_char(self, one_hot_vector):
        index = tf.argmax(one_hot_vector, axis=0)
        return self._available_chars[index]


class StringEncoder:
    """
    Encodes chars into integers.
    """
    def __init__(self, available_chars):
        self._available_chars = available_chars

    def encode_char(self, char: str):
        return self._available_chars.index(char)

    def encode(self, string):
        result = []
        for char in string:
            result.append(self.encode_char(char))
        return np.array(result)

    def decode_char(self, char_idx: int):
        return self._available_chars[char_idx]

    def decode(self, li):
        result = []
        for char in li:
            result.append(self.decode_char(char))
        return "".join(result)

    # def encode(self, input):
    #     result = []
    #     for x in input:
    #         result.append(self.encode_str(x))
    #     return np.array(result)