import numpy as np


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

    # def encode(self, input):
    #     result = []
    #     for x in input:
    #         result.append(self.encode_str(x))
    #     return np.array(result)