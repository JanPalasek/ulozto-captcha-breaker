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

    def decode_char(self, char_idx: int):
        return self._available_chars[char_idx]

    def decode(self, li):
        result = []
        for char in li:
            result.append(self.decode_char(char))
        return "".join(result)