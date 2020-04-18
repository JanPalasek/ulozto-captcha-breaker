import unittest
import numpy as np

from src.accuracy.correctly_classified_captcha_accuracy import CorrectlyClassifiedCaptchaAccuracy


class CorrectlyClassifiedCaptchaAccuracyTestCase(unittest.TestCase):
    def setUp(self):
        self._acc = CorrectlyClassifiedCaptchaAccuracy()

    def test_happy_path(self):
        logits = np.array([[[0.6, 0.2, 0.2], [0.1, 0.8, 0.1],  [0.1, 0.1, 0.8], [0.1, 0.8, 0.1]]])
        labels = np.array([[0, 1, 2, 1]])

        self._acc.update_state(labels, logits)

        self.assertEqual(1, self._acc.result())

    def test_fail(self):
        logits = np.array([[[0.2, 0.7, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8], [0.1, 0.8, 0.1]]])
        labels = np.array([[0, 1, 2, 1]])

        self._acc.update_state(labels, logits)

        self.assertEqual(0, self._acc.result())

    def test_succeed_fail(self):
        logits = np.array([[[0.6, 0.2, 0.2], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8], [0.1, 0.8, 0.1]]])
        labels = np.array([[0, 1, 2, 1]])
        self._acc.update_state(labels, logits)

        logits = np.array([[[0.2, 0.7, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8], [0.1, 0.8, 0.1]]])
        labels = np.array([[0, 1, 2, 1]])
        self._acc.update_state(labels, logits)

        result = self._acc.result()

        self.assertEqual(0.5, result)



if __name__ == '__main__':
    unittest.main()
