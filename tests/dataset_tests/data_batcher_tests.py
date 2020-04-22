import unittest

import numpy as np

from dataset.data_batcher import DataBatcher


class DataBatcherTests(unittest.TestCase):
    def test_something(self):
        inputs = np.array([2, 3, 0, 5])
        labels = np.array([1, 1, 8, 1])

        self._data_batcher = DataBatcher(batch_size=3, inputs=inputs, labels=labels)
        batches = list(self._data_batcher.batches())

        self.assertEqual(2, len(batches))


if __name__ == '__main__':
    unittest.main()
