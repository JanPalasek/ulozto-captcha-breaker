import random

import numpy as np


class DataBatcher:
    def __init__(self, batch_size: int, inputs, labels):
        self._inputs = inputs
        self._labels = labels
        self._batch_size = batch_size

    def batches(self):
        # TODO: currently ignoring last batch
        inputs = np.copy(self._inputs)
        labels = np.copy(self._labels)
        indices = list(range(len(self._inputs)))
        random.shuffle(indices)

        inputs = inputs[indices]
        labels = labels[indices]

        # TODO: shuffle
        for batch_idx in range(len(self._inputs) // self._batch_size):
            first_item_idx = batch_idx * self._batch_size

            batch_inputs = inputs[first_item_idx:first_item_idx + self._batch_size]
            batch_labels = labels[first_item_idx:first_item_idx + self._batch_size]

            yield (batch_inputs,
                   batch_labels)
