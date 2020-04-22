import random

import numpy as np


class DataBatcher:
    def __init__(self, batch_size: int, inputs, labels):
        self._inputs = inputs
        self._labels = labels
        self._batch_size = batch_size

    def batches(self):
        inputs = np.copy(self._inputs)
        labels = np.copy(self._labels)
        indices = list(range(len(self._inputs)))
        random.shuffle(indices)

        inputs = inputs[indices]
        labels = labels[indices]

        whole_batches_count = len(self._inputs) // self._batch_size
        has_remaining_batch = (len(self._inputs) % self._batch_size) > 0
        for batch_idx in range(whole_batches_count):
            first_item_idx = batch_idx * self._batch_size

            batch_inputs = inputs[first_item_idx:first_item_idx + self._batch_size]
            batch_labels = labels[first_item_idx:first_item_idx + self._batch_size]

            yield (batch_inputs,
                   batch_labels)

        if has_remaining_batch:
            first_item_idx = whole_batches_count * self._batch_size

            batch_inputs = inputs[first_item_idx:]
            batch_labels = labels[first_item_idx:]

            yield (batch_inputs,
                   batch_labels)
