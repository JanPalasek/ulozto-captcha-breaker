import numpy as np


class DataBatcher:
    def __init__(self, batch_size: int, inputs, labels):
        self._inputs = inputs
        self._labels = labels
        self._batch_size = batch_size

    def batches(self):
        # TODO: currently ignoring last batch
        for batch_idx in range(len(self._inputs // self._batch_size)):
            first_item_idx = batch_idx * self._batch_size

            batch_inputs = self._inputs[first_item_idx:first_item_idx + self._batch_size]
            batch_labels = self._labels[first_item_idx:first_item_idx + self._batch_size]

            yield (batch_inputs,
                   batch_labels)
