import sys

import tensorflow as tf


class CorrectlyClassifiedCaptchaAccuracy(tf.metrics.Metric):
    def __init__(self, name='correctly_classified', **kwargs):
        super(CorrectlyClassifiedCaptchaAccuracy, self).__init__(name=name, **kwargs)
        self._correctly_classified = self.add_weight(name='tp', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        if len(y_pred.shape) <= 2:
            y_pred = tf.expand_dims(y_pred, axis=1)
        if len(y_true.shape) <= 1:
            y_true = tf.expand_dims(y_true, axis=1)
        y_pred = tf.argmax(y_pred, axis=2)
        correct = y_true == y_pred
        # tf.print(f"Pred shape: {y_true.shape}", output_stream=sys.stdout)

        all_correct = tf.reduce_all(correct, axis=1)
        all_correct = tf.cast(all_correct, tf.dtypes.float32)
        update = tf.reduce_mean(all_correct)

        self._correctly_classified.assign_add(update)

    def result(self):
        return self._correctly_classified

    def reset_states(self):
        # The state of the metric will be reset at the start of each epoch.
        self._correctly_classified.assign(0.)
