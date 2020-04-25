import sys

import tensorflow as tf

def all_correct_acc(y_true: tf.Tensor, y_pred):
    if y_true.shape[0] is None and y_true.shape[1] is None and y_true.shape[2] is None:
        return tf.convert_to_tensor(0)

    # cast to int64 so we can compare it
    y_true = tf.cast(y_true, tf.dtypes.int64)

    if len(y_pred.shape) <= 2:
        y_pred = tf.expand_dims(y_pred, axis=1)
    if len(y_true.shape) <= 1:
        y_true = tf.expand_dims(y_true, axis=1)
    y_pred = tf.argmax(y_pred, axis=2)
    correct = y_true == y_pred
    # tf.print(f"Pred shape: {y_true.shape}", output_stream=sys.stdout)

    all_correct = tf.reduce_all(correct, axis=1)
    all_correct = tf.cast(all_correct, tf.dtypes.float32)

    return tf.reduce_mean(all_correct)

class CorrectlyClassifiedCaptchaAccuracy(tf.metrics.Metric):
    """
    Calculates how many captchas were correctly classified.
    E.g.:
    If target is "abcd" and we classified "abbd", then we didn't classify it correctly (we get 0).
    If target is "abcd" and we classified "abcd", then we did classify it correctly (thus we get 1).

    result() returns average of correctly classified labels.
    """
    def __init__(self, name='correctly_classified', **kwargs):
        super(CorrectlyClassifiedCaptchaAccuracy, self).__init__(name=name, **kwargs)
        # stores how many captchas were correctly classified completely
        # e.g. if the target is "abcd" and network predicts
        self._correctly_classified = self.add_weight(name='tp', initializer='zeros')
        self._counter = self.add_weight(name='c', initializer='zeros')

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
        self._counter.assign_add(len(all_correct))
        # tf.print(f"All correct: {len(all_correct)}", output_stream=sys.stdout)
        # tf.print(f"Counter: {self._counter}", output_stream=sys.stdout)
        update = tf.reduce_sum(all_correct)

        self._correctly_classified.assign_add(update)

    def result(self):
        return self._correctly_classified / self._counter

    def reset_states(self):
        # The state of the metric will be reset at the start of each epoch.
        self._correctly_classified.assign(0.)
        self._counter.assign(0)
