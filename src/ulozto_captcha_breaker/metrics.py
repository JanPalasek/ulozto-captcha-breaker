import tensorflow as tf


def all_correct_acc(y_true: tf.Tensor, y_pred: tf.Tensor):
    """
    Computes accuracy between y_true and y_pred in the following manner:

    - If i-th sample has all values on y_pred same as on y_true, then 1.
    - Otherwise 0.

    It is hence more restricting then a typical accuracy.

    Args:
        y_true (tf.Tensor): 2D tensor of shape (N, L), where N is the number of samples and L is length of the vector (number of characters).
        y_pred: 2D tensor of shape (N, L), where N is the number of samples and L is length of the vector (number of characters)

    Returns:
        Accuracy: number between [0, 1] denoting how many codes were predicted correctly.
    """
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

