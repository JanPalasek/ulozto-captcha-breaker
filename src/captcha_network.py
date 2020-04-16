import numpy as np
import tensorflow as tf
from tensorflow_core.python.keras.utils.vis_utils import plot_model

from dataset.captcha_dataset import CaptchaDataset
from dataset.data_batcher import DataBatcher
from dataset.preprocessing.image_preprocessors.image_cut_preprocessor import ImageCutPreprocessor
from dataset.preprocessing.image_preprocessors.image_preprocessor_pipeline import ImagePreprocessorPipeline
from dataset.preprocessing.image_preprocessors.normalize_image_preprocessor import NormalizeImagePreprocessor
from dataset.preprocessing.labels_preprocessors.label_preprocess_pipeline import LabelPreprocessPipeline
from dataset.preprocessing.labels_preprocessors.string_encoder import StringEncoder


class CaptchaNetwork:
    def __init__(self, image_shape, classes: int, time_steps: int, args):
        self._time_steps = time_steps
        input_shape = (time_steps, image_shape[0], image_shape[1] // time_steps, 1)

        input = tf.keras.layers.Input(shape=input_shape)
        layer = input
        layer = tf.keras.layers.TimeDistributed(tf.keras.layers.Convolution2D(
            filters=16, kernel_size=3, strides=1, padding="same", use_bias=False))(layer)
        layer = tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization())(layer)
        layer = tf.keras.layers.TimeDistributed(tf.keras.layers.ReLU())(layer)
        layer = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D(pool_size=(3, 3)))(layer)

        layer = tf.keras.layers.TimeDistributed(tf.keras.layers.Convolution2D(
            filters=32, kernel_size=3, strides=1, padding="same", use_bias=False))(layer)
        layer = tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization())(layer)
        layer = tf.keras.layers.TimeDistributed(tf.keras.layers.ReLU())(layer)
        layer = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D(pool_size=(3, 3)))(layer)
        layer = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())(layer)

        layer = tf.keras.layers.LSTM(units=50, return_sequences=True)(layer)
        layer = tf.keras.layers.LSTM(units=30, return_sequences=True)(layer)

        output = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(units=classes, activation="softmax"))(layer)

        self._model = tf.keras.Model(inputs=input, outputs=output)
        self._optimizer = tf.keras.optimizers.Adam()

        self._model.summary()

        self._metrics = {
            "loss": tf.metrics.Mean(),
            "accuracy": tf.keras.metrics.SparseCategoricalAccuracy()
        }
        self._writer = tf.summary.create_file_writer(args.logdir, flush_millis=10 * 1000)

    def train(self, data, args):
        image_preprocess_pipeline = ImagePreprocessorPipeline(ImageCutPreprocessor(self._time_steps), [
            NormalizeImagePreprocessor()
        ])
        label_preprocess_pipeline = LabelPreprocessPipeline(StringEncoder(available_chars="0123456789"))

        train = data["train"]
        train_inputs, train_labels = image_preprocess_pipeline(train["data"]), label_preprocess_pipeline(train["labels"])

        for epoch in range(args.epochs):
            self._train_epoch(train_inputs, train_labels, args.batch_size)
            train_loss = self._metrics["loss"].result()

            print(
                "\rEpoch: {0}, train-loss: {1:.4f}".format(epoch + 1, train_loss), flush=True)

    def _train_epoch(self, inputs, labels, batch_size):
        for i, (batch_inputs, batch_labels) in enumerate(DataBatcher(batch_size, inputs, labels).batches()):
            self._train_batch(batch_inputs, batch_labels)

            current_loss = self._metrics["loss"].result()
            print(
                '\rBatch: {0}/{1}, loss: {2:.4f}'.format(i * batch_size, len(inputs), current_loss), flush=True, end="")

    @tf.function
    def _train_batch(self, inputs, labels):
        with tf.GradientTape() as tape:
            logits = self._model(inputs, training=True)
            logit_length = tf.fill([tf.shape(logits)[0]], tf.shape(logits)[1])
            label_length = tf.fill([tf.shape(labels)[0]], tf.shape(labels)[1])

            # TODO
            loss = tf.nn.ctc_loss(
                labels=labels, logits=logits, label_length=label_length,
                logit_length=logit_length, logits_time_major=False)
            loss = tf.reduce_mean(loss)

        gradients = tape.gradient(target=loss, sources=self._model.trainable_variables)
        self._optimizer.apply_gradients(zip(gradients, self._model.trainable_variables))

        tf.summary.experimental.set_step(self._optimizer.iterations)
        with self._writer.as_default():
            for name, metric in self._metrics.items():
                metric.reset_states()
                if name == "loss":
                    metric(loss)
                tf.summary.scalar("train/{}".format(name), metric.result())
            tf.summary.scalar("train/{}".format("loss"), loss)
