import os

import numpy as np
import tensorflow as tf
from tensorflow_core.python.keras.utils.vis_utils import plot_model
from tensorflow.keras import backend as K

from accuracy.correctly_classified_captcha_accuracy import CorrectlyClassifiedCaptchaAccuracy
from dataset.captcha_dataset import CaptchaDataset
from dataset.data_batcher import DataBatcher
from dataset.preprocessing.image_preprocessors.image_cut_preprocessor import ImageCutPreprocessor
from dataset.preprocessing.image_preprocessors.image_preprocessor_pipeline import ImagePreprocessorPipeline
from dataset.preprocessing.image_preprocessors.normalize_image_preprocessor import NormalizeImagePreprocessor
from dataset.preprocessing.labels_preprocessors.label_preprocess_pipeline import LabelPreprocessPipeline
from dataset.preprocessing.labels_preprocessors.one_char_encoder import OneCharEncoder
from dataset.preprocessing.labels_preprocessors.string_encoder import StringEncoder


class CaptchaNetwork:
    def __init__(self, image_shape, classes: int, time_steps: int, args):
        # classes += 1
        self._classes = classes
        self._time_steps = time_steps
        input_shape = (image_shape[0], image_shape[1], 1)

        input = tf.keras.layers.Input(shape=input_shape)
        layer = input
        layer = tf.keras.layers.Convolution2D(
            filters=16, kernel_size=8, strides=4, padding="same", use_bias=False)(layer)
        layer = tf.keras.layers.BatchNormalization()(layer)
        layer = tf.keras.layers.ReLU()(layer)

        layer = tf.keras.layers.Convolution2D(
            filters=32, kernel_size=4, strides=2, padding="same", use_bias=False)(layer)
        layer = tf.keras.layers.BatchNormalization()(layer)
        layer = tf.keras.layers.ReLU()(layer)

        layer = tf.keras.layers.Convolution2D(
            filters=32, kernel_size=4, strides=2, padding="same", use_bias=False)(layer)
        layer = tf.keras.layers.BatchNormalization()(layer)
        layer = tf.keras.layers.ReLU()(layer)

        layer = tf.keras.layers.Convolution2D(
            filters=64, kernel_size=4, strides=2, padding="same", use_bias=False)(layer)
        layer = tf.keras.layers.BatchNormalization()(layer)
        layer = tf.keras.layers.ReLU()(layer)

        conv_to_rnn_shape = (layer.shape[1], layer.shape[2] * layer.shape[3])
        layer = tf.keras.layers.Reshape(target_shape=conv_to_rnn_shape)(layer)

        layer = tf.keras.layers.Dense(units=time_steps, activation="relu")(layer)
        layer = tf.keras.layers.Permute((2, 1))(layer)

        layer = tf.keras.layers.Flatten()(layer)

        layer = tf.keras.layers.Dense(units=100, activation="relu")(layer)
        layer = tf.keras.layers.Dropout(0.5)(layer)
        layer = tf.keras.layers.Dense(units=50, activation="relu")(layer)
        layer = tf.keras.layers.Dropout(0.5)(layer)
        output = tf.keras.layers.Dense(units=classes, activation="softmax")(layer)

        self._model = tf.keras.Model(inputs=input, outputs=output)
        self._optimizer = tf.keras.optimizers.Adam()

        self._model.summary()
        plot_model(self._model, to_file=os.path.join(args.out_dir, "model.png"), show_shapes=True)

        self._metrics = {
            "loss": tf.metrics.Mean(),
            "accuracy": tf.keras.metrics.SparseCategoricalAccuracy(),
            "correct_accuracy": CorrectlyClassifiedCaptchaAccuracy()
        }
        self._validation_metrics = {
            "loss": tf.metrics.Mean(),
            "accuracy": tf.keras.metrics.SparseCategoricalAccuracy(),
            "correct_accuracy": CorrectlyClassifiedCaptchaAccuracy()
        }
        self._writer = tf.summary.create_file_writer(args.logdir, flush_millis=10 * 1000)

    def train(self, data, args):
        image_preprocess_pipeline = ImagePreprocessorPipeline([
            NormalizeImagePreprocessor()
        ])
        label_preprocess_pipeline = LabelPreprocessPipeline(
            # StringEncoder(available_chars="0123456789")
            OneCharEncoder(available_chars="0123456789")
        )

        train = data["train"]
        train_inputs, train_labels = image_preprocess_pipeline(train["data"]), label_preprocess_pipeline(train["labels"])

        dev = data["dev"]
        dev_inputs, dev_labels = image_preprocess_pipeline(dev["data"]), label_preprocess_pipeline(
            dev["labels"])

        for epoch in range(args.epochs):
            self._train_epoch(train_inputs, train_labels, args.batch_size)
            self._evaluate_epoch(dev_inputs, dev_labels, args.batch_size)

            train_loss = self._metrics["loss"].result()
            train_acc = self._metrics["accuracy"].result()
            train_correct_acc = self._metrics["correct_accuracy"].result()

            dev_loss = self._validation_metrics["loss"].result()
            dev_acc = self._validation_metrics["accuracy"].result()
            dev_correct_acc = self._validation_metrics["correct_accuracy"].result()

            print(
                f"\rEpoch: {epoch + 1}, train-loss: {train_loss:.4f}, train-acc: {train_acc:.4f}, train-correct-dev: {train_correct_acc:.4f}, "
                f"dev-loss: {dev_loss:.4f}, dev-acc: {dev_acc:.4f}, dev-correct-acc: {dev_correct_acc:.4f}", flush=True)

    def _train_epoch(self, inputs, labels, batch_size):
        for i, (batch_inputs, batch_labels) in enumerate(DataBatcher(batch_size, inputs, labels).batches()):
            self._train_batch(batch_inputs, batch_labels)

            current_loss = self._metrics["loss"].result()
            current_acc = self._metrics["accuracy"].result()
            print(
                '\rBatch: {0}/{1}, loss: {2:.4f}, acc: {3:.4f}'.format(i * batch_size, len(inputs), current_loss,
                                                                       current_acc), flush=True, end="")

    def _evaluate_epoch(self, inputs, labels, batch_size):
        for i, (batch_inputs, batch_labels) in enumerate(DataBatcher(batch_size, inputs, labels).batches()):
            self._evaluate_batch(batch_inputs, batch_labels)

    @tf.function
    def _evaluate_batch(self, batch_inputs, batch_labels):
        logits = self._model(batch_inputs, training=False)
        batch_labels = tf.squeeze(batch_labels)
        loss = tf.losses.sparse_categorical_crossentropy(batch_labels, logits)

        # logits = tf.transpose(logits, [1, 0, 2])
        # label_length = tf.fill([tf.shape(batch_labels)[0]], tf.shape(batch_labels)[1])
        #
        # decoded = tf.nn.ctc_greedy_decoder(logits, sequence_length=label_length, merge_repeated=False)

        tf.summary.experimental.set_step(self._optimizer.iterations)
        with self._writer.as_default():
            for name, metric in self._validation_metrics.items():
                metric.reset_states()
                if name == "loss":
                    metric(loss)
                else:
                    metric.update_state(y_true=batch_labels, y_pred=logits)
                tf.summary.scalar("dev/{}".format(name), metric.result())

    @tf.function
    def _train_batch(self, inputs, labels):
        with tf.GradientTape() as tape:
            logits = self._model(inputs, training=True)
            labels = tf.squeeze(labels)

            loss = tf.losses.sparse_categorical_crossentropy(labels, logits)
            loss = tf.reduce_mean(loss)

        gradients = tape.gradient(target=loss, sources=self._model.trainable_variables)
        self._optimizer.apply_gradients(zip(gradients, self._model.trainable_variables))

        tf.summary.experimental.set_step(self._optimizer.iterations)
        with self._writer.as_default():
            for name, metric in self._metrics.items():
                metric.reset_states()
                if name == "loss":
                    metric(loss)
                else:
                    metric.update_state(y_true=labels, y_pred=logits)
                tf.summary.scalar("train/{}".format(name), metric.result())
