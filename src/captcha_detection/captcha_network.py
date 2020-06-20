#!/usr/bin/env python

import os

import numpy as np
import tensorflow as tf
from tensorflow.python.keras.utils.vis_utils import plot_model

from accuracy.correctly_classified_captcha_accuracy import all_correct_acc


class CaptchaNetwork:
    def __init__(self, image_shape, classes: int, image_preprocess_pipeline, label_preprocess_pipeline, args):
        """
        Initializes CaptchaNetwork instance.
        :param image_shape: Shape of image.
        :param classes: Number of classes that model recognizes. E.g. if we want it to detect abcdefghijklmnopqrstuvwxyz, then
        classes must be number 28.
        :param image_preprocess_pipeline: Specifies pipeline that is used before image is put as input to neural network.
        :param label_preprocess_pipeline: Specifies pipeline that transforms output of neural network from internal indices
        back into captcha characters.
        :param args:
        """

        assert args.weights_file is None or args.pretrained_model is None, "Cannot load pretrained model and weights file at the same time"

        self._image_preprocess_pipeline = image_preprocess_pipeline
        self._label_preprocess_pipeline = label_preprocess_pipeline

        self._classes = classes
        input_shape = (image_shape[0], image_shape[1], 1)

        input = tf.keras.layers.Input(shape=input_shape)

        layer = input

        if not args.pretrained_model:
            # to normalize input
            layer = tf.keras.layers.BatchNormalization()(layer)
            layer = tf.keras.layers.Convolution2D(
                filters=32, kernel_size=7, strides=2, padding="same", use_bias=False,
                kernel_regularizer=tf.keras.regularizers.l2(args.l2))(layer)
            layer = tf.keras.layers.BatchNormalization()(layer)
            layer = tf.keras.layers.ReLU()(layer)
            layer = tf.keras.layers.MaxPooling2D(strides=2)(layer)

            layer = self._create_residual_block(layer, filters=32, l2=args.l2)
            layer = self._create_residual_block(layer, filters=32, l2=args.l2)

            layer = tf.keras.layers.BatchNormalization()(layer)
            layer = tf.keras.layers.ReLU()(layer)
            layer = tf.keras.layers.Convolution2D(
                filters=64, kernel_size=3, strides=2, padding="same", use_bias=False,
                kernel_regularizer=tf.keras.regularizers.l2(args.l2))(layer)
            layer = self._create_residual_block(layer, filters=64, l2=args.l2)
            layer = self._create_residual_block(layer, filters=64, l2=args.l2)

            layer = tf.keras.layers.BatchNormalization()(layer)
            layer = tf.keras.layers.ReLU()(layer)
            layer = tf.keras.layers.Convolution2D(
                filters=128, kernel_size=3, strides=2, padding="same", use_bias=False,
                kernel_regularizer=tf.keras.regularizers.l2(args.l2))(layer)
            layer = self._create_residual_block(layer, filters=128, l2=args.l2)
            layer = self._create_residual_block(layer, filters=128, l2=args.l2)

            layer = tf.keras.layers.BatchNormalization()(layer)
            layer = tf.keras.layers.ReLU()(layer)
            layer = tf.keras.layers.Convolution2D(
                filters=256, kernel_size=3, strides=2, padding="same", use_bias=False,
                kernel_regularizer=tf.keras.regularizers.l2(args.l2))(layer)
            layer = self._create_residual_block(layer, filters=256, l2=args.l2)
            layer = self._create_residual_block(layer, filters=256, l2=args.l2)

            layer = tf.keras.layers.GlobalAveragePooling2D()(layer)

            layer = tf.keras.layers.Dense(units=args.captcha_length * classes,
                                          kernel_regularizer=tf.keras.regularizers.l2(args.l2))(layer)
            # # reshape into (batch, letters_count, rest)
            target_shape = (args.captcha_length, classes)
            layer = tf.keras.layers.Reshape(target_shape=target_shape)(layer)

            # layer = tf.keras.layers.Dense(units=100, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01))(layer)
            # layer = tf.keras.layers.Dropout(0.5)(layer)
            output = tf.keras.layers.Dense(units=classes, activation="softmax")(layer)

            self._model = tf.keras.Model(inputs=input, outputs=output)
        else:
            self._model = tf.keras.models.load_model(args.pretrained_model)

        if args.weights_file is not None:
            self._model.load_weights(args.weights_file)

        print(f"Total layers: {len(self._model.layers)}")
        if args.remove_layers:
            # remove classification header and add new one
            input = self._model.layers[0].input
            layer = self._model.layers[-1].input
            output = tf.keras.layers.Dense(units=classes, activation="softmax")(layer)

            self._model = tf.keras.Model(inputs=input, outputs=output)

        if args.freeze_layers > 0:
            for i in range(args.freeze_layers):
                self._model.layers[i].trainable = False


        self._loss = tf.keras.losses.SparseCategoricalCrossentropy()
        self._optimizer = tf.keras.optimizers.Adam()

        metrics = [tf.keras.metrics.sparse_categorical_accuracy]
        if not args.save_model_path:
            metrics.append(all_correct_acc)
        self._model.compile(optimizer=tf.keras.optimizers.Adam(),
                            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                            metrics=metrics)

        self._model.summary()
        plot_model(self._model, to_file=os.path.join(args.out_dir, "model.png"), show_shapes=True)

        self._tb_callback = tf.keras.callbacks.TensorBoard(args.logdir, update_freq=1000, profile_batch=1)
        self._tb_callback.on_train_end = lambda *_: None
        checkpoint_path = os.path.join(args.logdir, 'cp-{epoch:02d}.h5')
        self._check_callback = tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path, save_weights_only=True)

        if args.save_model_path:
            self.save_model(args.save_model_path)

    def _create_residual_block(self, layer: tf.keras.layers.Layer, filters: int, l2: float):
        prev_layer = layer
        layer = tf.keras.layers.BatchNormalization()(layer)
        layer = tf.keras.layers.ReLU()(layer)
        layer = tf.keras.layers.Convolution2D(
            filters=filters, kernel_size=3, strides=1, padding="same", use_bias=False,
            kernel_regularizer=tf.keras.regularizers.l2(l2))(layer)

        layer = tf.keras.layers.BatchNormalization()(layer)
        layer = tf.keras.layers.ReLU()(layer)
        layer = tf.keras.layers.Convolution2D(
            filters=filters, kernel_size=3, strides=1, padding="same", use_bias=False,
            kernel_regularizer=tf.keras.regularizers.l2(l2))(layer)
        layer = tf.keras.layers.Add()([prev_layer, layer])

        return layer
    
    def train(self, train_x, train_y, val_x, val_y, args):
        """
        Train the model.
        :param train_x: Numpy array with train captcha images.
        :param train_y: Numpy array with train captcha image labels (e.g. "abxz").
        :param val_x: Numpy array with validation captcha images.
        :param val_y: Numpy array with validation captcha image labels (e.g. "abxz").
        :param args:
        :return:
        """
        train_inputs, train_labels = self._image_preprocess_pipeline(train_x), self._label_preprocess_pipeline(train_y)
        dev_inputs, dev_labels = self._image_preprocess_pipeline(val_x), self._label_preprocess_pipeline(
            val_y)

        del train_x
        del train_y
        del val_x
        del val_y

        self._model.fit(x=train_inputs, y=train_labels, batch_size=args.batch_size, epochs=args.epochs,
                        validation_data=(dev_inputs, dev_labels),
                        callbacks=[self._check_callback, self._tb_callback])

    def save_model(self, out_path):
        tf.saved_model.save(self._model, out_path)

    def predict(self, inputs, args):
        inputs = self._image_preprocess_pipeline(inputs)

        y_pred = self._model.predict(inputs, args.batch_size)
        if len(y_pred.shape) <= 2:
            y_pred = np.expand_dims(y_pred, axis=1)
        y_pred = np.argmax(y_pred, axis=2)
        return y_pred