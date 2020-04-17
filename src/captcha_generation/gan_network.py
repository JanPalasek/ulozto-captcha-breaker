#!/usr/bin/env python3
# Honza
# 87947918-ba1c-11e7-a937-00505601122b
# Roman
# 2b38b15f-ba47-11e7-a937-00505601122b

import numpy as np
import tensorflow as tf

# from tensorflow.python.keras.utils import plot_model

# The neural network model
class Network:
    def __init__(self, args):
        self._z_dim = args.z_dim

        # Define `self.generator` as a Model, which
        # - takes vectors of [args.z_dim] shape on input
        generator_input = tf.keras.layers.Input(shape=[args.z_dim])
        # - applies len(args.generator_layers) dense layers with ReLU activation,
        #   i-th layer with args.generator_layers[i] units
        generator_layer = generator_input

        generator_layer = tf.keras.layers.Dense(units=100, activation="relu")(generator_layer)
        generator_layer = tf.keras.layers.Dense(units=300, activation="relu")(generator_layer)
        generator_layer = tf.keras.layers.Reshape(target_shape=[10, 10])(generator_layer)
        generator_layer = tf.keras.layers.Convolution2DTranspose(filters=16, kernel_size=3, strides=1)(generator_layer)
        generator_layer = tf.keras.layers.Convolution2DTranspose(filters=32, kernel_size=3, strides=1)(generator_layer)

        for i in range(len(args.generator_layers)):
            generator_layer = tf.keras.layers.Dense(units=args.generator_layers[i], activation="relu")(generator_layer)
        # - applies output dense layer with MNIST.H * MNIST.W * MNIST.C units
        #   and sigmoid activation
        generator_layer = tf.keras.layers.Dense(units=MNIST.H * MNIST.W * MNIST.C, activation="sigmoid")(generator_layer)
        # - reshapes the output (tf.keras.layers.Reshape) to [MNIST.H, MNIST.W, MNISt.C]
        generator_output = tf.keras.layers.Reshape(target_shape=[MNIST.H, MNIST.W, MNIST.C])(generator_layer)
        self._generator = keras.Model(inputs=generator_input, outputs=generator_output)

        # TODO: Define `self.discriminator` as a Model, which
        # - takes input images with shape [MNIST.H, MNIST.W, MNIST.C]
        discriminator_input = keras.layers.Input(shape=[MNIST.H, MNIST.W, MNIST.C])
        # - flattens them
        discriminator_layer = keras.layers.Flatten()(discriminator_input)
        # - applies len(args.discriminator_layers) dense layers with ReLU activation,
        #   i-th layer with args.discriminator_layers[i] units
        for i in range(len(args.discriminator_layers)):
            discriminator_layer = keras.layers.Dense(units=args.discriminator_layers[i], activation="relu")(discriminator_layer)
        # - applies output dense layer with one output and a suitable activation function
        discriminator_output = keras.layers.Dense(units=1, activation="sigmoid")(discriminator_layer)
        self.discriminator = keras.Model(inputs=discriminator_input, outputs=discriminator_output)

        # plot_model(self.generator, to_file='gan_generator.png', show_shapes=True)
        # plot_model(self.discriminator, to_file='gan_discriminator.png', show_shapes=True)

        self._generator_optimizer, self._discriminator_optimizer = tf.optimizers.Adam(), tf.optimizers.Adam()
        self._loss_fn = tf.losses.BinaryCrossentropy()
        self._discriminator_accuracy = tf.metrics.Mean()
        self._writer = tf.summary.create_file_writer(args.logdir, flush_millis=10 * 1000)

    def _sample_z(self, batch_size):
        """Sample random latent variable."""
        return tf.random.uniform([batch_size, self._z_dim], -1, 1, seed=42)

    # @tf.function
    def train_batch(self, images, batch_size):
        # Generator training. Using a Gradient tape:
        with tf.GradientTape() as tape:
            # - generate random images using a `generator`; do not forget about `training=True`
            random_images = self._sample_z(batch_size)
            generated_images = self._generator(random_images, training=True)
            # - run discriminator on the generated images, also using `training=True` (even if
            #   not updating discriminator parameters, we want to perform possible BatchNorm in it)
            discriminator_output = self.discriminator(generated_images, training=True)
            # - compute loss using `_loss_fn`, with target labels `tf.ones_like(discriminator_output)`
            # remarks: tf.ones_like because generator wants to get as close to ones in discriminator result as possible
            generator_loss = self._loss_fn(y_true=tf.ones_like(discriminator_output), y_pred=discriminator_output)
            # Then, compute the gradients with respect to generator trainable variables and update
            # generator trainable weights using self._generator_optimizer.
            generator_gradients = tape.gradient(target=generator_loss, sources=self._generator.variables)
            self._generator_optimizer.apply_gradients(zip(generator_gradients, self._generator.variables))

        # Discriminator training. Using a Gradient tape:
        with tf.GradientTape() as tape:
            # - discriminate `images`, storing results in `discriminated_real`
            discriminated_real = self.discriminator(images, training=True)
            # - discriminate images generated in generator training, storing results in `discriminated_fake`
            discriminated_fake = self.discriminator(generated_images, training=True)
            # - compute loss by summing
            #   - `_loss_fn` on discriminated_real with suitable target labels
            #   - `_loss_fn` on discriminated_fake with suitable targets (`tf.{ones,zeros}_like` come handy).
            # remarks: discriminator wants to hit correctly either real (wants to hit 1) and fake (wants to hit 0)
            discriminator_real_loss = self._loss_fn(y_true=tf.ones_like(discriminated_real),y_pred=discriminated_real)
            discriminator_fake_loss = self._loss_fn(y_true=tf.zeros_like(discriminated_fake), y_pred=discriminated_fake)
            discriminator_loss = discriminator_real_loss + discriminator_fake_loss
            # Then, compute the gradients with respect to discriminator trainable variables and update
            # discriminator trainable weights using self._discriminator_optimizer.
            discriminator_gradients = tape.gradient(target=discriminator_loss, sources=self.discriminator.variables)
            self._discriminator_optimizer.apply_gradients(zip(discriminator_gradients, self.discriminator.variables))

        self._discriminator_accuracy(tf.greater(discriminated_real, 0.5))
        self._discriminator_accuracy(tf.less(discriminated_fake, 0.5))
        tf.summary.experimental.set_step(self._discriminator_optimizer.iterations)
        with self._writer.as_default():
            tf.summary.scalar("gan/generator_loss", generator_loss)
            tf.summary.scalar("gan/discriminator_loss", discriminator_loss)
            tf.summary.scalar("gan/discriminator_accuracy", self._discriminator_accuracy.result())

        return generator_loss + discriminator_loss

    def generate(self):
        GRID = 20

        # Generate GRIDxGRID images
        random_images = self._generator(self._sample_z(GRID * GRID))

        starts, ends = self._sample_z(GRID), self._sample_z(GRID)
        interpolated_z = tf.concat(
            [starts[i] + (ends[i] - starts[i]) * tf.expand_dims(tf.linspace(0., 1., GRID), -1) for i in range(GRID)], axis=0)
        interpolated_images = self._generator(interpolated_z)

        # Stack the random images, then an empty row, and finally interpolated imates
        image = tf.concat(
            [tf.concat(list(images), axis=1) for images in tf.split(random_images, GRID)] +
            [tf.zeros([MNIST.H, MNIST.W * GRID, MNIST.C])] +
            [tf.concat(list(images), axis=1) for images in tf.split(interpolated_images, GRID)], axis=0)
        with self._writer.as_default():
            tf.summary.image("gan/images", tf.expand_dims(image, 0))

    def train_epoch(self, dataset, args):
        self._discriminator_accuracy.reset_states()
        loss = 0
        for batch in dataset.batches(args.batch_size):
            loss += self.train_batch(batch["images"])
        self.generate()
        return loss
