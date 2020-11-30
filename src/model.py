import tensorflow as tf

from blocks import ResBlock


def ResNet(input_shape, output_shape, init_filters: int, l2: float):
    input = tf.keras.layers.Input(shape=input_shape)

    layer = input
    filters = init_filters
    layer = tf.keras.layers.BatchNormalization()(layer)
    layer = tf.keras.layers.Convolution2D(
        filters=filters, kernel_size=7, strides=2, padding="same", use_bias=False,
        kernel_regularizer=tf.keras.regularizers.l2(l2))(layer)
    layer = tf.keras.layers.BatchNormalization()(layer)
    layer = tf.keras.layers.ReLU()(layer)
    layer = tf.keras.layers.MaxPooling2D(strides=2)(layer)

    layer = ResBlock(filters=filters, strides=1, l2=l2)(layer)
    layer = ResBlock(filters=filters, strides=1, l2=l2)(layer)

    for i in range(2):
        layer = ResBlock(filters=filters, strides=2, l2=l2)(layer)
        layer = ResBlock(filters=filters, strides=1, l2=l2)(layer)
        layer = ResBlock(filters=filters, strides=1, l2=l2)(layer)

    layer = tf.keras.layers.GlobalAveragePooling2D()(layer)

    captcha_length, classes = output_shape
    layer = tf.keras.layers.Dense(units=captcha_length * classes,
                                  kernel_regularizer=tf.keras.regularizers.l2(l2))(layer)
    layer = tf.keras.layers.Reshape(target_shape=output_shape)(layer)
    output = tf.keras.layers.Dense(units=classes, activation="softmax")(layer)

    model = tf.keras.Model(inputs=input, outputs=output)
    return model
