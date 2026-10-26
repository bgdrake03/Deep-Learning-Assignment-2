"""The baseline network, with the hidden activation as the one free choice.

    category -> Embedding(33, 4) -> Flatten ----+-> Concatenate (7)
    numeric  (price, order, duration) ----------+
        -> Dense 64 -> act -> Dense 32 -> act -> Dense 16 -> act -> Dense 1 (linear)

Each hidden layer is written as two layers: a Dense with NO activation, which
computes z = x @ W + b, followed by a separate activation layer a = f(z). It is
the same maths as Dense(64, activation='relu'), but it lets us treat all six
activations the same way - prelu and leaky relu only exist as layers - and lets
the diagnostics look at z and a separately.

The output is linear because the target (standardised log quantity) can be any
real number. Only the HIDDEN activation is varied.
"""

import tensorflow as tf
from config import *


def activation_layer(name, layer_name):
    """f(z) for one hidden layer.

    sigmoid    1 / (1 + e^-z)                  range (0, 1), f' <= 0.25
    tanh       (e^z - e^-z) / (e^z + e^-z)     range (-1, 1), f' <= 1
    relu       max(0, z)                       f' is 0 or 1
    leaky_relu z if z > 0 else slope * z       f' is slope or 1, never 0
    prelu      z if z > 0 else alpha * z       alpha is LEARNED, one per unit
                                               (Keras starts it at 0 = relu)
    elu        z if z > 0 else a(e^z - 1)      smooth, negative side -> -a
    """
    if name == 'leaky_relu':
        return tf.keras.layers.LeakyReLU(negative_slope=LEAKY_RELU_SLOPE, name=layer_name)
    if name == 'prelu':
        return tf.keras.layers.PReLU(name=layer_name)
    if name == 'elu':
        return tf.keras.layers.ELU(alpha=ELU_ALPHA, name=layer_name)
    return tf.keras.layers.Activation(name, name=layer_name)


def build_model(activation, vocab):
    cat_in = tf.keras.layers.Input(shape=(1,), name='category')
    num_in = tf.keras.layers.Input(shape=(len(NUMERIC),), name='numeric')

    cat_vec = tf.keras.layers.Embedding(vocab['category'], CATEGORY_EMBED_DIM,
                                        name='category_embedding')(cat_in)
    x = tf.keras.layers.Concatenate(name='concat')([
        tf.keras.layers.Flatten(name='category_flat')(cat_vec),   # (B, 1, 4) -> (B, 4)
        num_in,
    ])

    for i, units in enumerate(HIDDEN_LAYERS, start=1):
        x = tf.keras.layers.Dense(units, kernel_initializer=KERNEL_INITIALIZER,
                                  name=f'hidden{i}')(x)
        x = activation_layer(activation, f'act{i}')(x)

    out = tf.keras.layers.Dense(1, activation='linear', name='output')(x)

    model = tf.keras.Model(inputs=[cat_in, num_in], outputs=out,
                           name=f'dnn_{activation}')
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                  loss='mse')
    return model


if __name__ == '__main__':
    build_model(BASELINE_ACTIVATION, {'category': 33}).summary()
