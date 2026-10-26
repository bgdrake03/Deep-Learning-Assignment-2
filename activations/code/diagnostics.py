"""Looking for vanishing and exploding gradients while the network trains.

Backprop sends the error signal from the output back to the first layer. At
every hidden layer it is multiplied by the slope of the activation, f'(z), and
by the layer's weights. So:

  * if f'(z) is small (sigmoid's is at most 0.25, and ~0 once a unit
    saturates) the signal shrinks layer after layer -> VANISHING gradients,
    and the early layers learn slowly or not at all;
  * if the products are large the signal grows -> EXPLODING gradients, and the
    loss jumps around or becomes NaN.

At the end of every epoch (and once before training) this callback takes a
fixed sample of training rows and records, per hidden layer:

  grad_norm_<layer>   size of dLoss/dW for that layer's weights. Compare
                      hidden1 (furthest from the output) with output: a much
                      smaller hidden1 norm means the signal is vanishing.
  mean_slope_actN     average f'(z) - how much signal the activation lets through.
  stuck_frac_actN     share of units with f'(z) < STUCK_DERIVATIVE on every
                      sample row: dead relus or saturated sigmoid/tanh units.
"""

import numpy as np
import tensorflow as tf
from config import *

DENSE_LAYERS = [f'hidden{i}' for i in range(1, len(HIDDEN_LAYERS) + 1)] + ['output']
ACT_LAYERS = [f'act{i}' for i in range(1, len(HIDDEN_LAYERS) + 1)]


def sample_rows(X, y, n=DIAGNOSTIC_SAMPLE, seed=SPLIT_SEED):
    idx = np.random.default_rng(seed).choice(len(y), size=n, replace=False)
    return {k: v[idx] for k, v in X.items()}, y[idx]


class GradientMonitor(tf.keras.callbacks.Callback):

    def __init__(self, X_sample, y_sample):
        super().__init__()
        self.X = X_sample
        self.y = tf.constant(y_sample.reshape(-1, 1))
        self.rows = []

    def on_train_begin(self, logs=None):
        # A second model that shares the same layers but stops at each Dense
        # output, so we can see z (the input to each activation).
        self.z_model = tf.keras.Model(
            self.model.inputs, [self.model.get_layer(f'hidden{i}').output
                                for i in range(1, len(HIDDEN_LAYERS) + 1)])
        self._record(epoch=0)

    def on_epoch_end(self, epoch, logs=None):
        self._record(epoch=epoch + 1)

    def _record(self, epoch):
        row = {'epoch': epoch}

        # 1. Gradient of the loss with respect to each layer's weight matrix.
        kernels = [self.model.get_layer(n).kernel for n in DENSE_LAYERS]
        with tf.GradientTape() as tape:
            pred = self.model(self.X, training=False)
            loss = tf.reduce_mean(tf.square(self.y - pred))
        grads = tape.gradient(loss, kernels)
        for name, g in zip(DENSE_LAYERS, grads):
            row[f'grad_norm_{name}'] = float(tf.norm(g))

        # 2. The activation's own slope f'(z) at every row and unit. The
        #    activations work elementwise, so the gradient of sum(f(z)) with
        #    respect to z is exactly f'(z) at each position.
        for name, z in zip(ACT_LAYERS, self.z_model(self.X, training=False)):
            with tf.GradientTape() as tape:
                tape.watch(z)
                total = tf.reduce_sum(self.model.get_layer(name)(z))
            slope = np.abs(tape.gradient(total, z).numpy())
            row[f'mean_slope_{name}'] = float(slope.mean())
            row[f'stuck_frac_{name}'] = float((slope < STUCK_DERIVATIVE).all(axis=0).mean())

        self.rows.append(row)
