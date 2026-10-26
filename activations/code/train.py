"""Train one network (one activation, one seed) and save everything about it.

Used by run_baseline.py and run_activations.py. Each run writes a folder
results/runs/<activation>_seed<seed>/ containing

  history.csv      train and test loss after every epoch, seconds per epoch
  diagnostics.csv  gradient norms and activation slopes per epoch
  metrics.csv      final train/test error, in standardised-log and real units
"""

import os
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from config import *
from data import unscale
from diagnostics import GradientMonitor, sample_rows
from model import build_model

# Without this, TensorFlow may add numbers up in a different order from run to
# run, so the same seed can give slightly different results. With it, rerunning
# a seed reproduces every number exactly.
tf.config.experimental.enable_op_determinism()


def run_dir(activation, seed):
    return os.path.join(RUNS_DIR, f'{activation}_seed{seed}')


class EpochTimer(tf.keras.callbacks.Callback):
    """Wall-clock seconds per epoch (listed first, so it excludes diagnostics)."""

    def on_train_begin(self, logs=None):
        self.seconds = []

    def on_epoch_begin(self, epoch, logs=None):
        self._start = time.perf_counter()

    def on_epoch_end(self, epoch, logs=None):
        self.seconds.append(time.perf_counter() - self._start)


def errors(model, X, y, scaler):
    """Error on a whole data set, measured AFTER training.

    mse is the training loss itself (standardised log quantity). rmse and mae
    are in real units of quantity, which is what a manager would ask about.
    """
    pred = model.predict(X, batch_size=8192, verbose=0).ravel()
    q_true, q_pred = unscale(y, scaler), unscale(pred, scaler)
    mse = float(np.mean((y - pred) ** 2))
    return {
        'mse': mse,
        'r2': 1 - mse / float(np.var(y)),
        'rmse_units': float(np.sqrt(np.mean((q_true - q_pred) ** 2))),
        'mae_units': float(np.mean(np.abs(q_true - q_pred))),
    }


def train_one(activation, seed, data, epochs=EPOCHS):
    X_train, y_train, X_test, y_test, scaler, vocab = data
    out_dir = run_dir(activation, seed)
    os.makedirs(out_dir, exist_ok=True)

    # Same seed -> same starting weights and same batch order for every
    # activation, so the activation really is the only difference.
    tf.keras.utils.set_random_seed(seed)
    model = build_model(activation, vocab)

    timer = EpochTimer()
    monitor = GradientMonitor(*sample_rows(X_train, y_train))
    print(f'\n=== {activation}  seed {seed} ===')
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),   # watched only, never used to choose anything
        batch_size=BATCH_SIZE, epochs=epochs, shuffle=True, verbose=2,
        callbacks=[timer, monitor, tf.keras.callbacks.TerminateOnNaN()])

    # Keras's 'loss' is the average over the epoch's batches while the weights
    # were still moving; 'val_loss' is measured at the epoch's end. Close
    # enough for curves. The final table re-scores both sets properly below.
    hist = pd.DataFrame({
        'epoch': np.arange(1, len(history.history['loss']) + 1),
        'train_loss': history.history['loss'],
        'test_loss': history.history['val_loss'],
        'seconds': timer.seconds,
    })
    hist.to_csv(os.path.join(out_dir, 'history.csv'), index=False)
    pd.DataFrame(monitor.rows).to_csv(os.path.join(out_dir, 'diagnostics.csv'), index=False)

    diverged = not np.isfinite(hist['train_loss']).all() or len(hist) < epochs
    metrics = {'activation': activation, 'seed': seed, 'epochs_run': len(hist),
               'diverged': diverged, 'sec_per_epoch': float(np.mean(timer.seconds))}
    if not diverged:
        metrics.update({f'train_{k}': v for k, v in errors(model, X_train, y_train, scaler).items()})
        metrics.update({f'test_{k}': v for k, v in errors(model, X_test, y_test, scaler).items()})
    if activation == 'prelu':
        # The slope PReLU learned for z < 0, averaged over each layer's units.
        for i in range(1, len(HIDDEN_LAYERS) + 1):
            alpha = model.get_layer(f'act{i}').get_weights()[0]
            metrics[f'prelu_alpha_act{i}'] = float(alpha.mean())

    pd.Series(metrics).to_csv(os.path.join(out_dir, 'metrics.csv'), header=['value'],
                              index_label='metric')
    return model, hist, metrics
