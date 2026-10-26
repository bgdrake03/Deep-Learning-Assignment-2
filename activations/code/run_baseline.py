"""Step 1: the shared-spec baseline, trained end to end.

Run this before varying anything. The point is a sanity check: both curves
should fall and then flatten, and test loss should sit close to train loss.
If they don't, fix the spec first - comparing activations on a broken
baseline tells us nothing.

    python run_baseline.py
"""

import os
import matplotlib.pyplot as plt
from config import *
from data import load_data, summarise
from model import build_model
from train import train_one

if __name__ == '__main__':
    print(summarise().to_string())
    data = load_data()
    build_model(BASELINE_ACTIVATION, data[5]).summary()

    _, hist, metrics = train_one(BASELINE_ACTIVATION, RUN_SEEDS[0], data)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(hist['epoch'], hist['train_loss'], label='train')
    ax.plot(hist['epoch'], hist['test_loss'], label='test')
    ax.set_xlabel('epoch')
    ax.set_ylabel('MSE (standardised log quantity)')
    ax.set_title(f'Baseline ({BASELINE_ACTIVATION}): train vs test loss')
    ax.legend()
    fig.tight_layout()
    path = os.path.join(PLOT_DIR, 'baseline_loss_curve.png')
    fig.savefig(path, dpi=PLOT_DPI)

    print('\nFinal metrics:')
    for k, v in metrics.items():
        print(f'  {k:20s} {v}')
    print(f'\nLoss curve saved to {path}')
