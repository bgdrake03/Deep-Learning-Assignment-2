"""Step 2: train the same network with each of the six hidden activations.

Everything in the shared spec is held fixed; only the activation changes.
Runs that already have a metrics.csv are skipped, so the experiment can be
done in pieces and resumed after an interruption.

    python run_activations.py                       # all six, every seed
    python run_activations.py sigmoid tanh relu leaky_relu
    python run_activations.py prelu elu
    python run_activations.py --force relu          # redo a finished run
"""

import os
import sys
from config import *
from data import load_data
from train import run_dir, train_one

if __name__ == '__main__':
    args = sys.argv[1:]
    force = '--force' in args
    chosen = [a for a in args if a != '--force'] or ACTIVATIONS
    unknown = set(chosen) - set(ACTIVATIONS)
    if unknown:
        sys.exit(f'Unknown activation(s): {unknown}. Choose from {ACTIVATIONS}')

    data = load_data()
    for activation in chosen:
        for seed in RUN_SEEDS:
            done = os.path.exists(os.path.join(run_dir(activation, seed), 'metrics.csv'))
            if done and not force:
                print(f'skip {activation} seed {seed} (already done)')
                continue
            train_one(activation, seed, data)

    print('\nDone. Next: python analyze_activations.py')
