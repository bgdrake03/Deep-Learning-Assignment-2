"""Every setting for the project lives here. Change things here, not elsewhere.

The SHARED SPEC section is what the whole group agreed to hold fixed. If a
teammate's spec differs, reconcile it here and rerun - nothing else changes.
"""

import os

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------- data
# Same pricing.csv as assignment 1. It is confidential, so it is never committed.
# Put a copy in data/ next to code/ (or in data/ at the repo root); failing
# that we look for the assignment 1 folder sitting beside this project.
_PARENT = os.path.dirname(PROJECT_DIR)
_DATA_CANDIDATES = [
    os.path.join(PROJECT_DIR, 'data', 'pricing.csv'),
    os.path.join(_PARENT, 'data', 'pricing.csv'),
    os.path.join(_PARENT, 'Deep-Learning-Assignment-1', 'data', 'pricing.csv'),
    os.path.join(os.path.dirname(_PARENT), 'Deep-Learning-Assignment-1', 'data', 'pricing.csv'),
]
DATA_FILE = next((p for p in _DATA_CANDIDATES if os.path.exists(p)), _DATA_CANDIDATES[0])

CATEGORICAL = ['category']                       # integer code -> embedding
NUMERIC = ['price', 'order', 'duration']         # log1p (right-skewed), then standardised
TARGET = 'quantity'                              # log1p, then standardised
# sku is not used. See the SHARED SPEC note below.

# ================================================================ SHARED SPEC
TEST_SPLIT = 0.2      # 80/20 train/test
SPLIT_SEED = 42       # the split never changes, whatever else we vary

# category (32 codes, ~12,500 rows each) gets an embedding: a lookup table of
# learned vectors - a linear layer without the one-hot step.
#
# sku is DROPPED. It is a product ID, not a quantity, so using it as a number
# assumes IDs have a meaningful order. The proper way to use an ID is an
# embedding, but with 75,000 skus and ~5 training rows each it memorises
# individual products: test loss is best after epoch 1 and then climbs every
# epoch, even with L2 (see notes/activation_notes.md).
CATEGORY_EMBED_DIM = 4

HIDDEN_LAYERS = [64, 32, 16]        # three hidden layers, as in assignment 1
BASELINE_ACTIVATION = 'relu'
KERNEL_INITIALIZER = 'glorot_uniform'   # Keras default, held fixed for all six

LEARNING_RATE = 0.001     # Adam
BATCH_SIZE = 256
EPOCHS = 40               # at 20, every run was still improving
# ============================================================================

# ---------------------------------------------------------------- my experiment
# The ONLY thing that changes between runs is the hidden-layer activation.
ACTIVATIONS = ['sigmoid', 'tanh', 'relu', 'leaky_relu', 'prelu', 'elu']
LEAKY_RELU_SLOPE = 0.01   # slope for z < 0 (Keras's default would be 0.3)
ELU_ALPHA = 1.0           # ELU tends to -alpha for very negative z

# Each activation is trained once per seed. The seed sets the starting weights
# and the batch order; the train/test split stays fixed at SPLIT_SEED. Several
# seeds tell us whether a gap between two activations is real or just luck.
RUN_SEEDS = [42, 43, 44, 45, 46]

# "Convergence speed" = the first epoch at which a run's TEST loss gets within
# this factor of the best test loss any run reached. Same finish line for all.
# 1%, because every run lands within about 3% of the best one after epoch 1.
CONVERGENCE_TOLERANCE = 1.01

# Rows used to measure gradients and activations at the end of every epoch.
DIAGNOSTIC_SAMPLE = 4096
# A hidden unit is "stuck" if its slope f'(z) is below this on EVERY sample row:
# a dead relu (z < 0 everywhere, f' = 0) or a saturated sigmoid/tanh (|z| large,
# f' ~ 0). No gradient flows back through a stuck unit, so it stops learning.
STUCK_DERIVATIVE = 1e-3

# ---------------------------------------------------------------- output
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results')
RUNS_DIR = os.path.join(RESULTS_DIR, 'runs')      # one folder per activation/seed
PLOT_DIR = os.path.join(RESULTS_DIR, 'plots')
for _d in (RESULTS_DIR, RUNS_DIR, PLOT_DIR):
    os.makedirs(_d, exist_ok=True)

DATA_SUMMARY_PATH = os.path.join(RESULTS_DIR, 'data_summary.csv')
SUMMARY_PATH = os.path.join(RESULTS_DIR, 'activation_summary.csv')
SUMMARY_TABLE_PATH = os.path.join(RESULTS_DIR, 'activation_table.md')
PLOT_DPI = 200
