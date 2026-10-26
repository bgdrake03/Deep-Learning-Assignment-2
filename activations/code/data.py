"""Load pricing.csv, split it 80/20, and preprocess it. The shared spec.

500,000 rows of six columns is ~24 MB, so unlike assignment 1 we simply load
the whole file into memory. Nothing here depends on the activation function:
every run in the experiment sees exactly the same data.

Preprocessing, and why:
  * category       - an integer code, NOT a quantity (category 5 is not "more"
                     than category 4). Left as an integer, fed to an embedding.
  * sku            - not used: a product ID, and an embedding of it overfits
                     (see config.py).
  * price, order, duration - right-skewed (price runs to 275 with a mean of
                     0.8). log1p pulls in the long tail, then standardising
                     gives mean 0 / std 1 so no input dominates the first layer
                     or pushes sigmoid/tanh units into their flat regions.
  * quantity       - median 7 but max 4,165. With raw MSE a handful of huge
                     rows would dominate training. We fit log1p(quantity),
                     standardised; unscale() turns predictions back into units.

Every mean and std comes from the TRAINING rows only (no test-set leakage).
"""

import numpy as np
import pandas as pd
from config import *


def load_raw():
    return pd.read_csv(DATA_FILE)


def split(df):
    """Random 80/20 row split, always with SPLIT_SEED so it never moves."""
    rng = np.random.default_rng(SPLIT_SEED)
    order = rng.permutation(len(df))
    n_test = int(round(len(df) * TEST_SPLIT))
    return df.iloc[order[n_test:]], df.iloc[order[:n_test]]


def _log_skewed(df):
    """log1p on the numeric columns and the target (all right-skewed)."""
    return np.log1p(df[NUMERIC + [TARGET]].astype(float))


def fit_scaler(train):
    """Mean and std of the logged numeric columns and target."""
    logged = _log_skewed(train)
    return {'mean': logged.mean(), 'std': logged.std()}


def transform(df, scaler):
    """Raw rows -> (inputs dict for the model, standardised target)."""
    mean, std = scaler['mean'], scaler['std']
    logged = _log_skewed(df)
    numeric = (logged[NUMERIC] - mean[NUMERIC]) / std[NUMERIC]
    X = {
        'category': df['category'].to_numpy(np.int32),
        'numeric': numeric.to_numpy(np.float32),
    }
    y = ((logged[TARGET] - mean[TARGET]) / std[TARGET]).to_numpy(np.float32)
    return X, y


def unscale(y_scaled, scaler):
    """Standardised log quantity -> quantity in real units."""
    mean, std = scaler['mean'][TARGET], scaler['std'][TARGET]
    return np.expm1(y_scaled * std + mean)


def load_data():
    """Everything a training script needs, in one call.

    Returns:
        X_train, y_train, X_test, y_test, scaler, vocab
        vocab = {'category': n_codes} for the embedding size.
    """
    df = load_raw()
    train, test = split(df)
    scaler = fit_scaler(train)
    X_train, y_train = transform(train, scaler)
    X_test, y_test = transform(test, scaler)
    # Codes are IDs, not data about the target, so sizing the lookup table from
    # the whole file is not leakage.
    vocab = {c: int(df[c].max()) + 1 for c in CATEGORICAL}
    return X_train, y_train, X_test, y_test, scaler, vocab


def summarise():
    """Numbers for the data-overview slide. Writes DATA_SUMMARY_PATH."""
    df = load_raw()
    train, test = split(df)
    rows = {
        'rows_total': len(df),
        'rows_train': len(train),
        'rows_test': len(test),
        'n_sku': df['sku'].nunique(),
        'n_category': df['category'].nunique(),
        'quantity_median': df[TARGET].median(),
        'quantity_mean': df[TARGET].mean(),
        'quantity_p99': df[TARGET].quantile(0.99),
        'quantity_max': df[TARGET].max(),
        'price_max': df['price'].max(),
        'missing_values': int(df.isna().sum().sum()),
    }
    out = pd.Series(rows, name='value')
    out.to_csv(DATA_SUMMARY_PATH, index_label='metric')
    return out


if __name__ == '__main__':
    print(summarise().to_string())
