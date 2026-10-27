"""Baseline MLP for pricing.csv, following Baseline_Run.txt.

Usage: python baseline.py [path/to/pricing.csv]
Defaults to pricing.csv in the assignment folder (three levels above this file).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import keras
from keras import layers
from sklearn.model_selection import train_test_split

SEED = 42
N_CATEGORY_DIMS = 4
HIDDEN = [64, 32, 16]
LEARNING_RATE = 1e-3
BATCH_SIZE = 256
EPOCHS = 40
NUMERIC = ["price", "order", "duration"]


def load_data(path):
    df = pd.read_csv(path)
    # sku is dropped: it is a product ID and an embedding on it overfit after 1 epoch.
    return df.drop(columns=["sku"])


def split(df):
    train, test = train_test_split(df, test_size=0.2, random_state=SEED, shuffle=True)
    return train.reset_index(drop=True), test.reset_index(drop=True)


def fit_preprocessing(train):
    """Fit all preprocessing statistics on the training split only."""
    numeric_log = np.log1p(train[NUMERIC].to_numpy(dtype=np.float64))
    num_mean = numeric_log.mean(axis=0)
    num_std = numeric_log.std(axis=0)

    y_log = np.log1p(train["quantity"].to_numpy(dtype=np.float64))
    y_mean = y_log.mean()
    y_std = y_log.std()

    # Categories are integer codes that are not consecutive, so map them to
    # contiguous indices. The last index is reserved for unseen categories.
    categories = np.unique(train["category"].to_numpy())
    return {
        "num_mean": num_mean,
        "num_std": num_std,
        "y_mean": y_mean,
        "y_std": y_std,
        "categories": categories,
    }


def transform(df, prep):
    numeric = np.log1p(df[NUMERIC].to_numpy(dtype=np.float64))
    numeric = (numeric - prep["num_mean"]) / prep["num_std"]

    lookup = {c: i for i, c in enumerate(prep["categories"])}
    unknown = len(prep["categories"])
    cat = np.array([lookup.get(c, unknown) for c in df["category"].to_numpy()], dtype=np.int32)

    y = None
    if "quantity" in df:
        y = (np.log1p(df["quantity"].to_numpy(dtype=np.float64)) - prep["y_mean"]) / prep["y_std"]
    return numeric.astype(np.float32), cat, y


def build_model(n_categories, hidden=HIDDEN):
    numeric_in = keras.Input(shape=(len(NUMERIC),), name="numeric")
    category_in = keras.Input(shape=(1,), dtype="int32", name="category")

    emb = layers.Embedding(input_dim=n_categories + 1, output_dim=N_CATEGORY_DIMS)(category_in)
    emb = layers.Flatten()(emb)

    x = layers.Concatenate()([numeric_in, emb])
    for units in hidden:
        x = layers.Dense(units, activation="relu")(x)
    out = layers.Dense(1, activation="linear")(x)

    model = keras.Model([numeric_in, category_in], out)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="mse",
        metrics=["mae"],
    )
    return model


def back_to_quantity(pred_std, prep):
    log_q = pred_std * prep["y_std"] + prep["y_mean"]
    return np.maximum(np.expm1(log_q), 0.0)


def report(name, y_true_qty, y_pred_qty, y_true_log, y_pred_log):
    rmse_log = np.sqrt(np.mean((y_true_log - y_pred_log) ** 2))
    mae_qty = np.mean(np.abs(y_true_qty - y_pred_qty))
    rmse_qty = np.sqrt(np.mean((y_true_qty - y_pred_qty) ** 2))
    print(f"{name:<22} RMSE(log1p)={rmse_log:.4f}  MAE(qty)={mae_qty:.3f}  RMSE(qty)={rmse_qty:.3f}")


def save_curves(hist, results_dir, plots_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pd.DataFrame(hist).rename_axis("epoch_index").to_csv(results_dir / "baseline_history.csv")
    epochs = np.arange(1, len(hist["loss"]) + 1)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, hist["loss"], label="train")
    ax.plot(epochs, hist["val_loss"], label="test")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE (standardized log1p quantity)")
    ax.set_title("Baseline loss curves")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "baseline_loss_curve.png", dpi=150)


def main():
    default_path = Path(__file__).resolve().parents[3] / "pricing.csv"
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else default_path

    keras.utils.set_random_seed(SEED)

    df = load_data(path)
    train_df, test_df = split(df)
    print(f"Rows: total={len(df)} train={len(train_df)} test={len(test_df)}")

    prep = fit_preprocessing(train_df)
    x_num_tr, x_cat_tr, y_tr = transform(train_df, prep)
    x_num_te, x_cat_te, y_te = transform(test_df, prep)

    model = build_model(n_categories=len(prep["categories"]))
    model.summary()

    # The test split is passed as validation_data only to record a per-epoch test
    # loss curve; it is never used to pick or stop anything.
    history = model.fit(
        {"numeric": x_num_tr, "category": x_cat_tr},
        y_tr,
        validation_data=({"numeric": x_num_te, "category": x_cat_te}, y_te),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        shuffle=True,
        verbose=2,
    )
    root = Path(__file__).resolve().parent.parent
    (root / "results").mkdir(exist_ok=True)
    (root / "plots").mkdir(exist_ok=True)
    save_curves(history.history, root / "results", root / "plots")

    pred_tr = model.predict({"numeric": x_num_tr, "category": x_cat_tr}, verbose=0).ravel()
    pred_te = model.predict({"numeric": x_num_te, "category": x_cat_te}, verbose=0).ravel()

    qty_tr = train_df["quantity"].to_numpy(dtype=np.float64)
    qty_te = test_df["quantity"].to_numpy(dtype=np.float64)

    print()
    report("Train", qty_tr, back_to_quantity(pred_tr, prep), y_tr, pred_tr)
    report("Test", qty_te, back_to_quantity(pred_te, prep), y_te, pred_te)

    # Reference point: always predict the training mean on the log scale.
    naive_qty = np.full_like(qty_te, np.expm1(prep["y_mean"]))
    report("Test (mean baseline)", qty_te, naive_qty, y_te, np.zeros_like(y_te))


if __name__ == "__main__":
    main()
