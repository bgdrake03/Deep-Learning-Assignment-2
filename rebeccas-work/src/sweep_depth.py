"""Vary the number of hidden layers, holding everything else at the baseline.

Widths are a prefix of [64, 32, 16, 16, 16]: the baseline taper, extended at its
last width (16) so deeper nets do not add a narrow bottleneck.

Usage: python sweep_depth.py [path/to/pricing.csv]
"""

import sys
from pathlib import Path

import keras
import numpy as np
import pandas as pd

import baseline as b

WIDTHS = [64, 32, 16, 16, 16]
DEPTHS = [1, 2, 3, 4, 5]


def main():
    here = Path(__file__).resolve().parent
    results_dir, plots_dir = here.parent / "results", here.parent / "plots"
    results_dir.mkdir(exist_ok=True)
    plots_dir.mkdir(exist_ok=True)
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else here.parent.parent.parent / "pricing.csv"

    df = b.load_data(path)
    train_df, test_df = b.split(df)
    prep = b.fit_preprocessing(train_df)
    x_num_tr, x_cat_tr, y_tr = b.transform(train_df, prep)
    x_num_te, x_cat_te, y_te = b.transform(test_df, prep)
    qty_tr = train_df["quantity"].to_numpy(dtype=np.float64)
    qty_te = test_df["quantity"].to_numpy(dtype=np.float64)
    train_in = {"numeric": x_num_tr, "category": x_cat_tr}
    test_in = {"numeric": x_num_te, "category": x_cat_te}

    rows, curves = [], {}
    for depth in DEPTHS:
        hidden = WIDTHS[:depth]
        keras.utils.set_random_seed(b.SEED)
        model = b.build_model(len(prep["categories"]), hidden=hidden)
        hist = model.fit(
            train_in, y_tr, validation_data=(test_in, y_te),
            batch_size=b.BATCH_SIZE, epochs=b.EPOCHS, shuffle=True, verbose=0,
        ).history
        curves[depth] = hist

        p_tr = model.predict(train_in, verbose=0).ravel()
        p_te = model.predict(test_in, verbose=0).ravel()
        row = {"layers": depth, "hidden": "-".join(map(str, hidden)), "params": model.count_params()}
        for name, p, y, q in (("train", p_tr, y_tr, qty_tr), ("test", p_te, y_te, qty_te)):
            qp = b.back_to_quantity(p, prep)
            row[f"{name}_mse_std"] = float(np.mean((y - p) ** 2))
            row[f"{name}_rmse_log"] = float(np.sqrt(np.mean((y * prep['y_std'] - p * prep['y_std']) ** 2)))
            row[f"{name}_mae_qty"] = float(np.mean(np.abs(q - qp)))
            row[f"{name}_rmse_qty"] = float(np.sqrt(np.mean((q - qp) ** 2)))
        row["best_test_epoch"] = int(np.argmin(hist["val_loss"]) + 1)
        rows.append(row)
        print(row, flush=True)
        pd.DataFrame(rows).to_csv(results_dir / "results_depth.csv", index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    for depth, h in curves.items():
        line, = ax.plot(range(1, b.EPOCHS + 1), h["val_loss"], label=f"{depth} layers (test)")
        ax.plot(range(1, b.EPOCHS + 1), h["loss"], "--", color=line.get_color(), alpha=0.6)
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE (standardized log1p quantity)")
    ax.set_title("Depth sweep (solid = test, dashed = train)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "results_depth_curves.png", dpi=150)


if __name__ == "__main__":
    main()
