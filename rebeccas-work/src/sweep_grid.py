"""Fill in the depth x width grid and estimate seed-to-seed noise.

Depths {1, 3, 5} x width multipliers {0.25, 1, 4} on the base taper
[64, 32, 16, 16, 16][:depth]. Combinations already measured by sweep_depth.py
(mult 1) and sweep_width.py (depth 3) are reused, so only the four missing corners
are trained here. The baseline is also repeated with three other init seeds
(the data split stays at seed 42) to size the noise floor.

Usage: python sweep_grid.py [path/to/pricing.csv]
"""

import json
import sys
from pathlib import Path

import keras
import numpy as np
import pandas as pd

import baseline as b

WIDTHS = [64, 32, 16, 16, 16]
NEW_CONFIGS = [(1, 0.25), (1, 4), (5, 0.25), (5, 4)]
SEED_REPEATS = [1, 2, 3]


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

    runs = [(d, m, b.SEED) for d, m in NEW_CONFIGS] + [(3, 1, s) for s in SEED_REPEATS]
    rows, histories = [], {}
    for depth, mult, seed in runs:
        hidden = [int(u * mult) for u in WIDTHS[:depth]]
        keras.utils.set_random_seed(seed)
        model = b.build_model(len(prep["categories"]), hidden=hidden)
        hist = model.fit(
            train_in, y_tr, validation_data=(test_in, y_te),
            batch_size=b.BATCH_SIZE, epochs=b.EPOCHS, shuffle=True, verbose=0,
        ).history
        histories[f"d{depth}_m{mult}_s{seed}"] = {k: [float(v) for v in vals] for k, vals in hist.items()}

        p_tr = model.predict(train_in, verbose=0).ravel()
        p_te = model.predict(test_in, verbose=0).ravel()
        row = {"layers": depth, "multiplier": mult, "seed": seed,
               "hidden": "-".join(map(str, hidden)), "params": model.count_params()}
        for name, p, y, q in (("train", p_tr, y_tr, qty_tr), ("test", p_te, y_te, qty_te)):
            qp = b.back_to_quantity(p, prep)
            row[f"{name}_mse_std"] = float(np.mean((y - p) ** 2))
            row[f"{name}_rmse_log"] = float(np.sqrt(np.mean((y - p) ** 2)) * prep["y_std"])
            row[f"{name}_mae_qty"] = float(np.mean(np.abs(q - qp)))
            row[f"{name}_rmse_qty"] = float(np.sqrt(np.mean((q - qp) ** 2)))
        row["best_test_epoch"] = int(np.argmin(hist["val_loss"]) + 1)
        rows.append(row)
        print(row, flush=True)
        pd.DataFrame(rows).to_csv(results_dir / "results_grid.csv", index=False)
        (results_dir / "results_grid_histories.json").write_text(json.dumps(histories))

    

if __name__ == "__main__":
    main()
