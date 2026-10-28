"""Combine depth, width and grid sweeps; plot train vs test error.

Run after sweep_depth.py, sweep_width.py and sweep_grid.py have finished.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

here = Path(__file__).resolve().parent
results_dir, plots_dir = here.parent / "results", here.parent / "plots"
plots_dir.mkdir(exist_ok=True)
KEEP = ["layers", "multiplier", "seed", "hidden", "params", "train_mse_std", "test_mse_std",
        "train_mae_qty", "test_mae_qty", "best_test_epoch"]

depth = pd.read_csv(results_dir / "results_depth.csv").assign(multiplier=1.0, seed=42)
width = pd.read_csv(results_dir / "results_width.csv").assign(layers=3, seed=42)
grid = pd.read_csv(results_dir / "results_grid.csv")
allr = pd.concat([depth[KEEP], width[KEEP], grid[KEEP]], ignore_index=True)
allr = allr.drop_duplicates(subset=["layers", "multiplier", "seed"]).sort_values(
    ["seed", "layers", "multiplier"]).reset_index(drop=True)
allr["gap"] = allr.test_mse_std - allr.train_mse_std
allr.to_csv(results_dir / "results_combined.csv", index=False)

main = allr[allr.seed == 42]
seeds = allr[(allr.layers == 3) & (allr.multiplier == 1)]
print("Seed noise on baseline (test MSE):")
print(seeds[["seed", "train_mse_std", "test_mse_std", "gap"]].round(4).to_string(index=False))
print(f"mean={seeds.test_mse_std.mean():.4f} sd={seeds.test_mse_std.std(ddof=1):.4f} "
      f"range={seeds.test_mse_std.max() - seeds.test_mse_std.min():.4f}")

# 1. depth x width heatmap of test MSE (and gap)
depths, mults = [1, 3, 5], [0.25, 1, 4]
grid_t = main.pivot_table(index="layers", columns="multiplier", values="test_mse_std").loc[depths, mults]
grid_g = main.pivot_table(index="layers", columns="multiplier", values="gap").loc[depths, mults]
print("\nDepth x width test MSE:\n", grid_t.round(4))
print("\nDepth x width gap (test - train):\n", grid_g.round(4))
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
for ax, g, title in ((axes[0], grid_t, "Test MSE"), (axes[1], grid_g, "Gap (test - train)")):
    im = ax.imshow(g.values, cmap="viridis_r" if title == "Test MSE" else "magma_r")
    ax.set_xticks(range(3), [f"{m}x" for m in mults])
    ax.set_yticks(range(3), [f"{d} layer(s)" for d in depths])
    ax.set_xlabel("width multiplier (base 64-32-16)")
    ax.set_title(title)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{g.values[i, j]:.4f}", ha="center", va="center", color="white", fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.8)
fig.tight_layout()
fig.savefig(plots_dir / "arch_grid_heatmap.png", dpi=150)

# 2. train vs test error across every config
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
ax = axes[0]
sc = ax.scatter(main.train_mse_std, main.test_mse_std, c=np.log10(main.params), cmap="viridis", s=45)
lim = [main.train_mse_std.min() - 0.0005, main.test_mse_std.max() + 0.0005]
ax.plot(lim, lim, "k--", lw=1, label="test = train")
ax.set_xlabel("train MSE (standardized)")
ax.set_ylabel("test MSE (standardized)")
ax.set_title("Train vs test error, every architecture")
ax.legend()
fig.colorbar(sc, ax=ax, label="log10(params)")
ax = axes[1]
ax.scatter(main.params, main.gap, c="tab:blue", s=45, label="train MSE gap")
ax.set_xscale("log")
ax.set_xlabel("parameters")
ax.set_ylabel("test - train MSE")
ax.set_title("Generalization gap vs model size")
ax.axhline(seeds.gap.mean(), color="gray", ls=":", label="baseline gap (mean of seeds)")
ax.legend()
fig.tight_layout()
fig.savefig(plots_dir / "arch_train_vs_test.png", dpi=150)

# 3. learning curves from the grid runs
hist = json.loads((results_dir / "results_grid_histories.json").read_text())
fig, ax = plt.subplots(figsize=(7.5, 4.2))
for key, h in hist.items():
    if key.endswith("_s42"):
        d, m = key.split("_")[0][1:], key.split("_")[1][1:]
        line, = ax.plot(h["val_loss"], label=f"{d}L x{m} test")
        ax.plot(h["loss"], "--", color=line.get_color(), alpha=0.6)
ax.set_xlabel("epoch index")
ax.set_ylabel("MSE (standardized log1p quantity)")
ax.set_title("Grid corners (solid = test, dashed = train)")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(plots_dir / "arch_grid_curves.png", dpi=150)

print("\nFull table:")
print(main[["layers", "multiplier", "hidden", "params", "train_mse_std", "test_mse_std", "gap",
            "test_mae_qty", "best_test_epoch"]].round(4).to_string(index=False))
