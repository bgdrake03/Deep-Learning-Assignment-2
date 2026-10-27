# Architecture sensitivity notes (depth x width)

Setup held fixed (Baseline_Run.txt): 80/20 split seed 42, inputs price/order/duration (log1p + standardize) + 4-dim category embedding, `sku` dropped, target `log1p(quantity)` standardized, ReLU, Adam lr 0.001, batch 256, 40 epochs, MSE loss.

**Metric.** "MSE" below is on the standardized `log1p(quantity)` target, so 1.0 = predicting the mean and `1 - MSE` is the share of log-scale variance explained. Baseline test MSE 0.3112 means about **69% explained, 31% unexplained**. On the real `log1p` scale that is RMSE ≈ 0.691.

**Noise floor.** Baseline re-run with 4 init seeds (same split): test MSE 0.3112, 0.3113, 0.3113, 0.3116 (sd 0.0002, range 0.0004). Treat differences under ~0.0005 as not distinguishable. Single data split and one seed per non-baseline config, so this is a rough guide.

## Results (test MSE; gap = test - train)

| layers \ width | 0.25x | 1x (64-32-16 taper) | 4x |
|---|---|---|---|
| 1 | 0.3155 (gap 0.0019) | 0.3118 (0.0022) | 0.3109 (0.0025) |
| 3 | 0.3127 (0.0021) | 0.3112 (0.0035) | 0.3109 (0.0044) |
| 5 | 0.3146 (0.0022) | 0.3115 (0.0032) | 0.3112 (0.0047) |

3-layer width sweep (0.25x to 8x): test MSE 0.3127, 0.3122, 0.3112, 0.3108, 0.3109, 0.3109; train MSE 0.3106 down to 0.3063. Depth sweep at 1x (1 to 5 layers): test MSE 0.3118, 0.3114, 0.3112, 0.3118, 0.3115. Full table: `results_combined.csv`.

## What changed

1. **Depth barely matters.** At 1x width every depth from 1 to 5 sits within 0.0006 of the others, about the seed noise. At 4x width the three depths are within 0.0003.
2. **Width matters only at the narrow end.** Going from 0.25x to 1x lowers test MSE by 0.0015 (3 layers) to 0.0037 (1 layer), well above noise. From 2x upward test MSE is flat at about 0.3108-0.3109, even with 168,581 parameters; the 1-layer 256-unit net (2,437 params) ties it at 0.3109.
3. **Narrow + deep is the worst corner among deep nets.** 5 layers at 0.25x (16-8-4-4-4) is 0.3146 vs 0.3127 for 3 layers at the same scale. The 4-unit layers are a bottleneck. The single 16-unit layer is worst overall (0.3155).
4. **MAE in units tracks MSE** (12.39 to 12.52, about 1% spread). RMSE in units (33.2 to 33.8) is dominated by a few huge quantities (max 4,165) and is too noisy to rank architectures.

## Over / underfitting

- **Underfitting at the narrow end.** The smallest nets have the highest *train* error (0.3136 for 1 layer x 0.25) and a small gap (0.0019). Both curves are still falling at epoch 40 (best test epoch 39-40). More capacity helps here.
- **Mild overfitting at the wide end.** Train MSE keeps improving with size (0.3106 to 0.3063) while test MSE stalls, so the gap grows from about 0.002 to about 0.0046 (see `arch_train_vs_test.png`, right panel: gap rises with log parameters). The 3-layer nets also reach their best test epoch earlier as they widen (40, 32, 27, 23 epochs for 1x, 2x, 4x, 8x). It is never severe: test MSE does not climb materially after its minimum.
- The baseline sits at the knee: past underfitting, before the gap grows.

## Why (interpretation, not tested here)

- Once a net has roughly 1k+ parameters, extra depth or width lowers train error but not test error, so the remaining ~31% of variance looks like noise the inputs do not carry (three numeric features plus a 32-level category; `sku` is excluded). That points to the data, not the architecture, as the limit.
- Depth does not help because the signal appears to be a smooth, low-complexity function of few inputs that one hidden layer can already fit. This is a hypothesis; I did not test, for example, an interaction-heavy variant.
- Narrow nets underfit simply from too few units; the deep-narrow case is worse because 4-unit layers throw away information.

## Practical reading

- Best test MSE (0.3108-0.3109) is shared by 3 x 128-64-32, 256-128-64, 512-256-128 and 1 x 256, all within noise of the baseline (0.3112, 0.0004 gap, about 2 seed sd). No architecture change is clearly better than the baseline.
- If picking on cost, 1 x 256 (2,437 params) or the baseline are the sensible choices; the 168k-parameter net adds compute and gap for nothing.
- Next levers more likely to move error than architecture: optimizer / learning rate, batch size, activation, regularization or early stopping for the wide nets.

## Caveats

- One seed per non-baseline config; sd estimated from only 4 baseline runs.
- Test split also supplied the per-epoch curves and the "best epoch" numbers (never used to stop training), and configs are compared on it, so "best" is slightly optimistic. A separate validation split would avoid this for later tuning.
- Epoch 0 in `arch_grid_curves.png` shows train loss as the running average during the first epoch, so it starts above test loss.

## Files

`results/` (`results_depth.csv`, `results_width.csv`, `results_grid.csv`, `results_combined.csv`, `results_grid_histories.json`, `logs/`), `plots/` (`arch_grid_heatmap.png`, `arch_train_vs_test.png`, `arch_grid_curves.png`, plus the per-sweep curves), `src/` (`baseline.py`, `sweep_depth.py`, `sweep_width.py`, `sweep_grid.py`, `analyze_architecture.py`).
