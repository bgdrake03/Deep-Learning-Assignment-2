# Deep-Learning-Assignment-2

BZAN 554 Group Assignment 2: tuning deep neural networks on `pricing.csv` (predict `quantity`
from `price`, `order`, `duration`, `category`; `sku` is dropped). The assignment is a
presentation on sensitivity to tuning parameters; there is no code submission.

The shared baseline is specified in [Baseline_Run.txt](Baseline_Run.txt).

## Baseline

80/20 random split (seed 42). Target `log1p(quantity)`, standardized. `price`, `order`,
`duration`: `log1p` then standardize. `category`: 4-dim embedding. Hidden layers 64-32-16
(ReLU), linear output, Adam lr 0.001, batch size 256, 40 epochs, MSE loss. All preprocessing
statistics are fit on the training split only.

Baseline result: test MSE 0.3112 on the standardized target (train 0.3078), i.e. about 69% of
the log-scale variance explained. On the `log1p` scale that is test RMSE ≈ 0.691; test MAE
≈ 12.43 units sold.

## rebeccas-work/

Architecture sensitivity (hidden layers and neurons per layer), everything else at baseline.

```
rebeccas-work/
  ARCHITECTURE_NOTES.md   full write-up with numbers
  src/                    scripts
  results/                per-config CSVs, grid histories (JSON), run logs in results/logs/
  plots/                  loss curves, heatmap, train-vs-test plots
```

| Script (`src/`) | What it does |
|---|---|
| `baseline.py` | Baseline model and training; writes `results/baseline_history.csv`, `plots/baseline_loss_curve.png` |
| `sweep_depth.py` | 1 to 5 hidden layers (widths are the first entries of 64-32-16-16-16) |
| `sweep_width.py` | Width multiplier 0.25x to 8x on the 64-32-16 taper |
| `sweep_grid.py` | Fills in a depth x width grid and repeats the baseline with 3 extra init seeds |
| `analyze_architecture.py` | Merges all sweeps into `results/results_combined.csv` and makes the `arch_*.png` plots |

Run from `rebeccas-work/src/` (needs TensorFlow/Keras, pandas, scikit-learn, matplotlib). The
scripts read `pricing.csv` from the assignment folder (three levels up) by default, or take a
path as an argument. Run the sweeps before `analyze_architecture.py`:

```bash
python baseline.py
python sweep_depth.py
python sweep_width.py
python sweep_grid.py
python analyze_architecture.py
```

### Findings (test MSE, standardized `log1p(quantity)`)

Seed-to-seed noise on the baseline is sd ≈ 0.0002 (range 0.0004 over 4 init seeds), so
differences under about 0.0005 are not distinguishable.

| layers \ width | 0.25x | 1x (64-32-16) | 4x |
|---|---|---|---|
| 1 | 0.3155 | 0.3118 | 0.3109 |
| 3 | 0.3127 | 0.3112 | 0.3109 |
| 5 | 0.3146 | 0.3115 | 0.3112 |

- **Depth barely matters.** At a given width, 1 to 5 layers are within about 0.0006.
- **Narrow nets underfit.** The smallest nets have the highest train error and still improve at
  epoch 40. Widening from 0.25x to 1x lowers test MSE by 0.0015 to 0.0037.
- **Wide nets overfit mildly.** From 2x up, test MSE is flat at about 0.3108-0.3109 while train
  MSE keeps falling, so the train/test gap grows from about 0.002 to 0.0046 and the best test
  epoch moves earlier (40, 32, 27, 23 for 1x, 2x, 4x, 8x). Test error never rises materially.
- **No architecture clearly beats the baseline.** The best configs (e.g. 1 x 256 with 2,437
  parameters, 3 x 128-64-32) tie within noise. The remaining ~31% unexplained variance looks
  like it comes from the inputs, so the more promising levers are optimizer, learning rate,
  batch size, activation and regularization (hypothesis, not yet tested).

Caveats: one seed per non-baseline config, a single data split, and the test split was used to
compare configs (a separate validation split would avoid that for later tuning).
