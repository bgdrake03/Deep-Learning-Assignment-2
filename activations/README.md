# Group Assignment 2: Activation-function sensitivity (Ansley's section)

This folder holds the code for my part of the deck: **slides 7–9, Activation setup / results / takeaway.**
The six hidden-layer activations (sigmoid, tanh, relu, leaky relu, prelu, elu) are each trained on the same shared-spec network, and everything else is held fixed.

## How to run

1. Put `pricing.csv` in `activations/data/` (or `data/` at the repo root). It's confidential, so `.gitignore` keeps it out of git.
2. Install the packages: `pip install -r requirements.txt`
3. From the `code/` directory:

```
cd code
python run_baseline.py          # 1. shared-spec baseline, sanity-check the loss curve
python run_activations.py       # 2. all six activations x 5 seeds (~2 hours on CPU)
python analyze_activations.py   # 3. summary table + plots
```

`run_activations.py` skips runs that are already finished, so you can do it in pieces
(`python run_activations.py sigmoid tanh relu leaky_relu`, then `python run_activations.py prelu elu`).
Add `--force` to redo a run.

## Files

| File | What it does |
|---|---|
| `code/config.py` | **Every setting.** The shared spec is fenced off at the top; change things only here. |
| `code/data.py` | Loads `pricing.csv`, makes the 80/20 split (seed 42), preprocesses. `python data.py` prints the numbers for the data-overview slide. |
| `code/model.py` | Builds the baseline network with the hidden activation as the one free choice. |
| `code/diagnostics.py` | Callback that measures gradient size per layer and activation slopes f'(z) every epoch: the vanishing/exploding-gradient evidence. |
| `code/train.py` | Trains one (activation, seed) run and saves its history, diagnostics and final metrics. |
| `code/run_baseline.py` | Step 1. |
| `code/run_activations.py` | Step 2. |
| `code/analyze_activations.py` | Step 3. |
| `results/` | Output: `activation_summary.csv`, `activation_table.md`, `plots/`, one folder per run in `runs/`. |
| `notes/activation_notes.md` | Write-up of what changed, why, with numbers. Source for slides 7–9. |

## Shared spec (held fixed for every run)

Matches `Baseline_Run.txt` and `baseline_requirements.txt` at the repo root:
80/20 random split (seed 42) · target log(1+quantity), standardised · price/order/duration log(1+x) then standardised ·
**sku not used** · category → 4-dim embedding · hidden layers 64 → 32 → 16, linear output ·
Glorot init · MSE + Adam (lr 0.001) · batch 256 · 40 epochs · fully reproducible (op determinism on).
Each choice and the reasoning behind it is in `notes/activation_notes.md`.

Baseline check for the team: relu, seed 42 gives **test MSE 0.3104** (train 0.3070).

## Results at a glance (mean of 5 seeds)

| Activation | Test MSE | Epochs to converge |
|---|---|---|
| sigmoid | 0.3121 | 29.8 |
| tanh | 0.3111 | 12.4 |
| relu | 0.3101 | 6.0 |
| leaky relu | 0.3101 | 5.2 |
| prelu | 0.3102 | 6.4 |
| elu | 0.3113 | 16.6 |

Older runs are kept for reference: `results/before_spec_fix/` (sku embedding, overfit) and
`results/old_sku_as_number_20_epochs/` (sku as a number, 20 epochs, 3 seeds).
