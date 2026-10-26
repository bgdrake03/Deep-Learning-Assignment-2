# Activation sensitivity: notes for slides 7–9 (and the data slide)

All numbers below come from `results/activation_summary.csv` and `results/runs/`.
Rerun `python code/run_activations.py` then `python code/analyze_activations.py` to regenerate them.
Runs are fully reproducible: the same seed gives identical numbers every time.

---

## Data overview (slide 2)

- 500,000 rows, 6 columns, no missing values. Inputs used: price, order, duration, category. Target: quantity.
- 74,999 distinct SKUs (≈ 6.7 rows each), 32 categories.
- Quantity is very right-skewed: median 7, mean 22.8, 99th percentile 228, max 4,165.
- Price is skewed too (mean 0.82, max 275).
- Split: random 80/20, seed 42 → 400,000 train / 100,000 test rows.

---

## The shared spec (what was held fixed)

Same as `baseline_requirements.txt` in the team repo.

| Setting | Value | Why |
|---|---|---|
| Split | 80/20 random, seed 42 | Shared spec. Rows are shuffled before splitting, so the split isn't by SKU order. |
| Target | log(1 + quantity), standardised | With raw MSE, the few huge orders (up to 4,165) would dominate the loss. |
| price, order, duration | log(1 + x), then standardised | Skewed; puts every input on mean 0 / sd 1 so no input pushes sigmoid/tanh into their flat regions. |
| sku | **not used** | A product ID, not a quantity (see below). |
| category | Embedding, 4 dims | A code, not a quantity; 32 codes with ~12,500 rows each is plenty to learn from. |
| Hidden layers | 64 → 32 → 16 | Same shape as assignment 1. |
| Output | 1 unit, linear | Regression: the target can be any real number. |
| Loss / optimizer | MSE, Adam, learning rate 0.001 | |
| Batch size / epochs | 256 / 40 | Training rows reshuffled every epoch. |
| Weight init | Glorot uniform (Keras default) for every activation | Held fixed on purpose so the activation is the only change. |
| Statistics for scaling | from training rows only | No test-set leakage. |

Train/test error is re-measured on the full train and test sets **after** training.
Keras's per-epoch "loss" is an average taken while the weights are still changing.

### Why sku was dropped (one line on the setup slide)

SKU is a product ID. We tried it three ways (relu baseline):

| sku handling | Result |
|---|---|
| Embedding (8 dims, 600,000 weights) | Overfit right away. Test MSE was best after epoch 1 (0.298), then rose every epoch (0.419 by epoch 20). With ~5 rows per SKU it memorised products. L2 regularisation didn't fix it. |
| Standardised number | Stable, but assumes SKU 50,000 is "more" than 49,999, which is meaningless for an ID. |
| **Dropped** | Stable and honest. Costs about 0.003 in test MSE versus using it as a number. |

The overfit run is in `results/before_spec_fix/`.
The earlier sku-as-number results are in `results/old_sku_as_number_20_epochs/`.

---

## Slide 7: Activation setup

**Varied:** the activation on the three hidden layers, one of six:

| Activation | f(z) | f'(z) | Notes |
|---|---|---|---|
| sigmoid | 1 / (1 + e^-z) | f(1 − f), **at most 0.25** | output in (0, 1), not zero-centred |
| tanh | (e^z − e^-z)/(e^z + e^-z) | 1 − f², at most 1 | zero-centred, saturates at ±1 |
| relu | max(0, z) | 0 or 1 | can "die" (z < 0 on every row → gradient 0) |
| leaky relu | z if z > 0 else 0.01z | 0.01 or 1 | never exactly 0, so it can't die |
| prelu | z if z > 0 else αz | α or 1 | α is **learned**, one per unit (Keras starts it at 0, i.e. as relu) |
| elu | z if z > 0 else e^z − 1 | 1 or e^z | smooth; negative side flattens toward −1 |

**Held fixed:** everything in the shared-spec table, including the starting weights.
Each seed gives every activation the same starting weights and the same batch order.

**Repeats:** each activation was trained with 5 seeds (42–46): 30 runs in all.
The split always uses seed 42. The seed sd tells us which gaps are real.

**Measured:** train and test MSE; test MAE in real units; convergence speed; and, every epoch, gradient size per layer, average slope f'(z), and the share of "stuck" units (f'(z) ≈ 0 on every sample row).

**Convergence speed:** the first epoch where test MSE is within 1% of the best final test MSE (0.3101), so 0.3126 or lower.
Everyone races to the same finish line.

---

## Slide 8: Activation results

| Activation | Train MSE | Test MSE (± sd, 5 seeds) | Test MAE (units) | Epochs to converge | hidden1 / output gradient at start |
|---|---|---|---|---|---|
| sigmoid | 0.3104 | 0.3121 ± 0.0004 | 12.30 | 29.8 | **0.011** |
| tanh | 0.3087 | 0.3111 ± 0.0003 | 12.29 | 12.4 | 2.20 |
| relu | 0.3070 | **0.3101** ± 0.0004 | 12.27 | 6.0 | 2.25 |
| leaky relu | 0.3070 | **0.3101** ± 0.0003 | **12.25** | **5.2** | 2.26 |
| prelu | 0.3070 | 0.3102 ± 0.0004 | 12.26 | 6.4 | 2.25 |
| elu | 0.3090 | 0.3113 ± 0.0004 | 12.26 | 16.6 | 2.15 |

Plots: `results/plots/learning_curves.png`, `final_error.png`, `gradient_flow.png`, `activation_health.png`.

What the numbers say:

1. **There are three tiers.**
   The relu family (relu / leaky / prelu) is fastest and best, then tanh and elu, then sigmoid.
   relu vs sigmoid differs by 0.0020 in test MSE, about 5× the seed-to-seed sd, so it's real.
   relu vs leaky vs prelu differ by ≤ 0.0001, well inside seed noise, so they're a tie.
2. **The biggest difference is speed.**
   The relu family reaches the finish line in 5–6 epochs, tanh in 12, elu in 17, and sigmoid in 30.
   relu matches sigmoid's epoch-40 test error by epoch 7.
3. **The final error differences are small.**
   Test MSE spans only 0.3101–0.3121 (0.6%), and test MAE 12.25–12.30 units.
   For this small network, choosing the activation is mostly choosing how long you wait.
4. **The gap is still closing.**
   Over the last 10 epochs relu improved by only 0.0001, while sigmoid improved by 0.0005.
   relu has levelled off; sigmoid is still catching up. With enough epochs, the final-error gap would likely shrink further.
5. **No overfitting anywhere.**
   The train–test gap is 0.0018–0.0032. The slower activations have smaller gaps because they've fit less so far.
6. Time per epoch was 4.7–5.7 s. That's noisy (the laptop was in use), so speed is compared in epochs, not seconds.

---

## Slide 9: Activation takeaway

**Best:** relu, leaky relu and prelu, which are statistically tied (test MSE 0.3101–0.3102, 5–6 epochs to converge).
Leaky relu is a sensible default: same accuracy, slightly fastest, and it can't produce dead units.

**Vanishing gradients: clear evidence with sigmoid.**
Before training, the gradient reaching the first hidden layer was **1.1% of the size** of the output layer's (ratio 0.011).
With every other activation that ratio was ~2.2.
That's the textbook mechanism: on the way back to hidden1 the error passes through three sigmoid slopes, each at most 0.25.
0.25³ ≈ 0.016, close to the 0.011 measured (the measured slopes were 0.23–0.25). tanh and elu have slope ≈ 1 near 0, so nothing shrinks.
During training the ratio recovers to ~0.5. Adam scales each weight's step by that weight's own gradient history, which partly makes up for small gradients.
That is why sigmoid still learns, just slowly.

**Sigmoid also saturates.**
Its average slope in layer 2 fell from 0.23 to 0.06 over training, and 8.8% of layer-3 sigmoid units ended up stuck in the flat region.

**Dying relu: mild.**
By epoch 40, 6.2% of relu units in the last hidden layer (3.8% in layer 2) were dead.
For leaky relu it was 0%, and for prelu 1.2%.
That didn't hurt accuracy here, but it is the failure leaky/prelu are designed to prevent.

**PReLU learned negative slopes** (average α = −0.05, −0.05, −0.15 for layers 1–3).
So for z < 0 the output is slightly *positive*, a "V"-like shape that relu can't express. It didn't improve the error.

**Exploding gradients: none.**
No NaN losses and no divergence in 30 runs; the largest gradient norm seen was 3.4 (sigmoid's output layer at the start).
With standardised inputs, Adam and a small network, there was nothing to explode.

**Insight (1–2 sentences):**
On this data the activation mostly changes *how fast* the network learns, not *how well*: the relu family converged 2–3× faster than tanh/elu and 5× faster than sigmoid, while final test error differed by under 1%.
Sigmoid's slowness is visible directly as a vanishing gradient: the first layer received about 1% of the signal at the start.

**Caveats worth saying out loud:**
- Glorot initialisation was held fixed. He initialisation is the usual partner for the relu family and might widen their lead.
- tanh and elu don't show vanishing gradients, yet still converged 2–3× slower than relu. We didn't test why.
- These results are for this baseline (3 layers, Adam). With plain SGD or a deeper network, sigmoid's vanishing gradient would likely hurt much more. That's a good joint-slide check with the other tracks.

---

## Be ready to explain on paper (Test 2)

- **Why does sigmoid make gradients vanish?** Backprop multiplies by f'(z) at every layer. σ'(z) = σ(z)(1 − σ(z)) ≤ 0.25, so three layers shrink the signal by at least 0.25³ ≈ 1/64.
- **Why does relu avoid that?** f'(z) = 1 for z > 0, so the signal passes through unchanged. The price: f'(z) = 0 for z < 0, so a unit whose z is negative for every input never updates (dead relu).
- **Leaky vs PReLU:** both give z < 0 a small slope so gradients never hit exactly 0. Leaky fixes the slope (0.01); PReLU learns it by gradient descent like any other weight.
- **ELU:** like leaky relu, but smooth, and its negative side levels off at −1, pushing average activations toward 0.
- **Writing one in Keras:** `Dense(64)` followed by `LeakyReLU(negative_slope=0.01)` is the same as `Dense(64, activation=...)` with a leaky relu. PReLU and leaky relu are written as separate layers (see `code/model.py`).
