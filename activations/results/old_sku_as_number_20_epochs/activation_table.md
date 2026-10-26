| Activation | Train MSE | Test MSE (± sd) | Gap | Test R² | Test MAE (units) | Epochs to converge | Best epoch | Sec / epoch | Grad ratio h1/out (start → end) | Stuck units |
|---|---|---|---|---|---|---|---|---|---|---|
| sigmoid | 0.3109 | 0.3124 ± 0.0005 | 0.0015 | 0.686 | 12.30 | >20 | 20.0 | 5.1 | 0.011 → 0.675 | 1.0% |
| tanh | 0.3076 | 0.3100 ± 0.0006 | 0.0024 | 0.688 | 12.25 | 17.0 | 20.0 | 5.2 | 2.022 → 0.674 | 0.0% |
| relu | 0.3041 | 0.3075 ± 0.0007 | 0.0034 | 0.691 | 12.19 | 8.0 | 20.0 | 4.5 | 1.909 → 1.168 | 5.2% |
| leaky_relu | 0.3044 | 0.3078 ± 0.0005 | 0.0034 | 0.690 | 12.20 | 7.7 | 20.0 | 4.0 | 1.908 → 1.293 | 0.0% |
| prelu | 0.3045 | 0.3077 ± 0.0006 | 0.0032 | 0.690 | 12.20 | 8.0 | 19.7 | 4.5 | 1.909 → 1.062 | 0.7% |
| elu | 0.3084 | 0.3105 ± 0.0008 | 0.0021 | 0.688 | 12.23 | 19.0 | 20.0 | 4.7 | 2.012 → 0.489 | 0.0% |

Mean over seeds [42, 43, 44]. MSE is on standardised log(1+quantity), the training loss. "Epochs to converge" = first epoch with test MSE <= 0.3101 (101% of the best test MSE of any run).
