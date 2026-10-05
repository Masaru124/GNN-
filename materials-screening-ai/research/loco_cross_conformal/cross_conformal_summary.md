# LOCO cross-conformal summary (formation energy)

Score: `|y - mu| / sigma(DER), model.eval() deterministic forward`. Shipped q = 1.0254.

| cluster | n_test | MAE (eV/atom) | median half-width (eV) | coverage @1.0254 | 95% CI | q cross-conformal | coverage cross-conformal | coverage @ shift-aware q |
| ---: | ---: | ---: | ---: | ---: | :---: | ---: | ---: | ---: |
| 0 | 7178 | 0.0741 | 0.1223 | 0.8605 | [0.8523, 0.8684] | 1.8165 | 0.9858 | 0.9858 |
| 1 | 6898 | 0.0789 | 0.0833 | 0.6785 | [0.6673, 0.6894] | 1.7155 | 0.8897 | 0.9027 |
| 2 | 2527 | 0.0855 | 1.2987 | 1.0000 | [0.9985, 1.0000] | 1.7619 | 1.0000 | 1.0000 |
| 3 | 7678 | 0.0908 | 0.1056 | 0.6806 | [0.6701, 0.6910] | 1.7077 | 0.8827 | 0.9000 |
| 4 | 6103 | 0.0907 | 0.0990 | 0.6670 | [0.6551, 0.6788] | 1.7222 | 0.8974 | 0.9130 |
| 5 | 101 | 0.0958 | 1.4208 | 1.0000 | [0.9634, 1.0000] | 1.7253 | 1.0000 | 1.0000 |
| 6 | 5332 | 0.0608 | 0.0810 | 0.7764 | [0.7651, 0.7874] | 1.7694 | 0.9571 | 0.9614 |
| 7 | 6066 | 0.1296 | 0.1139 | 0.5547 | [0.5422, 0.5672] | 1.6178 | 0.7511 | 0.7962 |
| 8 | 6604 | 0.0818 | 0.0826 | 0.6217 | [0.6100, 0.6334] | 1.6882 | 0.8546 | 0.8813 |
| 9 | 511 | 0.1176 | 1.3623 | 1.0000 | [0.9925, 1.0000] | 1.7318 | 1.0000 | 1.0000 |

Pooled coverage at shipped q: **0.7124** (n = 48998).
Shift-aware q: **1.8143** (8-th smallest per-fold 90% quantile (>=80% of folds reach 90%)); pooled coverage 0.9123, 8/10 folds at or above 90%, median half-width 0.1814 eV.

Caveat: Each fold's model is trained on the other nine clusters only for the folds retrained here; fold 0 is the production checkpoint (also soap_loco fold 0). Folds share an identical wall-clock budget, so their budgets are equal but shorter than the production run.
