# LOCO cross-conformal summary (formation energy)

Score: `|y - mu| / sigma(DER), model.eval() deterministic forward`. Shipped q = 1.0254.

| cluster | n_test | MAE (eV/atom) | median half-width (eV) | coverage @1.0254 | 95% CI | q cross-conformal | coverage cross-conformal | coverage @ shift-aware q |
| ---: | ---: | ---: | ---: | ---: | :---: | ---: | ---: | ---: |
| 0 | 7178 | 0.0741 | 0.1223 | 0.8605 | [0.8523, 0.8684] | 1.6792 | 0.9784 | 0.9851 |
| 1 | 6898 | 0.0789 | 0.0833 | 0.6785 | [0.6673, 0.6894] | 1.3969 | 0.8121 | 0.9000 |
| 2 | 2527 | 0.0855 | 1.2987 | 1.0000 | [0.9985, 1.0000] | 1.5876 | 1.0000 | 1.0000 |
| 3 | 7678 | 0.0908 | 0.1056 | 0.6806 | [0.6701, 0.6910] | 1.3729 | 0.8065 | 0.8978 |
| 9 | 511 | 0.1176 | 1.3623 | 1.0000 | [0.9925, 1.0000] | 1.5385 | 1.0000 | 1.0000 |

Pooled coverage at shipped q: **0.7713** (n = 24792).
Shift-aware q: **1.7994** (4-th smallest per-fold 90% quantile (>=80% of folds reach 90%)); pooled coverage 0.9362, 4/5 folds at or above 90%, median half-width 0.1958 eV.

Caveat: Each fold's model is trained on the other nine clusters only for the folds retrained here; fold 0 is the production checkpoint (also soap_loco fold 0). Folds share an identical wall-clock budget, so their budgets are equal but shorter than the production run.
