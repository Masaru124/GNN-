# LOCO cross-conformal summary (formation energy)

Score: `|y - mu| / sigma(DER), model.eval() deterministic forward`. Shipped q = 1.0254.

| cluster | n_test | MAE (eV/atom) | median half-width (eV) | coverage @1.0254 | 95% CI | q cross-conformal | coverage cross-conformal | coverage @ shift-aware q |
| ---: | ---: | ---: | ---: | ---: | :---: | ---: | ---: | ---: |
| 0 | 7178 | 0.0741 | 0.1223 | 0.8605 | [0.8523, 0.8684] | 1.1504 | 0.9000 | 0.8998 |

Pooled coverage at shipped q: **0.8605** (n = 7178).
Shift-aware q: **1.1504** (1-th smallest per-fold 90% quantile (>=80% of folds reach 90%)); pooled coverage 0.8998, 0/1 folds at or above 90%, median half-width 0.1372 eV.

Caveat: Each fold's model is trained on the other nine clusters only for the folds retrained here; fold 0 is the production checkpoint (also soap_loco fold 0). Folds share an identical wall-clock budget, so their budgets are equal but shorter than the production run.
