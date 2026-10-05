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

Pooled coverage at shipped q: **0.7124** (n = 48998); macro-average (unweighted over folds) **0.7840**.
**Nested (leave-one-fold-out) result** — per-fold q fitted on the other 9 folds only: q spread min 1.6178 (cluster 7), median 1.7237, max 1.8165 (cluster 0); pooled coverage **0.8961**, macro **0.9218**; per-fold min 0.7511 (cluster 7), Q1 0.8844, median 0.9272, Q3 0.9964, max 1.0000; median half-width **0.1717 eV**.
Pooled 10-fold 90% q = **1.7242** (non-nested single q over all 10 folds): per-fold coverage min 0.7773 / Q1 0.8865 / median 0.9252 / Q3 0.9951, macro 0.9248, width 0.1724 eV; its nested leave-one-fold-out form is exactly the per-fold q above.
Oracle only (NOT a result, not adopted): shift-aware own-fold q = 1.8143 gives pooled 0.9123 / macro 0.9340 (7/10 folds ≥90% raw, 6/9 excl. n<500) at width 0.1814 eV — a non-nested own-fold statistic.
Nesting: q_cross_conformal for fold c = 90% quantile of the OTHER 9 folds' scores applied to fold c (leave-one-fold-out, nested). shift_aware_q is an oracle-style k-th smallest OWN-fold quantile, not nested.
LOFO (leave-one-fold-out) q spread: min 1.6178, median 1.7237, max 1.8165.

Caveat: Each fold's model is trained on the other nine clusters only for the folds retrained here; fold 0 is the production checkpoint (also soap_loco fold 0). Folds share an identical wall-clock budget, so their budgets are equal but shorter than the production run.
