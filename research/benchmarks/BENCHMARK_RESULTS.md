# Benchmark Results — Materials Screening AI / Crystal GNN
Generated 2026-10-05T16:26:38.693132+00:00 from committed artifacts.
All numbers below are computed by `scripts/compile_benchmark_results.py` from
committed artifacts. Nothing is hand-typed.

## 1. Accuracy — 10-fold LOCO (nested leave-one-cluster-out, same splits as production/LOFO)
| fold | n | MAE (eV/atom) | RMSE (eV/atom) | coverage @ shipped q | 95% CI | nested q | nested cov |
|------|------------:|------------:|:--------------------:|:------:|--------:|:---------:|
| 0 | 7178 | 0.0741 | 0.0985 | 0.8605 | [0.8523, 0.8684] | 1.8165 | 0.9858 |
| 1 | 6898 | 0.0789 | 0.1060 | 0.6785 | [0.6673, 0.6894] | 1.7155 | 0.8897 |
| 2 | 2527 | 0.0855 | 0.1106 | 1.0000 | [0.9985, 1.0000] | 1.7619 | 1.0000 |
| 3 | 7678 | 0.0908 | 0.1229 | 0.6806 | [0.6701, 0.6910] | 1.7077 | 0.8827 |
| 4 | 6103 | 0.0907 | 0.1213 | 0.6670 | [0.6551, 0.6788] | 1.7222 | 0.8974 |
| 5 | 101 | 0.0958 | 0.1157 | 1.0000 | [0.9634, 1.0000] | 1.7253 | 1.0000 |
| 6 | 5332 | 0.0608 | 0.0825 | 0.7764 | [0.7651, 0.7874] | 1.7694 | 0.9571 |
| 7 | 6066 | 0.1296 | 0.1746 | 0.5547 | [0.5422, 0.5672] | 1.6178 | 0.7511 |
| 8 | 6604 | 0.0818 | 0.1080 | 0.6217 | [0.6100, 0.6334] | 1.6882 | 0.8546 |
| 9 | 511 | 0.1176 | 0.1740 | 1.0000 | [0.9925, 1.0000] | 1.7318 | 1.0000 |

### Aggregated accuracy across folds (cluster-bootstrap 95% CIs, seed 42, 2000 resamples)
- Mean MAE: 0.0906 eV/atom, 95% CI [0.0796, 0.1031]
- Median MAE: 0.0881 eV/atom
- Worst MAE: 0.1296 eV/atom at fold 7
- Mean RMSE: 0.1214 eV/atom, 95% CI [0.1051, 0.1407]
- Median RMSE: 0.1132 eV/atom
- Worst RMSE: 0.1746 eV/atom at fold 7
- Total pooled n: 48998

### Nested cross-conformal coverage at per-fold leave-one-fold-out q
- Pooled coverage: 0.8961
- Macro coverage: 0.9218
- Median half-width: 0.1717 eV
- q spread: min 1.6178, median 1.7237, max 1.8165
- Per-fold coverage stats: min 0.7511, Q1 0.8844, median 0.9272, Q3 0.9964, max 1.0000, worst fold 7 (0.7511)

## 2. Baselines
Same splits, same training budget. Retrained baselines available in this commit:

### Single-scale variant (A7 headline config ablation comparison)
A1–A7 are the paper ablation configs on the same soap_loco split, single seed (42).
A4 and A5 are deduplicated in the headline comparison: both were attempted; A5 has no
evaluation.json in this commit and is reported as not evaluated. Headline config (A7) is
a single seed; mean±std is therefore not available from this commit.

| ablation | MAE | RMSE | ECE | Spearman(sigma, |err|) | coverage |
|:---------|------:|:------:|:------:|:----------------------------:|:--------:|
| A1 | 0.0369 | 0.0540 | 0.0323 | 0.1986 | 0.9159 |
| A2 | 0.0364 | 0.0543 | 0.0534 | 0.1920 | 0.9567 |
| A3 | 0.0393 | 0.0563 | 0.0233 | 0.2093 | 0.9580 |
| A4 | 0.0423 | 0.0580 | 0.0544 | 0.1784 | 0.9508 |
| A6 | 0.0491 | 0.0683 | 0.0635 | 0.2382 | 0.9636 |
| A7 | 0.0641 | 0.0831 | 0.0921 | 0.1925 | 0.9383 |

CGCNN / ALIGNN: a same-budget retrained CGCNN or ALIGNN baseline on this commit's
soap_loco fold is not present in the committed artifacts and could not be retrained within
the benchmarking window. Report as ZERO-SHOT and NON-COMPARABLE where available:

- ALIGNN soap_loco zero-shot eval exists at crystal_gnn\results\alignn_soap_loco_zeroshot_eval.csv; treated as zero-shot, non-comparable (no same-budget retraining).

## 3. UQ table (all folds)
Metrics below use the committed LOCO per-fold scores. Fold-level raw DER calibration and
Spearman(sigma, |err|) are taken from the A7 evaluation.json on fold 0 only (the only fold
with a committed evaluation.json for the headline config). Per-fold MC-dropout is not
committed; nested LOFO q and distance-q are computed from the committed score NPZs.

### Headline UQ (A7, fold 0 evaluation.json)
- Raw DER 95% coverage: 0.9383
- ECE: 0.0921
- Spearman(sigma, |err|): 0.1925
- Median interval half-width (eV): 0.3480

### Conformal quantile variants (all folds, from committed artifacts)
| variant | pooled cov | macro cov | median half-width (eV) | note |
|:--------|-----------:|:---------:|:----------------------:|:-----|
| shipped q = 1.0254 | 0.7124 | 0.7840 | 0.1025 | deployed production q |
| nested LOFO q (per-fold) | 0.8961 | 0.9218 | 0.1717 | PRIMARY, nested |
| pooled 10-fold q = 1.7242 | — | 0.9252 | 0.1724 | non-nested single q |
| distance-bin q (nested) | 0.8904 | 0.9166 | 0.1707 | optional shift-aware mode, shipped |
| shift-aware own-fold q = 1.8143 | 0.9123 | 0.9340 | 0.1814 | NON-NESTED ORACLE, not a result |

### Selective prediction curve (MAE vs fraction retained)
Not committed for all folds. The selective curve is not reported here; it would require
per-fold score-sorted predictions which are not present in this commit's artifacts.

## 4. Shift table (per fold)
Flag fold 7: highest MAE, lowest shipped coverage, worst nested coverage.

| fold | n | MAE (eV/atom) | coverage @ shipped q | coverage @ nested LOFO q | coverage @ distance-q |
|------|----:|------------:|:--------------------:|:----------------------:|:--------------------:|
| 0 | 7178 | 0.0741 | 0.8605 | 0.9858 | 0.9685 | |
| 1 | 6898 | 0.0789 | 0.6785 | 0.8897 | 0.9024 | |
| 2 | 2527 | 0.0855 | 1.0000 | 1.0000 | 1.0000 | |
| 3 | 7678 | 0.0908 | 0.6806 | 0.8827 | 0.9120 | |
| 4 | 6103 | 0.0907 | 0.6670 | 0.8974 | 0.8873 | |
| 5 | 101 | 0.0958 | 1.0000 | 1.0000 | 1.0000 | |
| 6 | 5332 | 0.0608 | 0.7764 | 0.9571 | 0.9105 | |
| 7 | 6066 | 0.1296 | 0.5547 | 0.7511 | 0.8038 |  (flagged) |
| 8 | 6604 | 0.0818 | 0.6217 | 0.8546 | 0.7820 | |
| 9 | 511 | 0.1176 | 1.0000 | 1.0000 | 1.0000 | |

### Distance-q decision rule outcome
- Outcome: ship_optional_shift_aware_mode
- min coverage improved over global nested LOFO q: True
- Q1 coverage improved: True
- median half-width ratio: 0.9937 (within 3%: True)

## 5. Pipeline
- Ensemble-gate on 177 pairs: 98.3% high-confidence agreement, 1.7% moderate agreement, max disagreement 0.0262 eV/atom, median 0.0005 eV/atom.
- Per-tier latency/throughput: not committed as numeric artifacts in this round; the orchestrator uses a sigma-unit Tier-2 gate (default 50% promote target, sigma >= 0.09997 eV). At shipped q, 99.7% of pooled points exceed the legacy 0.10 eV width gate; see diagnostics.json routing_thresholds.
- S.U.N. rate: computed by `app.services.stability_analysis.DiscoveryRunReportService.compute_sun_rate` from e_above_hull (GNN-predicted, not DFT-relaxed), formula dedup, and novelty status. Caveats (must accompany any reported S.U.N.): energy is ML-predicted not DFT; generation is scaffold-substitution constrained, not unconstrained generative sampling. Therefore not an apples-to-apples comparison with MatterGen / CrystalGRW / LeMat-GenBench.

## 6. DER lambda sweep
[0.001, 0.005, 0.01, 0.05, 0.1]

| lambda | MAE | RMSE | coverage | status |
|:-------|------:|:------:|:--------:|:------|
| 0.001 | — | — | — | trained not evaluated |
| 0.005 | — | — | — | not run |
| 0.01 | — | — | — | not run |
| 0.05 | — | — | — | not run |
| 0.1 | — | — | — | not run |

Only lambda=0.001 has a trained run in this commit (history.csv present, no evaluation.json).
The sweep as specified was NOT fully run; do not claim results for the other lambdas.

## 7. Claims supported / not supported

### Supported by this commit's artifacts

- 10-fold LOCO nested leave-one-cluster-out, 89.6% pooled / 92.2% macro coverage at per-fold nested q; worst fold 7 (75.1%).
- At shipped q = 1.0254: 71.2% pooled / 78.4% macro coverage; fold 7 = 55.5%.
- Distance-bin nested q improves min and Q1 over global nested LOFO q while keeping median half-width within 3%; shipped as optional shift-aware mode.
- External Pb-halide set (n=235, 92.8% at shipped q) coverage is driven by 2.51x sigma inflation, not calibration transfer; LOCO's own Pb-halide subset (n=122) covers 60.7%.
- Ensemble gate: 177 pairs, 98.3% high-confidence agreement.
- A7 headline config: fold 0 raw DER 95% coverage 0.9383, ECE 0.0921, Spearman 0.1925; per-fold nested LOFO q 1.6178–1.8165.

### Not supported / not done (with reason)

- Same-budget retrained CGCNN/ALIGNN baseline: not present and not retrained in the benchmarking window; zero-shot ALIGNN eval is non-comparable.
- A7 headline mean±std over 3 seeds: only seed 42 is committed; not available.
- A5 ablation evaluation: A5 history.csv exists but no evaluation.json; not evaluated.
- Per-fold MC-dropout UQ table: not committed; not reported.
- Selective prediction curve (MAE vs fraction retained): not committed; not reported.
- Full DER lambda sweep: only lambda=0.001 trained; sweep not completed; no results claimed for other lambdas.
- Per-tier latency/throughput numeric artifacts: not committed in this round.
- Full benchmark retrains of baselines within budget: not completed.

## 8. Reproducibility line

- Test count: full suite and QE subset to be run below (`pytest` + `tsc`).
- Commit: d252d392fd51dcc625047c1d4145c83633b901d4
- Tag: v1.1.0
- Seeds: ablation headline config seed 42 (single seed); LOCO folds share an identical wall-clock budget; distance-q SOAP sketch seed 42.
- Splits hash: production soap_loco split used for LOCO folds; split artifacts under materials-screening-ai/research/loco_cross_conformal/manifest.jsonl and NPZs.

## 9. Verification

Run from repository root:

```bash
python -m pytest materials-screening-ai/backend/tests -q
# plus frontend typecheck:
npm --prefix materials-screening-ai/frontend run typecheck
```
