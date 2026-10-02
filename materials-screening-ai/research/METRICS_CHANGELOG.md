# Metrics Changelog & Historical Figure Lineage

This document defines every historical metric reported during development of the $\Delta$-ML band-gap correction module, documenting its target formulation, regularization parameter $\alpha$, feature set, dataset size $N$, and exact reproduction status.

---

## 1. Summary of Canonical Frozen Metrics (`metrics.json`)

All production values are frozen and computed reproducibly by `eval_protocol.py` on the single-fidelity verified experimental optical gap calibration dataset ($N=9$ active, $n=6$ Pb-only perovskites):

| Metric | Canonical Value | Formulation / Method | Feature Set | $N$ |
| :--- | :--- | :--- | :--- | :--- |
| **LOOCV MAE (Ridge $\alpha=1.0$, Full Set)** | **0.4561 eV** | Ridge ($\alpha=1.0$), $\Delta = E_{\text{exp}} - E_{\text{PBE}}$ | 7 physics features (incl. $\epsilon_\infty$, Sanderson A-site) | 9 |
| **Linear PBE Scissor LOCO MAE (Cross-Family)** | **0.6597 eV** | $E_{\text{exp}} = 1.4840 \cdot E_{\text{PBE}} + 0.0251$ (Cross-Family Deployed) | 1 feature ($E_{\text{PBE}}$) | 9 |
| **In-Family Halide Perovskite LOOCV MAE** | **0.1296 eV** | Anion-Matched Mean $\Delta$ (In-Family Deployed Predictor) | Halide grouping ($\text{Cl}: 0.9708, \text{Br}: 0.7813, \text{I}: 0.2992\text{ eV}$) | 6 (Pb-only) |
| **In-Family Halide Perovskite Ridge LOOCV MAE** | **0.1474 eV** | Ridge ($\alpha=1.0$, No $\epsilon_\infty$, Documented Alternative) | 5 physics features | 6 (Pb-only) |
| **In-Family Halide Perovskite Linear PBE LOOCV MAE** | **0.1669 eV** | 2-Parameter Linear PBE Scissor ($E_{\text{exp}} = 1.8863 E_{\text{PBE}} - 0.7708$) | Baseline | 6 (Pb-only) |
| **In-Family Halide Perovskite Constant Scissor MAE** | **0.3077 eV** | Constant Scissor (Mean $\Delta$) | Baseline | 6 (Pb-only) |
| **Held-Out $\text{FAPbI}_3$ Absolute Error** | **0.0848 eV** | Anion-Matched Mean $\Delta$ (vs Ridge 0.2004 eV, Linear 0.1364 eV) | Prediction: 1.5648 eV (Target: 1.48 eV) | Held-out |
| **$\tilde{q}_{\text{in\_family}}$ (Halide Perovskite 80%)** | **0.2157 eV** | Finite-sample order statistic ($k = \lceil 7 \times 0.80 \rceil = 6 \le 6$) | $n=6$ Pb-only | 6 |
| **Held-Out Benchmark Interval Coverage** | **1/1 inside interval** | 1/1 held-out point inside the interval (not a coverage validation) | Interval: $[1.3491, 1.7804]$ eV | 1 |

---

## 2. Retraction of Unverified Mixed-Fidelity $N=21$ Numbers

> [!WARNING]
> **Provenance Retraction Notice (September 2026 Audit)**:
> The previously reported $N=21$ metrics (e.g. LOOCV MAE = 0.3416 eV, LOCO MAE = 0.4517 eV / 0.5975 eV) relied on an unsourced mixed-fidelity calibration set containing:
> 1. Nine (9) unverified targets (NaCl, NaBr, NaI, LiF, NaF, LiCl, CaO, BaO, RbPbBr3) that could not be matched to primary literature source text in the local corpus.
> 2. Two (2) theoretical QSGW+SOC targets (CsSnI3, CsSnBr3) mixed with experimental optical targets.
>
> All $N=21$ numbers are formally **RETRACTED** and archived below for historical traceability. The active calibration model is strictly refitted to the single-fidelity verified experimental optical gap dataset.

### Historical Metric Table & Attribution Lineage

| Figure | Context / Claimed Metric | Historical Configuration | Target | Alpha | $N$ | Reproduction Status / Producing Commit |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **0.3416 eV** / **0.7760 eV** | Retracted Mixed-Fidelity Model | Composition features with Sanderson geometric mean $\chi_{\text{eff}}=2.31$ (MA), 2.40 (FA). | Mixed HSE/Exp/GW | 1.0 | 21 | **Retracted**: Built on unsourced 21-row table; replaced by single-fidelity refit. |
| **0.4517 eV** | Retracted Linear Scissor LOCO | $E_{\text{HSE}} = 1.1370 \cdot E_{\text{PBE}} + 0.7553$ on mixed-fidelity set. | Mixed | n/a | 21 | **Retracted**: $N=21$ mixed-fidelity fit. |
| **0.5975 eV** | Retracted Nested Ridge LOCO | Nested alpha LOCO on mixed-fidelity set. | Mixed | Grid | 21 | **Retracted**: $N=21$ mixed-fidelity fit. |
| **0.2180 eV** | Retracted Perovskite In-Family LOOCV | Perovskite subset ($n=10$) with mixed QSGW/Exp targets. | Mixed | 1.0 | 10 | **Retracted**: Replaced by verified $n=6$ Pb-only experimental subset (0.1296 eV Anion-Matched / 0.1474 eV Ridge). |
| **0.4561 eV** / **1.1896 eV** | Single-Fidelity Experimental Model | Verified experimental optical gaps ($N=9$). | $E_{\text{exp}} - E_{\text{PBE}}$ | 1.0 | 9 | **Canonical Verified Active**: LOOCV 0.4561 eV, LOCO 1.1896 eV. |
| **0.6597 eV** / **1.0190 eV** | Single-Fidelity Deployed Decision | Linear Scissor (0.6597 eV) vs Nested Ridge (1.0190 eV). | $E_{\text{exp}} - E_{\text{PBE}}$ | Grid | 9 | **Canonical Verified Active**: Linear Scissor is cross-family winner. |

---

## 3. Deployed Predictor Rule & Rescoped Scientific Claim

### Fixed Decision Rule
1. **Cross-Family Criterion**: Linear PBE Scissor ($E_{\text{exp}} = 1.4840 \cdot E_{\text{PBE}} + 0.0251$, LOCO MAE **0.6597 eV**) beats Nested Ridge (**1.0190 eV**). Deployed cross-family.
2. **In-Family Criterion & Baseline Rule**:
   - Four baselines evaluated on $n=6$ Pb-only perovskites (LOOCV on identical folds):
     - Constant Scissor: LOOCV MAE = **0.3077 eV**, $\text{FAPbI}_3$ error = **0.4693 eV**
     - 2-Parameter Linear PBE Scissor: LOOCV MAE = **0.1669 eV**, $\text{FAPbI}_3$ error = **0.1364 eV**
     - Ridge ($\alpha=1.0$, No $\epsilon_\infty$): LOOCV MAE = **0.1474 eV**, $\text{FAPbI}_3$ error = **0.2004 eV**
     - Anion-Matched Mean $\Delta$: LOOCV MAE = **0.1296 eV**, $\text{FAPbI}_3$ error = **0.0848 eV**
   - **Rule Application**: Anion-Matched Mean $\Delta$ is within 0.05 eV of Ridge on LOOCV ($|0.1296 - 0.1474| = 0.0178 \le 0.05\text{ eV}$) and has lower error on held-out $\text{FAPbI}_3$ ($0.0848\text{ eV} < 0.2004\text{ eV}$).
   - **Deployment**: Deploy **Anion-Matched Mean $\Delta$** as the primary in-family predictor; retain Ridge as a documented alternative.
   - **Conformal Coverage**: Valid 80% conformal quantile $q_{80} = \mathbf{0.2157\text{ eV}}$. On held-out $\text{FAPbI}_3$, 1/1 held-out point inside the interval (not a coverage validation). $\text{MASnI}_3$ is gated as out-of-domain.

### Scientific Finding
*"On the single-fidelity verified experimental optical gap calibration set (N=9 active, n=6 Pb-only perovskites), a simple anion-matched scissor achieves LOOCV MAE 0.1296 eV and held-out FAPbI3 error 0.0848 eV, outperforming Ridge regression (0.1474 eV LOOCV, 0.2004 eV held-out). Across chemical families, a 2-parameter linear PBE scissor achieves superior generalization (LOCO MAE 0.6597 eV vs 1.0190 eV Ridge). Tin perovskites require a distinct SOC domain gate due to differing relativistic core physics."*
