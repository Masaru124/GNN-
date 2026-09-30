# Metrics Changelog & Historical Figure Lineage

This document defines every historical metric reported during development of the $\Delta$-ML band-gap correction module, documenting its target formulation, regularization parameter $\alpha$, feature set, dataset size $N$, and exact reproduction status.

---

## 1. Summary of Canonical Frozen Metrics (`metrics.json`)

All production values are frozen and computed reproducibly by `eval_protocol.py` on the single-fidelity verified experimental optical gap calibration dataset ($N=10$):

| Metric | Canonical Value | Formulation / Method | Feature Set | $N$ |
| :--- | :--- | :--- | :--- | :--- |
| **LOOCV MAE (Ridge $\alpha=1.0$)** | **0.4561 eV** | Ridge ($\alpha=1.0$), $\Delta = E_{\text{exp}} - E_{\text{PBE}}$ | 7 physics features (incl. $\epsilon_\infty$, Sanderson A-site) | 10 |
| **Linear PBE Scissor LOCO MAE** | **0.6597 eV** | $E_{\text{exp}} = 1.4840 \cdot E_{\text{PBE}} + 0.0251$ (Cross-Family Deployed) | 1 feature ($E_{\text{PBE}}$) | 10 |
| **Pooled LOCO MAE (Ridge $\alpha=1.0$)** | **1.1896 eV** | Leave-One-Chemistry-Out ($\alpha=1.0$) | 7 physics features | 10 |
| **Nested LOCO MAE (Extended $\alpha$ Grid)** | **1.0190 eV** | Inner LOCO $\alpha$-selection across $[10^{-4}, 10^5]$ | 7 physics features | 10 |
| **Paired $\Delta\text{MAE}$ (Nested Ridge vs Linear PBE)** | **+0.3593 eV** | Percentile compound bootstrap (95% CI: $[-0.1352, +0.7390]$ eV) | $N=10$ compounds | 10 |
| **In-Family Halide Perovskite LOOCV MAE** | **0.2683 eV** | Ridge ($\alpha=1.0$), In-Family Deployed Predictor | 7 physics features | 7 |
| **In-Family Halide Perovskite Linear PBE LOOCV MAE** | **0.6245 eV** | Linear PBE Scissor within halide perovskites | Baseline | 7 |
| **In-Family Halide Perovskite Family-Mean-$\Delta$** | **0.4614 eV** | Leave-one-out within halide perovskite family | Baseline | 7 |
| **In-Family Ridge Advantage over Linear Scissor** | **0.3562 eV** | 57.0% relative error reduction over linear scissor | In-family comparison | 7 |
| **In-Family Ridge Advantage over Family-Mean-$\Delta$** | **0.1931 eV** | 41.8% relative error reduction over family mean | In-family comparison | 7 |
| **$\tilde{q}_{\text{in\_family}}$ (Halide Perovskite)** | **0.7273 eV** | $s_i = \|e_i\| / \sqrt{1 + h_{i,\text{heldout}}}$ on in-family residuals | Query-normalized | 7 |
| **$\tilde{q}_{\text{cross\_family}}$ (Linear PBE Scissor)** | **0.7273 eV** | Pooled conformal quantile ($k = \lceil 11 \times 0.90 \rceil = 10$) | Query-normalized | 10 |

---

## 2. Retraction of Unverified Mixed-Fidelity $N=21$ Numbers

> [!WARNING]
> **Provenance Retraction Notice (September 2026 Audit)**:
> The previously reported $N=21$ metrics (e.g. LOOCV MAE = 0.3416 eV, LOCO MAE = 0.4517 eV / 0.5975 eV) relied on an unsourced mixed-fidelity calibration set containing:
> 1. Nine (9) unverified targets (NaCl, NaBr, NaI, LiF, NaF, LiCl, CaO, BaO, RbPbBr3) that could not be matched to primary literature source text in the local corpus.
> 2. Two (2) theoretical QSGW+SOC targets (CsSnI3, CsSnBr3) mixed with experimental optical targets.
>
> All $N=21$ numbers are formally **RETRACTED** and archived below for historical traceability. The active calibration model is strictly refitted to the $N=10$ verified experimental optical gap dataset.

### Historical Metric Table & Attribution Lineage

| Figure | Context / Claimed Metric | Historical Configuration | Target | Alpha | $N$ | Reproduction Status / Producing Commit |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **0.3416 eV** / **0.7760 eV** | Retracted Mixed-Fidelity Model | Composition features with Sanderson geometric mean $\chi_{\text{eff}}=2.31$ (MA), 2.40 (FA). | Mixed HSE/Exp/GW | 1.0 | 21 | **Retracted**: Built on unsourced 21-row table; replaced by $N=10$ single-fidelity refit. |
| **0.4517 eV** | Retracted Linear Scissor LOCO | $E_{\text{HSE}} = 1.1370 \cdot E_{\text{PBE}} + 0.7553$ on mixed-fidelity set. | Mixed | n/a | 21 | **Retracted**: $N=21$ mixed-fidelity fit. |
| **0.5975 eV** | Retracted Nested Ridge LOCO | Nested alpha LOCO on mixed-fidelity set. | Mixed | Grid | 21 | **Retracted**: $N=21$ mixed-fidelity fit. |
| **0.2180 eV** | Retracted Perovskite In-Family LOOCV | Perovskite subset ($n=10$) with mixed QSGW/Exp targets. | Mixed | 1.0 | 10 | **Retracted**: Replaced by verified $n=7$ experimental perovskite subset (0.2683 eV). |
| **0.4561 eV** / **1.1896 eV** | Single-Fidelity Experimental Model | Verified experimental optical gaps ($N=10$). | $E_{\text{exp}} - E_{\text{PBE}}$ | 1.0 | 10 | **Canonical Verified Active**: LOOCV 0.4561 eV, LOCO 1.1896 eV. |
| **0.6597 eV** / **1.0190 eV** | Single-Fidelity Deployed Decision | Linear Scissor (0.6597 eV) vs Nested Ridge (1.0190 eV). | $E_{\text{exp}} - E_{\text{PBE}}$ | Grid | 10 | **Canonical Verified Active**: Linear Scissor is cross-family winner. |

---

## 3. Deployed Predictor Rule & Rescoped Scientific Claim

### Fixed Decision Rule
1. **Primary Criterion**: Nested LOCO MAE with paired compound-level bootstrap $\Delta\text{MAE}$ vs. Linear PBE Scissor ($E_{\text{exp}} = a \cdot E_{\text{PBE}} + b$, $N=10$, 95% CI).
2. **Outcome**: Nested Ridge achieves LOCO MAE **1.0190 eV** vs. Linear PBE Scissor **0.6597 eV** ($\Delta\text{MAE} = +0.3593\text{ eV}$, 95% CI $[-0.1352, +0.7390]\text{ eV}$). Ridge is not significantly better cross-family.
3. **Deployment**:
   - **Out-of-Family / Cross-Family**: Deploy **Linear PBE Scissor** ($E_{\text{exp}} = 1.4840 \cdot E_{\text{PBE}} + 0.0251$, LOCO MAE **0.6597 eV**).
   - **In-Family (Halide Perovskites $n=7$)**: Deploy **Ridge ($\alpha=1.0$)** because it beats linear scissor by **0.3562 eV** (LOOCV MAE **0.2683 eV** vs **0.6245 eV**) and beats family-mean-$\Delta$ by **0.1931 eV** (LOOCV MAE **0.2683 eV** vs **0.4614 eV**).
   - Note: Because all individual families have $n < 9$ in the $N=10$ dataset, conformal intervals fall back to pooled calibration with $k = \lceil 11 \times 0.90 \rceil = 10$.

### Scientific Finding
*"On the single-fidelity verified experimental optical gap calibration set (N=10), no Δ-ML variant beats a linear PBE scissor cross-family (LOCO MAE 0.6597 eV vs 1.0190 eV); in-family, Ridge regression provides substantial gains exclusively for halide perovskites (LOOCV MAE 0.2683 eV vs 0.6245 eV linear scissor). Conformal prediction intervals use pooled calibration due to family sample size limitations (all n_fam < 9)."*
