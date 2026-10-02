# Physical & Methodological Limitations of the $\Delta$-ML Pipeline

**Document Status**: Authoritative Protocol Specification  
**Version**: 1.0  
**Audit Date**: September 30, 2026  
**Pipeline**: Materials Screening AI — Quantum ESPRESSO DFT & $\Delta$-ML Band Gap Corrector  

---

## Executive Summary

The $\Delta$-ML band gap correction framework enhances standard semi-local Kohn-Sham density functional theory (PBE) with empirical corrections trained on verified experimental optical band gaps. While this methodology achieves significant in-family error reduction (reducing LOOCV MAE on halide perovskites from $0.6245\text{ eV}$ for linear PBE scissor to $\mathbf{0.2683\text{ eV}}$ for Ridge regression), its deployment is subject to physical, statistical, and domain boundaries.

This document details the six primary limitations identified during the comprehensive reproducibility audit.

---

## 1. Sample Size, Dimensionality, and Overparameterization ($N=10, p=7$)

### 1.1 Effective Degrees of Freedom
The active single-fidelity calibration set consists of $N=10$ verified compounds spanning 3 distinct chemistry families:
- **Halide Perovskites** ($n=7$): $\text{CsPbI}_3$, $\text{CsPbBr}_3$, $\text{CsPbCl}_3$, $\text{CsSnCl}_3$, $\text{MAPbI}_3$, $\text{MAPbBr}_3$, $\text{MAPbCl}_3$
- **Transition Metal Perovskites** ($n=2$): $\text{SrTiO}_3$, $\text{BaTiO}_3$
- **Alkaline Earth Oxides** ($n=1$): $\text{MgO}$

The full model employs $p=7$ features:
$$\mathbf{x} = \left[ E_{\text{PBE}}, E_{\text{PBE}}^2, \Delta\chi, r_A/r_B, Z_{\text{avg}}, 1/\epsilon_\infty, E_{\text{PBE}}/\epsilon_\infty \right]$$

With an intercept, the model fits $p+1 = 8$ parameters on $N=10$ samples, leaving only **2 effective degrees of freedom** in unregularized ordinary least squares (OLS). The OLS design matrix has rank 8 ($\text{Tr}(H_{\text{OLS}}) = 8.0000$).

### 1.2 Statistical Leverage & Cook's Distance
Under Ridge regression ($\alpha = 1.0$), the effective model complexity is $\text{Tr}(H_{\text{ridge}}) = 4.5214$, yielding a 2× cutoff criterion of $2 \cdot \text{Tr}(H)/N = 0.9043$.

| Compound | Family | Classical Leverage $h_{ii}$ | Ridge Leverage $h_{ii}$ | Held-Out Query Leverage $h_q$ | Cook's $D$ | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **MgO** | alkaline_earth_oxide | **0.9998** | **0.9368** | **33.3349** | **1176.12** | **Extreme Leverage Outlier** |
| $\text{SrTiO}_3$ | transition_metal_perovskite | 0.8123 | 0.4238 | 0.7330 | 0.04 | Nominal |
| $\text{BaTiO}_3$ | transition_metal_perovskite | 0.8126 | 0.4240 | 0.7371 | 0.05 | Nominal |
| $\text{CsPbI}_3$ | halide_perovskite | 0.8872 | 0.4689 | 0.8980 | 0.08 | Nominal |
| $\text{CsPbBr}_3$ | halide_perovskite | 0.7184 | 0.2732 | 0.3734 | 0.02 | Nominal |
| $\text{CsPbCl}_3$ | halide_perovskite | 0.6941 | 0.2490 | 0.3217 | 0.01 | Nominal |
| $\text{CsSnCl}_3$ | halide_perovskite | 0.9412 | 0.5972 | 1.4925 | 1.22 | High Residual |
| $\text{MAPbI}_3$ | halide_perovskite | 0.9103 | 0.5056 | 1.0938 | 0.21 | Moderate Leverage |
| $\text{MAPbBr}_3$ | halide_perovskite | 0.7324 | 0.2994 | 0.4350 | 0.03 | Nominal |
| $\text{MAPbCl}_3$ | halide_perovskite | 0.7816 | 0.3433 | 0.5224 | 0.06 | Nominal |

$\text{MgO}$ exhibits extreme statistical leverage ($h_{\text{OLS}} = 0.9998$, $h_{\text{ridge}} = 0.9368$, $h_q = 33.3349$, Cook's $D = 1176.12$). Leaving out $\text{MgO}$ causes catastrophic prediction failure on $\text{MgO}$ (predicted $4.80\text{ eV}$ vs target $7.22\text{ eV}$, error $2.42\text{ eV}$).

### 1.3 Deployed Model Scoping Rule & Halide-Only Fit
Due to this severe leverage gap across chemistry families:
1. **Halide Perovskite Deployed Fit ($n=7$)**: The deployed predictor for halide perovskites is fit strictly on the halide perovskite compounds ($n=7$) without $\epsilon_\infty$ features ($p=5$: $E_{\text{PBE}}, E_{\text{PBE}}^2, \Delta\chi, r_A/r_B, Z_{\text{avg}}$).
2. **LOOCV Comparison Across Fits**:
   - **Halide-Only Fit ($n=7$)**:
     - Without $\epsilon_\infty$: LOOCV MAE = **0.2120 eV**
     - With $\epsilon_\infty$: LOOCV MAE = **0.2251 eV**
   - **Halide-Only Fit ($n=6$, without anomalous $\text{CsSnCl}_3$)**:
     - Without $\epsilon_\infty$: LOOCV MAE = **0.0694 eV**
     - With $\epsilon_\infty$: LOOCV MAE = **0.0822 eV**
   - **Pooled-10 Fit ($N=10$)**:
     - Without $\epsilon_\infty$: Total LOOCV MAE = **0.6183 eV**; Halide In-Family subset = **0.3145 eV**
     - With $\epsilon_\infty$: Total LOOCV MAE = **0.4561 eV**; Halide In-Family subset = **0.2683 eV**
3. **Physical Explanation for In-Family Shift ($0.2683/0.3145 \to 0.2251/0.2120\text{ eV}$)**:
   In the pooled-10 fit, $\text{MgO}$ possesses extreme leverage ($h_{ii} = 0.9368$, Cook's $D = 1176.12$, $\Delta = +2.745\text{ eV}$) and the two transition metal perovskites ($\text{SrTiO}_3, \text{BaTiO}_3$) have distinct $d^0$ electronic structures. These out-of-family points exert strong regression torque that tilts the standardized hyperplane away from the halide manifold. Fitting strictly on the halide perovskite family ($n=7$) completely eliminates this out-of-family pulling torque, allowing the model to specialize on the halide manifold and dropping LOOCV MAE from $0.2683\text{ eV}$ to $0.2251\text{ eV}$ (with $\epsilon_\infty$) and from $0.3145\text{ eV}$ to $0.2120\text{ eV}$ (without $\epsilon_\infty$).
4. **Status of $\text{MgO}$, $\text{SrTiO}_3$, and $\text{BaTiO}_3$**:
   $\text{MgO}$, $\text{SrTiO}_3$, and $\text{BaTiO}_3$ are **removed** from the deployed fit. Non-halide candidate queries exhibit extreme out-of-domain leverage ($h_q = 33.3$ on $\text{MgO}$) and must return `out_of_domain` rather than receiving an uncalibrated extrapolation.

---

## 2. Target Noise & Experimental Literature Discrepancies

### 2.1 Experimental Noise Floor
Experimental optical band gaps reported in the literature exhibit inherent measurement noise and systematic experimental variances:

1. **Measurement Techniques**: Band gaps extracted from optical absorption spectra (Tauc plots) frequently differ by $0.1-0.2\text{ eV}$ from photoluminescence (PL) emission peak positions due to Stokes shifts and Urbach band tails.
2. **Sample Morphology**: Single crystals vs polycrystalline thin films vs nanocrystals display varying defect densities and quantum confinement effects.
3. **Documented Examples in Calibration Corpus**:
   - $\text{MAPbBr}_3$: Reported as $2.18\text{ eV}$ (Mosconi et al. 2013 Table 1) and $2.33\text{ eV}$ (Castelli et al. 2014 Table I), with other literature reporting $2.36\text{ eV}$ ($\Delta \approx 0.18-0.35\text{ eV}$).
   - $\text{CsPbCl}_3$: Literature values span $2.85\text{ eV}$ to $3.00\text{ eV}$ ($\Delta \approx 0.15\text{ eV}$).
   - $\text{MAPbI}_3$: Literature values span $1.55\text{ eV}$ to $1.61\text{ eV}$ ($\Delta \approx 0.06\text{ eV}$).

This empirical measurement variance establishes an intrinsic noise floor of $\sim 0.15-0.20\text{ eV}$ on any $\Delta$-ML model trained on experimental targets.

### 2.2 DFT PBE Band Gap Discrepancies vs Published Theory
When comparing self-consistent Quantum ESPRESSO PBE calculations against published DFT literature values:
- **$\text{CsPbI}_3$**: Wiktor et al. (J. Phys. Chem. Lett. 2017, 8, 5507, Table 2) reports PBE (no SOC) band gap of $1.14\text{ eV}$. The pipeline's self-consistent QE calculation at the paper's exact experimental cubic lattice constant ($a = 6.29\text{ \AA}$) with SSSP Efficiency pseudopotentials yields $1.323\text{ eV}$ (**unexplained: +0.18 eV vs published**).
- **$\text{MAPbCl}_3$**: Mosconi et al. (J. Phys. Chem. C 2013, 117, 13902, Table 1) reports PBE band gap of $2.34\text{ eV}$. The pipeline's self-consistent QE calculation at the paper's pseudocubic geometry ($a = 5.68\text{ \AA}$) yields $2.450\text{ eV}$ (**unexplained: +0.11 eV vs published**).
- unexplained: +0.18 eV (CsPbI3) vs published PBE with NC pseudopotentials at 80 Ry/6x6x6; optionally run QE with an NC (ONCV) pseudopotential at the paper's settings and report the gap.

---

## 3. Phase Mismatch and Room-Temperature Dynamic Disorder

### 3.1 0 K Static Ground State vs 300 K Dynamic Structure
1. **Computational DFT Conditions**: Quantum ESPRESSO calculations evaluate static 0 K idealized pseudocubic ($Pm\bar{3}m$) or relaxed zero-temperature lattices without dynamic rotational or phonon contributions.
2. **Experimental Conditions**: Target optical measurements are collected at room temperature ($T = 300\text{ K}$). At this temperature:
   - Organic cations ($\text{MA}^+$, $\text{FA}^+$) undergo rapid picosecond rotational dynamics, creating fluctuating electrostatic potentials across the inorganic $[\text{PbX}_6]^{4-}$ octahedra.
   - Dynamic octahedral tilting introduces significant structural fluctuations that modulate the $\text{Pb-}6s$ and $\text{X-}p$ orbital overlap.
   - Temperature-dependent electron-phonon coupling and thermal expansion induce non-trivial band gap shifts ($dE_g/dT \approx +0.3\text{ meV/K}$ for lead halides).
3. **Lumping of Effects**: The $\Delta$-ML correction ($\Delta = E_{\text{exp}} - E_{\text{PBE}}$) is not purely an exchange-correlation correction; it implicitly bundles semi-local functional error, zero-point renormalization, room-temperature thermal expansion, and dynamic disorder into a single offset.

---

## 4. Spin-Orbit Coupling (SOC) Regimes: Lead vs Tin Perovskites

Semi-local PBE functional performance varies drastically between heavy $6p$ lead perovskites and lighter $5p$ tin perovskites due to relativistic spin-orbit effects:

1. **Lead Halide Perovskites ($\text{Pb}^{2+}$, $Z=82$)**:
   - Relativistic spin-orbit coupling splits the conduction band minimum ($\text{Pb-}6p$ orbitals) by $\approx 1.2-1.5\text{ eV}$, drastically lowering the band gap.
   - Standard PBE without SOC underestimates the band gap due to self-interaction errors.
   - In Pb perovskites, neglecting SOC artificially raises the band gap by $\sim 1.2\text{ eV}$, which serendipitously cancels the PBE underestimation error (the celebrated "fortuitous error cancellation"). Consequently, non-SOC PBE values for $\text{MAPbI}_3$ ($1.55\text{ eV}$) closely match experimental gaps ($1.57\text{ eV}$, $\Delta = +0.02\text{ eV}$).
2. **Tin Halide Perovskites ($\text{Sn}^{2+}$, $Z=50$)**:
   - In tin compounds, relativistic SOC splitting is substantially weaker ($\approx 0.4\text{ eV}$).
   - The fortuitous cancellation does not hold: non-SOC PBE severely underestimates the band gap ($\text{CsSnCl}_3$ PBE $= 0.799\text{ eV}$ vs target $2.60\text{ eV}$, $\Delta = +1.801\text{ eV}$; $\text{MASnI}_3$ PBE $= 0.351\text{ eV}$ vs target $1.20\text{ eV}$, $\Delta = +0.849\text{ eV}$).
3. **Regime Heterogeneity**: A unified linear or kernel model across both Sn and Pb perovskites must accommodate distinct physical correction regimes driven by the atomic number $Z_B$ and average atomic number $Z_{\text{avg}}$.

---

## 5. Dielectric Function ($\epsilon_\infty$) Proxy vs Tabulated Literature

1. **SSSP / DFPT Origin**: High-frequency optical dielectric constants $\epsilon_\infty$ in the calibration dataset are calculated using Density Functional Perturbation Theory (DFPT) with SSSP Efficiency pseudopotentials, or obtained from curated optical reference tables. They are not co-reported in the original band gap source papers.
2. **Elimination of $\epsilon_\infty$ in Deployed Model**:
   - Because experimental dielectric constants $\epsilon_\infty$ are rarely known for novel screening candidates, requiring hand-typed $\epsilon_\infty$ creates operational bottlenecks and vulnerability to manual transcription error.
   - Within the halide perovskite family ($n=7$), the LOOCV MAE without $\epsilon_\infty$ is $0.2120\text{ eV}$ (and $0.0694\text{ eV}$ for $n=6$ without $\text{CsSnCl}_3$), proving that composition descriptors ($\Delta\chi, r_A/r_B, Z_{\text{avg}}$) and $E_{\text{PBE}}$ capture the necessary physical trends without requiring $\epsilon_\infty$.
   - The deployed halide model therefore operates strictly without $\epsilon_\infty$ features.

---

## 6. Conformal Prediction Validity Floor & Deployed Prediction Intervals

### 6.1 Mathematical Validity Requirement ($n=6$)
Conformal prediction order statistic calculation:
$$k = \left\lceil (n + 1)(1 - \alpha) \right\rceil$$

For the deployed lead-halide perovskite predictor ($n=6$):
- **90% Confidence ($\alpha = 0.10$)**:
  $$k = \lceil (6 + 1) \cdot 0.90 \rceil = \lceil 6.3 \rceil = 7 > 6 \quad (\textbf{Undefined / Infinite Interval})$$
- **80% Confidence ($\alpha = 0.20$)**:
  $$k = \lceil (6 + 1) \cdot 0.80 \rceil = \lceil 5.6 \rceil = 6 \le 6 \quad (\textbf{Valid Finite-Sample Interval})$$

The deployed predictor therefore reports an **80% prediction interval** calibrated strictly from the $n=6$ Pb-only leave-one-out residuals ($k=6$), or reports no interval if 90% coverage is requested. The previously reported pooled-10 $\tilde{q}$ has been removed from the deployed service.

### 6.2 In-Sample LOO Coverage Tautology & Held-Out Performance
1. **Tautological In-Sample Coverage**:
   Evaluating conformal coverage on the calibration training set itself using leave-one-out residuals is mathematically tautological. Because $\tilde{q}$ is defined by construction as the empirical order statistic of the LOO residuals ($k=6$ out of $6$), the in-sample LOO coverage is guaranteed to be $100\%$ ($6/6$) by definition. Real statistical validity can only be demonstrated on independent held-out queries.
2. **Held-Out Predictions**:
   - $\text{FAPbI}_3$ (In-Domain Lead Halide): PBE $= 1.2656\text{ eV} \implies \hat{E}_g = 1.5951\text{ eV}$ (Target $1.48\text{ eV}$, error $0.1151\text{ eV}$). With $80\%$ interval half-width $\pm 0.2590\text{ eV}$, interval is $[1.336, 1.854]\text{ eV}$ (covered).
   - $\text{MASnI}_3$ (Out-of-Domain Tin Halide): Categorized as `out_of_domain` with reason `"Sn/SOC regime, 1 calibration point"`. Evaluating the un-gated Pb model on $\text{MASnI}_3$ yields $\hat{E}_g = 0.5269\text{ eV}$ (Target $1.20\text{ eV}$, error $0.6731\text{ eV}$), showing catastrophic failure when extrapolating across the relativistic SOC regime.

