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

### 1.3 Deployed Model Scoping Rule
Due to this severe leverage gap across chemistry families:
1. **Inside Halide Perovskites**: The $p=7$ Ridge model is retained because it outperforms linear scissor by $0.3562\text{ eV}$ and captures chemical trends across halide substitutions ($\text{I} \to \text{Br} \to \text{Cl}$) and cation substitutions ($\text{Cs} \to \text{MA} \to \text{FA}$).
2. **Outside Halide Perovskites**: The system must NOT extrapolate using the $p=7$ Ridge model. Any candidate outside the halide perovskite chemistry domain must return `out_of_domain` or fall back to the global linear PBE scissor ($E_{\text{exp}} = 1.4840 E_{\text{PBE}} + 0.0251$).

---

## 2. Target Noise & Experimental Literature Discrepancies

Experimental optical band gaps reported in the literature exhibit inherent measurement noise and systematic experimental variances:

1. **Measurement Techniques**: Band gaps extracted from optical absorption spectra (Tauc plots) frequently differ by $0.1-0.2\text{ eV}$ from photoluminescence (PL) emission peak positions due to Stokes shifts and Urbach band tails.
2. **Sample Morphology**: Single crystals vs polycrystalline thin films vs nanocrystals display varying defect densities and quantum confinement effects.
3. **Documented Examples in Calibration Corpus**:
   - $\text{MAPbBr}_3$: Reported as $2.18\text{ eV}$ (Mosconi et al. 2013 Table 1) and $2.33\text{ eV}$ (Castelli et al. 2014 Table I), with other literature reporting $2.36\text{ eV}$ ($\Delta \approx 0.18-0.35\text{ eV}$).
   - $\text{CsPbCl}_3$: Literature values span $2.85\text{ eV}$ to $3.00\text{ eV}$ ($\Delta \approx 0.15\text{ eV}$).
   - $\text{MAPbI}_3$: Literature values span $1.55\text{ eV}$ to $1.61\text{ eV}$ ($\Delta \approx 0.06\text{ eV}$).

This empirical measurement variance establishes an intrinsic noise floor of $\sim 0.15-0.20\text{ eV}$ on any $\Delta$-ML model trained on experimental targets.

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
2. **Descriptor Sensitivity**:
   - Omitting $\epsilon_\infty$ ($p=5$ features: $E_{\text{PBE}}, E_{\text{PBE}}^2, \Delta\chi, r_A/r_B, Z_{\text{avg}}$) increases full-sample LOOCV MAE from $0.4561\text{ eV}$ to $0.6183\text{ eV}$ (+0.1622 eV penalty).
   - Within the halide perovskite family ($n=7$), the LOOCV MAE without $\epsilon_\infty$ is $0.2120\text{ eV}$, demonstrating that composition and PBE gap features carry the dominant predictive signal for in-family trends.
3. **Operational Impact**: For candidate screening where DFPT response calculations are computationally expensive, the pipeline utilizes dielectric proxy estimates or composition models.

---

## 6. Conformal Prediction Validity Floor ($n_{\text{cal}} \ge 9$)

### 6.1 Mathematical Validity Requirement
Split-conformal prediction at significance level $\alpha = 0.10$ requires computing the order statistic:
$$k = \left\lceil (n_{\text{cal}} + 1)(1 - \alpha) \right\rceil = \left\lceil (n_{\text{cal}} + 1) \cdot 0.90 \right\rceil$$

For a finite, valid prediction interval, the $k$-th order statistic must exist within the calibration set:
$$k \le n_{\text{cal}} \iff \left\lceil (n_{\text{cal}} + 1) \cdot 0.90 \right\rceil \le n_{\text{cal}}$$

Evaluating this inequality yields the exact integer threshold:
- $n_{\text{cal}} = 8 \implies k = \lceil 9 \times 0.90 \rceil = \lceil 8.1 \rceil = 9 > 8$ (**Invalid / Infinite Interval**)
- $n_{\text{cal}} = 9 \implies k = \lceil 10 \times 0.90 \rceil = 9 \le 9$ (**Valid**)

### 6.2 Family-Level and LOFO Conformal Breakdown
1. **In-Family Calibration**:
   - Halide Perovskites ($n_{\text{cal}} = 7$): $k = \lceil 8 \times 0.90 \rceil = 8 > 7$. Conformal 90% prediction intervals cannot be formed mathematically without ad-hoc index clamping.
   - Transition Metal Perovskites ($n_{\text{cal}} = 2$) and Alkaline Earth Oxides ($n_{\text{cal}} = 1$): Violate validity ($k > n_{\text{cal}}$).
2. **Leave-One-Family-Out (LOFO) Calibration**:
   - Leaving out Alkaline Earth Oxide: $n_{\text{cal}} = 9 \implies k = 9 \le 9$. **Valid** ($q_{\text{cal}} = 0.3583\text{ eV}$ for Ridge, $1.1076\text{ eV}$ for Scissor).
   - Leaving out Transition Metal Perovskite: $n_{\text{cal}} = 8 \implies k = 9 > 8$. **Invalid (Infinite)**.
   - Leaving out Halide Perovskite: $n_{\text{cal}} = 3 \implies k = 4 > 3$. **Invalid (Infinite)**.
3. **Production Resolution**:
   - In production inference, the conformal module pools all $N=10$ verified calibration records ($k = \lceil 11 \times 0.90 \rceil = 10 \le 10$), producing a mathematically valid, finite interval normalized by leverage:
     $$\Delta \in \left[ \hat{\Delta} - \tilde{q} \sqrt{1 + h_q}, \, \hat{\Delta} + \tilde{q} \sqrt{1 + h_q} \right], \quad \tilde{q}_{\text{pooled}} = 0.7273\text{ eV}.$$
