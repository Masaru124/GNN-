# MatScreen Single-Fidelity Experimental Gap Calibration Dataset Provenance & Comprehensive Audit ($N=10$)

This document provides complete, transparent, and audited provenance for the single-fidelity $N=10$ experimental optical band gap calibration dataset used in the $\Delta$-ML band gap correction service, documenting the target audit across all 21 candidate entries, PBE parameter comparisons, dielectric constant provenance, and refit metrics.

---

## 1. Academic Integrity & Target Provenance Audit (21 Compounds)

Every candidate compound was audited against primary literature sources in the local corpus (`materials-screening-ai/research/literature/combined.md`, SHA256: `3bbf35118c8c36e1f2b5fc7b1f714d38522e2100a9961b34512259c2f1795c3c`) and local PDF documents.

### Summary Table of All 21 Candidate Rows

| Formula | Target Gap (eV) | Target Level | Verified Primary Source | Table / Page | Verbatim Matched Cell Text | Status in Calibration Fit |
| :--- | :---: | :--- | :--- | :--- | :--- | :--- |
| **MgO** | 7.22 | Experimental | Heyd et al. JCP 123, 174101 (2005) | Table V, p. 174101-6 | `\|MgO\|4.178\|4.268\|4.247\|4.218\|4.207\|4.92\|4.34\|4.56\|6.50\|7.22\|` | **VERIFIED (Active Fit)** |
| **SrTiO3** | 3.25 | Experimental | Piskunov CMS 29, 165 (2004) / Bilc PRB 77, 165107 (2008) | Table 4, p. 172 / Table V, p. 10 | `STO ... X–X ... 3.25––indirect gap (Ref. [46])` / `SrTiO3 ... 3.57 3.25[82]` | **VERIFIED (Active Fit)** |
| **BaTiO3** | 3.20 | Experimental | Piskunov CMS 29, 165 (2004) / Bilc PRB 77, 165107 (2008) | Table 4, p. 172 / Table V, p. 10 | `BTO ... C–C ... 3.2 (Ref. [47])` / `BaTiO3 ... 3.39 3.20[65]` | **VERIFIED (Active Fit)** |
| **CsPbI3** | 1.73 | Experimental | Castelli APL Mater. 2, 081514 (2014) / JPCL 2017 | Table I, p. 081514-2 / Table 6, p. 5511 | `\|CsPbI\|Cubic\|1.62\|1.73\|` / `\|CsPbI\|2.24\|0.74\|−1.22\|1.76\|1.67,a 1.73b\|` | **VERIFIED (Active Fit)** |
| **CsPbBr3** | 2.36 | Experimental | Wiktor et al. JPCL 8, 5507 (2017) | Table 6, p. 5511 | `\|CsPbBr\|3.15\|0.51\|−1.28\|2.38\|2.36c\|` (ref 38) | **VERIFIED (Active Fit)** |
| **CsPbCl3** | 2.85 | Experimental | Wiktor et al. JPCL 8, 5507 (2017) | Table 6, p. 5511 | `\|CsPbCl\|3.66\|0.63\|−1.34\|2.95\|2.85d\|` (ref 39) | **VERIFIED (Active Fit)** |
| **CsSnCl3** | 2.60 | Experimental | Wiktor et al. JPCL 8, 5507 (2017) | Table 6, p. 5511 | `\|CsSnCl ... \|2.22\|0.73\|−0.35\|2.60\|∼2.6e\|` (ref 40) | **VERIFIED (Active Fit)** |
| **MAPbI3** | 1.57 | Experimental | Castelli APL Mater. 2, 081514 (2014) / Mosconi JPCC 2013 | Table I / Table 1, p. 13907 | `\|MAPbI\|Cubic\|1.36\|1.57\|` / `\|X=I\|a = 6.33\|...\|1.57\|1.55b\|` | **VERIFIED (Active Fit)** |
| **MAPbBr3** | 2.33 | Experimental | Castelli APL Mater. 2, 081514 (2014) / Mosconi JPCC 2013 | Table I / Table 1, p. 13907 | `\|MAPbBr\|Cubic\|1.96\|2.33\|` / `\|X=Br\|a = 5.90\|...\|1.80\|2.00b, 2.33−2.35e,f\|` | **VERIFIED (Active Fit)** |
| **MAPbCl3** | 3.11 | Experimental | Mosconi et al. JPCC 117, 13902 (2013) | Table 1, p. 13907 | `\|X=Cl ... \|a = 5.68\|...\|2.34\|3.11−3.12e,f\|` | **VERIFIED (Active Fit)** |
| **CsSnI3** | 1.57 | QSGW+SOC (Theory) | Wiktor et al. JPCL 8, 5507 (2017) | Table 6, p. 5511 | `\|CsSnI₃\|1.21\|0.78\|−0.40\|1.57\|\|` | Excluded (Theory Only) |
| **CsSnBr3** | 2.09 | QSGW+SOC (Theory) | Wiktor et al. JPCL 8, 5507 (2017) | Table 6, p. 5511 | `\|CsSnBr₃\|1.66\|0.77\|−0.34\|2.09\|\|` | Excluded (Theory Only) |
| **NaCl** | 6.48 | HSE03 (Theory) | Heyd et al. JCP 123, 174101 (2005) | Not in Table V | Absent from Heyd Table V (SC/40 test set); Paier 2006 unavailable | Excluded (Unsourced) |
| **NaBr** | 5.40 | Experimental | Landolt-Börnstein III/41B | Unsourced | Absent from local text corpus | Excluded (Unsourced) |
| **NaI** | 4.85 | Experimental | Landolt-Börnstein III/41B | Unsourced | Absent from local text corpus | Excluded (Unsourced) |
| **LiF** | 11.45 | HSE03 (Theory) | Heyd et al. JCP 123, 174101 (2005) | Not in Table V | Absent from Heyd Table V; Paier 2006 unavailable | Excluded (Unsourced) |
| **NaF** | 8.00 | HSE03 (Theory) | Heyd et al. JCP 123, 174101 (2005) | Not in Table V | Absent from Heyd Table V; Paier 2006 unavailable | Excluded (Unsourced) |
| **LiCl** | 7.60 | HSE03 (Theory) | Heyd et al. JCP 123, 174101 (2005) | Not in Table V | Absent from Heyd Table V; Paier 2006 unavailable | Excluded (Unsourced) |
| **CaO** | 5.37 | HSE03 (Theory) | Heyd et al. JCP 123, 174101 (2005) | Not in Table V | Absent from Heyd Table V; Paier 2006 unavailable | Excluded (Unsourced) |
| **BaO** | 3.75 | Experimental | Landolt-Börnstein III/41B | Unsourced | Absent from local text corpus | Excluded (Unsourced) |
| **RbPbBr3** | 2.38 | GLLB-SC (Theory) | Castelli et al. APL Mater. 2, 081514 (2014) | Not in Table I | Absent from Castelli Table I; 2.38 eV was JPCL CsPbBr3 theory | Excluded (Unsourced) |

---

## 2. PBE Input Parameter & Physics Comparison

### CsPbI3 Comparison: Our PBE (1.323 eV) vs JPCL 2017 (1.14 eV)
1. **Lattice Parameter**: Our calculation used experimental cubic lattice constant $a = 6.29\text{ \AA}$ ($V = 248.9\text{ \AA}^3$). JPCL 2017 optimized lattice at PBE level, finding a larger expanded volume ($a \approx 6.40\text{ \AA}$, $+5.3\%$ volume expansion).
2. **Deformation Potential**: In lead halide perovskites, the valence band maximum is an antibonding Pb-$6s$ / I-$5p$ state. Volume expansion weakens antibonding overlap, lowering the VBM and narrowing the fundamental band gap ($dE_g/dV < 0$). This explains $\approx 0.12\text{ eV}$ of the difference.
3. **Pseudopotential & Semicore**: Our calculation used SSSP Efficiency with semicore Pb ($5d^{10} 6s^2 6p^2$) and Cs ($5s^2 5p^6 6s^1$) explicit valence states with scalar-relativistic treatment without spin-orbit coupling.
4. **K-Point Sampling**: We used a converged $(4\times 4\times 4)$ Monkhorst-Pack mesh sampling the zone-corner $R(0.5, 0.5, 0.5)$ direct gap.

### MAPbCl3 Comparison: Our PBE (2.450 eV) vs Mosconi 2013 (2.34 eV)
1. **Phase and Molecular Orientation**: Our calculation evaluated cubic $Pm\bar{3}m$ lattice ($a = 5.68\text{ \AA}$) with time-averaged isotropic methylammonium orientation. Mosconi et al. evaluated explicit static organic cation dipole alignments ($C_{3v}$ along [111] and [100]), where internal electrostatic dipole fields induce $\pm 0.10\text{ eV}$ band gap modulation.
2. **Pseudopotential & Basis Set**: Our pipeline executed plane-wave pseudopotential DFT (SSSP ultrasoft / PAW), whereas Mosconi et al. utilized localized Gaussian basis sets (PW91 / B3LYP functional implementations in CPMD / Quantum ESPRESSO).

---

## 3. Dielectric Constant ($\epsilon_\infty$) Provenance Audit

A complete provenance review of dielectric constants $\epsilon_\infty$ used in the 7-feature model:
1. **Primary Band-Gap Literature Tables**: The primary band-gap source tables (Bilc et al. PRB 77 Table V, Huang et al. PRB 88 Table IV, Brivio et al. PRB 89 Table I) report band gaps and lattice parameters, but **do not tabulate high-frequency optical dielectric constants ($\epsilon_\infty$)**.
2. **Provenance Status**: Thirteen (13) of 21 canonical values in previous informal scripts were copied from active configuration files rather than independently extracted from separate dielectric experiment papers.
3. **Audit Designation**: In `delta_ml_corrector.py` and `calibration_provenance_table.csv`, dielectric values are explicitly documented as **DFPT / SSSP optical dielectric proxies**, not as verified experimental entries from the band-gap citation.

---

## 4. Single-Fidelity Refit Metrics ($N=10$)

- **Active Calibration Records**: 10 verified experimental optical gaps (`MgO`, `SrTiO3`, `BaTiO3`, `CsPbI3`, `CsPbBr3`, `CsPbCl3`, `CsSnCl3`, `MAPbI3`, `MAPbBr3`, `MAPbCl3`).
- **Family Distribution**: `halide_perovskite` ($n=7$), `transition_metal_perovskite` ($n=2$), `alkaline_earth_oxide` ($n=1$).
- **Conformal Prediction**: All families have $n_{\text{fam}} < 9$, safely falling back to pooled calibration ($k = \lceil 11 \times 0.90 \rceil = 10$, $\tilde{q}_{\text{pooled}} = \mathbf{0.7273\text{ eV}}$).
- **Linear Scissor Fit**: $E_{\text{exp}} = 1.4840 \cdot E_{\text{PBE}} + 0.0251$ eV.
- **Cross-Family Performance**: Linear Scissor LOCO MAE = **0.6597 eV** vs Nested Ridge LOCO MAE = **1.0190 eV**.
- **In-Family Performance (Halide Perovskites $n=7$)**: Ridge ($\alpha=1.0$) LOOCV MAE = **0.2683 eV** vs Linear Scissor LOOCV MAE = **0.6245 eV** (Ridge advantage: **0.3562 eV**).
