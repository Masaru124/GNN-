# DFT/ML Pipeline Verification Guide — Standing Verification Standard

> **Operational Mandate**: Before reporting any result as verified, run through the applicable checks in this guide. Show the **raw evidence** for each check (actual API responses, actual eigenvalues, actual k-points, script stdout), not a summary claiming it passed. Plausible-sounding summaries without raw tool output are prohibited.

Every item here corresponds to a real bug or fabrication identified and resolved in the MatScreen AI pipeline.

---

## 1. Citations and Literature Claims

**Failure mode found:** Two fabricated citations (Chantis et al., PRB 73, 045114 (2006) and Manghnani et al., PRB 60, 6171 (1999) — real author/journal, wrong paper entirely), plus a DOI typo (`10.1063/1.2087449` vs. `10.1063/1.2085170`), plus unverified table claims (KCl, KBr, KI attributed to Paier Table IV where they did not exist).

**Checklist:**
- [ ] Every DOI is queried against the live CrossRef API (`https://api.crossref.org/works/{doi}`) or OpenAlex API, never recalled from model memory.
- [ ] The raw title/author/journal string from the API response is printed and displayed.
- [ ] For a specific numeric claim ("Table IV reports $X = 6.05\text{ eV}$"), confirm the *specific number* appears in that *specific table* of the primary source.
- [ ] If a value cannot be found in the claimed primary source, it must be placed in an explicitly segregated `EXCLUDED_UNVERIFIED_RECORDS` bucket — never silently kept in a fit or dressed up with a surrogate citation.

**Verification Command:**
```bash
python scratch/raw_crossref_audit.py
```

---

## 2. Crystal Structures and Prototypes

**Failure mode found:** Wrong Wyckoff position placed atoms on a general (36-fold) site instead of the correct special position, silently generating 30+ atoms instead of 4 — turning a real insulator into a false metal.

**Checklist:**
- [ ] For any generated prototype structure, print the space group symbol, space group number, atom count, and fractional coordinates of each site.
- [ ] Compare against the known reference structure from the Materials Project (MP-ID) or ICSD.
- [ ] Check for suspiciously short interatomic distances ($d < 1.0\text{ \AA}$) indicative of overlapping sites.
- [ ] For cubic/high-symmetry space groups with multiple origin choices (e.g. $Fd\bar{3}m$ for diamond), verify the origin choice matches pymatgen's convention (Origin Choice 1 vs. 2).

**Verification Command:**
```bash
python -c "from app.services.simulation_service import generate_crystal_prototype; s = generate_crystal_prototype('KZrCl3'); print(s.spacegroup, len(s), s.lattice.abc)"
```

---

## 3. Silent Fallbacks That Mask Failure

**Failure mode found:** When `nbnd` was unset, Quantum ESPRESSO computed zero conduction bands; the parser's fallback for "no conduction band found" reported `gap_eV: 0.0, "metallic"`, turning a missing-bands setup bug into a plausible-looking wrong answer.

**Checklist:**
- [ ] Any fallback path returning a specific plausible value (`0.0 eV`, `"metallic"`, or default gap) on missing data is prohibited. It must return `None`, `"unknown"`, or raise an explicit exception.
- [ ] Explicitly audit `except:` blocks and `.get(..., default)` calls to ensure defaults cannot be mistaken for valid physical output.
- [ ] In electronic structure calculations, explicitly set `nbnd` above the occupied count, and verify conduction bands exist.

---

## 4. Cross-Code / Cross-Source Consistency

**Failure mode found:** The $\Delta$-ML correction model was initially trained on PBE inputs borrowed from disparate literature sources (all-electron Gaussian, PAW, different pseudopotentials) while being applied at inference time to Quantum ESPRESSO + SSSP PBE outputs, introducing unmodeled systematic offsets.

**Checklist:**
- [ ] When training a model that maps "Method A $\to$ Method B", ensure all training inputs for Method A were computed with the *exact same* code, pseudopotentials, cutoff energy, and settings used at inference time.
- [ ] When target values stem from mixed methodologies (HSE06, QSGW, B3PW), label each source explicitly in data tables, UI responses, and provenance records.

---

## 5. Numerical Sanity / Physical Plausibility Checks

**Failure mode found:** $\text{CsPbI}_3$ and $\text{CsSnI}_3$ showed PBE gaps exceeding hybrid-functional targets by $\approx 1.8\text{ eV}$ ($\Delta < 0$), violating fundamental DFT gap-underestimation physics. This was initially rationalized as "mesh discretization" before being diagnosed as $R$-point omission.

**Checklist:**
- [ ] For band gap corrections, verify the sign and magnitude of every residual $\Delta = E_g^{\text{target}} - E_g^{\text{PBE}}$. Underestimation requires $\Delta > 0$ for standard semi-local DFT. Any negative residual is an automatic bug flag.
- [ ] Verify monotonic trends down chemical series (e.g. halogen series: $\text{Cl} > \text{Br} > \text{I}$). A trend violation indicates a computational error.
- [ ] Never accept an unverified hypothesis ("probably mesh error") as an explanation; inspect eigenvalues and $k$-points directly.

---

## 6. $k$-Point Mesh / Brillouin Zone Sampling

**Failure mode found:** Odd-$N$ Monkhorst-Pack meshes mathematically never sample $k_i = 1/2$, silently omitting zone-boundary points ($R(0.5, 0.5, 0.5)$, $X(0.5, 0, 0)$, $M(0.5, 0.5, 0)$) where perovskite extrema reside.

**Checklist:**
- [ ] Automatic $k$-mesh generators must guarantee even $N_i \ge 2$ along each axis: $N_i = \max(2, 2 \lceil \text{raw\_N}_i / 2 \rceil)$.
- [ ] Recognize the scope of this guarantee: `force_even` guarantees sampling of zone-boundary points with fractional coordinate $1/2$ in simple-cubic, tetragonal, and orthorhombic lattices.
- [ ] It does **not** guarantee capturing band extrema that reside off high-symmetry points (even in cubic systems, e.g. Si CBM at $\approx 0.85$ along $\Gamma - X$), nor at generic incommensurate $k$-points in lower-symmetry structures (monoclinic, triclinic, distorted perovskite phases).
- [ ] For candidates, do not rely on parity alone: run an explicit multi-grid convergence scan ($k$-refinement until $\Delta E_g < 25\text{ meV}$) or a dense NSCF / band structure calculation.
- [ ] For open-shell transition metal candidates (e.g. formal $d^2 \text{Zr}^{2+}$ in hypothetical $\text{KZrCl}_3$), audit for metallicity before treating an unrelaxed prototype gap as valid.

---

## 7. Statistics on Top of Unverified Data & Silent Statistical Drift

**Failure mode found:** Cross-validation metrics (LOOCV, LOCO, $\hat{q}$) were computed on unverified datasets containing flawed points. Separately, the reported LiF leverage value changed three times ($0.926 \to 0.742 \to 0.892$) across drafts without explanation, and a reconstructed narrative was offered instead of an honest declaration of non-reproducibility.

**Checklist:**
- [ ] Never report statistical metrics until every individual constituent data point has passed all physical and integrity audits.
- [ ] All statistics must be generated by **one canonical script** reading live data from `delta_ml_corrector.py` (`backend/scripts/compute_canonical_leverage.py`), exporting to `research/canonical_leverage_table.csv`.
- [ ] If a historical number cannot be reproduced under tested configurations, state plainly that it is **unverified and not reproducible** — never construct an unverified narrative to explain it away.
- [ ] For Ridge hat matrices, ensure the hat matrix matches the fitted model (e.g. unpenalized intercept in `sklearn.linear_model.Ridge(fit_intercept=True)`: $H = \frac{1}{n}\mathbf{1}\mathbf{1}^T + Z(Z^TZ + \alpha I)^{-1}Z^T$).
- [ ] Check leverage threshold margins: points within rounding distance of $2p/n$ (e.g. BaO at $0.7620$ vs $0.7619$) must be reported as **Borderline**, not hard outliers. Note parameter saturation ($\bar{h} = p/n$) and collinearity ($r(\text{PBE}, \text{PBE}^2) = 0.9542$).
- [ ] **LOOCV shortcut warning**: The analytic shortcut $e_i / (1 - h_{ii})$ is **invalid** for pipelines that refit a `StandardScaler` per fold. On this dataset (n=21, p=8), the max per-compound discrepancy between brute-force LOOCV and the analytic shortcut is **0.42 eV** (LiF). Always use brute-force refitting.
- [ ] **Nested $\alpha$ selection**: When $\alpha$ is tuned per fold via inner 5-fold CV (over $[0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0]$), the nested LOOCV MAE is **0.277 eV** vs **0.361 eV** for fixed $\alpha = 1.0$. The inner CV selects $\alpha = 0.01$ for 16/21 folds, suggesting the current $\alpha = 1.0$ over-regularizes relative to this small dataset.

---

## 8. Cross-Artifact Consistency

**Failure mode found:** The $n=21$ dataset was duplicated across Python source, a Markdown table, and a CSV matrix, with rows in the CSV permuted relative to the source records.

**Checklist:**
- [ ] All derived artifacts (CSV matrices, markdown tables) must be regenerated programmatically from the single canonical data source.
- [ ] Verify 1-to-1 isomorphism between artifacts by formula key, not by row position.
- [ ] Check DOIs, values, and order programmatically before finalizing documentation.

---

## The Meta-Instruction

> **Always request and provide raw tool output, never a narrative summary claiming verification. direct evidence (actual numbers, actual structures, raw API responses) must precede any analytical conclusion.**

---

## 9. Metallicity Detection via Signed Overlap

**Failure mode found:** The HOMO/LUMO parser path used `gap = max(0.0, cbm - vbm)`, silently clipping a negative band overlap to zero and reporting `gap_type: "insulator"` with `gap_eV: 0.0`. This would misclassify a metallic system with overlapping bands as an insulator.

**Fix applied:** All three parser paths (HOMO/LUMO, Fermi-level, valence-electron fallback) now explicitly check `if cbm < vbm` and return `gap_type: "metallic"` with `gap_eV: 0.0`. The `max(0.0, ...)` clip has been removed.

**Checklist:**
- [ ] Never use `max(0.0, cbm - vbm)` to compute a band gap. Compute `cbm - vbm` directly and check the sign.
- [ ] If `cbm < vbm` (signed overlap), report metallic — do not clip to zero and call it insulating.
- [ ] Verify all gap parsing paths handle the `cbm < vbm` case consistently.

---

## 10. Multi-Method Beyond-PBE Reference Target & SOC Consistency

**Audited Reference Methodologies**:
The $n=21$ calibration targets represent a verified multi-method beyond-PBE benchmark set:
- **Heyd et al. 2005 / Paier et al. 2006** (Table V / IV): All-electron Gaussian / plane-wave **HSE06 hybrid functional** for alkali halides (LiF, LiCl, NaF, NaCl, NaBr, NaI) and alkaline earth oxides (MgO, CaO, BaO).
- **Castelli et al. 2014** (APL Mater. 2, 081514): GPAW **GLLB-SC potential with spin-orbit coupling (SOC) correction ($\sim -1.02\text{ eV}$)** for Pb perovskites ($\text{CsPbI}_3=1.73$, $\text{CsPbBr}_3=2.25$, $\text{CsPbCl}_3=2.85$, $\text{RbPbBr}_3=2.38\text{ eV}$), matching experimental cubic phase optical gaps.
- **Huang & Lambrecht 2013** (PRB 88, 165203, Table IV): Quasiparticle Self-Consistent GW with SOC (**QSGW+SOC**) for Sn perovskites ($\text{CsSnI}_3=1.30$, $\text{CsSnBr}_3=1.75$, $\text{CsSnCl}_3=2.45\text{ eV}$).
- **Brivio et al. 2014** (PRB 89, 155204): Relativistic Quasiparticle Self-Consistent GW (**QSGW+SOC**) for $\text{MAPbI}_3=1.67\text{ eV}$.
- **Mosconi et al. 2013** (JPCC 117, 13902): **PBE0 / Hybrid DFT** for $\text{MAPbBr}_3=2.20$, $\text{MAPbCl}_3=2.95\text{ eV}$.
- **Bilc et al. 2008 / Piskunov et al. 2004**: B1-WC / B3PW Hybrid DFT for $\text{BaTiO}_3=3.20$, $\text{SrTiO}_3=3.25\text{ eV}$.

**Baseline PBE & SOC Absorption Policy**:
- Baseline PBE inputs are computed self-consistently with standard scalar-relativistic SSSP Efficiency pseudopotentials on $(4\times 4\times 4)$ or denser grids (e.g. $\text{CsPbI}_3 = 1.323\text{ eV}$).
- For heavy elements (Pb, Sn), the empirical $\Delta$-ML correction ($\Delta = E_{\text{target}} - E_{\text{PBE}}^{\text{scalar-rel}}$) inherently absorbs both the exchange-correlation functional correction ($\Delta_{\text{XC}}$) and any residual spin-orbit coupling correction ($\Delta_{\text{SOC}}$) between the scalar-relativistic PBE baseline and the relativistic benchmark target.

**Checklist:**
- [x] Explicitly label each compound's high-level electronic structure method (HSE06, GLLB-SC+SOC, QSGW+SOC, PBE0) in `evidence_matrix.csv` and documentation.
- [x] Verify crystal structure consistency: cubic ($Pm\bar{3}m$) targets are paired strictly with cubic PBE inputs.
- [x] Disclose that the $\Delta$-ML mapping absorbs functional and spin-orbit corrections simultaneously for heavy-element perovskite families.

