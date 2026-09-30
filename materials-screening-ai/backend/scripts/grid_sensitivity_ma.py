import sys, os, itertools
sys.path.insert(0, os.path.abspath("materials-screening-ai/backend"))
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from app.services.delta_ml_corrector import DeltaMLGapCorrector, CALIBRATION_RECORDS, _composition_features

corrector = DeltaMLGapCorrector()

# Sensitivity Grid Ranges:
# r_eff: Kieslich et al. (Chem. Sci. 2014) reports r_eff = 2.17 A; plausible tolerance range 2.00 to 2.30 A
r_eff_grid = [2.00, 2.17, 2.30]
# chi_eff: Pauling scale group electronegativity for protonated amine cation [CH3NH3]+: plausible range 1.20 to 1.50
chi_eff_grid = [1.20, 1.30, 1.40, 1.50]
# Z_eff: Effective charge / proton number: CH3NH3+ has 19 total protons (6+6+7), 18 electrons; valence core ~14-19
z_eff_grid = [14.0, 16.0, 18.0, 19.0]

print("================================================================================")
print("ITEM 3: MA PSEUDO-A-SITE SENSITIVITY GRID (r_eff x chi_eff x Z_eff)")
print("================================================================================")
print(f"Testing {len(r_eff_grid) * len(chi_eff_grid) * len(z_eff_grid)} parameter combinations...")

results = []

for r_eff, chi_eff, z_eff in itertools.product(r_eff_grid, chi_eff_grid, z_eff_grid):
    # Build feature modified dataset
    records_grid = []
    for r in CALIBRATION_RECORDS:
        form = r[6]
        if "MAPb" in form:
            if form == "MAPbI3":
                chi_d = round(2.66 - chi_eff, 4)
                r_rat = round(1.40 / r_eff, 4)
                z_avg = round((z_eff + 82 + 3*53)/5, 4)
            elif form == "MAPbBr3":
                chi_d = round(2.96 - chi_eff, 4)
                r_rat = round(1.15 / r_eff, 4)
                z_avg = round((z_eff + 82 + 3*35)/5, 4)
            elif form == "MAPbCl3":
                chi_d = round(3.16 - chi_eff, 4)
                r_rat = round(0.99 / r_eff, 4)
                z_avg = round((z_eff + 82 + 3*17)/5, 4)
            records_grid.append((r[0], r[1], chi_d, r_rat, z_avg, r[5], r[6], r[7], r[8], r[9]))
        else:
            records_grid.append(r)
            
    pbes = np.array([rec[0] for rec in records_grid])
    targets = np.array([rec[1] for rec in records_grid])
    y_delta = targets - pbes
    families = [rec[9] for rec in records_grid]
    n = len(records_grid)
    X = np.array([corrector._make_features(rec[0], rec[2], rec[3], rec[4], rec[5]) for rec in records_grid])
    
    # LOOCV
    loocv_errs = []
    for i in range(n):
        tr = [j for j in range(n) if j != i]
        pipe = Pipeline([('scaler', StandardScaler()), ('ridge', Ridge(alpha=1.0, fit_intercept=True))])
        pipe.fit(X[tr], y_delta[tr])
        p_d = pipe.predict(X[i:i+1])[0]
        loocv_errs.append(abs(targets[i] - (pbes[i] + p_d)))
    m_loocv = np.mean(loocv_errs)
    
    # LOCO
    fam_errs = {}
    all_loco = []
    for fam in sorted(list(set(families))):
        tr = [j for j in range(n) if families[j] != fam]
        te = [j for j in range(n) if families[j] == fam]
        pipe = Pipeline([('scaler', StandardScaler()), ('ridge', Ridge(alpha=1.0, fit_intercept=True))])
        pipe.fit(X[tr], y_delta[tr])
        p_d = pipe.predict(X[te])
        e = list(np.abs(targets[te] - (pbes[te] + p_d)))
        fam_errs[fam] = float(np.mean(e))
        all_loco.extend(e)
    pooled_loco = float(np.mean(all_loco))
    macro_loco = float(np.mean(list(fam_errs.values())))
    
    results.append({
        "r_eff": r_eff,
        "chi_eff": chi_eff,
        "z_eff": z_eff,
        "loocv_mae": m_loocv,
        "pooled_loco": pooled_loco,
        "macro_loco": macro_loco,
        "alkali_halide": fam_errs["alkali_halide"],
        "ae_oxide": fam_errs["alkaline_earth_oxide"],
        "halide_perov": fam_errs["halide_perovskite"],
        "tm_perov": fam_errs["transition_metal_perovskite"]
    })

# Summarize Min, Max, Mean across grid
print(f"\n--- SENSITIVITY GRID SUMMARY across {len(results)} configurations ---")
for key in ["pooled_loco", "macro_loco", "halide_perov", "alkali_halide", "ae_oxide", "tm_perov", "loocv_mae"]:
    vals = [res[key] for res in results]
    print(f"{key:25s}: Min = {min(vals):.4f} eV | Max = {max(vals):.4f} eV | Spread = {max(vals)-min(vals):.4f} eV | Mean = {np.mean(vals):.4f} eV")

print("\nTop 5 Grid Configurations (ranked by Pooled LOCO):")
sorted_res = sorted(results, key=lambda x: x["pooled_loco"])
print(f"{'Rank':4s} | {'r_eff':5s} | {'chi_eff':7s} | {'Z_eff':5s} | {'Pooled LOCO':11s} | {'Macro LOCO':10s} | {'Halide Perov':12s} | {'Alkali Halide':13s} | {'AE Oxide':8s} | {'TM Perov':8s}")
print("-" * 110)
for idx, res in enumerate(sorted_res[:5], 1):
    print(f"{idx:4d} | {res['r_eff']:5.2f} | {res['chi_eff']:7.2f} | {res['z_eff']:5.1f} | {res['pooled_loco']:11.4f} | {res['macro_loco']:10.4f} | {res['halide_perov']:12.4f} | {res['alkali_halide']:13.4f} | {res['ae_oxide']:8.4f} | {res['tm_perov']:8.4f}")
