import sys, os
sys.path.insert(0, os.path.abspath("materials-screening-ai/backend"))
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from app.services.delta_ml_corrector import DeltaMLGapCorrector, CALIBRATION_RECORDS, _composition_features

corrector = DeltaMLGapCorrector()

# Let's define pseudo-A cation / BX3 sublattice composition features:
def _sublattice_features(formula: str):
    if "MA" in formula:
        # MAPbX3 -> BX3 inorganic frame with MA+ as A-site (r_eff = 2.17 A, chi_eff = 1.30, Z_eff = 16)
        # B = Pb (Z=82, r=1.80, chi=2.33), X = I (Z=53, r=1.40, chi=2.66), Br (Z=35, r=1.15, chi=2.96), Cl (Z=17, r=0.99, chi=3.16)
        if "MAPbI3" in formula:
            chi_diff = round(2.66 - 1.30, 4) # 1.36
            r_ratio = round(1.40 / 2.17, 4)   # 0.6452
            Z_avg = round((16 + 82 + 3*53)/5, 4) # 51.4
            return chi_diff, r_ratio, Z_avg
        elif "MAPbBr3" in formula:
            chi_diff = round(2.96 - 1.30, 4) # 1.66
            r_ratio = round(1.15 / 2.17, 4)   # 0.5299
            Z_avg = round((16 + 82 + 3*35)/5, 4) # 40.6
            return chi_diff, r_ratio, Z_avg
        elif "MAPbCl3" in formula:
            chi_diff = round(3.16 - 1.30, 4) # 1.86
            r_ratio = round(0.99 / 2.17, 4)   # 0.4562
            Z_avg = round((16 + 82 + 3*17)/5, 4) # 29.8
            return chi_diff, r_ratio, Z_avg
    return _composition_features(formula)

# Build records with sublattice features
SUBLATTICE_RECORDS = []
for r in CALIBRATION_RECORDS:
    chi, r_rat, z_avg = _sublattice_features(r[6])
    SUBLATTICE_RECORDS.append((r[0], r[1], chi, r_rat, z_avg, r[5], r[6], r[7], r[8], r[9]))

def evaluate(records, name=""):
    formulas = [r[6] for r in records]
    families = [r[9] for r in records]
    pbes = np.array([r[0] for r in records])
    targets = np.array([r[1] for r in records])
    y_delta = targets - pbes
    n = len(records)
    
    X = np.array([corrector._make_features(r[0], r[2], r[3], r[4], r[5]) for r in records])
    
    # LOOCV
    loocv_residuals = {}
    for i in range(n):
        train_idx = [j for j in range(n) if j != i]
        pipe = Pipeline([('scaler', StandardScaler()), ('ridge', Ridge(alpha=1.0, fit_intercept=True))])
        pipe.fit(X[train_idx], y_delta[train_idx])
        pred_delta = pipe.predict(X[i:i+1])[0]
        pred_gap = pbes[i] + pred_delta
        err = abs(targets[i] - pred_gap)
        loocv_residuals[formulas[i]] = err
        
    mae = np.mean(list(loocv_residuals.values()))
    rmse = np.sqrt(np.mean(np.array(list(loocv_residuals.values()))**2))
    k = int(np.ceil((n + 1) * 0.90))
    sorted_errs = sorted(loocv_residuals.items(), key=lambda x: x[1])
    q_hat_fs = sorted_errs[min(k-1, n-1)][1]
    
    # LOCO
    fam_errs = {}
    all_loco_errs = []
    for fam in sorted(list(set(families))):
        train_idx = [j for j in range(n) if families[j] != fam]
        test_idx = [j for j in range(n) if families[j] == fam]
        pipe = Pipeline([('scaler', StandardScaler()), ('ridge', Ridge(alpha=1.0, fit_intercept=True))])
        pipe.fit(X[train_idx], y_delta[train_idx])
        preds_delta = pipe.predict(X[test_idx])
        preds_gap = pbes[test_idx] + preds_delta
        errs = list(np.abs(targets[test_idx] - preds_gap))
        fam_errs[fam] = (len(test_idx), np.mean(errs))
        all_loco_errs.extend(errs)
        
    pooled_loco = np.mean(all_loco_errs)
    macro_loco = np.mean([v[1] for v in fam_errs.values()])
    
    print(f"=== {name} ===")
    print(f"LOOCV MAE: {mae:.4f} eV | RMSE: {rmse:.4f} eV | q_hat (20/21): {q_hat_fs:.4f} eV")
    print(f"LOCO Pooled MAE: {pooled_loco:.4f} eV | Macro MAE: {macro_loco:.4f} eV")
    for fam, (cnt, f_mae) in fam_errs.items():
        print(f"  - {fam:25s} (n={cnt}): {f_mae:.4f} eV")
    print()

evaluate(CALIBRATION_RECORDS, name="[1] Active Rebuilt (Elemental Min/Max on all atoms including H)")
evaluate(SUBLATTICE_RECORDS, name="[2] Sublattice / Pseudo-A Site (MA+ as effective A cation)")
