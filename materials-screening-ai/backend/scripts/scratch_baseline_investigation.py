import sys, os, subprocess
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from pymatgen.core import Composition
from app.services.delta_ml_corrector import DeltaMLGapCorrector

corrector = DeltaMLGapCorrector()

# Pre-rebuild historical CALIBRATION_RECORDS from before Option (i):
OLD_RECORDS = [
    (5.106, 6.48, 2.23, 0.5636, 14.0, 2.54, "NaCl", "alkali_halide"),
    (4.165, 5.40, 2.03, 0.5283, 24.0, 2.65, "NaBr", "alkali_halide"),
    (3.640, 4.85, 1.73, 0.4682, 32.0, 3.00, "NaI", "alkali_halide"),
    (9.241, 11.45, 3.00, 0.5755, 6.0, 1.96, "LiF", "alkali_halide"),
    (6.354, 8.00, 3.05, 0.7698, 10.0, 1.75, "NaF", "alkali_halide"),
    (6.360, 7.60, 2.18, 0.4208, 10.0, 2.75, "LiCl", "alkali_halide"),
    (4.475, 6.50, 2.13, 0.5143, 10.0, 3.00, "MgO", "alkaline_earth_oxide"),
    (3.751, 5.37, 2.44, 0.7143, 14.0, 3.35, "CaO", "alkaline_earth_oxide"),
    (1.976, 3.75, 2.55, 0.9643, 32.0, 3.60, "BaO", "alkaline_earth_oxide"),
    (2.179, 3.25, 2.49, 0.8429, 22.4, 6.10, "SrTiO3", "transition_metal_perovskite"),
    (2.068, 3.20, 2.55, 0.9429, 27.2, 6.30, "BaTiO3", "transition_metal_perovskite"),
    (1.323, 1.73, 1.26, 0.7100, 57.0, 5.80, "CsPbI3", "halide_perovskite"),
    (1.532, 2.25, 1.46, 0.6500, 48.0, 5.30, "CsPbBr3", "halide_perovskite"),
    (1.919, 2.85, 1.60, 0.6000, 39.0, 4.80, "CsPbCl3", "halide_perovskite"),
    (0.327, 1.30, 1.26, 0.7100, 50.0, 6.20, "CsSnI3", "halide_perovskite"),
    (0.392, 1.75, 1.46, 0.6500, 41.0, 5.80, "CsSnBr3", "halide_perovskite"),
    (0.799, 2.45, 1.60, 0.6000, 32.0, 5.20, "CsSnCl3", "halide_perovskite"),
    (1.550, 1.67, 1.40, 0.6500, 20.0, 5.70, "MAPbI3", "halide_perovskite"),
    (1.900, 2.20, 1.60, 0.6000, 18.0, 5.40, "MAPbBr3", "halide_perovskite"),
    (2.450, 2.95, 1.74, 0.5500, 15.0, 4.80, "MAPbCl3", "halide_perovskite"),
    (1.395, 2.38, 1.46, 0.6500, 44.0, 4.70, "RbPbBr3", "halide_perovskite"),
]

def run_evaluation(records, target_type="delta", alpha=1.0):
    formulas = [r[6] for r in records]
    families = [r[7] for r in records]
    pbes = np.array([r[0] for r in records])
    targets = np.array([r[1] for r in records])
    
    X = np.array([corrector._make_features(r[0], r[2], r[3], r[4], r[5]) for r in records])
    y = (targets - pbes) if target_type == "delta" else targets
    n = len(records)
    
    # LOOCV
    loocv_errors = []
    res_dict = {}
    for i in range(n):
        train_idx = [j for j in range(n) if j != i]
        pipe = Pipeline([('scaler', StandardScaler()), ('ridge', Ridge(alpha=alpha, fit_intercept=True))])
        pipe.fit(X[train_idx], y[train_idx])
        pred_raw = pipe.predict(X[i:i+1])[0]
        pred_gap = (pbes[i] + pred_raw) if target_type == "delta" else pred_raw
        err = abs(targets[i] - pred_gap)
        loocv_errors.append(err)
        res_dict[formulas[i]] = (targets[i], pred_gap, err)
        
    loocv_mae = np.mean(loocv_errors)
    loocv_rmse = np.sqrt(np.mean(np.array(loocv_errors)**2))
    
    # Quantiles
    # Method 1: finite-sample corrected index ceil((n+1)*0.9) - 1
    k = int(np.ceil((n + 1) * 0.90))
    sorted_errs = np.sort(loocv_errors)
    q_hat_fs = sorted_errs[min(k - 1, n - 1)]
    # Method 2: standard np.quantile with 0.90
    q_hat_std = np.quantile(loocv_errors, 0.90)
    
    # LOCO
    fam_errs = {}
    all_loco_errs = []
    for fam in sorted(list(set(families))):
        train_idx = [j for j in range(n) if families[j] != fam]
        test_idx = [j for j in range(n) if families[j] == fam]
        pipe = Pipeline([('scaler', StandardScaler()), ('ridge', Ridge(alpha=alpha, fit_intercept=True))])
        pipe.fit(X[train_idx], y[train_idx])
        preds_raw = pipe.predict(X[test_idx])
        preds_gap = (pbes[test_idx] + preds_raw) if target_type == "delta" else preds_raw
        errs = list(np.abs(targets[test_idx] - preds_gap))
        fam_errs[fam] = (len(test_idx), np.mean(errs))
        all_loco_errs.extend(errs)
        
    pooled_loco = np.mean(all_loco_errs)
    macro_loco = np.mean([v[1] for v in fam_errs.values()])
    
    return {
        "target_type": target_type,
        "alpha": alpha,
        "loocv_mae": loocv_mae,
        "loocv_rmse": loocv_rmse,
        "q_hat_fs": q_hat_fs,
        "q_hat_std": q_hat_std,
        "pooled_loco": pooled_loco,
        "macro_loco": macro_loco,
        "fam_errs": fam_errs,
        "sorted_res": sorted(res_dict.items(), key=lambda x: x[1][2]),
        "res_dict": res_dict
    }

print("=== 1. PRE-REBUILD OLD FEATURES EVALUATION ===")
res_old_abs = run_evaluation(OLD_RECORDS, target_type="absolute", alpha=1.0)
print(f"ABSOLUTE TARGET (y = HSE, alpha=1.0):")
print(f"  LOOCV MAE: {res_old_abs['loocv_mae']:.4f} eV, RMSE: {res_old_abs['loocv_rmse']:.4f} eV")
print(f"  q_hat (finite sample): {res_old_abs['q_hat_fs']:.4f} eV, q_hat (std 90%): {res_old_abs['q_hat_std']:.4f} eV")
print(f"  Pooled LOCO: {res_old_abs['pooled_loco']:.4f} eV, Macro LOCO: {res_old_abs['macro_loco']:.4f} eV")
for fam, (cnt, mae) in res_old_abs['fam_errs'].items():
    print(f"    - {fam:25s} (n={cnt}): {mae:.4f} eV")

print()
res_old_delta = run_evaluation(OLD_RECORDS, target_type="delta", alpha=1.0)
print(f"DELTA TARGET (y = HSE - PBE, alpha=1.0):")
print(f"  LOOCV MAE: {res_old_delta['loocv_mae']:.4f} eV, RMSE: {res_old_delta['loocv_rmse']:.4f} eV")
print(f"  q_hat (finite sample): {res_old_delta['q_hat_fs']:.4f} eV, q_hat (std 90%): {res_old_delta['q_hat_std']:.4f} eV")
print(f"  Pooled LOCO: {res_old_delta['pooled_loco']:.4f} eV, Macro LOCO: {res_old_delta['macro_loco']:.4f} eV")
for fam, (cnt, mae) in res_old_delta['fam_errs'].items():
    print(f"    - {fam:25s} (n={cnt}): {mae:.4f} eV")
