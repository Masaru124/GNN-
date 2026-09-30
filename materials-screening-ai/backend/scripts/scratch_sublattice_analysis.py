import sys, os
sys.path.insert(0, os.path.abspath("materials-screening-ai/backend"))
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from app.services.delta_ml_corrector import DeltaMLGapCorrector, CALIBRATION_RECORDS

corrector = DeltaMLGapCorrector()

# 1. Historical pre-rebuild dataset with absolute target vs delta target
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

def analyze_model(records, name="Model"):
    formulas = [r[6] for r in records]
    families = [r[7] if len(r)==8 else r[9] for r in records]
    pbes = np.array([r[0] for r in records])
    targets = np.array([r[1] for r in records])
    y_delta = targets - pbes
    n = len(records)
    
    X = np.array([corrector._make_features(r[0], r[2], r[3], r[4], r[5]) for r in records])
    
    # LOOCV Delta formulation
    loocv_residuals = {}
    for i in range(n):
        train_idx = [j for j in range(n) if j != i]
        pipe = Pipeline([('scaler', StandardScaler()), ('ridge', Ridge(alpha=1.0, fit_intercept=True))])
        pipe.fit(X[train_idx], y_delta[train_idx])
        pred_delta = pipe.predict(X[i:i+1])[0]
        pred_gap = pbes[i] + pred_delta
        err = abs(targets[i] - pred_gap)
        loocv_residuals[formulas[i]] = err
        
    loocv_err_list = list(loocv_residuals.values())
    mae = np.mean(loocv_err_list)
    rmse = np.sqrt(np.mean(np.array(loocv_err_list)**2))
    k = int(np.ceil((n + 1) * 0.90))
    sorted_errs = sorted(loocv_residuals.items(), key=lambda x: x[1])
    q_hat_fs = sorted_errs[min(k-1, n-1)][1]
    q_hat_std = np.quantile(loocv_err_list, 0.90)
    
    # LOCO
    fam_errs = {}
    all_loco_errs = []
    loco_res_by_formula = {}
    for fam in sorted(list(set(families))):
        train_idx = [j for j in range(n) if families[j] != fam]
        test_idx = [j for j in range(n) if families[j] == fam]
        pipe = Pipeline([('scaler', StandardScaler()), ('ridge', Ridge(alpha=1.0, fit_intercept=True))])
        pipe.fit(X[train_idx], y_delta[train_idx])
        preds_delta = pipe.predict(X[test_idx])
        preds_gap = pbes[test_idx] + preds_delta
        errs = list(np.abs(targets[test_idx] - preds_gap))
        for t_i, g_i in enumerate(test_idx):
            loco_res_by_formula[formulas[g_i]] = errs[t_i]
        fam_errs[fam] = (len(test_idx), np.mean(errs))
        all_loco_errs.extend(errs)
        
    pooled_loco = np.mean(all_loco_errs)
    macro_loco = np.mean([v[1] for v in fam_maes.values()] if 'fam_maes' in locals() else [v[1] for v in fam_errs.values()])
    
    print(f"=== {name} ===")
    print(f"LOOCV MAE: {mae:.4f} eV | RMSE: {rmse:.4f} eV | q_hat (finite-sample 20/21): {q_hat_fs:.4f} eV | q_hat (np.quantile 0.9): {q_hat_std:.4f} eV")
    print(f"LOCO Pooled MAE: {pooled_loco:.4f} eV | Macro MAE: {macro_loco:.4f} eV")
    for fam, (cnt, f_mae) in fam_errs.items():
        print(f"  - {fam:25s} (n={cnt}): {f_mae:.4f} eV")
    print("\nSorted LOOCV Residuals (Ascending):")
    for rank, (f, err) in enumerate(sorted_errs, 1):
        print(f"  {rank:2d}. {f:10s}: {err:.4f} eV")
    print("-" * 75)
    return {
        "sorted_loocv": sorted_errs,
        "loco_res": loco_res_by_formula,
        "mae": mae,
        "rmse": rmse,
        "q_hat_fs": q_hat_fs,
        "pooled_loco": pooled_loco,
        "macro_loco": macro_loco
    }

res_before = analyze_model(OLD_RECORDS, name="[A] PRE-REBUILD OLD FEATURES (Hand Curated)")
res_after = analyze_model(CALIBRATION_RECORDS, name="[B] ACTIVE REBUILT COMPOSITION FEATURES")
