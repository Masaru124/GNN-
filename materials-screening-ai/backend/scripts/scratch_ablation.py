import sys, os
sys.path.insert(0, os.path.abspath("materials-screening-ai/backend"))
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from pymatgen.core import Composition
from app.services.delta_ml_corrector import _composition_features, DeltaMLGapCorrector, CALIBRATION_RECORDS

corrector = DeltaMLGapCorrector()

# Historical records (before option i rebuild)
# Old features: (pbe_gap, hse06_gap, chi_diff, r_ratio, Z_avg, eps_inf, formula, family)
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

ALT_EPS = {
    "NaCl": 2.34, "NaBr": 2.62, "NaI": 3.01, "LiF": 1.92, "NaF": 1.74, "LiCl": 2.75,
    "MgO": 2.95, "CaO": 3.33, "BaO": 3.61, "SrTiO3": 5.30, "BaTiO3": 5.20,
    "CsPbI3": 6.10, "CsPbBr3": 4.80, "CsPbCl3": 4.00, "CsSnI3": 6.50, "CsSnBr3": 5.20,
    "CsSnCl3": 4.40, "MAPbI3": 5.80, "MAPbBr3": 4.70, "MAPbCl3": 3.90, "RbPbBr3": 4.60
}

def evaluate_model(records, alpha=1.0):
    formulas = [r[6] for r in records]
    families = [r[7] if len(r) == 8 else r[9] for r in records]
    X_raw = []
    y_raw = []
    
    for r in records:
        pbe = r[0]
        hse = r[1]
        chi = r[2]
        r_rat = r[3]
        z_avg = r[4]
        eps = r[5]
        feats = corrector._make_features(pbe, chi, r_rat, z_avg, eps)
        X_raw.append(feats)
        y_raw.append(hse - pbe)
        
    X = np.array(X_raw)
    y = np.array(y_raw)
    N = len(y)
    
    # In-sample
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    model = Ridge(alpha=alpha)
    model.fit(X_s, y)
    y_pred_delta = model.predict(X_s)
    y_pred_abs = np.array([r[0] for r in records]) + y_pred_delta
    targets = np.array([r[1] for r in records])
    in_sample_mae = np.mean(np.abs(targets - y_pred_abs))
    in_sample_rmse = np.sqrt(np.mean((targets - y_pred_abs)**2))
    
    # LOOCV
    loocv_errors = []
    for i in range(N):
        train_idx = [j for j in range(N) if j != i]
        sc = StandardScaler()
        X_tr = sc.fit_transform(X[train_idx])
        y_tr = y[train_idx]
        m = Ridge(alpha=alpha)
        m.fit(X_tr, y_tr)
        
        X_te = sc.transform(X[i:i+1])
        d_pred = m.predict(X_te)[0]
        pred_abs = records[i][0] + d_pred
        loocv_errors.append(abs(targets[i] - pred_abs))
        
    loocv_mae = np.mean(loocv_errors)
    loocv_rmse = np.sqrt(np.mean(np.array(loocv_errors)**2))
    
    # q-hat (finite-sample corrected)
    alpha_conf = 0.10
    q_level = np.ceil((N + 1) * (1.0 - alpha_conf)) / N
    q_hat = float(np.quantile(loocv_errors, min(q_level, 1.0)))
    
    # LOCO
    unique_fams = sorted(list(set(families)))
    fam_maes = {}
    loco_all_errs = []
    withheld_count = 0
    for fam in unique_fams:
        test_idx = [j for j in range(N) if families[j] == fam]
        train_idx = [j for j in range(N) if families[j] != fam]
        sc = StandardScaler()
        X_tr = sc.fit_transform(X[train_idx])
        y_tr = y[train_idx]
        m = Ridge(alpha=alpha)
        m.fit(X_tr, y_tr)
        
        X_te = sc.transform(X[test_idx])
        d_preds = m.predict(X_te)
        fam_errs = []
        for idx_in_test, global_idx in enumerate(test_idx):
            d_p = d_preds[idx_in_test]
            if d_p <= 0:
                withheld_count += 1
            pred_abs = records[global_idx][0] + d_p
            err = abs(targets[global_idx] - pred_abs)
            fam_errs.append(err)
            loco_all_errs.append(err)
        fam_maes[fam] = (len(test_idx), np.mean(fam_errs))
        
    pooled_loco = np.mean(loco_all_errs)
    macro_loco = np.mean([v[1] for v in fam_maes.values()])
    
    # Leverage (Hat matrix)
    X_aug = np.column_stack([np.ones(N), X_s])
    I_p = np.eye(X_aug.shape[1])
    I_p[0, 0] = 0.0 # Don't regularize intercept
    H = X_aug @ np.linalg.inv(X_aug.T @ X_aug + alpha * I_p) @ X_aug.T
    leverages = np.diag(H)
    
    return {
        "in_sample_mae": in_sample_mae,
        "in_sample_rmse": in_sample_rmse,
        "loocv_mae": loocv_mae,
        "loocv_rmse": loocv_rmse,
        "q_hat": q_hat,
        "pooled_loco": pooled_loco,
        "macro_loco": macro_loco,
        "fam_maes": fam_maes,
        "withheld_count": withheld_count,
        "leverages": leverages,
        "formulas": formulas,
        "tr_H": np.trace(H)
    }

print("=== RUNNING ABLATION STUDY FOR LOCO & LOOCV METRICS ===")
# 1. Old features + Old eps
res_old = evaluate_model(OLD_RECORDS, alpha=1.0)
print("\n[1] OLD FEATURES + OLD EPS (alpha=1.0):")
print(f"  LOOCV MAE: {res_old['loocv_mae']:.4f} eV, RMSE: {res_old['loocv_rmse']:.4f} eV, q-hat: {res_old['q_hat']:.4f} eV")
print(f"  Pooled LOCO: {res_old['pooled_loco']:.4f} eV, Macro LOCO: {res_old['macro_loco']:.4f} eV")
for fam, (cnt, mae) in res_old['fam_maes'].items():
    print(f"    - {fam:25s} (n={cnt}): {mae:.4f} eV")

# 2. Old features + New eps (ALT_EPS)
OLD_FEATS_NEW_EPS = []
for r in OLD_RECORDS:
    OLD_FEATS_NEW_EPS.append((r[0], r[1], r[2], r[3], r[4], ALT_EPS[r[6]], r[6], r[7]))
res_oldf_neweps = evaluate_model(OLD_FEATS_NEW_EPS, alpha=1.0)
print("\n[2] OLD FEATURES + NEW EPS (alpha=1.0):")
print(f"  LOOCV MAE: {res_oldf_neweps['loocv_mae']:.4f} eV, RMSE: {res_oldf_neweps['loocv_rmse']:.4f} eV, q-hat: {res_oldf_neweps['q_hat']:.4f} eV")
print(f"  Pooled LOCO: {res_oldf_neweps['pooled_loco']:.4f} eV, Macro LOCO: {res_oldf_neweps['macro_loco']:.4f} eV")
for fam, (cnt, mae) in res_oldf_neweps['fam_maes'].items():
    print(f"    - {fam:25s} (n={cnt}): {mae:.4f} eV")

# 3. New composition features + Old eps
NEW_FEATS_OLD_EPS = []
for r, r_old in zip(CALIBRATION_RECORDS, OLD_RECORDS):
    NEW_FEATS_OLD_EPS.append((r[0], r[1], r[2], r[3], r[4], r_old[5], r[6], r[7], r[8], r[9]))
res_newf_oldeps = evaluate_model(NEW_FEATS_OLD_EPS, alpha=1.0)
print("\n[3] NEW COMPOSITION FEATURES + OLD EPS (alpha=1.0):")
print(f"  LOOCV MAE: {res_newf_oldeps['loocv_mae']:.4f} eV, RMSE: {res_newf_oldeps['loocv_rmse']:.4f} eV, q-hat: {res_newf_oldeps['q_hat']:.4f} eV")
print(f"  Pooled LOCO: {res_newf_oldeps['pooled_loco']:.4f} eV, Macro LOCO: {res_newf_oldeps['macro_loco']:.4f} eV")
for fam, (cnt, mae) in res_newf_oldeps['fam_maes'].items():
    print(f"    - {fam:25s} (n={cnt}): {mae:.4f} eV")

# 4. New composition features + New eps (ALT_EPS)
NEW_FEATS_ALT_EPS = []
for r in CALIBRATION_RECORDS:
    NEW_FEATS_ALT_EPS.append((r[0], r[1], r[2], r[3], r[4], ALT_EPS[r[6]], r[6], r[7], r[8], r[9]))
res_newf_alteps = evaluate_model(NEW_FEATS_ALT_EPS, alpha=1.0)
print("\n[4] NEW COMPOSITION FEATURES + NEW EPS (alpha=1.0):")
print(f"  LOOCV MAE: {res_newf_alteps['loocv_mae']:.4f} eV, RMSE: {res_newf_alteps['loocv_rmse']:.4f} eV, q-hat: {res_newf_alteps['q_hat']:.4f} eV")
print(f"  Pooled LOCO: {res_newf_alteps['pooled_loco']:.4f} eV, Macro LOCO: {res_newf_alteps['macro_loco']:.4f} eV")
for fam, (cnt, mae) in res_newf_alteps['fam_maes'].items():
    print(f"    - {fam:25s} (n={cnt}): {mae:.4f} eV")

print("\n=== LEVERAGE TABLE (NEW COMPOSITION FEATURES + CALIBRATION_RECORDS EPS, alpha=1.0) ===")
print(f"{'Formula':10s} | {'h_ii':8s} | {'pbe_gap':8s} | {'eps_inf':8s} | {'High Lev (Ridge cutoff=0.5521)'}")
print("-" * 65)
for form, h, r in zip(res_newf_oldeps['formulas'], res_newf_oldeps['leverages'], CALIBRATION_RECORDS):
    flag = "YES (*)" if h > 2 * res_newf_oldeps['tr_H'] / 21 else "NO"
    print(f"{form:10s} | {h:8.4f} | {r[0]:8.3f} | {r[5]:8.2f} | {flag}")
