import sys, os, json
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

sys.path.insert(0, os.path.abspath("materials-screening-ai/backend"))
from app.services.delta_ml_corrector import (
    DeltaMLGapCorrector,
    CALIBRATION_RECORDS,
    _composition_features,
    ORGANIC_CATIONS
)

records = CALIBRATION_RECORDS
n_samples = len(records)
formulas = [r[6] for r in records]
families = [r[9] for r in records]
pbes = np.array([r[0] for r in records])
targets = np.array([r[1] for r in records])
y_delta = targets - pbes

corrector = DeltaMLGapCorrector()
X_full = np.array([corrector._make_features(r[0], r[2], r[3], r[4], r[5]) for r in records])

# -----------------------------------------------------------------------------
# 1. Held-out query leverage for LOOCV and LOCO
# -----------------------------------------------------------------------------
loocv_query_h = np.zeros(n_samples)
loocv_errors = np.zeros(n_samples)
for i in range(n_samples):
    tr = [j for j in range(n_samples) if j != i]
    sc = StandardScaler()
    Z_tr = sc.fit_transform(X_full[tr])
    p_dim = Z_tr.shape[1]
    n_tr = len(tr)
    m = Ridge(alpha=1.0, fit_intercept=True)
    m.fit(Z_tr, y_delta[tr])
    
    Z_test = sc.transform(X_full[i:i+1])
    d_pred = m.predict(Z_test)[0]
    loocv_errors[i] = abs(targets[i] - (pbes[i] + d_pred))
    
    H_inv = np.linalg.inv(Z_tr.T @ Z_tr + 1.0 * np.eye(p_dim))
    loocv_query_h[i] = float((1.0 / n_tr) + (Z_test @ H_inv @ Z_test.T)[0, 0])

print("=== HELD-OUT QUERY LEVERAGE (LOOCV) ===")
for i in range(n_samples):
    print(f"{formulas[i]:10s} | In-sample h: {corrector.leverages[i]:.4f} | Held-out query h: {loocv_query_h[i]:.4f} | Error: {loocv_errors[i]:.4f} eV")

# Compare q_tilde in-family
# Halide perovskite subset (n=10)
pero_indices = [i for i in range(n_samples) if families[i] == "halide_perovskite"]
n_pero = len(pero_indices)
k_pero = int(np.ceil((n_pero + 1) * 0.90)) # 10th of 10

scores_pero_insample = [loocv_errors[i] / np.sqrt(1.0 + corrector.leverages[i]) for i in pero_indices]
scores_pero_query = [loocv_errors[i] / np.sqrt(1.0 + loocv_query_h[i]) for i in pero_indices]

q_tilde_pero_insample = float(sorted(scores_pero_insample)[min(k_pero - 1, n_pero - 1)])
q_tilde_pero_query = float(sorted(scores_pero_query)[min(k_pero - 1, n_pero - 1)])
print(f"\nHalide Perovskite In-Family q_tilde (n=10, rank 10/10):")
print(f"  Using in-sample leverage: {q_tilde_pero_insample:.4f} eV")
print(f"  Using held-out query leverage: {q_tilde_pero_query:.4f} eV")
print(f"  Halide Perovskite Raw max error: {max(loocv_errors[pero_indices]):.4f} eV ({formulas[pero_indices[np.argmax(loocv_errors[pero_indices])]]})")

# -----------------------------------------------------------------------------
# 2. Leave-One-Family-Out Held-Out Conformal Coverage
# -----------------------------------------------------------------------------
print("\n=== LEAVE-ONE-FAMILY-OUT HELD-OUT CONFORMAL COVERAGE ===")
unique_fams = sorted(list(set(families)))
coverage_by_fam = {}
for fam in unique_fams:
    tr = [i for i in range(n_samples) if families[i] != fam]
    te = [i for i in range(n_samples) if families[i] == fam]
    n_tr = len(tr)
    n_te = len(te)
    
    # Train on tr
    sc = StandardScaler()
    Z_tr = sc.fit_transform(X_full[tr])
    p_dim = Z_tr.shape[1]
    m = Ridge(alpha=1.0, fit_intercept=True)
    m.fit(Z_tr, y_delta[tr])
    
    # Calibrate q_tilde on tr using LOOCV within tr
    tr_scores = []
    for idx_in_tr, orig_i in enumerate(tr):
        inner_tr = [k for k in range(n_tr) if k != idx_in_tr]
        sc_in = StandardScaler()
        Z_in_tr = sc_in.fit_transform(X_full[[tr[k] for k in inner_tr]])
        m_in = Ridge(alpha=1.0, fit_intercept=True)
        m_in.fit(Z_in_tr, y_delta[[tr[k] for k in inner_tr]])
        
        Z_in_te = sc_in.transform(X_full[orig_i:orig_i+1])
        d_pred = m_in.predict(Z_in_te)[0]
        e_val = abs(targets[orig_i] - (pbes[orig_i] + d_pred))
        
        H_in_inv = np.linalg.inv(Z_in_tr.T @ Z_in_tr + 1.0 * np.eye(p_dim))
        h_val = float((1.0 / len(inner_tr)) + (Z_in_te @ H_in_inv @ Z_in_te.T)[0, 0])
        tr_scores.append(e_val / np.sqrt(1.0 + h_val))
        
    k_tr = int(np.ceil((n_tr + 1) * 0.90))
    q_tilde_cal = float(sorted(tr_scores)[min(k_tr - 1, n_tr - 1)])
    
    # Test on held-out family te
    Z_te = sc.transform(X_full[te])
    d_te_preds = m.predict(Z_te)
    H_tr_inv = np.linalg.inv(Z_tr.T @ Z_tr + 1.0 * np.eye(p_dim))
    
    fam_covered = []
    print(f"\nFamily: {fam} (n={n_te}, Calibrated q_tilde from other {n_tr} points = {q_tilde_cal:.4f} eV):")
    for pos, orig_idx in enumerate(te):
        z_pt = Z_te[pos:pos+1]
        h_query = float((1.0 / n_tr) + (z_pt @ H_tr_inv @ z_pt.T)[0, 0])
        half_width = q_tilde_cal * np.sqrt(1.0 + h_query)
        err = abs(targets[orig_idx] - (pbes[orig_idx] + d_te_preds[pos]))
        is_cov = err <= half_width
        fam_covered.append(is_cov)
        print(f"  {formulas[orig_idx]:10s} | Error: {err:.4f} eV | Half-width: ±{half_width:.4f} eV (h={h_query:.4f}) | Covered: {is_cov}")
        
    cov_rate = float(np.mean(fam_covered))
    coverage_by_fam[fam] = {"covered": int(sum(fam_covered)), "total": n_te, "rate": cov_rate}
    print(f"-> Coverage for {fam}: {sum(fam_covered)}/{n_te} ({cov_rate*100:.1f}%)")

print("\n=== SUMMARY HELD-OUT LOFO CONFORMAL COVERAGE ===")
for f_name, stats in coverage_by_fam.items():
    print(f"{f_name:30s} | {stats['covered']}/{stats['total']} ({stats['rate']*100:.1f}%)")
