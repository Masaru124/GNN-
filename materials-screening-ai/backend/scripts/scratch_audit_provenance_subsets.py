import sys, os
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

sys.path.insert(0, os.path.abspath("materials-screening-ai/backend"))
from app.services.delta_ml_corrector import CALIBRATION_RECORDS, DeltaMLGapCorrector

records = CALIBRATION_RECORDS
formulas = [r[6] for r in records]
families = [r[9] for r in records]
pbes = np.array([r[0] for r in records])
targets = np.array([r[1] for r in records])
y_delta = targets - pbes

corrector = DeltaMLGapCorrector()
X_full = np.array([corrector._make_features(r[0], r[2], r[3], r[4], r[5]) for r in records])

# Classification of records:
# 1. Pure HSE06 / HSE06+SOC benchmark
# 2. Proxy functional (B3PW, B1-WC, GLLB-SC, QSGW)
is_pure_hse = []
for r in records:
    src = r[7]
    form = r[6]
    if "proxy" in src.lower() or "gllb" in src.lower() or "b3pw" in src.lower() or "bilc" in src.lower() or "castelli" in src.lower():
        is_pure_hse.append(False)
    else:
        is_pure_hse.append(True)

pure_indices = [i for i, val in enumerate(is_pure_hse) if val]
mismatched_indices = [i for i, val in enumerate(is_pure_hse) if not val]

print("=== DATASET FUNCTIONAL & PROVENANCE BREAKDOWN ===")
print(f"Total entries: {len(records)}")
print(f"Pure HSE06/HSE06+SOC entries: {len(pure_indices)} -> {[formulas[i] for i in pure_indices]}")
print(f"Proxy functional / mismatched entries: {len(mismatched_indices)} -> {[formulas[i] for i in mismatched_indices]}")

# LOCO on Full vs Pure HSE
def run_loco(indices):
    sub_records = [records[i] for i in indices]
    sub_n = len(sub_records)
    sub_fams = [r[9] for r in sub_records]
    sub_pbes = np.array([r[0] for r in sub_records])
    sub_targets = np.array([r[1] for r in sub_records])
    sub_y = sub_targets - sub_pbes
    sub_X = X_full[indices]
    
    # Linear PBE baseline
    lin_preds = np.zeros(sub_n)
    for f in sorted(list(set(sub_fams))):
        tr = [j for j in range(sub_n) if sub_fams[j] != f]
        te = [j for j in range(sub_n) if sub_fams[j] == f]
        if not tr or not te:
            continue
        p_tr, t_tr = sub_pbes[tr], sub_targets[tr]
        A = np.column_stack([p_tr, np.ones(len(p_tr))])
        coeff, intercept = np.linalg.lstsq(A, t_tr, rcond=None)[0]
        lin_preds[te] = coeff * sub_pbes[te] + intercept
        
    lin_mae = float(np.mean(np.abs(sub_targets - lin_preds)))
    
    # Ridge alpha=1.0
    ridge_preds = np.zeros(sub_n)
    for f in sorted(list(set(sub_fams))):
        tr = [j for j in range(sub_n) if sub_fams[j] != f]
        te = [j for j in range(sub_n) if sub_fams[j] == f]
        if not tr or not te:
            continue
        pipe = Pipeline([('scaler', StandardScaler()), ('ridge', Ridge(alpha=1.0, fit_intercept=True))])
        pipe.fit(sub_X[tr], sub_y[tr])
        ridge_preds[te] = sub_pbes[te] + pipe.predict(sub_X[te])
        
    ridge_mae = float(np.mean(np.abs(sub_targets - ridge_preds)))
    
    return lin_mae, ridge_mae

full_lin, full_ridge = run_loco(list(range(len(records))))
pure_lin, pure_ridge = run_loco(pure_indices)

print(f"\nFull n=21:     Linear PBE LOCO = {full_lin:.4f} eV | Ridge alpha=1.0 LOCO = {full_ridge:.4f} eV")
print(f"Pure HSE n=14: Linear PBE LOCO = {pure_lin:.4f} eV | Ridge alpha=1.0 LOCO = {pure_ridge:.4f} eV")
print(f"Delta on Pure HSE: Linear PBE change = {pure_lin - full_lin:+.4f} eV | Ridge change = {pure_ridge - full_ridge:+.4f} eV")
