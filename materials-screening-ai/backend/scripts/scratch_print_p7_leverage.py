import sys, os
import numpy as np
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.abspath("materials-screening-ai/backend"))
from app.services.delta_ml_corrector import CALIBRATION_RECORDS, DeltaMLGapCorrector

corrector = DeltaMLGapCorrector()
records = CALIBRATION_RECORDS
n = len(records)
formulas = [r[6] for r in records]
families = [r[9] for r in records]
pbes = [r[0] for r in records]
targets = [r[1] for r in records]

X = np.array([corrector._make_features(r[0], r[2], r[3], r[4], r[5]) for r in records])
p_feat = X.shape[1] # 7

# 1. Classical OLS Hat Matrix with constant column (p = 8)
X_b = np.column_stack([np.ones(n), X])
H_classic = X_b @ np.linalg.pinv(X_b)
lev_classic = np.diag(H_classic)
tr_classic = float(np.trace(H_classic))
cutoff_classic = 2.0 * tr_classic / n

# 2. Pipeline Ridge Hat Matrix (alpha=1.0, unpenalized intercept)
Z = StandardScaler().fit_transform(X)
H_ridge = (1.0 / n) * np.ones((n, n)) + Z @ np.linalg.inv(Z.T @ Z + 1.0 * np.eye(p_feat)) @ Z.T
lev_ridge = np.diag(H_ridge)
tr_ridge = float(np.trace(H_ridge))
cutoff_ridge = 2.0 * tr_ridge / n

# Sort descending by ridge leverage
order = np.argsort(lev_ridge)[::-1]

print("=========================================================================================================")
print(f"CANONICAL STATISTICAL LEVERAGE AUDIT (n={n}, p_feat={p_feat}, Ridge Cutoff = {cutoff_ridge:.4f}, OLS Cutoff = {cutoff_classic:.4f})")
print("=========================================================================================================")
print(f"{'Rank':<4} | {'Formula':<10} | {'Family':<26} | {'PBE (eV)':<8} | {'Target (eV)':<11} | {'Classical h':<12} | {'Ridge h':<10} | {'Status'}")
print("-" * 105)

high_lev_ridge = []
for rank, idx in enumerate(order, start=1):
    f = formulas[idx]
    fam = families[idx]
    pbe = pbes[idx]
    tgt = targets[idx]
    hc = lev_classic[idx]
    hr = lev_ridge[idx]
    status = "HIGH" if hr > cutoff_ridge else "Normal"
    if hr > cutoff_ridge:
        high_lev_ridge.append(f)
    print(f"{rank:<4} | {f:<10} | {fam:<26} | {pbe:<8.3f} | {tgt:<11.2f} | {hc:<12.4f} | {hr:<10.4f} | {status}")

print("-" * 105)
print(f"Ridge Trace Tr(H_ridge): {tr_ridge:.4f}")
print(f"Ridge Mean Leverage:     {tr_ridge/n:.4f}")
print(f"Ridge Cutoff (2*mean_h): {cutoff_ridge:.4f}")
print(f"High Leverage Set (Ridge): {high_lev_ridge}")
print(f"Classical OLS Trace:     {tr_classic:.4f}")
print(f"Classical OLS Cutoff:    {cutoff_classic:.4f}")
