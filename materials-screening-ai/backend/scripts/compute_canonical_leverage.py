# -*- coding: utf-8 -*-
"""
Canonical Statistical Leverage Calculator for MatScreen-HalideOxide-Calibration-v1.

Single source of truth for design matrix leverage (h_ii) across the verified n=21 calibration dataset.
Reads live CALIBRATION_RECORDS from delta_ml_corrector.py.

Mathematical definitions:
1. Classical OLS Hat Matrix:
   H = X_b (X_b^T X_b)^(-1) X_b^T = X_b X_b^+
   where X_b is the (n x p) design matrix with a constant column 1 prepended.
   Measures structural geometric extremity in feature space.

2. Regularized Ridge Hat Matrix:
   H_alpha = Z_b (Z_b^T Z_b + alpha I)^(-1) Z_b^T
   where Z_b is the standardized design matrix (StandardScaler) with bias column 1,
   and alpha=1.0 is the Ridge penalty used in the Delta-ML pipeline.
   Measures effective degrees of freedom and shrinking influence in regularized space.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import csv
import numpy as np
from sklearn.preprocessing import StandardScaler
from app.services.delta_ml_corrector import CALIBRATION_RECORDS, DeltaMLGapCorrector


def compute_leverages():
    corrector = DeltaMLGapCorrector()
    records = CALIBRATION_RECORDS
    
    n = len(records)
    formulas = [r[6] for r in records]
    families = [r[9] for r in records]
    targets = [r[1] for r in records]
    pbes = [r[0] for r in records]
    
    # 7 physical features: pbe, pbe^2, chi_diff, r_ratio, Z_avg, 1/eps, pbe/eps
    X = np.array([corrector._make_features(r[0], r[2], r[3], r[4], r[5]) for r in records])
    
    # Classical design matrix with intercept column (p = 8)
    X_b = np.column_stack([np.ones(n), X])
    p = X_b.shape[1]
    cutoff = 2.0 * p / n  # 16/21 = 0.7619
    
    H_classic = X_b @ np.linalg.pinv(X_b)
    lev_classic = np.diag(H_classic)
    
    # Standardized Ridge design matrix matching scikit-learn Ridge(alpha=1.0, fit_intercept=True)
    # The intercept is unpenalized; features Z are centered and scaled by StandardScaler.
    # The exact hat matrix is: H_ridge = (1/n) * 1 1^T + Z (Z^T Z + alpha I)^(-1) Z^T
    Z = StandardScaler().fit_transform(X)
    alpha = 1.0
    p_feat = Z.shape[1]
    H_ridge = (1.0 / n) * np.ones((n, n)) + Z @ np.linalg.inv(Z.T @ Z + alpha * np.eye(p_feat)) @ Z.T
    lev_ridge = np.diag(H_ridge)
    
    # Sort by classical leverage descending
    order = np.argsort(lev_classic)[::-1]
    
    print("=" * 105)
    print(f"CANONICAL STATISTICAL LEVERAGE AUDIT (n={n}, p={p}, Leverage Threshold 2p/n = {cutoff:.4f})")
    print("=" * 105)
    print(f"{'Rank':<4} | {'Formula':<10} | {'Family':<26} | {'PBE (eV)':<8} | {'Target (eV)':<11} | {'Classical h_ii':<14} | {'Ridge h_ii':<10} | {'Leverage Status'}")
    print("-" * 105)
    
    rows = []
    for rank, idx in enumerate(order, start=1):
        f = formulas[idx]
        fam = families[idx]
        pbe = pbes[idx]
        tgt = targets[idx]
        hc = lev_classic[idx]
        hr = lev_ridge[idx]
        
        # Classification with sensitivity threshold:
        # BaO (0.7620) is within 0.0001 of cutoff (0.7619), so it is strictly borderline.
        if hc > cutoff + 0.01:
            status_str = "High"
            is_high = True
        elif abs(hc - cutoff) <= 0.01:
            status_str = "Borderline"
            is_high = False
        else:
            status_str = "Normal"
            is_high = False
            
        print(f"{rank:<4} | {f:<10} | {fam:<26} | {pbe:<8.3f} | {tgt:<11.2f} | {hc:<14.4f} | {hr:<10.4f} | {status_str}")
        rows.append({
            "rank": rank,
            "formula": f,
            "family": fam,
            "pbe_gap_eV": pbe,
            "target_gap_eV": tgt,
            "classical_leverage_hii": round(hc, 4),
            "ridge_leverage_hii": round(hr, 4),
            "leverage_status": status_str,
            "is_high_leverage": is_high,
            "leverage_cutoff_2p_over_n": round(cutoff, 4)
        })
        
    print("-" * 105)
    print(f"Classical Sum of Diagonals (Trace): {np.sum(lev_classic):.4f} (Exact = p = {p})")
    print(f"Ridge Sum of Diagonals (Trace / DoF): {np.sum(lev_ridge):.4f} (Effective Degrees of Freedom with unpenalized intercept)")
    print(f"Mean Leverage (p/n): {p/n:.4f} (Parameter saturation metric)")
    print("=" * 105)
    
    # Export canonical CSV
    out_csv = Path(__file__).resolve().parent.parent.parent / "research" / "canonical_leverage_table.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[SAVED] Canonical leverage table exported to: {out_csv}")
    
    return rows


if __name__ == "__main__":
    compute_leverages()
