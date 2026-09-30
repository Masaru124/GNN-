# -*- coding: utf-8 -*-
"""
Generates research/loco_residuals_alpha_comparison.csv and computes:
1. Per-compound LOCO residuals for alpha=1.0 vs alpha=0.01
2. Per-family MAE
3. Macro-averaged MAE vs Pooled MAE
"""
import sys
from pathlib import Path
sys.path.insert(0, r"c:\Users\User\Desktop\GNN\materials-screening-ai\backend")

import csv
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error

from app.services.delta_ml_corrector import CALIBRATION_RECORDS, DeltaMLGapCorrector

corrector = DeltaMLGapCorrector()
records = CALIBRATION_RECORDS
formulas = [r[6] for r in records]
families = [r[9] for r in records]
pbes = np.array([r[0] for r in records])
targets = np.array([r[1] for r in records])
y_delta = targets - pbes

X = np.array([corrector._make_features(r[0], r[2], r[3], r[4], r[5]) for r in records])

unique_families = sorted(list(set(families)))

def get_loco_predictions(alpha_val):
    preds = np.zeros(len(records))
    for fam in unique_families:
        train_idx = [i for i, f in enumerate(families) if f != fam]
        test_idx = [i for i, f in enumerate(families) if f == fam]
        
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('ridge', Ridge(alpha=alpha_val, fit_intercept=True))
        ])
        pipe.fit(X[train_idx], y_delta[train_idx])
        preds[test_idx] = pipe.predict(X[test_idx])
    return preds

preds_1_0 = get_loco_predictions(1.0)
preds_0_01 = get_loco_predictions(0.01)

err_1_0 = y_delta - preds_1_0
abs_err_1_0 = np.abs(err_1_0)

err_0_01 = y_delta - preds_0_01
abs_err_0_01 = np.abs(err_0_01)

# Write CSV
out_csv = Path(r"c:\Users\User\Desktop\GNN\materials-screening-ai\research\loco_residuals_alpha_comparison.csv")
with open(out_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([
        "formula", "family", "pbe_gap_eV", "target_gap_eV", "target_delta_eV",
        "pred_delta_alpha_1_0_eV", "abs_err_alpha_1_0_eV",
        "pred_delta_alpha_0_01_eV", "abs_err_alpha_0_01_eV"
    ])
    for i in range(len(records)):
        writer.writerow([
            formulas[i], families[i], round(pbes[i], 3), round(targets[i], 3), round(y_delta[i], 3),
            round(preds_1_0[i], 4), round(abs_err_1_0[i], 4),
            round(preds_0_01[i], 4), round(abs_err_0_01[i], 4)
        ])

print(f"[SUCCESS] Exported per-compound LOCO residuals to: {out_csv}")

# Family breakdown
print("\n=== PER-FAMILY LOCO MAE SUMMARY ===")
family_maes_1_0 = []
family_maes_0_01 = []

print(f"{'Chemistry Family':30s} | {'n':3s} | {'alpha=1.0 MAE (eV)':18s} | {'alpha=0.01 MAE (eV)':18s}")
print("-" * 75)
for fam in unique_families:
    idx = [i for i, f in enumerate(families) if f == fam]
    m_1 = np.mean(abs_err_1_0[idx])
    m_001 = np.mean(abs_err_0_01[idx])
    family_maes_1_0.append(m_1)
    family_maes_0_01.append(m_001)
    print(f"{fam:30s} | {len(idx):3d} | {m_1:18.4f} | {m_001:18.4f}")

pooled_1_0 = np.mean(abs_err_1_0)
pooled_0_01 = np.mean(abs_err_0_01)

macro_1_0 = np.mean(family_maes_1_0)
macro_0_01 = np.mean(family_maes_0_01)

print("-" * 75)
print(f"{'POOLED LOCO MAE (n=21)':30s} | {len(records):3d} | {pooled_1_0:18.4f} | {pooled_0_01:18.4f}")
print(f"{'MACRO-AVERAGED LOCO MAE (k=4)':30s} |   4 | {macro_1_0:18.4f} | {macro_0_01:18.4f}")
