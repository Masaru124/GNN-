import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app.services.delta_ml_corrector import CALIBRATION_RECORDS, DeltaMLGapCorrector
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

records = CALIBRATION_RECORDS
n_samples = len(records)
formulas = [r[6] for r in records]
families = [r[9] for r in records]
pbes = np.array([r[0] for r in records])
targets = np.array([r[1] for r in records])
y_delta = targets - pbes

corrector = DeltaMLGapCorrector()
X_full = np.array([corrector._make_features(r[0], r[2], r[3], r[4], r[5]) for r in records])

# Scissor LOCO
bl_loco_scissor = [0.0]*n_samples
for fam in sorted(list(set(families))):
    tr = [j for j in range(n_samples) if families[j] != fam]
    te = [j for j in range(n_samples) if families[j] == fam]
    A = np.column_stack([pbes[tr], np.ones(len(tr))])
    coeff, intercept = np.linalg.lstsq(A, targets[tr], rcond=None)[0]
    for idx in te:
        pred_lin = coeff * pbes[idx] + intercept
        bl_loco_scissor[idx] = abs(targets[idx] - pred_lin)

alpha_grid = [1e-4, 1e-3, 1e-2, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 1e3, 1e4, 1e5]

# Inner LOCO Nested Ridge
loco_nested_preds = [0.0]*n_samples
loco_nested_errs = [0.0]*n_samples
chosen_alphas_loco = {}

for fam in sorted(list(set(families))):
    outer_tr = [j for j in range(n_samples) if families[j] != fam]
    outer_te = [j for j in range(n_samples) if families[j] == fam]
    unique_inner_fams = sorted(list(set([families[j] for j in outer_tr])))
    alpha_scores = []
    for a_cand in alpha_grid:
        inner_errs = []
        for infam in unique_inner_fams:
            in_tr = [idx for idx, j in enumerate(outer_tr) if families[j] != infam]
            in_te = [idx for idx, j in enumerate(outer_tr) if families[j] == infam]
            if not in_te or not in_tr: continue
            sc = StandardScaler()
            X_in_tr = sc.fit_transform(X_full[[outer_tr[k] for k in in_tr]])
            y_in_tr = y_delta[[outer_tr[k] for k in in_tr]]
            m_in = Ridge(alpha=a_cand, fit_intercept=True).fit(X_in_tr, y_in_tr)
            X_in_te = sc.transform(X_full[[outer_tr[k] for k in in_te]])
            p_in = m_in.predict(X_in_te)
            inner_errs.extend(list(np.abs(targets[[outer_tr[k] for k in in_te]] - (pbes[[outer_tr[k] for k in in_te]] + p_in))))
        alpha_scores.append(np.mean(inner_errs))
    best_alpha = alpha_grid[int(np.argmin(alpha_scores))]
    chosen_alphas_loco[fam] = (best_alpha, min(alpha_scores))
    
    sc_out = StandardScaler()
    X_out_tr = sc_out.fit_transform(X_full[outer_tr])
    y_out_tr = y_delta[outer_tr]
    m_out = Ridge(alpha=best_alpha, fit_intercept=True).fit(X_out_tr, y_out_tr)
    X_out_te = sc_out.transform(X_full[outer_te])
    d_pred = m_out.predict(X_out_te)
    gap_pred = pbes[outer_te] + d_pred
    for k_idx, global_idx in enumerate(outer_te):
        loco_nested_preds[global_idx] = float(gap_pred[k_idx])
        loco_nested_errs[global_idx] = float(abs(targets[global_idx] - gap_pred[k_idx]))

print("=== INNER-LOCO NESTED RIDGE (STRICTLY NON-LEAKY) ===")
for fam in sorted(list(set(families))):
    a_sel, in_mae = chosen_alphas_loco[fam]
    print(f"Outer Fold Family: {fam:28s} | Selected alpha = {a_sel:8.1f} (Inner LOCO MAE = {in_mae:.4f} eV)")

print("\n=== PER-COMPOUND LOCO ERRORS & PAIRED DIFFS (NESTED RIDGE vs SCISSOR) ===")
print(f"{'Formula':10s} {'Family':28s} {'Target':7s} {'NestedPred':10s} {'NestedErr':10s} {'ScissorErr':10s} {'Diff(Ridge-Sciss)':17s}")
for i in range(n_samples):
    d_err = loco_nested_errs[i] - bl_loco_scissor[i]
    print(f"{formulas[i]:10s} {families[i]:28s} {targets[i]:7.2f} {loco_nested_preds[i]:10.4f} {loco_nested_errs[i]:10.4f} {bl_loco_scissor[i]:10.4f} {d_err:+17.4f}")

print("\n=== FAMILY-LEVEL SUMMARY & PAIRED DIFFERENCES ===")
for fam in sorted(list(set(families))):
    f_idx = [j for j in range(n_samples) if families[j] == fam]
    m_nest = np.mean([loco_nested_errs[k] for k in f_idx])
    m_scis = np.mean([bl_loco_scissor[k] for k in f_idx])
    diff_fam = m_nest - m_scis
    print(f"{fam:28s} (n={len(f_idx):2d}) | Nested MAE: {m_nest:.4f} eV | Scissor MAE: {m_scis:.4f} eV | Diff: {diff_fam:+.4f} eV")

print(f"\nOverall Pooled Nested Ridge LOCO MAE: {np.mean(loco_nested_errs):.4f} eV")
print(f"Overall Pooled Scissor LOCO MAE:      {np.mean(bl_loco_scissor):.4f} eV")
print(f"Overall Paired Delta MAE:             {np.mean(np.array(loco_nested_errs) - np.array(bl_loco_scissor)):+.4f} eV")
