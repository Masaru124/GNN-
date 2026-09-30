import sys, os, json, hashlib
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
X_no_eps = np.array([[r[0], r[0]**2, r[2], r[3], r[4]] for r in records])

# -----------------------------------------------------------------------------
# 1. Extended alpha grid & Standardized Coefficients
# -----------------------------------------------------------------------------
alpha_extended = [1e-4, 1e-3, 1e-2, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0, 1e3, 1e4, 1e5]
scaler = StandardScaler()
X_s = scaler.fit_transform(X_full)

print("=== STANDARDIZED COEFFICIENTS VS ALPHA ===")
feature_names = ["pbe", "pbe^2", "chi_diff", "r_ratio", "Z_avg", "1/eps", "pbe/eps"]
for a in [1.0, 10.0, 50.0, 1e3, 1e5]:
    m = Ridge(alpha=a, fit_intercept=True)
    m.fit(X_s, y_delta)
    coefs_str = ", ".join([f"{name}: {c:+.4f}" for name, c in zip(feature_names, m.coef_)])
    print(f"alpha={a:7.1f} | Intercept: {m.intercept_:+.4f} | {coefs_str}")

# -----------------------------------------------------------------------------
# 2. Nested LOCO alpha selection with extended grid & one-SE rule
# -----------------------------------------------------------------------------
def run_nested_loco(X_data, grid):
    loco_preds = np.zeros(n_samples)
    chosen_alphas = []
    unique_fams = sorted(list(set(families)))
    for fam in unique_fams:
        outer_tr = [j for j in range(n_samples) if families[j] != fam]
        outer_te = [j for j in range(n_samples) if families[j] == fam]
        
        inner_fams = sorted(list(set([families[j] for j in outer_tr])))
        alpha_scores = []
        alpha_ses = []
        for a_cand in grid:
            in_errs = []
            for infam in inner_fams:
                in_tr = [idx for idx, j in enumerate(outer_tr) if families[j] != infam]
                in_te = [idx for idx, j in enumerate(outer_tr) if families[j] == infam]
                if not in_te or not in_tr:
                    continue
                sc = StandardScaler()
                X_in_tr = sc.fit_transform(X_data[[outer_tr[k] for k in in_tr]])
                y_in_tr = y_delta[[outer_tr[k] for k in in_tr]]
                m_in = Ridge(alpha=a_cand, fit_intercept=True)
                m_in.fit(X_in_tr, y_in_tr)
                X_in_te = sc.transform(X_data[[outer_tr[k] for k in in_te]])
                p_in = m_in.predict(X_in_te)
                in_errs.extend(list(np.abs(y_delta[[outer_tr[k] for k in in_te]] - p_in)))
            alpha_scores.append(np.mean(in_errs))
            alpha_ses.append(np.std(in_errs) / np.sqrt(len(in_errs)))
            
        best_idx = int(np.argmin(alpha_scores))
        min_err = alpha_scores[best_idx]
        se_thresh = min_err + alpha_ses[best_idx]
        eligible = [idx for idx, sc in enumerate(alpha_scores) if sc <= se_thresh]
        chosen_idx = max(eligible)
        chosen_a = grid[chosen_idx]
        chosen_alphas.append((fam, chosen_a))
        
        sc_out = StandardScaler()
        X_out_tr = sc_out.fit_transform(X_data[outer_tr])
        y_out_tr = y_delta[outer_tr]
        m_out = Ridge(alpha=chosen_a, fit_intercept=True)
        m_out.fit(X_out_tr, y_out_tr)
        
        X_out_te = sc_out.transform(X_data[outer_te])
        d_pred = m_out.predict(X_out_te)
        loco_preds[outer_te] = pbes[outer_te] + d_pred
        
    errs = np.abs(targets - loco_preds)
    return float(np.mean(errs)), loco_preds, chosen_alphas

nested_mae_full, nested_preds_full, chosen_a_full = run_nested_loco(X_full, alpha_extended)
nested_mae_noeps, nested_preds_noeps, chosen_a_noeps = run_nested_loco(X_no_eps, alpha_extended)

print(f"\nNested LOCO Full (7 feats): MAE = {nested_mae_full:.4f} eV, Alphas: {chosen_a_full}")
print(f"Nested LOCO No eps_inf (5 feats): MAE = {nested_mae_noeps:.4f} eV, Alphas: {chosen_a_noeps}")

# -----------------------------------------------------------------------------
# 3. Linear PBE baseline under LOCO
# -----------------------------------------------------------------------------
lin_loco_preds = np.zeros(n_samples)
for fam in sorted(list(set(families))):
    outer_tr = [j for j in range(n_samples) if families[j] != fam]
    outer_te = [j for j in range(n_samples) if families[j] == fam]
    p_tr, t_tr = pbes[outer_tr], targets[outer_tr]
    A = np.column_stack([p_tr, np.ones(len(p_tr))])
    coeff, intercept = np.linalg.lstsq(A, t_tr, rcond=None)[0]
    lin_loco_preds[outer_te] = coeff * pbes[outer_te] + intercept

lin_loco_errs = np.abs(targets - lin_loco_preds)
lin_loco_mae = float(np.mean(lin_loco_errs))
print(f"Linear PBE baseline LOCO MAE: {lin_loco_mae:.4f} eV")

# Paired Delta-MAE bootstrap over compounds (N=21)
rng = np.random.default_rng(42)
d_errs = np.abs(targets - nested_preds_full) - lin_loco_errs # Ridge err - Lin err
boot_d_maes = []
for _ in range(2000):
    b_idx = rng.choice(n_samples, size=n_samples, replace=True)
    boot_d_maes.append(np.mean(d_errs[b_idx]))

point_d_mae = float(np.mean(d_errs))
ci_d_mae = [float(np.quantile(boot_d_maes, 0.025)), float(np.quantile(boot_d_maes, 0.975))]
print(f"Paired Delta-MAE (Ridge - Linear PBE) over N={n_samples} compounds: {point_d_mae:+.4f} eV [95% CI: {ci_d_mae[0]:+.4f}, {ci_d_mae[1]:+.4f}]")
if ci_d_mae[0] <= 0.0 <= ci_d_mae[1]:
    print("-> Result: Ridge is NOT statistically significantly better than Linear PBE scissor (CI contains 0).")

# -----------------------------------------------------------------------------
# 4. In-family: Family-Mean-Delta vs Ridge LOOCV
# -----------------------------------------------------------------------------
fam_mean_d_loocv_preds = np.zeros(n_samples)
ridge_loocv_preds = np.zeros(n_samples)
for i in range(n_samples):
    tr = [j for j in range(n_samples) if j != i]
    # Family mean delta
    fam_tr = [j for j in tr if families[j] == families[i]]
    if fam_tr:
        fam_mean_d_loocv_preds[i] = pbes[i] + np.mean(y_delta[fam_tr])
    else:
        fam_mean_d_loocv_preds[i] = pbes[i] + np.mean(y_delta[tr])
    # Ridge LOOCV
    sc = StandardScaler()
    X_tr = sc.fit_transform(X_full[tr])
    y_tr = y_delta[tr]
    m = Ridge(alpha=1.0, fit_intercept=True)
    m.fit(X_tr, y_tr)
    pred_d = m.predict(sc.transform(X_full[i:i+1]))[0]
    ridge_loocv_preds[i] = pbes[i] + pred_d

fam_mean_errs = np.abs(targets - fam_mean_d_loocv_preds)
ridge_loocv_errs = np.abs(targets - ridge_loocv_preds)
print(f"Family-Mean-Delta LOOCV MAE: {np.mean(fam_mean_errs):.4f} eV")
print(f"Ridge (alpha=1.0) LOOCV MAE: {np.mean(ridge_loocv_errs):.4f} eV")

# Halide perovskite subset (n=10)
pero_idx = [i for i in range(n_samples) if families[i] == "halide_perovskite"]
print(f"Halide Perovskites only: Family-Mean-Delta MAE = {np.mean(fam_mean_errs[pero_idx]):.4f} eV vs Ridge LOOCV MAE = {np.mean(ridge_loocv_errs[pero_idx]):.4f} eV")
