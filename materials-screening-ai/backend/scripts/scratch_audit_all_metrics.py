import os, sys, json, hashlib
import numpy as np
import sklearn
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from pymatgen.core import Composition

sys.path.insert(0, os.path.abspath("materials-screening-ai/backend"))
from app.services.delta_ml_corrector import CALIBRATION_RECORDS, DeltaMLGapCorrector

def audit():
    records = CALIBRATION_RECORDS
    n_samples = len(records)
    formulas = [r[6] for r in records]
    families = [r[9] for r in records]
    pbes = np.array([r[0] for r in records])
    targets = np.array([r[1] for r in records])
    y_delta = targets - pbes
    eps_vals = np.array([r[5] for r in records])

    corrector = DeltaMLGapCorrector()
    X_full = np.array([corrector._make_features(r[0], r[2], r[3], r[4], r[5]) for r in records])

    print("================================================================================")
    print("1. DEPLOYED PREDICTOR EVALUATION (Nested LOCO vs Linear PBE)")
    print("================================================================================")
    
    # Linear PBE baseline
    loco_lin_errs = np.zeros(n_samples)
    for fam in set(families):
        tr = [j for j in range(n_samples) if families[j] != fam]
        te = [j for j in range(n_samples) if families[j] == fam]
        A_tr = np.column_stack([pbes[tr], np.ones(len(tr))])
        c_tr, int_tr = np.linalg.lstsq(A_tr, targets[tr], rcond=None)[0]
        for idx in te:
            loco_lin_errs[idx] = abs(targets[idx] - (c_tr * pbes[idx] + int_tr))

    A_all = np.column_stack([pbes, np.ones(n_samples)])
    lin_coeff, lin_int = np.linalg.lstsq(A_all, targets, rcond=None)[0]
    print(f"Linear PBE Fit (N=21): HSE = {lin_coeff:.4f} * PBE + {lin_int:.4f}")
    print(f"Linear PBE LOCO MAE:   {np.mean(loco_lin_errs):.4f} eV")

    # Extended alpha grid
    alpha_grid = [1e-4, 1e-3, 1e-2, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 1e3, 1e4, 1e5]
    loco_nested_errs = np.zeros(n_samples)
    chosen_alphas = []
    for fam in sorted(list(set(families))):
        outer_tr = [j for j in range(n_samples) if families[j] != fam]
        outer_te = [j for j in range(n_samples) if families[j] == fam]
        inner_fams = sorted(list(set([families[j] for j in outer_tr])))
        a_scores = []
        a_ses = []
        for a in alpha_grid:
            in_errs = []
            for infam in inner_fams:
                in_tr = [idx for idx, j in enumerate(outer_tr) if families[j] != infam]
                in_te = [idx for idx, j in enumerate(outer_tr) if families[j] == infam]
                sc = StandardScaler()
                X_in_tr = sc.fit_transform(X_full[[outer_tr[k] for k in in_tr]])
                m_in = Ridge(alpha=a).fit(X_in_tr, y_delta[[outer_tr[k] for k in in_tr]])
                p_in = m_in.predict(sc.transform(X_full[[outer_tr[k] for k in in_te]]))
                in_errs.extend(list(np.abs(targets[[outer_tr[k] for k in in_te]] - (pbes[[outer_tr[k] for k in in_te]] + p_in))))
            a_scores.append(np.mean(in_errs))
            a_ses.append(np.std(in_errs) / np.sqrt(len(in_errs)))
        best_idx = int(np.argmin(a_scores))
        best_alpha = alpha_grid[best_idx]
        chosen_alphas.append((fam, best_alpha))
        sc_out = StandardScaler()
        X_out_tr = sc_out.fit_transform(X_full[outer_tr])
        m_out = Ridge(alpha=best_alpha).fit(X_out_tr, y_delta[outer_tr])
        p_out = m_out.predict(sc_out.transform(X_full[outer_te]))
        for k_pos, idx in enumerate(outer_te):
            loco_nested_errs[idx] = abs(targets[idx] - (pbes[idx] + p_out[k_pos]))

    print(f"Nested Ridge LOCO MAE: {np.mean(loco_nested_errs):.4f} eV")
    print(f"Chosen Alphas: {chosen_alphas}")

    # Paired Delta MAE Bootstrap
    rng = np.random.default_rng(42)
    deltas = loco_nested_errs - loco_lin_errs
    boot_deltas = []
    for _ in range(10000):
        b_idx = rng.choice(n_samples, size=n_samples, replace=True)
        boot_deltas.append(np.mean(deltas[b_idx]))
    ci_lower = np.percentile(boot_deltas, 2.5)
    ci_upper = np.percentile(boot_deltas, 97.5)
    point_delta = np.mean(deltas)
    print(f"Paired Delta MAE (Nested Ridge - Linear PBE): {point_delta:+.4f} eV, 95% CI: [{ci_lower:+.4f}, {ci_upper:+.4f}] eV (N={n_samples})")

    # In-family Halide Perovskite check
    pero_indices = [i for i in range(n_samples) if families[i] == "halide_perovskite"]
    ridge_loocv_errs_pero = []
    fam_mean_errs_pero = []
    for i in pero_indices:
        tr = [j for j in range(n_samples) if j != i]
        sc = StandardScaler()
        m = Ridge(alpha=1.0).fit(sc.fit_transform(X_full[tr]), y_delta[tr])
        p_d = m.predict(sc.transform(X_full[i:i+1]))[0]
        ridge_loocv_errs_pero.append(abs(targets[i] - (pbes[i] + p_d)))

        fam_tr = [j for j in pero_indices if j != i]
        fam_d = np.mean(y_delta[fam_tr])
        fam_mean_errs_pero.append(abs(targets[i] - (pbes[i] + fam_d)))

    print("\nIn-Family Halide Perovskites (n=10):")
    print(f"  Ridge (alpha=1.0) LOOCV MAE: {np.mean(ridge_loocv_errs_pero):.4f} eV")
    print(f"  Family-Mean-Delta LOOCV MAE: {np.mean(fam_mean_errs_pero):.4f} eV")
    print(f"  Ridge Advantage:             {np.mean(fam_mean_errs_pero) - np.mean(ridge_loocv_errs_pero):.4f} eV")

    print("\n================================================================================")
    print("2. STANDARDIZED COEFFICIENTS & EXTENDED ALPHA GRID")
    print("================================================================================")
    feat_names = ["PBE", "PBE^2", "chi_diff", "r_ratio", "Z_avg", "1/eps_inf", "PBE/eps_inf"]
    alphas_to_print = [1.0, 10.0, 50.0, 1000.0, 100000.0]
    print(f"{'Feature':15s} | " + " | ".join([f"alpha={a:<8.0f}" for a in alphas_to_print]))
    print("-" * 75)
    scaler_full = StandardScaler().fit(X_full)
    Z_full = scaler_full.transform(X_full)
    for f_idx, fn in enumerate(feat_names):
        coeffs = []
        for a in alphas_to_print:
            m = Ridge(alpha=a, fit_intercept=True).fit(Z_full, y_delta)
            coeffs.append(f"{m.coef_[f_idx]:8.4f}")
        print(f"{fn:15s} | " + " | ".join(coeffs))

    # LOCO MAE for each alpha
    loco_by_alpha = []
    for a in alphas_to_print:
        l_errs = []
        for fam in set(families):
            tr = [j for j in range(n_samples) if families[j] != fam]
            te = [j for j in range(n_samples) if families[j] == fam]
            sc = StandardScaler()
            m = Ridge(alpha=a).fit(sc.fit_transform(X_full[tr]), y_delta[tr])
            p = m.predict(sc.transform(X_full[te]))
            l_errs.extend(list(np.abs(targets[te] - (pbes[te] + p))))
        loco_by_alpha.append(f"{np.mean(l_errs):8.4f}")
    print("-" * 75)
    print(f"{'LOCO MAE (eV)':15s} | " + " | ".join(loco_by_alpha))

    print("\n================================================================================")
    print("3. LEAVE-ONE-FAMILY-OUT (LOFO) HELD-OUT CONFORMAL COVERAGE")
    print("================================================================================")
    unique_fams = sorted(list(set(families)))
    tot_covered = 0
    p_dim = X_full.shape[1]
    for fam in unique_fams:
        tr_idx = [j for j in range(n_samples) if families[j] != fam]
        te_idx = [j for j in range(n_samples) if families[j] == fam]
        n_tr = len(tr_idx)
        
        # LOOCV on tr_idx with query leverage
        tr_loo_norm_scores = []
        for k_pos, tr_i in enumerate(tr_idx):
            sub_tr = [j for j in tr_idx if j != tr_i]
            sc_sub = StandardScaler().fit(X_full[sub_tr])
            Z_sub = sc_sub.transform(X_full[sub_tr])
            z_k = sc_sub.transform(X_full[tr_i:tr_i+1])
            H_inv = np.linalg.inv(Z_sub.T @ Z_sub + 1.0 * np.eye(p_dim))
            h_k = float((1.0 / len(sub_tr)) + (z_k @ H_inv @ z_k.T)[0, 0])
            
            m_sub = Ridge(alpha=1.0).fit(Z_sub, y_delta[sub_tr])
            pred_sub = m_sub.predict(z_k)[0]
            err_sub = abs(targets[tr_i] - (pbes[tr_i] + pred_sub))
            tr_loo_norm_scores.append(err_sub / np.sqrt(1.0 + h_k))
            
        k_order = int(np.ceil((n_tr + 1) * 0.90))
        q_tilde_cal = float(sorted(tr_loo_norm_scores)[min(k_order - 1, n_tr - 1)])
        
        sc_tr = StandardScaler().fit(X_full[tr_idx])
        Z_tr = sc_tr.transform(X_full[tr_idx])
        m_tr = Ridge(alpha=1.0).fit(Z_tr, y_delta[tr_idx])
        H_tr_inv = np.linalg.inv(Z_tr.T @ Z_tr + 1.0 * np.eye(p_dim))
        
        covered_count = 0
        for te_i in te_idx:
            z_te = sc_tr.transform(X_full[te_i:te_i+1])
            h_te = float((1.0 / n_tr) + (z_te @ H_tr_inv @ z_te.T)[0, 0])
            pred_delta = m_tr.predict(z_te)[0]
            pred_gap = pbes[te_i] + pred_delta
            err_te = abs(targets[te_i] - pred_gap)
            half_width = q_tilde_cal * np.sqrt(1.0 + h_te)
            is_cov = (err_te <= half_width)
            if is_cov:
                covered_count += 1
            tot_covered += int(is_cov)
        pct = (covered_count / len(te_idx)) * 100.0
        print(f"Family '{fam:28s}' (n={len(te_idx)}): {covered_count}/{len(te_idx)} ({pct:5.1f}%) | q_tilde_cal = {q_tilde_cal:.4f} eV")

    print(f"Overall LOFO Coverage: {tot_covered}/{n_samples} ({tot_covered/n_samples*100:.1f}%)")

    # Within-family repeated splits for Halide Perovskites
    pero_arr = np.array(pero_indices)
    within_covs = []
    for s_idx in range(50):
        rng_split = np.random.default_rng(100 + s_idx)
        perm = rng_split.permutation(len(pero_arr))
        tr_p = pero_arr[perm[:7]]
        te_p = pero_arr[perm[7:]]
        
        # LOOCV on tr_p
        tr_norm = []
        for ti in tr_p:
            sub = [j for j in tr_p if j != ti]
            sc_s = StandardScaler().fit(X_full[sub])
            m_s = Ridge(alpha=1.0).fit(sc_s.transform(X_full[sub]), y_delta[sub])
            z_ti = sc_s.transform(X_full[ti:ti+1])
            H_inv = np.linalg.inv(sc_s.transform(X_full[sub]).T @ sc_s.transform(X_full[sub]) + 1.0 * np.eye(p_dim))
            h_ti = float((1.0 / len(sub)) + (z_ti @ H_inv @ z_ti.T)[0, 0])
            e_ti = abs(targets[ti] - (pbes[ti] + m_s.predict(z_ti)[0]))
            tr_norm.append(e_ti / np.sqrt(1.0 + h_ti))
        k_p = int(np.ceil((len(tr_p) + 1) * 0.90))
        q_p = float(sorted(tr_norm)[min(k_p - 1, len(tr_p) - 1)])
        
        sc_tp = StandardScaler().fit(X_full[tr_p])
        m_tp = Ridge(alpha=1.0).fit(sc_tp.transform(X_full[tr_p]), y_delta[tr_p])
        H_tp_inv = np.linalg.inv(sc_tp.transform(X_full[tr_p]).T @ sc_tp.transform(X_full[tr_p]) + 1.0 * np.eye(p_dim))
        
        cov_te = 0
        for ti in te_p:
            z_ti = sc_tp.transform(X_full[ti:ti+1])
            h_ti = float((1.0 / len(tr_p)) + (z_ti @ H_tp_inv @ z_ti.T)[0, 0])
            pred_gap = pbes[ti] + m_tp.predict(z_ti)[0]
            err = abs(targets[ti] - pred_gap)
            if err <= q_p * np.sqrt(1.0 + h_ti):
                cov_te += 1
        within_covs.append(cov_te / len(te_p))
    print(f"Within-Family Repeated Splits Coverage (Halide Perovskites 70/30, 50 splits): {np.mean(within_covs)*100:.1f}% +/- {np.std(within_covs)*100:.1f}%")

    print("\n================================================================================")
    print("4. EPS_INF ABLATION UNDER NESTED ALPHA")
    print("================================================================================")
    X_no_eps = np.array([[r[0], r[0]**2, r[2], r[3], r[4]] for r in records])
    loco_no_eps_nested = np.zeros(n_samples)
    for fam in sorted(list(set(families))):
        outer_tr = [j for j in range(n_samples) if families[j] != fam]
        outer_te = [j for j in range(n_samples) if families[j] == fam]
        inner_fams = sorted(list(set([families[j] for j in outer_tr])))
        a_scores = []
        for a in alpha_grid:
            in_errs = []
            for infam in inner_fams:
                in_tr = [idx for idx, j in enumerate(outer_tr) if families[j] != infam]
                in_te = [idx for idx, j in enumerate(outer_tr) if families[j] == infam]
                sc = StandardScaler()
                X_in_tr = sc.fit_transform(X_no_eps[[outer_tr[k] for k in in_tr]])
                m_in = Ridge(alpha=a).fit(X_in_tr, y_delta[[outer_tr[k] for k in in_tr]])
                p_in = m_in.predict(sc.transform(X_no_eps[[outer_tr[k] for k in in_te]]))
                in_errs.extend(list(np.abs(targets[[outer_tr[k] for k in in_te]] - (pbes[[outer_tr[k] for k in in_te]] + p_in))))
            a_scores.append(np.mean(in_errs))
        best_alpha = alpha_grid[int(np.argmin(a_scores))]
        sc_out = StandardScaler()
        m_out = Ridge(alpha=best_alpha).fit(sc_out.fit_transform(X_no_eps[outer_tr]), y_delta[outer_tr])
        p_out = m_out.predict(sc_out.transform(X_no_eps[outer_te]))
        for k_pos, idx in enumerate(outer_te):
            loco_no_eps_nested[idx] = abs(targets[idx] - (pbes[idx] + p_out[k_pos]))

    print(f"With eps_inf (Nested LOCO MAE):    {np.mean(loco_nested_errs):.4f} eV")
    print(f"Without eps_inf (Nested LOCO MAE): {np.mean(loco_no_eps_nested):.4f} eV")
    print(f"Delta (Benefit of eps_inf):        {np.mean(loco_no_eps_nested) - np.mean(loco_nested_errs):.4f} eV")

    print("\n================================================================================")
    print("5. PURE HSE SUBSET LOCO (N=14 vs N=21)")
    print("================================================================================")
    # Mismatches: SrTiO3 (B3PW), MAPbI3 (QSGW), CsPbI3/CsPbBr3/CsPbCl3/RbPbBr3 (no SOC proxy)
    pure_hse_idx = [i for i in range(n_samples) if formulas[i] not in ("SrTiO3", "MAPbI3", "CsPbI3", "CsPbBr3", "CsPbCl3", "RbPbBr3", "KCl", "KBr", "KI")]
    print(f"Pure HSE Subset size: N={len(pure_hse_idx)} compounds: {[formulas[i] for i in pure_hse_idx]}")
    pure_fams = [families[i] for i in pure_hse_idx]
    loco_pure_errs = []
    for p_i in range(len(pure_hse_idx)):
        glob_i = pure_hse_idx[p_i]
        fam_i = pure_fams[p_i]
        tr_p = [pure_hse_idx[k] for k in range(len(pure_hse_idx)) if pure_fams[k] != fam_i]
        if not tr_p:
            continue
        sc_p = StandardScaler().fit(X_full[tr_p])
        m_p = Ridge(alpha=1.0).fit(sc_p.transform(X_full[tr_p]), y_delta[tr_p])
        p_val = m_p.predict(sc_p.transform(X_full[glob_i:glob_i+1]))[0]
        loco_pure_errs.append(abs(targets[glob_i] - (pbes[glob_i] + p_val)))
    loco_ridge_full = 0.7845
    print(f"Pure HSE06 (N={len(pure_hse_idx)}) LOCO MAE: {np.mean(loco_pure_errs):.4f} eV (vs Full N=21 LOCO MAE {loco_ridge_full:.4f} eV)")

if __name__ == "__main__":
    audit()
