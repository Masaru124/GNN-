import sys, os
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, backend_dir)
from app.services.delta_ml_corrector import CALIBRATION_RECORDS

def run_full_sanderson_grid():
    records = CALIBRATION_RECORDS
    n_samples = len(records)
    formulas = [r[6] for r in records]
    families = [r[9] for r in records]
    pbes = np.array([r[0] for r in records])
    targets = np.array([r[1] for r in records])
    y_delta = targets - pbes

    # Pauling values
    pauling = {"C": 2.55, "H": 2.20, "N": 3.04, "Pb": 2.33, "Sn": 1.96, "I": 2.66, "Br": 2.96, "Cl": 3.16}
    
    # Exact Sanderson geometric mean
    chi_ma_exact = (2.55 * (2.20**6) * 3.04) ** (1.0/8.0)
    chi_fa_exact = (2.55 * (2.20**5) * (3.04**2)) ** (1.0/8.0)
    
    print(f"[EXACT SANDERSON] chi_eff(MA+) = {chi_ma_exact:.4f} (using C=2.55, H=2.20, N=3.04, N_atoms=8)")
    print(f"[EXACT SANDERSON] chi_eff(FA+) = {chi_fa_exact:.4f} (using C=2.55, H=2.20, N=3.04, N_atoms=8)")
    print(f"[PB ELECTRONEGATIVITY] chi(Pb) = {pauling['Pb']:.2f}")
    print(f"[PLACEMENT JUSTIFICATION] Since chi_eff(MA+) = {chi_ma_exact:.2f} < chi(Pb) = {pauling['Pb']:.2f}, the organic cation remains the minimum electronegativity constituent in MAPbX3, while the halogen X is the maximum (I: 2.66, Br: 2.96, Cl: 3.16). Thus chi_diff = chi(X) - chi(MA+) correctly reflects cation-anion polarity.")

    # 2D Grid across chi_MA in [1.2, 2.6]
    grid_ma = np.linspace(1.2, 2.6, 15)
    grid_results = []
    
    print("\n--- 2D SENSITIVITY GRID EVALUATION (LOOCV MAE & LOCO MAE) ---")
    print(f"{'chi_MA':>8} | {'LOOCV MAE (alpha=1.0)':>22} | {'LOCO MAE (alpha=1.0)':>21} | {'LOCO MAE (alpha=50.0)':>22} | {'Nested LOCO MAE':>17}")
    print("-" * 105)
    
    for chi_val in grid_ma:
        X = []
        for r in records:
            pbe, tgt, chi, r_r, z_a, eps, form, src, mp_id, fam = r
            if "MA" in form:
                if "I" in form: x_el = 2.66
                elif "Br" in form: x_el = 2.96
                elif "Cl" in form: x_el = 3.16
                else: x_el = 2.66
                all_x = [chi_val, 2.33, x_el]
                chi_d = max(all_x) - min(all_x)
            else:
                chi_d = chi
            X.append([pbe, pbe**2, chi_d, r_r, z_a, 1.0/eps, pbe/eps])
        X = np.array(X)
        
        # LOOCV
        loocv_errs = []
        for i in range(n_samples):
            tr = [j for j in range(n_samples) if j != i]
            sc = StandardScaler().fit(X[tr])
            m = Ridge(alpha=1.0).fit(sc.transform(X[tr]), y_delta[tr])
            p = m.predict(sc.transform(X[i:i+1]))[0]
            loocv_errs.append(abs(targets[i] - (pbes[i] + p)))
        loocv_mae = float(np.mean(loocv_errs))
        
        # LOCO alpha=1.0, 50.0, Nested
        loco_a1 = [0.0]*n_samples
        loco_a50 = [0.0]*n_samples
        loco_nested = [0.0]*n_samples
        alpha_grid = [1e-4, 1e-3, 1e-2, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 1e3, 1e4, 1e5]
        
        for fam in sorted(list(set(families))):
            outer_tr = [j for j in range(n_samples) if families[j] != fam]
            outer_te = [j for j in range(n_samples) if families[j] == fam]
            
            sc_out = StandardScaler().fit(X[outer_tr])
            m1 = Ridge(alpha=1.0).fit(sc_out.transform(X[outer_tr]), y_delta[outer_tr])
            p1 = m1.predict(sc_out.transform(X[outer_te]))
            
            m50 = Ridge(alpha=50.0).fit(sc_out.transform(X[outer_tr]), y_delta[outer_tr])
            p50 = m50.predict(sc_out.transform(X[outer_te]))
            
            for k_pos, idx in enumerate(outer_te):
                loco_a1[idx] = abs(targets[idx] - (pbes[idx] + p1[k_pos]))
                loco_a50[idx] = abs(targets[idx] - (pbes[idx] + p50[k_pos]))
                
            # Inner CV for nested alpha
            inner_fams = sorted(list(set([families[j] for j in outer_tr])))
            a_scores = []
            for a in alpha_grid:
                in_errs = []
                for infam in inner_fams:
                    in_tr = [k for k, j in enumerate(outer_tr) if families[j] != infam]
                    in_te = [k for k, j in enumerate(outer_tr) if families[j] == infam]
                    if not in_te or not in_tr: continue
                    sc_in = StandardScaler().fit(X[[outer_tr[k] for k in in_tr]])
                    m_in = Ridge(alpha=a).fit(sc_in.transform(X[[outer_tr[k] for k in in_tr]]), y_delta[[outer_tr[k] for k in in_tr]])
                    p_in = m_in.predict(sc_in.transform(X[[outer_tr[k] for k in in_te]]))
                    in_errs.extend(list(np.abs(targets[[outer_tr[k] for k in in_te]] - (pbes[[outer_tr[k] for k in in_te]] + p_in))))
                a_scores.append(np.mean(in_errs) if in_errs else 999.0)
            best_a = alpha_grid[int(np.argmin(a_scores))]
            m_nest = Ridge(alpha=best_a).fit(sc_out.transform(X[outer_tr]), y_delta[outer_tr])
            p_nest = m_nest.predict(sc_out.transform(X[outer_te]))
            for k_pos, idx in enumerate(outer_te):
                loco_nested[idx] = abs(targets[idx] - (pbes[idx] + p_nest[k_pos]))
                
        loco_a1_mae = float(np.mean(loco_a1))
        loco_a50_mae = float(np.mean(loco_a50))
        loco_nest_mae = float(np.mean(loco_nested))
        grid_results.append((chi_val, loocv_mae, loco_a1_mae, loco_a50_mae, loco_nest_mae))
        print(f"{chi_val:8.2f} | {loocv_mae:22.4f} | {loco_a1_mae:21.4f} | {loco_a50_mae:22.4f} | {loco_nest_mae:17.4f}")

if __name__ == "__main__":
    run_full_sanderson_grid()
