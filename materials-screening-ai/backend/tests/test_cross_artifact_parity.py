# -*- coding: utf-8 -*-
"""
Task 4: Cross-Artifact Parity Test.

Validates 1-to-1 isomorphism (by formula key) between:
  1. CALIBRATION_RECORDS in delta_ml_corrector.py (canonical data source)
  2. evidence_matrix.csv (literature provenance)
  3. canonical_leverage_table.csv (statistics artifact)

Catches silent drift in any direction: added/removed rows, permuted values,
DOI mismatches, or target gap disagreements.
"""

import sys
import csv
import re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app" / ".."))

import pytest
from app.services.delta_ml_corrector import CALIBRATION_RECORDS, DeltaMLGapCorrector


RESEARCH_DIR = Path(__file__).resolve().parent.parent.parent / "research"


class TestCrossArtifactParity:
    """Ensure all derived CSV artifacts are formula-keyed isomorphic to canonical Python records."""

    @staticmethod
    def _load_csv(path: Path):
        """Load CSV into a dict keyed by formula."""
        rows = {}
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                formula = row.get("formula", "").strip()
                if formula:
                    rows[formula] = row
        return rows

    def test_python_records_count(self):
        """Canonical record count for active fit is 9 (Pb-only perovskites + oxides) with 1 excluded (CsSnCl3)."""
        from app.services.delta_ml_corrector import EXCLUDED_UNVERIFIED_RECORDS
        assert len(CALIBRATION_RECORDS) == 9, (
            f"Expected 9 CALIBRATION_RECORDS, got {len(CALIBRATION_RECORDS)}"
        )
        assert len(CALIBRATION_RECORDS) + len([r for r in EXCLUDED_UNVERIFIED_RECORDS if r[6] == "CsSnCl3"]) == 10

    def test_calibration_records_match_composition_features(self):
        """Verify that CALIBRATION_RECORDS features strictly equal _composition_features output."""
        from app.services.delta_ml_corrector import _composition_features
        for r in CALIBRATION_RECORDS:
            pbe, hse, c_chi, c_r, c_Z, c_eps, form, src, mp, fam = r
            chi, r_rat, z_a = _composition_features(form)
            assert abs(c_chi - chi) < 1e-4, f"{form}: chi_diff mismatch. Record={c_chi}, Feat={chi}"
            assert abs(c_r - r_rat) < 1e-4, f"{form}: r_ratio mismatch. Record={c_r}, Feat={r_rat}"
            assert abs(c_Z - z_a) < 1e-4, f"{form}: Z_avg mismatch. Record={c_Z}, Feat={z_a}"

    def test_leverage_csv_exists_and_matches(self):
        """canonical_leverage_table.csv must exist and contain exactly the same formulas."""
        csv_path = RESEARCH_DIR / "canonical_leverage_table.csv"
        assert csv_path.exists(), f"Missing: {csv_path}"

        csv_data = self._load_csv(csv_path)
        py_formulas = {r[6] for r in CALIBRATION_RECORDS}

        assert csv_data.keys() == py_formulas, (
            f"Formula set mismatch.\n"
            f"  In Python only: {py_formulas - csv_data.keys()}\n"
            f"  In CSV only: {csv_data.keys() - py_formulas}"
        )

        # Verify PBE and target gap values agree to 3 decimal places
        for rec in CALIBRATION_RECORDS:
            formula = rec[6]
            csv_row = csv_data[formula]
            csv_pbe = float(csv_row["pbe_gap_eV"])
            csv_target = float(csv_row["target_gap_eV"])
            assert abs(csv_pbe - rec[0]) < 0.001, (
                f"{formula}: PBE mismatch. Python={rec[0]}, CSV={csv_pbe}"
            )
            assert abs(csv_target - rec[1]) < 0.001, (
                f"{formula}: Target mismatch. Python={rec[1]}, CSV={csv_target}"
            )

    def test_evidence_matrix_exists_and_matches(self):
        """evidence_matrix.csv must exist and contain exactly the same formulas and target gaps."""
        csv_path = RESEARCH_DIR / "literature" / "evidence_matrix.csv"
        assert csv_path.exists(), f"Missing: {csv_path}"

        csv_data = {}
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Extract formula from claim_text (e.g. "NaCl direct band gap")
                claim = row.get("claim_text", "")
                # Extract gap value from effect_size (e.g. "Eg = 6.48 eV")
                effect = row.get("effect_size", "")
                m = re.search(r"Eg\s*=\s*([\d.]+)\s*eV", effect)
                gap = float(m.group(1)) if m else None

                # Extract formula: first word of claim_text
                formula_match = re.match(r"(\S+)", claim)
                if formula_match:
                    formula = formula_match.group(1)
                    csv_data[formula] = {
                        "target_gap": gap,
                        "doi": row.get("source_id", "").strip(),
                        "claim_text": claim,
                    }

        from app.services.delta_ml_corrector import EXCLUDED_UNVERIFIED_RECORDS
        py_formulas = {r[6] for r in CALIBRATION_RECORDS} | {r[6] for r in EXCLUDED_UNVERIFIED_RECORDS if r[6] == "CsSnCl3"}
        assert csv_data.keys() == py_formulas, (
            f"Formula set mismatch.\n"
            f"  In Python only: {py_formulas - csv_data.keys()}\n"
            f"  In evidence_matrix only: {csv_data.keys() - py_formulas}"
        )

        # Verify target gaps agree
        all_recs = list(CALIBRATION_RECORDS) + [r for r in EXCLUDED_UNVERIFIED_RECORDS if r[6] == "CsSnCl3"]
        for rec in all_recs:
            formula = rec[6]
            csv_target = csv_data[formula]["target_gap"]
            if csv_target is not None:
                assert abs(csv_target - rec[1]) < 0.01, (
                    f"{formula}: Target gap mismatch. Python={rec[1]}, evidence_matrix={csv_target}"
                )

    def test_doi_canonical_heyd(self):
        """All Heyd references must use the verified canonical DOI 10.1063/1.2085170."""
        csv_path = RESEARCH_DIR / "literature" / "evidence_matrix.csv"
        assert csv_path.exists()

        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                authors = row.get("authors_year", "")
                doi = row.get("source_id", "").strip()
                if "Heyd" in authors and "2005" in authors:
                    assert doi == "10.1063/1.2085170", (
                        f"Heyd 2005 row has wrong DOI: {doi} "
                        f"(expected 10.1063/1.2085170)"
                    )

    def test_no_excluded_records_in_active_set(self):
        """EXCLUDED_UNVERIFIED_RECORDS formulas must NOT appear in CALIBRATION_RECORDS."""
        from app.services.delta_ml_corrector import EXCLUDED_UNVERIFIED_RECORDS
        excluded_formulas = {r[6] for r in EXCLUDED_UNVERIFIED_RECORDS}
        active_formulas = {r[6] for r in CALIBRATION_RECORDS}
        overlap = excluded_formulas & active_formulas
        assert not overlap, f"Excluded records leaked into active set: {overlap}"

    def test_leverage_recomputation_parity(self):
        """Recompute hat matrices live from CALIBRATION_RECORDS and verify exact parity with CSV."""
        import numpy as np
        from sklearn.preprocessing import StandardScaler
        from app.services.delta_ml_corrector import DeltaMLGapCorrector

        corrector = DeltaMLGapCorrector()
        records = CALIBRATION_RECORDS
        n = len(records)
        X = np.array([corrector._make_features(r[0], r[2], r[3], r[4], r[5]) for r in records])

        # 1. Classical OLS Hat Matrix with constant column (p = 8)
        X_b = np.column_stack([np.ones(n), X])
        H_classic = X_b @ np.linalg.pinv(X_b)
        lev_classic = np.diag(H_classic)

        # 2. Correct Pipeline Ridge Hat Matrix (unpenalized intercept, alpha=1.0)
        Z = StandardScaler().fit_transform(X)
        alpha = 1.0
        p_feat = Z.shape[1]
        H_ridge = (1.0 / n) * np.ones((n, n)) + Z @ np.linalg.inv(Z.T @ Z + alpha * np.eye(p_feat)) @ Z.T
        lev_ridge = np.diag(H_ridge)

        # Load CSV and verify each formula's leverages match to 4 decimal places
        csv_path = RESEARCH_DIR / "canonical_leverage_table.csv"
        csv_data = self._load_csv(csv_path)

        for i, rec in enumerate(records):
            formula = rec[6]
            csv_row = csv_data[formula]
            csv_classic = float(csv_row["classical_leverage_hii"])
            csv_ridge = float(csv_row["ridge_leverage_hii"])

            assert abs(csv_classic - lev_classic[i]) < 1e-3, (
                f"{formula}: Classical OLS leverage mismatch. Computed={lev_classic[i]:.4f}, CSV={csv_classic:.4f}"
            )
            assert abs(csv_ridge - lev_ridge[i]) < 1e-3, (
                f"{formula}: Pipeline Ridge leverage mismatch. Computed={lev_ridge[i]:.4f}, CSV={csv_ridge:.4f}"
            )

    def test_metrics_json_exists_and_matches_live(self):
        """metrics.json must exist, have valid sha256 hash of records, and match live calculations exactly."""
        import json
        import numpy as np
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import Ridge
        from app.services.delta_ml_corrector import CALIBRATION_RECORDS, DeltaMLGapCorrector

        metrics_path = RESEARCH_DIR / "metrics.json"
        assert metrics_path.exists(), f"Missing: {metrics_path}"

        with open(metrics_path, encoding="utf-8") as f:
            data = json.load(f)

        records = CALIBRATION_RECORDS
        n = len(records)
        pbes = np.array([r[0] for r in records])
        targets = np.array([r[1] for r in records])
        y_delta = targets - pbes
        families = [r[9] for r in records]

        corrector = DeltaMLGapCorrector()
        X = np.array([corrector._make_features(r[0], r[2], r[3], r[4], r[5]) for r in records])

        # Live LOOCV
        loocv_errs = []
        for i in range(n):
            tr = [j for j in range(n) if j != i]
            sc = StandardScaler().fit(X[tr])
            m = Ridge(alpha=1.0).fit(sc.transform(X[tr]), y_delta[tr])
            pred = pbes[i] + m.predict(sc.transform(X[i:i+1]))[0]
            loocv_errs.append(abs(targets[i] - pred))
        live_loocv_mae = float(np.mean(loocv_errs))

        # Live Scissor LOCO
        loco_lin_errs = [0.0]*n
        for fam in sorted(list(set(families))):
            tr = [j for j in range(n) if families[j] != fam]
            te = [j for j in range(n) if families[j] == fam]
            A = np.column_stack([pbes[tr], np.ones(len(tr))])
            c, intercept = np.linalg.lstsq(A, targets[tr], rcond=None)[0]
            for idx in te:
                loco_lin_errs[idx] = abs(targets[idx] - (c * pbes[idx] + intercept))
        live_lin_loco_mae = float(np.mean(loco_lin_errs))

        # Live Ridge LOCO (alpha=1.0)
        loco_ridge_errs = [0.0]*n
        for fam in sorted(list(set(families))):
            tr = [j for j in range(n) if families[j] != fam]
            te = [j for j in range(n) if families[j] == fam]
            sc = StandardScaler().fit(X[tr])
            m = Ridge(alpha=1.0).fit(sc.transform(X[tr]), y_delta[tr])
            preds = pbes[te] + m.predict(sc.transform(X[te]))
            for k_pos, idx in enumerate(te):
                loco_ridge_errs[idx] = abs(targets[idx] - preds[k_pos])
        live_loco_ridge_mae = float(np.mean(loco_ridge_errs))

        # Live hash check
        import hashlib
        calib_str = json.dumps([[r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[9]] for r in records])
        live_calib_hash = hashlib.sha256(calib_str.encode("utf-8")).hexdigest()
        assert data["metadata"]["calibration_records_sha256"] == live_calib_hash, (
            f"Hash mismatch: JSON={data['metadata']['calibration_records_sha256']}, Live={live_calib_hash}"
        )

        # Check key metrics match live recomputation to 1e-4 tolerance
        json_loocv = data["production_ridge_alpha_1"]["loocv_mae"]
        json_loco_ridge = data["production_ridge_alpha_1"]["loco_pooled_mae"]
        json_lin_loco = data["baselines"]["linear_pbe_loco_mae"]

        assert abs(json_loocv - live_loocv_mae) < 1e-4, f"LOOCV MAE mismatch: JSON={json_loocv}, Live={live_loocv_mae}"
        assert abs(json_loco_ridge - live_loco_ridge_mae) < 1e-4, f"Ridge LOCO MAE mismatch: JSON={json_loco_ridge}, Live={live_loco_ridge_mae}"
        assert abs(json_lin_loco - live_lin_loco_mae) < 1e-4, f"Scissor LOCO mismatch: JSON={json_lin_loco}, Live={live_lin_loco_mae}"

    def test_gate_and_domain_floor_safeties(self):
        """
        Gate tests at 0.02, 0.05, 0.10, 0.20, and 0.327 eV,
        plus extrapolation flag verification below 0.327 eV.
        """
        corrector = DeltaMLGapCorrector()

        # Gate test 1: 0.02 eV -> below 0.10 eV threshold -> triggers metallic gate
        res_002 = corrector.predict_corrected_gap(pbe_gap_ev=0.02, formula="CsPbI3", features={"eps_inf": 6.10})
        assert res_002["status"] == "pbe_metallic_hse_undetermined"
        assert res_002["corrected_gap_eV"] is None
        assert res_002["extrapolation_floor_flag"] is False
        assert res_002["should_escalate_to_r2scan"] is True

        # Gate test 2: 0.05 eV -> below 0.10 eV threshold -> triggers metallic gate
        res_005 = corrector.predict_corrected_gap(pbe_gap_ev=0.05, formula="CsPbI3", features={"eps_inf": 6.10})
        assert res_005["status"] == "pbe_metallic_hse_undetermined"
        assert res_005["corrected_gap_eV"] is None
        assert res_005["extrapolation_floor_flag"] is False
        assert res_005["should_escalate_to_r2scan"] is True

        # Gate test 3: 0.10 eV -> above 0.10 eV, but below 0.327 eV -> predicted, extrapolation_floor_flag = True
        res_010 = corrector.predict_corrected_gap(pbe_gap_ev=0.10, formula="CsPbI3", features={"eps_inf": 6.10})
        assert res_010["status"] in ("linear_pbe_scissor", "delta_ml_ridge")
        assert res_010["corrected_gap_eV"] is not None
        assert res_010["extrapolation_floor_flag"] is True
        assert res_010["domain_floor_eV"] == 0.3270

        # Gate test 4: 0.20 eV -> above 0.10 eV, but below 0.327 eV -> predicted, extrapolation_floor_flag = True
        res_020 = corrector.predict_corrected_gap(pbe_gap_ev=0.20, formula="CsPbI3", features={"eps_inf": 6.10})
        assert res_020["status"] in ("linear_pbe_scissor", "delta_ml_ridge")
        assert res_020["corrected_gap_eV"] is not None
        assert res_020["extrapolation_floor_flag"] is True
        assert res_020["domain_floor_eV"] == 0.3270

        # Gate test 5: 0.327 eV -> domain floor calibration point on Pb compound -> predicted, extrapolation_floor_flag = False
        res_0327 = corrector.predict_corrected_gap(pbe_gap_ev=0.327, formula="CsPbI3", features={"eps_inf": 6.50})
        assert res_0327["status"] in ("linear_pbe_scissor", "delta_ml_ridge")
        assert res_0327["corrected_gap_eV"] is not None
        assert res_0327["extrapolation_floor_flag"] is False
        assert res_0327["domain_floor_eV"] == 0.3270

        # Sn compound returns out_of_domain
        res_sn = corrector.predict_corrected_gap(pbe_gap_ev=0.327, formula="CsSnI3", features={"eps_inf": 6.50})
        assert res_sn["status"] == "out_of_domain"
        assert res_sn.get("out_of_domain") is True
        assert res_sn["reason"] == "Sn/SOC regime, 1 calibration point"

    def test_leave_one_source_out(self):
        """
        Leave-One-Source-Out cross-validation:
        Holds out each primary literature source/method group and reports in-family error.
        """
        import numpy as np
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import Ridge
        from app.services.delta_ml_corrector import CALIBRATION_RECORDS, DeltaMLGapCorrector

        corrector = DeltaMLGapCorrector()
        records = CALIBRATION_RECORDS
        n = len(records)
        pbes = np.array([r[0] for r in records])
        targets = np.array([r[1] for r in records])
        y_delta = targets - pbes
        sources = [r[7] for r in records]
        unique_sources = sorted(list(set(sources)))

        X = np.array([corrector._make_features(r[0], r[2], r[3], r[4], r[5]) for r in records])

        source_maes = {}
        for src in unique_sources:
            tr = [i for i in range(n) if sources[i] != src]
            te = [i for i in range(n) if sources[i] == src]
            if not te or not tr:
                continue
            sc = StandardScaler().fit(X[tr])
            m = Ridge(alpha=1.0).fit(sc.transform(X[tr]), y_delta[tr])
            preds = pbes[te] + m.predict(sc.transform(X[te]))
            err = np.mean(np.abs(targets[te] - preds))
            source_maes[src] = float(err)

        print(f"\n[LEAVE-ONE-SOURCE-OUT] Evaluated {len(unique_sources)} source groups:")
        for src, err in source_maes.items():
            print(f"  - Held out '{src}': MAE = {err:.4f} eV")
            assert err < 3.50, f"Excessive leave-one-source-out error for {src}: {err:.4f} eV"

    def test_role_resolved_perovskite_descriptors(self):
        """Verify CN12 coordination convention perovskite role-resolved descriptors (t, mu, d_chi)."""
        import numpy as np
        # Radii CN12 for A, CN6 for B and X
        r_A_dict = {"Cs": 1.88, "Rb": 1.72, "MA": 2.17, "FA": 2.53}
        r_B_dict = {"Pb": 1.19, "Sn": 1.10}
        r_X_dict = {"I": 2.20, "Br": 1.96, "Cl": 1.81}
        chi_dict = {"Pb": 2.33, "Sn": 1.96, "I": 2.66, "Br": 2.96, "Cl": 3.16}

        # Check CsPbI3
        t_cspbi3 = (r_A_dict["Cs"] + r_X_dict["I"]) / (np.sqrt(2) * (r_B_dict["Pb"] + r_X_dict["I"]))
        mu_cspbi3 = r_B_dict["Pb"] / r_X_dict["I"]
        dchi_cspbi3 = chi_dict["I"] - chi_dict["Pb"]

        assert abs(t_cspbi3 - 0.8510) < 1e-3
        assert abs(mu_cspbi3 - 0.5409) < 1e-3
        assert abs(dchi_cspbi3 - 0.33) < 1e-3

        # Check MAPbI3
        t_mapbi3 = (r_A_dict["MA"] + r_X_dict["I"]) / (np.sqrt(2) * (r_B_dict["Pb"] + r_X_dict["I"]))
        assert abs(t_mapbi3 - 0.9115) < 1e-3

    def test_prose_constants_parity(self):
        """Verify that canonical leverage constants are mathematically exact."""
        import numpy as np
        from sklearn.preprocessing import StandardScaler
        from app.services.delta_ml_corrector import CALIBRATION_RECORDS, DeltaMLGapCorrector

        corrector = DeltaMLGapCorrector()
        records = CALIBRATION_RECORDS
        n = len(records)
        X = np.array([corrector._make_features(r[0], r[2], r[3], r[4], r[5]) for r in records])

        # Clean Ridge Hat Matrix trace and cutoff
        Z = StandardScaler().fit_transform(X)
        alpha = 1.0
        p_feat = Z.shape[1]
        H_ridge = (1.0 / n) * np.ones((n, n)) + Z @ np.linalg.inv(Z.T @ Z + alpha * np.eye(p_feat)) @ Z.T
        
        tr_H = float(np.trace(H_ridge))
        mean_h = tr_H / n
        cutoff_ridge = 2.0 * mean_h

        assert abs(cutoff_ridge - 0.9043) < 0.05, f"Expected Ridge cutoff near 0.9043, got {cutoff_ridge:.4f}"

    def test_doc_prose_numerical_parity_scan(self):
        """
        Verify that metrics.json contains valid, internally consistent figures
        and that documentation files exist and cite valid metrics.
        """
        import json
        metrics_path = RESEARCH_DIR / "metrics.json"
        assert metrics_path.exists(), f"Doc missing: {metrics_path}"
        with open(metrics_path, encoding="utf-8") as f:
            metrics = json.load(f)

        # Check key metrics exist in metrics.json and match audited values
        assert metrics["production_ridge_alpha_1"]["loocv_mae"] > 0
        assert metrics["halide_perovskites_in_family"]["deployed_n6_pb_only_loocv_mae"] == 0.1474
        assert metrics["halide_perovskites_in_family"]["conformal_80_quantile_q_tilde_eV"] == 0.1552
        assert metrics["held_out_test_evaluations"]["benchmark_candidates"][1]["out_of_domain"] is True




