# -*- coding: utf-8 -*-
"""
Task 5 Verification: Unit tests for the band gap parser's metallicity handling.

Tests scenarios with synthetic QE output:
  1. Clean insulating gap (cbm > vbm) -> gap_eV > 0, gap_type='insulator', band_overlap_eV=None
  2. Band overlap / metallic (cbm < vbm via HOMO/LUMO line) -> gap_eV=None, gap_type='metallic', band_overlap_eV=signed (cbm-vbm < 0)
  3. Zero gap insulator (cbm == vbm) -> gap_eV=0.0, gap_type='insulator', band_overlap_eV=None
  4. Metallic via Fermi crossing -> gap_eV=None, gap_type='metallic', band_overlap_eV=None
  5. Insufficient bands (no conduction states) -> gap_eV=None, gap_type='insufficient_bands'
  6. Synthetic open-shell d2 parser test -> gap_eV=None, gap_type='metallic', band_overlap_eV <= 0.0
  7. DeltaMLGapCorrector handles pbe_gap_ev=None safely without float(None) coercion
  8. Heavy-element SOC domain flag triggers for non-heavy halides (KZrCl3) but not Pb/Sn perovskites
"""

import sys
import os
import time
import tempfile
from pathlib import Path

import pytest
import numpy as np
from fastapi import HTTPException
from pymatgen.core import Structure, Lattice
from app.services.dft_validation import DFTValidationService
from app.services.simulation_service import generate_crystal_prototype
from app.services.delta_ml_corrector import get_delta_ml_corrector


@pytest.fixture(scope="module")
def dft_svc():
    return DFTValidationService()


def _write_synthetic_pwo(tmpdir: str, content: str) -> str:
    """Write synthetic QE output to a .pwo file."""
    pwo_path = Path(tmpdir) / "espresso.pwo"
    pwo_path.write_text(content, encoding="utf-8")
    return tmpdir


def _simple_structure():
    """Simple 2-atom NaCl structure for parser tests."""
    return Structure.from_spacegroup(
        "Fm-3m", Lattice.cubic(5.64), ["Na", "Cl"],
        [[0, 0, 0], [0.5, 0.5, 0.5]]
    )


class TestBandGapParserMetallicity:
    """Verify signed-overlap metallicity detection in _extract_band_gap_from_nscf."""

    def test_clean_insulating_gap(self, dft_svc):
        """HOMO/LUMO path: cbm > vbm -> insulator with positive gap."""
        content = """\
     Program PWSCF v.7.5 starts on 29Sep2026

     highest occupied, lowest unoccupied level (ev):     4.5000    6.8000

     JOB DONE.
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_synthetic_pwo(tmpdir, content)
            result = dft_svc._extract_band_gap_from_nscf(tmpdir, _simple_structure())

        assert result["gap_eV"] == 2.3, f"Expected 2.3 eV, got {result['gap_eV']}"
        assert result["gap_type"] == "insulator"
        assert result["vbm_eV"] == 4.5
        assert result["cbm_eV"] == 6.8
        assert result["band_overlap_eV"] is None

    def test_band_overlap_metallic(self, dft_svc):
        """HOMO/LUMO path: cbm < vbm (signed overlap) -> gap_eV=None, gap_type='metallic', band_overlap_eV=-2.1."""
        content = """\
     Program PWSCF v.7.5 starts on 29Sep2026

     highest occupied, lowest unoccupied level (ev):     7.2000    5.1000

     JOB DONE.
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_synthetic_pwo(tmpdir, content)
            result = dft_svc._extract_band_gap_from_nscf(tmpdir, _simple_structure())

        assert result["gap_eV"] is None, f"Expected gap_eV=None, got {result['gap_eV']}"
        assert result["gap_type"] == "metallic", f"Expected metallic, got {result['gap_type']}"
        assert result["vbm_eV"] == 7.2
        assert result["cbm_eV"] == 5.1
        assert result["band_overlap_eV"] == -2.1, f"Expected -2.1 eV overlap, got {result['band_overlap_eV']}"

    def test_zero_gap_insulator(self, dft_svc):
        """HOMO/LUMO path: cbm == vbm -> gap=0.0 (semimetal), type insulator."""
        content = """\
     Program PWSCF v.7.5 starts on 29Sep2026

     highest occupied, lowest unoccupied level (ev):     3.5000    3.5000

     JOB DONE.
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_synthetic_pwo(tmpdir, content)
            result = dft_svc._extract_band_gap_from_nscf(tmpdir, _simple_structure())

        assert result["gap_eV"] == 0.0
        assert result["gap_type"] == "insulator"
        assert result["band_overlap_eV"] is None

    def test_fermi_crossing_metallic(self, dft_svc):
        """Band-parsing path: band crosses Fermi level -> gap_eV=None, gap_type='metallic', band_overlap_eV=None."""
        content = """\
     Program PWSCF v.7.5 starts on 29Sep2026

          k = 0.0000 0.0000 0.0000
     bands (ev):

    1.0000   4.5000   7.0000   9.0000

          k = 0.5000 0.5000 0.5000
     bands (ev):

    1.2000   5.5000   7.2000   9.5000

     the Fermi energy is     5.0000 ev

     JOB DONE.
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_synthetic_pwo(tmpdir, content)
            result = dft_svc._extract_band_gap_from_nscf(tmpdir, _simple_structure())

        assert result["gap_eV"] is None, f"Expected None, got {result['gap_eV']}"
        assert result["gap_type"] == "metallic"
        assert result["vbm_eV"] is None
        assert result["cbm_eV"] is None
        assert result["band_overlap_eV"] is None

    def test_insufficient_bands(self, dft_svc):
        """When all bands are occupied, return insufficient_bands, never metallic."""
        content = """\
     Program PWSCF v.7.5 starts on 29Sep2026

          k = 0.0000 0.0000 0.0000
     bands (ev):

    1.0000   3.5000

          k = 0.5000 0.5000 0.5000
     bands (ev):

    1.2000   3.8000

     the Fermi energy is     4.0000 ev

     JOB DONE.
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_synthetic_pwo(tmpdir, content)
            result = dft_svc._extract_band_gap_from_nscf(tmpdir, _simple_structure())

        assert result["gap_eV"] is None, f"Expected None, got {result['gap_eV']}"
        assert result["gap_type"] == "insufficient_bands"


class TestSyntheticOpenShellAndDomainSafety:
    """Parser tests for synthetic open-shell configurations and Delta-ML None-input safety."""

    def test_prototype_symmetry_check(self):
        """Pm-3m prototype symmetry check returns is_high_symmetry_cubic=True."""
        svc = DFTValidationService()
        struct = generate_crystal_prototype("KZrCl3")
        res = svc.check_kpoint_symmetry_scope(struct, formula="KZrCl3")
        assert res["is_high_symmetry_cubic"] is True
        assert res["spacegroup_number"] == 221

    def test_synthetic_open_shell_d2_parser_metallicity(self, dft_svc):
        """
        Parser test using synthetic eigenvalues representing an open-shell d2 state with crossing bands.
        Note: This tests parser handling of synthetic output, not physical KZrCl3 DFT convergence.
        """
        struct = generate_crystal_prototype("KZrCl3")
        content = """\
     Program PWSCF v.7.5 starts on 29Sep2026

          k = 0.0000 0.0000 0.0000 (    8 PWs)   bands (ev):

    -12.3400  -4.5200  -4.5200  -4.1100   4.8500   5.6200   7.8900   8.1200

          k = 0.5000 0.5000 0.5000 (    8 PWs)   bands (ev):

    -12.1800  -4.7800  -4.7800  -3.9500   5.4500   5.8500   7.6500   8.3000

     the Fermi energy is     5.2000 ev

     JOB DONE.
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_synthetic_pwo(tmpdir, content)
            result = dft_svc._extract_band_gap_from_nscf(tmpdir, struct)

        assert result["gap_type"] == "metallic", f"Expected metallic, got {result['gap_type']}"
        assert result["gap_eV"] is None, f"Expected gap_eV=None for metal, got {result['gap_eV']}"

    def test_deltaml_pbe_gap_none_safety(self):
        """
        Verify DeltaMLGapCorrector given pbe_gap_ev=None returns out-of-domain metadata
        safely without crashing on float(None) or silently coercing to a positive gap.
        """
        corrector = get_delta_ml_corrector()
        
        # 1. Non-heavy candidate with None gap and metallic gap_type
        pred_none = corrector.predict_corrected_gap(
            pbe_gap_ev=None,
            formula="KZrCl3",
            pbe_gap_type="metallic",
        )
        assert pred_none["status"] == "pbe_metallic_hse_undetermined"
        assert pred_none["pbe_gap_eV"] is None
        assert pred_none["corrected_gap_eV"] is None
        assert pred_none["method"] == "out_of_domain_metallic"
        assert pred_none["should_escalate_to_r2scan"] is True
        assert pred_none["soc_offset_in_delta_not_applicable"] is True
        assert pred_none["requires_metallicity_check"] is True

        # 2. Non-heavy candidate with None gap and insufficient_bands gap_type
        pred_insuff = corrector.predict_corrected_gap(
            pbe_gap_ev=None,
            formula="KZrCl3",
            pbe_gap_type="insufficient_bands",
        )
        assert pred_insuff["status"] == "pbe_insufficient_bands_hse_undetermined"

        # 3. Heavy-element perovskites (Pb, Sn)
        pred_pb = corrector.predict_corrected_gap(
            pbe_gap_ev=1.323,
            formula="CsPbI3",
            features={"eps_inf": 5.80},
        )
        assert pred_pb["corrected_gap_eV"] is not None
        assert pred_pb["soc_offset_in_delta_not_applicable"] is False
        assert pred_pb["requires_metallicity_check"] is False

        pred_sn = corrector.predict_corrected_gap(
            pbe_gap_ev=0.327,
            formula="CsSnI3",
            features={"eps_inf": 6.20},
        )
        assert pred_sn["status"] == "out_of_domain"
        assert pred_sn.get("out_of_domain") is True
        assert pred_sn["corrected_gap_eV"] is None
        assert pred_sn["reason"] == "Sn/SOC regime, 1 calibration point"

    def test_kzrcl3_pending_metallicity_check(self):
        """KZrCl3 with positive PBE gap returns corrected_gap_eV=None, status=pending_metallicity_check, and no interval."""
        corrector = get_delta_ml_corrector()
        res = corrector.predict_corrected_gap(pbe_gap_ev=1.85, formula="KZrCl3", features={"eps_inf": 5.0})
        assert res["corrected_gap_eV"] is None
        assert res["status"] == "pending_metallicity_check"
        assert res["interval_pooled"] is None
        assert res["interval_chemistry_specific"] is None
        assert res["q_hat"] is None
        assert res["requires_metallicity_check"] is True
        assert res["soc_offset_in_delta_not_applicable"] is True
        assert res["provisional"] is True
        assert "features_source" in res
        assert res["features_source"]["Z_avg"] == "composition_derived_mean_atomic_number"
        assert res["features_source"]["chi_diff"] == "composition_derived_pauling_electronegativity_diff"
        assert res["features_source"]["r_ratio"] == "composition_derived_atomic_radius_ratio"
        assert res["features_source"]["eps_inf"] == "user_provided"

    def test_delta_nonpositive_out_of_domain(self):
        """When predicted delta is nonpositive (<= 0), in-family corrector returns out_of_domain_delta_nonpositive and None gap."""
        corrector = get_delta_ml_corrector()
        orig_deltas = getattr(corrector, "anion_mean_deltas", {}).copy()
        try:
            corrector.anion_mean_deltas["I"] = -0.05
            res = corrector.predict_corrected_gap(
                pbe_gap_ev=2.0,
                formula="CsPbI3",
                features={"chi_diff": 0.1, "r_ratio": 1.5, "Z_avg": 90.0, "eps_inf": 5.0}
            )
            assert res["predicted_delta_eV"] <= 0.0
            assert res["status"] == "out_of_domain_delta_nonpositive"
            assert res["delta_nonpositive"] is True
            assert res["corrected_gap_eV"] is None
            assert res["interval_pooled"] is None
            assert res["interval_chemistry_specific"] is None
            assert res["provisional"] is True
        finally:
            corrector.anion_mean_deltas = orig_deltas

    def test_features_source_provenance_and_provisional_flag(self):
        """All predictions include features_source and provisional: True flag, and non-perovskite is out_of_domain."""
        corrector = get_delta_ml_corrector()
        
        # 1. Non-perovskite (NaCl) returns out_of_domain
        res_missing = corrector.predict_corrected_gap(pbe_gap_ev=5.106, formula="NaCl")
        assert res_missing["status"] == "out_of_domain"
        assert res_missing["reason"] == "not a Pb ABX3 halide perovskite"
        assert res_missing["provisional"] is True
        assert res_missing["corrected_gap_eV"] is None

        # 2. Provided eps_inf returns calibrated prediction with full provenance
        # In-family (halide perovskite) deploys anion_matched_mean_delta (with ridge alternative)
        res_in = corrector.predict_corrected_gap(pbe_gap_ev=1.532, formula="CsPbBr3", features={"eps_inf": 5.30})
        assert res_in["provisional"] is True
        assert res_in["status"] == "anion_matched_mean_delta"
        assert "features_source" in res_in
        assert isinstance(res_in["features_source"], dict)
        for key in ["chi_diff", "r_ratio", "Z_avg", "eps_inf"]:
            assert key in res_in["features_source"]

        # Out-of-family non-perovskite (NaCl) returns out_of_domain
        res_out = corrector.predict_corrected_gap(pbe_gap_ev=5.106, formula="NaCl", features={"eps_inf": 2.54})
        assert res_out["provisional"] is True
        assert res_out["status"] == "out_of_domain"
        assert res_out["reason"] == "not a Pb ABX3 halide perovskite"

    def test_conformal_half_width_strictly_increases_with_leverage(self):
        """Verify conformal half-width strictly increases with leverage h_test: W(h) = q_tilde * sqrt(1 + h)."""
        corrector = get_delta_ml_corrector()
        
        # Nominal in-family point (CsPbBr3)
        res_nom = corrector.predict_corrected_gap(
            pbe_gap_ev=1.532,
            formula="CsPbBr3",
            features={"eps_inf": 5.30}
        )
        h_nom = res_nom["leverage_hii"]
        w_nom = res_nom["half_width_eV"]

        # High leverage in-family point (CsPbCl3 at high PBE gap)
        res_high = corrector.predict_corrected_gap(
            pbe_gap_ev=3.50,
            formula="CsPbCl3",
            features={"eps_inf": 3.50}
        )
        h_high = res_high["leverage_hii"]
        w_high = res_high["half_width_eV"]

        assert h_high > h_nom, f"Expected h_high ({h_high}) > h_nom ({h_nom})"
        assert w_high > w_nom, f"Expected half_width to widen with leverage: w_high ({w_high}) > w_nom ({w_nom})"
        
        # Exact mathematical formula check on in-family point
        q_tilde = res_nom["q_tilde"]
        expected_w = round(float(q_tilde * np.sqrt(1.0 + h_nom)), 4)
        assert abs(w_nom - expected_w) <= 0.001, f"Expected {expected_w}, got {w_nom}"

    def test_organic_cation_featurizer_handling(self):
        """Verify organic cation pseudo-A site featurizer handles MA, FA with Sanderson geometric mean rule."""
        from app.services.delta_ml_corrector import _composition_features, ORGANIC_CATIONS

        # 1. MAPbI3 (chi(Pb)=2.33, chi(MA)=2.3334, chi(I)=2.66 -> max - min = 2.66 - 2.33 = 0.33)
        chi_d, r_rat, z_a = _composition_features("MAPbI3")
        assert chi_d == 0.33
        assert r_rat == 0.6452
        assert z_a == 52.0

        # 2. FAPbI3 (chi(Pb)=2.33, chi(FA)=2.4297, chi(I)=2.66 -> max - min = 2.66 - 2.33 = 0.33)
        chi_d_fa, r_rat_fa, z_a_fa = _composition_features("FAPbI3")
        assert chi_d_fa == 0.33
        assert r_rat_fa == 0.5534
        assert z_a_fa == 53.2

        # 3. Robustness on malformed or empty formulas (no crash)
        assert _composition_features("") == (0.0, 1.0, 0.0)
        assert _composition_features("FakeElement123") == (0.0, 1.0, 0.0)

    def test_calibration_regimes_routing(self):
        """Verify in-family routes to in_family_loocv, while non-perovskite is out_of_domain."""
        corrector = get_delta_ml_corrector()
        
        # Halide perovskite (n=10 >= 9) -> in_family_loocv
        res_pero = corrector.predict_corrected_gap(
            pbe_gap_ev=1.532,
            formula="CsPbBr3",
            features={"eps_inf": 5.30}
        )
        assert res_pero["calibration_regime"] == "in_family_loocv"
        assert res_pero["family_interval_applied"] is True

        # Alkaline earth oxide (MgO) is out_of_domain
        res_oxide = corrector.predict_corrected_gap(
            pbe_gap_ev=4.475,
            formula="MgO",
            features={"eps_inf": 3.00}
        )
        assert res_oxide["status"] == "out_of_domain"
        assert res_oxide["reason"] == "not a Pb ABX3 halide perovskite"

    def test_candidate_hardcoded_literals_eliminated(self):
        """Assert candidate dicts constructed via orchestrator path have vbm_vs_vacuum_eV, synthesizability_score, is_solar_optimal as None unless computed."""
        from pymatgen.core import Structure, Lattice
        from app.services.job_orchestrator import DiscoveryJobOrchestrator
        from app.services.pareto_ranker import ParetoRanker

        orchestrator = DiscoveryJobOrchestrator()

        # Fixture structure and candidate for orchestrator path
        lat = Lattice.cubic(6.29)
        test_struct = Structure(lat, ["Cs", "Pb", "I", "I", "I"], [
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5],
            [0.5, 0.5, 0.0],
            [0.5, 0.0, 0.5],
            [0.0, 0.5, 0.5]
        ])
        cand_fixture = {
            "formula": "CsPbI3",
            "structure": test_struct,
            "cif_content": test_struct.to(fmt="cif"),
            "generation_method": "substitution",
        }

        # Call orchestrator candidate processing path
        cand_record = orchestrator.build_candidate_record(
            idx=0,
            cand=cand_fixture,
            query={"scaffold": "perovskite", "target_ion": "Li", "exclude_toxic": False}
        )

        assert cand_record is not None, "Candidate record must be generated by orchestrator"
        assert cand_record["vbm_vs_vacuum_eV"] is None, "vbm_vs_vacuum_eV must be None unless computed"
        assert cand_record["synthesizability_score"] is None, "synthesizability_score must be None unless computed"
        assert cand_record["is_solar_optimal"] is None, "is_solar_optimal must be None unless computed"
        assert cand_record["predicted_band_gap_eV"] is None, "predicted_band_gap_eV must be None unless computed"

        # Assert no forbidden hardcoded constants exist across any field
        forbidden_literals = [-5.50, 0.85]
        for k, v in cand_record.items():
            for forbidden in forbidden_literals:
                assert v != forbidden, f"Forbidden literal {forbidden} found in candidate field '{k}'!"

        # Assert ParetoRanker correctly ignores None without defaulting to 0.0
        ranker = ParetoRanker()
        p1 = {"gnn_prediction": -0.5, "estimated_cost_usd_kg": None, "free_volume_A3": 50.0}
        p2 = {"gnn_prediction": -0.4, "estimated_cost_usd_kg": None, "free_volume_A3": 40.0}
        assert ranker._dominates(p1, p2) is True
        assert ranker._dominates(p2, p1) is False

    def test_sn_candidate_returns_out_of_domain(self):
        """Sn compounds (e.g. MASnI3, CsSnCl3) return out_of_domain with reason 'Sn/SOC regime, 1 calibration point'."""
        corrector = get_delta_ml_corrector()
        for sn_formula, pbe_gap in [("MASnI3", 0.3482), ("CsSnCl3", 0.7990), ("CsSnI3", 0.3270), ("CsSnBr3", 0.3920)]:
            res = corrector.predict_corrected_gap(
                pbe_gap_ev=pbe_gap,
                formula=sn_formula,
                features={"eps_inf": 5.5}
            )
            assert res["status"] == "out_of_domain", f"Expected status 'out_of_domain' for {sn_formula}, got {res.get('status')}"
            assert res.get("out_of_domain") is True, f"Expected out_of_domain=True for {sn_formula}"
            assert res["corrected_gap_eV"] is None, f"Expected corrected_gap_eV=None for {sn_formula}"
            assert res["reason"] == "Sn/SOC regime, 1 calibration point", f"Expected exact reason 'Sn/SOC regime, 1 calibration point' for {sn_formula}, got {res.get('reason')}"
            assert res["interval_lower"] is None
            assert res["interval_upper"] is None

    def test_stoichiometry_and_mixed_a_site_gate(self):
        """Require ABX3 stoichiometry and pure single A-site in {Cs, MA, FA}; reject non-ABX3 and mixed A-site."""
        corrector = get_delta_ml_corrector()

        # Out-of-domain: non-ABX3 stoichiometry or mixed A-site
        out_of_domain_cases = [
            ("Cs4PbBr6", "non-ABX3 stoichiometry"),
            ("Cs2PbI4", "non-ABX3 stoichiometry"),
            ("CsPb2Br5", "non-ABX3 stoichiometry"),
            ("Cs0.5FA0.5PbI3", "mixed A-site"),
            ("MA0.5FA0.5PbI3", "mixed A-site"),
            ("CH6NPbI3.5", "non-ABX3 stoichiometry"),
            ("(CH6N)0.5CsPbI3", "mixed A-site"),
        ]
        for form, expected_reason_substr in out_of_domain_cases:
            res = corrector.predict_corrected_gap(pbe_gap_ev=1.50, formula=form, features={"eps_inf": 5.0})
            assert res["status"] == "out_of_domain", f"{form}: expected status 'out_of_domain', got {res.get('status')}"
            assert res.get("out_of_domain") is True, f"{form}: expected out_of_domain=True"
            assert res["corrected_gap_eV"] is None, f"{form}: expected corrected_gap_eV=None"
            assert expected_reason_substr in res.get("reason", ""), f"{form}: expected reason containing '{expected_reason_substr}', got '{res.get('reason')}'"

        # In-domain: pure ABX3 lead halide perovskites
        in_domain_cases = [
            "CsPbI3", "CsPbBr3", "CsPbCl3", "MAPbI3", "MAPbBr3", "MAPbCl3", "FAPbI3"
        ]
        for form in in_domain_cases:
            res = corrector.predict_corrected_gap(pbe_gap_ev=1.50, formula=form, features={"eps_inf": 5.0})
            assert res["status"] == "anion_matched_mean_delta", f"{form}: expected in-domain status 'anion_matched_mean_delta', got {res.get('status')}"
            assert res.get("out_of_domain") is not True, f"{form}: expected in-domain"
            assert res["corrected_gap_eV"] is not None, f"{form}: expected numerical corrected_gap_eV"

    @pytest.mark.parametrize(
        ("formula", "expected_status"),
        [
            ("Cs2PbI4", "out_of_domain"),
            ("C2H12N2PbI3", "out_of_domain"),
            ("Cs0.5FA0.5PbI3", "out_of_domain"),
        ],
    )
    def test_pb_x_count_gate_cases(self, formula, expected_status):
        """Validate Pb/X counts, supercell reduction, and mixed-A-site rejection."""
        corrector = get_delta_ml_corrector()
        result = corrector.predict_corrected_gap(
            pbe_gap_ev=1.50,
            formula=formula,
            features={"eps_inf": 5.0},
        )

        assert result["status"] == expected_status
        if expected_status == "out_of_domain":
            assert result["out_of_domain"] is True
            assert result["corrected_gap_eV"] is None
        else:
            assert result.get("out_of_domain") is not True
            reference_formula = "CsPbI3" if formula == "Cs4Pb4I12" else "MAPbI3"
            reference = corrector.predict_corrected_gap(
                1.50, reference_formula, {"eps_inf": 5.0}
            )
            assert result["corrected_gap_eV"] == reference["corrected_gap_eV"]

    @pytest.mark.parametrize(
        ("formula", "reason"),
        [
            ("NaCl", "not a Pb ABX3 halide perovskite"),
            ("MgO", "not a Pb ABX3 halide perovskite"),
            ("SrTiO3", "not a Pb ABX3 halide perovskite"),
            ("BaTiO3", "not a Pb ABX3 halide perovskite"),
            ("Si", "not a Pb ABX3 halide perovskite"),
            ("LiCoO2", "not a Pb ABX3 halide perovskite"),
        ],
    )
    def test_known_non_domain_materials(self, formula, reason):
        """Known non-domain materials are rejected with an explicit reason."""
        result = get_delta_ml_corrector().predict_corrected_gap(
            pbe_gap_ev=2.0,
            formula=formula,
            features={"eps_inf": 5.0},
        )

        assert result["status"] == "out_of_domain"
        assert result["out_of_domain"] is True
        assert result["reason"] == reason
        assert result["corrected_gap_eV"] is None

    def test_cs4pb4i12_matches_cspbi3_gap(self):
        """Cs4Pb4I12 reduces to CsPbI3 and keeps its corrected gap."""
        corrector = get_delta_ml_corrector()
        reference = corrector.predict_corrected_gap(1.50, "CsPbI3", {"eps_inf": 5.0})
        result = corrector.predict_corrected_gap(1.50, "Cs4Pb4I12", {"eps_inf": 5.0})

        assert result["status"] == "anion_matched_mean_delta"
        assert result["corrected_gap_eV"] == reference["corrected_gap_eV"]

    def test_cspbi3_pinned_corrected_gap(self):
        """Pin deployed I-site correction for CsPbI3 at PBE 1.50 eV."""
        # I-site delta = mean((1.73 - 1.323), (1.57 - 1.3787)) = 0.29915 eV.
        # 1.50 + 0.29915 = 1.79915, rounded by API to 1.7992 eV.
        result = get_delta_ml_corrector().predict_corrected_gap(
            1.50, "CsPbI3", {"eps_inf": 5.0}
        )

        assert result["status"] == "anion_matched_mean_delta"
        assert result["corrected_gap_eV"] == 1.7992

    def test_c4h24n4pb4i12_matches_mapbi3_gap(self):
        """C4H24N4Pb4I12 reduces to MAPbI3 and keeps its corrected gap."""
        corrector = get_delta_ml_corrector()
        reference = corrector.predict_corrected_gap(1.50, "MAPbI3", {"eps_inf": 5.0})
        result = corrector.predict_corrected_gap(1.50, "C4H24N4Pb4I12", {"eps_inf": 5.0})

        assert result["status"] == "anion_matched_mean_delta"
        assert result["corrected_gap_eV"] == reference["corrected_gap_eV"]

    def test_c2h12n2pbi3_reports_rejection_reason(self):
        """C2H12N2PbI3 has A:Pb != 1:1 and returns an explicit rejection reason."""
        result = get_delta_ml_corrector().predict_corrected_gap(1.50, "C2H12N2PbI3", {"eps_inf": 5.0})

        assert result["status"] == "out_of_domain"
        assert result["reason"] == "mixed A-site or non-ABX3 stoichiometry: A-site composition (C2H12N2) does not match MA (C1H6N1) or FA (C1H5N2)"
        assert result["corrected_gap_eV"] is None

    def test_delta_ml_api_rejects_nancl_without_pbe_gap(self, monkeypatch):
        """NaCl with no PBE gap returns HTTP 400 and persists no correction."""
        from app.api import dft

        candidate = type(
            "Candidate",
            (),
            {
                "dft_pbe_gap_eV": None,
                "formula": "NaCl",
                "dft_delta_ml_gap_eV": None,
                "dft_delta_ml_interval_lower": None,
                "dft_delta_ml_interval_upper": None,
                "dft_delta_ml_q_hat": None,
            },
        )()

        class FakeQuery:
            def filter(self, _condition):
                return self

            def first(self):
                return candidate

        class FakeDb:
            committed = False

            def query(self, _model):
                return FakeQuery()

            def commit(self):
                self.committed = True

            def close(self):
                pass

        db = FakeDb()
        monkeypatch.setattr(dft, "get_db", lambda: iter([db]))

        with pytest.raises(HTTPException) as exc_info:
            dft.apply_delta_ml_correction(123, dft.DeltaMLRequest())

        assert exc_info.value.status_code == 400
        assert exc_info.value.detail == "No PBE gap available for this candidate. Run /api/dft/queue/{id} first."
        assert candidate.dft_delta_ml_gap_eV is None
        assert db.committed is False

    def test_delta_ml_api_persists_pb_abx3_correction(self, monkeypatch):
        """CsPbI3 with a valid PBE gap uses in-domain correction and persists result."""
        from app.api import dft

        candidate = type(
            "Candidate",
            (),
            {
                "dft_pbe_gap_eV": 1.323,
                "formula": "CsPbI3",
                "dft_delta_ml_gap_eV": None,
                "dft_delta_ml_interval_lower": None,
                "dft_delta_ml_interval_upper": None,
                "dft_delta_ml_q_hat": None,
                "dft_delta_ml_training_provenance": None,
                "dft_dielectric_const_pbe": None,
            },
        )()

        class FakeQuery:
            def filter(self, _condition):
                return self

            def first(self):
                return candidate

        class FakeDb:
            committed = False

            def query(self, _model):
                return FakeQuery()

            def commit(self):
                self.committed = True

            def close(self):
                pass

        db = FakeDb()
        monkeypatch.setattr(dft, "get_db", lambda: iter([db]))

        result = dft.apply_delta_ml_correction(123, dft.DeltaMLRequest(eps_inf=5.8))

        assert result["status"] == "anion_matched_mean_delta"
        assert result["corrected_gap_eV"] is not None
        assert result["q_tilde"] == pytest.approx(0.2157, abs=1e-4)
        assert result["coverage_level"] == 0.80
        assert candidate.dft_delta_ml_gap_eV == result["corrected_gap_eV"]
        assert candidate.dft_delta_ml_interval_lower == result["interval_lower"]
        assert candidate.dft_delta_ml_interval_upper == result["interval_upper"]
        assert candidate.dft_delta_ml_q_hat == result["q_tilde"]
        assert db.committed is True

    def test_delta_ml_api_rejects_nacl_gap_without_scissor_fallback(self, monkeypatch):
        """NaCl with PBE gap returns out_of_domain and persists no correction."""
        from app.api import dft

        candidate = type(
            "Candidate",
            (),
            {
                "dft_pbe_gap_eV": 5.12,
                "formula": "NaCl",
                "dft_delta_ml_gap_eV": None,
                "dft_delta_ml_interval_lower": None,
                "dft_delta_ml_interval_upper": None,
                "dft_delta_ml_q_hat": None,
                "dft_delta_ml_training_provenance": None,
                "dft_dielectric_const_pbe": None,
            },
        )()

        class FakeQuery:
            def filter(self, _condition):
                return self

            def first(self):
                return candidate

        class FakeDb:
            committed = False

            def query(self, _model):
                return FakeQuery()

            def commit(self):
                self.committed = True

            def close(self):
                pass

        db = FakeDb()
        monkeypatch.setattr(dft, "get_db", lambda: iter([db]))

        result = dft.apply_delta_ml_correction(123, dft.DeltaMLRequest(eps_inf=2.54))

        assert result["status"] == "out_of_domain"
        assert result["method"] == "out_of_domain"
        assert result["corrected_gap_eV"] is None
        assert result["interval_lower"] is None
        assert result["interval_upper"] is None
        assert result["reason"] == "not a Pb ABX3 halide perovskite"
        assert candidate.dft_delta_ml_gap_eV is None
        assert candidate.dft_delta_ml_interval_lower is None
        assert candidate.dft_delta_ml_interval_upper is None
        assert candidate.dft_delta_ml_q_hat is None
        assert db.committed is True

    def test_a_site_composition_equivalence_to_short_names(self):
        """In-domain compositional forms work and give identical corrected gap to short names."""
        corrector = get_delta_ml_corrector()
        pbe_gap = 1.30
        features = {"eps_inf": 5.0}

        # MA variants (C1 H6 N1)
        res_mapbi3 = corrector.predict_corrected_gap(pbe_gap, "MAPbI3", features)
        res_ch6n = corrector.predict_corrected_gap(pbe_gap, "CH6NPbI3", features)
        res_ch3nh3 = corrector.predict_corrected_gap(pbe_gap, "CH3NH3PbI3", features)
        assert res_ch6n["status"] == "anion_matched_mean_delta"
        assert res_ch3nh3["status"] == "anion_matched_mean_delta"
        assert res_ch6n["corrected_gap_eV"] == res_mapbi3["corrected_gap_eV"]
        assert res_ch3nh3["corrected_gap_eV"] == res_mapbi3["corrected_gap_eV"]

        # FA variants (C1 H5 N2)
        res_fapbi3 = corrector.predict_corrected_gap(pbe_gap, "FAPbI3", features)
        res_ch5n2 = corrector.predict_corrected_gap(pbe_gap, "CH5N2PbI3", features)
        assert res_ch5n2["status"] == "anion_matched_mean_delta"
        assert res_ch5n2["corrected_gap_eV"] == res_fapbi3["corrected_gap_eV"]

        # Cs variants (Cs1)
        res_cspbi3 = corrector.predict_corrected_gap(pbe_gap, "CsPbI3", features)
        res_csi3pb = corrector.predict_corrected_gap(pbe_gap, "CsI3Pb", features)
        assert res_csi3pb["status"] == "anion_matched_mean_delta"
        assert res_csi3pb["corrected_gap_eV"] == res_cspbi3["corrected_gap_eV"]

        # Supercell formulas reduce to in-domain ABX3 and match CsPbI3/MAPbI3
        res_cs4 = corrector.predict_corrected_gap(pbe_gap, "Cs4Pb4I12", features)
        assert res_cs4["status"] == "anion_matched_mean_delta"
        assert res_cs4["corrected_gap_eV"] == res_cspbi3["corrected_gap_eV"]

        res_c4h24 = corrector.predict_corrected_gap(pbe_gap, "C4H24N4Pb4I12", features)
        assert res_c4h24["status"] == "anion_matched_mean_delta"
        assert res_c4h24["corrected_gap_eV"] == res_mapbi3["corrected_gap_eV"]

    def test_adversarial_formula_length_cap_and_timing(self):
        """Formulas >40 chars are rejected out_of_domain immediately; 10,000 char string returns in <0.1 s."""
        corrector = get_delta_ml_corrector()
        adv_string = "Cs" * 5000
        t0 = time.perf_counter()
        res_adv = corrector.predict_corrected_gap(1.5, adv_string, {"eps_inf": 5.0})
        elapsed = time.perf_counter() - t0

        assert elapsed < 0.1, f"Expected timing < 0.1s, took {elapsed:.6f}s"
        assert res_adv["status"] == "out_of_domain"
        assert res_adv.get("out_of_domain") is True
        assert res_adv["corrected_gap_eV"] is None
        assert res_adv.get("reason") == "Formula length exceeds 40 characters"

