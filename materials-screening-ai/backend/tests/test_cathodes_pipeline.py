# -*- coding: utf-8 -*-
"""
Regression Test Fixture for Battery Cathode Discovery & Physics Validation.

Verifies:
  1. Charge-neutrality gate (pymatgen oxi_state_guesses) correctly passes real cathodes (LiCoO2, LiFePO4)
     and rejects unphysical formulas (CaTiF2, CaMnF2, MgMoF2).
  2. Mobile ion preservation (Li/Na) in scaffold substitution.
  3. Tier 2 MLIP structure relaxation trajectory and energy convergence.
"""

import pytest
from pymatgen.core import Structure, Composition

from app.services.generation_engine import StructureGenerationEngine, passes_charge_neutrality
from app.services.physics_validation import PhysicsValidationLayer
from app.services.constraint_parser import ConstraintParser


def test_known_cathodes_pass_charge_neutrality():
    """Verify real known cathodes pass charge neutrality gate."""
    c_licoo2 = Composition("LiCoO2")
    c_lifepo4 = Composition("LiFePO4")

    assert len(c_licoo2.oxi_state_guesses()) > 0, "LiCoO2 must have valid oxidation state guesses."
    assert len(c_lifepo4.oxi_state_guesses()) > 0, "LiFePO4 must have valid oxidation state guesses."


def test_unphysical_formulas_rejected_by_charge_gate():
    """Verify unphysical charge-imbalanced formulas are rejected."""
    c_catif2 = Composition("CaTiF2")
    c_camnf2 = Composition("CaMnF2")
    c_mgmof2 = Composition("MgMoF2")

    assert len(c_catif2.oxi_state_guesses()) == 0, "CaTiF2 must be rejected by oxidation state gate."
    assert len(c_camnf2.oxi_state_guesses()) == 0, "CaMnF2 must be rejected by oxidation state gate."
    assert len(c_mgmof2.oxi_state_guesses()) == 0, "MgMoF2 must be rejected by oxidation state gate."


def test_generator_preserves_mobile_ion_and_charge_neutrality():
    """Verify generated layered oxide candidates preserve target ion Li and pass charge neutrality."""
    engine = StructureGenerationEngine(seed=42)
    candidates = engine.generate_candidates(
        scaffold_type="layered_oxide",
        target_ion="Li",
        num_candidates=6
    )

    assert len(candidates) > 0, "Generator must produce valid candidates."
    for cand in candidates:
        struct = cand["structure"]
        assert "Li" in [el.symbol for el in struct.composition.elements], f"Candidate {cand['formula']} must contain Li."
        assert "O" in [el.symbol for el in struct.composition.elements], f"Candidate {cand['formula']} must contain O."
        assert passes_charge_neutrality(struct), f"Candidate {cand['formula']} must pass charge neutrality."


def test_physics_validation_trajectory():
    """Verify Tier 2 MLIP structure relaxation records step-by-step energy trajectory."""
    engine = StructureGenerationEngine(seed=42)
    cands = engine.generate_candidates(scaffold_type="layered_oxide", target_ion="Li", num_candidates=1)
    assert len(cands) > 0

    validator = PhysicsValidationLayer()
    val_res = validator.validate_candidate(
        structure=cands[0]["structure"],
        predicted_gnn_energy=-2.5
    )

    assert "mlip_relaxed_energy_eV" in val_res
    assert "mlip_stability_flag" in val_res
    assert "trajectory" in val_res
    assert len(val_res["trajectory"]) > 0, "Trajectory must contain relaxation steps."
    assert val_res["confidence_tier"] == "Tier 2 (Physics Validated)"
