# -*- coding: utf-8 -*-
"""Final-round behaviour tests: sigma-unit Tier-2 gate + shift-mode flags."""

import pytest

from app.services.multi_fidelity_orchestrator import (
    TIER2_SIGMA_GATE_DEFAULT,
    TIER2_SIGMA_GATE_TARGETS,
    MultiFidelityOrchestrator,
)

from pathlib import Path

_SHIFT_AWARE_ARTIFACTS = (
    Path(__file__).resolve().parents[3]
    / "crystal_gnn" / "data" / "cache" / "shift_aware_q_artifacts.npz"
)


class TestTier2SigmaGate:
    def test_default_is_50pct_target(self):
        assert TIER2_SIGMA_GATE_DEFAULT == TIER2_SIGMA_GATE_TARGETS["50pct_promote"]
        assert TIER2_SIGMA_GATE_DEFAULT == pytest.approx(0.09997, abs=1e-5)
        assert MultiFidelityOrchestrator().sigma_threshold == TIER2_SIGMA_GATE_DEFAULT

    def test_targets_are_ordered_and_q_invariant(self):
        # higher promote target requires a LOWER sigma gate
        t = TIER2_SIGMA_GATE_TARGETS
        assert t["70pct_promote"] < t["50pct_promote"] < t["30pct_promote"]
        # sigma units: the gate does not reference any q
        org = MultiFidelityOrchestrator(sigma_threshold=t["50pct_promote"])
        # width 0.10 eV -> sigma = 0.10 / (2*1.0254) = 0.0488 eV < gate -> hold
        assert org.decide_next_action(
            0.5, -0.05, 0.05, True, target_threshold=-0.20
        ) == "hold_for_more_data"
        # width 0.50 eV -> sigma = 0.2438 eV >= gate -> promote
        assert org.decide_next_action(
            0.5, -0.25, 0.25, True, target_threshold=-0.20
        ) == "promote_to_tier2"

    def test_hard_filter_and_stable_candidates_unaffected(self):
        org = MultiFidelityOrchestrator()
        assert org.decide_next_action(
            0.5, -0.25, 0.25, False, target_threshold=-0.20
        ) == "reject"
        # thermodynamically stable candidate promotes regardless of sigma
        assert org.decide_next_action(
            -0.30, -0.05, 0.05, True, target_threshold=-0.20
        ) == "promote_to_tier2"


class TestShiftModeFlags:
    def test_shift_robust_constant_and_default(self):
        from app.services.predictor import Q_SHIPPED_CONFORMAL, Q_SHIFT_ROBUST

        assert Q_SHIPPED_CONFORMAL == 1.0254
        assert Q_SHIFT_ROBUST == 1.7242
        assert Q_SHIFT_ROBUST > Q_SHIPPED_CONFORMAL  # shift-robust intervals are wider

    def test_shift_aware_falls_back_when_artifacts_missing(self, tmp_path, monkeypatch):
        import app.services.shift_aware_q as saq

        monkeypatch.setattr(saq, "ARTIFACTS", tmp_path / "missing.npz")
        monkeypatch.setattr(saq, "FINGERPRINTS", tmp_path / "missing2.npz")
        model = saq.ShiftAwareQModel()
        # artifacts missing -> compute() must return None so serving uses default q
        assert model.compute(None) is None

    @pytest.mark.skipif(
        not _SHIFT_AWARE_ARTIFACTS.exists(),
        reason="shift-aware build artifacts not present (rebuild via loco_distance_q.py)",
    )
    def test_shift_aware_end_to_end_on_synthetic_structure(self):
        from pymatgen.core import Lattice, Structure

        from app.services.shift_aware_q import compute_shift_aware_q

        s = Structure(Lattice.cubic(5.8), ["Pb", "Cl", "Cl"],
                      [[0.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.5, 0.0, 0.5]])
        out = compute_shift_aware_q(s)
        assert out is not None, "artifacts present -> compute must succeed"
        assert 0.0 <= out["distance"] <= 2.0
        assert 0 <= out["bin"] <= 9
        assert 1.0 < out["q"] < 3.0  # deployment bin q values live in ~[1.28, 2.45]
