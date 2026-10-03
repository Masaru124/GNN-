"""Regression tests: GNNPredictorService.predict must be deterministic by default.

Hermetic: builds a tiny random-weight MultiScaleGNN, so no 77MB checkpoint is needed.
Guards the MCDropout gating added because MCDropout is always-on (ignores eval mode),
which previously made every predict() call sample a fresh dropout mask (±0.06 eV/atom).
"""
import inspect

import torch
from pymatgen.core import Lattice, Structure

from app.services.predictor import GNNPredictorService
from crystal_gnn.models.ms_gnn import MultiScaleGNN
from crystal_gnn.uncertainty.mc_dropout import MCDropout


def _make_structure() -> Structure:
    return Structure.from_spacegroup(
        "Pm-3m", Lattice.cubic(6.29), ["Cs", "Pb", "I"],
        [[0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0.5, 0]],
    )


def _make_service(deterministic: bool = True, seed=None, dropout_rate: float = 0.5) -> GNNPredictorService:
    torch.manual_seed(0)
    svc = object.__new__(GNNPredictorService)
    svc.device = "cpu"
    svc.deterministic = deterministic
    svc.seed = seed
    svc.q_hat_conformal = 0.4954
    svc.model = MultiScaleGNN(
        hidden_dim=16,
        num_encoder_layers=1,
        dropout_rate=dropout_rate,
        use_attention_fusion=True,
        use_der=True,
        radii=[4.0, 6.0, 8.0],
    ).to("cpu")
    svc.model.eval()
    svc._configure_mc_dropout()
    return svc


def test_deterministic_is_default():
    params = inspect.signature(GNNPredictorService.__init__).parameters
    assert params["deterministic"].default is True
    assert params["seed"].default is None


def test_mc_dropout_gated_off_when_deterministic():
    svc = _make_service(deterministic=True, dropout_rate=0.5)
    mods = [m for m in svc.model.modules() if isinstance(m, MCDropout)]
    assert mods, "tiny model should contain MCDropout modules"
    assert all(m.p == 0.0 for m in mods)
    # Opt back in restores the original rate.
    svc._set_mc_dropout(enabled=True)
    assert all(m.p == 0.5 for m in mods)


def test_two_identical_predict_calls_identical_output():
    """Deterministic default: two identical calls give byte-identical results."""
    svc = _make_service(deterministic=True, dropout_rate=0.5)
    struct = _make_structure()
    first = svc.predict(struct)
    second = svc.predict(struct)
    assert first == second
    # Dropout would break this within a handful of calls.
    for _ in range(5):
        assert svc.predict(struct) == first


def test_stochastic_mode_seed_reproducible():
    """deterministic=False restores MC-dropout sampling; seed pins it."""
    svc = _make_service(deterministic=False, dropout_rate=0.5)
    struct = _make_structure()
    a = svc.predict(struct, seed=42)
    b = svc.predict(struct, seed=42)
    c = svc.predict(struct, seed=43)
    assert a == b
    assert a["predicted_formation_energy_per_atom_eV"] != c["predicted_formation_energy_per_atom_eV"]


def test_mc_samples_reports_mean_and_std():
    svc = _make_service(deterministic=True, dropout_rate=0.5)
    struct = _make_structure()
    res = svc.predict(struct, seed=0, mc_samples=8)
    assert res["mc_samples"] == 8
    assert res["mc_std_eV"] > 0.0
    # Gating is restored afterwards: point prediction is deterministic again.
    assert all(m.p == 0.0 for m in svc._mc_dropout_modules)
    assert svc.predict(struct) == svc.predict(struct)
