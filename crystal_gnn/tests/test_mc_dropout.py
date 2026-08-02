from __future__ import annotations

import torch

from crystal_gnn.models.ms_gnn import MultiScaleGNN
from crystal_gnn.uncertainty.mc_dropout import MCDropout, mc_inference


def test_mcdropout_always_on():
    d = MCDropout(p=0.5)
    d.eval()
    x = torch.ones(100)
    y1 = d(x)
    y2 = d(x)
    assert not torch.allclose(y1, y2)


def test_mc_inference_output_keys(batch_triple):
    b1, b2, b3, _, _ = batch_triple
    model = MultiScaleGNN(hidden_dim=32, num_encoder_layers=2)
    out = mc_inference(model, b1, b2, b3, T=5, device="cpu")
    assert set(["mu", "mc_epistemic", "der_aleatoric", "der_epistemic", "total_epistemic", "total_uncertainty"]).issubset(out.keys())


def test_mc_inference_t_zero_raises(batch_triple):
    b1, b2, b3, _, _ = batch_triple
    model = MultiScaleGNN(hidden_dim=32, num_encoder_layers=2)
    import pytest

    with pytest.raises(ValueError):
        mc_inference(model, b1, b2, b3, T=0, device="cpu")
