from __future__ import annotations

import torch

from crystal_gnn.models.ms_gnn import MultiScaleGNN


def test_full_forward_pass(batch_triple):
    b1, b2, b3, _, _ = batch_triple
    model = MultiScaleGNN(hidden_dim=32, num_encoder_layers=2)
    mu, v, alpha, beta = model(b1, b2, b3)
    assert mu.shape[1] == 1
    assert torch.all(v > 0) and torch.all(alpha > 1) and torch.all(beta > 0)


def test_single_scale_ablation(batch_triple):
    b1, _, _, _, _ = batch_triple
    model = MultiScaleGNN(hidden_dim=32, num_encoder_layers=2, radii=[4.0])
    out = model(b1, None, None)
    assert out is not None


def test_predict_with_uncertainty(batch_triple):
    b1, b2, b3, _, _ = batch_triple
    model = MultiScaleGNN(hidden_dim=32, num_encoder_layers=2)
    out = model.predict_with_uncertainty(b1, b2, b3, T=10)
    assert set(["prediction", "epistemic", "aleatoric", "all_passes", "mc_epistemic", "der_epistemic"]).issubset(out.keys())
    assert torch.all(out["epistemic"] >= 0)
    assert torch.all(out["aleatoric"] >= 0)
    assert torch.allclose(out["epistemic"], out["mc_epistemic"] + out["der_epistemic"], atol=1e-5)


def test_mc_variance_positive(batch_triple):
    b1, b2, b3, _, _ = batch_triple
    model = MultiScaleGNN(hidden_dim=32, num_encoder_layers=2)
    out = model.predict_with_uncertainty(b1, b2, b3, T=10)
    assert torch.all(out["mc_epistemic"] >= 0)


def test_attention_fusion_vs_concat(batch_triple):
    b1, b2, b3, _, _ = batch_triple
    m1 = MultiScaleGNN(hidden_dim=32, num_encoder_layers=2, use_attention_fusion=True)
    m2 = MultiScaleGNN(hidden_dim=32, num_encoder_layers=2, use_attention_fusion=False)
    o1 = m1(b1, b2, b3)[0]
    o2 = m2(b1, b2, b3)[0]
    assert o1.shape == o2.shape
    assert not torch.allclose(o1, o2)


def test_checkpoint_save_load(batch_triple, tmp_path):
    b1, b2, b3, _, _ = batch_triple
    m1 = MultiScaleGNN(hidden_dim=32, num_encoder_layers=2)
    p = tmp_path / "m.pt"
    torch.save(m1.state_dict(), p)
    m2 = MultiScaleGNN(hidden_dim=32, num_encoder_layers=2)
    m2.load_state_dict(torch.load(p, map_location="cpu"))
    y1 = m1(b1, b2, b3)[0]
    y2 = m2(b1, b2, b3)[0]
    assert y1.shape == y2.shape


def test_no_der_ablation(batch_triple):
    b1, b2, b3, _, _ = batch_triple
    m = MultiScaleGNN(hidden_dim=32, num_encoder_layers=2, use_der=False)
    out = m(b1, b2, b3)
    assert out.shape[1] == 1
