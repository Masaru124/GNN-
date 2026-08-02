from __future__ import annotations

import pytest
import torch

from crystal_gnn.models.fusion import ScaleFusionAttention


def test_output_shape():
    m = ScaleFusionAttention(hidden_dim=128, nhead=4)
    h1, h2, h3 = torch.randn(4, 128), torch.randn(4, 128), torch.randn(4, 128)
    out, _ = m(h1, h2, h3)
    assert out.shape == (4, 128)


def test_attention_weights_sum_to_one():
    m = ScaleFusionAttention(128, 4)
    _, att = m(torch.randn(4, 128), torch.randn(4, 128), torch.randn(4, 128))
    assert torch.allclose(att.sum(dim=1), torch.ones(4), atol=1e-5)


def test_attention_weights_vary_per_sample():
    m = ScaleFusionAttention(128, 4)
    _, att = m(torch.randn(4, 128), torch.randn(4, 128), torch.randn(4, 128))
    assert torch.var(att) > 1e-6


def test_attention_weights_stored():
    m = ScaleFusionAttention(128, 4)
    m(torch.randn(2, 128), torch.randn(2, 128), torch.randn(2, 128))
    assert m.last_attn_weights is not None
    assert m.last_attn_weights.shape == (2, 4, 1, 3)


def test_invalid_hidden_dim():
    with pytest.raises(ValueError):
        ScaleFusionAttention(hidden_dim=10, nhead=3)


def test_residual_connection():
    m = ScaleFusionAttention(128, 4)
    h1 = torch.randn(4, 128)
    out, _ = m(h1, torch.zeros_like(h1), torch.zeros_like(h1))
    assert torch.norm(out - h1) < torch.norm(out - torch.randn_like(h1))


def test_batch_size_one():
    m = ScaleFusionAttention(128, 4)
    out, _ = m(torch.randn(1, 128), torch.randn(1, 128), torch.randn(1, 128))
    assert out.shape == (1, 128)
