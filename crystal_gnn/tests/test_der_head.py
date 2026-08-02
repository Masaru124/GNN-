from __future__ import annotations

import torch

from crystal_gnn.models.der_head import DERHead


def test_shapes_and_positivity():
    h = DERHead(hidden_dim=32)
    mu, v, alpha, beta = h(torch.randn(4, 32))
    assert mu.shape == (4, 1)
    assert torch.all(v > 0)
    assert torch.all(alpha > 1)
    assert torch.all(beta > 0)


def test_uncertainties_positive():
    h = DERHead(hidden_dim=16)
    _, v, a, b = h(torch.randn(3, 16))
    assert torch.all(DERHead.aleatoric(v, a, b) >= 0)
    assert torch.all(DERHead.epistemic(v, a, b) >= 0)
