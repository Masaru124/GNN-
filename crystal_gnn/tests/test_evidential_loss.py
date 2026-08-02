from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
import pytest

from crystal_gnn.losses.evidential import combined_loss, evidential_loss, warm_up_loss


class TinyDER(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(1, 4)

    def forward(self, x):
        out = self.fc(x)
        mu = out[:, :1]
        v = torch.nn.functional.softplus(out[:, 1:2]) + 1e-6
        alpha = torch.nn.functional.softplus(out[:, 2:3]) + 1.1
        beta = torch.nn.functional.softplus(out[:, 3:4]) + 1e-6
        return mu, v, alpha, beta


def test_loss_decreases_during_training():
    x = torch.linspace(-1, 1, 64).reshape(-1, 1)
    y = 2 * x
    model = TinyDER()
    opt = optim.Adam(model.parameters(), lr=0.05)
    losses = []
    for _ in range(50):
        mu, v, a, b = model(x)
        loss, _, _ = evidential_loss(mu, v, a, b, y, lam=0.01)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0]


def test_warm_up_loss_is_mse():
    mu = torch.tensor([[1.0], [2.0]])
    y = torch.tensor([[1.5], [2.5]])
    assert torch.allclose(warm_up_loss(mu, y, epoch=0), torch.nn.functional.mse_loss(mu, y))


def test_nan_in_targets_raises():
    with pytest.raises(ValueError):
        evidential_loss(torch.zeros(1, 1), torch.ones(1, 1), torch.ones(1, 1) + 1, torch.ones(1, 1), torch.tensor([[float("nan")]]))


def test_nan_in_output_returns_zero_with_warning():
    y = torch.ones(1, 1)
    loss, _, _ = evidential_loss(torch.tensor([[float("nan")]]), torch.ones(1, 1), torch.ones(1, 1) + 1, torch.ones(1, 1), y)
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-8)


def test_regularization_term_positive():
    mu = torch.zeros(4, 1)
    v = torch.ones(4, 1)
    a = torch.ones(4, 1) + 1.1
    b = torch.ones(4, 1)
    y = torch.ones(4, 1)
    _, _, reg = evidential_loss(mu, v, a, b, y)
    assert reg >= 0


def test_combined_loss_switches_at_warmup():
    mu, v, a, b, y = torch.zeros(4, 1), torch.ones(4, 1), torch.ones(4, 1) + 1.1, torch.ones(4, 1), torch.zeros(4, 1)
    l1, meta1 = combined_loss(mu, v, a, b, y, epoch=0, warm_up_epochs=20)
    l2, meta2 = combined_loss(mu, v, a, b, y, epoch=20, warm_up_epochs=20)
    assert meta1["mode"] == "warmup_mse"
    assert meta2["mode"] == "evidential"
    assert l1 >= 0 and l2 >= 0


def test_loss_gradients_nonzero():
    x = torch.randn(16, 1)
    y = 2 * x
    model = TinyDER()
    mu, v, a, b = model(x)
    loss, _, _ = evidential_loss(mu, v, a, b, y)
    loss.backward()
    grads = [p.grad for p in model.parameters()]
    assert all(g is not None for g in grads)
    assert any(torch.any(g != 0) for g in grads)
