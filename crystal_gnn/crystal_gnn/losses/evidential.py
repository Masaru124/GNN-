"""Evidential regression losses."""

from __future__ import annotations

import logging
import math

import torch
import torch.nn.functional as F

LOGGER = logging.getLogger(__name__)


def evidential_loss(
    mu: torch.Tensor,
    v: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    y: torch.Tensor,
    lam: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return total evidential loss and its NLL/regularization components."""
    if torch.isnan(y).any():
        raise ValueError(f"evidential_loss received NaN targets with shape {tuple(y.shape)}")

    if any(torch.isnan(t).any() for t in [mu, v, alpha, beta]):
        LOGGER.warning("NaN detected in model outputs; skipping batch by returning zero loss.")
        zero = torch.zeros((), device=y.device, dtype=y.dtype, requires_grad=True)
        return zero, zero.detach(), zero.detach()

    v = torch.clamp(v, min=1e-6, max=1e6)
    alpha = torch.clamp(alpha, min=1.01, max=1e6)
    beta = torch.clamp(beta, min=1e-6, max=1e6)

    two_beta_one_v = 2.0 * beta * (1.0 + v)
    sq_error_term = (y - mu).pow(2) * v + two_beta_one_v

    nll = (
        0.5 * torch.log(torch.tensor(math.pi, device=y.device, dtype=y.dtype) / v)
        - alpha * torch.log(two_beta_one_v)
        + (alpha + 0.5) * torch.log(torch.clamp(sq_error_term, min=1e-12))
        + torch.lgamma(alpha)
        - torch.lgamma(alpha + 0.5)
    )

    reg = (2.0 * v + alpha) * torch.abs(y - mu)
    total = (nll + lam * reg).mean()
    return total, nll.mean().detach(), reg.mean().detach()


def warm_up_loss(mu: torch.Tensor, y: torch.Tensor, epoch: int, warm_up_epochs: int = 20) -> torch.Tensor:
    """Return MSE during warm-up epochs."""
    del epoch, warm_up_epochs
    return F.mse_loss(mu, y)


def combined_loss(
    mu: torch.Tensor,
    v: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    y: torch.Tensor,
    epoch: int,
    lam: float = 0.1,
    warm_up_epochs: int = 20,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Switch from MSE warm-up to evidential loss after warm-up epochs."""
    if epoch < warm_up_epochs:
        mse = warm_up_loss(mu, y, epoch, warm_up_epochs)
        return mse, {"mode": "warmup_mse", "mse": float(mse.detach().cpu().item())}

    total, nll, reg = evidential_loss(mu, v, alpha, beta, y, lam=lam)
    return total, {
        "mode": "evidential",
        "nll": float(nll.cpu().item()),
        "reg": float(reg.cpu().item()),
    }
