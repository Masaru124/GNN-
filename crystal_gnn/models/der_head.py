"""Deep Evidential Regression output head."""

from __future__ import annotations

import torch
from torch import nn


class DERHead(nn.Module):
    """Predict NIG parameters for evidential regression."""

    def __init__(self, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 4),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (mu, v, alpha, beta) each shaped [B, 1]."""
        out = self.net(x)
        mu, v_raw, alpha_raw, beta_raw = torch.chunk(out, 4, dim=-1)

        v = torch.nn.functional.softplus(v_raw) + 1e-6
        alpha = torch.nn.functional.softplus(alpha_raw) + 1.0
        alpha = torch.clamp(alpha, min=1.01, max=1e6)
        beta = torch.nn.functional.softplus(beta_raw) + 1e-6
        beta = torch.clamp(beta, min=1e-6, max=1e6)

        for name, tensor in {"mu": mu, "v": v, "alpha": alpha, "beta": beta}.items():
            if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                raise RuntimeError(
                    f"NaN/Inf detected in DERHead output '{name}'. "
                    "This often indicates unstable evidential lambda early in training."
                )

        return mu, v, alpha, beta

    @staticmethod
    def aleatoric(v: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        """Return aleatoric uncertainty and clamp to finite range."""
        alpha = torch.clamp(alpha, min=1.01, max=1e6)
        v = torch.clamp(v, min=1e-6, max=1e6)
        beta = torch.clamp(beta, min=1e-6, max=1e6)
        out = beta / (v * (alpha - 1.0))
        return torch.clamp(out, min=0.0, max=1e6)

    @staticmethod
    def epistemic(v: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        """Return epistemic uncertainty and clamp to finite range."""
        ale = DERHead.aleatoric(v, alpha, beta)
        v = torch.clamp(v, min=1e-6, max=1e6)
        out = ale / v
        return torch.clamp(out, min=0.0, max=1e6)
