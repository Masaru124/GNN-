"""MC Dropout utilities."""

from __future__ import annotations

import logging
import warnings

import torch
import torch.nn.functional as F

LOGGER = logging.getLogger(__name__)


class MCDropout(torch.nn.Dropout):
    """Dropout module that is always active regardless of model mode."""

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Apply dropout with training=True even under eval mode."""
        return F.dropout(input, self.p, training=True, inplace=self.inplace)


def mc_inference(
    model: torch.nn.Module,
    batch_r1,
    batch_r2,
    batch_r3,
    T: int = 30,
    device: str = "cuda",
    chunk_size: int = 10,
) -> dict[str, torch.Tensor]:
    """Run T stochastic forward passes and return uncertainty statistics."""
    if T <= 0:
        raise ValueError("T must be > 0 for MC inference.")

    model = model.to(device)
    model.train()

    has_mc_dropout = any(isinstance(m, MCDropout) for m in model.modules())
    if not has_mc_dropout:
        warnings.warn("Model has no MCDropout layers; MC epistemic may collapse to zero.", UserWarning)

    mu_passes = []
    ale_passes = []
    epi_passes = []

    with torch.no_grad():
        for start in range(0, T, chunk_size):
            end = min(start + chunk_size, T)
            for _ in range(start, end):
                out = model(batch_r1, batch_r2, batch_r3)
                if isinstance(out, tuple) and len(out) == 4:
                    mu_t, v_t, alpha_t, beta_t = out
                    ale_t = model.head.aleatoric(v_t, alpha_t, beta_t)
                    epi_t = model.head.epistemic(v_t, alpha_t, beta_t)
                else:
                    mu_t = out
                    ale_t = torch.zeros_like(mu_t)
                    epi_t = torch.zeros_like(mu_t)
                mu_passes.append(mu_t.detach())
                ale_passes.append(ale_t.detach())
                epi_passes.append(epi_t.detach())

            if torch.cuda.is_available() and str(device).startswith("cuda"):
                torch.cuda.empty_cache()

    mu_stack = torch.stack(mu_passes, dim=0)
    ale_stack = torch.stack(ale_passes, dim=0)
    epi_stack = torch.stack(epi_passes, dim=0)

    mu_mean = mu_stack.mean(dim=0)
    mu_var = mu_stack.var(dim=0, unbiased=False)
    ale_mean = ale_stack.mean(dim=0)
    epi_mean = epi_stack.mean(dim=0)

    return {
        "mu": mu_mean,
        "mc_epistemic": mu_var,
        "der_aleatoric": ale_mean,
        "der_epistemic": epi_mean,
        "total_epistemic": mu_var + epi_mean,
        "total_uncertainty": mu_var + epi_mean + ale_mean,
    }
