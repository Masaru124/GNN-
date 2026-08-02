"""Multi-scale crystal GNN model assembly."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from crystal_gnn.models.der_head import DERHead
from crystal_gnn.models.encoder import NNConvEncoder
from crystal_gnn.models.fusion import ScaleFusionAttention


class MultiScaleGNN(nn.Module):
    """Three-scale model with attention/concat fusion and optional DER head."""

    def __init__(
        self,
        hidden_dim: int = 128,
        num_encoder_layers: int = 3,
        dropout_rate: float = 0.1,
        use_attention_fusion: bool = True,
        use_der: bool = True,
        radii: list[float] | None = None,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_attention_fusion = use_attention_fusion
        self.use_der = use_der
        self.radii = radii or [4.0, 6.0, 8.0]

        self.encoder_r1 = NNConvEncoder(hidden_dim=hidden_dim, num_layers=num_encoder_layers, dropout_rate=dropout_rate)
        self.encoder_r2 = NNConvEncoder(hidden_dim=hidden_dim, num_layers=num_encoder_layers, dropout_rate=dropout_rate)
        self.encoder_r3 = NNConvEncoder(hidden_dim=hidden_dim, num_layers=num_encoder_layers, dropout_rate=dropout_rate)

        if use_attention_fusion:
            self.fusion = ScaleFusionAttention(hidden_dim=hidden_dim, nhead=4, dropout=dropout_rate)
            self.concat_proj = None
        else:
            self.fusion = None
            self.concat_proj = nn.Linear(3 * hidden_dim, hidden_dim)

        if use_der:
            self.head = DERHead(hidden_dim=hidden_dim)
        else:
            self.head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1))

        self._last_attention: torch.Tensor | None = None

    def forward(self, batch_r1, batch_r2=None, batch_r3=None):
        """Run forward pass and return DER tuple or scalar prediction."""
        h1 = self.encoder_r1(batch_r1)

        if batch_r2 is None:
            batch_r2 = batch_r1
        if batch_r3 is None:
            batch_r3 = batch_r1

        h2 = self.encoder_r2(batch_r2)
        h3 = self.encoder_r3(batch_r3)

        if self.use_attention_fusion:
            fused, attn = self.fusion(h1, h2, h3)
            self._last_attention = attn
        else:
            fused = self.concat_proj(torch.cat([h1, h2, h3], dim=-1))
            self._last_attention = None

        out = self.head(fused)
        return out

    def predict_with_uncertainty(self, batch_r1, batch_r2, batch_r3, T: int = 30) -> dict[str, torch.Tensor]:
        """Compute MC+DER uncertainty statistics over T stochastic passes."""
        if T < 2:
            raise ValueError("T must be >= 2 to estimate variance.")

        mu_list = []
        der_ale_list = []
        der_epi_list = []

        self.train()
        with torch.no_grad():
            for start in range(0, T, 10):
                end = min(start + 10, T)
                for _ in range(start, end):
                    out = self(batch_r1, batch_r2, batch_r3)
                    if self.use_der:
                        mu, v, alpha, beta = out
                        der_ale = DERHead.aleatoric(v, alpha, beta)
                        der_epi = DERHead.epistemic(v, alpha, beta)
                    else:
                        mu = out
                        der_ale = torch.zeros_like(mu)
                        der_epi = torch.zeros_like(mu)
                    mu_list.append(mu)
                    der_ale_list.append(der_ale)
                    der_epi_list.append(der_epi)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        all_passes = torch.stack(mu_list, dim=0)
        mean_pred = all_passes.mean(dim=0)
        mc_epistemic = all_passes.var(dim=0, unbiased=False)
        der_aleatoric = torch.stack(der_ale_list, dim=0).mean(dim=0)
        der_epistemic = torch.stack(der_epi_list, dim=0).mean(dim=0)
        combined_epistemic = mc_epistemic + der_epistemic

        return {
            "prediction": mean_pred,
            "epistemic": combined_epistemic,
            "aleatoric": der_aleatoric,
            "mc_epistemic": mc_epistemic,
            "der_epistemic": der_epistemic,
            "all_passes": all_passes,
        }

    def get_attention_weights(self) -> dict[str, Any]:
        """Return attention weights from the last forward pass."""
        if self._last_attention is None:
            return {}
        return {"last_batch": self._last_attention}


class SingleScaleGNN(nn.Module):
    """Single-scale ablation model using only r1 encoder."""

    def __init__(
        self,
        hidden_dim: int = 128,
        num_encoder_layers: int = 3,
        dropout_rate: float = 0.1,
        use_der: bool = False,
        **_: Any,
    ) -> None:
        super().__init__()
        self.encoder_r1 = NNConvEncoder(hidden_dim=hidden_dim, num_layers=num_encoder_layers, dropout_rate=dropout_rate)
        self.use_der = use_der
        if use_der:
            self.head = DERHead(hidden_dim=hidden_dim)
        else:
            self.head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1))

    def forward(self, batch_r1, batch_r2=None, batch_r3=None):
        """Forward pass using only first-radius graph batch."""
        h = self.encoder_r1(batch_r1)
        return self.head(h)
