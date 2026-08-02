"""Cross-attention fusion across scale embeddings."""

from __future__ import annotations

import torch
from torch import nn


class ScaleFusionAttention(nn.Module):
    """Fuse 3 scale embeddings using attention with h1 as anchor query."""

    def __init__(self, hidden_dim: int = 128, nhead: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        if hidden_dim % nhead != 0:
            raise ValueError("hidden_dim must be divisible by nhead")
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=nhead, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.last_attn_weights: torch.Tensor | None = None

    def forward(self, h1: torch.Tensor, h2: torch.Tensor, h3: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return fused embeddings and mean-per-head attention weights [B, 3]."""
        h1 = torch.nan_to_num(h1, nan=0.0, posinf=1e4, neginf=-1e4)
        h2 = torch.nan_to_num(h2, nan=0.0, posinf=1e4, neginf=-1e4)
        h3 = torch.nan_to_num(h3, nan=0.0, posinf=1e4, neginf=-1e4)
        stacked = torch.stack([h1, h2, h3], dim=1)
        query = h1.unsqueeze(1)
        out, attn_weights = self.attn(
            query=query,
            key=stacked,
            value=stacked,
            need_weights=True,
            average_attn_weights=False,
        )
        self.last_attn_weights = attn_weights
        fused = self.norm(out.squeeze(1) + h1)
        fused = torch.nan_to_num(fused, nan=0.0, posinf=1e4, neginf=-1e4)
        attn_mean = attn_weights.mean(dim=1).squeeze(1)
        attn_mean = attn_mean / (attn_mean.sum(dim=-1, keepdim=True) + 1e-12)
        return fused, attn_mean

    def get_attention_weights(self) -> torch.Tensor | None:
        """Return stored attention weights from the last forward pass."""
        return self.last_attn_weights
