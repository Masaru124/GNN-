"""NNConv encoder for single-radius crystal graphs."""

from __future__ import annotations

from typing import List

import torch
from torch import nn
from torch_geometric.nn import NNConv, global_mean_pool

from crystal_gnn.uncertainty.mc_dropout import MCDropout


class NNConvEncoder(nn.Module):
    """Stacked NNConv encoder with always-on MC dropout."""

    def __init__(
        self,
        node_feat_dim: int = 123,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout_rate: float = 0.1,
        edge_feat_dim: int = 50,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(node_feat_dim, hidden_dim)

        self.edge_mlps = nn.ModuleList()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        in_channels = hidden_dim
        for _ in range(num_layers):
            edge_nn = nn.Sequential(
                nn.Linear(edge_feat_dim, 128),
                nn.SiLU(),
                nn.Linear(128, in_channels * hidden_dim),
            )
            conv = NNConv(in_channels, hidden_dim, nn=edge_nn, aggr="mean")
            self.edge_mlps.append(edge_nn)
            self.convs.append(conv)
            self.norms.append(nn.LayerNorm(hidden_dim))
            self.dropouts.append(MCDropout(p=dropout_rate))
            in_channels = hidden_dim

        self.act = nn.SiLU()
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, batch) -> torch.Tensor:
        """Encode PyG batch into crystal-level embeddings [B, hidden_dim]."""
        x = self.input_proj(batch.x)
        edge_index = batch.edge_index
        edge_attr = batch.edge_attr

        if edge_index.numel() == 0:
            # Preserve behavior on isolated graphs by using projected node features.
            for norm, dropout in zip(self.norms, self.dropouts):
                x = dropout(self.act(norm(x)))
        else:
            for conv, norm, dropout in zip(self.convs, self.norms, self.dropouts):
                x = conv(x, edge_index, edge_attr)
                x = norm(x)
                x = self.act(x)
                x = dropout(x)

        pooled = global_mean_pool(x, batch.batch)
        return pooled
