from __future__ import annotations

import torch
from torch_geometric.data import Batch, Data

from crystal_gnn.models.encoder import NNConvEncoder


def _simple_batch(batch_size=4):
    data_list = []
    for _ in range(batch_size):
        x = torch.randn(5, 123)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
        edge_attr = torch.rand(edge_index.shape[1], 50)
        data_list.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr))
    return Batch.from_data_list(data_list)


def test_output_shape():
    model = NNConvEncoder(hidden_dim=64)
    out = model(_simple_batch(4))
    assert out.shape == (4, 64)


def test_mcdropout_active_in_eval():
    model = NNConvEncoder(hidden_dim=32, dropout_rate=0.5)
    model.eval()
    b = _simple_batch(4)
    o1 = model(b)
    o2 = model(b)
    assert not torch.allclose(o1, o2)


def test_mcdropout_variance_nonzero():
    model = NNConvEncoder(hidden_dim=32, dropout_rate=0.5)
    b = _simple_batch(4)
    outs = torch.stack([model(b) for _ in range(10)], dim=0)
    assert torch.all(outs.var(dim=0) > 0)


def test_gradient_flow():
    model = NNConvEncoder(hidden_dim=32)
    b = _simple_batch(4)
    loss = model(b).sum()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert all(g is not None for g in grads)
    assert any(torch.any(g != 0) for g in grads)


def test_xavier_init():
    model = NNConvEncoder(hidden_dim=32)
    lin = [m for m in model.modules() if isinstance(m, torch.nn.Linear)][0]
    w = lin.weight.detach()
    assert torch.var(w) > 0


def test_layernorm_applied():
    model = NNConvEncoder(hidden_dim=32)
    out = model(_simple_batch(4))
    assert abs(out.mean().item()) < 1.0


def test_edge_case_no_edges():
    model = NNConvEncoder(hidden_dim=16)
    d = Data(x=torch.randn(1, 123), edge_index=torch.zeros((2, 0), dtype=torch.long), edge_attr=torch.zeros((0, 50)))
    b = Batch.from_data_list([d])
    out = model(b)
    assert out.shape == (1, 16)
    assert not torch.isnan(out).any()
