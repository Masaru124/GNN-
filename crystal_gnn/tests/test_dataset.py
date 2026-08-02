from __future__ import annotations

import torch
from pymatgen.core import Lattice, Structure

from crystal_gnn.data.dataset import MultiScaleCollate, MultiScaleDataset
from crystal_gnn.data.preprocessing import rbf_encode_distance


class TestMultiScaleDataset:
    def test_getitem_returns_triple(self, tiny_dataset):
        out = tiny_dataset[0]
        assert len(out) == 5
        g1, g2, g3, y, _ = out
        for g in [g1, g2, g3]:
            assert hasattr(g, "x") and hasattr(g, "edge_index") and hasattr(g, "edge_attr") and hasattr(g, "y")
            assert g.x.shape[1] == 123
            assert g.edge_attr.shape[1] == 50
        assert y.shape == (1,)

    def test_rbf_encoding_range(self):
        rbf = rbf_encode_distance([0.0, 1.0, 2.0], radius=4.0, n_rbf=50)
        assert torch.all(rbf >= 0.0) and torch.all(rbf <= 1.0)
        assert torch.allclose(rbf.sum(dim=1), torch.ones(rbf.shape[0]), atol=1e-4)
        assert torch.argmax(rbf[0]).item() == 0

    def test_periodic_boundary_conditions(self, tiny_labels, tmp_path):
        s = Structure(Lattice.cubic(3.0), ["Cu"], [[0, 0, 0]])
        s.properties["material_id"] = "mp-fake-0"
        ds = MultiScaleDataset([s], {"mp-fake-0": tiny_labels["mp-fake-0"]}, radii=[4.0, 6.0, 8.0], cache_dir=str(tmp_path))
        g1, _, _, _, _ = ds[0]
        assert g1.edge_index.shape[1] > 0

    def test_neighbor_cap(self, tiny_labels, tmp_path):
        s = Structure(Lattice.cubic(3.0), ["Cu"] * 8, [[0, 0, 0], [0, 0.5, 0], [0.5, 0, 0], [0.5, 0.5, 0], [0, 0, 0.5], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0.5]])
        s.properties["material_id"] = "mp-fake-0"
        ds = MultiScaleDataset([s], {"mp-fake-0": tiny_labels["mp-fake-0"]}, radii=[4.0, 6.0, 8.0], cache_dir=str(tmp_path), max_neighbors=8)
        g1, _, _, _, _ = ds[0]
        assert g1.edge_index.shape[1] <= 8 * g1.x.shape[0]

    def test_caching(self, tiny_dataset):
        _ = tiny_dataset[0]
        cache_files = list(tiny_dataset.cache_dir.glob("*.pt"))
        assert len(cache_files) > 0
        one = cache_files[0]
        one.unlink()
        _ = tiny_dataset[0]
        assert one.exists()

    def test_nan_node_features(self, tmp_path):
        s = Structure(Lattice.cubic(4.0), ["He"], [[0, 0, 0]])
        s.properties["material_id"] = "mp-fake-0"
        labels = {"mp-fake-0": {"formation_energy_per_atom": -1.0, "band_gap": 0.1}}
        ds = MultiScaleDataset([s], labels, radii=[4.0, 6.0, 8.0], cache_dir=str(tmp_path))
        g1, _, _, _, _ = ds[0]
        assert not torch.isnan(g1.x).any()

    def test_single_atom_structure(self, tmp_path):
        s = Structure(Lattice.cubic(4.0), ["Cu"], [[0, 0, 0]])
        s.properties["material_id"] = "mp-fake-0"
        labels = {"mp-fake-0": {"formation_energy_per_atom": -1.0, "band_gap": 0.1}}
        ds = MultiScaleDataset([s], labels, radii=[4.0, 6.0, 8.0], cache_dir=str(tmp_path))
        g1, _, _, _, _ = ds[0]
        assert g1.x.shape[0] == 1

    def test_empty_neighbors(self, tiny_structure_list, tiny_labels, tmp_path):
        ds = MultiScaleDataset(tiny_structure_list, tiny_labels, radii=[0.1, 0.2, 0.3], cache_dir=str(tmp_path))
        assert len(ds) <= len(tiny_structure_list)

    def test_collate_function(self, tiny_dataset):
        items = [tiny_dataset[i] for i in range(4)]
        b1, b2, b3, y, ids = MultiScaleCollate()(items)
        assert y.shape[0] == 4
        assert len(ids) == 4
        assert b1.num_graphs == 4 and b2.num_graphs == 4 and b3.num_graphs == 4
