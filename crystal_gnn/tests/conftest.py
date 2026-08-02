"""Shared pytest fixtures for Crystal GNN."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from pymatgen.core import Lattice, Structure
from torch.utils.data import DataLoader

from crystal_gnn.data.dataset import MultiScaleCollate, MultiScaleDataset
from crystal_gnn.models.ms_gnn import MultiScaleGNN


@pytest.fixture
def tiny_structure_list():
    rng = np.random.default_rng(42)
    structs = []
    specs = [
        ("Pm-3m", "cubic"),
        ("P6_3/mmc", "hexagonal"),
        ("Pnma", "orthorhombic"),
        ("P2_1/c", "monoclinic"),
    ]
    for i in range(20):
        sg, _ = specs[i // 5]
        a = float(rng.uniform(3.0, 5.5))
        c = float(rng.uniform(4.0, 7.0))
        if "hex" in sg:
            lattice = Lattice.hexagonal(a, c)
        elif "Pnma" in sg:
            lattice = Lattice.orthorhombic(a, a + 0.2, c)
        elif "P2" in sg:
            lattice = Lattice.monoclinic(a, a + 0.3, c, 105)
        else:
            lattice = Lattice.cubic(a)

        species = ["Ti", "O", "O", "Ti"]
        coords = rng.random((4, 3)).tolist()
        s = Structure(lattice=lattice, species=species, coords=coords)
        s.properties["material_id"] = f"mp-fake-{i}"
        structs.append(s)
    return structs


@pytest.fixture
def tiny_labels():
    rng = np.random.default_rng(7)
    labels = {}
    for i in range(20):
        labels[f"mp-fake-{i}"] = {
            "formation_energy_per_atom": float(rng.uniform(-4.0, 0.0)),
            "band_gap": float(rng.uniform(0.0, 5.0)),
            "crystal_system": ["cubic", "hexagonal", "orthorhombic", "monoclinic"][i // 5],
            "nelements": int(rng.integers(1, 5)),
        }
    return labels


@pytest.fixture
def tiny_dataset(tiny_structure_list, tiny_labels, tmp_path):
    return MultiScaleDataset(
        structures=tiny_structure_list,
        labels=tiny_labels,
        radii=[4.0, 6.0, 8.0],
        cache_dir=str(tmp_path / "cache"),
    )


@pytest.fixture
def batch_triple(tiny_dataset):
    loader = DataLoader(tiny_dataset, batch_size=4, collate_fn=MultiScaleCollate(), shuffle=False)
    return next(iter(loader))


@pytest.fixture
def small_model():
    return MultiScaleGNN(hidden_dim=32, num_encoder_layers=2)


@pytest.fixture
def trained_predictions():
    rng = np.random.default_rng(123)
    y_pred = rng.normal(0, 1, 100)
    sigma = np.abs(rng.normal(0.5, 0.2, 100))
    y_true = y_pred + rng.normal(0, 0.3, 100)
    return y_pred, sigma, y_true


@pytest.fixture
def tio2_pair_structures():
    base = Path(__file__).parent / "fixtures"
    s1 = Structure.from_file(base / "mp-1341203.cif")
    s2 = Structure.from_file(base / "mp-2901430.cif")
    return s1, s2
