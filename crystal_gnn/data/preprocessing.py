"""Preprocessing helpers for crystal graph construction."""

from __future__ import annotations

from typing import Iterable, List

import numpy as np
import torch
from pymatgen.core import Element, Structure


_NODE_FEAT_DIM = 123


def safe_float(value: float | None, default: float = 0.0) -> float:
    """Convert to float while replacing invalid values with a default."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if np.isnan(out) or np.isinf(out):
        return default
    return out


def atomic_number_onehot(atomic_number: int, max_z: int = 118) -> np.ndarray:
    """Return one-hot encoding for atomic number in [1, max_z]."""
    onehot = np.zeros(max_z, dtype=np.float32)
    if 1 <= atomic_number <= max_z:
        onehot[atomic_number - 1] = 1.0
    return onehot


def element_scalar_features(element: Element) -> np.ndarray:
    """Return scalar chemistry features used with one-hot element encoding."""
    en = safe_float(element.X, default=0.0)
    radius = safe_float(element.atomic_radius, default=0.0)
    group = safe_float(element.group, default=0.0)
    period = safe_float(element.row, default=0.0)
    atomic_mass = safe_float(element.atomic_mass, default=0.0)
    return np.array([en, radius, group, period, atomic_mass], dtype=np.float32)


def build_node_features(structure: Structure) -> torch.Tensor:
    """Build node feature tensor of shape [n_atoms, 123]."""
    features: List[np.ndarray] = []
    for site in structure.sites:
        z = int(site.specie.Z)
        element = site.specie.element if hasattr(site.specie, "element") else site.specie
        vec = np.concatenate(
            [atomic_number_onehot(z), element_scalar_features(element)],
            axis=0,
        )
        if vec.shape[0] != _NODE_FEAT_DIM:
            raise ValueError(f"Node feature dimension mismatch: got {vec.shape[0]}, expected {_NODE_FEAT_DIM}.")
        vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
        features.append(vec)
    return torch.tensor(np.stack(features), dtype=torch.float32)


def rbf_encode_distance(distances: Iterable[float], radius: float, n_rbf: int = 50, width: float = 0.5) -> torch.Tensor:
    """Encode distances with Gaussian RBF and normalize rows into [0, 1]."""
    d = np.asarray(list(distances), dtype=np.float32).reshape(-1, 1)
    if d.size == 0:
        return torch.zeros((0, n_rbf), dtype=torch.float32)
    centers = np.linspace(0.0, radius, n_rbf, dtype=np.float32).reshape(1, -1)
    rbf = np.exp(-((d - centers) ** 2) / (2.0 * width**2))
    denom = np.sum(rbf, axis=1, keepdims=True) + 1e-12
    rbf = np.clip(rbf / denom, 0.0, 1.0)
    return torch.tensor(rbf, dtype=torch.float32)


def node_feature_dim() -> int:
    """Return the fixed node feature dimension."""
    return _NODE_FEAT_DIM
