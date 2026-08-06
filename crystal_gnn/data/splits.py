"""Split builders for random and structure-aware OOD protocols."""

from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
from dscribe.descriptors import SOAP
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

LOGGER = logging.getLogger(__name__)


def random_split(dataset: Sequence[Any], ratios: tuple[float, float, float] = (0.8, 0.1, 0.1), seed: int = 42) -> dict[str, list[int]]:
    """Return reproducible random train/val/test index split."""
    n = len(dataset)
    if n == 0:
        raise ValueError("random_split received empty dataset")
    if not np.isclose(sum(ratios), 1.0):
        raise ValueError("Split ratios must sum to 1.0")

    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)

    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    train_idx = idx[:n_train].tolist()
    val_idx = idx[n_train : n_train + n_val].tolist()
    test_idx = idx[n_train + n_val :].tolist()
    return {"train": train_idx, "val": val_idx, "test": test_idx}


def _species_union(structures: Sequence[Structure]) -> list[str]:
    elems = sorted({str(site.specie.symbol) for s in structures for site in s.sites})
    return elems or ["H"]


def _material_id_for_idx(labels: dict[str, dict[str, Any]], idx: int, structure: Structure) -> str:
    sid = getattr(structure, "material_id", None)
    if sid is None and hasattr(structure, "properties"):
        sid = structure.properties.get("material_id")
    if sid is not None:
        return str(sid)
    keys = list(labels.keys())
    if idx < len(keys):
        return keys[idx]
    return f"material-{idx}"


def soap_loco_split(structures: Sequence[Structure], labels: dict[str, dict[str, Any]], n_clusters: int = 10, seed: int = 42) -> list[dict[str, list[int]]]:
    """Build SOAP-LOCO splits and return a split for each retained cluster."""
    if len(structures) == 0:
        raise ValueError("SOAP-LOCO requires non-empty structures")
    if n_clusters > len(structures):
        raise ValueError("n_clusters cannot exceed number of structures")

    species = _species_union(structures)
    soap = SOAP(species=species, r_cut=6.0, n_max=9, l_max=9, sigma=0.5, periodic=True, sparse=False)

    soap_vectors: list[np.ndarray] = []
    valid_indices: list[int] = []
    adaptor = AseAtomsAdaptor()
    for i, s in enumerate(structures):
        try:
            ase_atoms = adaptor.get_atoms(s)
            vec = soap.create(ase_atoms, n_jobs=1)
            if vec.ndim == 2:
                vec = vec.mean(axis=0)
            soap_vectors.append(np.asarray(vec, dtype=np.float32))
            valid_indices.append(i)
        except Exception as exc:  # noqa: BLE001
            mid = _material_id_for_idx(labels, i, s)
            LOGGER.warning("SOAP failed for %s: %s", mid, exc)

    if len(soap_vectors) == 0:
        raise ValueError("SOAP-LOCO failed for all structures; no descriptors were generated.")

    X = np.stack(soap_vectors)
    X = StandardScaler().fit_transform(X)

    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    cluster_ids = km.fit_predict(X)

    # Merge tiny clusters into nearest center by distance.
    centers = km.cluster_centers_
    min_cluster_size = 50
    for c in range(n_clusters):
        members = np.where(cluster_ids == c)[0]
        if len(members) >= min_cluster_size:
            continue
        if len(members) == 0:
            continue
        for m in members:
            dists = np.linalg.norm(centers - X[m], axis=1)
            dists[c] = np.inf
            cluster_ids[m] = int(np.argmin(dists))

    splits: list[dict[str, list[int]]] = []
    all_valid = np.array(valid_indices)
    final_clusters = sorted(set(cluster_ids.tolist()))
    for c in final_clusters:
        test_mask = cluster_ids == c
        test_idx = all_valid[test_mask].tolist()
        train_idx = all_valid[~test_mask].tolist()
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue
        tr, val = train_test_split(train_idx, test_size=0.1, random_state=seed, shuffle=True)
        splits.append({"train": list(tr), "val": list(val), "test": test_idx})
    return splits


def crystal_system_split(structures: Sequence[Structure], labels: dict[str, dict[str, Any]]) -> dict[str, dict[str, list[int]]]:
    """Leave-one-crystal-system-out splits keyed by system name."""
    systems = {
        "cubic",
        "hexagonal",
        "tetragonal",
        "orthorhombic",
        "monoclinic",
        "triclinic",
        "trigonal",
    }

    grouped: dict[str, list[int]] = {k: [] for k in systems}
    for i, s in enumerate(structures):
        mid = _material_id_for_idx(labels, i, s)
        cs = str(labels.get(mid, {}).get("crystal_system", "triclinic")).lower()
        if cs not in systems:
            cs = "triclinic"
        grouped[cs].append(i)

    out: dict[str, dict[str, list[int]]] = {}
    for cs, idxs in grouped.items():
        if len(idxs) < 100:
            LOGGER.warning("Skipping crystal system %s due to <100 structures.", cs)
            continue
        test_idx = idxs
        train_pool = [i for i in range(len(structures)) if i not in set(test_idx)]
        tr, val = train_test_split(train_pool, test_size=0.1, random_state=42, shuffle=True)
        out[cs] = {"train_idx": list(tr), "val_idx": list(val), "test_idx": list(test_idx)}
    return out


def composition_split(structures: Sequence[Structure], labels: dict[str, dict[str, Any]], seed: int = 42) -> dict[str, list[int]]:
    """Train on <=3 elements and test on >=4 elements."""
    train_idx: list[int] = []
    test_idx: list[int] = []
    for i, s in enumerate(structures):
        mid = _material_id_for_idx(labels, i, s)
        n_elem = int(labels.get(mid, {}).get("nelements", len({str(site.specie.symbol) for site in s.sites})))
        if n_elem >= 4:
            test_idx.append(i)
        else:
            train_idx.append(i)

    if len(test_idx) < 1000:
        warnings.warn("Composition split test set has <1000 structures; OOD severity may be weak.", UserWarning)

    tr, val = train_test_split(train_idx, test_size=0.1, random_state=seed, shuffle=True) if len(train_idx) > 1 else (train_idx, [])
    return {"train_idx": list(tr), "val_idx": list(val), "test_idx": list(test_idx)}


def save_splits(split_dict: dict[str, Any], path: str) -> None:
    """Persist split metadata to JSON."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f_out:
        json.dump(split_dict, f_out, indent=2)


def load_splits(path: str) -> dict[str, Any]:
    """Load split metadata from JSON."""
    with Path(path).open("r", encoding="utf-8") as f_in:
        return json.load(f_in)
