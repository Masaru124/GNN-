"""Split builders for random and structure-aware OOD protocols."""

from __future__ import annotations

import json
import logging
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
try:
    from dscribe.descriptors import SOAP
except ImportError:
    SOAP = None
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


def _species_union(structures: Sequence[Structure], max_species: int | None = None) -> list[str]:
    counts: dict[str, int] = {}
    for s in structures:
        for site in s.sites:
            sym = str(site.specie.symbol)
            counts[sym] = counts.get(sym, 0) + 1

    if not counts:
        return ["H"]

    if max_species is not None and max_species > 0 and len(counts) > max_species:
        top = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:max_species]
        species = sorted([k for k, _ in top])
        LOGGER.warning(
            "SOAP-LOCO species capped to top-%d elements by frequency (from %d total)",
            max_species,
            len(counts),
        )
        return species

    return sorted(counts.keys())


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


def soap_loco_split(
    structures: Sequence[Structure],
    labels: dict[str, dict[str, Any]],
    n_clusters: int = 10,
    seed: int = 42,
    soap_n_jobs: int = 1,
    log_every: int = 1000,
    soap_r_cut: float = 6.0,
    soap_n_max: int = 9,
    soap_l_max: int = 9,
    soap_sigma: float = 0.5,
    soap_average: str = "inner",
    soap_max_species: int | None = None,
) -> list[dict[str, list[int]]]:
    """Build SOAP-LOCO splits and return a split for each retained cluster."""
    if len(structures) == 0:
        raise ValueError("SOAP-LOCO requires non-empty structures")
    if n_clusters > len(structures):
        raise ValueError("n_clusters cannot exceed number of structures")
    if SOAP is None:
        raise RuntimeError("dscribe package is required for SOAP-LOCO split generation. Please install dscribe.")

    t0 = time.time()
    species = _species_union(structures, max_species=soap_max_species)
    soap = SOAP(
        species=species,
        r_cut=soap_r_cut,
        n_max=soap_n_max,
        l_max=soap_l_max,
        sigma=soap_sigma,
        periodic=True,
        sparse=True,
        average=soap_average,
    )
    LOGGER.info("SOAP-LOCO: starting descriptor generation for %d structures (soap_n_jobs=%d)", len(structures), soap_n_jobs)
    LOGGER.info(
        "SOAP-LOCO params: species=%d, r_cut=%.2f, n_max=%d, l_max=%d, sigma=%.2f, average=%s",
        len(species),
        soap_r_cut,
        soap_n_max,
        soap_l_max,
        soap_sigma,
        soap_average,
    )
    if soap_n_jobs > 1:
        LOGGER.info(
            "SOAP-LOCO note: higher soap_n_jobs can increase peak RAM usage for large descriptor spaces."
        )

    soap_vectors: list[np.ndarray] = []
    valid_indices: list[int] = []
    species_set = set(species)
    skipped_unknown_species = 0
    skipped_examples: list[tuple[str, list[str]]] = []
    soap_failures = 0
    adaptor = AseAtomsAdaptor()
    for i, s in enumerate(structures):
        struct_species = {str(site.specie.symbol) for site in s.sites}
        unknown_species = sorted(struct_species - species_set)
        if unknown_species:
            skipped_unknown_species += 1
            if len(skipped_examples) < 5:
                mid = _material_id_for_idx(labels, i, s)
                skipped_examples.append((mid, unknown_species))
            continue

        try:
            ase_atoms = adaptor.get_atoms(s)
            vec = soap.create(ase_atoms, n_jobs=soap_n_jobs)

            # dscribe may return sparse matrices for large descriptor spaces; average per-atom
            # descriptors without materializing full dense (n_atoms, n_features) arrays.
            if hasattr(vec, "mean"):
                vec = vec.mean(axis=0)
            if hasattr(vec, "todense"):
                vec = vec.todense()
            if hasattr(vec, "toarray"):
                vec = vec.toarray()
            vec = np.asarray(vec, dtype=np.float32).reshape(-1)

            soap_vectors.append(vec)
            valid_indices.append(i)
        except Exception as exc:  # noqa: BLE001
            mid = _material_id_for_idx(labels, i, s)
            soap_failures += 1
            if soap_failures <= 5:
                LOGGER.warning("SOAP failed for %s: %s", mid, exc)

        if (i + 1) % log_every == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / max(elapsed, 1e-9)
            LOGGER.info(
                "SOAP-LOCO progress: %d/%d (%.1f%%), valid=%d, skipped_species=%d, failed=%d, elapsed=%.1fs, rate=%.2f structs/s",
                i + 1,
                len(structures),
                100.0 * (i + 1) / max(len(structures), 1),
                len(valid_indices),
                skipped_unknown_species,
                soap_failures,
                elapsed,
                rate,
            )

    if len(soap_vectors) == 0:
        raise ValueError("SOAP-LOCO failed for all structures; no descriptors were generated.")

    LOGGER.info(
        "SOAP-LOCO: generated descriptors for %d/%d structures (skipped_species=%d, failed=%d)",
        len(soap_vectors),
        len(structures),
        skipped_unknown_species,
        soap_failures,
    )
    for mid, unknown in skipped_examples:
        LOGGER.info("SOAP-LOCO skipped example %s due to unknown species: %s", mid, unknown)
    X = np.stack(soap_vectors)
    LOGGER.info("SOAP-LOCO: scaling descriptors")
    X = StandardScaler().fit_transform(X)

    LOGGER.info("SOAP-LOCO: running KMeans with n_clusters=%d", n_clusters)
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

    LOGGER.info("SOAP-LOCO: completed %d LOCO splits in %.1fs", len(splits), time.time() - t0)
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
        # Avoid rebuilding set(test_idx) for each element, which is expensive on large datasets.
        test_set = set(test_idx)
        train_pool = [i for i in range(len(structures)) if i not in test_set]
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
