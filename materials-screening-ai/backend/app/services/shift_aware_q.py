# -*- coding: utf-8 -*-
"""Optional "shift-aware" conformal q — distance-conditional, flag-gated.

Prototype (research/loco_cross_conformal/distance_q.json): nested
distance-decile q improves worst-fold coverage 0.7511 -> 0.8038 and Q1
0.8845 -> 0.8911 vs the global nested LOFO q at 0.994x median half-width,
so the decision rule shipped it as an OPTIONAL mode. Disabled by default:
opt in with GNNPredictorService(shift_robust-style flag) shift_aware=True or
env MATSCREEN_SHIFT_AWARE_Q=1. When artifacts or dscribe are unavailable,
compute() returns None and the caller falls back to q_hat_conformal.

How it works (identical recipe to crystal_gnn/scripts/loco_distance_q.py):
  SOAP (r_cut=6, n_max=9, l_max=9, sigma=0.5, average="inner", 80-species
  union) -> count-sketch 512-d (seed 42) -> cosine distance to the nearest
  structure of the PRODUCTION train -> pooled distance decile -> per-bin
  deployment q (pooled 90% quantile per bin).

Build output (gitignored cache, regenerate with loco_distance_q.py):
  crystal_gnn/data/cache/shift_aware_q_artifacts.npz
  crystal_gnn/data/cache/soap_fingerprint512_seed42.npz
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[4]
CACHE_DIR = REPO_ROOT / "crystal_gnn" / "data" / "cache"
ARTIFACTS = CACHE_DIR / "shift_aware_q_artifacts.npz"
FINGERPRINTS = CACHE_DIR / "soap_fingerprint512_seed42.npz"


class ShiftAwareQModel:
    """Lazy-loaded distance-conditional q model (thread-safe init)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._loaded = False
        self._failed = False
        self._soap = None
        self._H: Optional[np.ndarray] = None
        self._SGN: Optional[np.ndarray] = None
        self._edges: Optional[np.ndarray] = None
        self._bin_q: Optional[np.ndarray] = None
        self._ref: Optional[np.ndarray] = None
        self._meta: Dict[str, Any] = {}

    def _load(self) -> bool:
        if self._loaded:
            return not self._failed
        with self._lock:
            if self._loaded:
                return not self._failed
            try:
                if not ARTIFACTS.exists() or not FINGERPRINTS.exists():
                    raise FileNotFoundError(
                        f"shift-aware artifacts missing ({ARTIFACTS} / {FINGERPRINTS}); "
                        "rebuild with crystal_gnn/scripts/loco_distance_q.py")
                from dscribe.descriptors import SOAP  # heavy import, lazy

                art = np.load(ARTIFACTS, allow_pickle=True)
                species = [str(s) for s in art["species"]]
                self._H = art["H"].astype(np.int64)  # bincount indices
                self._SGN = art["SGN"].astype(np.float64)
                self._edges = art["edges"].astype(np.float64)
                self._bin_q = art["bin_q"].astype(np.float64)
                self._meta = json.loads(str(art["meta"]))
                self._embed_dim = int(self._meta["embed"]["dim"])
                scfg = self._meta["soap"]
                self._soap = SOAP(
                    species=species, r_cut=scfg["r_cut"], n_max=scfg["n_max"],
                    l_max=scfg["l_max"], sigma=scfg["sigma"],
                    periodic=scfg["periodic"], sparse=scfg["sparse"],
                    average=scfg["average"],
                )
                fp = np.load(FINGERPRINTS)
                ref_rows = art["ref_rows"].astype(np.int64)
                self._ref = fp["X"][ref_rows]  # (N, 512) float32
                if self._H.size != int(self._meta["embed"]["soap_dim"]):
                    raise ValueError("artifact soap_dim mismatch")
                self._loaded = True
                logger.info(
                    "[shift-aware] loaded: species=%d ref=%d bins=%d",
                    len(species), len(self._ref), len(self._bin_q))
            except Exception as exc:  # noqa: BLE001 — degrade to default q
                self._loaded = True
                self._failed = True
                logger.warning("[shift-aware] disabled, falling back to q_hat_conformal: %s", exc)
            return not self._failed

    def compute(self, structure) -> Optional[Dict[str, Any]]:
        """Return {'q','distance','bin'} for a pymatgen Structure, or None."""
        if not self._load():
            return None
        try:
            from pymatgen.io.ase import AseAtomsAdaptor

            v = self._soap.create(AseAtomsAdaptor.get_atoms(structure), n_jobs=1)
            if hasattr(v, "todense"):
                v = v.todense()
            if hasattr(v, "toarray"):
                v = v.toarray()
            v = np.asarray(v, dtype=np.float64).reshape(-1)
            if v.size != self._H.size:
                raise ValueError(f"descriptor dim {v.size} != artifact dim {self._H.size}")
            p = np.bincount(self._H, weights=self._SGN * v,
                            minlength=self._embed_dim)[:self._embed_dim]
            n = float(np.linalg.norm(p))
            if n > 0:
                p = p / n
            p = p.astype(np.float32)
            d = float(1.0 - (self._ref @ p).max())
            b = int(np.clip(np.searchsorted(self._edges, d, side="right") - 1,
                            0, len(self._bin_q) - 1))
            return {"q": float(self._bin_q[b]), "distance": d, "bin": b}
        except Exception as exc:  # noqa: BLE001 — one bad structure must not break serving
            logger.warning("[shift-aware] q computation failed, using default q: %s", exc)
            return None


_MODEL = ShiftAwareQModel()


def compute_shift_aware_q(structure) -> Optional[Dict[str, Any]]:
    """Public entry: distance-conditional q, or None to use the default q."""
    return _MODEL.compute(structure)
