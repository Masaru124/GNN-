# -*- coding: utf-8 -*-
"""
Leave-one-cluster-out (LOCO) cross-conformal calibration over the 10 soap_loco clusters.

For each held-out cluster c:
  * train a fresh model on the remaining 9 clusters (scripts/train.py, production
    recipe + seed policy)
  * score cluster c with that model: conformal score = |y - mu| / sigma (DER)
  * cross-conformal pooling: q_c = 90% quantile of scores from the OTHER folds
    (never from fold c itself), applied to fold c

Fold 0 is the production run (paper_A7_soap_loco_formation_energy_per_atom); it is
scored with the production checkpoint and never retrained.

Outputs (tracked, small):
  research/loco_cross_conformal/manifest.jsonl   cluster, split sha256, ckpt sha256,
                                                seed, wall time, n_test, val MAE, epoch
  research/loco_cross_conformal/fold<c>_scores.npz

Usage (run from crystal_gnn/):
  python scripts/loco_cross_conformal.py cache                 # prebuild graph cache once
  python scripts/loco_cross_conformal.py run --folds 3,9,1,2,4,5,6,7,8 --max-hours 6
  python scripts/loco_cross_conformal.py score --folds 0        # production fold only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from torch_geometric.loader import DataLoader  # noqa: E402

from train import _build_model, load_data  # noqa: E402


def _load_dataset_module():
    """Return the packaged dataset module (the one recording orig_to_dataset_idx).

    Two copies of crystal_gnn/data/dataset.py exist in this tree; only the packaged
    one maps original indices to dataset indices, so import that explicitly.
    """
    import importlib
    import inspect

    for name in ("crystal_gnn.crystal_gnn.data.dataset", "crystal_gnn.data.dataset"):
        try:
            mod = importlib.import_module(name)
        except ImportError:
            continue
        if "orig_to_dataset_idx" in inspect.getsource(mod.MultiScaleDataset.__init__):
            return mod
    raise ImportError("packaged crystal_gnn dataset module not found")


_DS = _load_dataset_module()
MultiScaleDataset = _DS.MultiScaleDataset
MultiScaleCollate = _DS.MultiScaleCollate

REPO = ROOT.parent
OUT_DIR = REPO / "materials-screening-ai" / "research" / "loco_cross_conformal"
PROD_CKPT = ROOT / "checkpoints/paper_A7_soap_loco_formation_energy_per_atom/best.pt"
SPLIT_FILE = ROOT / "data/splits/soap_loco.json"
COVERAGE = 0.90
N_CLUSTERS = 10


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_folds(max_structures: int = 50000) -> list[dict]:
    """Filtered soap_loco folds in index space (i < max_structures)."""
    raw = json.loads(SPLIT_FILE.read_text(encoding="utf-8"))["splits"]
    return [
        {
            "cluster": c,
            "train": [i for i in s["train"] if i < max_structures],
            "val": [i for i in s["val"] if i < max_structures],
            "test": [i for i in s["test"] if i < max_structures],
        }
        for c, s in enumerate(raw[:N_CLUSTERS])
    ]


def split_sha(fold: dict) -> str:
    payload = json.dumps({k: sorted(v) for k, v in fold.items()}, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_dataset(cache_dir: str = "./data/cache"):
    structures, labels = load_data(str(ROOT / "data" / "raw"))
    structures = structures[:50000]
    ds = MultiScaleDataset(
        structures=structures,
        labels=labels,
        radii=[4.0, 6.0, 8.0],
        target="formation_energy_per_atom",
        cache_dir=cache_dir,
        max_neighbors=24,
    )
    return ds, labels


def to_dataset_idx(ds, indices: list[int]) -> list[int]:
    mapping = ds.orig_to_dataset_idx
    return [mapping[i] for i in indices if i in mapping]


def warm_cache(ds, num_workers: int = 8, limit: int | None = None) -> None:
    """Build every graph once so all folds reuse the on-disk cache."""
    idx = list(range(len(ds)))
    if limit:
        idx = idx[:limit]
    loader = DataLoader(
        Subset(ds, idx),
        batch_size=16,
        shuffle=False,
        collate_fn=MultiScaleCollate(),
        num_workers=num_workers,
    )
    t0 = time.time()
    done = 0
    for _ in loader:
        done += 16
        if done % 1600 == 0:
            print(f"[cache] {done}/{len(idx)} graphs  {time.time() - t0:.0f}s", flush=True)
    print(f"[cache] done {len(idx)} graphs in {time.time() - t0:.0f}s", flush=True)


def score_indices(ckpt_path: Path, ds, indices: list[int], device: str, batch_size: int = 32):
    """Return (material_id, y, mu, sigma, score) for the given dataset indices.

    Graphs are served from the on-disk cache (./data/cache) when present, so the
    one-off graph build is paid once and every fold reuses it.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = _build_model(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()  # deterministic: MC dropout inactive in eval
    loader = DataLoader(
        Subset(ds, indices),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=MultiScaleCollate(),
        num_workers=4,
    )
    ids, ys, mus, sigmas = [], [], [], []
    with torch.no_grad():
        for b1, b2, b3, y, bid in loader:
            b1, b2, b3, y = b1.to(device), b2.to(device), b3.to(device), y.to(device)
            mu, v, alpha, beta = model(b1, b2, b3)
            mu = mu.reshape(-1)
            v = torch.clamp(v.reshape(-1), min=1e-4)
            alpha = torch.clamp(alpha.reshape(-1), min=1.0001)
            beta = torch.clamp(beta.reshape(-1), min=1e-4)
            sigma = torch.sqrt(torch.clamp((beta * (1.0 + 1.0 / v)) / (alpha - 1.0), min=1e-8))
            ids.extend(bid)
            ys.extend(y.reshape(-1).cpu().numpy().tolist())
            mus.extend(mu.cpu().numpy().tolist())
            sigmas.extend(sigma.cpu().numpy().tolist())
    y_arr = np.asarray(ys, dtype=np.float64)
    mu_arr = np.asarray(mus, dtype=np.float64)
    sig_arr = np.asarray(sigmas, dtype=np.float64)
    score = np.abs(y_arr - mu_arr) / np.maximum(sig_arr, 1e-8)
    return ids, y_arr, mu_arr, sig_arr, score


def train_fold(cluster: int, max_hours: float, seed: int, num_workers: int) -> Path:
    run_id = f"loco_A7_soap_cluster{cluster}_seed{seed}"
    ckpt = ROOT / "checkpoints" / run_id / "best.pt"
    if ckpt.exists():
        print(f"[train] fold {cluster}: checkpoint exists, skipping", flush=True)
        return ckpt
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "train.py"),
        "--ablation", "A7",
        "--split", "soap_loco",
        "--soap_loco_idx", str(cluster),
        "--max_structures", "50000",
        "--max_neighbors", "24",
        "--batch_size", "8",
        "--accumulate_grad_batches", "4",
        "--max_epochs", "100",
        "--warm_up_epochs", "5",
        "--seed", str(seed),
        "--num_workers", str(num_workers),
        "--max_hours", str(max_hours),
        "--auto_resume",
        "--run_id", run_id,
    ]
    print(f"[train] fold {cluster}: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)
    if not ckpt.exists():
        raise RuntimeError(f"fold {cluster}: no checkpoint at {ckpt}")
    return ckpt


def append_manifest(row: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "manifest.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def record(cluster: int, ckpt: Path, fold: dict, seed: int, wall: float, ds) -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    idx = to_dataset_idx(ds, fold["test"])
    ids, y, mu, sigma, score = score_indices(ckpt, ds, idx, "cuda" if torch.cuda.is_available() else "cpu")
    np.savez_compressed(
        OUT_DIR / f"fold{cluster}_scores.npz",
        material_ids=np.asarray(ids),
        target=y,
        mu=mu,
        sigma=sigma,
        score=score,
    )
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    row = {
        "cluster": cluster,
        "run_id": ck.get("run_id"),
        "split_sha256": split_sha(fold),
        "n_test_index": len(fold["test"]),
        "n_test_scored": int(len(score)),
        "checkpoint": str(ckpt.relative_to(REPO)).replace("\\", "/"),
        "checkpoint_sha256": sha256_file(ckpt),
        "seed": seed,
        "wall_time_s": round(wall, 1),
        "best_epoch": ck.get("epoch"),
        "best_val_mae": ck.get("best_val_mae"),
        "test_mae_eV_per_atom": float(np.abs(y - mu).mean()),
        "score_quantile_90": float(np.quantile(score, COVERAGE)),
        "score_quantile_80": float(np.quantile(score, 0.80)),
        "median_half_width_eV_at_q1p0254": float(1.0254 * np.median(sigma)),
        "note": "cluster held out of training; score = |y-mu|/sigma(DER)",
    }
    append_manifest(row)
    print(json.dumps(row, indent=1), flush=True)
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("cache")
    p.add_argument("--num-workers", type=int, default=8)

    p = sub.add_parser("run")
    p.add_argument("--folds", type=str, required=True)
    p.add_argument("--max-hours", type=float, default=6.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=6)

    p = sub.add_parser("score")
    p.add_argument("--folds", type=str, default="0")
    p.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()

    if args.cmd == "cache":
        ds, _ = build_dataset()
        print(f"[cache] dataset size {len(ds)}", flush=True)
        warm_cache(ds, num_workers=args.num_workers)
        return

    folds = load_folds()
    ds, _ = build_dataset()

    for c in [int(x) for x in args.folds.split(",")]:
        fold = folds[c]
        t0 = time.time()
        if c == 0:
            ckpt = PROD_CKPT
            print(f"[score] fold 0 uses production checkpoint (no retrain)", flush=True)
        else:
            ckpt = train_fold(c, args.max_hours, args.seed, args.num_workers)
        if (OUT_DIR / f"fold{c}_scores.npz").exists():
            print(f"[score] fold {c} scores exist, skipping record", flush=True)
            continue
        record(c, ckpt, fold, args.seed, time.time() - t0, ds)


if __name__ == "__main__":
    main()
