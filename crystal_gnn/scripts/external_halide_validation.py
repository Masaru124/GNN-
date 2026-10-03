"""
External Pb-halide validation set (memory-light).

Streams data/raw/mp_structures.json.gz once and keeps only Pb-halide structures
that are provably outside the production checkpoint's training indices, then
scores them with the shipped A7 model under the deterministic default.

Memory-light on purpose: it never builds the 50k-structure MultiScaleDataset, so
it can run while a training job holds the rest of the RAM.

Run from crystal_gnn/:
    python scripts/external_halide_validation.py --min-n 100
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(REPO / "materials-screening-ai" / "backend"))

OUT = REPO / "materials-screening-ai" / "research" / "coverage_reports"
LABELS = ROOT / "data" / "raw" / "mp_labels.csv"
STRUCTURES = ROOT / "data" / "raw" / "mp_structures.json.gz"
SPLITS = ROOT / "data" / "data" / "splits" / "soap_loco.json"
if not SPLITS.exists():
    SPLITS = ROOT / "data" / "splits" / "soap_loco.json"
CKPT = ROOT / "checkpoints" / "paper_A7_soap_loco_formation_energy_per_atom" / "best.pt"
Q_SHIPPED = 1.0254
MAX_STRUCTURES = 50000
PB_HALOGENS = {"I", "Br", "Cl"}


def wilson(k: int, n: int, z: float = 1.959963985):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def load_labels():
    out = {}
    with open(LABELS, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            fe = row.get("formation_energy_per_atom")
            if fe in (None, "", "None"):
                continue
            out[row["material_id"]] = float(fe)
    return out


def train_indices():
    """Filtered soap_loco fold-0 train indices (the production training set)."""
    splits = json.loads(SPLITS.read_text(encoding="utf-8"))["splits"]
    tr = splits[0]["train"]
    return {i for i in tr if i < MAX_STRUCTURES}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--min-n", type=int, default=100)
    ap.add_argument("--out-prefix", default="external_halide_validation")
    args = ap.parse_args()

    labels = load_labels()
    print(f"[external] labels with formation energy: {len(labels)}")
    tr = train_indices()
    print(f"[external] production train indices: {len(tr)}")

    from pymatgen.core import Structure

    selected = []  # (index, material_id, structure)
    with gzip.open(STRUCTURES, "rt", encoding="utf-8") as fh:
        records = json.load(fh)
    print(f"[external] raw records: {len(records)}")

    for idx, rec in enumerate(records):
        mid = rec.get("material_id")
        if mid in labels and idx not in tr:
            doc = rec["structure"]
            els = {s["label"] for s in doc["sites"]} if "sites" in doc else set()
            if "Pb" in els and els & PB_HALOGENS:
                selected.append((idx, mid, Structure.from_dict(doc)))
    print(f"[external] Pb-halide structures outside train: {len(selected)}")
    outside_cutoff = sum(1 for idx, _, _ in selected if idx >= MAX_STRUCTURES)
    print(f"[external] of which index >= {MAX_STRUCTURES} (never loaded by the checkpoint): {outside_cutoff}")
    if len(selected) < args.min_n:
        print(f"[external] WARNING: {len(selected)} < requested {args.min_n}")

    # score with the shipped model under the deterministic default
    import torch

    from app.services.predictor import GNNPredictorService

    predictor = GNNPredictorService.get_instance()
    y, mu, sigma = [], [], []
    for _idx, _mid, struct in selected:
        res = predictor.predict(struct)
        y.append(labels[_mid])
        mu.append(res["predicted_formation_energy_per_atom_eV"])
        sigma.append(res["evidential_std_eV"])
    y = np.asarray(y)
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    score = np.abs(y - mu) / np.maximum(sigma, 1e-8)
    covered = score <= Q_SHIPPED
    k, n = int(covered.sum()), len(covered)
    lo, hi = wilson(k, n)
    abs_err = np.abs(y - mu)

    payload = {
        "definition": (
            "Pb + (I/Br/Cl) structures with Materials Project PBE formation energy whose "
            "material index is absent from the production checkpoint's soap_loco train "
            "indices; the pool spans the whole 129k-record file, so most entries also lie "
            "beyond the checkpoint's data cutoff (i < 50000)"
        ),
        "n_beyond_checkpoint_data_cutoff": outside_cutoff,
        "checkpoint": "paper_A7_soap_loco_formation_energy_per_atom/best.pt",
        "inference": "deterministic default (MCDropout off, model.eval())",
        "q_used": Q_SHIPPED,
        "n": n,
        "n_requested_min": args.min_n,
        "mae_eV_per_atom": round(float(abs_err.mean()), 4),
        "median_abs_error_eV_per_atom": round(float(np.median(abs_err)), 4),
        "rmse_eV_per_atom": round(float(np.sqrt((abs_err ** 2).mean())), 4),
        "coverage": round(float(covered.mean()), 4),
        "coverage_ci95": [round(lo, 4), round(hi, 4)],
        "median_half_width_eV": round(float(Q_SHIPPED * np.median(sigma)), 4),
        "mean_half_width_eV": round(float(Q_SHIPPED * sigma.mean()), 4),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{args.out_prefix}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with open(OUT / f"{args.out_prefix}.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["material_id", "target_eV_per_atom", "predicted_eV_per_atom",
                    "abs_error_eV", "sigma_eV", "score", "covered_at_q_1p0254"])
        for (idx, mid, _s), yy, mm, ss, sc, cv in zip(selected, y, mu, sigma, score, covered):
            w.writerow([mid, f"{yy:.6f}", f"{mm:.6f}", f"{abs(yy - mm):.6f}",
                        f"{ss:.6f}", f"{sc:.6f}", int(cv)])
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
