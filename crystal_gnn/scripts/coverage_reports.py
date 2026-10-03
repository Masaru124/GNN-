"""
Coverage reports for the LOCO study.

Subcommands (run from crystal_gnn/):
  n-cal      per-chemistry-class calibration counts on the production val split
  bandgap    leave-one-cluster-out cross-conformal coverage for the band-gap
             heuristic (no training: the estimator is analytic, so every fold is
             chemically clean)
  external   external halide validation set: Pb / halide-perovskite structures with
             MP PBE formation energies, verified absent from the checkpoint's train
             indices, scored with the production GNN

Writes JSON + CSV under materials-screening-ai/research/coverage_reports/.
"""

from __future__ import annotations

import argparse
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
Q_SHIPPED = 1.0254          # GNN conformal q (production val, soap_loco)
Q_BANDGAP_SHIPPED = 5.5833  # band-gap heuristic q (production val)
COVERAGE = 0.90
N_CLUSTERS = 10
MAX_STRUCTURES = 50000


def wilson(k: int, n: int, z: float = 1.959963985) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def load_all():
    """structures, labels, dataset (packaged class), fold list, split hashes."""
    from train import load_data  # noqa: E402
    from loco_cross_conformal import (  # noqa: E402
        MultiScaleCollate,
        MultiScaleDataset,
        load_folds,
    )

    structures, labels = load_data(str(ROOT / "data" / "raw"))
    structures = structures[:MAX_STRUCTURES]
    ds = MultiScaleDataset(
        structures=structures,
        labels=labels,
        radii=[4.0, 6.0, 8.0],
        target="formation_energy_per_atom",
        cache_dir=str(ROOT / "data" / "cache"),
        max_neighbors=24,
    )
    folds = load_folds(MAX_STRUCTURES)
    return structures, labels, ds, folds


def dataset_material_ids(ds):
    """material_id per dataset index (entries are appended in index order)."""
    return [mid for mid, _ in ds._entries]


def dataset_formulas(ds):
    """material_id -> reduced formula, taken from the built pymatgen structures."""
    out = {}
    for mid, s in ds._entries:
        try:
            out[mid] = s.composition.reduced_formula
        except Exception:
            continue
    return out


def n_cal_command():
    from app.services.chemistry_class import CLASSES, classify

    structures, labels, ds, folds = load_all()
    mids = dataset_material_ids(ds)
    formulas = dataset_formulas(ds)
    mapping = ds.orig_to_dataset_idx
    counts = {c: 0 for c in CLASSES}
    for i in folds[0]["val"]:
        di = mapping.get(i)
        if di is None:
            continue
        cls = classify(formulas.get(mids[di]))
        counts[cls] = counts.get(cls, 0) + 1
    total = sum(counts.values())
    payload = {"source": "production val split (soap_loco fold 0, i<50000)", "total": total, "n_cal_by_class": counts}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "n_cal_by_class.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print("\nPaste into app/services/chemistry_class.py N_CAL_TABLE:")
    print(json.dumps(counts, indent=4))


def bandgap_command():
    """LOCO cross-conformal for the band-gap heuristic (analytic estimator)."""
    from bandgap_estimator import BandGapEstimatorService

    structures, labels, ds, folds = load_all()
    mids = dataset_material_ids(ds)
    mapping = ds.orig_to_dataset_idx
    est = BandGapEstimatorService.get_instance()

    fold_scores, fold_rows = {}, []
    for c, fold in enumerate(folds):
        scores, halves = [], []
        for i in fold["test"]:
            di = mapping.get(i)
            if di is None:
                continue
            mid = mids[di]
            lab = labels[mid]
            gap = lab.get("band_gap")
            if gap is None:
                continue
            from pymatgen.core import Structure  # noqa: E402

            s = structures[i]
            if isinstance(s, dict):
                s = Structure.from_dict(s)
            res = est.estimate_band_gap(s)
            e = res["estimated_band_gap_eV"]
            sig = float(res.get("estimation_error_1sigma_eV") or 0.0)
            if sig <= 0:
                continue
            scores.append(abs(float(gap) - float(e)) / sig)
            halves.append(res.get("conformal_90_interval_eV"))
        fold_scores[c] = np.asarray(scores, dtype=np.float64)
        folded_halves = [h for h in halves if h]
        med_half = float(np.median([(h[1] - h[0]) / 2.0 for h in folded_halves])) if folded_halves else float("nan")
        fold_rows.append({
            "cluster": c,
            "n_scored": len(scores),
            "median_interval_half_width_eV": round(med_half, 4),
            "score_q90": round(float(np.quantile(scores, COVERAGE)), 4) if len(scores) else None,
        })
        print(f"[bandgap] cluster {c}: n={len(scores)} q90={fold_rows[-1]['score_q90']} half={fold_rows[-1]['median_interval_half_width_eV']}")

    # cross-conformal: q_c from the other folds, applied to fold c
    cross_rows = []
    pooled = np.concatenate([fold_scores[c] for c in range(N_CLUSTERS) if len(fold_scores[c])])
    for c in range(N_CLUSTERS):
        others = np.concatenate([fold_scores[o] for o in range(N_CLUSTERS) if o != c and len(fold_scores[o])]) \
            if any(len(fold_scores[o]) for o in range(N_CLUSTERS) if o != c) else np.asarray([])
        if not len(others) or not len(fold_scores[c]):
            continue
        q_c = float(np.quantile(others, COVERAGE))
        cov = float((fold_scores[c] <= q_c).mean())
        lo, hi = wilson(int((fold_scores[c] <= q_c).sum()), len(fold_scores[c]))
        cross_rows.append({
            "cluster": c,
            "n_test": len(fold_scores[c]),
            "q_cross_conformal": round(q_c, 4),
            "coverage": round(cov, 4),
            "ci95": [round(lo, 4), round(hi, 4)],
            "coverage_at_shipped_q": round(float((fold_scores[c] <= Q_BANDGAP_SHIPPED).mean()), 4),
        })
        print(f"[bandgap][cross] cluster {c}: q={q_c:.4f} cov={cov:.4f} CI=[{lo:.4f},{hi:.4f}]")

    q_needed = sorted(float(np.quantile(fold_scores[c], COVERAGE)) for c in range(N_CLUSTERS) if len(fold_scores[c]))
    n_folds = len(q_needed)
    k = math.ceil(0.8 * n_folds)
    q_shift = q_needed[k - 1] if k <= n_folds else q_needed[-1]
    per_fold_at_shift = {
        c: round(float((fold_scores[c] <= q_shift).mean()), 4)
        for c in range(N_CLUSTERS) if len(fold_scores[c])
    }
    n_meeting = sum(1 for v in per_fold_at_shift.values() if v >= COVERAGE)
    halves = [r["median_interval_half_width_eV"] for r in fold_rows if r["median_interval_half_width_eV"] == r["median_interval_half_width_eV"]]
    payload = {
        "estimator": "band-gap heuristic (electronegativity / matgl fallback)",
        "coverage_target": COVERAGE,
        "shipped_q": Q_BANDGAP_SHIPPED,
        "per_fold": fold_rows,
        "cross_conformal": cross_rows,
        "pooled_coverage_at_shipped_q": round(float((pooled <= Q_BANDGAP_SHIPPED).mean()), 4),
        "shift_aware_q": round(q_shift, 4),
        "shift_aware_q_definition": f"{k}-th smallest per-fold 90% quantile (>=80% of folds reach 90%)",
        "coverage_at_shift_aware_q_per_fold": per_fold_at_shift,
        "folds_meeting_90pct_at_shift_aware_q": n_meeting,
        "n_folds": n_folds,
        "median_interval_half_width_eV_across_folds": round(float(np.median(halves)), 4) if halves else None,
        "pooled_n": int(len(pooled)),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "bandgap_loco_cross_conformal.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


def external_command(min_n: int = 100):
    """External Pb/halide validation set, verified outside the training indices."""
    import torch
    from torch.utils.data import Subset
    from torch_geometric.loader import DataLoader

    from app.services.chemistry_class import classify
    from loco_cross_conformal import MultiScaleCollate, score_indices

    structures, labels, ds, folds = load_all()
    mids = dataset_material_ids(ds)
    formulas = dataset_formulas(ds)
    mapping = ds.orig_to_dataset_idx
    train_ids = {
        mids[mapping[i]]
        for i in folds[0]["train"]
        if i in mapping
    }
    print(f"[external] train material ids: {len(train_ids)}")

    selected = []
    for mid, lab in labels.items():
        if mid in train_ids or mid not in formulas:
            continue
        f = formulas[mid]
        if lab.get("formation_energy_per_atom") is None:
            continue
        if classify(f) != "pb_halide":
            continue
        selected.append((mid, f))
    print(f"[external] Pb/halide materials outside train: {len(selected)}")

    idx_by_mid = {mid: di for di, mid in enumerate(mids)}
    usable = [(mid, f, idx_by_mid[mid]) for mid, f in selected if mid in idx_by_mid]
    print(f"[external] usable (in filtered dataset): {len(usable)}")
    if len(usable) < min_n:
        print(f"[external] WARNING: only {len(usable)} < {min_n} requested")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = ROOT / "checkpoints" / "paper_A7_soap_loco_formation_energy_per_atom" / "best.pt"
    ids, y, mu, sigma, score = score_indices(ckpt, ds, [u[2] for u in usable], device)
    abs_err = np.abs(y - mu)
    covered = score <= Q_SHIPPED
    k, n = int(covered.sum()), len(covered)
    lo, hi = wilson(k, n)
    payload = {
        "definition": "Pb-containing halide materials with MP PBE formation energy, material_id absent from the production checkpoint train indices",
        "checkpoint": "paper_A7_soap_loco_formation_energy_per_atom/best.pt (deterministic eval)",
        "q_used": Q_SHIPPED,
        "n": n,
        "n_requested_min": min_n,
        "mae_eV_per_atom": round(float(abs_err.mean()), 4),
        "median_abs_error_eV_per_atom": round(float(np.median(abs_err)), 4),
        "rmse_eV_per_atom": round(float(np.sqrt((abs_err ** 2).mean())), 4),
        "coverage": round(float(covered.mean()), 4),
        "coverage_ci95": [round(lo, 4), round(hi, 4)],
        "median_half_width_eV": round(float(Q_SHIPPED * np.median(sigma)), 4),
        "n_missing_from_train_check": len(train_ids),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "external_halide_validation.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with open(OUT / "external_halide_validation.csv", "w", encoding="utf-8", newline="") as fh:
        import csv

        w = csv.writer(fh)
        w.writerow(["material_id", "formula", "target_eV_per_atom", "predicted_eV_per_atom",
                    "abs_error_eV", "sigma_eV", "score", "covered_at_q"])
        for (mid, f, _), yy, mm, ss, sc, cv in zip(usable, y, mu, sigma, score, covered):
            w.writerow([mid, f, f"{yy:.6f}", f"{mm:.6f}", f"{abs(yy - mm):.6f}", f"{ss:.6f}", f"{sc:.6f}", int(cv)])
    print(json.dumps(payload, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=["n-cal", "bandgap", "external"])
    ap.add_argument("--min-n", type=int, default=100)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    {"n-cal": n_cal_command, "bandgap": bandgap_command, "external": external_command}[args.command](
        **({"min_n": args.min_n} if args.command == "external" else {})
    )


if __name__ == "__main__":
    main()
