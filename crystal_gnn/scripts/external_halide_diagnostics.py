"""External Pb-halide set diagnostics (item 4).

Reports:
  * n and composition of the external set (halogen breakdown, top elements)
  * overlap with the production training set and with each LOCO fold's
    train/val/test indices (index-level and material-id level)
  * how many Pb-halide structures the model DID see in training
  * coverage decomposition: why 0.9277 at q=1.0254 externally vs 0.7124 pooled LOCO
    (error vs sigma scale, median half-width, median score)

Run from repo root:
    .venv311/Scripts/python.exe crystal_gnn/scripts/external_halide_diagnostics.py
"""

from __future__ import annotations

import csv
import gzip
import json
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "crystal_gnn"
LABELS = ROOT / "data" / "raw" / "mp_labels.csv"
STRUCTURES = ROOT / "data" / "raw" / "mp_structures.json.gz"
SPLITS = ROOT / "data" / "splits" / "soap_loco.json"
LOCO_DIR = REPO / "materials-screening-ai" / "research" / "loco_cross_conformal"
EXT_CSV = REPO / "materials-screening-ai" / "research" / "coverage_reports" / "external_halide_validation.csv"
EXT_JSON = REPO / "materials-screening-ai" / "research" / "coverage_reports" / "external_halide_validation.json"
OUT = LOCO_DIR / "external_halide_diagnostics.json"
Q = 1.0254
MAX_STRUCTURES = 50000
PB_HALOGENS = {"I", "Br", "Cl"}
HALOGENS = {"F", "Cl", "Br", "I"}


def main() -> None:
    labels = set()
    for row in csv.DictReader(open(LABELS, newline="", encoding="utf-8")):
        fe = row.get("formation_energy_per_atom")
        if fe not in (None, "", "None"):
            labels.add(row["material_id"])

    splits = json.loads(SPLITS.read_text(encoding="utf-8"))["splits"]
    train_sets = [{i for i in s["train"]} for s in splits]
    val_sets = [{i for i in s["val"]} for s in splits]
    test_sets = [{i for i in s["test"]} for s in splits]

    # LOCO fold npz ids
    npz_ids = {}
    for npz in sorted(LOCO_DIR.glob("fold*_scores.npz")):
        c = int(npz.stem.replace("fold", "").replace("_scores", ""))
        npz_ids[c] = set(str(x) for x in np.load(npz, allow_pickle=True)["material_ids"])

    with gzip.open(STRUCTURES, "rt", encoding="utf-8") as fh:
        records = json.load(fh)

    # index -> elements (only for labeled records)
    idx_of = {}
    elems_of = {}
    pb_halide_idx = []
    for idx, rec in enumerate(records):
        mid = rec.get("material_id")
        if mid not in labels:
            continue
        idx_of[mid] = idx
        els = {s["label"] for s in rec["structure"]["sites"]}
        elems_of[idx] = els
        if "Pb" in els and (els & PB_HALOGENS):
            pb_halide_idx.append(idx)

    # external set membership (same rule as external_halide_validation.py)
    ext_rows = list(csv.DictReader(open(EXT_CSV, newline="", encoding="utf-8")))
    ext_ids = [r["material_id"] for r in ext_rows]
    ext_idx = [idx_of[m] for m in ext_ids if m in idx_of]

    ext_el = Counter()
    ext_hal = Counter()
    for i in ext_idx:
        e = elems_of[i]
        ext_el.update(e)
        for h in e & PB_HALOGENS:
            ext_hal[h] += 1

    # overlap with each fold's train/val/test (index level, raw + i<50000 cutoff)
    overlap = {}
    for c in range(10):
        tr50 = {i for i in train_sets[c] if i < MAX_STRUCTURES}
        va50 = {i for i in val_sets[c] if i < MAX_STRUCTURES}
        te50 = {i for i in test_sets[c] if i < MAX_STRUCTURES}
        overlap[str(c)] = {
            "in_train": len(set(ext_idx) & train_sets[c]),
            "in_val": len(set(ext_idx) & val_sets[c]),
            "in_split_test": len(set(ext_idx) & test_sets[c]),
            "in_npz_test_ids": len(set(ext_ids) & npz_ids[c]),
            "in_train_idx_lt_50000": len(set(ext_idx) & tr50),
            "in_val_idx_lt_50000": len(set(ext_idx) & va50),
            "in_split_test_idx_lt_50000": len(set(ext_idx) & te50),
        }
    ext_within = [i for i in ext_idx if i < MAX_STRUCTURES]
    cluster_of = {}
    for c in range(10):
        for i in test_sets[c]:
            cluster_of[i] = c
    within_membership = {
        "n_idx_lt_50000": len(ext_within),
        "in_train0_lt50000": len(set(ext_within) & {i for i in train_sets[0] if i < MAX_STRUCTURES}),
        "in_val0_lt50000": len(set(ext_within) & {i for i in val_sets[0] if i < MAX_STRUCTURES}),
        "cluster_of_within_cutoff": {
            str(c): sum(1 for i in ext_within if cluster_of.get(i) == c) for c in range(10)
        },
    }

    # how many Pb-halide structures training saw
    train0_ph = sum(1 for i in pb_halide_idx if i < MAX_STRUCTURES and i in train_sets[0])
    train0_ph_any = sum(1 for i in pb_halide_idx if i in train_sets[0])
    train_any_fold_ph = {str(c): sum(1 for i in pb_halide_idx if i in train_sets[c])
                         for c in range(10)}

    # Pb-halide coverage inside LOCO pooled test (chemistry-matched comparison)
    ext_cov = np.array([float(r["covered_at_q_1p0254"]) for r in ext_rows])
    ext_err = np.array([float(r["abs_error_eV"]) for r in ext_rows])
    ext_sig = np.array([float(r["sigma_eV"]) for r in ext_rows])
    ext_scr = np.array([float(r["score"]) for r in ext_rows])

    loco_err, loco_sig, loco_scr, loco_ph_cov, loco_ph_n = [], [], [], [], 0
    for c in sorted(npz_ids):
        d = np.load(LOCO_DIR / f"fold{c}_scores.npz", allow_pickle=True)
        loco_err.append(np.abs(d["target"] - d["mu"]))
        loco_sig.append(d["sigma"])
        loco_scr.append(d["score"])
        # per-point Pb-halide flags
        ids_c = [str(x) for x in d["material_ids"]]
        flags = [1 if (idx_of.get(m) is not None and "Pb" in elems_of.get(idx_of[m], set())
                       and elems_of[idx_of[m]] & PB_HALOGENS) else 0 for m in ids_c]
        f = np.array(flags)
        loco_ph_n += int(f.sum())
        loco_ph_cov.append(d["score"][f == 1] <= Q)
    le = np.concatenate(loco_err)
    ls = np.concatenate(loco_sig)
    lsc = np.concatenate(loco_scr)
    ph_cov = np.concatenate([a for a in loco_ph_cov if len(a)]) if loco_ph_cov else np.array([])

    ext_js = json.loads(EXT_JSON.read_text(encoding="utf-8"))
    out = {
        "n": len(ext_ids),
        "n_in_records_with_labels": len(ext_idx),
        "n_beyond_checkpoint_cutoff": ext_js.get("n_beyond_checkpoint_data_cutoff"),
        "composition": {
            "halogen_presence_counts": dict(ext_hal),
            "top_elements": [f"{e}:{v}" for e, v in ext_el.most_common(10)],
            "all_pb_halide_by_construction": True,
        },
        "overlap_with_folds": overlap,
        "overlap_summary": {
            "in_production_train0_idx_lt_50000": len(set(ext_idx) & {i for i in train_sets[0] if i < MAX_STRUCTURES}),
            "in_any_fold_train": {str(c): overlap[str(c)]["in_train"] for c in range(10)},
            "in_any_fold_split_test": sum(overlap[str(c)]["in_split_test"] for c in range(10)),
            "in_loco_npz_test_total": sum(overlap[str(c)]["in_npz_test_ids"] for c in range(10)),
            "within_checkpoint_data_cutoff": within_membership,
        },
        "training_set_ph_halides": {
            "total_pb_halide_in_labeled_pool": len(pb_halide_idx),
            "in_production_train_idx_lt_50000": train0_ph,
            "in_production_train_any_idx": train0_ph_any,
            "per_fold_train": train_any_fold_ph,
        },
        "coverage_decomposition": {
            "external": {
                "n": len(ext_ids),
                "mae": round(float(ext_err.mean()), 4),
                "median_err": round(float(np.median(ext_err)), 4),
                "median_sigma": round(float(np.median(ext_sig)), 4),
                "median_score": round(float(np.median(ext_scr)), 4),
                "mean_score": round(float(ext_scr.mean()), 4),
                "median_half_width_eV": round(float(np.median(Q * ext_sig)), 4),
                "coverage_at_Q": round(float(ext_cov.mean()), 4),
            },
            "loco_pooled": {
                "n": int(len(lsc)),
                "mae": round(float(le.mean()), 4),
                "median_err": round(float(np.median(le)), 4),
                "median_sigma": round(float(np.median(ls)), 4),
                "median_score": round(float(np.median(lsc)), 4),
                "mean_score": round(float(lsc.mean()), 4),
                "median_half_width_eV": round(float(np.median(Q * ls)), 4),
                "coverage_at_Q": round(float((lsc <= Q).mean()), 4),
            },
            "loco_pb_halide_subset": {
                "n": int(ph_cov.size),
                "coverage_at_Q": round(float(ph_cov.mean()), 4) if ph_cov.size else None,
            },
            "verdict_inputs": {
                "median_sigma_ratio_ext_over_loco": round(
                    float(np.median(ext_sig) / np.median(ls)), 3),
                "mae_ratio_ext_over_loco": round(float(le.mean() and ext_err.mean() / le.mean()), 3),
                "median_score_ext": round(float(np.median(ext_scr)), 4),
                "median_score_loco": round(float(np.median(lsc)), 4),
            },
        },
    }
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
