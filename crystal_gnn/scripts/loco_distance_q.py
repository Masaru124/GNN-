"""Distance-conditional q prototype (item 5).

SOAP fingerprints with the split-building params (r_cut=6, n_max=9, l_max=9,
sigma=0.5, average="inner", species = full union of the pool exactly like
soap_loco_split with soap_max_species=None) for every labeled structure with
index < 50000 (the LOCO folds trained with --max_structures 50000). Each
1-D COO descriptor (average="inner" already averages over atoms) is reduced to
a 512-dim count-sketch embedding (seed 42, deterministic per worker), then the
cosine distance to the nearest structure in each fold's train indices
(split[c]['train'] with i < 50000, val excluded) is computed for every scored
LOCO test point.

Then, NESTED across folds (q for fold c fitted on the other 9 folds only):
  * distance-bin conditional q (10 bins on pooled other-fold distances)
  * coverage by distance bin, fold-7 coverage, median half-width vs the
    global nested q_LOFO and shipped q (1.0254).

Outputs: research/loco_cross_conformal/distance_q.json
Caches: crystal_gnn/data/cache/soap_fingerprint512_seed42.npz

Run from repo root:
    .venv311/Scripts/python.exe crystal_gnn/scripts/loco_distance_q.py
"""

from __future__ import annotations

import csv
import gzip
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "crystal_gnn"
LABELS = ROOT / "data" / "raw" / "mp_labels.csv"
STRUCTURES = ROOT / "data" / "raw" / "mp_structures.json.gz"
SPLITS = ROOT / "data" / "splits" / "soap_loco.json"
LOCO_DIR = REPO / "materials-screening-ai" / "research" / "loco_cross_conformal"
CACHE = ROOT / "data" / "cache" / "soap_fingerprint512_seed42.npz"
OUT = LOCO_DIR / "distance_q.json"
MAX_STRUCTURES = 50000  # LOCO folds trained with --max_structures 50000
Q_SHIPPED = 1.0254
COVERAGE = 0.90
N_BINS = 10
EMBED_DIM = 512
SEED = 42
N_WORKERS = 16
STAGE1 = 4000  # pre-filter chunk size

_SOAP = None
_SPECIES = None
_H = None
_SGN = None
_DIM = None


def _init_worker(species):
    global _SOAP, _SPECIES, _H, _SGN, _DIM
    _SPECIES = set(species)
    _H = None
    _SGN = None
    _DIM = None
    from dscribe.descriptors import SOAP
    _SOAP = SOAP(species=species, r_cut=6.0, n_max=9, l_max=9, sigma=0.5,
                 periodic=True, sparse=True, average="inner")


def _embed(v):
    """Count-sketch a dense descriptor to EMBED_DIM dims, L2-normalised."""
    global _H, _SGN, _DIM
    if _H is None:
        _DIM = int(v.size)
        rng = np.random.default_rng(SEED)
        _H = rng.integers(0, EMBED_DIM, size=_DIM, dtype=np.int32)
        _SGN = (rng.integers(0, 2, size=_DIM, dtype=np.int8) * 2 - 1).astype(np.float64)
    p = np.bincount(_H, weights=_SGN * v, minlength=EMBED_DIM)[:EMBED_DIM]
    n = float(np.linalg.norm(p))
    if n > 0:
        p = p / n
    return p.astype(np.float32)


def _process(chunk):
    """chunk: list of (idx, record). Returns (idx_list, vectors[N,EMBED_DIM])."""
    from pymatgen.core import Structure
    from pymatgen.io.ase import AseAtomsAdaptor
    out_idx, out_vec = [], []
    for idx, rec in chunk:
        els = {s["label"] for s in rec["structure"]["sites"]}
        if not els <= _SPECIES:
            continue
        try:
            s = Structure.from_dict(rec["structure"])
            v = _SOAP.create(AseAtomsAdaptor.get_atoms(s), n_jobs=1)
            # average="inner" already yields ONE vector per structure (1-D COO)
            if hasattr(v, "todense"):
                v = v.todense()
            if hasattr(v, "toarray"):
                v = v.toarray()
            v = np.asarray(v, dtype=np.float64).reshape(-1)
            out_idx.append(idx)
            out_vec.append(_embed(v))
        except Exception:
            continue
    return out_idx, (np.stack(out_vec) if out_vec else np.empty((0, EMBED_DIM), np.float32))


def get_vectors():
    if CACHE.exists():
        d = np.load(CACHE)
        return d["idx"], d["X"]
    t0 = time.time()
    labels = set()
    for row in csv.DictReader(open(LABELS, newline="", encoding="utf-8")):
        fe = row.get("formation_energy_per_atom")
        if fe not in (None, "", "None"):
            labels.add(row["material_id"])
    with gzip.open(STRUCTURES, "rt", encoding="utf-8") as fh:
        records = json.load(fh)

    splits = json.loads(SPLITS.read_text(encoding="utf-8"))["splits"]
    needed = set()
    for s in splits:
        for key in ("train", "val", "test"):
            needed.update(i for i in s[key] if i < MAX_STRUCTURES)
    needed = sorted(needed)

    species = set()
    recs = {}
    for i in needed:
        rec = records[i]
        recs[i] = rec
        species.update(s["label"] for s in rec["structure"]["sites"])
    species = sorted(species)  # full union, exactly like soap_loco_split (no cap)
    print(f"[soap] needed={len(needed)} species={len(species)}", flush=True)
    del records

    items = list(recs.items())
    del recs
    chunks = [items[i:i + STAGE1] for i in range(0, len(items), STAGE1)]
    del items

    all_idx, all_vec = [], []
    with mp.Pool(N_WORKERS, initializer=_init_worker, initargs=(species,)) as pool:
        for n, (idxs, vecs) in enumerate(pool.imap_unordered(_process, chunks, chunksize=1)):
            all_idx.extend(idxs)
            if len(vecs):
                all_vec.append(vecs)
            if (n + 1) % 5 == 0:
                print(f"[soap] chunk {n+1}/{len(chunks)} valid={len(all_idx)} "
                      f"elapsed={time.time()-t0:.0f}s", flush=True)
    X = np.concatenate(all_vec) if all_vec else np.empty((0, EMBED_DIM), np.float32)
    idx = np.array(all_idx, dtype=np.int64)
    order = np.argsort(idx)
    idx, X = idx[order], X[order]
    np.savez_compressed(CACHE, idx=idx, X=X)
    print(f"[soap] done: valid={len(idx)} in {time.time()-t0:.0f}s -> {CACHE}", flush=True)
    return idx, X


def main() -> None:
    t0 = time.time()
    idx_arr, X = get_vectors()
    pos = {int(i): n for n, i in enumerate(idx_arr)}

    splits = json.loads(SPLITS.read_text(encoding="utf-8"))["splits"]
    # match loco_cross_conformal.load_folds(): training indices are i < 50000
    train_sets = [{i for i in s["train"] if i < MAX_STRUCTURES} for s in splits]

    with gzip.open(STRUCTURES, "rt", encoding="utf-8") as fh:
        records = json.load(fh)
    idx_of = {}
    for i, rec in enumerate(records):
        mid = rec.get("material_id")
        if mid:
            idx_of[mid] = i
    del records

    order, scores_a, dist_a, sigma_a, missing = [], {}, {}, {}, {}
    for npz in sorted(LOCO_DIR.glob("fold*_scores.npz")):
        c = int(npz.stem.replace("fold", "").replace("_scores", ""))
        d = dict(np.load(npz, allow_pickle=True))
        ids = [str(m) for m in d["material_ids"]]
        mask = np.array([idx_of.get(m) in pos for m in ids])

        tr_rows = [pos[i] for i in train_sets[c] if i in pos]
        T = X[tr_rows]
        Qv = X[[pos[idx_of[m]] for m, mk in zip(ids, mask) if mk]]
        dmin = np.empty(len(Qv), dtype=np.float32)
        bs = 512
        for s in range(0, len(Qv), bs):
            sim = Qv[s:s + bs] @ T.T
            dmin[s:s + bs] = 1.0 - sim.max(axis=1)

        order.append(c)
        scores_a[c] = d["score"][mask]
        dist_a[c] = dmin
        sigma_a[c] = d["sigma"][mask]
        missing[c] = int((~mask).sum())
        print(f"[knn] fold {c}: train_vecs={len(T)} q={len(Qv)} missing={missing[c]} "
              f"median_d={np.median(dmin):.4f} {time.time()-t0:.0f}s", flush=True)
    order.sort()

    pooled_d = np.concatenate([dist_a[c] for c in order])
    pooled_s = np.concatenate([scores_a[c] for c in order])

    # ---- shipped-q coverage by distance quintile -------------------------
    qcuts = np.quantile(pooled_d, [0.2, 0.4, 0.6, 0.8, 1.0])
    edges5 = np.concatenate([[np.nextafter(pooled_d.min(), -np.inf)], qcuts])
    bid = np.clip(np.searchsorted(edges5, pooled_d, side="right") - 1, 0, 4)
    by_bin_ship = []
    for b in range(5):
        m = bid == b
        if not m.any():
            by_bin_ship.append({"bin": b, "n": 0})
            continue
        by_bin_ship.append({
            "bin": b,
            "d_max": round(float(edges5[b + 1]), 4),
            "n": int(m.sum()),
            "coverage_at_shipped_q": round(float((pooled_s[m] <= Q_SHIPPED).mean()), 4),
            "median_score": round(float(np.median(pooled_s[m])), 4),
        })

    # ---- nested distance-bin conditional q (10 bins) ---------------------
    per_fold_cov, all_cover, all_hw = {}, [], []
    bin_cov_pool, bin_n = {b: [] for b in range(N_BINS)}, {b: 0 for b in range(N_BINS)}
    for c in order:
        o_d = np.concatenate([dist_a[o] for o in order if o != c])
        o_s = np.concatenate([scores_a[o] for o in order if o != c])
        edges = np.unique(np.quantile(o_d, np.linspace(0, 1, N_BINS + 1)))
        if len(edges) < 3:
            edges = np.array([-np.inf, np.inf])
        oidx = np.clip(np.searchsorted(edges, o_d, side="right") - 1, 0, len(edges) - 2)
        qidx = np.clip(np.searchsorted(edges, dist_a[c], side="right") - 1, 0, len(edges) - 2)
        q_glob = float(np.quantile(o_s, COVERAGE))
        qs = np.empty(len(edges) - 1)
        for b in range(len(edges) - 1):
            sb = o_s[oidx == b]
            qs[b] = float(np.quantile(sb, COVERAGE)) if len(sb) >= 100 else q_glob
        qb = qs[qidx]
        cov = scores_a[c] <= qb
        per_fold_cov[c] = round(float(cov.mean()), 4)
        all_cover.append(cov)
        all_hw.append(qb * sigma_a[c])
        for b in range(len(edges) - 1):
            m = qidx == b
            if m.any():
                bin_cov_pool[b].append(cov[m])
                bin_n[b] += int(m.sum())
    # pooled shipped-q coverage per distance decile (for the table)
    dec_edges = np.unique(np.quantile(pooled_d, np.linspace(0, 1, N_BINS + 1)))
    dec_id = np.clip(np.searchsorted(dec_edges, pooled_d, side="right") - 1, 0, len(dec_edges) - 2)
    bin_rows = []
    for b in sorted(bin_cov_pool):
        nested_cov = (float(np.concatenate(bin_cov_pool[b]).mean())
                      if bin_cov_pool[b] else None)
        m = dec_id == b
        bin_rows.append({
            "bin": b,
            "n": int(bin_n[b]),
            "nested_cov": round(nested_cov, 4) if nested_cov is not None else None,
            "coverage_at_shipped_q": round(float((pooled_s[m] <= Q_SHIPPED).mean()), 4)
            if m.any() else None,
            "median_score": round(float(np.median(pooled_s[m])), 4) if m.any() else None,
        })

    # global nested q_LOFO on the same aligned points
    lofo = {c: float(np.quantile(
        np.concatenate([scores_a[o] for o in order if o != c]), COVERAGE)) for c in order}
    cov_lofo = {c: round(float((scores_a[c] <= lofo[c]).mean()), 4) for c in order}
    hw_lofo = float(np.median(np.concatenate([lofo[c] * sigma_a[c] for c in order])))

    out = {
        "method": ("SOAP (r_cut=6, n_max=9, l_max=9, sigma=0.5, average=inner, full species "
                   "union like soap_loco_split) -> count-sketch 512d (seed 42) -> cosine "
                   "distance to nearest structure in the fold's train indices (i<50000, val "
                   "excluded). Distance bins = pooled other-fold deciles; per-bin q = 90% "
                   "quantile of other folds' scores (nested)."),
        "n_vectors": int(len(idx_arr)),
        "n_test_points_total": int(sum(len(scores_a[c]) for c in order)),
        "n_test_points_expected": int(sum(
            len(np.load(LOCO_DIR / f"fold{c}_scores.npz", allow_pickle=True)["score"])
            for c in order)),
        "missing_per_fold": {str(c): missing[c] for c in order},
        "coverage_by_distance_quintile_at_shipped_q": by_bin_ship,
        "median_score_by_distance_quintile": [b.get("median_score") for b in by_bin_ship],
        "nested_distance_bin_q": {
            "pooled_cov": round(float(np.concatenate(all_cover).mean()), 4),
            "macro_cov": round(float(np.mean([per_fold_cov[c] for c in order])), 4),
            "per_fold_cov": {str(c): per_fold_cov[c] for c in order},
            "fold7_cov": per_fold_cov.get(7),
            "median_half_width_eV": round(float(np.median(np.concatenate(all_hw))), 4),
            "coverage_by_distance_bin": bin_rows,
        },
        "global_nested_q_lofo_same_points": {
            "per_fold_q": {str(c): round(lofo[c], 4) for c in order},
            "pooled_cov": round(float(np.concatenate(
                [(scores_a[c] <= lofo[c]) for c in order]).mean()), 4),
            "fold7_cov": cov_lofo[7],
            "median_half_width_eV": round(hw_lofo, 4),
        },
        "shipped_q": {
            "pooled_cov": round(float((pooled_s <= Q_SHIPPED).mean()), 4),
            "fold7_cov": round(float((scores_a[7] <= Q_SHIPPED).mean()), 4),
        },
        "elapsed_s": round(time.time() - t0, 1),
    }
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
