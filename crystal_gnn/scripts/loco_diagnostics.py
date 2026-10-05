"""LOCO cross-conformal diagnostics.

Answers, with numbers:
  1. Is per-fold cross-conformal q nested (fitted on the OTHER 9 folds only)?
     + LOFO q spread.
  2. Macro-average (unweighted) coverage at shipped and shift-aware q; folds
     reaching 90% excluding folds with n < 500.
  3. Fold 7 diagnosis: cluster composition (Pb/halide content), sigma-vs-error
     calibration, MAE vs other folds.
  4. Sigma-binned nested q vs global nested q vs shift-aware q: coverage/width
     tradeoff. Cluster-conditional (own-fold, non-nested) q as an oracle bound.
  5. Routing/screening thresholds (interval width >= 0.10 eV) recomputed at
     q = 1.0254 vs q = 1.8143 on the pooled LOCO sigma distribution.

Run from repo root:
    .venv311/Scripts/python.exe crystal_gnn/scripts/loco_diagnostics.py
"""

from __future__ import annotations

import gzip
import json
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
LOCO_DIR = REPO / "materials-screening-ai" / "research" / "loco_cross_conformal"
STRUCTURES = REPO / "crystal_gnn" / "data" / "raw" / "mp_structures.json.gz"
Q_SHIPPED = 1.0254
COVERAGE = 0.90
MIN_N = 500

HALOGENS = {"F", "Cl", "Br", "I"}
TRANSITION_METALS = set("Sc Ti V Cr Mn Fe Co Ni Cu Zn Y Zr Nb Mo Tc Ru Rh Pd Ag Cd "
                        "Hf Ta W Re Os Ir Pt Au Hg La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho "
                        "Er Tm Yb Lu Ac Th Pa U Np Pu".split())
ALKALINE_EARTH = {"Be", "Mg", "Ca", "Sr", "Ba", "Ra"}
OXIDE_ONLY = TRANSITION_METALS | ALKALINE_EARTH


def classify_elems(elems: set) -> str:
    """Same rule table as backend chemistry_class.classify, on element sets."""
    hal = elems & HALOGENS
    if hal and "Pb" in elems:
        return "pb_halide"
    if hal:
        return "halide_other"
    has_o = "O" in elems
    if has_o and (elems & TRANSITION_METALS):
        return "transition_metal_oxide"
    if has_o and (elems & ALKALINE_EARTH):
        return "alkaline_earth_oxide"
    if has_o:
        return "other_oxide"
    return "other"


def load_folds():
    folds = {}
    for npz in sorted(LOCO_DIR.glob("fold*_scores.npz")):
        c = int(npz.stem.replace("fold", "").replace("_scores", ""))
        folds[c] = dict(np.load(npz, allow_pickle=True))
    return folds


def wilson(k: int, n: int, z: float = 1.959963985):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def main() -> None:
    folds = load_folds()
    scores = {c: folds[c]["score"] for c in folds}
    sigma = {c: folds[c]["sigma"] for c in folds}
    mu = {c: folds[c]["mu"] for c in folds}
    tgt = {c: folds[c]["target"] for c in folds}
    ids = {c: folds[c]["material_ids"] for c in folds}
    order = sorted(folds)
    pooled = np.concatenate([scores[c] for c in order])
    pooled_sigma = np.concatenate([sigma[c] for c in order])
    n_total = len(pooled)

    out: dict = {}

    # ---- (1) nesting + LOFO q spread -------------------------------------
    lofo, own90 = {}, {}
    for c in order:
        others = np.concatenate([scores[o] for o in order if o != c])
        lofo[c] = float(np.quantile(others, COVERAGE))   # nested: other 9 folds
        own90[c] = float(np.quantile(scores[c], COVERAGE))  # own fold (oracle)
    out["nesting"] = {
        "definition": "q_c = 90% quantile of scores from the OTHER 9 folds, applied to fold c (verified in loco_report.py:66)",
        "q_lofo_per_fold": {str(c): round(lofo[c], 4) for c in order},
        "q_lofo_min": round(min(lofo.values()), 4),
        "q_lofo_median": round(float(np.median(list(lofo.values()))), 4),
        "q_lofo_max": round(max(lofo.values()), 4),
        "q_lofo_argmin": min(lofo, key=lofo.get),
        "q_lofo_argmax": max(lofo, key=lofo.get),
        "q_own_fold90_spread": [round(min(own90.values()), 4), round(max(own90.values()), 4)],
        "shift_aware_q_is_nested": False,
        "shift_aware_q_note": "q=1.8143 is the k-th smallest OWN-fold 90% quantile (oracle-style), NOT nested; the nested analogue is the LOFO range above.",
    }
    # nested coverage of each fold at its own LOFO q (sanity: should be ~>= 0.9 on average)
    lofo_cov = {c: float((scores[c] <= lofo[c]).mean()) for c in order}
    out["nesting"]["coverage_at_own_lofo_q"] = {str(c): round(lofo_cov[c], 4) for c in order}
    out["nesting"]["macro_coverage_at_lofo_q"] = round(float(np.mean(list(lofo_cov.values()))), 4)

    # ---- (2) macro coverage + n>=500 exclusion ---------------------------
    cov_ship = {c: float((scores[c] <= Q_SHIPPED).mean()) for c in order}
    # shift-aware q at FULL precision (8th smallest own-fold q90 of 10), matching loco_report.py
    q_shift = sorted(own90.values())[max(1, int(np.ceil(0.8 * len(order)))) - 1]
    cov_shift = {c: float((scores[c] <= q_shift).mean()) for c in order}
    eligible = [c for c in order if len(scores[c]) >= MIN_N]
    out["coverage"] = {
        "per_fold_n": {str(c): int(len(scores[c])) for c in order},
        "pooled_coverage_at_shipped_q": round(float((pooled <= Q_SHIPPED).mean()), 4),
        "macro_coverage_at_shipped_q": round(float(np.mean([cov_ship[c] for c in order])), 4),
        "pooled_coverage_at_shift_aware_q": round(float((pooled <= q_shift).mean()), 4),
        "shift_aware_q_exact": q_shift,
        "macro_coverage_at_shift_aware_q": round(float(np.mean([cov_shift[c] for c in order])), 4),
        "folds_ge_90_at_shift_aware_q": f"{sum(1 for c in order if cov_shift[c] >= COVERAGE)}/{len(order)}",
        "folds_excluded_n_lt_500": [c for c in order if c not in eligible],
        "folds_ge_90_at_shift_aware_q_excl_n_lt_500":
            f"{sum(1 for c in eligible if cov_shift[c] >= COVERAGE)}/{len(eligible)}",
        "folds_ge_90_at_shipped_q_excl_n_lt_500":
            f"{sum(1 for c in eligible if cov_ship[c] >= COVERAGE)}/{len(eligible)}",
        "macro_coverage_at_shift_aware_q_excl_n_lt_500":
            round(float(np.mean([cov_shift[c] for c in eligible])), 4),
        "worst_folds_at_shift_aware_q": sorted(
            ({"cluster": c, "n": int(len(scores[c])), "cov": round(cov_shift[c], 4)} for c in order),
            key=lambda r: r["cov"])[:3],
    }

    # ---- (3) fold 7 diagnosis -------------------------------------------
    wanted = set()
    for c in order:
        wanted.update(str(x) for x in ids[c])
    elem_of: dict = {}
    with gzip.open(STRUCTURES, "rt", encoding="utf-8") as fh:
        data = json.load(fh)
    for rec in data:
        mid = rec["material_id"]
        if mid in wanted:
            elem_of[mid] = {s["species"][0]["element"] for s in rec["structure"]["sites"]}
    del data

    comp = {}
    for c in order:
        classes = Counter()
        elems = Counter()
        missing = 0
        for mid in ids[c]:
            e = elem_of.get(str(mid))
            if e is None:
                missing += 1
                continue
            classes[classify_elems(e)] += 1
            elems.update(e)
        n = len(ids[c])
        comp[c] = {"n": n, "missing": missing, "classes": classes, "elems": elems}

    c7 = 7
    def comp_block(c):
        n = comp[c]["n"] or 1
        return {
            "n": comp[c]["n"],
            "pb_struct_pct": round(100.0 * comp[c]["elems"]["Pb"] / n, 1),
            "pb_halide_pct": round(100.0 * comp[c]["classes"]["pb_halide"] / n, 1),
            "halide_any_pct": round(
                100.0 * (comp[c]["classes"]["pb_halide"] + comp[c]["classes"]["halide_other"]) / n, 1),
            "class_pct": {k: round(100.0 * v / n, 1) for k, v in comp[c]["classes"].most_common()},
            "top_elements": [f"{e}:{v}" for e, v in comp[c]["elems"].most_common(8)],
        }

    # pooled other-fold composition
    other_elems = Counter()
    other_classes = Counter()
    other_n = 0
    for c in order:
        if c == c7:
            continue
        other_elems.update(comp[c]["elems"])
        other_classes.update(comp[c]["classes"])
        other_n += comp[c]["n"]

    calib = {}
    for c in order:
        err = np.abs(tgt[c] - mu[c])
        s = scores[c]
        calib[c] = {
            "mae": float(err.mean()),
            "rmse": float(np.sqrt((err ** 2).mean())),
            "median_sigma": float(np.median(sigma[c])),
            "median_err": float(np.median(err)),
            "mean_ratio_err_over_sigma": float((err / np.maximum(sigma[c], 1e-12)).mean()),
            "frac_err_le_1sigma": float((s <= 1.0).mean()),
            "score_p90": float(np.quantile(s, 0.90)),
            "score_p99": float(np.quantile(s, 0.99)),
            "cov_shipped": cov_ship[c],
        }

    out["fold7"] = {
        "composition_fold7": comp_block(c7),
        "composition_pooled_other9": {
            "n": other_n,
            "class_pct": {k: round(100.0 * v / other_n, 1) for k, v in other_classes.most_common()},
            "top_elements": [f"{e}:{v}" for e, v in other_elems.most_common(8)],
        },
        "per_fold_error_sigma": {str(c): {k: round(v, 4) for k, v in calib[c].items()} for c in order},
        "fold7_rank_by_mae": sorted(order, key=lambda c: -calib[c]["mae"]).index(c7) + 1,
        "fold7_rank_by_miscoverage_shipped": sorted(
            order, key=lambda c: cov_ship[c]).index(c7) + 1,
        "verdict_inputs": {
            "fold7_mae": round(calib[c7]["mae"], 4),
            "median_other_folds_mae": round(float(np.median([calib[c]["mae"] for c in order if c != c7])), 4),
            "fold7_median_sigma": round(calib[c7]["median_sigma"], 4),
            "median_other_folds_sigma": round(float(np.median([calib[c]["median_sigma"] for c in order if c != c7])), 4),
            "fold7_frac_err_le_1sigma": round(calib[c7]["frac_err_le_1sigma"], 4),
            "median_other_frac_err_le_1sigma": round(
                float(np.median([calib[c]["frac_err_le_1sigma"] for c in order if c != c7])), 4),
        },
    }

    # ---- (4) sigma-binned nested q --------------------------------------
    n_bins = 10
    bin_cov = {}
    all_cover = []
    all_hw = []
    for c in order:
        o_sig = np.concatenate([sigma[o] for o in order if o != c])
        o_sco = np.concatenate([scores[o] for o in order if o != c])
        edges = np.unique(np.quantile(o_sig, np.linspace(0, 1, n_bins + 1)))
        if len(edges) < 3:
            edges = np.array([0.0, np.inf])
        idx = np.clip(np.searchsorted(edges, sigma[c], side="right") - 1, 0, len(edges) - 2)
        oidx = np.clip(np.searchsorted(edges, o_sig, side="right") - 1, 0, len(edges) - 2)
        q_glob = float(np.quantile(o_sco, COVERAGE))
        qs = np.empty(len(edges) - 1)
        for b in range(len(edges) - 1):
            sb = o_sco[oidx == b]
            qs[b] = float(np.quantile(sb, COVERAGE)) if len(sb) >= 100 else q_glob
        qb = qs[idx]
        bin_cov[c] = float((scores[c] <= qb).mean())
        all_cover.append((scores[c] <= qb))
        all_hw.append(qb * sigma[c])
    pooled_bin_cov_val = float(np.concatenate(all_cover).mean())
    pooled_bin_hw_val = float(np.median(np.concatenate(all_hw)))

    global_hw_shipped = float(np.median(Q_SHIPPED * pooled_sigma))
    global_hw_shift = float(np.median(q_shift * pooled_sigma))
    lofo_hw = float(np.median(np.concatenate(
        [np.full(len(sigma[c]), lofo[c]) * sigma[c] for c in order])))
    oracle_hw = float(np.median(np.concatenate(
        [np.full(len(sigma[c]), own90[c]) * sigma[c] for c in order])))

    out["q_variants"] = {
        "global_nested_lofo_q": {
            "per_fold": {str(c): round(lofo[c], 4) for c in order},
            "macro_cov": round(float(np.mean(list(lofo_cov.values()))), 4),
            "pooled_cov": round(float(np.concatenate(
                [(scores[c] <= lofo[c]) for c in order]).mean()), 4),
            "median_half_width_eV": round(lofo_hw, 4),
        },
        "sigma_binned_nested_q_10bins": {
            "macro_cov": round(float(np.mean([bin_cov[c] for c in order])), 4),
            "per_fold_cov": {str(c): round(bin_cov[c], 4) for c in order},
            "pooled_cov": round(pooled_bin_cov_val, 4),
            "median_half_width_eV": round(pooled_bin_hw_val, 4),
        },
        "shift_aware_q_not_nested": {
            "q": round(q_shift, 4),
            "macro_cov": round(float(np.mean([cov_shift[c] for c in order])), 4),
            "pooled_cov": round(float((pooled <= q_shift).mean()), 4),
            "median_half_width_eV": round(global_hw_shift, 4),
        },
        "shipped_q_1p0254": {
            "macro_cov": round(float(np.mean([cov_ship[c] for c in order])), 4),
            "pooled_cov": round(float((pooled <= Q_SHIPPED).mean()), 4),
            "median_half_width_eV": round(global_hw_shipped, 4),
        },
        "cluster_conditional_oracle_not_nested": {
            "macro_cov": round(float(np.mean([float((scores[c] <= own90[c]).mean()) for c in order])), 4),
            "median_half_width_eV": round(oracle_hw, 4),
            "note": "own-fold 90% quantile per cluster; requires knowing the cluster at prediction time and uses held-out scores",
        },
    }

    # ---- (2b) nested per-fold coverage + pooled 10-fold 90% q -----------
    nested_cov = {c: float((scores[c] <= lofo[c]).mean()) for c in order}
    nv = np.array([nested_cov[c] for c in order])

    def stats(v: dict) -> dict:
        a = np.array(list(v.values()))
        return {
            "min": round(float(a.min()), 4),
            "q1": round(float(np.quantile(a, 0.25)), 4),
            "median": round(float(np.median(a)), 4),
            "q3": round(float(np.quantile(a, 0.75)), 4),
            "max": round(float(a.max()), 4),
            "fold7": round(float(v[7]), 4) if 7 in v else None,
            "argmin": int(min(v, key=v.get)),
        }

    out["nested_coverage_at_q_lofo"] = {
        "definition": "coverage of fold c at its own leave-one-fold-out q (fitted on the other 9 folds)",
        "stats": stats(nested_cov),
        "per_fold": {str(c): round(nested_cov[c], 4) for c in order},
        "pooled": round(float(np.concatenate(
            [(scores[c] <= lofo[c]) for c in order]).mean()), 4),
        "macro": round(float(nv.mean()), 4),
        "median_half_width_eV": round(lofo_hw, 4),
    }

    # pooled 10-fold 90% q: non-nested (all folds) + nested (other-9 pooled per fold)
    q_pool = float(np.quantile(pooled, COVERAGE))
    pool_naive = {c: float((scores[c] <= q_pool).mean()) for c in order}
    q_lopo, pool_nested = {}, {}
    for c in order:
        o = np.concatenate([scores[x] for x in order if x != c])
        q_lopo[c] = float(np.quantile(o, COVERAGE))
        pool_nested[c] = float((scores[c] <= q_lopo[c]).mean())
    lopo_hw = float(np.median(np.concatenate(
        [q_lopo[c] * sigma[c] for c in order])))
    out["pooled_10fold_q90"] = {
        "q_pooled_all_folds": round(q_pool, 4),
        "q_pooled_all_folds_note": "non-nested: fitted on all 10 folds incl. each held-out fold",
        "per_fold_cov_at_q_pooled": stats(pool_naive),
        "q_lopo_nested_per_fold": {str(c): round(q_lopo[c], 4) for c in order},
        "q_lopo_spread": [round(min(q_lopo.values()), 4), round(max(q_lopo.values()), 4)],
        "nested_coverage_stats": stats(pool_nested),
        "nested_coverage_per_fold": {str(c): round(pool_nested[c], 4) for c in order},
        "nested_pooled_cov": round(float(np.concatenate(
            [(scores[c] <= q_lopo[c]) for c in order]).mean()), 4),
        "nested_macro_cov": round(float(np.mean(list(pool_nested.values()))), 4),
        "nested_median_half_width_eV": round(lopo_hw, 4),
        "median_half_width_eV_at_q_pooled": round(float(np.median(q_pool * pooled_sigma)), 4),
    }

    # ---- (5) routing / screening thresholds -----------------------------
    def route(q: float) -> dict:
        width = 2 * q * pooled_sigma
        return {
            "q": q,
            "promote_rate_width_ge_0p10": round(float((width >= 0.10).mean()), 4),
            "median_interval_width_eV": round(float(np.median(width)), 4),
            "p90_interval_width_eV": round(float(np.quantile(width, 0.90)), 4),
            "uq_signal_mean_width_over_0p5": round(float(np.minimum(width / 0.5, 1.0).mean()), 4),
            "dft_priority_ge_0p15_rate_only_uq": round(
                float((0.35 * np.minimum(width / 0.5, 1.0) >= 0.15).mean()), 4),
        }

    out["routing_thresholds"] = {
        "rule": "multi_fidelity_orchestrator: promote_to_tier2 if interval_width = 2*q*sigma >= 0.10 eV",
        "confidence_tiers_note": "predictor.py sigma tiers (<0.15/<0.35) do not depend on q",
        "at_shipped_q": route(Q_SHIPPED),
        "at_shift_aware_q": route(q_shift),
        "at_pooled_q": route(q_pool),
        "promote_curve": {
            "thresholds_eV": [0.05, 0.075, 0.10, 0.125, 0.15, 0.175, 0.20, 0.25, 0.30, 0.40, 0.50],
            "at_q_1p0254": [round(float((2 * Q_SHIPPED * pooled_sigma >= t).mean()), 4)
                            for t in [0.05, 0.075, 0.10, 0.125, 0.15, 0.175, 0.20, 0.25, 0.30, 0.40, 0.50]],
            "at_pooled_q": [round(float((2 * q_pool * pooled_sigma >= t).mean()), 4)
                            for t in [0.05, 0.075, 0.10, 0.125, 0.15, 0.175, 0.20, 0.25, 0.30, 0.40, 0.50]],
        },
        "sigma_unit_gate": {
            "derivation": "2*q*sigma >= 0.10 at q=1.0254 <=> sigma >= 0.04876 eV; q-invariant",
            "promote_rate_sigma_ge_0.04876": round(float((pooled_sigma >= 0.10 / (2 * Q_SHIPPED)).mean()), 4),
        },
        "dft_uq_signal": {
            "current": "uq_signal = min(width/0.5, 1), width = 2*q*sigma; saturates at sigma >= 0.5/(2q)",
            "saturation_frac_at_q_1p0254": round(float((2 * Q_SHIPPED * pooled_sigma >= 0.5).mean()), 4),
            "saturation_frac_at_pooled_q": round(float((2 * q_pool * pooled_sigma >= 0.5).mean()), 4),
            "sigma_unit_equivalent": "min(sigma/0.2438, 1) — identical to current signal at q=1.0254, invariant to q",
        },
    }

    print(json.dumps(out, indent=2))
    (LOCO_DIR / "diagnostics.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {LOCO_DIR / 'diagnostics.json'}")


if __name__ == "__main__":
    main()
