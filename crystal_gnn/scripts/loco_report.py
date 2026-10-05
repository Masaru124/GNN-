"""
Aggregate the LOCO cross-conformal fold score arrays into coverage tables.

Reads research/loco_cross_conformal/fold*_scores.npz (score = |y-mu|/sigma(DER),
one array per held-out soap_loco cluster) and reports:

  * per-fold coverage at the shipped q = 1.0254
  * per-fold cross-conformal coverage: q fitted on the OTHER folds, applied here
  * the shift-aware q that reaches >= 90% coverage in >= 80% of folds, with its
    pooled coverage and median half-width

Run from crystal_gnn/:
    python scripts/loco_report.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np

REPO = Path(__file__).resolve().parents[2]
LOCO_DIR = REPO / "materials-screening-ai" / "research" / "loco_cross_conformal"
OUT_JSON = LOCO_DIR / "cross_conformal_summary.json"
OUT_CSV = LOCO_DIR / "cross_conformal_per_fold.csv"
OUT_MD = LOCO_DIR / "cross_conformal_summary.md"
Q_SHIPPED = 1.0254
COVERAGE = 0.90
MIN_N = 500  # folds with fewer test points are excluded from the '>=90% of folds' count


def wilson(k: int, n: int, z: float = 1.959963985):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def load_folds():
    folds = {}
    for npz in sorted(LOCO_DIR.glob("fold*_scores.npz")):
        c = int(npz.stem.replace("fold", "").replace("_scores", ""))
        d = np.load(npz, allow_pickle=True)
        folds[c] = d
    return folds


def main() -> None:
    folds = load_folds()
    if not folds:
        raise SystemExit(f"no fold score arrays in {LOCO_DIR}")
    print(f"folds available: {sorted(folds)}")

    scores = {c: folds[c]["score"] for c in folds}
    sigma = {c: folds[c]["sigma"] for c in folds}
    pooled = np.concatenate([scores[c] for c in sorted(scores)])

    rows = []
    for c in sorted(scores):
        s = scores[c]
        others = np.concatenate([scores[o] for o in sorted(scores) if o != c]) if len(scores) > 1 else s
        q_cc = float(np.quantile(others, COVERAGE))
        k = int((s <= Q_SHIPPED).sum())
        k_cc = int((s <= q_cc).sum())
        lo, hi = wilson(k, len(s))
        lo_cc, hi_cc = wilson(k_cc, len(s))
        rows.append({
            "cluster": c,
            "n_test": int(len(s)),
            "test_mae_eV_per_atom": float(np.abs(folds[c]["target"] - folds[c]["mu"]).mean()),
            "median_half_width_eV_at_shipped_q": float(Q_SHIPPED * np.median(sigma[c])),
            "coverage_at_shipped_q_1p0254": round(k / len(s), 4),
            "ci95_at_shipped_q": [round(lo, 4), round(hi, 4)],
            "q_cross_conformal": round(q_cc, 4),
            "q_cross_conformal_exact": q_cc,
            "coverage_cross_conformal": round(k_cc / len(s), 4),
            "coverage_cross_conformal_exact": k_cc / len(s),
            "ci95_cross_conformal": [round(lo_cc, 4), round(hi_cc, 4)],
            "own_fold_q90": round(float(np.quantile(s, COVERAGE)), 4),
            "own_fold_q90_exact": float(np.quantile(s, COVERAGE)),
        })

    # shift-aware q: smallest q that reaches 90% coverage in >= 80% of folds.
    # Computed at FULL precision (rounding to 4 dp drops fold 3 below 90%);
    # rounding is applied only when printing.
    q_needed = sorted(r["own_fold_q90_exact"] for r in rows)
    n_folds = len(q_needed)
    k80 = max(1, math.ceil(0.8 * n_folds))
    q_shift = q_needed[k80 - 1]
    per_fold_shift = {}
    cov_shift_raw = {}
    for c in sorted(scores):
        cov_shift_raw[c] = float((scores[c] <= q_shift).mean())
        per_fold_shift[str(c)] = round(cov_shift_raw[c], 4)
    n_meeting = sum(1 for v in cov_shift_raw.values() if v >= COVERAGE)
    n_meeting_r4 = sum(1 for v in cov_shift_raw.values() if round(v, 4) >= COVERAGE)
    border = sorted(
        (c for c in cov_shift_raw if COVERAGE > cov_shift_raw[c] >= COVERAGE - 1e-3),
        key=lambda c: cov_shift_raw[c],
    )
    eligible = [c for c in sorted(scores) if len(scores[c]) >= MIN_N]
    n_meeting_eligible = sum(1 for c in eligible if cov_shift_raw[c] >= COVERAGE)
    cov_ship_raw = {c: float((scores[c] <= Q_SHIPPED).mean()) for c in sorted(scores)}
    macro_ship = float(np.mean(list(cov_ship_raw.values())))
    macro_shift = float(np.mean(list(cov_shift_raw.values())))
    macro_shift_eligible = float(np.mean([cov_shift_raw[c] for c in eligible]))
    lofo_vals = [r["q_cross_conformal"] for r in rows]

    # nested (leave-one-fold-out) headline + pooled 10-fold 90% q
    nested_exact = {r["cluster"]: r["coverage_cross_conformal_exact"] for r in rows}
    nested_sorted = np.array([nested_exact[c] for c in sorted(nested_exact)])
    nested_pooled = float(sum(r["coverage_cross_conformal_exact"] * r["n_test"] for r in rows)
                          / sum(r["n_test"] for r in rows))
    nested_hw = float(np.median(np.concatenate(
        [r["q_cross_conformal"] * sigma[r["cluster"]] for r in rows])))
    q_pool = float(np.quantile(pooled, COVERAGE))  # non-nested single q over all 10 folds
    pool_cov = {c: float((scores[c] <= q_pool).mean()) for c in sorted(scores)}
    pool_sorted = np.array([pool_cov[c] for c in sorted(pool_cov)])
    pool_hw = float(np.median(q_pool * np.concatenate(
        [sigma[c] for c in sorted(sigma)])))
    pooled_at_shift = float((pooled <= q_shift).mean())
    med_half_shift = float(q_shift * np.median(np.concatenate([sigma[c] for c in sorted(sigma)])))

    summary = {
        "score_definition": "|y - mu| / sigma(DER), model.eval() deterministic forward",
        "q_shipped": Q_SHIPPED,
        "coverage_target": COVERAGE,
        "folds_available": sorted(scores),
        "n_folds": n_folds,
        "pooled_n": int(len(pooled)),
        "pooled_coverage_at_shipped_q": round(float((pooled <= Q_SHIPPED).mean()), 4),
        "macro_coverage_at_shipped_q": round(macro_ship, 4),
        "per_fold_n": {str(c): int(len(scores[c])) for c in sorted(scores)},
        "q_nesting": (
            "q_cross_conformal for fold c = 90% quantile of the OTHER 9 folds' scores "
            "applied to fold c (leave-one-fold-out, nested). shift_aware_q is an "
            "oracle-style k-th smallest OWN-fold quantile, not nested."
        ),
        "nested_lofo_result": {
            "role": "PRIMARY RESULT — nested leave-one-fold-out q fitted on the other 9 folds",
            "q_spread": {
                "min": round(min(lofo_vals), 4),
                "median": round(float(np.median(lofo_vals)), 4),
                "max": round(max(lofo_vals), 4),
            },
            "pooled_coverage": round(nested_pooled, 4),
            "macro_coverage": round(float(nested_sorted.mean()), 4),
            "per_fold_coverage": {
                "min": round(float(nested_sorted.min()), 4),
                "q1": round(float(np.quantile(nested_sorted, 0.25)), 4),
                "median": round(float(np.median(nested_sorted)), 4),
                "q3": round(float(np.quantile(nested_sorted, 0.75)), 4),
                "max": round(float(nested_sorted.max()), 4),
                "worst_fold": int(min(nested_exact, key=nested_exact.get)),
            },
            "median_half_width_eV": round(nested_hw, 4),
        },
        "pooled_q90_non_nested": {
            "q": round(q_pool, 4),
            "note": "90% quantile of all 10 folds pooled (each held-out fold contributes its own scores — not nested)",
            "per_fold_coverage": {
                "min": round(float(pool_sorted.min()), 4),
                "q1": round(float(np.quantile(pool_sorted, 0.25)), 4),
                "median": round(float(np.median(pool_sorted)), 4),
                "q3": round(float(np.quantile(pool_sorted, 0.75)), 4),
                "max": round(float(pool_sorted.max()), 4),
            },
            "macro_coverage": round(float(pool_sorted.mean()), 4),
            "median_half_width_eV": round(pool_hw, 4),
        },
        "lofo_q_spread": {
            "min": round(min(lofo_vals), 4),
            "median": round(float(np.median(lofo_vals)), 4),
            "max": round(max(lofo_vals), 4),
        },
        "shift_aware_q": round(q_shift, 4),
        "shift_aware_q_role": (
            "NON-NESTED ORACLE (k-th smallest own-fold 90% quantile) — reported only as an "
            "oracle bound, not cited as a result and not adopted"
        ),
        "shift_aware_q_exact": q_shift,
        "shift_aware_q_definition": f"{k80}-th smallest per-fold 90% quantile (>=80% of folds reach 90%), full precision",
        "pooled_coverage_at_shift_aware_q": round(pooled_at_shift, 4),
        "macro_coverage_at_shift_aware_q": round(macro_shift, 4),
        "folds_meeting_90pct_at_shift_aware_q": n_meeting,
        "folds_meeting_90pct_at_shift_aware_q_display_precision": n_meeting_r4,
        "borderline_folds_below_90pct_within_1e-3": {
            str(c): round(cov_shift_raw[c], 6) for c in border
        },
        "folds_excluded_n_lt_500": [c for c in sorted(scores) if c not in eligible],
        "folds_meeting_90pct_at_shift_aware_q_excluding_n_lt_500": f"{n_meeting_eligible}/{len(eligible)}",
        "macro_coverage_at_shift_aware_q_excluding_n_lt_500": round(macro_shift_eligible, 4),
        "coverage_at_shift_aware_q_per_fold": per_fold_shift,
        "median_half_width_eV_at_shift_aware_q": round(med_half_shift, 4),
        "per_fold": rows,
        "caveat": (
            "Each fold's model is trained on the other nine clusters only for the folds "
            "retrained here; fold 0 is the production checkpoint (also soap_loco fold 0). "
            "Folds share an identical wall-clock budget, so their budgets are equal but "
            "shorter than the production run."
        ),
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    with open(OUT_CSV, "w", encoding="utf-8", newline="") as fh:
        import csv

        w = csv.writer(fh)
        w.writerow([
            "cluster", "n_test", "test_mae_eV_per_atom", "median_half_width_eV_at_shipped_q",
            "coverage_at_shipped_q_1p0254", "ci95_low", "ci95_high",
            "q_cross_conformal", "coverage_cross_conformal", "cc_ci95_low", "cc_ci95_high",
            "own_fold_q90", "coverage_at_shift_aware_q",
        ])
        for r in rows:
            w.writerow([
                r["cluster"], r["n_test"], r["test_mae_eV_per_atom"],
                r["median_half_width_eV_at_shipped_q"], r["coverage_at_shipped_q_1p0254"],
                r["ci95_at_shipped_q"][0], r["ci95_at_shipped_q"][1],
                r["q_cross_conformal"], r["coverage_cross_conformal"],
                r["ci95_cross_conformal"][0], r["ci95_cross_conformal"][1],
                r["own_fold_q90"], per_fold_shift[str(r["cluster"])],
            ])

    md = [
        "# LOCO cross-conformal summary (formation energy)",
        "",
        f"Score: `{summary['score_definition']}`. Shipped q = {Q_SHIPPED}.",
        "",
        "| cluster | n_test | MAE (eV/atom) | median half-width (eV) | coverage @1.0254 | 95% CI | q cross-conformal | coverage cross-conformal | coverage @ shift-aware q |",
        "| ---: | ---: | ---: | ---: | ---: | :---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        md.append(
            f"| {r['cluster']} | {r['n_test']} | {r['test_mae_eV_per_atom']:.4f} | "
            f"{r['median_half_width_eV_at_shipped_q']:.4f} | {r['coverage_at_shipped_q_1p0254']:.4f} | "
            f"[{r['ci95_at_shipped_q'][0]:.4f}, {r['ci95_at_shipped_q'][1]:.4f}] | "
            f"{r['q_cross_conformal']:.4f} | {r['coverage_cross_conformal']:.4f} | "
            f"{per_fold_shift[str(r['cluster'])]:.4f} |"
        )
    md += [
        "",
        f"Pooled coverage at shipped q: **{summary['pooled_coverage_at_shipped_q']:.4f}** (n = {summary['pooled_n']}); macro-average (unweighted over folds) **{macro_ship:.4f}**.",
        f"**Nested (leave-one-fold-out) result** — per-fold q fitted on the other 9 folds only: "
        f"q spread min {summary['lofo_q_spread']['min']:.4f} (cluster 7), median {summary['lofo_q_spread']['median']:.4f}, "
        f"max {summary['lofo_q_spread']['max']:.4f} (cluster 0); pooled coverage **{nested_pooled:.4f}**, "
        f"macro **{float(nested_sorted.mean()):.4f}**; per-fold min {float(nested_sorted.min()):.4f} (cluster {int(min(nested_exact, key=nested_exact.get))}), "
        f"Q1 {float(np.quantile(nested_sorted, 0.25)):.4f}, median {float(np.median(nested_sorted)):.4f}, "
        f"Q3 {float(np.quantile(nested_sorted, 0.75)):.4f}, max {float(nested_sorted.max()):.4f}; "
        f"median half-width **{nested_hw:.4f} eV**.",
        f"Pooled 10-fold 90% q = **{q_pool:.4f}** (non-nested single q over all 10 folds): "
        f"per-fold coverage min {float(pool_sorted.min()):.4f} / Q1 {float(np.quantile(pool_sorted, 0.25)):.4f} / "
        f"median {float(np.median(pool_sorted)):.4f} / Q3 {float(np.quantile(pool_sorted, 0.75)):.4f}, "
        f"macro {float(pool_sorted.mean()):.4f}, width {pool_hw:.4f} eV; its nested leave-one-fold-out form is exactly the per-fold q above.",
        f"Oracle only (NOT a result, not adopted): shift-aware own-fold q = {summary['shift_aware_q']:.4f} "
        f"gives pooled {summary['pooled_coverage_at_shift_aware_q']:.4f} / macro {macro_shift:.4f} "
        f"({n_meeting}/{n_folds} folds ≥90% raw, {n_meeting_eligible}/{len(eligible)} excl. n<{MIN_N}) "
        f"at width {summary['median_half_width_eV_at_shift_aware_q']:.4f} eV — a non-nested own-fold statistic.",
        f"Nesting: {summary['q_nesting']}",
        f"LOFO (leave-one-fold-out) q spread: min {summary['lofo_q_spread']['min']:.4f}, "
        f"median {summary['lofo_q_spread']['median']:.4f}, max {summary['lofo_q_spread']['max']:.4f}.",
        "",
        f"Caveat: {summary['caveat']}",
        "",
    ]
    OUT_MD.write_text("\n".join(md), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "per_fold"}, indent=2))
    print("\n".join(md))


if __name__ == "__main__":
    main()
