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
from pathlib import Path

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
            "coverage_cross_conformal": round(k_cc / len(s), 4),
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
        "lofo_q_spread": {
            "min": round(min(lofo_vals), 4),
            "median": round(float(np.median(lofo_vals)), 4),
            "max": round(max(lofo_vals), 4),
        },
        "shift_aware_q": round(q_shift, 4),
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
        f"Pooled coverage at shipped q: **{summary['pooled_coverage_at_shipped_q']:.4f}** (n = {summary['pooled_n']}); "
        f"macro-average (unweighted over folds) **{macro_ship:.4f}**.",
        f"Shift-aware q: **{summary['shift_aware_q']:.4f}** ({summary['shift_aware_q_definition']}); "
        f"pooled coverage {summary['pooled_coverage_at_shift_aware_q']:.4f}, "
        f"macro-average {macro_shift:.4f}, "
        f"{n_meeting}/{n_folds} folds at or above 90% (raw, full-precision q; "
        f"{n_meeting_r4}/{n_folds} at display precision"
        + (f" — borderline: " + ", ".join(
            f"cluster {c} = {cov_shift_raw[c]:.5f}" for c in border) if border else "")
        + f"), {n_meeting_eligible}/{len(eligible)} excluding folds with n < {MIN_N} "
        f"(cluster {','.join(str(x) for x in summary['folds_excluded_n_lt_500'])}), "
        f"median half-width {summary['median_half_width_eV_at_shift_aware_q']:.4f} eV.",
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
