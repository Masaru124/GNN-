"""
Independent audit of results/chgnet_collision_audit_177_pairs.csv.
Run this directly against the real file -- don't trust the summary table
until this passes cleanly.

Usage:
    python audit_collision_csv.py results/chgnet_collision_audit_177_pairs.csv
"""

import sys
import csv
from collections import defaultdict


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "results/chgnet_collision_audit_177_pairs.csv"

    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))

    print(f"Loaded {len(rows)} rows from {path}")
    if len(rows) != 177:
        print(f"!! MISMATCH: expected 177 rows, found {len(rows)} -- investigate before trusting anything else.")

    # 1. Recompute resolution_ratio and re-derive classification independently
    mismatches = []
    label_counts = defaultdict(int)
    for r in rows:
        true_gap = float(r["true_energy_gap"])
        pred_gap = float(r["chgnet_pred_gap"])
        stated_ratio = float(r["resolution_ratio"])
        recomputed_ratio = pred_gap / true_gap if true_gap != 0 else float("nan")

        if abs(recomputed_ratio - stated_ratio) > 1e-4:
            mismatches.append((r["pair_index"], "ratio", stated_ratio, recomputed_ratio))

        # Re-derive label from ratio using the stated thresholds
        # (>=0.5 Fully Resolved, 0.1-0.5 Partially Resolved, <0.1 Collapsed)
        if recomputed_ratio >= 0.5 or pred_gap >= 0.025:
            expected_label = "Fully Resolved"
        elif recomputed_ratio >= 0.1:
            expected_label = "Partially Resolved"
        else:
            expected_label = "Collapsed"

        if expected_label != r["classification"]:
            mismatches.append((r["pair_index"], "label", r["classification"], expected_label))

        label_counts[r["classification"]] += 1

    print(f"\nRatio/label mismatches: {len(mismatches)}")
    for m in mismatches[:20]:
        print(" ", m)
    if len(mismatches) > 20:
        print(f"  ... and {len(mismatches) - 20} more")

    # 2. Recompute aggregate percentages independently
    total = len(rows)
    print("\nIndependently recomputed aggregate breakdown:")
    for label in ["Fully Resolved", "Partially Resolved", "Collapsed"]:
        n = label_counts[label]
        print(f"  {label:20s}: {n:3d} ({100*n/total:.1f}%)")

    # 3. Check for duplicate / degenerate pairs (same two structure indices listed twice,
    #    in either order)
    seen = defaultdict(list)
    for r in rows:
        key = tuple(sorted([r["struct_idx_1"], r["struct_idx_2"]]))
        seen[key].append(r["pair_index"])
    dupes = {k: v for k, v in seen.items() if len(v) > 1}
    print(f"\nDuplicate structure-pairs (same 2 indices listed >once): {len(dupes)}")
    for k, v in list(dupes.items())[:10]:
        print(" ", k, "-> pair_index rows:", v)

    # 4. Sanity check: same structure appearing in many pairs should belong to one
    #    consistent composition (a real polymorph cluster), not mixed formulas
    #    (which would indicate a bucketing/hashing bug in the original collision scan)
    struct_formula = {}
    conflicts = []
    for r in rows:
        for idx_col, formula_col in [("struct_idx_1", "formula_1"), ("struct_idx_2", "formula_2")]:
            idx = r[idx_col]
            formula = r[formula_col]
            if idx in struct_formula and struct_formula[idx] != formula:
                conflicts.append((idx, struct_formula[idx], formula))
            struct_formula[idx] = formula
    print(f"\nStructure-index formula conflicts (same idx, different formula across rows): {len(conflicts)}")
    for c in conflicts[:10]:
        print(" ", c)

    print("\nIf mismatches/duplicates/conflicts are all zero, the CSV is internally "
          "consistent and safe to cite. Any nonzero count above needs investigation "
          "before this goes into the paper or supplementary material.")


if __name__ == "__main__":
    main()
