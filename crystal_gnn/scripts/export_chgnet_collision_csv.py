"""
Export Full CSV for 177 Colliding Pairs Evaluated with CHGNet and Select Spot-Check Samples.
"""

import sys
import os
import csv
import itertools
import time
import numpy as np
import torch
from collections import defaultdict
from pymatgen.core import Structure
from chgnet.model import CHGNet

sys.path.append(os.path.join(os.getcwd(), "scripts"))
from collision_scan_full import load_dataset_subsampled

OUT_CSV = "results/chgnet_collision_audit_177_pairs.csv"
SPOTCHECK_DIR = "results/spotcheck_cifs"

def main():
    print("--- GENERATING FULL CHGNET COLLISION AUDIT CSV (177 PAIRS) ---", flush=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(SPOTCHECK_DIR, exist_ok=True)

    # 1. Load dataset (subsampled with seed 0)
    dataset = load_dataset_subsampled(50000)
    max_radius = 4.0
    threshold = 0.05
    max_neighbors = 24

    all_structures_neighs = []
    for idx, (structure, _) in enumerate(dataset):
        neighs = structure.get_all_neighbors(r=max_radius, include_index=True)
        all_structures_neighs.append(neighs)

    targets = [t for _, t in dataset]
    radii = [2.0, 2.5, 3.0, 3.5, 4.0]
    
    all_unique_pairs = set()

    for r in radii:
        descriptors = []
        for idx, (structure, _) in enumerate(dataset):
            neighs_max = all_structures_neighs[idx]
            per_atom_descriptors = []
            for i, neighs in enumerate(neighs_max):
                sorted_neighs = [n for n in neighs if n.nn_distance <= r]
                sorted_neighs = sorted(sorted_neighs, key=lambda n: n.nn_distance)[:max_neighbors]
                neigh_tuples = []
                for n in sorted_neighs:
                    species_num = int(n.specie.number)
                    distance_bin = int(round(n.nn_distance / 0.05))
                    neigh_tuples.append((species_num, distance_bin))
                neigh_tuples.sort()
                central_species = int(structure[i].specie.number)
                per_atom_descriptors.append((central_species, tuple(neigh_tuples)))
            per_atom_descriptors.sort()
            descriptors.append(tuple(per_atom_descriptors))

        buckets = defaultdict(list)
        for idx, desc in enumerate(descriptors):
            buckets[desc].append(idx)

        for desc, idxs in buckets.items():
            if len(idxs) < 2:
                continue
            for i, j in itertools.combinations(idxs, 2):
                diff = abs(targets[i] - targets[j])
                if diff > threshold:
                    pair_tuple = (min(i, j), max(i, j))
                    all_unique_pairs.add(pair_tuple)

    sorted_pairs = sorted(list(all_unique_pairs))
    print(f"Total Unique Colliding Pairs: {len(sorted_pairs)}", flush=True)

    # Collect unique structure indices needed
    unique_struct_indices = sorted(list(set(itertools.chain(*sorted_pairs))))
    print(f"Unique Structures: {len(unique_struct_indices)}", flush=True)

    # 2. Run CHGNet inference on all structures
    chgnet = CHGNet.load()
    chgnet_preds = {}

    for count, idx in enumerate(unique_struct_indices):
        struct, _ = dataset[idx]
        res = chgnet.predict_structure(struct)
        chgnet_preds[idx] = float(res['e'])

    # 3. Build CSV rows
    rows = []
    fully_resolved_pairs = []
    partially_resolved_pairs = []
    collapsed_pairs = []

    for pair_idx, (i, j) in enumerate(sorted_pairs):
        struct_i, target_i = dataset[i]
        struct_j, target_j = dataset[j]

        id_i = getattr(struct_i, "material_id", f"idx_{i}")
        id_j = getattr(struct_j, "material_id", f"idx_{j}")

        formula_i = struct_i.formula
        formula_j = struct_j.formula

        true_gap = abs(target_i - target_j)
        pred_i = chgnet_preds[i]
        pred_j = chgnet_preds[j]
        pred_gap = abs(pred_i - pred_j)

        ratio = pred_gap / true_gap if true_gap > 0 else 0

        if ratio >= 0.50 or pred_gap >= 0.025:
            classification = "Fully Resolved"
            fully_resolved_pairs.append((pair_idx, i, j, id_i, id_j, formula_i, target_i, target_j, true_gap, pred_i, pred_j, pred_gap))
        elif ratio >= 0.10:
            classification = "Partially Resolved"
            partially_resolved_pairs.append((pair_idx, i, j, id_i, id_j, formula_i, target_i, target_j, true_gap, pred_i, pred_j, pred_gap))
        else:
            classification = "Collapsed"
            collapsed_pairs.append((pair_idx, i, j, id_i, id_j, formula_i, target_i, target_j, true_gap, pred_i, pred_j, pred_gap))

        rows.append({
            "pair_index": pair_idx + 1,
            "struct_idx_1": i,
            "struct_idx_2": j,
            "mp_id_1": id_i,
            "mp_id_2": id_j,
            "formula_1": formula_i,
            "formula_2": formula_j,
            "target_energy_1": target_i,
            "target_energy_2": target_j,
            "true_energy_gap": true_gap,
            "chgnet_pred_1": pred_i,
            "chgnet_pred_2": pred_j,
            "chgnet_pred_gap": pred_gap,
            "resolution_ratio": ratio,
            "classification": classification,
        })

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[DONE] Full 177-pair CSV written to: {OUT_CSV}", flush=True)
    print(f"Summary: Fully Resolved={len(fully_resolved_pairs)}, Partially Resolved={len(partially_resolved_pairs)}, Collapsed={len(collapsed_pairs)}")

    # 4. Select 5 spot-check pairs
    # Pick 2 Fully Resolved, 2 Collapsed, 1 Partially Resolved
    spot_check_candidates = [
        ("Fully Resolved #1", fully_resolved_pairs[0]),
        ("Fully Resolved #2", fully_resolved_pairs[len(fully_resolved_pairs)//2]),
        ("Collapsed #1", collapsed_pairs[0]),
        ("Collapsed #2", collapsed_pairs[len(collapsed_pairs)//2]),
        ("Partially Resolved #1", partially_resolved_pairs[0]),
    ]

    print("\n" + "=" * 90, flush=True)
    print("SELECTED 5 SPOT-CHECK PAIRS FOR INDEPENDENT VERIFICATION:")
    print("=" * 90, flush=True)

    spotcheck_info = []
    for label, item in spot_check_candidates:
        pair_idx, i, j, id_i, id_j, form_i, target_i, target_j, true_gap, pred_i, pred_j, pred_gap = item
        struct_i, _ = dataset[i]
        struct_j, _ = dataset[j]

        file_i = os.path.join(SPOTCHECK_DIR, f"spot_{pair_idx}_1_{id_i}.cif")
        file_j = os.path.join(SPOTCHECK_DIR, f"spot_{pair_idx}_2_{id_j}.cif")

        struct_i.to(filename=file_i)
        struct_j.to(filename=file_j)

        spotcheck_info.append({
            "label": label,
            "id_1": id_i, "id_2": id_j,
            "formula": form_i,
            "file_1": file_i, "file_2": file_j,
            "target_1": target_i, "target_2": target_j, "true_gap": true_gap,
            "expected_pred_1": pred_i, "expected_pred_2": pred_j, "expected_pred_gap": pred_gap
        })

        print(f"[{label}] Pair #{pair_idx+1}: {form_i} ({id_i} vs {id_j})")
        print(f"  Targets: {target_i:.6f} vs {target_j:.6f} -> True Gap = {true_gap:.6f} eV/atom")
        print(f"  CHGNet Preds: {pred_i:.6f} vs {pred_j:.6f} -> Pred Gap = {pred_gap:.6f} eV/atom")
        print(f"  Saved CIFs: {file_i}, {file_j}")
        print("-" * 90, flush=True)

if __name__ == "__main__":
    main()
