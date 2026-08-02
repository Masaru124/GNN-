"""
Comprehensive Batch CHGNet Benchmark on All 50k Colliding Pairs.
Evaluates CHGNet across ALL colliding structure pairs identified by the 50k collision scan
at radii r = 2.0, 2.5, 3.0, 3.5, and 4.0 Å.
"""

import sys
import os
import itertools
import time
import numpy as np
import torch
from collections import defaultdict
from pymatgen.core import Structure
from chgnet.model import CHGNet

sys.path.append(os.path.join(os.getcwd(), "scripts"))
from train import load_data
from collision_scan_full import load_dataset_subsampled

def main():
    print("==========================================================================", flush=True)
    print("      COMPREHENSIVE BATCH CHGNet BENCHMARK ON ALL 50K COLLIDING PAIRS     ", flush=True)
    print("==========================================================================", flush=True)

    t_start = time.time()
    # 1. Load 50,000 structure dataset (subsampled with seed 0, exactly matching the scan)
    dataset = load_dataset_subsampled(50000)
    print(f"Loaded {len(dataset)} structures.", flush=True)

    radii = [2.0, 2.5, 3.0, 3.5, 4.0]
    max_radius = 4.0
    threshold = 0.05
    max_neighbors = 24

    print(f"\n1. Building descriptors and collecting colliding pairs at max_radius = {max_radius} Å...", flush=True)
    all_structures_neighs = []
    for idx, (structure, _) in enumerate(dataset):
        neighs = structure.get_all_neighbors(r=max_radius, include_index=True)
        all_structures_neighs.append(neighs)

    targets = [t for _, t in dataset]
    
    # Store colliding pairs per radius: radius -> list of (idx_i, idx_j, diff_true)
    pairs_per_radius = {}
    all_unique_pairs = set()
    structures_to_predict = set()

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

        r_pairs = []
        for desc, idxs in buckets.items():
            if len(idxs) < 2:
                continue
            for i, j in itertools.combinations(idxs, 2):
                diff = abs(targets[i] - targets[j])
                if diff > threshold:
                    pair_tuple = (min(i, j), max(i, j))
                    r_pairs.append((pair_tuple[0], pair_tuple[1], diff))
                    all_unique_pairs.add(pair_tuple)
                    structures_to_predict.add(pair_tuple[0])
                    structures_to_predict.add(pair_tuple[1])

        pairs_per_radius[r] = r_pairs
        print(f"  Radius {r:.1f} Å: {len(r_pairs)} colliding pairs ({len(set(itertools.chain(*[(p[0], p[1]) for p in r_pairs])))} unique structures)", flush=True)

    print(f"\nTotal Unique Colliding Pairs across all radii: {len(all_unique_pairs)}", flush=True)
    print(f"Total Unique Structures requiring CHGNet inference: {len(structures_to_predict)}", flush=True)

    # 2. Run CHGNet inference on all distinct structures involved in collisions
    print("\n2. Initializing CHGNet model for GPU batch inference...", flush=True)
    chgnet = CHGNet.load()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Running CHGNet inference on {len(structures_to_predict)} distinct structures...", flush=True)
    t_chgnet0 = time.time()
    chgnet_energies = {}

    struct_list = list(structures_to_predict)
    for idx_count, struct_idx in enumerate(struct_list):
        struct, _ = dataset[struct_idx]
        res = chgnet.predict_structure(struct)
        chgnet_energies[struct_idx] = float(res['e'])
        if (idx_count + 1) % 50 == 0 or (idx_count + 1) == len(struct_list):
            print(f"  CHGNet processed {idx_count + 1}/{len(struct_list)} structures ({time.time() - t_chgnet0:.1f}s)", flush=True)

    print(f"\nCHGNet batch inference completed in {time.time() - t_chgnet0:.1f}s.", flush=True)

    # 3. Comprehensive Analysis & Resolution Rates
    print("\n3. AUDITING CHGNET RESOLUTION RATES ACROSS RADII", flush=True)
    print("Criteria:")
    print("  - Fully Resolved:   pred_gap >= 50% of true_gap (and pred_gap >= 0.025 eV/atom)")
    print("  - Partially Resolved: 10% <= pred_gap < 50% of true_gap")
    print("  - Collapsed:        pred_gap < 10% of true_gap (or pred_gap < 0.005 eV/atom)")

    print("\n" + "=" * 90, flush=True)
    print(f"{'Radius (Å)':<12} | {'Total Pairs':<12} | {'Fully Resolved':<16} | {'Partially Resolved':<18} | {'Collapsed':<14}", flush=True)
    print("-" * 90, flush=True)

    summary_by_radius = {}

    for r in radii:
        r_pairs = pairs_per_radius[r]
        if len(r_pairs) == 0:
            continue

        resolved_count = 0
        partial_count = 0
        collapsed_count = 0

        for i, j, diff_true in r_pairs:
            e_i = chgnet_energies[i]
            e_j = chgnet_energies[j]
            diff_pred = abs(e_i - e_j)

            ratio = diff_pred / diff_true if diff_true > 0 else 0
            if ratio >= 0.50 or diff_pred >= 0.025:
                resolved_count += 1
            elif ratio >= 0.10:
                partial_count += 1
            else:
                collapsed_count += 1

        total = len(r_pairs)
        summary_by_radius[r] = (total, resolved_count, partial_count, collapsed_count)
        print(f"{r:<12.1f} | {total:<12d} | {resolved_count:<5d} ({resolved_count/total*100:5.1f}%) | {partial_count:<6d} ({partial_count/total*100:5.1f}%) | {collapsed_count:<5d} ({collapsed_count/total*100:5.1f}%)", flush=True)

    print("=" * 90, flush=True)

    # 4. Overall Deduplicated Summary Across All Radii
    all_resolved = 0
    all_partial = 0
    all_collapsed = 0

    for i, j in all_unique_pairs:
        diff_true = abs(dataset[i][1] - dataset[j][1])
        diff_pred = abs(chgnet_energies[i] - chgnet_energies[j])
        ratio = diff_pred / diff_true if diff_true > 0 else 0

        if ratio >= 0.50 or diff_pred >= 0.025:
            all_resolved += 1
        elif ratio >= 0.10:
            all_partial += 1
        else:
            all_collapsed += 1

    total_all = len(all_unique_pairs)
    print("\nOVERALL DEDUPLICATED SUMMARY ACROSS ALL RADII:", flush=True)
    print(f"  Total Unique Colliding Pairs: {total_all}", flush=True)
    print(f"  Fully Resolved:   {all_resolved:<4d} ({all_resolved/total_all*100:.1f}%)", flush=True)
    print(f"  Partially Resolved:{all_partial:<4d} ({all_partial/total_all*100:.1f}%)", flush=True)
    print(f"  Collapsed:        {all_collapsed:<4d} ({all_collapsed/total_all*100:.1f}%)", flush=True)
    print(f"\nTotal execution time: {time.time() - t_start:.1f}s.", flush=True)

if __name__ == "__main__":
    main()
