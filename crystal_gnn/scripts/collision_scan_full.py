"""
Collision scan on a statistically robust sample of 50,000 structures
(matching the paper training set size).
Performs single-pass neighbor query at max_radius = 4.0 Å to speed up neighbor search by 8x.
"""

import sys
import os
import argparse
import itertools
import numpy as np
import time
import logging
from collections import defaultdict
from pymatgen.core import Structure
sys.path.append(os.path.join(os.getcwd(), "scripts"))
from train import load_data

logging.basicConfig(level=logging.WARNING)


def load_dataset_subsampled(n_samples):
    """
    Load raw structures, resolve material IDs, and subsample/convert to pymatgen Structures.
    """
    print("Loading raw dataset structures and labels...", flush=True)
    structures_raw, labels = load_data("data/raw")
    print(f"Loaded {len(structures_raw)} raw structures and {len(labels)} label entries.", flush=True)
    
    label_keys = list(labels.keys())
    
    def resolve_material_id(idx, labels, s_dict, label_keys):
        sid = s_dict.get("material_id")
        if sid is None and "properties" in s_dict:
            sid = s_dict["properties"].get("material_id")
        if sid is not None and str(sid) in labels:
            return str(sid)
        fallback = f"mp-fake-{idx}"
        if fallback in labels:
            return fallback
        if idx < len(label_keys):
            return label_keys[idx]
        return None

    print("Filtering valid structure indices...", flush=True)
    valid_indices = []
    mapped_targets = []
    for idx, s_dict in enumerate(structures_raw):
        mid = resolve_material_id(idx, labels, s_dict, label_keys)
        if mid is not None and mid in labels:
            valid_indices.append(idx)
            mapped_targets.append(labels[mid]["formation_energy_per_atom"])
            
    print(f"Total valid structures: {len(valid_indices)}", flush=True)
    
    # Subsample indices
    rng = np.random.default_rng(0)
    if len(valid_indices) > n_samples:
        print(f"Subsampling {n_samples} structures...", flush=True)
        sub_idxs = rng.choice(len(valid_indices), size=n_samples, replace=False)
        chosen_indices = [valid_indices[i] for i in sub_idxs]
        chosen_targets = [mapped_targets[i] for i in sub_idxs]
    else:
        chosen_indices = valid_indices
        chosen_targets = mapped_targets
        
    print("Converting chosen structures to pymatgen Structure objects...", flush=True)
    dataset = []
    t0 = time.time()
    for idx, target in zip(chosen_indices, chosen_targets):
        try:
            s_dict = structures_raw[idx]
            structure = Structure.from_dict(s_dict)
            dataset.append((structure, target))
        except Exception:
            continue
        if (len(dataset)) % 10000 == 0:
            print(f"  Converted {len(dataset)}/{len(chosen_indices)} structures ({time.time()-t0:.1f}s)", flush=True)
            
    print(f"Loaded {len(dataset)} valid structures for collision scan.", flush=True)
    return dataset


def build_descriptors_for_all_radii(dataset, radii, threshold=0.05):
    """
    Computes structural descriptors for all specified radii using a single-pass neighbor search at max_radius.
    Returns a dictionary of results mapping radius -> (collision_rate, n_pairs, example_pairs)
    """
    max_radius = max(radii)
    print(f"\nPerforming single-pass neighbor query at max_radius = {max_radius} Å...", flush=True)
    t0 = time.time()
    
    # Store neighbor lists at max_radius
    all_structures_neighs = []
    for idx, (structure, _) in enumerate(dataset):
        neighs = structure.get_all_neighbors(r=max_radius, include_index=True)
        all_structures_neighs.append(neighs)
        if (idx + 1) % 10000 == 0:
            print(f"  Queried neighbors for {idx+1}/{len(dataset)} structures ({time.time()-t0:.1f}s)", flush=True)

    results = {}
    max_neighbors = 24

    for r in sorted(radii):
        print(f"\nComputing collision rate at radius r = {r} Å...", flush=True)
        t_r0 = time.time()
        
        descriptors = []
        for idx, (structure, _) in enumerate(dataset):
            neighs_max = all_structures_neighs[idx]
            per_atom_descriptors = []
            
            for i, neighs in enumerate(neighs_max):
                # Filter neighbors for radius r
                sorted_neighs = [n for n in neighs if n.nn_distance <= r]
                # Sort neighbors by distance to keep closest 24
                sorted_neighs = sorted(sorted_neighs, key=lambda n: n.nn_distance)
                if len(sorted_neighs) > max_neighbors:
                    sorted_neighs = sorted_neighs[:max_neighbors]
                    
                # Species atomic number and discretized distance bin (0.05 A bins)
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

        # Perform collision detection
        targets = [t for _, t in dataset]
        buckets = defaultdict(list)
        for idx, desc in enumerate(descriptors):
            buckets[desc].append(idx)

        colliding_structures = set()
        example_pairs = []
        n_pairs = 0

        for desc, idxs in buckets.items():
            if len(idxs) < 2:
                continue
            for i, j in itertools.combinations(idxs, 2):
                diff = abs(targets[i] - targets[j])
                if diff > threshold:
                    n_pairs += 1
                    colliding_structures.add(i)
                    colliding_structures.add(j)
                    if len(example_pairs) < 5:
                        example_pairs.append((i, j, diff))

        collision_rate = len(colliding_structures) / len(dataset)
        results[r] = (collision_rate, n_pairs, example_pairs)
        print(f"  Radius {r} finished in {time.time()-t_r0:.1f}s. Rate: {collision_rate:.6f} ({n_pairs} pairs)", flush=True)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.05,
                         help="Min |target diff| (eV/atom) to count as a meaningful collision")
    parser.add_argument("--n_samples", type=int, default=50000)
    args = parser.parse_args()

    t_start = time.time()
    dataset = load_dataset_subsampled(args.n_samples)
    
    # We test range 2.0 to 4.0 in 0.5 Å steps to get the exact falling curve
    radii = [2.0, 2.5, 3.0, 3.5, 4.0]
    results = build_descriptors_for_all_radii(dataset, radii, args.threshold)

    print("\n" + "=" * 60, flush=True)
    print(f"      COLLISION SCAN RESULTS (N = {len(dataset)} structures)", flush=True)
    print("=" * 60, flush=True)
    print(f"{'Radius (Å)':>12} | {'Collision Rate':>18} | {'# Colliding Pairs':>18}", flush=True)
    print("-" * 60, flush=True)
    for r in sorted(radii):
        rate, n_pairs, _ = results[r]
        print(f"{r:>12.1f} | {rate:>18.6f} | {n_pairs:>18d}", flush=True)
    print("=" * 60, flush=True)

    # Print example pairs at radius 2.0 and 2.5
    for r in [2.0, 2.5, 3.0]:
        rate, n_pairs, examples = results[r]
        if n_pairs > 0:
            print(f"\nExample colliding pairs at radius {r} Å:", flush=True)
            for i, j, diff in examples[:3]:
                mid_i = dataset[i][0].properties.get("material_id", f"idx-{i}")
                mid_j = dataset[j][0].properties.get("material_id", f"idx-{j}")
                print(f"  {mid_i} vs {mid_j}: energy diff = {diff:.6f} eV/atom", flush=True)
                
    print(f"\nTotal script execution time: {time.time()-t_start:.1f} seconds.", flush=True)


if __name__ == "__main__":
    main()
