"""
Collision scan: quantifies how often distinct crystal structures produce
near-identical local-environment graphs at a given cutoff radius, despite
having meaningfully different target property values.

Highly optimized to avoid O(N) dict key casting inside loop.
"""

import argparse
import itertools
import numpy as np
import logging
from collections import defaultdict
from pymatgen.core import Structure
from scripts.train import load_data

logging.basicConfig(level=logging.WARNING)


def load_dataset_subsampled(n_samples):
    """
    Load raw structures, subsample them in raw dict format, then convert to pymatgen Structure.
    """
    print("Loading raw dataset structures and labels...", flush=True)
    structures_raw, labels = load_data("data/raw")
    print(f"Loaded {len(structures_raw)} raw structures and {len(labels)} label entries.", flush=True)
    
    # Pre-build keys list once to avoid massive O(N^2) complexity in loop
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
            target_val = labels[mid]["formation_energy_per_atom"]
            valid_indices.append(idx)
            mapped_targets.append(target_val)
            
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
    for idx, target in zip(chosen_indices, chosen_targets):
        try:
            s_dict = structures_raw[idx]
            structure = Structure.from_dict(s_dict)
            dataset.append((structure, target))
        except Exception as e:
            continue
            
    print(f"Loaded {len(dataset)} valid structures for collision scan.", flush=True)
    return dataset


def build_graph_descriptor(structure, radius):
    """
    Build a local environment descriptor matching the GNN's message passing:
    For each atom, build sorted tuple of (neighbor_species, distance_bin) within cutoff radius.
    Aggregate all per-atom lists into a sorted tuple representing the whole crystal structure.
    """
    max_neighbors = 24
    per_atom_descriptors = []
    
    all_neighs = structure.get_all_neighbors(r=radius, include_index=True)
    
    for i, neighs in enumerate(all_neighs):
        # Sort neighbors by distance to keep closest ones if exceeding max_neighbors
        sorted_neighs = sorted(neighs, key=lambda n: n.nn_distance)
        if len(sorted_neighs) > max_neighbors:
            sorted_neighs = sorted_neighs[:max_neighbors]
            
        # Species atomic number and discretized distance bin (0.05 A bins)
        neigh_tuples = []
        for n in sorted_neighs:
            species_num = int(n.specie.number)
            distance_bin = int(round(n.nn_distance / 0.05))
            neigh_tuples.append((species_num, distance_bin))
            
        # Neighbors list is sorted to guarantee permutation invariance
        neigh_tuples.sort()
        
        # Central atom species + its sorted neighbor environments
        central_species = int(structure[i].specie.number)
        per_atom_descriptors.append((central_species, tuple(neigh_tuples)))
        
    # Sort all per-atom descriptors to ensure translation/permutation invariance of structures
    per_atom_descriptors.sort()
    return tuple(per_atom_descriptors)


def collision_rate_at_radius(dataset, radius, threshold):
    """
    Returns:
      collision_rate: fraction of structures involved in at least one
        collision (identical descriptor at this radius, but |target
        difference| > threshold).
      n_pairs: number of colliding pairs found.
      example_pairs: up to 5 example (i, j, target_diff) tuples for
        manual inspection.
    """
    print(f"Building descriptors for radius {radius}...", flush=True)
    descriptors = [build_graph_descriptor(s, radius) for s, _ in dataset]
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
    return collision_rate, n_pairs, example_pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=10000)
    parser.add_argument("--radii", type=float, nargs="+", default=[3.0, 4.0, 6.0, 8.0])
    parser.add_argument("--threshold", type=float, default=0.05,
                         help="Min |target diff| (eV/atom) to count as a meaningful collision")
    args = parser.parse_args()

    dataset = load_dataset_subsampled(args.n_samples)

    print(f"Running collision scan on {len(dataset)} structures\n", flush=True)
    print(f"{'Radius':>8} | {'Collision Rate':>15} | {'# Colliding Pairs':>18}", flush=True)
    print("-" * 50, flush=True)

    results = {}
    for r in args.radii:
        rate, n_pairs, examples = collision_rate_at_radius(dataset, r, args.threshold)
        results[r] = (rate, n_pairs, examples)
        print(f"{r:>8.1f} | {rate:>15.4f} | {n_pairs:>18d}", flush=True)

    print("\nExample colliding pairs at smallest radius tested:", flush=True)
    smallest_r = min(args.radii)
    for i, j, diff in results[smallest_r][2]:
        print(f"  structure[{i}] vs structure[{j}]: target diff = {diff:.4f} eV/atom", flush=True)


if __name__ == "__main__":
    main()
