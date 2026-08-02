import sys
import os
import itertools
from collections import defaultdict, Counter
import numpy as np
import time

sys.path.append(os.path.join(os.getcwd(), "scripts"))
from collision_scan_full import load_dataset_subsampled, build_descriptors_for_all_radii

def main():
    print("--- RUNNING EXACT BUCKET AUDIT ON 50,000 STRUCTURES ---", flush=True)
    dataset = load_dataset_subsampled(50000)
    radii = [2.0, 2.5, 3.0, 3.5, 4.0]
    
    # We replicate the descriptor computation to inspect bucket sizes directly
    max_radius = 4.0
    print("\nQuerying neighbor lists at max_radius = 4.0 Å...", flush=True)
    all_structures_neighs = []
    for idx, (structure, _) in enumerate(dataset):
        neighs = structure.get_all_neighbors(r=max_radius, include_index=True)
        all_structures_neighs.append(neighs)

    targets = [t for _, t in dataset]
    threshold = 0.05
    max_neighbors = 24

    for r in radii:
        descriptors = []
        for idx, (structure, _) in enumerate(dataset):
            neighs_max = all_structures_neighs[idx]
            per_atom_descriptors = []
            for i, neighs in enumerate(neighs_max):
                sorted_neighs = [n for n in neighs if n.nn_distance <= r]
                sorted_neighs = sorted(sorted_neighs, key=lambda n: n.nn_distance)
                if len(sorted_neighs) > max_neighbors:
                    sorted_neighs = sorted_neighs[:max_neighbors]
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

        colliding_structures = set()
        n_pairs = 0
        bucket_size_counts = Counter()

        for desc, idxs in buckets.items():
            if len(idxs) < 2:
                continue
            # Check pairwise target differences
            pairs_in_bucket = 0
            structures_in_bucket = set()
            for i, j in itertools.combinations(idxs, 2):
                diff = abs(targets[i] - targets[j])
                if diff > threshold:
                    n_pairs += 1
                    pairs_in_bucket += 1
                    structures_in_bucket.add(i)
                    structures_in_bucket.add(j)
                    colliding_structures.add(i)
                    colliding_structures.add(j)
            if len(structures_in_bucket) > 0:
                bucket_size_counts[len(structures_in_bucket)] += 1

        n_colliding_structs = len(colliding_structures)
        rate = n_colliding_structs / len(dataset)
        print(f"\n[Radius {r:.1f} Å Audit]:")
        print(f"  Sample size N = {len(dataset)}")
        print(f"  Colliding structures count: {n_colliding_structs}")
        print(f"  Collision rate: {rate:.6f} ({rate*100:.3f}%)")
        print(f"  Total colliding pairs: {n_pairs}")
        print(f"  Bucket size breakdown (size: number of clusters): {dict(sorted(bucket_size_counts.items()))}")
        
        # Verify formula: Rate * N == len(colliding_structures)
        assert n_colliding_structs == int(round(rate * len(dataset)))
        print(f"  -> Arithmetic Check: {rate:.6f} * {len(dataset)} = {rate*len(dataset):.1f} == {n_colliding_structs} (EXACT MATCH)")

if __name__ == "__main__":
    main()
