import sys
import os
from pathlib import Path
import numpy as np
import itertools
from collections import defaultdict
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

sys.path.append(os.path.join(os.getcwd(), "scripts"))
from train import load_data

sys.path.append(os.getcwd())


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


def get_structure_info(structure, target_val, mid):
    composition = structure.composition.reduced_formula
    try:
        sga = SpacegroupAnalyzer(structure)
        space_group = sga.get_space_group_symbol()
    except Exception:
        space_group = "Unknown"
    return {
        "material_id": mid,
        "composition": composition,
        "space_group": space_group,
        "target": target_val,
        "num_sites": len(structure)
    }


def main():
    print("Loading raw dataset structures and labels...", flush=True)
    structures_raw, labels = load_data("data/raw")
    label_keys = list(labels.keys())

    print("Filtering valid structure indices...", flush=True)
    valid_indices = []
    mapped_targets = []
    for idx, s_dict in enumerate(structures_raw):
        mid = resolve_material_id(idx, labels, s_dict, label_keys)
        if mid is not None and mid in labels:
            target_val = labels[mid]["formation_energy_per_atom"]
            valid_indices.append(idx)
            mapped_targets.append(target_val)

    # Recreate the 10k subsampled dataset using seed 0
    rng = np.random.default_rng(0)
    sub_idxs = rng.choice(len(valid_indices), size=10000, replace=False)
    chosen_indices = [valid_indices[i] for i in sub_idxs]
    chosen_targets = [mapped_targets[i] for i in sub_idxs]

    # Convert only the chosen structures to get the exact ones for 1231, 2981, 5066, 9137
    print("\n--- ANALYZING HIGH-LIGHTED COLLIDING PAIRS (FROM 10K SUB-SAMPLE) ---", flush=True)
    pair_indices = [1231, 2981, 5066, 9137]
    structures_10k = {}
    for idx in pair_indices:
        raw_idx = chosen_indices[idx]
        target = chosen_targets[idx]
        s_dict = structures_raw[raw_idx]
        structure = Structure.from_dict(s_dict)
        mid = resolve_material_id(raw_idx, labels, s_dict, label_keys)
        structures_10k[idx] = (structure, target, mid)

    for i in pair_indices:
        struct, target, mid = structures_10k[i]
        info = get_structure_info(struct, target, mid)
        print(f"Structure[{i}]:")
        print(f"  mp-id: {info['material_id']}")
        print(f"  composition: {info['composition']}")
        print(f"  space group: {info['space_group']}")
        print(f"  num sites: {info['num_sites']}")
        print(f"  target formation energy: {info['target']:.8f} eV/atom")

    # Manually recompute differences
    t_1231 = structures_10k[1231][1]
    t_2981 = structures_10k[2981][1]
    diff_1 = abs(t_1231 - t_2981)
    
    t_5066 = structures_10k[5066][1]
    t_9137 = structures_10k[9137][1]
    diff_2 = abs(t_5066 - t_9137)

    print("\n--- MANUAL DIFFERENCE RE-COMPUTATION ---", flush=True)
    print(f"abs(target[1231] - target[2981]) = |{t_1231:.8f} - {t_2981:.8f}| = {diff_1:.8f} eV/atom")
    print(f"abs(target[5066] - target[9137]) = |{t_5066:.8f} - {t_9137:.8f}| = {diff_2:.8f} eV/atom")

if __name__ == "__main__":
    main()
