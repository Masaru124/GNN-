import sys
import os
import json
import gzip
import csv
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

def main():
    print("--- VERIFYING RAW MP DATASET ENTRIES ---", flush=True)
    labels_path = "data/raw/mp_labels.csv"
    structures_path = "data/raw/mp_structures.json.gz"

    labels = {}
    with open(labels_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels[row["material_id"]] = row

    target_ids = ["mp-754388", "mp-1178504", "mp-1949090", "mp-1039332"]
    
    print("\nReading mp_labels.csv:")
    for mp_id in target_ids:
        if mp_id in labels:
            print(f"\n[FOUND IN LABELS] {mp_id}:")
            print(json.dumps(labels[mp_id], indent=2))
        else:
            print(f"\n[NOT FOUND] {mp_id}")

    print("\nReading mp_structures.json.gz:")
    with gzip.open(structures_path, "rt", encoding="utf-8") as f:
        structures_raw = json.load(f)

    label_keys = list(labels.keys())
    found_count = 0
    for idx, s_dict in enumerate(structures_raw):
        sid = s_dict.get("material_id")
        if sid is None and "properties" in s_dict:
            sid = s_dict["properties"].get("material_id")
        if sid is None and idx < len(label_keys):
            sid = label_keys[idx]
        sid_str = str(sid)
        
        if sid_str in target_ids:
            found_count += 1
            struct = Structure.from_dict(s_dict)
            sga = SpacegroupAnalyzer(struct)
            sg_sym = sga.get_space_group_symbol()
            sg_num = sga.get_space_group_number()
            formula = struct.composition.reduced_formula
            target_energy = float(labels[sid_str]["formation_energy_per_atom"])
            
            print(f"\n[STRUCTURE MATCH] {sid_str} (raw index {idx}):")
            print(f"  Formula: {formula}")
            print(f"  Space Group: {sg_sym} (No. {sg_num})")
            print(f"  Num sites: {len(struct)}")
            print(f"  Lattice: a={struct.lattice.a:.4f}, b={struct.lattice.b:.4f}, c={struct.lattice.c:.4f}")
            print(f"  Formation Energy: {target_energy:.8f} eV/atom")

if __name__ == "__main__":
    main()
