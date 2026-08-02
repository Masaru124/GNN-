import sys
import os
from pathlib import Path
sys.path.append(os.getcwd())

from scripts.train import load_data
from pymatgen.core import Structure

print("Testing dataset loading...")
structures, labels = load_data("data/raw")
print(f"Loaded {len(structures)} raw structures and {len(labels)} label entries.")

# Print one sample structure and label to inspect keys
first_struct = structures[0]
first_pmg = Structure.from_dict(first_struct)
print("First structure properties:", getattr(first_pmg, "properties", None))
print("First structure material_id attr:", getattr(first_pmg, "material_id", None))

# Test resolving material ID
def resolve_material_id(idx, labels, structure):
    sid = getattr(structure, "material_id", None)
    if sid is None and hasattr(structure, "properties"):
        sid = structure.properties.get("material_id")
    if sid is not None and str(sid) in labels:
        return str(sid)
    fallback = f"mp-fake-{idx}"
    if fallback in labels:
        return fallback
    keys = list(labels.keys())
    if idx < len(keys):
        return keys[idx]
    return None

mid = resolve_material_id(0, labels, first_pmg)
print(f"Resolved material ID for index 0: {mid}")
if mid:
    print(f"Target value (formation_energy_per_atom): {labels[mid]['formation_energy_per_atom']}")
