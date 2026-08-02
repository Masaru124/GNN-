"""
Pulls the two verified colliding pairs (BaSrI4, Ca2Mg) fresh from the
Materials Project API and saves them as CIF files, ready to feed into
ComFormer / ALIGNN for the collision-resolution check.

Requires: pip install mp-api pymatgen
"""

import os
import sys
from mp_api.client import MPRester

OUT_DIR = "comformer_check_structures"
MP_IDS = {
    "BaSrI4_pair": ["mp-754388", "mp-1178504"],
    "Ca2Mg_pair": ["mp-1949090", "mp-1039332"],
}

# Ground-truth formation energies from verification (eV/atom)
KNOWN_TARGETS = {
    "mp-754388": -1.91428489,
    "mp-1178504": -1.98049591,
    "mp-1949090": -0.04146314,
    "mp-1039332": 0.02474444,
}


def main():
    api_key = "EYbuHOo0WB47mRls32dKKuIuACxNUkzj"
    print(f"--- EXPORTING COLLIDING CIF STRUCTURES ---", flush=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    fields = ["material_id", "structure", "formula_pretty"]

    with MPRester(api_key) as mpr:
        for pair_name, ids in MP_IDS.items():
            print(f"\nProcessing {pair_name} ({ids})...", flush=True)
            for mp_id in ids:
                docs = mpr.materials.summary.search(material_ids=[mp_id], fields=fields)
                if not docs:
                    print(f"  Error: Could not retrieve {mp_id} from MP API.")
                    continue
                structure = docs[0].structure
                cif_path = os.path.join(OUT_DIR, f"{mp_id}.cif")
                structure.to(filename=cif_path)
                print(f"  Saved {mp_id} -> {cif_path}")
                print(f"    Formula: {docs[0].formula_pretty}")
                print(f"    Sites: {len(structure)}")
                print(f"    Known Target Formation Energy: {KNOWN_TARGETS[mp_id]:.6f} eV/atom")

    print(f"\n[DONE] All CIF files saved to '{OUT_DIR}/' directory.")

if __name__ == "__main__":
    main()
