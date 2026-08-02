"""
CHGNet (Nature Machine Intelligence 2023) Benchmark
Evaluates the official pretrained CHGNet model on the colliding CIF pairs.
"""

import sys
import os
import torch
from pymatgen.core import Structure
from chgnet.model import CHGNet

CIF_FILES = {
    "mp-754388": "comformer_check_structures/mp-754388.cif",
    "mp-1178504": "comformer_check_structures/mp-1178504.cif",
    "mp-1949090": "comformer_check_structures/mp-1949090.cif",
    "mp-1039332": "comformer_check_structures/mp-1039332.cif",
}

KNOWN_TARGETS = {
    "mp-754388": -1.91428489,
    "mp-1178504": -1.98049591,
    "mp-1949090": -0.04146314,
    "mp-1039332": 0.02474444,
}

def main():
    print("=======================================================", flush=True)
    print("   EVALUATING CHGNet (Nature Machine Intelligence 2023)", flush=True)
    print("=======================================================", flush=True)
    
    print("Loading pretrained CHGNet model...", flush=True)
    chgnet = CHGNet.load()
    
    preds_energy = {}
    preds_eform = {}
    
    for mp_id, path in CIF_FILES.items():
        struct = Structure.from_file(path)
        prediction = chgnet.predict_structure(struct)
        
        # prediction contains 'e' (energy per atom), 'forces', 'stress'
        e_per_atom = float(prediction['e'])
        preds_energy[mp_id] = e_per_atom
        print(f"  [CHGNet Energy] {mp_id}: {e_per_atom:.6f} eV/atom (Ground Truth Formation Energy: {KNOWN_TARGETS[mp_id]:.6f})", flush=True)
        
    gap_basri = abs(preds_energy["mp-754388"] - preds_energy["mp-1178504"])
    gap_ca2mg = abs(preds_energy["mp-1949090"] - preds_energy["mp-1039332"])
    
    print("\n" + "=" * 70, flush=True)
    print("                CHGNet BENCHMARK RESULTS", flush=True)
    print("=" * 70, flush=True)
    print(f"  --> CHGNet Predicted Gap BaSrI4: {gap_basri:.6f} eV/atom (True Gap: 0.066211)")
    print(f"  --> CHGNet Predicted Gap Ca2Mg:  {gap_ca2mg:.6f} eV/atom (True Gap: 0.066208)")
    print("=" * 70, flush=True)

if __name__ == "__main__":
    main()
