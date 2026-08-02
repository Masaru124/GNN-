import sys
import os
import torch
from pymatgen.core import Structure

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

def run_alignn():
    print("\n=======================================================", flush=True)
    print("        RUNNING ALIGNN (PRETRAINED SOTA MODEL)", flush=True)
    print("=======================================================", flush=True)
    try:
        from alignn.pretrained import pretrained
        # Load ALIGNN pretrained model on formation energy
        # Typical models: "jv_formation_energy_per_atom_alignn" or "mp_e_form_alignn"
        model_name = "jv_formation_energy_per_atom_alignn"
        print(f"Loading pretrained ALIGNN model: '{model_name}'...", flush=True)
        model = pretrained(model_name)
        
        preds = {}
        for mp_id, path in CIF_FILES.items():
            # ALIGNN prediction on structure / cif
            out = model.predict_structure(path)
            # jarvis/alignn returns energy or property in eV or eV/atom
            preds[mp_id] = float(out)
            print(f"  ALIGNN Prediction for {mp_id}: {out:.6f} (Known: {KNOWN_TARGETS[mp_id]:.6f})")
            
        gap_basri = abs(preds["mp-754388"] - preds["mp-1178504"])
        gap_ca2mg = abs(preds["mp-1949090"] - preds["mp-1039332"])
        print(f"\nALIGNN Predicted Gap BaSrI4: {gap_basri:.6f} eV/atom (Real: 0.066211)")
        print(f"ALIGNN Predicted Gap Ca2Mg:  {gap_ca2mg:.6f} eV/atom (Real: 0.066208)")
        return preds
    except Exception as exc:
        print(f"ALIGNN Error: {exc}", flush=True)
        import traceback
        traceback.print_exc()
        return None

def run_matgl():
    print("\n=======================================================", flush=True)
    print("        RUNNING MATGL (MEGNet / M3GNet SOTA MODELS)", flush=True)
    print("=======================================================", flush=True)
    try:
        import matgl
        from matgl.ext.ase import M3GNetCalculator, MEGNetCalculator
        
        # Check available pretrained models in matgl
        print("Available MatGL models:", matgl.get_available_pretrained_models(), flush=True)
        
        # Load MEGNet formation energy model
        megnet_name = "MEGNet-MP-2018.6.1-Eform"
        print(f"\nLoading pretrained MatGL model: '{megnet_name}'...", flush=True)
        model = matgl.load_model(megnet_name)
        
        preds = {}
        for mp_id, path in CIF_FILES.items():
            struct = Structure.from_file(path)
            eform = model.predict_structure(struct)
            preds[mp_id] = float(eform)
            print(f"  MEGNet Prediction for {mp_id}: {eform:.6f} (Known: {KNOWN_TARGETS[mp_id]:.6f})")
            
        gap_basri = abs(preds["mp-754388"] - preds["mp-1178504"])
        gap_ca2mg = abs(preds["mp-1949090"] - preds["mp-1039332"])
        print(f"\nMEGNet Predicted Gap BaSrI4: {gap_basri:.6f} eV/atom (Real: 0.066211)")
        print(f"MEGNet Predicted Gap Ca2Mg:  {gap_ca2mg:.6f} eV/atom (Real: 0.066208)")
        return preds
    except Exception as exc:
        print(f"MatGL Error: {exc}", flush=True)
        import traceback
        traceback.print_exc()
        return None

def main():
    print("--- EXTERNAL SOTA MODEL INFERENCE BENCHMARK ---", flush=True)
    alignn_preds = run_alignn()
    matgl_preds = run_matgl()

if __name__ == "__main__":
    main()
