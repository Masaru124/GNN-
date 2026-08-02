"""
External SOTA Model Benchmark: ALIGNN & MatGL (MEGNet / M3GNet)
Evaluates published pretrained models on the two verified colliding CIF pairs.
"""

import sys
import os
import torch
import numpy as np
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

def evaluate_alignn():
    print("\n" + "=" * 70, flush=True)
    print("1. EVALUATING ALIGNN (jv_formation_energy_peratom_alignn)", flush=True)
    print("=" * 70, flush=True)
    try:
        from alignn.pretrained import get_prediction
        from jarvis.core.atoms import Atoms
        
        model_name = "jv_formation_energy_peratom_alignn"
        preds = {}
        
        for mp_id, path in CIF_FILES.items():
            atoms = Atoms.from_cif(path)
            # get_prediction returns float
            pred = get_prediction(model_name=model_name, atoms=atoms)
            if isinstance(pred, (tuple, list)):
                pred = pred[0]
            preds[mp_id] = float(pred)
            print(f"  [ALIGNN] {mp_id}: {preds[mp_id]:.6f} eV/atom (Ground Truth: {KNOWN_TARGETS[mp_id]:.6f})", flush=True)
            
        gap_basri = abs(preds["mp-754388"] - preds["mp-1178504"])
        gap_ca2mg = abs(preds["mp-1949090"] - preds["mp-1039332"])
        print(f"\n  --> ALIGNN Predicted Gap BaSrI4: {gap_basri:.6f} eV/atom (True Gap: 0.066211)")
        print(f"  --> ALIGNN Predicted Gap Ca2Mg:  {gap_ca2mg:.6f} eV/atom (True Gap: 0.066208)")
        return preds
    except Exception as exc:
        print(f"ALIGNN Evaluation Error: {exc}", flush=True)
        import traceback
        traceback.print_exc()
        return None

def evaluate_matgl():
    print("\n" + "=" * 70, flush=True)
    print("2. EVALUATING MATGL (MEGNet / M3GNet)", flush=True)
    print("=" * 70, flush=True)
    try:
        import matgl
        available = matgl.get_available_pretrained_models()
        print(f"Available MatGL pretrained models: {available}", flush=True)
        
        # Select first available formation energy model
        eform_models = [m for m in available if "Eform" in m or "eform" in m or "E_form" in m]
        print(f"Eform models: {eform_models}", flush=True)
        
        results = {}
        for m_name in eform_models[:2]:
            print(f"\nLoading MatGL model: '{m_name}'...", flush=True)
            try:
                model = matgl.load_model(m_name)
                preds = {}
                for mp_id, path in CIF_FILES.items():
                    struct = Structure.from_file(path)
                    eform = model.predict_structure(struct)
                    if hasattr(eform, "item"):
                        eform = eform.item()
                    preds[mp_id] = float(eform)
                    print(f"  [{m_name}] {mp_id}: {eform:.6f} eV/atom (Ground Truth: {KNOWN_TARGETS[mp_id]:.6f})", flush=True)
                gap_basri = abs(preds["mp-754388"] - preds["mp-1178504"])
                gap_ca2mg = abs(preds["mp-1949090"] - preds["mp-1039332"])
                print(f"  --> {m_name} Predicted Gap BaSrI4: {gap_basri:.6f} eV/atom (True Gap: 0.066211)")
                print(f"  --> {m_name} Predicted Gap Ca2Mg:  {gap_ca2mg:.6f} eV/atom (True Gap: 0.066208)")
                results[m_name] = preds
            except Exception as e:
                print(f"  Failed to evaluate {m_name}: {e}")
        return results
    except Exception as exc:
        print(f"MatGL Evaluation Error: {exc}", flush=True)
        import traceback
        traceback.print_exc()
        return None

def main():
    print("--- EXTERNAL SOTA MODEL BENCHMARK ON COLLIDING CIF PAIRS ---", flush=True)
    print("Ground Truth Gaps:")
    print("  BaSrI4 pair (mp-754388 vs mp-1178504): 0.066211 eV/atom")
    print("  Ca2Mg  pair (mp-1949090 vs mp-1039332): 0.066208 eV/atom")
    
    alignn_preds = evaluate_alignn()
    matgl_res = evaluate_matgl()
    
    print("\n" + "=" * 90)
    print("                 PUBLISHED SOTA EXTERNAL MODEL SUMMARY")
    print("=" * 90)
    print(f"{'Model Name':<35} | {'BaSrI4 Pred Gap':<22} | {'Ca2Mg Pred Gap':<22}")
    print("-" * 90)
    
    if alignn_preds:
        g1 = abs(alignn_preds["mp-754388"] - alignn_preds["mp-1178504"])
        g2 = abs(alignn_preds["mp-1949090"] - alignn_preds["mp-1039332"])
        print(f"{'ALIGNN (jv_formation_energy)':<35} | {g1:<22.6f} | {g2:<22.6f}")
        
    if matgl_res:
        for m_name, p in matgl_res.items():
            g1 = abs(p["mp-754388"] - p["mp-1178504"])
            g2 = abs(p["mp-1949090"] - p["mp-1039332"])
            print(f"{m_name:<35} | {g1:<22.6f} | {g2:<22.6f}")
    print("=" * 90)

if __name__ == "__main__":
    main()
