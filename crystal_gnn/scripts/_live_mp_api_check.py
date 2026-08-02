import sys
import os
import json

def main():
    api_key = "EYbuHOo0WB47mRls32dKKuIuACxNUkzj"
    print("--- LIVE MATERIALS PROJECT API QUERY VIA MPRESTER ---", flush=True)
    target_ids = ["mp-754388", "mp-1178504", "mp-1949090", "mp-1039332", "mp-752402"]
    
    from mp_api.client import MPRester
    
    fields = ["material_id", "formula_pretty", "symmetry", "formation_energy_per_atom", "energy_above_hull", "band_gap"]
    
    with MPRester(api_key) as mpr:
        print(f"Querying summary data for material_ids={target_ids}...", flush=True)
        docs = mpr.materials.summary.search(material_ids=target_ids, fields=fields)
        
        print(f"\nRetrieved {len(docs)} documents from Materials Project live database:\n", flush=True)
        for doc in docs:
            mid = str(doc.material_id)
            formula = doc.formula_pretty
            sg_num = doc.symmetry.number
            sg_sym = doc.symmetry.symbol
            c_sys = doc.symmetry.crystal_system
            fe = doc.formation_energy_per_atom
            ehull = doc.energy_above_hull
            bg = doc.band_gap
            
            print(f"[{mid}]:")
            print(f"  Formula: {formula}")
            print(f"  Space Group: {sg_sym} (No. {sg_num}, {c_sys})")
            print(f"  Formation Energy per Atom: {fe:.8f} eV/atom")
            print(f"  Energy Above Hull: {ehull:.8f} eV/atom")
            print(f"  Band Gap: {bg:.4f} eV")
            print("-" * 50, flush=True)

if __name__ == "__main__":
    main()
