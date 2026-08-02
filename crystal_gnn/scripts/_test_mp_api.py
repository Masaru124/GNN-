import sys
import os
import json

def main():
    print("--- TESTING MP API QUERY ---", flush=True)
    # Check if MP_API_KEY environment variable exists or if mp-api package is available
    api_key = os.environ.get("MP_API_KEY") or os.environ.get("PMG_MAPI_KEY")
    print(f"API Key present in environment: {bool(api_key)}")
    
    try:
        from mp_api.client import MPRester
        print("mp_api package is installed!")
    except ImportError:
        print("mp_api package is NOT installed.")
        MPRester = None

    target_ids = ["mp-754388", "mp-1178504", "mp-1949090", "mp-1039332"]

    if MPRester and api_key:
        print(f"\nQuerying Materials Project live database for {target_ids}...")
        with MPRester(api_key) as mpr:
            docs = mpr.materials.summary.search(material_ids=target_ids)
            for doc in docs:
                mid = str(doc.material_id)
                formula = doc.formula_pretty
                sg_num = doc.symmetry.number
                sg_sym = doc.symmetry.symbol
                fe = doc.formation_energy_per_atom
                print(f"\n[LIVE MP API RESPONSE] {mid}:")
                print(f"  Formula: {formula}")
                print(f"  Space Group: {sg_sym} (No. {sg_num})")
                print(f"  Formation Energy: {fe:.8f} eV/atom")
    else:
        print("\nNo MP API key or mp_api package found in current shell environment.")
        print("Checking if pymatgen MPRester legacy interface works or if we can make a direct HTTPS request to Next-Gen MP API...")

        import urllib.request
        # Try direct query to public API endpoint or legacy REST API if available
        # MP API v2 requires header "X-API-KEY"
        if api_key:
            url = f"https://api.materialsproject.org/v2/materials/summary/?material_ids={','.join(target_ids)}&_fields=material_id,formula_pretty,symmetry,formation_energy_per_atom"
            req = urllib.request.Request(url, headers={"X-API-KEY": api_key})
            try:
                with urllib.request.urlopen(req) as resp:
                    data = json.loads(resp.read().decode())
                    print("Live API Data:", json.dumps(data, indent=2))
            except Exception as e:
                print("Direct HTTP request error:", e)

if __name__ == "__main__":
    main()
