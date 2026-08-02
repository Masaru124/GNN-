from mp_api.client import MPRester
import os
from pathlib import Path

def load_env_key():
    # Check .env file
    env_path = Path(".env")
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("MP_API_KEY="):
                return line.split("=", 1)[1].strip()
    # Check system environment
    import os
    return os.environ.get("MP_API_KEY")

def main():
    api_key = load_env_key()
    if not api_key:
        print("Error: MP_API_KEY is not set in environment or .env file.", flush=True)
        return
        
    with MPRester(api_key) as mpr:
        ids = ["mp-1041984", "mp-1392145", "mp-1178392", "mp-776606"]
        docs = mpr.summary.search(
            material_ids=ids,
            fields=["material_id", "formation_energy_per_atom",
                    "band_gap", "symmetry", "nsites"]
        )
        for d in docs:
            print(d.material_id,
                  d.formation_energy_per_atom,
                  d.band_gap,
                  d.symmetry.symbol,
                  d.nsites,
                  flush=True)

if __name__ == "__main__":
    main()
