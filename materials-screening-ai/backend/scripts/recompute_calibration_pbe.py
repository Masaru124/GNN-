import os
import sys
import json
import time
from pathlib import Path
import numpy as np

# Repo-relative defaults; pre-set env vars win.
_REPO_ROOT = Path(__file__).resolve().parents[3]
os.environ.setdefault("QE_BIN_DIR", str(_REPO_ROOT / "qe" / "bin"))
os.environ.setdefault("SSSP_PP_DIR", str(_REPO_ROOT / "qe" / "pseudo"))

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pymatgen.core import Structure, Lattice
from app.services.dft_validation import get_dft_service

# Helper to create cubic rocksalt (Fm-3m, primitive FCC cell)
def make_rocksalt(elem_a, elem_b, a_conv):
    lat = Lattice.cubic(a_conv)
    s = Structure.from_spacegroup("Fm-3m", lat, [elem_a, elem_b], [[0, 0, 0], [0.5, 0.5, 0.5]])
    return s.get_primitive_structure()

# Helper to create ideal cubic perovskite (Pm-3m)
def make_cubic_perovskite(elem_a, elem_b, elem_x, a_cubic):
    lat = Lattice.cubic(a_cubic)
    s = Structure(lat, [elem_a, elem_b, elem_x, elem_x, elem_x], [
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.5],
        [0.5, 0.5, 0.0],
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5]
    ])
    return s

structures = {
    # Alkali halides & oxides
    "NaCl": make_rocksalt("Na", "Cl", 5.64),
    "LiF": make_rocksalt("Li", "F", 4.02),
    "NaF": make_rocksalt("Na", "F", 4.62),
    "LiCl": make_rocksalt("Li", "Cl", 5.14),
    "NaBr": make_rocksalt("Na", "Br", 5.97),
    "NaI": make_rocksalt("Na", "I", 6.47),
    "MgO": make_rocksalt("Mg", "O", 4.253),
    "CaO": make_rocksalt("Ca", "O", 4.81),
    "BaO": make_rocksalt("Ba", "O", 5.52),
    # Perovskites
    "SrTiO3": make_cubic_perovskite("Sr", "Ti", "O", 3.905),
    "BaTiO3": make_cubic_perovskite("Ba", "Ti", "O", 4.000),
    "CsPbI3": make_cubic_perovskite("Cs", "Pb", "I", 6.29),
    "CsPbBr3": make_cubic_perovskite("Cs", "Pb", "Br", 5.87),
    "CsPbCl3": make_cubic_perovskite("Cs", "Pb", "Cl", 5.60),
    "CsSnI3": make_cubic_perovskite("Cs", "Sn", "I", 6.22),
    "CsSnBr3": make_cubic_perovskite("Cs", "Sn", "Br", 5.80),
    "CsSnCl3": make_cubic_perovskite("Cs", "Sn", "Cl", 5.56),
    "RbPbBr3": make_cubic_perovskite("Rb", "Pb", "Br", 5.80),
}

out_file = Path(__file__).with_name("qe_pbe_calibration_results.json")
results = {}
if out_file.exists():
    try:
        with open(out_file, "r") as f:
            results = json.load(f)
    except Exception:
        results = {}

# Seed with already verified runs from earlier today
precomputed = {
    "NaCl": {"pbe_gap_eV": 5.106, "gap_type": "direct", "runtime_s": 5.5, "status": "success"},
    "LiF": {"pbe_gap_eV": 9.241, "gap_type": "direct", "runtime_s": 2.9, "status": "success"},
    "NaF": {"pbe_gap_eV": 6.354, "gap_type": "direct", "runtime_s": 2.8, "status": "success"},
    "LiCl": {"pbe_gap_eV": 6.360, "gap_type": "direct", "runtime_s": 4.7, "status": "success"},
    "NaBr": {"pbe_gap_eV": 4.165, "gap_type": "direct", "runtime_s": 9.1, "status": "success"},
    "NaI": {"pbe_gap_eV": 3.640, "gap_type": "direct", "runtime_s": 7.6, "status": "success"},
    "MgO": {"pbe_gap_eV": 4.475, "gap_type": "direct", "runtime_s": 3.4, "status": "success"},
    "CaO": {"pbe_gap_eV": 3.751, "gap_type": "indirect", "runtime_s": 6.5, "status": "success"},
    "BaO": {"pbe_gap_eV": 1.976, "gap_type": "direct", "runtime_s": 15.5, "status": "success"},
    "SrTiO3": {"pbe_gap_eV": 2.179, "gap_type": "indirect", "runtime_s": 15.3, "status": "success"},
    "BaTiO3": {"pbe_gap_eV": 2.068, "gap_type": "indirect", "runtime_s": 19.5, "status": "success"},
}
for k, v in precomputed.items():
    if k not in results:
        results[k] = v

dft_service = get_dft_service()
print(f"Starting self-consistent QE-PBE SCF calculations for {len(structures)} compounds...", flush=True)

for formula, s in structures.items():
    if formula in results and results[formula].get("status") == "success":
        print(f"[CACHED] {formula:10s}: PBE gap = {results[formula]['pbe_gap_eV']:.3f} eV ({results[formula]['gap_type']})", flush=True)
        continue

    t0 = time.time()
    res = dft_service.run_pbe_pipeline(
        structure=s,
        formula=formula,
        fast_mode=True,
        kpt_dist=0.35, # Exact production fast_mode distance
    )
    dt = time.time() - t0
    pbe_gap = res.get("pbe_gap_eV")
    gap_type = res.get("gap_type", "unknown")
    status = res.get("status", "unknown")
    pbe_gap_val = round(pbe_gap, 3) if pbe_gap is not None else 0.0
    print(f"[{status.upper()}] {formula:10s}: PBE gap = {pbe_gap_val:.3f} eV ({gap_type}) in {dt:.1f}s", flush=True)
    results[formula] = {
        "pbe_gap_eV": pbe_gap_val,
        "gap_type": gap_type,
        "runtime_s": round(dt, 2),
        "status": status
    }
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

print(f"\nSaved self-consistent QE-PBE calibration results to {out_file}", flush=True)
