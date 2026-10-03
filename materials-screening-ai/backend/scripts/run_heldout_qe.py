# -*- coding: utf-8 -*-
"""
Programmatic Runner for Held-Out Perovskite PBE DFT Calculations (Item 2).

Builds physically realistic atomic unit cells with explicit C, N, H atoms
(no placeholder / virtual A-site atoms) for:
  - FAPbI3: a = 6.362 Å (Citations: Weller et al., J. Mater. Chem. A 3, 9208 (2015); Castelli et al., APL Mater. 2, 081514 (2014) Table I)
  - MASnI3: a = 6.230 Å (Citations: Stoumpos et al., Inorg. Chem. 52, 9019 (2013); Castelli et al., APL Mater. 2, 081514 (2014) Table I)

Executes Quantum ESPRESSO PBE pipeline via DFTValidationService and writes output
to heldout_qe_results.json for eval_protocol.py to ingest dynamically.
"""

import os
import sys
import json
import time
from pathlib import Path
import numpy as np

# Ensure backend root is on sys.path
backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

# Set QE environment defaults (repo-relative) if not set
_QE_ROOT = Path(__file__).resolve().parents[3] / "qe"
if "QE_BIN_DIR" not in os.environ:
    os.environ["QE_BIN_DIR"] = str(_QE_ROOT / "bin")
if "SSSP_PP_DIR" not in os.environ:
    os.environ["SSSP_PP_DIR"] = str(_QE_ROOT / "pseudo")

from pymatgen.core import Structure, Lattice
from app.services.dft_validation import get_dft_service


def build_fa_perovskite(b_elem: str, x_elem: str, a_cubic: float) -> Structure:
    """
    Build cubic ABX3 perovskite cell with planar formamidinium [HC(NH2)2]+ cation
    centered in the cuboctahedral cavity at (0, 0, 0).
    Lattice constant a_cubic in Angstroms.
    """
    lat = Lattice.cubic(a_cubic)
    # B at (0.5, 0.5, 0.5), X at face centers (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)
    species = [b_elem, x_elem, x_elem, x_elem]
    frac_coords = [
        [0.5, 0.5, 0.5],
        [0.5, 0.5, 0.0],
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5]
    ]

    # Planar Formamidinium cation in xy plane centered at origin:
    # C-N = 1.32 Å, C-H = 1.08 Å, N-H = 1.00 Å, N-C-N angle = 125 deg
    c_cart = np.array([0.0, 0.0, 0.0])
    h_c = np.array([0.0, 1.08, 0.0])
    n1 = np.array([1.1708, -0.6095, 0.0])
    n2 = np.array([-1.1708, -0.6095, 0.0])
    h1a = np.array([2.0368, -1.1095, 0.0])
    h1b = np.array([1.1708, 0.3905, 0.0])
    h2a = np.array([-2.0368, -1.1095, 0.0])
    h2b = np.array([-1.1708, 0.3905, 0.0])

    fa_species = ['C', 'N', 'N', 'H', 'H', 'H', 'H', 'H']
    fa_carts = [c_cart, n1, n2, h_c, h1a, h1b, h2a, h2b]
    for sp, cart in zip(fa_species, fa_carts):
        species.append(sp)
        frac = (cart / a_cubic) % 1.0
        frac_coords.append(frac.tolist())

    return Structure(lat, species, frac_coords)


def build_ma_perovskite(b_elem: str, x_elem: str, a_cubic: float) -> Structure:
    """
    Build cubic ABX3 perovskite cell with methylammonium [CH3NH3]+ cation
    oriented along [100] and centered in the cuboctahedral cavity at (0, 0, 0).
    Lattice constant a_cubic in Angstroms.
    """
    lat = Lattice.cubic(a_cubic)
    species = [b_elem, x_elem, x_elem, x_elem]
    frac_coords = [
        [0.5, 0.5, 0.5],
        [0.5, 0.5, 0.0],
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5]
    ]

    # C-N bond along x: C at -0.735 Å, N at +0.735 Å (d_CN = 1.47 Å)
    c_cart = np.array([-0.735, 0.0, 0.0])
    n_cart = np.array([+0.735, 0.0, 0.0])
    # Hydrogens on C (d_CH = 1.09 Å, tetrahedral)
    h_c1 = np.array([-1.0983, 1.0276, 0.0])
    h_c2 = np.array([-1.0983, -0.5138, 0.8900])
    h_c3 = np.array([-1.0983, -0.5138, -0.8900])
    # Hydrogens on N (d_NH = 1.01 Å, tetrahedral, staggered)
    h_n1 = np.array([1.0717, 0.4761, 0.8246])
    h_n2 = np.array([1.0717, -0.9522, 0.0])
    h_n3 = np.array([1.0717, 0.4761, -0.8246])

    ma_species = ['C', 'N', 'H', 'H', 'H', 'H', 'H', 'H']
    ma_carts = [c_cart, n_cart, h_c1, h_c2, h_c3, h_n1, h_n2, h_n3]
    for sp, cart in zip(ma_species, ma_carts):
        species.append(sp)
        frac = (cart / a_cubic) % 1.0
        frac_coords.append(frac.tolist())

    return Structure(lat, species, frac_coords)


def get_heldout_structures() -> dict:
    return {
        "FAPbI3": {
            "structure": build_fa_perovskite("Pb", "I", 6.362),
            "a_cubic_angstrom": 6.362,
            "target_exp_gap_eV": 1.48,
            "domain_class": "in_domain_lead_halide",
            "citation": "Weller et al., J. Mater. Chem. A 3, 9208 (2015); Castelli et al., APL Mater. 2, 081514 (2014) Table I"
        },
        "MASnI3": {
            "structure": build_ma_perovskite("Sn", "I", 6.230),
            "a_cubic_angstrom": 6.230,
            "target_exp_gap_eV": 1.20,
            "domain_class": "out_of_domain_tin_halide",
            "citation": "Stoumpos et al., Inorg. Chem. 52, 9019 (2013); Castelli et al., APL Mater. 2, 081514 (2014) Table I"
        }
    }


def run_or_load_heldout_qe(force_rerun: bool = False) -> dict:
    out_paths = [
        Path(backend_dir) / "scripts" / "heldout_qe_results.json",
        Path(backend_dir) / ".." / "research" / "heldout_qe_results.json",
    ]

    results = {}
    primary_out = out_paths[0]
    if not force_rerun and primary_out.exists():
        try:
            with open(primary_out, "r", encoding="utf-8") as f:
                results = json.load(f)
        except Exception:
            results = {}

    structures_meta = get_heldout_structures()
    dft_service = get_dft_service()

    for formula, meta in structures_meta.items():
        s = meta["structure"]
        poscar_str = s.to(fmt="poscar")

        print(f"\n================================================================================")
        print(f"HELD-OUT CANDIDATE: {formula}")
        print(f"Lattice Constant a = {meta['a_cubic_angstrom']} A | Target Exp Gap = {meta['target_exp_gap_eV']} eV")
        print(f"Citation: {meta['citation']}")
        print(f"POSCAR ACTUAL GEOMETRY EXECUTED BY QE:")
        print(f"--------------------------------------------------------------------------------")
        print(poscar_str.strip())
        print(f"--------------------------------------------------------------------------------")

        if not force_rerun and formula in results and results[formula].get("status") == "success":
            print(f"[CACHED RESULT] {formula}: PBE gap = {results[formula]['pbe_gap_eV']} eV ({results[formula]['gap_type']})")
            continue

        print(f"Executing self-consistent Quantum ESPRESSO PBE calculation for {formula}...")
        t0 = time.time()
        res = dft_service.run_pbe_pipeline(s, formula, fast_mode=True, kpt_dist=0.35)
        dt = time.time() - t0

        pbe_gap = res.get("pbe_gap_eV")
        gap_type = res.get("gap_type", "unknown")
        status = res.get("status", "unknown")
        print(f"[{status.upper()}] {formula}: PBE gap = {pbe_gap} eV ({gap_type}) in {dt:.1f}s")

        results[formula] = {
            "formula": formula,
            "a_cubic_angstrom": meta["a_cubic_angstrom"],
            "citation": meta["citation"],
            "target_exp_gap_eV": meta["target_exp_gap_eV"],
            "domain_class": meta["domain_class"],
            "pbe_gap_eV": round(float(pbe_gap), 4) if pbe_gap is not None else None,
            "gap_type": gap_type,
            "runtime_seconds": round(dt, 2),
            "status": status,
            "poscar": poscar_str,
        }

    # Save to both backend/scripts/ and research/
    for p in out_paths:
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Saved held-out results to: {p.resolve()}")

    return results


if __name__ == "__main__":
    force = "--force" in sys.argv
    run_or_load_heldout_qe(force_rerun=force)
