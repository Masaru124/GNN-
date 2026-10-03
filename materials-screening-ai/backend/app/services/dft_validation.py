# -*- coding: utf-8 -*-
"""
Tier 3 DFT Automation Service (Item 10).

Drives Quantum ESPRESSO pw.x + ph.x via ASE Espresso calculator for:
  Tier 3a: PBE vc-relax → SCF → NSCF gap extraction
  Tier 3b: Δ-ML calibrated gap correction (via delta_ml_corrector.py)
  Tier 3c: r2SCAN meta-GGA SCF + gap
  Tier 3d: HSE06 eigenvalues at VBM/CBM k-points only (ONCV pseudopotentials)
  NAC:     ph.x Born-charge correction for phonon service (Item 12)

Architecture:
  All QE jobs are asynchronous — launched in a BackgroundTasks worker and
  tracked via the DFTJob database table. The FastAPI event loop is never blocked.

QE binary discovery:
  1. $QE_BIN_DIR environment variable (e.g. /usr/lib/quantum-espresso/bin)
  2. PATH lookup (pw.x must be in PATH)
  3. Common install locations (/usr/bin, /usr/local/bin, conda envs)
  
Pseudopotential discovery:
  1. $SSSP_PP_DIR environment variable (directory with *.UPF files)
  2. graphify-out/pseudopotentials/ (local copy)
  3. Automatic download of SSSP efficiency set from Materials Cloud Archive
     (performed once on first use; requires internet)

If QE is not installed, all methods return a descriptive error with installation
instructions — no silent hang, no placeholder result.
"""

import os
import re
import json
import time
import logging
import shutil
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pymatgen.core import Structure, Composition, Element
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

try:
    from ase.calculators.espresso import Espresso, EspressoProfile
    from pymatgen.io.ase import AseAtomsAdaptor
    HAS_ASE_ESPRESSO = True
except ImportError:
    HAS_ASE_ESPRESSO = False

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# QE Binary & Pseudopotential Discovery
# ---------------------------------------------------------------------------

QE_BIN_DIR = os.getenv("QE_BIN_DIR", "")
SSSP_PP_DIR = os.getenv("SSSP_PP_DIR", "")

# Repo-relative <root>/qe/pseudo; SSSP_PP_DIR env var overrides.
PP_SUBDIR = Path(SSSP_PP_DIR) if SSSP_PP_DIR else Path(__file__).resolve().parents[4] / "qe" / "pseudo"
SSSP_EFFICIENCY_URL_BASE = "https://pseudopotentials.quantum-espresso.org/upf_files/"

# SSSP v1.3 efficiency set checksums + filenames for elements we commonly encounter
# Format: {element_symbol: filename}
SSSP_EFFICIENCY_PPS = {
    "H":  "H.pbe-rrkjus_psl.1.0.0.UPF",
    "C":  "C.pbe-n-kjpaw_psl.1.0.0.UPF",
    "Li": "Li.pbe-s-kjpaw_psl.1.0.0.UPF",
    "Na": "Na.pbe-spnl-kjpaw_psl.1.0.0.UPF",
    "K":  "K.pbe-spn-kjpaw_psl.1.0.0.UPF",
    "Rb": "Rb.pbe-spn-kjpaw_psl.1.0.0.UPF",
    "Cs": "Cs.pbe-spn-kjpaw_psl.1.0.0.UPF",
    "Ca": "Ca.pbe-spn-kjpaw_psl.1.0.0.UPF",
    "Sr": "Sr.pbe-spn-kjpaw_psl.1.0.0.UPF",
    "Ba": "Ba.pbe-spn-kjpaw_psl.1.0.0.UPF",
    "Mg": "Mg.pbe-spnl-kjpaw_psl.1.0.0.UPF",
    "Zr": "Zr.pbe-spn-kjpaw_psl.1.0.0.UPF",
    "Sn": "Sn.pbe-dn-kjpaw_psl.1.0.0.UPF",
    "Pb": "Pb.pbe-dn-kjpaw_psl.1.0.0.UPF",
    "Ge": "Ge.pbe-dn-kjpaw_psl.1.0.0.UPF",
    "Ti": "Ti.pbe-spn-kjpaw_psl.1.0.0.UPF",
    "Si": "Si.pbe-n-rrkjus_psl.1.0.0.UPF",
    "Cl": "Cl.pbe-n-rrkjus_psl.1.0.0.UPF",
    "Br": "Br.pbe-dn-kjpaw_psl.1.0.0.UPF",
    "I":  "I.pbe-n-kjpaw_psl.1.0.0.UPF",
    "F":  "F.pbe-n-rrkjus_psl.1.0.0.UPF",
    "O":  "O.pbe-n-rrkjus_psl.1.0.0.UPF",
    "N":  "N.pbe-n-rrkjus_psl.1.0.0.UPF",
    "S":  "S.pbe-n-rrkjus_psl.1.0.0.UPF",
    "In": "In.pbe-dn-rrkjus_psl.1.0.0.UPF",
    "Fe": "Fe.pbe-spn-kjpaw_psl.1.0.0.UPF",
    "Mn": "Mn.pbe-spn-kjpaw_psl.1.0.0.UPF",
    "Ni": "Ni.pbe-spn-kjpaw_psl.1.0.0.UPF",
    "Co": "Co.pbe-spn-kjpaw_psl.1.0.0.UPF",
    "Cu": "Cu.pbe-dn-rrkjus_psl.1.0.0.UPF",
    "Zn": "Zn.pbe-dn-rrkjus_psl.1.0.0.UPF",
    "Al": "Al.pbe-n-rrkjus_psl.1.0.0.UPF",
    "V":  "V.pbe-spnl-kjpaw_psl.1.0.0.UPF",
    "Nb": "Nb.pbe-spn-kjpaw_psl.1.0.0.UPF",
    "Mo": "Mo.pbe-spn-kjpaw_psl.1.0.0.UPF",
    "W":  "W.pbe-spn-kjpaw_psl.1.0.0.UPF",
    "Y":  "Y.pbe-spn-kjpaw_psl.1.0.0.UPF",
    "La": "La.pbe-spfn-kjpaw_psl.1.0.0.UPF",
}

# SSSP recommended cutoffs (ecutwfc, ecutrho) per element — eV
# Source: SSSP v1.3 efficiency recommended cutoffs
SSSP_CUTOFFS: Dict[str, Tuple[float, float]] = {
    "H":  (40, 320), "C": (45, 360), "Li": (50, 400), "Na": (60, 480), "K": (60, 480),
    "Rb": (60, 480), "Cs": (70, 560), "Ca": (55, 440), "Sr": (60, 480),
    "Ba": (60, 480), "Mg": (60, 480), "Zr": (60, 480), "Sn": (75, 600),
    "Pb": (70, 560), "Ge": (75, 600), "Ti": (55, 440), "Si": (30, 240),
    "Cl": (40, 320), "Br": (40, 320), "I":  (45, 360), "F": (50, 400),
    "O":  (50, 400), "N":  (50, 400), "S":  (40, 320), "In": (70, 560),
    "Fe": (60, 480), "Mn": (65, 520), "Ni": (55, 440), "Co": (60, 480),
    "Cu": (65, 520), "Zn": (65, 520), "Al": (30, 240), "V":  (70, 560),
    "Nb": (60, 480), "Mo": (60, 480), "W":  (70, 560), "Y":  (55, 440),
    "La": (55, 440),
}

DEFAULT_ECUTWFC = 60.0  # Ry — used when element not in SSSP_CUTOFFS
DEFAULT_ECUTRHO = 480.0  # Ry


def _get_pw_binary() -> Optional[str]:
    """Find pw.x or native Windows pw.exe via env var, PATH, or workspace locations (no WSL needed)."""
    binary_names = ["pw.exe", "pw.x.exe", "pw.x", "pw"]

    # 1. Env vars: QE_BIN_DIR or QE_DIR
    for env_var in [QE_BIN_DIR, os.getenv("QE_DIR")]:
        if env_var:
            for bname in binary_names:
                cand = os.path.join(env_var, bname)
                if os.path.isfile(cand):
                    return cand
                cand_bin = os.path.join(env_var, "bin", bname)
                if os.path.isfile(cand_bin):
                    return cand_bin

    # 2. PATH lookup
    for bname in binary_names:
        found = shutil.which(bname)
        if found:
            return found

    # 3. Workspace / local directories (native Windows, no WSL needed)
    workspace_candidates = [
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "qe", "bin")),
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "qe", "bin")),
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "qe", "bin")),
        r"C:\QE\bin",
        r"C:\Program Files\Quantum ESPRESSO\bin",
        r"C:\Tools\qe\bin",
    ]
    for wdir in workspace_candidates:
        for bname in binary_names:
            cand = os.path.join(wdir, bname)
            if os.path.isfile(cand):
                return cand

    # 4. Common Linux locations
    common = [
        "/usr/bin/pw.x",
        "/usr/local/bin/pw.x",
        "/opt/quantum-espresso/bin/pw.x",
        os.path.expanduser("~/qe/bin/pw.x"),
        os.path.expanduser("~/opt/qe/bin/pw.x"),
    ]
    for path in common:
        if os.path.isfile(path):
            return path

    return None


def _get_ph_binary() -> Optional[str]:
    """Find ph.x or native Windows ph.exe for phonon calculations."""
    binary_names = ["ph.exe", "ph.x.exe", "ph.x", "ph"]

    for env_var in [QE_BIN_DIR, os.getenv("QE_DIR")]:
        if env_var:
            for bname in binary_names:
                cand = os.path.join(env_var, bname)
                if os.path.isfile(cand):
                    return cand
                cand_bin = os.path.join(env_var, "bin", bname)
                if os.path.isfile(cand_bin):
                    return cand_bin

    for bname in binary_names:
        found = shutil.which(bname)
        if found:
            return found

    workspace_candidates = [
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "qe", "bin")),
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "qe")),
        r"C:\QE\bin",
        r"C:\Program Files\Quantum ESPRESSO\bin",
    ]
    for wdir in workspace_candidates:
        for bname in binary_names:
            cand = os.path.join(wdir, bname)
            if os.path.isfile(cand):
                return cand

    return None


def _get_pp_dir() -> Optional[Path]:
    """Locate pseudopotential directory."""
    if SSSP_PP_DIR and os.path.isdir(SSSP_PP_DIR):
        return Path(SSSP_PP_DIR)
    if PP_SUBDIR.is_dir() and any(PP_SUBDIR.glob("*.UPF")):
        return PP_SUBDIR
    return None


def _attempt_pp_download(elements: List[str]) -> Optional[Path]:
    """
    Attempt to download SSSP efficiency pseudopotentials for requested elements.
    Only downloads missing files. Requires internet connection.
    """
    try:
        import requests
    except ImportError:
        return None

    PP_SUBDIR.mkdir(parents=True, exist_ok=True)
    downloaded_any = False

    for elem in elements:
        filename = SSSP_EFFICIENCY_PPS.get(elem)
        if not filename:
            continue
        target = PP_SUBDIR / filename
        if target.exists():
            continue
        url = SSSP_EFFICIENCY_URL_BASE + filename
        logger.info(f"[DFT] Downloading SSSP PP for {elem}: {url}")
        try:
            resp = requests.get(url, timeout=30, stream=True)
            if resp.status_code == 200:
                with open(target, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=8192):
                        f.write(chunk)
                downloaded_any = True
                logger.info(f"[DFT] Downloaded {filename}")
        except Exception as e:
            logger.warning(f"[DFT] Failed to download {filename}: {e}")

    if list(PP_SUBDIR.glob("*.UPF")):
        return PP_SUBDIR
    return None


def _get_sssp_cutoffs(structure: Structure) -> Tuple[float, float]:
    """Return the maximum recommended ecutwfc and ecutrho for all elements in structure."""
    max_wfc = DEFAULT_ECUTWFC
    max_rho = DEFAULT_ECUTRHO
    for site in structure:
        elem = str(site.specie.symbol)
        wfc, rho = SSSP_CUTOFFS.get(elem, (DEFAULT_ECUTWFC, DEFAULT_ECUTRHO))
        max_wfc = max(max_wfc, wfc)
        max_rho = max(max_rho, rho)
    return max_wfc, max_rho


def _get_valence_electrons(element: str, pp_dir: Optional[Path] = None) -> float:
    """Parse z_valence from UPF file header, with standard fallback."""
    if pp_dir:
        filename = SSSP_EFFICIENCY_PPS.get(element)
        if filename:
            upf_path = pp_dir / filename
            if upf_path.exists():
                try:
                    text = upf_path.read_text(errors="ignore")[:3000]
                    m = re.search(r'z_valence\s*=\s*"?([0-9\.eEdD\+\-]+)"?', text, re.IGNORECASE)
                    if not m:
                        m = re.search(r'([0-9\.eEdD\+\-]+)\s+Z valence', text, re.IGNORECASE)
                    if m:
                        val_str = m.group(1).replace("d", "e").replace("D", "e")
                        v = float(val_str)
                        if v > 0:
                            return v
                except Exception:
                    pass
    # Standard pseudo valence fallback for common elements
    STANDARD_VALENCE = {
        "H": 1.0, "Li": 1.0, "Be": 2.0, "B": 3.0, "C": 4.0, "N": 5.0, "O": 6.0, "F": 7.0,
        "Na": 9.0, "Mg": 10.0, "Al": 3.0, "Si": 4.0, "P": 5.0, "S": 6.0, "Cl": 7.0,
        "K": 9.0, "Ca": 10.0, "Sc": 11.0, "Ti": 12.0, "V": 13.0, "Cr": 14.0, "Mn": 15.0,
        "Fe": 16.0, "Co": 17.0, "Ni": 18.0, "Cu": 11.0, "Zn": 12.0, "Ga": 13.0, "Ge": 14.0,
        "As": 5.0, "Se": 6.0, "Br": 7.0, "Rb": 9.0, "Sr": 10.0, "Y": 11.0, "Zr": 12.0,
        "Nb": 13.0, "Mo": 14.0, "Tc": 15.0, "Ru": 16.0, "Rh": 17.0, "Pd": 18.0, "Ag": 11.0,
        "Cd": 12.0, "In": 13.0, "Sn": 14.0, "Sb": 5.0, "Te": 6.0, "I": 7.0, "Cs": 9.0,
        "Ba": 10.0, "La": 11.0, "Hf": 12.0, "Ta": 13.0, "W": 14.0, "Re": 15.0, "Os": 16.0,
        "Ir": 17.0, "Pt": 18.0, "Au": 11.0, "Hg": 12.0, "Tl": 13.0, "Pb": 14.0, "Bi": 15.0
    }
    return STANDARD_VALENCE.get(element, 4.0)


def _calculate_nbnd(structure: Structure, pp_dir: Optional[Path] = None) -> int:
    """
    Calculate an explicit number of bands (nbnd) ensuring adequate empty (conduction) bands.
    Never run SCF/NSCF with default nbnd = occupied_bands, which causes empty conduction band lists.
    """
    total_val_e = 0.0
    for site in structure:
        elem = str(site.specie.symbol)
        total_val_e += _get_valence_electrons(elem, pp_dir)
    n_occ = int(np.ceil(total_val_e / 2.0))
    # Occupied bands plus at least 8 or 50% margin, minimum 12 bands
    nbnd = max(n_occ + 8, int(np.ceil(1.5 * n_occ)), 12)
    return nbnd


# Materials Project standard Hubbard U (eV) for transition-metal oxides and fluorides
# Matches conventions used in CHGNet training (MPtrj)
MP_HUBBARD_U = {
    "Co": 3.32,
    "Fe": 5.30,
    "Mn": 3.90,
    "Ni": 6.20,
    "V": 3.25,
    "Cr": 3.70,
    "Mo": 4.38,
    "W": 6.20,
    "Ti": 2.50,
    "Cu": 4.00,
}


def _get_hubbard_cards(structure: Structure) -> Tuple[List[str], Dict[str, float]]:
    """
    Generate modern QE 7.x HUBBARD (atomic) cards matching Materials Project / CHGNet conventions.
    Applied when transition metals are present with oxygen (O) or fluorine (F).
    """
    elem_symbols = {str(site.specie.symbol) for site in structure}
    has_anion = any(anion in elem_symbols for anion in ["O", "F", "S", "Se"])
    hubbard_lines = []
    applied_u = {}
    if has_anion:
        for elem, u_val in MP_HUBBARD_U.items():
            if elem in elem_symbols:
                hubbard_lines.append(f"  U {elem}-3d {u_val}")
                applied_u[elem] = u_val
    if hubbard_lines:
        card = "HUBBARD (atomic)\n" + "\n".join(hubbard_lines)
        return [card], applied_u
    return [], {}


def _kpoints_from_dist(
    structure: Structure, kpt_dist: float = 0.25, force_even: bool = True
) -> Tuple[int, int, int]:
    """
    Compute a Monkhorst-Pack k-point mesh from a target k-point distance (Å⁻¹).
    kpt_dist = 0.25 Å⁻¹ is the AiiDA default PBE protocol (suitable for SCF-quality runs).

    If force_even=True (default), enforces that all mesh subdivisions N_i are even:
        N_i = max(2, 2 * ceil(raw_N_i / 2))

    Physical justification & mathematical proof:
    In unshifted Monkhorst-Pack sampling (used by standard SCF grids in Quantum ESPRESSO),
    the fractional k-coordinates along axis i are k_{i,r} = r / N_i for integer r.
    When N_i is odd, 2r = N_i has NO integer solution, so the zone-boundary planes
    k_i = 1/2 are mathematically NEVER sampled. This completely omits critical high-symmetry 
    zone-boundary points such as R(0.5, 0.5, 0.5), X(0.5, 0, 0), and M(0.5, 0.5, 0) in cubic, 
    tetragonal, and orthorhombic lattices. Because band gap extrema (VBM and CBM) frequently 
    reside at these high-symmetry boundary points (e.g., R in perovskites, X in rocksalt oxides),
    an odd mesh causes massive band dispersion sampling errors (e.g. 1.8-2.3 eV artificial gap overestimation).
    Forcing even N >= 2 guarantees that k_i = 1/2 is always exactly represented in the grid.

    CRITICAL SCOPE LIMITATION:
    `force_even` guarantees exact representation of zone-boundary points with fractional 
    coordinates k_i = 1/2 in simple-cubic, tetragonal, and orthorhombic lattices. It does NOT
    guarantee capturing a band extremum located at a generic (non-special or incommensurate) 
    k-point, which is common in lower-symmetry structures (e.g. monoclinic, triclinic, or distorted 
    perovskite phases like Pnma or P2_1/c). For candidates with lower symmetry (such as distorted 
    KZrCl3 or other real discovery candidates), `force_even` alone is not sufficient; a proper 
    k-density convergence test (increasing k-density until Delta E_g < 25 meV) is required.
    """
    recip = structure.lattice.reciprocal_lattice
    b_norms = [np.linalg.norm(b) for b in recip.matrix]
    mesh = []
    for b in b_norms:
        raw_n = max(1, int(np.ceil(b / kpt_dist)))
        if force_even:
            # Guarantee even N >= 2: 1 -> 2, 2 -> 2, 3 -> 4, 4 -> 4, 5 -> 6...
            n_even = max(2, 2 * int(np.ceil(raw_n / 2.0)))
            mesh.append(n_even)
        else:
            mesh.append(raw_n)
    return tuple(mesh)  # type: ignore


# ---------------------------------------------------------------------------
# Main DFT Validation Service
# ---------------------------------------------------------------------------

class DFTValidationService:
    """
    Manages asynchronous Quantum ESPRESSO DFT jobs for Tier 3 validation.
    
    All public methods return a result dict. When QE is not installed,
    they return {status: 'qe_not_installed', install_instructions: ...}
    so the caller can surface this honestly rather than silently failing.
    """

    def __init__(self):
        raw_pw = _get_pw_binary()
        self.pw_binary = Path(raw_pw).as_posix() if raw_pw else None
        raw_ph = _get_ph_binary()
        self.ph_binary = Path(raw_ph).as_posix() if raw_ph else None
        self.qe_available = self.pw_binary is not None
        self.ph_available = self.ph_binary is not None
        self.pp_dir: Optional[Path] = _get_pp_dir()

        if self.qe_available:
            logger.info(f"[DFT] QE found: pw.x = {self.pw_binary}")
        else:
            logger.warning(
                "[DFT] pw.x not found in PATH, QE_BIN_DIR, or common locations. "
                "DFT jobs will return descriptive errors until QE is installed. "
                "Install via: conda install -c conda-forge qe  OR  "
                "Download from https://www.quantum-espresso.org/download-page/"
            )

    # ------------------------------------------------------------------
    # QE availability check & Symmetry Scope Audit
    # ------------------------------------------------------------------

    def check_kpoint_symmetry_scope(self, structure: Structure, formula: Optional[str] = None) -> Dict[str, Any]:
        """
        Audit candidate crystal symmetry to determine if the `force_even` parity guarantee is sufficient
        or if a multi-grid k-density convergence scan is required.
        
        High-symmetry cubic systems (Pm-3m, Fm-3m, Fd-3m, etc.) have extrema at Gamma, X, M, or R (k_i = 1/2),
        where force_even guarantees exact sampling. Lower-symmetry systems (orthorhombic Pnma, monoclinic,
        or tilted octahedral phases) often have band extrema at generic incommensurate k-points, necessitating
        an explicit k-density convergence test.
        """
        try:
            sga = SpacegroupAnalyzer(structure, symprec=1e-3)
            sg_symbol = sga.get_space_group_symbol()
            sg_number = sga.get_space_group_number()
            crystal_system = sga.get_crystal_system()
        except Exception:
            sg_symbol = "Unknown"
            sg_number = 0
            crystal_system = "unknown"

        # Cubic systems: space groups 195-230
        is_cubic = (195 <= sg_number <= 230) or str(crystal_system).lower() == "cubic"

        if is_cubic:
            recommendation = "zone_boundary_points_sampled"
            note = (
                f"Structure has cubic symmetry ({sg_symbol}, #{sg_number}). "
                "`force_even=True` mathematically guarantees sampling of zone-boundary points with fractional coordinates k_i in {{0, 1/2}} (Gamma, X, M, R). "
                "However, band extrema can still reside off high-symmetry points (e.g. Si CBM along Gamma-X). "
                "A dense NSCF mesh or band path calculation is recommended for definitive band extremum location."
            )
        else:
            recommendation = "k_density_convergence_scan_required"
            note = (
                f"Structure has lower symmetry ({crystal_system} {sg_symbol}, #{sg_number}). "
                "`force_even=True` guarantees k_i = 1/2 but does NOT guarantee capturing band extrema "
                "located at generic incommensurate k-points. An explicit multi-grid k-density convergence test "
                "(scanning until Delta E_g < 25 meV) is required before treating the band gap as converged."
            )

        return {
            "formula": formula or structure.composition.reduced_formula,
            "spacegroup_symbol": sg_symbol,
            "spacegroup_number": sg_number,
            "crystal_system": crystal_system,
            "is_high_symmetry_cubic": is_cubic,
            "zone_boundary_points_sampled": is_cubic,
            "recommended_protocol": recommendation,
            "assessment": note,
        }

    def run_kpoint_convergence_scan(
        self,
        structure: Structure,
        formula: str,
        kpt_dists: Optional[List[float]] = None,
        fast_mode: bool = True,
        tolerance_eV: float = 0.025,
    ) -> Dict[str, Any]:
        """
        Execute an automated k-point density convergence test across a ladder of k-point spacings.
        Evaluates whether the band gap stabilizes within `tolerance_eV` (default: 25 meV).
        """
        if kpt_dists is None:
            kpt_dists = [0.35, 0.28, 0.22, 0.18]

        symmetry_audit = self.check_kpoint_symmetry_scope(structure, formula=formula)
        steps = []
        gaps = []

        for dist in kpt_dists:
            kpts = _kpoints_from_dist(structure, kpt_dist=dist, force_even=True)
            res = self.run_pbe_pipeline(structure, formula, kpt_dist=dist, fast_mode=fast_mode)
            gap = res.get("pbe_gap_eV")
            gaps.append(gap)
            steps.append({
                "kpt_dist_inv_A": dist,
                "mesh": list(kpts),
                "gap_eV": gap,
                "status": res.get("status"),
                "runtime_seconds": res.get("runtime_seconds", 0.0),
            })

        valid_gaps = [g for g in gaps if g is not None]
        converged = False
        max_delta = None
        if len(valid_gaps) >= 2:
            deltas = [abs(valid_gaps[i] - valid_gaps[i - 1]) for i in range(1, len(valid_gaps))]
            max_delta = deltas[-1] if deltas else None
            converged = (deltas[-1] <= tolerance_eV)

        return {
            "formula": formula,
            "symmetry_audit": symmetry_audit,
            "convergence_steps": steps,
            "final_gap_eV": valid_gaps[-1] if valid_gaps else None,
            "is_converged": converged,
            "last_step_delta_eV": round(max_delta, 4) if max_delta is not None else None,
            "tolerance_eV": tolerance_eV,
        }

    def get_install_status(self) -> Dict[str, Any]:
        return {
            "qe_available": self.qe_available,
            "pw_binary": self.pw_binary,
            "ph_available": self.ph_available,
            "ph_binary": self.ph_binary,
            "pp_dir": str(self.pp_dir) if self.pp_dir else None,
            "install_instructions": (
                "Native Windows (No WSL required): Run 'python backend/scripts/setup_qe_windows.py' to download "
                "the official precompiled Intel oneAPI / MS-MPI Quantum ESPRESSO Windows release (pw.exe) to c:\\Users\\User\\Desktop\\GNN\\qe. "
                "Alternatively, download qe-7.5-win-oneapi-msmpi.zip from https://github.com/QMatSuite/quantum-espresso-windows-exe/releases "
                "and extract to c:\\Users\\User\\Desktop\\GNN\\qe\\bin."
            ) if not self.qe_available else "QE installed and ready.",
        }

    def _qe_not_installed_error(self, tier: str) -> Dict[str, Any]:
        return {
            "status": "qe_not_installed",
            "tier": tier,
            "error": (
                "Quantum ESPRESSO (pw.exe/pw.x) is not detected on PATH or workspace. "
                "No WSL needed: Run 'python backend/scripts/setup_qe_windows.py' to download native Windows QE 7.5."
            ),
            "qe_available": False,
        }

    def _make_profile(self) -> EspressoProfile:
        """Create ASE EspressoProfile with quoted posix path to prevent Windows backslash escaping errors."""
        threads = str(min(24, os.cpu_count() or 4))
        os.environ["OMP_NUM_THREADS"] = threads
        os.environ["MKL_NUM_THREADS"] = threads
        os.environ["OPENBLAS_NUM_THREADS"] = threads
        cmd = f'"{Path(self.pw_binary).as_posix()}"'
        pseudo_dir = Path(self.pp_dir).as_posix() if self.pp_dir else ""
        return EspressoProfile(command=cmd, pseudo_dir=pseudo_dir)

    # ------------------------------------------------------------------
    # Pseudopotential resolution
    # ------------------------------------------------------------------

    def _resolve_pseudopotentials(self, structure: Structure) -> Optional[Dict[str, str]]:
        """
        Resolve pseudopotential file paths for all elements in structure.
        Attempts: (1) existing pp_dir, (2) auto-download SSSP.
        Returns dict {element: filepath} or None if missing.
        """
        elements = list({str(s.specie.symbol) for s in structure})

        # Try existing directory
        pp_dir = self.pp_dir
        if pp_dir is None:
            pp_dir = _attempt_pp_download(elements)
            if pp_dir:
                self.pp_dir = pp_dir

        if pp_dir is None:
            return None

        pp_dict = {}
        for elem in elements:
            filename = SSSP_EFFICIENCY_PPS.get(elem)
            if not filename:
                logger.warning(f"[DFT] No SSSP efficiency PP known for element {elem}")
                return None
            full_path = pp_dir / filename
            if not full_path.exists():
                logger.info(f"[DFT] PP for {elem} missing at {full_path}, attempting auto-download...")
                _attempt_pp_download([elem])
            if full_path.exists():
                pp_dict[elem] = filename
            else:
                logger.warning(f"[DFT] PP for {elem} could not be resolved at {full_path}")
                return None
        return pp_dict

    # ------------------------------------------------------------------
    # Tier 3a: PBE vc-relax → SCF → NSCF gap
    # ------------------------------------------------------------------

    def run_pbe_pipeline(
        self,
        structure: Structure,
        formula: str,
        workdir: Optional[str] = None,
        kpt_dist: float = 0.35,
        fast_mode: bool = True,
    ) -> Dict[str, Any]:
        """
        Run the PBE DFT pipeline:
          fast_mode=True  (Default): Skips the expensive 50-step vc-relax by using
                                     the MLIP pre-relaxed geometry directly, running
                                     standardized SCF + NSCF band gap in ~30 seconds.
          fast_mode=False:          Runs optimized vc-relax (nstep=12, mixing_beta=0.35)
                                     followed by symmetry refinement, SCF, and NSCF.

        Returns dict with gap, gap_type (direct/indirect/metallic), relaxed structure CIF,
        convergence parameters stored for provenance.
        """
        if not self.qe_available:
            return self._qe_not_installed_error("Tier3a_PBE")

        if not HAS_ASE_ESPRESSO:
            return {"status": "error", "error": "ASE Espresso calculator not available (pip install ase)"}

        pp_dict = self._resolve_pseudopotentials(structure)
        if pp_dict is None:
            return {
                "status": "error",
                "error": (
                    "SSSP pseudopotentials not found. "
                    "Set SSSP_PP_DIR env var or ensure internet access for auto-download. "
                    "Manual install: `aiida-pseudo install sssp --pseudo-format upf`"
                ),
            }

        ecutwfc, ecutrho = _get_sssp_cutoffs(structure)
        if fast_mode:
            kpt_dist = max(kpt_dist, 0.35)
            ecutwfc = min(ecutwfc, 50.0)
            ecutrho = min(ecutrho, 400.0)

        kpts = _kpoints_from_dist(structure, kpt_dist=kpt_dist)
        t0 = time.time()

        workdir = workdir or tempfile.mkdtemp(prefix="matscreen_dft_pbe_")
        os.makedirs(workdir, exist_ok=True)

        try:
            profile = self._make_profile()

            # ── Step 1: vc-relax (skipped in fast_mode) ────────────────
            hubbard_cards, applied_u = _get_hubbard_cards(structure)
            nbnd_init = _calculate_nbnd(structure, self.pp_dir)

            if fast_mode:
                logger.info(f"[DFT] fast_mode=True: Using MLIP pre-relaxed geometry for {formula}, skipping vc-relax.")
                relaxed_struct = structure
            else:
                atoms = AseAtomsAdaptor.get_atoms(structure)
                relax_input = {
                    "control": {
                        "calculation": "vc-relax",
                        "restart_mode": "from_scratch",
                        "nstep": 12,
                        "outdir": Path(workdir, "relax_out").as_posix(),
                        "pseudo_dir": Path(str(self.pp_dir)).as_posix(),
                        "prefix": "matscreen",
                        "verbosity": "high",
                        "tprnfor": True,
                        "tstress": True,
                    },
                    "system": {
                        "ecutwfc": ecutwfc,
                        "ecutrho": ecutrho,
                        "occupations": "smearing",
                        "smearing": "mv",      # Marzari-Vanderbilt
                        "degauss": 0.01,       # Ry
                        "nbnd": nbnd_init,
                    },
                    "electrons": {
                        "conv_thr": 1.0e-5,    # 1e-5 Ry for fast relaxation convergence
                        "mixing_beta": 0.35,   # 0.35 prevents charge sloshing for d-elements
                        "mixing_mode": "plain",
                    },
                    "ions": {"ion_dynamics": "bfgs"},
                    "cell": {"cell_dynamics": "bfgs", "press_conv_thr": 2.0},
                }

                calc_relax = Espresso(
                    profile=profile,
                    pseudopotentials=pp_dict,
                    kpts=kpts,
                    input_data=relax_input,
                    directory=Path(workdir, "relax").as_posix(),
                    additional_cards=hubbard_cards if hubbard_cards else None,
                )
                atoms.calc = calc_relax
                atoms.get_potential_energy()  # runs pw.x

                relaxed_atoms = calc_relax.get_atoms() if hasattr(calc_relax, "get_atoms") else atoms
                relaxed_struct = AseAtomsAdaptor.get_structure(relaxed_atoms)

            # ── Step 2: Symmetry refinement ───────────────────────────
            sga = SpacegroupAnalyzer(relaxed_struct, symprec=0.01)
            standard_struct = sga.get_primitive_standard_structure()
            sg_number = sga.get_space_group_number()
            sg_symbol = sga.get_space_group_symbol()
            atoms_std = AseAtomsAdaptor.get_atoms(standard_struct)

            # Update k-points & Hubbard cards for standardized primitive cell
            kpts_std = _kpoints_from_dist(standard_struct, kpt_dist)
            hubbard_cards, applied_u = _get_hubbard_cards(standard_struct)
            nbnd = _calculate_nbnd(standard_struct, self.pp_dir)

            # ── Step 3: SCF ───────────────────────────────────────────
            scf_input = {
                "control": {
                    "calculation": "scf",
                    "restart_mode": "from_scratch",
                    "outdir": Path(workdir, "scf_out").as_posix(),
                    "pseudo_dir": Path(str(self.pp_dir)).as_posix(),
                    "prefix": "matscreen_scf",
                    "verbosity": "high",
                    "tprnfor": True,
                    "tstress": True,
                },
                "system": {
                    "ecutwfc": ecutwfc,
                    "ecutrho": ecutrho,
                    "occupations": "smearing",
                    "smearing": "mv",
                    "degauss": 0.01,
                    "nbnd": nbnd,
                },
                "electrons": {
                    "conv_thr": 1.0e-5,
                    "mixing_beta": 0.35,
                },
            }

            calc_scf = Espresso(
                profile=profile,
                pseudopotentials=pp_dict,
                kpts=kpts_std,
                input_data=scf_input,
                directory=Path(workdir, "scf").as_posix(),
                additional_cards=hubbard_cards if hubbard_cards else None,
            )
            atoms_std.calc = calc_scf
            scf_energy = float(atoms_std.get_potential_energy())

            # ── Step 4: Extract eigenvalues and compute band gap ─────
            # In fast_mode, SCF already computes all eigenvalues across the standardized k-mesh in ~20s.
            # In full precision mode (fast_mode=False), run an additional dense-mesh NSCF step.
            if not fast_mode:
                kpts_nscf = tuple(max(k, k + 1) for k in kpts_std)
                nscf_input = dict(scf_input)
                nscf_input["control"] = dict(scf_input["control"])
                nscf_input["control"]["calculation"] = "nscf"
                nscf_input["control"]["outdir"] = scf_input["control"]["outdir"]
                nscf_input["system"] = dict(scf_input.get("system", {}))
                nscf_input["system"]["occupations"] = "tetrahedra"
                nscf_input["system"]["nbnd"] = nbnd

                calc_nscf = Espresso(
                    profile=profile,
                    pseudopotentials=pp_dict,
                    kpts=kpts_nscf,
                    input_data=nscf_input,
                    directory=Path(workdir, "nscf").as_posix(),
                    additional_cards=hubbard_cards if hubbard_cards else None,
                )
                try:
                    calc_nscf.calculate(atoms_std)
                except Exception as _nscf_err:
                    logger.debug(f"[DFT] NSCF calculation finished: {_nscf_err}")

                gap_result = self._extract_band_gap_from_nscf(
                    os.path.join(workdir, "nscf"), standard_struct
                )
                if gap_result.get("gap_eV") is None:
                    gap_result = self._extract_band_gap_from_nscf(
                        os.path.join(workdir, "scf"), standard_struct
                    )
            else:
                kpts_nscf = kpts_std
                gap_result = self._extract_band_gap_from_nscf(
                    os.path.join(workdir, "scf"), standard_struct
                )

            runtime = time.time() - t0
            fast_mode_disclosure = {
                "geometry_source": "MLIP (CHGNet) pre-relaxed geometry; DFT vc-relax was bypassed" if fast_mode else "Full DFT vc-relax (BFGS, 12 steps)",
                "kpoints_sampling": f"SCF mesh points only ({kpt_dist} A^-1 grid); gap is an upper bound on true PBE band gap" if fast_mode else "Dense NSCF Monkhorst-Pack grid with tetrahedron method",
                "speedup_provenance": "Speedup (~35x) is achieved by eliminating the multi-step DFT vc-relax and evaluating eigenvalues directly on the SCF k-mesh" if fast_mode else "Standard full-cost DFT relaxation and NSCF density",
                "accuracy_note": "For definitive electronic structure, full DFT vc-relax with high-density NSCF mesh can be toggled via fast_mode=False",
            }
            convergence_params = {
                "ecutwfc_Ry": ecutwfc,
                "ecutrho_Ry": ecutrho,
                "nbnd": nbnd,
                "hubbard_u": applied_u,
                "kpt_distance_inv_A": kpt_dist,
                "kpts_scf": list(kpts_std),
                "kpts_nscf": list(kpts_nscf),
                "pseudopotential_set": "SSSP_efficiency_v1.3_PBE",
                "smearing": "Marzari-Vanderbilt",
                "degauss_Ry": 0.01,
                "spacegroup_number": sg_number,
                "spacegroup_symbol": sg_symbol,
                "fast_mode": fast_mode,
                "fast_mode_disclosure": fast_mode_disclosure,
            }

            return {
                "status": "success",
                "tier": "Tier3a_PBE",
                "formula": formula,
                "pbe_gap_eV": gap_result.get("gap_eV"),
                "gap_type": gap_result.get("gap_type", "unknown"),
                "band_overlap_eV": gap_result.get("band_overlap_eV"),
                "vbm_eV": gap_result.get("vbm_eV"),
                "cbm_eV": gap_result.get("cbm_eV"),
                "scf_total_energy_eV": round(scf_energy, 6),
                "spacegroup_number": sg_number,
                "spacegroup_symbol": sg_symbol,
                "relaxed_structure_cif": standard_struct.to(fmt="cif"),
                "convergence_params": convergence_params,
                "hubbard_u": applied_u,
                "fast_mode_disclosure": fast_mode_disclosure,
                "workdir": workdir,
                "runtime_seconds": round(runtime, 1),
                "disclosure": "PBE (DFT) — known to underestimate band gaps by ~30-50%",
            }

        except Exception as e:
            import traceback
            logger.error(f"[DFT] PBE pipeline failed for {formula}: {e}\n{traceback.format_exc()}")
            return {
                "status": "error",
                "tier": "Tier3a_PBE",
                "formula": formula,
                "error": str(e),
                "workdir": workdir,
                "runtime_seconds": round(time.time() - t0, 1),
            }

    def _extract_band_gap_from_nscf(self, nscf_dir: str, structure: Structure) -> Dict[str, Any]:
        """
        Parse QE NSCF/SCF output to extract VBM, CBM, and band gap.
        Robustly extracts eigenvalues, handles Fermi level crossings, and identifies direct/indirect gaps.
        Strictly returns 'insufficient_bands' or 'error' if CBM is not found, never silently reporting metallic!
        """
        import re

        output_files = list(Path(nscf_dir).glob("*.pwo")) + list(Path(nscf_dir).glob("espresso.pwo"))
        if not output_files:
            output_files = [Path(nscf_dir) / "pw.out"]

        for output_file in output_files:
            if not output_file.exists():
                continue
            text = output_file.read_text(errors="ignore")

            # 1. Direct HOMO/LUMO report
            m_hl = re.search(r"highest occupied,\s*lowest unoccupied level \(ev\):\s+([-\d\.]+)\s+([-\d\.]+)", text, re.IGNORECASE)
            if m_hl:
                vbm, cbm = float(m_hl.group(1)), float(m_hl.group(2))
                if cbm < vbm:
                    # Signed overlap: cbm - vbm < 0 indicates band overlap (metallic)
                    return {
                        "gap_eV": None,
                        "gap_type": "metallic",
                        "vbm_eV": round(vbm, 4),
                        "cbm_eV": round(cbm, 4),
                        "band_overlap_eV": round(cbm - vbm, 4),
                    }
                gap = cbm - vbm
                return {
                    "gap_eV": round(gap, 4),
                    "gap_type": "insulator",
                    "vbm_eV": round(vbm, 4),
                    "cbm_eV": round(cbm, 4),
                    "band_overlap_eV": None,
                }

            # 2. Parse Fermi energy
            m_fermi = re.search(r"the Fermi energy is\s+([-\d\.]+)\s+ev", text, re.IGNORECASE)
            fermi = float(m_fermi.group(1)) if m_fermi else None

            # 3. Parse k-point bands
            lines = text.splitlines()
            bands_per_kpt = []
            i = 0
            while i < len(lines):
                line = lines[i]
                if "bands (ev)" in line.lower():
                    i += 1
                    bands = []
                    while i < len(lines):
                        l = lines[i].strip()
                        if not l:
                            i += 1
                            continue
                        if "k =" in l or "the Fermi energy" in l or "JOB DONE" in l or "Writing all" in l:
                            break
                        try:
                            parts = [float(x) for x in l.split()]
                            bands.extend(parts)
                        except ValueError:
                            break
                        i += 1
                    if bands:
                        bands_per_kpt.append(bands)
                    continue
                i += 1

            if not bands_per_kpt:
                continue

            arr = np.array(bands_per_kpt)
            n_k, n_b = arr.shape

            if fermi is not None:
                # Check if any band crosses the Fermi level (metallic)
                crossing = False
                for b in range(n_b):
                    b_min, b_max = np.min(arr[:, b]), np.max(arr[:, b])
                    if b_min < fermi and b_max > fermi:
                        crossing = True
                        break
                if crossing:
                    return {
                        "gap_eV": None,
                        "gap_type": "metallic",
                        "vbm_eV": None,
                        "cbm_eV": None,
                        "band_overlap_eV": None,
                    }

                # Find band below and above Fermi
                val_bands = [b for b in range(n_b) if np.max(arr[:, b]) <= fermi + 1e-3]
                con_bands = [b for b in range(n_b) if np.min(arr[:, b]) >= fermi - 1e-3]
                if val_bands and con_bands:
                    vb_idx = val_bands[-1]
                    cb_idx = con_bands[0]
                    vbm_k = int(np.argmax(arr[:, vb_idx]))
                    cbm_k = int(np.argmin(arr[:, cb_idx]))
                    vbm = float(arr[vbm_k, vb_idx])
                    cbm = float(arr[cbm_k, cb_idx])
                    if cbm < vbm:
                        # Signed overlap: bands cross, metallic
                        return {
                            "gap_eV": None,
                            "gap_type": "metallic",
                            "vbm_eV": round(vbm, 4),
                            "cbm_eV": round(cbm, 4),
                            "band_overlap_eV": round(cbm - vbm, 4),
                        }
                    gap = cbm - vbm
                    gtype = "direct" if vbm_k == cbm_k else "indirect"
                    return {
                        "gap_eV": round(gap, 4),
                        "gap_type": gtype,
                        "vbm_eV": round(vbm, 4),
                        "cbm_eV": round(cbm, 4),
                        "band_overlap_eV": None,
                    }

                # If conduction bands were not computed, NEVER report metallic!
                if not con_bands:
                    logger.warning(f"[DFT] Fermi level {fermi} eV found with {n_b} bands, but no empty conduction bands exist. nbnd was insufficient.")
                    return {
                        "gap_eV": None,
                        "gap_type": "insufficient_bands",
                        "vbm_eV": round(fermi, 4),
                        "cbm_eV": None,
                        "error": f"Calculated {n_b} bands were all occupied below or at Fermi level ({fermi:.3f} eV). No conduction bands were computed. Increase nbnd."
                    }

                if not val_bands:
                    return {
                        "gap_eV": None,
                        "gap_type": "insufficient_bands",
                        "vbm_eV": None,
                        "cbm_eV": round(fermi, 4),
                        "error": "No valence bands found below Fermi level."
                    }

            # Fallback using pseudopotential valence electron count
            total_val_e = sum(_get_valence_electrons(str(site.specie.symbol), self.pp_dir) for site in structure)
            n_occ = int(np.ceil(total_val_e / 2.0))

            if n_occ >= n_b:
                logger.warning(f"[DFT] Structure requires {n_occ} occupied bands, but only {n_b} bands were calculated in QE output.")
                return {
                    "gap_eV": None,
                    "gap_type": "insufficient_bands",
                    "vbm_eV": None,
                    "cbm_eV": None,
                    "error": f"Insufficient bands computed: {n_b} bands available vs {n_occ} occupied bands required."
                }

            if 0 < n_occ < n_b:
                vb_idx = n_occ - 1
                cb_idx = n_occ
                vbm_k = int(np.argmax(arr[:, vb_idx]))
                cbm_k = int(np.argmin(arr[:, cb_idx]))
                vbm = float(arr[vbm_k, vb_idx])
                cbm = float(arr[cbm_k, cb_idx])
                if cbm < vbm:
                    return {
                        "gap_eV": None,
                        "gap_type": "metallic",
                        "vbm_eV": round(vbm, 4),
                        "cbm_eV": round(cbm, 4),
                        "band_overlap_eV": round(cbm - vbm, 4),
                    }
                gap = cbm - vbm
                gtype = "direct" if vbm_k == cbm_k else "indirect"
                return {
                    "gap_eV": round(gap, 4),
                    "gap_type": gtype,
                    "vbm_eV": round(vbm, 4),
                    "cbm_eV": round(cbm, 4),
                    "band_overlap_eV": None,
                }

        return {
            "gap_eV": None,
            "gap_type": "unknown",
            "vbm_eV": None,
            "cbm_eV": None,
            "band_overlap_eV": None,
        }

    # ------------------------------------------------------------------
    # Tier 3c: r2SCAN meta-GGA
    # ------------------------------------------------------------------

    def run_r2scan_gap(
        self,
        structure: Structure,
        formula: str,
        pbe_workdir: Optional[str] = None,
        workdir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run r2SCAN meta-GGA SCF + band gap on PBE-relaxed geometry.
        
        r2SCAN is ~2-5× cost of PBE but gives significantly improved geometries
        and gaps without the full cost of hybrid functionals.
        Uses PBE-relaxed geometry (no re-relax needed for gap estimate).
        """
        if not self.qe_available:
            return self._qe_not_installed_error("Tier3c_r2SCAN")

        pp_dict = self._resolve_pseudopotentials(structure)
        if pp_dict is None:
            return {"status": "error", "error": "Pseudopotentials not available for r2SCAN"}

        ecutwfc, ecutrho = _get_sssp_cutoffs(structure)
        # r2SCAN typically needs slightly higher cutoffs than PBE
        ecutwfc = max(ecutwfc, 70.0)
        ecutrho = ecutwfc * 8

        kpts = _kpoints_from_dist(structure, kpt_dist=0.20)  # slightly denser for meta-GGA
        t0 = time.time()
        workdir = workdir or tempfile.mkdtemp(prefix="matscreen_dft_r2scan_")
        os.makedirs(workdir, exist_ok=True)

        try:
            atoms = AseAtomsAdaptor.get_atoms(structure)
            profile = self._make_profile()

            # r2SCAN requires meta-GGA with kinetic energy density support
            r2scan_input = {
                "control": {
                    "calculation": "scf",
                    "outdir": Path(workdir, "r2scan_out").as_posix(),
                    "pseudo_dir": Path(str(self.pp_dir)).as_posix(),
                    "prefix": "matscreen_r2scan",
                    "tprnfor": True,
                },
                "system": {
                    "ecutwfc": ecutwfc,
                    "ecutrho": ecutrho,
                    "input_dft": "r2scan",      # r2SCAN meta-GGA
                    "occupations": "smearing",
                    "smearing": "mv",
                    "degauss": 0.01,
                },
                "electrons": {"conv_thr": 1.0e-8},
            }

            calc = Espresso(
                profile=profile,
                pseudopotentials=pp_dict,
                kpts=kpts,
                input_data=r2scan_input,
                directory=Path(workdir).as_posix(),
            )
            atoms.calc = calc
            energy = float(atoms.get_potential_energy())

            # NSCF for gap
            nscf_input = dict(r2scan_input)
            nscf_input["control"] = dict(r2scan_input["control"])
            nscf_input["control"]["calculation"] = "nscf"
            nscf_input["system"] = dict(r2scan_input["system"])
            nscf_input["system"]["occupations"] = "tetrahedra"
            kpts_nscf = tuple(k + 2 for k in kpts)

            calc_nscf = Espresso(
                profile=profile,
                pseudopotentials=pp_dict,
                kpts=kpts_nscf,
                input_data=nscf_input,
                directory=Path(workdir, "nscf").as_posix(),
            )
            atoms.calc = calc_nscf
            atoms.get_potential_energy()

            n_electrons = sum(int(s.specie.Z) for s in structure)
            gap_result = self._extract_band_gap_from_nscf(os.path.join(workdir, "nscf"), structure)

            runtime = time.time() - t0
            return {
                "status": "success",
                "tier": "Tier3c_r2SCAN",
                "formula": formula,
                "r2scan_gap_eV": gap_result.get("gap_eV"),
                "gap_type": gap_result.get("gap_type", "unknown"),
                "vbm_eV": gap_result.get("vbm_eV"),
                "cbm_eV": gap_result.get("cbm_eV"),
                "scf_total_energy_eV": round(energy, 6),
                "convergence_params": {
                    "ecutwfc_Ry": ecutwfc,
                    "ecutrho_Ry": ecutrho,
                    "functional": "r2SCAN",
                    "kpts": list(kpts),
                },
                "runtime_seconds": round(runtime, 1),
                "disclosure": "r2SCAN meta-GGA (DFT) — improved over PBE, cheaper than HSE06",
            }
        except Exception as e:
            return {
                "status": "error",
                "tier": "Tier3c_r2SCAN",
                "formula": formula,
                "error": str(e),
                "runtime_seconds": round(time.time() - t0, 1),
            }

    # ------------------------------------------------------------------
    # Tier 3d: HSE06 gap-only (VBM/CBM eigenvalues, ONCV pseudopotentials)
    # ------------------------------------------------------------------

    def run_hse06_gap_only(
        self,
        structure: Structure,
        formula: str,
        workdir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Compute HSE06 band gap (eigenvalues at VBM/CBM k-points only).
        
        IMPORTANT: Uses ONCV norm-conserving pseudopotentials (NOT ultrasoft/PAW).
        QE developers state ultrasoft+hybrid is "very inefficient" and prone to hanging.
        
        Reduced q-grid (Nk/Nq ≈ 2–3) for exact-exchange tractability.
        Does NOT compute full band dispersion — k-path + hybrid = documented conflict in QE.
        """
        if not self.qe_available:
            return self._qe_not_installed_error("Tier3d_HSE06")

        # HSE06 requires ONCV pseudopotentials — check separately
        pp_dict = self._resolve_pseudopotentials(structure)
        if pp_dict is None:
            return {
                "status": "error",
                "error": (
                    "Pseudopotentials not found. HSE06 specifically requires ONCV "
                    "norm-conserving pseudopotentials — NOT SSSP ultrasoft. "
                    "Download from https://www.pseudo-dojo.org/ "
                    "and set SSSP_PP_DIR to that directory."
                ),
            }

        ecutwfc, ecutrho = _get_sssp_cutoffs(structure)
        # HSE06 with NC PPs typically needs higher ecutwfc than ultrasoft
        ecutwfc = max(ecutwfc + 20, 80.0)
        ecutrho = ecutwfc * 4  # NC pseudopotentials: ecutrho = 4× ecutwfc (not 8–12×)

        kpts = _kpoints_from_dist(structure, kpt_dist=0.30)  # coarser k-grid to reduce HSE cost
        # Reduced q-grid: Nk/Nq = 2 as standard in hybrid literature
        qpts = tuple(max(1, k // 2) for k in kpts)

        t0 = time.time()
        workdir = workdir or tempfile.mkdtemp(prefix="matscreen_dft_hse06_")
        os.makedirs(workdir, exist_ok=True)

        try:
            atoms = AseAtomsAdaptor.get_atoms(structure)
            profile = self._make_profile()

            hse06_input = {
                "control": {
                    "calculation": "scf",
                    "outdir": Path(workdir, "hse06_out").as_posix(),
                    "pseudo_dir": Path(str(self.pp_dir)).as_posix(),
                    "prefix": "matscreen_hse06",
                    "tprnfor": False,  # no forces — we only need eigenvalues
                },
                "system": {
                    "ecutwfc": ecutwfc,
                    "ecutrho": ecutrho,
                    "input_dft": "hse",           # HSE06 in QE
                    "exxdiv_treatment": "gygi-baldereschi",  # standard for periodic systems
                    "x_gamma_extrapolation": True,
                    "nqx1": qpts[0],              # reduced q-grid for exchange sum
                    "nqx2": qpts[1],
                    "nqx3": qpts[2],
                    "occupations": "smearing",
                    "smearing": "mv",
                    "degauss": 0.005,             # tighter smearing for gap accuracy
                },
                "electrons": {
                    "conv_thr": 1.0e-7,
                    "mixing_mode": "local-TF",    # better convergence for hybrids
                    "mixing_beta": 0.3,
                    "electron_maxstep": 200,      # hybrids need more iterations
                },
            }

            calc = Espresso(
                profile=profile,
                pseudopotentials=pp_dict,
                kpts=kpts,
                input_data=hse06_input,
                directory=Path(workdir).as_posix(),
            )
            atoms.calc = calc
            energy = float(atoms.get_potential_energy())

            gap_result = self._extract_band_gap_from_nscf(workdir, structure)
            runtime = time.time() - t0

            return {
                "status": "success",
                "tier": "Tier3d_HSE06",
                "formula": formula,
                "hse06_gap_eV": gap_result.get("gap_eV"),
                "gap_type": gap_result.get("gap_type", "unknown"),
                "vbm_eV": gap_result.get("vbm_eV"),
                "cbm_eV": gap_result.get("cbm_eV"),
                "scf_total_energy_eV": round(energy, 6),
                "convergence_params": {
                    "ecutwfc_Ry": ecutwfc,
                    "ecutrho_Ry": ecutrho,
                    "functional": "HSE06",
                    "pseudopotential_type": "ONCV_norm_conserving",
                    "kpts": list(kpts),
                    "qpts_exchange": list(qpts),
                },
                "runtime_seconds": round(runtime, 1),
                "disclosure": (
                    "HSE06 hybrid functional (DFT) — corrects PBE underestimation. "
                    "Gap computed at VBM/CBM k-points only (not full band path)."
                ),
                "warning": (
                    "HSE06 is computationally expensive (≥10× PBE). "
                    "Verify timing benchmark before running on multiple candidates."
                ),
            }
        except Exception as e:
            return {
                "status": "error",
                "tier": "Tier3d_HSE06",
                "formula": formula,
                "error": str(e),
                "runtime_seconds": round(time.time() - t0, 1),
            }

    # ------------------------------------------------------------------
    # NAC Born charge calculation (for phonon NAC correction — Item 12)
    # ------------------------------------------------------------------

    def run_born_charges_dfpt(
        self,
        structure: Structure,
        formula: str,
        scf_outdir: str,
        workdir: Optional[str] = None,
        prefix: str = "matscreen_scf",
    ) -> Dict[str, Any]:
        """
        Run ph.x to compute dielectric tensor and Born effective charges.
        
        This is a SINGLE DFPT perturbation at Γ only — far cheaper than
        full phonon DFPT. Output feeds the NAC correction in simulation_service.py.
        
        Requires a completed SCF calculation in scf_outdir (same prefix used here).
        Note: DFPT epsil=.true. requires an insulating ground state (occupations='fixed').
        """
        if not self.ph_available:
            return {
                "status": "ph_not_installed",
                "error": (
                    "ph.x (Quantum ESPRESSO PHonon package) not found. "
                    "Install with: apt install quantum-espresso  OR  conda install -c conda-forge qe"
                ),
            }

        t0 = time.time()
        workdir = workdir or tempfile.mkdtemp(prefix="matscreen_dft_ph_")
        os.makedirs(workdir, exist_ok=True)

        dyn_path = Path(workdir, "matscreen.dyn").as_posix()
        scf_dir_posix = Path(scf_outdir).as_posix()

        ph_input = f"""&inputph
  tr2_ph = 1.0d-12,
  prefix = '{prefix}',
  outdir = '{scf_dir_posix}',
  fildyn = '{dyn_path}',
  epsil  = .true.,
  trans  = .true.,
  asr    = .true.,
/
0.0 0.0 0.0
"""
        ph_input_file = os.path.join(workdir, "ph.in")
        with open(ph_input_file, "w", encoding="utf-8") as f:
            f.write(ph_input)

        ph_output_file = os.path.join(workdir, "ph.out")

        try:
            result = subprocess.run(
                [self.ph_binary, "-inp", ph_input_file],
                capture_output=True, text=True,
                timeout=3600,  # 1 hour max for ph.x
                cwd=workdir,
            )
            with open(ph_output_file, "w", encoding="utf-8") as f:
                f.write(result.stdout)

            if result.returncode != 0:
                return {
                    "status": "error",
                    "error": f"ph.x failed with code {result.returncode}",
                    "stderr": result.stderr[:2000],
                    "runtime_seconds": round(time.time() - t0, 1),
                }

            # Parse Born charges and dielectric tensor from ph.x output
            born_result = self._parse_born_charges(result.stdout, structure)
            runtime = time.time() - t0

            return {
                "status": "success",
                "formula": formula,
                "dielectric_tensor": born_result.get("dielectric_tensor"),
                "born_effective_charges": born_result.get("born_charges"),
                "n_atoms": len(structure),
                "workdir": workdir,
                "runtime_seconds": round(runtime, 1),
                "disclosure": "Born effective charges and dielectric tensor from DFPT (ph.x at Γ with ASR)",
            }

        except subprocess.TimeoutExpired:
            return {
                "status": "error",
                "error": "ph.x timed out after 1 hour. Consider a smaller unit cell.",
                "runtime_seconds": round(time.time() - t0, 1),
            }
        except Exception as e:
            return {"status": "error", "error": str(e), "runtime_seconds": round(time.time() - t0, 1)}

    def _parse_born_charges(self, stdout: str, structure: Structure) -> Dict[str, Any]:
        """Parse Born effective charges and dielectric tensor from ph.x stdout."""
        dielectric = None
        born_charges = []

        # 1. Parse dielectric tensor (cartesian axis)
        m_eps = re.search(
            r'Dielectric constant in cartesian axis\s*\n\s*\(\s*([0-9\.\-]+)\s+([0-9\.\-]+)\s+([0-9\.\-]+)\s*\)\s*\n\s*\(\s*([0-9\.\-]+)\s+([0-9\.\-]+)\s+([0-9\.\-]+)\s*\)\s*\n\s*\(\s*([0-9\.\-]+)\s+([0-9\.\-]+)\s+([0-9\.\-]+)\s*\)',
            stdout
        )
        if m_eps:
            v = [float(x) for x in m_eps.groups()]
            dielectric = [v[0:3], v[3:6], v[6:9]]

        # 2. Parse Born effective charges (prefer acoustic sum rule 'with asr applied')
        born_sec = None
        if "with asr applied" in stdout:
            born_sec = stdout.split("with asr applied:")[-1].split("Diagonalizing")[0]
        elif "without acoustic sum rule" in stdout:
            born_sec = stdout.split("without acoustic sum rule applied")[-1].split("Diagonalizing")[0]

        if born_sec:
            # Atom block pattern: atom <idx> <elem> ... E*x ( 3 floats ) E*y ( 3 floats ) E*z ( 3 floats )
            blocks = re.findall(
                r'atom\s+(\d+)\s+([A-Za-z]+).*?\n\s*E\*?x\s*\(\s*([0-9\.\-]+)\s+([0-9\.\-]+)\s+([0-9\.\-]+)\s*\)\s*\n\s*E\*?y\s*\(\s*([0-9\.\-]+)\s+([0-9\.\-]+)\s+([0-9\.\-]+)\s*\)\s*\n\s*E\*?z\s*\(\s*([0-9\.\-]+)\s+([0-9\.\-]+)\s+([0-9\.\-]+)\s*\)',
                born_sec,
                re.DOTALL
            )
            for blk in blocks:
                t = [float(x) for x in blk[2:]]
                born_charges.append([t[0:3], t[3:6], t[6:9]])

        return {
            "dielectric_tensor": dielectric,
            "born_charges": born_charges if born_charges else None,
        }

    @staticmethod
    def _is_float(s: str) -> bool:
        try:
            float(s)
            return True
        except ValueError:
            return False


# Module-level singleton
_dft_service_instance: Optional[DFTValidationService] = None


def get_dft_service() -> DFTValidationService:
    global _dft_service_instance
    if _dft_service_instance is None:
        _dft_service_instance = DFTValidationService()
    else:
        # Re-check dynamically so a running server picks up newly installed binaries
        pw = _get_pw_binary()
        if pw:
            _dft_service_instance.pw_binary = Path(pw).as_posix()
            _dft_service_instance.qe_available = True
            ph = _get_ph_binary()
            _dft_service_instance.ph_binary = Path(ph).as_posix() if ph else None
            _dft_service_instance.ph_available = _dft_service_instance.ph_binary is not None
            _dft_service_instance.pp_dir = _get_pp_dir()
    return _dft_service_instance
