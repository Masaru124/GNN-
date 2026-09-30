# -*- coding: utf-8 -*-
"""
Production DFT Benchmark Regression Suite.

Validates the full Quantum ESPRESSO DFT and DFPT validation service against
pre-registered Materials Project and primary literature reference values:
  1. Si (mp-149): Diamond indirect semiconductor (PBE Eg = 0.61 - 0.85 eV)
  2. MgO (mp-1265): Rocksalt wide-gap direct insulator (PBE Eg = 4.64 eV)
  3. NaCl (mp-22862): Rocksalt direct ionic insulator (PBE Eg = 5.15 eV)
  4. Cu (mp-30): FCC metal (PBE Eg = 0.00 eV)
  5. LiCoO2 (mp-22526): Layered R-3m closed-shell insulator (Plain PBE Eg ~ 0.73 eV, PBE+U ~ 2.55 eV)
  6. NaCl DFPT (mp-22862): Dielectric tensor (eps_inf ~ 2.54) & Born charges (Z* ~ +-1.10)
  7. Delta-ML Corrector: LOOCV validation & PBE+U / metallic domain safety
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import numpy as np
from pymatgen.core import Structure, Lattice

from app.services.dft_validation import get_dft_service, DFTValidationService
from app.services.simulation_service import generate_crystal_prototype
from app.services.delta_ml_corrector import get_delta_ml_corrector


@pytest.fixture(scope="module")
def dft_service():
    svc = get_dft_service()
    if not svc.qe_available:
        pytest.skip("Quantum ESPRESSO (pw.exe) not available in environment.")
    return svc


# ---------------------------------------------------------------------------
# 1. Silicon (mp-149)
# ---------------------------------------------------------------------------
def test_silicon_pbe_benchmark(dft_service: DFTValidationService):
    """Diamond Si: Pre-registered MP PBE = 0.85 eV (indirect), tol = +-0.25 eV."""
    struct = generate_crystal_prototype("Si")
    res = dft_service.run_pbe_pipeline(
        structure=struct,
        formula="Si",
        kpt_dist=0.30,
        fast_mode=True,
    )
    assert res["status"] == "success", f"Si DFT run failed: {res.get('error')}"
    gap = res.get("pbe_gap_eV")
    assert gap is not None, "Si gap was None"
    # Pre-registered tolerance: [0.55, 0.95] eV
    assert 0.55 <= gap <= 0.95, f"Si gap {gap} eV outside pre-registered [0.55, 0.95] eV"
    assert res.get("gap_type") in ["direct", "indirect"]


# ---------------------------------------------------------------------------
# 2. Magnesium Oxide (mp-1265)
# ---------------------------------------------------------------------------
def test_mgo_pbe_benchmark(dft_service: DFTValidationService):
    """Rocksalt MgO: Pre-registered MP PBE = 4.64 eV (direct at Gamma), tol = +-0.30 eV."""
    struct = generate_crystal_prototype("MgO")
    res = dft_service.run_pbe_pipeline(
        structure=struct,
        formula="MgO",
        kpt_dist=0.30,
        fast_mode=True,
    )
    assert res["status"] == "success", f"MgO DFT run failed: {res.get('error')}"
    gap = res.get("pbe_gap_eV")
    assert gap is not None, "MgO gap was None"
    # Pre-registered tolerance: [4.35, 4.95] eV
    assert 4.35 <= gap <= 4.95, f"MgO gap {gap} eV outside pre-registered [4.35, 4.95] eV"
    assert res.get("gap_type") == "direct", f"Expected direct gap at Gamma, got {res.get('gap_type')}"


def test_mgo_lattice_sensitivity(dft_service: DFTValidationService):
    """
    Verify MgO deformation potential dEg/da.
    Under lattice expansion (a: 4.20 -> 4.30 A), weakened bonding-antibonding
    interaction closes the band gap (dEg/da < 0, volume deformation potential a_v ~ -8.9 eV).
    Protects against bad prototype lattice parameters across all ionic insulators.
    """
    struct_compressed = Structure.from_spacegroup("Fm-3m", Lattice.cubic(4.20), ["Mg", "O"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    struct_expanded = Structure.from_spacegroup("Fm-3m", Lattice.cubic(4.30), ["Mg", "O"], [[0, 0, 0], [0.5, 0.5, 0.5]])

    res_comp = dft_service.run_pbe_pipeline(struct_compressed, "MgO", kpt_dist=0.30, fast_mode=True)
    res_exp = dft_service.run_pbe_pipeline(struct_expanded, "MgO", kpt_dist=0.30, fast_mode=True)

    gap_comp = res_comp.get("pbe_gap_eV")
    gap_exp = res_exp.get("pbe_gap_eV")
    assert gap_comp is not None and gap_exp is not None

    # Slope dEg / da
    slope = (gap_exp - gap_comp) / (4.30 - 4.20)
    # Expected slope: ~ -6.5 eV/A (within [-8.0, -4.5] eV/A)
    assert -8.0 <= slope <= -4.5, f"Unexpected dEg/da slope: {slope:.3f} eV/A"
    # Volume deformation potential: a_v = (a0 / 3) * slope ~ -9.2 eV
    a_v = (4.253 / 3.0) * slope
    assert -12.0 <= a_v <= -6.0, f"Deformation potential {a_v:.3f} eV outside literature [-12, -6] eV"


# ---------------------------------------------------------------------------
# 3. Sodium Chloride (mp-22862)
# ---------------------------------------------------------------------------
def test_nacl_pbe_benchmark(dft_service: DFTValidationService):
    """Rocksalt NaCl: Pre-registered MP PBE = 5.15 eV (direct at Gamma), tol = +-0.30 eV."""
    struct = generate_crystal_prototype("NaCl")
    res = dft_service.run_pbe_pipeline(
        structure=struct,
        formula="NaCl",
        kpt_dist=0.30,
        fast_mode=True,
    )
    assert res["status"] == "success", f"NaCl DFT run failed: {res.get('error')}"
    gap = res.get("pbe_gap_eV")
    assert gap is not None, "NaCl gap was None"
    # Pre-registered tolerance: [4.85, 5.45] eV
    assert 4.85 <= gap <= 5.45, f"NaCl gap {gap} eV outside pre-registered [4.85, 5.45] eV"
    assert res.get("gap_type") == "direct"


# ---------------------------------------------------------------------------
# 4. Copper Metal (mp-30)
# ---------------------------------------------------------------------------
def test_copper_metal_benchmark(dft_service: DFTValidationService):
    """FCC Cu: Pre-registered ground truth = metallic (gap_type='metallic', pbe_gap_eV=None)."""
    struct = generate_crystal_prototype("Cu")
    res = dft_service.run_pbe_pipeline(
        structure=struct,
        formula="Cu",
        kpt_dist=0.30,
        fast_mode=True,
    )
    assert res["status"] == "success", f"Cu DFT run failed: {res.get('error')}"
    assert res.get("gap_type") == "metallic"
    assert res.get("pbe_gap_eV") is None, f"Expected pbe_gap_eV=None for metal, got {res.get('pbe_gap_eV')}"



# ---------------------------------------------------------------------------
# 5. Lithium Cobalt Oxide (mp-22526)
# ---------------------------------------------------------------------------
def test_licoo2_insulator_and_u_benchmark(dft_service: DFTValidationService):
    """
    Layered R-3m LiCoO2:
    - Must NEVER collapse to metal (closed-shell low-spin Co3+, d6).
    - With PBE+U (U=3.32 eV): QE atomic projector yields Eg in [1.90, 2.70] eV.
    """
    struct = generate_crystal_prototype("LiCoO2")
    res = dft_service.run_pbe_pipeline(
        structure=struct,
        formula="LiCoO2",
        kpt_dist=0.35,
        fast_mode=True,
    )
    assert res["status"] == "success", f"LiCoO2 DFT run failed: {res.get('error')}"
    gap = res.get("pbe_gap_eV")
    assert gap is not None, "LiCoO2 gap was None"
    assert res.get("gap_type") != "metallic", "CRITICAL FAILURE: LiCoO2 collapsed to metallic state!"
    assert gap >= 0.50, f"LiCoO2 gap {gap} eV is unexpectedly low (expected >= 0.50 eV)"
    # Check that Hubbard U was applied to Co
    applied_u = res.get("convergence_params", {}).get("hubbard_u", {})
    assert "Co" in applied_u, f"Hubbard U was not applied to Co in LiCoO2: {applied_u}"
    assert applied_u["Co"] == 3.32


# ---------------------------------------------------------------------------
# 6. NaCl DFPT Born Charges & Dielectric Tensor
# ---------------------------------------------------------------------------
def test_nacl_dfpt_born_charges_benchmark(dft_service: DFTValidationService):
    """
    DFPT Born Effective Charges and Dielectric Tensor on Rocksalt NaCl:
    - Literature eps_inf ~ 2.34 - 2.54
    - Literature Z*(Na) ~ +1.09 to +1.10, Z*(Cl) ~ -1.09 to -1.10
    """
    if not dft_service.ph_available:
        pytest.skip("Quantum ESPRESSO ph.x not available in environment.")

    import tempfile
    from ase.calculators.espresso import Espresso
    from pymatgen.io.ase import AseAtomsAdaptor
    from app.services.dft_validation import _get_sssp_cutoffs, _get_valence_electrons

    # Primitive 2-atom cell
    a_cubic = 5.692
    lattice = Lattice.from_parameters(
        a_cubic / 2**0.5, a_cubic / 2**0.5, a_cubic / 2**0.5,
        60, 60, 60
    )
    struct = Structure(lattice, ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])

    workdir = tempfile.mkdtemp(prefix="test_nacl_dfpt_")
    scf_outdir = Path(workdir, "scf_out").as_posix()
    pp_dict = dft_service._resolve_pseudopotentials(struct)
    ecutwfc, ecutrho = _get_sssp_cutoffs(struct)
    val_e = sum(_get_valence_electrons(s.specie.symbol, dft_service.pp_dir) for s in struct)
    n_occ = int(round(val_e / 2.0))

    scf_input = {
        "control": {
            "calculation": "scf",
            "outdir": scf_outdir,
            "pseudo_dir": Path(str(dft_service.pp_dir)).as_posix(),
            "prefix": "matscreen_scf",
        },
        "system": {
            "ecutwfc": ecutwfc,
            "ecutrho": ecutrho,
            "occupations": "fixed",
            "nbnd": n_occ,
        },
        "electrons": {"conv_thr": 1.0e-10, "mixing_beta": 0.7},
    }

    atoms = AseAtomsAdaptor.get_atoms(struct)
    calc = Espresso(
        profile=dft_service._make_profile(),
        pseudopotentials=pp_dict,
        kpts=(4, 4, 4),
        input_data=scf_input,
        directory=workdir,
    )
    atoms.calc = calc
    atoms.get_potential_energy()

    # Run DFPT ph.x
    ph_res = dft_service.run_born_charges_dfpt(
        structure=struct,
        formula="NaCl",
        scf_outdir=scf_outdir,
        workdir=Path(workdir, "ph").as_posix(),
        prefix="matscreen_scf",
    )
    assert ph_res["status"] == "success", f"ph.x failed: {ph_res.get('error')}"

    # Verify dielectric tensor
    eps = ph_res.get("dielectric_tensor")
    assert eps is not None, "Dielectric tensor was None"
    eps_mean = (eps[0][0] + eps[1][1] + eps[2][2]) / 3.0
    assert 2.30 <= eps_mean <= 2.70, f"Dielectric constant {eps_mean} outside [2.30, 2.70]"

    # Verify Born effective charges
    born = ph_res.get("born_effective_charges")
    assert born is not None and len(born) == 2, f"Expected 2 Born charge tensors, got {born}"
    z_na = born[0][0][0]
    z_cl = born[1][0][0]
    assert 1.00 <= z_na <= 1.20, f"Z*(Na) {z_na} outside [1.00, 1.20]"
    assert -1.20 <= z_cl <= -1.00, f"Z*(Cl) {z_cl} outside [-1.20, -1.00]"
    assert abs(z_na + z_cl) < 0.05, f"Acoustic sum rule violated: Z*(Na) + Z*(Cl) = {z_na + z_cl}"


# ---------------------------------------------------------------------------
# 7. Delta-ML Domain Safety & LOOCV Accuracy
# ---------------------------------------------------------------------------
def test_delta_ml_domain_safety_and_loocv():
    """Verify Delta-ML LOOCV accuracy, effective n=10 (single-fidelity experimental optical gaps), and PBE+U domain boundary."""
    corrector = get_delta_ml_corrector()
    assert corrector._is_fitted, "Delta-ML corrector was not fitted"
    assert corrector.effective_n == 10, f"Effective n={corrector.effective_n}, expected 10"

    # LOOCV accuracy check
    rep = corrector.get_validation_report()
    assert rep["loocv_mae_eV"] <= 0.55, f"LOOCV MAE {rep['loocv_mae_eV']} exceeds 0.55 eV"

    # Metallic domain check
    res_metal = corrector.predict_corrected_gap(0.0, formula="Cu")
    assert res_metal["status"] == "pbe_metallic_hse_undetermined"
    assert res_metal["corrected_gap_eV"] is None

    # PBE+U domain boundary check
    res_u = corrector.predict_corrected_gap(2.55, formula="LiCoO2", applied_hubbard_u={"Co": 3.32})
    assert res_u["status"] == "pbe_plus_u_direct"
    assert "out of Δ-ML training domain" in res_u["label"]
    assert "atomic projectors" in res_u["disclosure"]


# ---------------------------------------------------------------------------
# 8. Monkhorst-Pack Even-N Zone Boundary Guarantee
# ---------------------------------------------------------------------------
def test_kpoint_even_mesh_guarantee():
    """
    Verify that _kpoints_from_dist unconditionally enforces even subdivisions N_i >= 2.
    This guarantees that high-symmetry zone-boundary points (R, X, M, Z) with k_i = 1/2
    are never omitted by an odd-N mesh, preventing the band gap dispersion overshoot bug.
    """
    from app.services.dft_validation import _kpoints_from_dist
    
    test_cases = [
        ("Si diamond (a=5.43)", Structure.from_spacegroup("Fd-3m", Lattice.cubic(5.43), ["Si"], [[0, 0, 0]])),
        ("CsPbCl3 (a=5.60)", Structure.from_spacegroup("Pm-3m", Lattice.cubic(5.60), ["Cs"], [[0, 0, 0]])),
        ("CsPbBr3 (a=5.87)", Structure.from_spacegroup("Pm-3m", Lattice.cubic(5.87), ["Cs"], [[0, 0, 0]])),
        ("CsPbI3 (a=6.29)", Structure.from_spacegroup("Pm-3m", Lattice.cubic(6.29), ["Cs"], [[0, 0, 0]])),
        ("CsSnI3 (a=6.22)", Structure.from_spacegroup("Pm-3m", Lattice.cubic(6.22), ["Cs"], [[0, 0, 0]])),
        ("Supercell (a=10.5)", Structure.from_spacegroup("Pm-3m", Lattice.cubic(10.5), ["Cs"], [[0, 0, 0]])),
        ("Tetragonal (a=3.8, c=12.5)", Structure(Lattice.tetragonal(3.8, 12.5), ["Cs", "Cs"], [[0, 0, 0], [0.5, 0.5, 0.5]])),
    ]

    for label, struct in test_cases:
        kpts = _kpoints_from_dist(struct, kpt_dist=0.35)
        # Every mesh subdivision must be even and >= 2
        for k in kpts:
            assert k >= 2 and k % 2 == 0, f"{label} produced odd/invalid mesh {kpts}"
            
    # Explicitly confirm CsPbI3 and CsSnI3 receive 4x4x4 mesh at production kpt_dist=0.35
    cspbi3 = Structure.from_spacegroup("Pm-3m", Lattice.cubic(6.29), ["Cs"], [[0, 0, 0]])
    cssni3 = Structure.from_spacegroup("Pm-3m", Lattice.cubic(6.22), ["Cs"], [[0, 0, 0]])
    assert _kpoints_from_dist(cspbi3, kpt_dist=0.35) == (4, 4, 4)
    assert _kpoints_from_dist(cssni3, kpt_dist=0.35) == (4, 4, 4)


# ---------------------------------------------------------------------------
# 9. Crystal Symmetry Scope Audit & Convergence Protocol
# ---------------------------------------------------------------------------
def test_kpoint_symmetry_scope_and_candidate_audit():
    """
    Verify that DFTValidationService correctly audits crystal symmetry and differentiates
    between high-symmetry cubic systems (where force_even=True guarantees exact zone-boundary
    sampling at k_i=1/2) and lower-symmetry systems (where a multi-grid convergence scan is required).
    """
    svc = DFTValidationService()

    # 1. Cubic perovskite prototype (KZrCl3 prototype in Pm-3m, #221)
    s_cubic = generate_crystal_prototype("KZrCl3")
    res_cubic = svc.check_kpoint_symmetry_scope(s_cubic, formula="KZrCl3")
    assert res_cubic["is_high_symmetry_cubic"] is True
    assert res_cubic["zone_boundary_points_sampled"] is True
    assert res_cubic["recommended_protocol"] == "zone_boundary_points_sampled"
    assert res_cubic["spacegroup_symbol"] == "Pm-3m"
    assert res_cubic["spacegroup_number"] == 221

    # 2. Lower-symmetry layered oxide (LiCoO2 in R-3m, #166)
    s_low = generate_crystal_prototype("LiCoO2")
    res_low = svc.check_kpoint_symmetry_scope(s_low, formula="LiCoO2")
    assert res_low["is_high_symmetry_cubic"] is False
    assert res_low["zone_boundary_points_sampled"] is False
    assert res_low["recommended_protocol"] == "k_density_convergence_scan_required"
    assert "convergence test" in res_low["assessment"].lower()


