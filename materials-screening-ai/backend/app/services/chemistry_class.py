"""
Chemistry grouping for conformal coverage reporting and the n_cal gate.

The conformal interval shipped by the GNN is a single marginal quantile fitted on
the production soap_loco validation split (n=4289). Marginal validity does not
transfer to every chemistry class: measured LOCO coverage differs by class (see
research/loco_cross_conformal/), so any class-specific claim requires a minimum
number of calibration points. MIN_N_CAL encodes that requirement and
N_CAL_TABLE records how many calibration points each class actually has.

Classes (first match wins):
  pb_halide              Pb + halogen (I/Br/Cl)
  halide_other           halogen without Pb
  transition_metal_oxide oxide + transition metal
  alkaline_earth_oxide   oxide + alkaline earth, no transition metal
  other_oxide            oxide, neither of the above
  other                  everything else
"""

from __future__ import annotations

from typing import Dict, Optional

from pymatgen.core import Composition

CLASSES = (
    "pb_halide",
    "halide_other",
    "transition_metal_oxide",
    "alkaline_earth_oxide",
    "other_oxide",
    "other",
)

# Minimum calibration points before a class may carry its own coverage claim.
MIN_N_CAL = 30

# Calibration-point counts measured on the production checkpoint's val split
# (soap_loco chemistry groups, i < max_structures=50000, n=4289 total).
# Regenerate with: python crystal_gnn/scripts/coverage_reports.py n-cal
# (writes research/coverage_reports/n_cal_by_class.json)
N_CAL_TABLE: Dict[str, int] = {
    "pb_halide": 25,
    "halide_other": 641,
    "transition_metal_oxide": 1057,
    "alkaline_earth_oxide": 67,
    "other_oxide": 226,
    "other": 2273,
}

HALOGENS = {"F", "Cl", "Br", "I"}
PB_HALOGEN_HALOGENS = {"Cl", "Br", "I"}
TRANSITION_METALS = {
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
}
ALKALI = {"Li", "Na", "K", "Rb", "Cs"}
ALKALINE_EARTH = {"Be", "Mg", "Ca", "Sr", "Ba", "Ra"}
POST_TRANSITION = {"Pb", "Sn", "Ge", "Bi", "Sb", "In", "Tl"}


def classify(formula: Optional[str]) -> str:
    """Return the chemistry class for a formula string (falls back to 'other')."""
    if not formula:
        return "other"
    try:
        comp = Composition(str(formula).replace("MA", "CH3NH3").replace("FA", "CH5N2"))
        elements = {el.symbol for el in comp.elements}
    except Exception:
        return "other"

    halogens = elements & HALOGENS
    has_oxygen = "O" in elements
    has_tm = bool(elements & TRANSITION_METALS)
    has_ae = bool(elements & ALKALINE_EARTH)

    if halogens and "Pb" in elements:
        return "pb_halide"
    if halogens:
        return "halide_other"
    if has_oxygen and has_tm:
        return "transition_metal_oxide"
    if has_oxygen and has_ae:
        return "alkaline_earth_oxide"
    if has_oxygen:
        return "other_oxide"
    return "other"


def n_cal(chem_class: str) -> Optional[int]:
    """Calibration points available for a class (None if the class is unknown)."""
    return N_CAL_TABLE.get(chem_class)


def class_calibration_status(chem_class: str) -> Dict[str, object]:
    """
    Gate for class-level claims.

    Returns n_cal, the threshold, and a status:
      calibrated        n_cal >= MIN_N_CAL -> class-level statements allowed
      under_calibrated  0 < n_cal < MIN_N_CAL -> marginal coverage only
      no_calibration    class absent from the calibration split
    """
    n = n_cal(chem_class)
    if n is None:
        status = "no_calibration"
        n = 0
    elif n >= MIN_N_CAL:
        status = "calibrated"
    elif n > 0:
        status = "under_calibrated"
    else:
        status = "no_calibration"
    return {
        "chemistry_class": chem_class,
        "n_cal": n,
        "min_n_cal": MIN_N_CAL,
        "status": status,
        "claim_scope": (
            "marginal coverage only (q pooled over all classes)"
            if status != "calibrated"
            else "class-level coverage permitted"
        ),
    }


if __name__ == "__main__":  # pragma: no cover - manual inspection helper
    for f in ("CsPbI3", "MAPbBr3", "NaCl", "TiCl4", "MgO", "SrTiO3", "Fe2O3", "LiCoO2", "Si"):
        print(f"{f:10s} -> {classify(f):24s} {class_calibration_status(classify(f))}")
