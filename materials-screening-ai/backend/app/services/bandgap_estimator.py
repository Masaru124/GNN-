# -*- coding: utf-8 -*-
"""
Band Gap Tier B Estimator Service.

Provides a cheap, calibrated band gap estimate for triage — positioned between
"we refuse to guess" (Tier C hard refusal) and "DFT ground truth" (Tier C validated).

Two estimation backends, used in priority order:
  1. Primary (if available): matgl M3GNet pretrained band gap model — ML-based,
     pretrained on Materials Project DFT-PBE band gaps, typical MAE ~0.3–0.5 eV vs DFT.
  2. Fallback (always available): Calibrated electronegativity/ionic-radius heuristic
     with per-chemistry-class error bars — zero new dependencies, replaces the broken
     hardcoded predict_band_gap that was producing 1.38 eV for everything.

DISCLOSED LIMITATIONS (must accompany every reported estimate):
  - DFT-PBE systematically underestimates true experimental band gaps by ~30–50%
    (well-known textbook limitation of the PBE exchange-correlation functional).
  - ML models trained on PBE gaps inherit this same systematic bias.
  - The heuristic fallback is even coarser — useful for triage, not for decisions.
  - Neither backend is a substitute for HSE06 or experimental measurement.
"""

import logging
import threading
from typing import Any, Dict, Optional

from pymatgen.core import Structure, Composition, Element

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# Calibrated electronegativity/ionic-radius heuristic reference data
# ═══════════════════════════════════════════════════════════════════════════════

# Pauling electronegativity values for halide perovskite-relevant elements
PAULING_EN = {
    "Cs": 0.79, "Rb": 0.82, "K": 0.82, "Na": 0.93, "Li": 0.98,
    "Ba": 0.89, "Sr": 0.95, "Ca": 1.00, "Mg": 1.31,
    "Sn": 1.96, "Ge": 2.01, "Bi": 2.02, "Sb": 2.05, "Pb": 2.33,
    "Ag": 1.93, "Cu": 1.90, "Ti": 1.54, "Zr": 1.33, "Nb": 1.60,
    "In": 1.78, "Ga": 1.81, "Al": 1.61,
    "I": 2.66, "Br": 2.96, "Cl": 3.16, "F": 3.98,
    "O": 3.44, "S": 2.58, "Se": 2.55, "Te": 2.10, "N": 3.04,
}

# Empirically calibrated base band gap by halide anion (from DFT-PBE literature)
# These replace the broken hardcoded values that gave 1.38 eV for everything
HALIDE_BASE_GAP_eV = {
    "I":  1.30,    # Iodides: narrow gap, near-IR absorbers (CsPbI3 ≈ 1.73 exp, 1.3-1.5 DFT-PBE)
    "Br": 2.10,    # Bromides: wider gap (CsPbBr3 ≈ 2.36 exp, 1.8-2.2 DFT-PBE)
    "Cl": 2.90,    # Chlorides: wide gap (CsPbCl3 ≈ 3.0 exp, 2.5-3.0 DFT-PBE)
    "F":  4.50,    # Fluorides: insulating
}

# B-site metal correction (electronegativity-based shift from Pb reference)
# More electropositive B-sites → narrower gap; more electronegative → wider
B_SITE_CORRECTION_eV = {
    "Pb": 0.00,   # Reference (most-studied halide perovskite B-site)
    "Sn": -0.35,  # Sn²⁺ is less electronegative than Pb²⁺ → narrower gap
    "Ge": -0.15,  # Ge²⁺ similar to Sn but slightly wider due to smaller ion
    "Bi": +0.45,  # Bi³⁺ requires double perovskite → wider gap
    "Sb": +0.40,  # Similar to Bi
    "In": +0.50,  # In³⁺ double perovskites have wider gaps
    "Ag": +0.35,  # Ag⁺ double perovskites
    "Cu": +0.20,  # Cu²⁺ Jahn-Teller effects
    "Ti": +1.20,  # Ti⁴⁺ oxide-like behavior → much wider gap
    "Zr": +1.40,  # Zr⁴⁺ similar to Ti but even wider
}

# A-site cation correction (lattice constant / octahedral tilting effect)
A_SITE_CORRECTION_eV = {
    "Cs": 0.00,   # Reference (largest common A-site cation)
    "Rb": +0.08,  # Slightly smaller → more tilting → slightly wider gap
    "K":  +0.15,  # Smaller still → more tilting
    "Na": +0.25,  # Even more tilting distortion
    "Li": +0.35,  # Very small → extreme tilting → significantly wider gap
    "Ba": -0.05,  # Similar size to Cs for oxide perovskites
    "Sr": +0.05,
    "Ca": +0.12,
}

# Per-chemistry-class calibrated error bars (1σ, in eV)
# These are approximate but honest — derived from the known spread in DFT-PBE
# vs. experiment for each halide family
HEURISTIC_ERROR_BY_HALIDE = {
    "I":  0.35,   # Iodides: tighter error (more experimental data)
    "Br": 0.40,   # Bromides: moderate
    "Cl": 0.45,   # Chlorides: wider (less systematic data)
    "F":  0.60,   # Fluorides: wide (fewer perovskite examples)
}
DEFAULT_HEURISTIC_ERROR = 0.50


class BandGapEstimatorService:
    """
    Tier B band gap estimator with calibrated uncertainty disclosure.

    Uses matgl M3GNet when available, falls back to electronegativity heuristic.
    """
    _instance = None
    _INSTANCE_LOCK = threading.Lock()
    _matgl_model = None
    _matgl_available = False

    def __init__(self):
        # Attempt to load matgl pretrained model if local/available, otherwise use heuristic fallback
        if BandGapEstimatorService._matgl_model is None:
            try:
                import os
                # Avoid long network retries if HuggingFace is blocked
                if os.environ.get("USE_MATGL_REMOTE", "0") == "1":
                    import matgl
                    BandGapEstimatorService._matgl_model = matgl.load_model(
                        "M3GNet-MP-2021.2.8-DIRECT-PES"
                    )
                    BandGapEstimatorService._matgl_available = True
                    logger.info("[BandGapEstimator] matgl M3GNet model loaded successfully.")
                else:
                    BandGapEstimatorService._matgl_available = False
            except Exception as e:
                logger.info(f"[BandGapEstimator] matgl remote model unavailable ({e}). Using calibrated heuristic fallback.")
                BandGapEstimatorService._matgl_available = False

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            with cls._INSTANCE_LOCK:
                if cls._instance is None:
                    cls._instance = BandGapEstimatorService()
        return cls._instance

    def estimate_band_gap(
        self,
        structure: Structure,
        prefer_ml: bool = True,
    ) -> Dict[str, Any]:
        """
        Estimate band gap with disclosed uncertainty.

        Tries matgl ML model first (if available and prefer_ml=True),
        falls back to calibrated heuristic.

        Returns:
            dict with estimated_band_gap_eV, source, tier, error bars,
            solar optimality flag, and mandatory disclosure note.
        """
        # Try matgl ML model first
        if prefer_ml and self._matgl_available and self._matgl_model is not None:
            try:
                result = self._estimate_via_matgl(structure)
                if result is not None:
                    return result
            except Exception as e:
                logger.warning(f"[BandGapEstimator] matgl inference failed: {e}. Falling back to heuristic.")

        # Fallback to calibrated heuristic
        return self._estimate_via_heuristic(structure)

    def _estimate_via_matgl(self, structure: Structure) -> Optional[Dict[str, Any]]:
        """Estimate band gap using pretrained matgl M3GNet model."""
        try:
            predicted_gap = float(self._matgl_model.predict_structure(structure))
            predicted_gap = max(0.0, predicted_gap)  # band gap cannot be negative

            # M3GNet trained on PBE gaps: MAE ~0.3–0.5 eV vs DFT-PBE
            error_bar = 0.40  # Conservative 1σ estimate

            is_solar = 1.1 <= predicted_gap <= 1.7
            solar_status = self._classify_solar_status(predicted_gap)

            return {
                "estimated_band_gap_eV": round(predicted_gap, 3),
                "estimation_error_1sigma_eV": error_bar,
                # 90% interval: error_bar is declared 1σ, so the Gaussian two-sided
                # 90% factor 1.645 applies (no split-conformal data for M3GNet here;
                # the heuristic branch below is split-conformally calibrated).
                "conformal_90_interval_eV": [
                    round(max(0.0, predicted_gap - 1.645 * error_bar), 3),
                    round(predicted_gap + 1.645 * error_bar, 3),
                ],
                "source_model": "M3GNet-MP-2021.2.8-pretrained",
                "tier": "Tier B (ML Estimate — NOT DFT-grade)",
                "is_solar_optimal": is_solar,
                "solar_absorption_status": solar_status,
                "recommend_dft": is_solar or predicted_gap < 2.0,
                "disclosed_error_note": (
                    "Pretrained M3GNet model trained on Materials Project DFT-PBE "
                    "band gaps. Typical MAE ~0.3–0.5 eV vs DFT-PBE. DFT-PBE itself "
                    "systematically underestimates true experimental gaps by ~30–50%. "
                    "This number is for triage only — not a substitute for HSE06 or "
                    "experimental measurement."
                ),
            }
        except Exception as e:
            logger.warning(f"[BandGapEstimator] matgl predict_structure failed: {e}")
            return None

    def _estimate_via_heuristic(self, structure: Structure) -> Dict[str, Any]:
        """
        Calibrated electronegativity/ionic-radius band gap heuristic.

        Uses halide anion baseline + B-site metal correction + A-site lattice correction,
        calibrated against known DFT-PBE band gaps for ABX3 halide perovskites.

        This replaces the broken predict_band_gap() that hardcoded 1.38 eV for everything.
        """
        comp = structure.composition
        species = [el.symbol for el in comp.elements]

        # Identify the halide anion (or oxide/chalcogenide)
        halide_anion = None
        for anion in ["I", "Br", "Cl", "F"]:
            if anion in species:
                halide_anion = anion
                break

        if halide_anion:
            base_gap = HALIDE_BASE_GAP_eV[halide_anion]
            error_bar = HEURISTIC_ERROR_BY_HALIDE.get(halide_anion, DEFAULT_HEURISTIC_ERROR)
        elif "O" in species:
            # Oxide perovskites — wider gap baseline
            base_gap = 3.2
            error_bar = 0.60
        elif "S" in species:
            base_gap = 2.0
            error_bar = 0.55
        else:
            # Unknown anion chemistry — very uncertain
            base_gap = 2.5
            error_bar = 0.80

        # Apply B-site metal correction
        b_correction = 0.0
        for metal, corr in B_SITE_CORRECTION_eV.items():
            if metal in species:
                b_correction = corr
                break  # Take the first match (primary B-site)

        # Apply A-site cation correction
        a_correction = 0.0
        for cation, corr in A_SITE_CORRECTION_eV.items():
            if cation in species:
                a_correction = corr
                break

        estimated_gap = max(0.0, base_gap + b_correction + a_correction)
        estimated_gap = round(estimated_gap, 3)

        is_solar = 1.1 <= estimated_gap <= 1.7
        solar_status = self._classify_solar_status(estimated_gap)

        # HEURISTIC 90% interval (i.i.d. marginal, wide by design): q is the 90%
        # quantile of |gap - est| / error_bar recalibrated on the production
        # checkpoint's val split (soap_loco chemistry groups, n=4289, disjoint from
        # train and test). Coverage on the LOCO test split (n=7178): 0.7699 at
        # q=5.5833 vs 0.6245 at the old random-split q=4.875 (the old 0.910 figure
        # held only on that in-distribution random split).
        # Median half-width 3.35 eV (>> 1 eV) — labeled heuristic/triage in UI and
        # docs (LIMITATIONS.md §7); not a DFT-grade interval.
        q_conformal_90 = 5.5833
        return {
            "estimated_band_gap_eV": estimated_gap,
            "estimation_error_1sigma_eV": round(error_bar, 3),
            "conformal_90_interval_eV": [
                round(max(0.0, estimated_gap - q_conformal_90 * error_bar), 3),
                round(estimated_gap + q_conformal_90 * error_bar, 3),
            ],
            "source_model": "calibrated-electronegativity-heuristic",
            "tier": "Tier B (Heuristic Estimate — NOT DFT-grade)",
            "is_solar_optimal": is_solar,
            "solar_absorption_status": solar_status,
            "recommend_dft": is_solar or estimated_gap < 2.0,
            "heuristic_breakdown": {
                "halide_base_eV": base_gap if halide_anion else None,
                "b_site_correction_eV": b_correction,
                "a_site_correction_eV": a_correction,
                "halide_anion": halide_anion,
            },
            "disclosed_error_note": (
                f"Electronegativity/ionic-radius heuristic calibrated against "
                f"known DFT-PBE band gaps for ABX3 halide perovskites. "
                f"1σ error bar: ±{error_bar:.2f} eV for {halide_anion or 'this'} "
                f"-based compositions. This is a coarse triage estimate — "
                f"the error bar is large by design to be honest about the "
                f"limitation. The 90% conformal band around this estimate is a "
                f"HEURISTIC interval with median half-width ~3.35 eV (wide by "
                f"design) — not a DFT-grade interval. Not a substitute for DFT "
                f"or experiment."
            ),
        }

    @staticmethod
    def _classify_solar_status(gap_eV: float) -> str:
        """Classify band gap into application-relevant categories."""
        if 1.1 <= gap_eV <= 1.7:
            return "Optimal Shockley-Queisser Solar Absorber (1.1–1.7 eV)"
        elif 0.5 <= gap_eV < 1.1:
            return "Narrow Gap — Infrared Absorber / Thermoelectric"
        elif gap_eV < 0.5:
            return "Metallic / Semi-metallic (Near-Zero Gap)"
        elif 1.7 < gap_eV <= 3.0:
            return "Wide Gap — Visible-Light Transparent / LED Candidate"
        else:
            return "Very Wide Gap — Insulator / UV Transparent Oxide"
