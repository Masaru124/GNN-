# -*- coding: utf-8 -*-
"""
Convex Hull Decomposition Stability & S.U.N. Rate Analysis Service.

Computes energy-above-hull (e_above_hull) for candidate structures using
pymatgen PhaseDiagram against Materials Project reference entries,
and reports the field-standard Stable/Unique/Novel (S.U.N.) metric
(MatterGen, CDVAE, CrystalGRW, LeMat-GenBench evaluation framework).

Two fidelity paths:
  - Live MP API (requires MP_API_KEY): real DFT-computed reference entries → defensible hull.
  - Offline fallback: returns 'insufficient_reference_data' rather than computing
    a bogus hull from non-formation-energy data — a wrong e_above_hull is worse
    than a missing one.
"""

import hashlib
import logging
from typing import Any, Dict, List, Optional, Tuple

from pymatgen.core import Composition, Element
from pymatgen.analysis.phase_diagram import PhaseDiagram, PDEntry

logger = logging.getLogger(__name__)

# Standard stability thresholds from the literature (MatterGen, CrystalGRW, LeMat-GenBench)
E_HULL_STRICT_THRESHOLD = 0.0       # On-hull: thermodynamically stable
E_HULL_METASTABLE_THRESHOLD = 0.1   # eV/atom — standard synthesizability cutoff


class StabilityAnalysisService:
    """
    Computes energy-above-hull for candidate materials using pymatgen PhaseDiagram.

    The hull is constructed from Materials Project reference entries for the
    candidate's chemical system (e.g., K-Zr-Cl), with the candidate inserted
    as a hypothetical entry using its GNN/MLIP-predicted formation energy.
    """

    def __init__(self, mp_service=None):
        """
        Args:
            mp_service: MaterialsProjectService instance (from mp_api.py).
                        If None, live MP API calls are attempted via import.
        """
        self._pd_cache: Dict[tuple, List[PDEntry]] = {}
        self.mp_service = mp_service

    def _get_mp_phase_entries(self, elements: List[str]) -> Tuple[List[PDEntry], str]:
        """
        Fetch phase diagram reference entries for a chemical system.

        Returns:
            entries: List of PDEntry objects (empty if unavailable).
            source: 'mp_api_live' | 'insufficient_reference_data'
        """
        cache_key = tuple(sorted(elements))
        if cache_key in self._pd_cache:
            return self._pd_cache[cache_key], "mp_api_live_cached"

        # Attempt live MP API path
        import os
        mp_api_key = os.environ.get("MP_API_KEY", "")

        if mp_api_key:
            try:
                from mp_api.client import MPRester
                chemsys = "-".join(sorted(elements))
                with MPRester(mp_api_key) as mpr:
                    # Fetch thermodynamic entries for the full chemical system
                    # This includes all known compounds in the element space
                    docs = mpr.materials.thermo.search(
                        chemsys=chemsys,
                        fields=[
                            "material_id",
                            "formula_pretty",
                            "energy_per_atom",
                            "composition",
                            "energy_above_hull",
                            "is_stable",
                        ]
                    )

                    if not docs:
                        logger.warning(
                            f"[StabilityAnalysis] No MP entries found for system {chemsys}. "
                            f"Returning insufficient_reference_data."
                        )
                        return [], "insufficient_reference_data"

                    entries = []
                    for doc in docs:
                        try:
                            comp = doc.composition
                            # MP energy_per_atom is total DFT energy per atom;
                            # PDEntry expects total energy for the full composition
                            total_energy = doc.energy_per_atom * comp.num_atoms
                            entry = PDEntry(composition=comp, energy=total_energy)
                            entries.append(entry)
                        except Exception as e:
                            logger.debug(f"[StabilityAnalysis] Skipping MP entry: {e}")
                            continue

                    if entries:
                        self._pd_cache[cache_key] = entries
                        logger.info(
                            f"[StabilityAnalysis] Fetched {len(entries)} MP reference entries "
                            f"for system {chemsys}."
                        )
                        return entries, "mp_api_live"

            except ImportError:
                logger.warning("[StabilityAnalysis] mp_api package not installed.")
            except Exception as e:
                logger.warning(f"[StabilityAnalysis] MP API call failed: {e}")

        # Offline fallback: OFFLINE_MATERIALS_DB does NOT contain formation energies,
        # only band_gap_eV and structural metadata. Constructing a phase diagram
        # from band gaps would produce numerically meaningless hull positions.
        # Return honest "insufficient" rather than computing a bogus number.
        logger.info(
            f"[StabilityAnalysis] No MP API key or API call failed for {elements}. "
            f"Returning insufficient_reference_data — offline DB lacks formation energies "
            f"needed for valid hull construction."
        )
        return [], "insufficient_reference_data"

    def compute_e_above_hull(
        self,
        composition: Composition,
        predicted_formation_energy_per_atom: float,
    ) -> Dict[str, Any]:
        """
        Compute energy above the convex hull for a candidate material.

        Args:
            composition: pymatgen Composition of the candidate.
            predicted_formation_energy_per_atom: GNN/MLIP predicted Ef (eV/atom).

        Returns:
            dict with e_above_hull_eV, classification, decomposition products, and status.
        """
        elements = [el.symbol for el in composition.elements]
        known_entries, source = self._get_mp_phase_entries(elements)

        if not known_entries:
            return {
                "e_above_hull_eV": None,
                "hull_classification": "insufficient_reference_data",
                "decomposition_products": [],
                "num_reference_compounds": 0,
                "data_source": source,
                "status": "insufficient_reference_data",
                "note": (
                    "No reference compounds with formation energies found for this "
                    "chemical system. Hull position cannot be computed. This is NOT "
                    "'stable' — it means 'unknown'. Requires live Materials Project "
                    "API access (MP_API_KEY) for a valid hull computation."
                ),
            }

        # Construct candidate PDEntry from predicted formation energy
        # PDEntry expects TOTAL energy (not per-atom), computed as:
        # E_total = E_f_per_atom * N_atoms
        candidate_total_energy = (
            predicted_formation_energy_per_atom * composition.num_atoms
        )
        candidate_entry = PDEntry(
            composition=composition, energy=candidate_total_energy
        )

        # Build phase diagram with all reference entries + candidate
        all_entries = known_entries + [candidate_entry]

        try:
            pd = PhaseDiagram(all_entries)
            e_hull = pd.get_e_above_hull(candidate_entry)

            # Get decomposition products (what the candidate would decompose into)
            decomp_dict, decomp_energy = pd.get_decomp_and_e_above_hull(candidate_entry)
            decomp_formulas = [
                str(entry.composition.reduced_formula)
                for entry in decomp_dict.keys()
            ]

        except Exception as e:
            logger.error(f"[StabilityAnalysis] PhaseDiagram construction failed: {e}")
            return {
                "e_above_hull_eV": None,
                "hull_classification": "hull_construction_failed",
                "decomposition_products": [],
                "num_reference_compounds": len(known_entries),
                "data_source": source,
                "status": "hull_construction_failed",
                "error": str(e),
                "note": (
                    "Phase diagram construction failed — this can happen when the "
                    "chemical system has too few reference entries to form a valid hull "
                    "in the composition space dimensionality."
                ),
            }

        # Classify hull position using field-standard thresholds
        e_hull_rounded = round(float(e_hull), 4)

        if e_hull_rounded <= 1e-4:
            classification = "on_hull_stable"
        elif e_hull_rounded <= 0.05:
            classification = "likely_synthesizable_metastable"
        elif e_hull_rounded <= E_HULL_METASTABLE_THRESHOLD:
            classification = "borderline_metastable"
        else:
            classification = "likely_unstable_decomposes"

        return {
            "e_above_hull_eV": e_hull_rounded,
            "hull_classification": classification,
            "decomposition_products": decomp_formulas,
            "num_reference_compounds": len(known_entries),
            "data_source": source,
            "status": "computed",
            "note": (
                f"Hull computed from {len(known_entries)} MP reference entries using "
                f"GNN-predicted formation energy ({predicted_formation_energy_per_atom:.4f} eV/atom). "
                f"This is a Tier 1 estimate — the candidate's energy is ML-predicted, "
                f"not DFT-computed, so the hull position carries the model's own uncertainty. "
                f"MP reference entries are DFT-computed (PBE/PBE+U)."
            ),
        }


class DiscoveryRunReportService:
    """
    Computes run-level S.U.N. (Stable/Unique/Novel) and M.S.U.N.
    (Metastable-Stable/Unique/Novel) rates using the field-standard thresholds:

    - S.U.N.: e_above_hull <= 0 eV/atom (strict, on-hull)
    - M.S.U.N.: e_above_hull <= 0.1 eV/atom (relaxed, standard synthesizability cutoff)

    These thresholds match MatterGen, CrystalGRW, and LeMat-GenBench exactly,
    making the reported percentages directly comparable to published results.

    DISCLOSED CAVEATS (must be stated alongside any reported S.U.N. number):
    1. e_above_hull is computed from Tier 1 GNN or Tier 2 MLIP energy, not DFT-relaxed
       energy as published benchmarks use — stability classification carries more
       uncertainty than theirs.
    2. Generation is template-substitution-constrained to 5 scaffolds, a fundamentally
       different (and typically easier) task than unconstrained generative sampling.
    """

    @staticmethod
    def compute_sun_rate(candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compute S.U.N. and M.S.U.N. rates for a list of candidate dicts.

        Each candidate dict is expected to have:
          - e_above_hull_eV (float or None)
          - novelty_status (str): 'novel' or 'known_match'
          - formula (str): reduced formula for deduplication

        Returns:
            dict with SUN/MSUN counts, rates, and the mandatory disclosure note.
        """
        total = len(candidates)
        if total == 0:
            return {
                "total_candidates_evaluated": 0,
                "SUN_rate_pct": 0.0,
                "SUN_count": 0,
                "MSUN_rate_pct": 0.0,
                "MSUN_count": 0,
                "candidates_with_hull_data": 0,
                "candidates_without_hull_data": 0,
                "stability_threshold_note": "",
            }

        # Separate candidates with and without hull data
        with_hull = [
            c for c in candidates if c.get("e_above_hull_eV") is not None
        ]
        without_hull = [
            c for c in candidates if c.get("e_above_hull_eV") is None
        ]

        # 1. STABLE filter (sequential funnel)
        strict_stable = [
            c for c in with_hull if c["e_above_hull_eV"] <= E_HULL_STRICT_THRESHOLD
        ]
        metastable_pool = [
            c for c in with_hull if c["e_above_hull_eV"] <= E_HULL_METASTABLE_THRESHOLD
        ]

        # 2. UNIQUE filter — deduplicate by reduced formula (proxy for WL hash)
        def dedupe_by_formula(cands: List[Dict]) -> List[Dict]:
            seen = set()
            unique = []
            for c in cands:
                formula = c.get("formula", "")
                if formula not in seen:
                    seen.add(formula)
                    unique.append(c)
            return unique

        strict_unique = dedupe_by_formula(strict_stable)
        meta_unique = dedupe_by_formula(metastable_pool)

        # 3. NOVEL filter — exclude known_match
        strict_novel = [
            c for c in strict_unique if c.get("novelty_status") == "novel"
        ]
        meta_novel = [
            c for c in meta_unique if c.get("novelty_status") == "novel"
        ]

        sun_rate = round(100.0 * len(strict_novel) / total, 2) if total else 0.0
        msun_rate = round(100.0 * len(meta_novel) / total, 2) if total else 0.0

        return {
            "total_candidates_evaluated": total,
            "SUN_rate_pct": sun_rate,
            "SUN_count": len(strict_novel),
            "MSUN_rate_pct": msun_rate,
            "MSUN_count": len(meta_novel),
            "candidates_with_hull_data": len(with_hull),
            "candidates_without_hull_data": len(without_hull),
            "funnel_breakdown": {
                "strict_stable": len(strict_stable),
                "strict_stable_unique": len(strict_unique),
                "strict_stable_unique_novel": len(strict_novel),
                "metastable_pool": len(metastable_pool),
                "metastable_unique": len(meta_unique),
                "metastable_unique_novel": len(meta_novel),
            },
            "stability_threshold_note": (
                "S.U.N. uses e_above_hull <= 0 eV/atom (strict, on-hull); "
                "M.S.U.N. relaxes to <= 0.1 eV/atom, the standard synthesizability "
                "cutoff used by MatterGen, CrystalGRW, and LeMat-GenBench — chosen "
                "specifically so this number is directly comparable to published "
                "generative-model results rather than being a bespoke metric. "
                "CAVEAT: e_above_hull here is computed from GNN/MLIP-predicted energy, "
                "not DFT-relaxed; and generation is scaffold-substitution-constrained, "
                "not unconstrained generative sampling. Both factors mean this S.U.N. "
                "rate is not a direct apples-to-apples comparison with published numbers."
            ),
        }
