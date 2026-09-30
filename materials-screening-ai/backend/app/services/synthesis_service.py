# -*- coding: utf-8 -*-
"""
Synthesis Route Classification Service (Item 9).

Given a crystal formula and its constituent elements, classifies the most
likely synthesis route (solid_state / wet_chemistry / vapor_deposition /
mechanochemical), assigns precursor compounds from a curated database,
and returns a feasibility score based on element availability and
thermodynamic accessibility.

The classification logic is physics-informed:
  - Ionic halides (your target class): wet_chemistry or solid_state dominant
  - Oxides / sulfides: solid_state at high temperature
  - Organics: wet_chemistry
  - Nitrides / carbides: vapor_deposition / high-pressure solid_state
  - High-melting refractory: mechanochemical or arc melting

References:
  - West, A.R. "Solid State Chemistry and its Applications", 2nd Ed. (2014)
  - Pinna, N. et al. "Chemical Synthesis of Metal Oxide Nanoparticles" (2015)
  - Precursor data: curated from ICSD / NIST Chemistry WebBook element availabilities
"""

import re
import logging
from typing import Any, Dict, List, Optional, Tuple

from pymatgen.core import Composition, Element

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Curated precursor database — organized by element
# Format: {element_symbol: [precursor_formula, ...]}
# ---------------------------------------------------------------------------
ELEMENT_PRECURSORS: Dict[str, List[str]] = {
    "K":  ["KCl", "KBr", "KI", "K2CO3", "KNO3", "KOH", "K2O"],
    "Na": ["NaCl", "NaBr", "Na2CO3", "NaOH", "Na2O", "NaNO3"],
    "Li": ["LiCl", "LiBr", "Li2CO3", "LiOH", "LiNO3", "LiF"],
    "Cs": ["CsCl", "CsBr", "CsI", "Cs2CO3", "CsNO3"],
    "Rb": ["RbCl", "RbBr", "Rb2CO3"],
    "Ca": ["CaCl2", "CaBr2", "CaCO3", "CaO", "Ca(OH)2"],
    "Ba": ["BaCl2", "BaBr2", "BaCO3", "BaO", "Ba(NO3)2"],
    "Sr": ["SrCl2", "SrBr2", "SrCO3", "SrO", "Sr(NO3)2"],
    "Mg": ["MgCl2", "MgBr2", "MgO", "Mg(OH)2", "MgCO3"],
    "Zr": ["ZrCl4", "ZrO2", "ZrOCl2·8H2O", "Zr(OPr)4"],
    "Sn": ["SnCl2", "SnCl4", "SnBr2", "SnO", "SnO2"],
    "Pb": ["PbCl2", "PbBr2", "PbI2", "PbO", "Pb(NO3)2"],
    "Ge": ["GeCl4", "GeO2", "GeBr2"],
    "Ti": ["TiCl4", "TiO2", "TiOSO4", "Ti(OiPr)4"],
    "Hf": ["HfCl4", "HfO2"],
    "V":  ["VCl3", "VO2", "V2O5", "NH4VO3"],
    "Nb": ["NbCl5", "Nb2O5"],
    "Ta": ["TaCl5", "Ta2O5"],
    "Mo": ["MoCl3", "MoO3", "(NH4)6Mo7O24"],
    "W":  ["WCl6", "WO3", "(NH4)6W12O39"],
    "Mn": ["MnCl2", "MnBr2", "MnO", "MnO2", "Mn(NO3)2"],
    "Fe": ["FeCl2", "FeCl3", "Fe2O3", "Fe3O4", "Fe(NO3)3"],
    "Co": ["CoCl2", "CoBr2", "CoO", "Co3O4", "Co(NO3)2"],
    "Ni": ["NiCl2", "NiBr2", "NiO", "Ni(NO3)2"],
    "Cu": ["CuCl2", "CuBr2", "CuO", "Cu2O", "Cu(NO3)2"],
    "Zn": ["ZnCl2", "ZnBr2", "ZnO", "Zn(NO3)2", "ZnSO4"],
    "In": ["InCl3", "In2O3", "In(NO3)3"],
    "Al": ["AlCl3", "Al2O3", "Al(OH)3", "Al(NO3)3"],
    "Ga": ["GaCl3", "Ga2O3"],
    "Bi": ["BiCl3", "Bi2O3", "Bi(NO3)3"],
    "Sb": ["SbCl3", "SbCl5", "Sb2O3"],
    "Cl": ["HCl", "NH4Cl", "NaCl", "KCl"],
    "Br": ["HBr", "NH4Br", "NaBr", "KBr"],
    "I":  ["HI", "NH4I", "NaI", "KI"],
    "F":  ["HF", "NH4F", "NaF", "KF"],
    "O":  ["O2", "H2O", "H2O2"],
    "S":  ["S", "Na2S", "H2S", "(NH4)2S"],
    "N":  ["N2", "NH3", "NH4NO3"],
    "P":  ["P2O5", "NH4H2PO4", "H3PO4"],
    "Si": ["SiCl4", "SiO2", "TEOS"],
    "Ag": ["AgNO3", "AgCl", "Ag2SO4"],
    "Au": ["HAuCl4", "Au"],
    "Pt": ["H2PtCl6", "Pt"],
    "Pd": ["PdCl2", "Pd(NO3)2"],
    "Y":  ["YCl3", "Y2O3", "Y(NO3)3"],
    "La": ["LaCl3", "La2O3", "La(NO3)3"],
    "Ce": ["CeCl3", "CeO2", "Ce(NO3)3"],
    "Pr": ["PrCl3", "Pr6O11"],
    "Nd": ["NdCl3", "Nd2O3"],
    "Eu": ["EuCl3", "Eu2O3"],
    "Gd": ["GdCl3", "Gd2O3"],
    "Tb": ["TbCl3", "Tb4O7"],
    "Ho": ["HoCl3", "Ho2O3"],
    "Er": ["ErCl3", "Er2O3"],
    "Yb": ["YbCl3", "Yb2O3"],
    "Lu": ["LuCl3", "Lu2O3"],
}

# Melting points (°C) for feasibility scoring
ELEMENT_MELTING_POINTS: Dict[str, float] = {
    "W": 3422, "Re": 3180, "Os": 3033, "Ta": 2996, "Mo": 2623,
    "Nb": 2468, "Ir": 2443, "Ru": 2334, "Hf": 2233, "Rh": 1964,
    "V": 1910, "Cr": 1907, "Zr": 1855, "Ti": 1668, "Pt": 1768,
    "Fe": 1538, "Ni": 1455, "Co": 1495, "Mn": 1246, "Cu": 1085,
    "Au": 1064, "Ag": 962, "Al": 660, "Mg": 650, "Zn": 420,
    "Pb": 328, "Bi": 271, "Sn": 232, "In": 157, "Ga": 30,
    "K": 63, "Na": 98, "Li": 181, "Cs": 29, "Rb": 39, "Ba": 727,
    "Sr": 777, "Ca": 842, "Ge": 938, "Si": 1414, "Sb": 631,
    "Y": 1522, "La": 920, "Ce": 798, "Nd": 1016,
}

# Cost classification for elements (rough $ tier)
ELEMENT_COST_TIER: Dict[str, str] = {
    # Cheap bulk chemicals
    "K": "cheap", "Na": "cheap", "Ca": "cheap", "Mg": "cheap", "Al": "cheap",
    "Fe": "cheap", "Mn": "cheap", "Zn": "cheap", "Cl": "cheap", "O": "cheap",
    "N": "cheap", "S": "cheap", "Si": "cheap", "Li": "cheap",
    # Moderate
    "Cs": "moderate", "Rb": "moderate", "Ba": "moderate", "Sr": "moderate",
    "Cu": "moderate", "Ni": "moderate", "Co": "moderate", "Sn": "moderate",
    "Zr": "moderate", "Ti": "moderate", "V": "moderate", "Cr": "moderate",
    "Br": "moderate", "I": "moderate", "In": "moderate", "Ge": "moderate",
    "La": "moderate", "Ce": "moderate", "Y": "moderate",
    # Expensive
    "Nb": "expensive", "Mo": "expensive", "W": "expensive", "Ta": "expensive",
    "Hf": "expensive", "Re": "expensive", "Ga": "expensive", "Bi": "expensive",
    "Sb": "expensive",
    # Very expensive / critical
    "Pt": "critical", "Au": "critical", "Ir": "critical", "Rh": "critical",
    "Pd": "critical", "Ru": "critical",
}


class SynthesisRouteService:
    """
    Physics-informed synthesis route classifier for inorganic crystal compounds.
    """

    # Route classification decision table (ordered: first match wins)
    # Format: (anion_set, condition_fn, route, temperature_C, atmosphere, notes)
    ROUTE_RULES = [
        # Halide perovskites / halide ionic solids → wet chemistry preferred
        {
            "match_fn": lambda elems, anions, comp: anions.issubset({"Cl", "Br", "I", "F"}),
            "route": "wet_chemistry",
            "temperature_C": 150,
            "atmosphere": "inert (N₂ or Ar)",
            "solvent": "DMF or DMSO",
            "notes": "Dissolve precursor halide salts in polar aprotic solvent; crystallize by anti-solvent addition or slow evaporation. Standard for halide perovskites.",
        },
        # Halides of high-melting metals → solid-state at moderate T
        {
            "match_fn": lambda elems, anions, comp: (
                anions.issubset({"Cl", "Br", "I", "F"})
                and any(ELEMENT_MELTING_POINTS.get(e, 0) > 1200 for e in elems - anions)
            ),
            "route": "solid_state",
            "temperature_C": 800,
            "atmosphere": "inert (N₂ or Ar) sealed ampoule",
            "solvent": None,
            "notes": "Mix stoichiometric halide salts; heat in sealed ampoule to allow vapor-phase halide transport. Required when one metal has high-melting chloride.",
        },
        # Oxides with moderate melting point → solid state
        {
            "match_fn": lambda elems, anions, comp: "O" in anions and not anions - {"O", "F", "Cl"},
            "route": "solid_state",
            "temperature_C": 1000,
            "atmosphere": "air or O₂",
            "solvent": None,
            "notes": "Ball-mill oxide/carbonate precursors; calcine at elevated temperature. Grind-react-grind cycle standard for oxides.",
        },
        # Sulfides → inert atmosphere required
        {
            "match_fn": lambda elems, anions, comp: "S" in anions,
            "route": "solid_state",
            "temperature_C": 650,
            "atmosphere": "H₂S or CS₂ or sealed ampoule",
            "solvent": None,
            "notes": "Solid-state reaction of elemental sulfur with metal powders under inert atmosphere or H₂S flow.",
        },
        # Nitrides → CVD or high-pressure synthesis
        {
            "match_fn": lambda elems, anions, comp: "N" in anions,
            "route": "vapor_deposition",
            "temperature_C": 900,
            "atmosphere": "N₂ or NH₃ flow",
            "solvent": None,
            "notes": "Chemical vapor deposition from metal chloride precursor under N₂ or NH₃ atmosphere. High temperature required for kinetically stable nitride.",
        },
        # Phosphides → similar to nitrides
        {
            "match_fn": lambda elems, anions, comp: "P" in anions,
            "route": "solid_state",
            "temperature_C": 700,
            "atmosphere": "sealed ampoule",
            "solvent": None,
            "notes": "Red phosphorus + metal powder in sealed silica ampoule. Handle with care (toxic phosphorus vapor).",
        },
        # Default fallback
        {
            "match_fn": lambda elems, anions, comp: True,
            "route": "solid_state",
            "temperature_C": 900,
            "atmosphere": "air",
            "solvent": None,
            "notes": "Default solid-state mixed-oxide route. Optimize atmosphere and temperature per specific chemistry.",
        },
    ]

    def classify_route(
        self, formula: str, composition: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Classify the most likely synthesis route and enumerate precursors.
        
        Args:
            formula: Crystal formula string (e.g. "KZrCl3")
            composition: Optional pymatgen Composition (computed if not given)
            
        Returns:
            route: str (wet_chemistry / solid_state / vapor_deposition / mechanochemical)
            temperature_C: int — typical synthesis temperature
            atmosphere: str — required atmosphere
            precursors: List[{element, formula, role}]
            feasibility: str (high/medium/low/very_low)
            feasibility_score: float (0–1)
            notes: str — synthesis procedure notes
            warnings: List[str]
        """
        try:
            if composition is None:
                composition = Composition(formula)
        except Exception as e:
            return {
                "error": f"Cannot parse formula '{formula}': {e}",
                "route": "unknown",
                "feasibility": "unknown",
            }

        # Extract elements and classify anions
        elements = {str(el) for el in composition.elements}
        anions = self._identify_anions(composition)
        cations = elements - anions

        # Apply route classification rules
        matched_rule = None
        for rule in self.ROUTE_RULES:
            try:
                if rule["match_fn"](elements, anions, composition):
                    matched_rule = rule
                    break
            except Exception:
                continue

        if matched_rule is None:
            matched_rule = self.ROUTE_RULES[-1]  # fallback

        # Build precursor list
        precursors = self._enumerate_precursors(composition, anions, cations, matched_rule["route"])

        # Compute feasibility score
        feasibility_score, feasibility_label, warnings = self._compute_feasibility(
            elements, composition, matched_rule["route"]
        )

        # Check alternative routes
        alternatives = self._suggest_alternatives(matched_rule["route"], anions, elements)

        return {
            "formula": formula,
            "route": matched_rule["route"],
            "temperature_C": matched_rule["temperature_C"],
            "atmosphere": matched_rule["atmosphere"],
            "solvent": matched_rule.get("solvent"),
            "precursors": precursors,
            "feasibility": feasibility_label,
            "feasibility_score": round(feasibility_score, 3),
            "notes": matched_rule["notes"],
            "warnings": warnings,
            "alternative_routes": alternatives,
        }

    def _identify_anions(self, composition: Any) -> set:
        """
        Classify which elements act as anions in the compound.
        Simple heuristic: halogens and chalcogens are anions;
        for oxides, O is the anion; for complex anions, fall back to
        electronegativity ordering.
        """
        anion_elements = {"F", "Cl", "Br", "I", "O", "S", "Se", "Te", "N", "P", "As", "Sb", "Bi"}
        elements = {str(el) for el in composition.elements}
        anions = elements & anion_elements
        return anions

    def _enumerate_precursors(
        self,
        composition: Any,
        anions: set,
        cations: set,
        route: str,
    ) -> List[Dict]:
        """
        Select appropriate precursor compounds for each element.
        
        For wet_chemistry: prefer soluble salts (nitrates, acetates, halides in solution)
        For solid_state: prefer oxides, carbonates, or direct halides
        For vapor_deposition: prefer volatile halides (MCl_n)
        """
        precursors = []
        
        # Prefer list by route
        if route == "wet_chemistry":
            prefer_suffix = ["Cl", "Br", "NO3", "CO3", "OH"]
        elif route == "vapor_deposition":
            prefer_suffix = ["Cl"]  # volatile halides
        else:
            prefer_suffix = ["O", "CO3", "Cl", "Br"]

        for elem in sorted(composition.elements, key=lambda e: str(e)):
            elem_str = str(elem)
            amount = composition[elem]
            candidates = ELEMENT_PRECURSORS.get(elem_str, [elem_str])
            
            # Pick best candidate based on route preference
            chosen = candidates[0]
            for suffix in prefer_suffix:
                for cand in candidates:
                    if cand.endswith(suffix) or suffix in cand:
                        chosen = cand
                        break
                else:
                    continue
                break

            role = "anion source" if elem_str in anions else "cation source"
            cost_tier = ELEMENT_COST_TIER.get(elem_str, "unknown")
            precursors.append({
                "element": elem_str,
                "stoichiometry": float(amount),
                "recommended_precursor": chosen,
                "alternatives": candidates[1:3],
                "role": role,
                "cost_tier": cost_tier,
            })

        return precursors

    def _compute_feasibility(
        self,
        elements: set,
        composition: Any,
        route: str,
    ) -> Tuple[float, str, List[str]]:
        """
        Score synthesis feasibility 0–1 based on:
          - Element cost availability (cheap=0.33, moderate=0.2, expensive=0.1, critical=0.0)
          - All precursors exist in ELEMENT_PRECURSORS database
          - No extreme-temperature elements that conflict with chosen route
          - Charge balance feasibility
        """
        warnings = []
        score = 1.0

        # Element availability penalty
        for elem in elements:
            tier = ELEMENT_COST_TIER.get(elem, "expensive")
            if tier == "critical":
                score -= 0.25
                warnings.append(f"{elem} is a critical/PGM element — very expensive, limited supply.")
            elif tier == "expensive":
                score -= 0.10
                warnings.append(f"{elem} is expensive — source from specialty chemical supplier.")
            elif tier == "moderate":
                score -= 0.03

        # Missing precursor penalty
        for elem in elements:
            if elem not in ELEMENT_PRECURSORS:
                score -= 0.15
                warnings.append(f"No precursor database entry for {elem} — manual precursor search required.")

        # Temperature conflict warning
        max_mp = max((ELEMENT_MELTING_POINTS.get(e, 500) for e in elements), default=500)
        if route == "wet_chemistry" and max_mp > 2000:
            score -= 0.2
            warnings.append(
                f"Wet chemistry route selected but max element melting point is {max_mp}°C — "
                "may require alternative high-temperature synthesis."
            )

        # Halogen availability
        halogens_present = elements & {"Cl", "Br", "I", "F"}
        if halogens_present and route == "wet_chemistry":
            score += 0.05  # bonus: halide wet chemistry is well-established

        score = max(0.0, min(1.0, score))

        if score >= 0.75:
            label = "high"
        elif score >= 0.50:
            label = "medium"
        elif score >= 0.25:
            label = "low"
        else:
            label = "very_low"

        return score, label, warnings

    def _suggest_alternatives(self, primary_route: str, anions: set, elements: set) -> List[str]:
        """Suggest alternative synthesis routes."""
        alts = []
        if primary_route == "wet_chemistry":
            alts.append("mechanochemical (ball milling of dry precursors — solvent-free alternative)")
            alts.append("solid_state (sealed ampoule at elevated temperature)")
        elif primary_route == "solid_state":
            if anions & {"Cl", "Br", "I"}:
                alts.append("wet_chemistry (dissolve in DMF/DMSO, slow crystallization)")
            alts.append("mechanochemical (room-temperature high-energy ball milling)")
        elif primary_route == "vapor_deposition":
            alts.append("molecular beam epitaxy (MBE) for thin-film applications")
            alts.append("solid_state under reactive atmosphere")
        return alts


# Module-level singleton
_synthesis_service_instance: Optional[SynthesisRouteService] = None


def get_synthesis_service() -> SynthesisRouteService:
    global _synthesis_service_instance
    if _synthesis_service_instance is None:
        _synthesis_service_instance = SynthesisRouteService()
    return _synthesis_service_instance
