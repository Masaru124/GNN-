# -*- coding: utf-8 -*-
"""
Constraint Parser & Hard Filter Service for Materials Discovery.

Evaluates candidate structures against:
  1. RoHS Toxicity Exclusion List (Pb, Cd, As, Hg, Tl, Be, Cr, Bq).
  2. USGS Mineral Commodity Cost Ceilings ($/kg).
  3. Structural Mass Density Ceilings (g/cm³).
  4. Soft target property ranges & scoring.
"""

from typing import Any, Dict, List, Optional, Tuple
from pymatgen.core import Element, Structure

# RoHS Restricted & Hazardous Element Categories
ROHS_TOXIC_ELEMENTS = {
    "Pb", "Cd", "As", "Hg", "Tl", "Be", "Cr", "Bq", "U", "Th", "Pu", "Ra"
}

# USGS Mineral Commodity Price Reference (Approximate USD / kg, 2024 data)
# Elements with high abundance/cheap: Fe, Al, Si, Mg, Ca, Na, K, Ti, C, O, S, P
# Precious/Expensive: Pt, Au, Ir, Rh, Ru, Re, Sc, Te, Ge, Li, Co
USGS_ELEMENT_COST_USD_KG: Dict[str, float] = {
    "H": 2.50, "Li": 45.00, "Be": 850.00, "B": 4.50, "C": 0.15, "N": 0.10, "O": 0.05, "F": 2.00,
    "Na": 0.35, "Mg": 3.20, "Al": 2.40, "Si": 1.80, "P": 1.50, "S": 0.20, "Cl": 0.15,
    "K": 0.45, "Ca": 2.10, "Sc": 3500.00, "Ti": 8.50, "V": 25.00, "Cr": 11.00, "Mn": 2.20,
    "Fe": 0.12, "Co": 32.00, "Ni": 16.50, "Cu": 8.80, "Zn": 2.70, "Ga": 350.00, "Ge": 1400.00,
    "As": 2.00, "Se": 28.00, "Br": 4.50, "Rb": 1200.00, "Sr": 6.50, "Y": 35.00, "Zr": 35.00,
    "Nb": 45.00, "Mo": 42.00, "Ru": 15000.00, "Rh": 140000.00, "Pd": 35000.00, "Ag": 850.00,
    "Cd": 3.50, "In": 240.00, "Sn": 26.00, "Sb": 12.00, "Te": 75.00, "I": 35.00, "Cs": 1800.00,
    "Ba": 2.50, "La": 4.50, "Ce": 3.50, "Pr": 75.00, "Nd": 85.00, "Sm": 4.50, "Eu": 250.00,
    "Gd": 35.00, "Tb": 1200.00, "Dy": 320.00, "Ho": 650.00, "Er": 45.00, "Tm": 2200.00,
    "Yb": 25.00, "Lu": 1400.00, "Hf": 950.00, "Ta": 180.00, "W": 35.00, "Re": 3200.00,
    "Os": 12000.00, "Ir": 150000.00, "Pt": 31000.00, "Au": 68000.00, "Hg": 35.00, "Tl": 85.00,
    "Pb": 2.10, "Bi": 11.50, "Th": 450.00, "U": 120.00
}
DEFAULT_UNLISTED_ELEMENT_COST = 50.0  # $/kg fallback for unlisted elements


class ConstraintParser:
    """Parses user query constraints and applies hard/soft filters."""

    def __init__(
        self,
        exclude_toxic: bool = True,
        custom_toxic_elements: Optional[List[str]] = None,
        max_cost_usd_kg: Optional[float] = None,
        max_density_g_cm3: Optional[float] = None,
        target_property_min: Optional[float] = None,
        target_property_max: Optional[float] = None,
    ):
        self.exclude_toxic = exclude_toxic
        self.toxic_set = set(custom_toxic_elements) if custom_toxic_elements else ROHS_TOXIC_ELEMENTS
        self.max_cost_usd_kg = max_cost_usd_kg
        self.max_density_g_cm3 = max_density_g_cm3
        self.target_property_min = target_property_min
        self.target_property_max = target_property_max

    def calculate_estimated_cost(self, structure: Structure) -> float:
        """Calculate weighted average raw material cost ($/kg) for the crystal composition."""
        comp = structure.composition
        total_mass = comp.weight
        if total_mass <= 0:
            return 0.0

        weighted_cost = 0.0
        for el, amt in comp.items():
            el_symbol = el.symbol
            el_mass_fraction = (el.atomic_mass * amt) / total_mass
            el_cost = USGS_ELEMENT_COST_USD_KG.get(el_symbol, DEFAULT_UNLISTED_ELEMENT_COST)
            weighted_cost += el_mass_fraction * el_cost

        return round(float(weighted_cost), 2)

    def calculate_density(self, structure: Structure) -> float:
        """Calculate mass density in g/cm³."""
        return round(float(structure.density), 3)

    def evaluate_structure(
        self,
        structure: Structure,
        predicted_property: Optional[float] = None
    ) -> Tuple[bool, List[str], Dict[str, Any]]:
        """
        Evaluate a candidate crystal structure against hard filters.

        Returns:
            pass_filter (bool): True if structure passes all hard filters.
            reasons (List[str]): List of rejection reasons if failed.
            metrics (Dict[str, Any]): Evaluated metrics (cost, density, toxicity status).
        """
        reasons = []
        comp = structure.composition
        elements_present = [el.symbol for el in comp.elements]

        # 1. Toxicity Exclusion Filter
        has_toxic = False
        toxic_matches = []
        if self.exclude_toxic:
            for el in elements_present:
                if el in self.toxic_set:
                    has_toxic = True
                    toxic_matches.append(el)
            if has_toxic:
                reasons.append(f"Contains restricted toxic element(s): {', '.join(toxic_matches)}")

        # 2. Cost Ceiling Filter
        cost = self.calculate_estimated_cost(structure)
        if self.max_cost_usd_kg is not None and cost > self.max_cost_usd_kg:
            reasons.append(f"Estimated cost ${cost:.2f}/kg exceeds limit of ${self.max_cost_usd_kg:.2f}/kg")

        # 3. Density Ceiling Filter
        density = self.calculate_density(structure)
        if self.max_density_g_cm3 is not None and density > self.max_density_g_cm3:
            reasons.append(f"Density {density:.2f} g/cm³ exceeds limit of {self.max_density_g_cm3:.2f} g/cm³")

        # 4. Property Range Filter (if predicted property is passed)
        if predicted_property is not None:
            if self.target_property_min is not None and predicted_property < self.target_property_min:
                reasons.append(f"Predicted property {predicted_property:.3f} is below min target {self.target_property_min:.3f}")
            if self.target_property_max is not None and predicted_property > self.target_property_max:
                reasons.append(f"Predicted property {predicted_property:.3f} is above max target {self.target_property_max:.3f}")

        pass_filter = len(reasons) == 0
        metrics = {
            "cost_usd_kg": cost,
            "density_g_cm3": density,
            "has_toxic_elements": has_toxic,
            "toxic_elements": toxic_matches,
            "elements": elements_present,
            "formula": comp.reduced_formula,
        }

        return pass_filter, reasons, metrics
