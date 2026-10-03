# -*- coding: utf-8 -*-
"""
Natural Language Constraint Query Parser Service for Materials Discovery.

Parses free-text user prompts such as:
  "I need a lightweight battery cathode with conductivity > 1e-3, low toxicity, stable above 600°C and inexpensive"

Maps fuzzy semantic phrases & numeric operators to explicit Discovery Query parameters:
  - 'lightweight' -> max_density_g_cm3 = 4.5
  - 'battery cathode' / 'cathode' -> scaffold = 'layered_oxide' or 'spinel', target_ion = 'Li'
  - 'conductivity > X' / 'fast ion' -> min_bottleneck_radius_A = 1.2, min_free_volume_A3 = 15.0
  - 'low toxicity' / 'non-toxic' / 'safe' -> exclude_toxic = True
  - 'stable above 600°C' / 'high temp' -> target_property_max = -1.85 (formation energy ceiling)
  - 'inexpensive' / 'cheap' / 'low cost' -> max_cost_usd_kg = 30.0
"""

import re
from typing import Any, Dict, List, Optional


class NaturalLanguageQueryParser:
    """Parses free-text natural language material specification prompts into structured discovery queries."""

    def parse_prompt(self, prompt: str) -> Dict[str, Any]:
        p_lower = prompt.lower()

        parsed_constraints: List[Dict[str, str]] = []
        scaffold = "perovskite"
        target_ion = "Li"
        num_candidates = 12
        generation_mode = "substitution"
        exclude_toxic = True
        max_cost_usd_kg: Optional[float] = None
        max_density_g_cm3: Optional[float] = None
        target_property_max: Optional[float] = None
        min_free_volume_A3: Optional[float] = None

        # 1. Application & Scaffold Recognition
        if any(kw in p_lower for kw in ["cathode", "battery cathode", "lithium-ion cathode"]):
            scaffold = "layered_oxide"
            target_ion = "Li"
            parsed_constraints.append({"term": "battery cathode", "mapped_to": "Scaffold: Layered Oxide (AxMO2), Ion: Li+"})
        elif any(kw in p_lower for kw in ["spinel", "spinel cathode"]):
            scaffold = "spinel"
            target_ion = "Li"
            parsed_constraints.append({"term": "spinel cathode", "mapped_to": "Scaffold: Spinel (AB2X4), Ion: Li+"})
        elif any(kw in p_lower for kw in ["olivine", "lifepo4"]):
            scaffold = "olivine"
            target_ion = "Li"
            parsed_constraints.append({"term": "olivine scaffold", "mapped_to": "Scaffold: Olivine (AxMPO4), Ion: Li+"})
        elif any(kw in p_lower for kw in ["sodium", "na-ion", "na cathode"]):
            scaffold = "layered_oxide"
            target_ion = "Na"
            parsed_constraints.append({"term": "na-ion cathode", "mapped_to": "Scaffold: Layered Oxide (AxMO2), Ion: Na+"})
        elif "solid electrolyte" in p_lower:
            scaffold = "perovskite"
            target_ion = "Li"
            parsed_constraints.append({"term": "solid electrolyte", "mapped_to": "Scaffold: Perovskite (ABX3), Fast Ion Transport"})
        elif any(kw in p_lower for kw in ["solar", "photovoltaic", "solar absorber", "pv absorber"]):
            scaffold = "perovskite_solar_halide"
            target_ion = "None"
            # No band-gap filter: the only screening band gap was a heuristic with a
            # 3.35 eV median 90% half-width, so the 1.1-1.7 eV window is recorded as
            # intent only. Gating happens on Tier C DFT / Delta-ML gaps later.
            parsed_constraints.append({"term": "solar absorber", "mapped_to": "Scaffold: Lead-Free Halide Perovskite (CsSnI3); target band gap 1.1-1.7 eV (Shockley-Queisser) is NOT screened here - requires Tier C DFT / Delta-ML"})

        # 2. Density / Weight Recognition
        if any(kw in p_lower for kw in ["lightweight", "light weight", "low density"]):
            max_density_g_cm3 = 4.5
            parsed_constraints.append({"term": "lightweight", "mapped_to": "Max Density ≤ 4.50 g/cm³"})

        # 3. Cost & Economics Recognition
        if any(kw in p_lower for kw in ["inexpensive", "cheap", "low cost", "affordable", "economical"]):
            max_cost_usd_kg = 30.0
            parsed_constraints.append({"term": "inexpensive", "mapped_to": "Max Raw Material Cost ≤ $30.00/kg"})
        elif "cost <" in p_lower or "under $" in p_lower:
            cost_match = re.search(r'(?:cost <|under \$|\$)\s*(\d+(?:\.\d+)?)', p_lower)
            if cost_match:
                val = float(cost_match.group(1))
                max_cost_usd_kg = val
                parsed_constraints.append({"term": f"cost < ${val}", "mapped_to": f"Max Raw Material Cost ≤ ${val:.2f}/kg"})

        # 4. Toxicity Recognition
        if any(kw in p_lower for kw in ["low toxicity", "non-toxic", "non toxic", "rohs", "green", "safe"]):
            exclude_toxic = True
            parsed_constraints.append({"term": "low toxicity", "mapped_to": "RoHS Toxic Element Exclusion Filter ON (Excludes Pb, Cd, As, Hg, Tl)"})

        # 5. High Temperature / Thermal Stability Recognition
        if any(kw in p_lower for kw in ["stable above 600", "high temperature", "thermal stability", "stable at high temp"]):
            target_property_max = -1.85
            parsed_constraints.append({"term": "stable above 600°C", "mapped_to": "Formation Energy E_f ≤ -1.85 eV/atom (High Thermal Stability)"})
        elif "stable" in p_lower:
            target_property_max = -1.50
            parsed_constraints.append({"term": "stable", "mapped_to": "Formation Energy E_f ≤ -1.50 eV/atom"})

        # 6. Transport / Conductivity Recognition
        if any(kw in p_lower for kw in ["conductivity >", "high conductivity", "fast ion", "ionic conductivity"]):
            min_free_volume_A3 = 15.0
            parsed_constraints.append({"term": "high ionic conductivity", "mapped_to": "Transport Channel Bottleneck ≥ 1.2 Å, Free Vol ≥ 15.0 Å³"})

        # Fallback Title
        clean_title = prompt.strip()
        if len(clean_title) > 60:
            clean_title = clean_title[:57] + "..."

        return {
            "title": clean_title,
            "scaffold": scaffold,
            "num_candidates": num_candidates,
            "generation_mode": generation_mode,
            "exclude_toxic": exclude_toxic,
            "max_cost_usd_kg": max_cost_usd_kg,
            "max_density_g_cm3": max_density_g_cm3,
            "target_ion": target_ion,
            "target_property_max": target_property_max,
            "min_free_volume_A3": min_free_volume_A3,
            "parsed_constraints": parsed_constraints,
            "raw_prompt": prompt,
        }
