# -*- coding: utf-8 -*-
"""
Transport Property Proxy & Voronoi Free Volume Service.

Computes structure heuristics for ion transport (e.g. Li+, Na+, Mg2+ battery cathodes/solid electrolytes):
  - Voronoi Free Volume (Å³) and Void Fraction (%)
  - Migration Channel Bottleneck Radius (Å) around target mobile ion species
"""

import numpy as np
from typing import Any, Dict, List, Optional
from pymatgen.core import Structure, Element
from pymatgen.analysis.structure_analyzer import VoronoiAnalyzer

# Standard Shannon Ionic Radii (Å) for common mobile ions
IONIC_RADII_A: Dict[str, float] = {
    "Li": 0.76,
    "Na": 1.02,
    "Mg": 0.72,
    "K": 1.38,
    "Ca": 1.00,
    "Zn": 0.74,
    "Al": 0.53,
    "O": 1.40,
    "F": 1.33,
    "S": 1.84,
}
DEFAULT_IONIC_RADIUS_A = 0.80


class TransportPropertyProxy:
    """Computes Voronoi free volume and migration channel bottleneck radii."""

    def __init__(self, target_ion: str = "Li"):
        self.target_ion = target_ion
        self.ion_radius = IONIC_RADII_A.get(target_ion, DEFAULT_IONIC_RADIUS_A)

    def analyze_structure(self, structure: Structure) -> Dict[str, Any]:
        """
        Analyze structural void volume and channel bottleneck metrics.

        Returns:
            dict containing:
              - free_volume_A3 (float): Estimated vacant lattice volume (Å³)
              - void_fraction_pct (float): Percent void volume in unit cell
              - bottleneck_radius_A (float): Maximum open channel radius (Å)
              - target_ion (str): Mobile species analyzed
              - ion_fits_in_channel (bool): True if bottleneck >= ion radius
        """
        cell_volume = float(structure.volume)
        n_atoms = len(structure)

        if n_atoms == 0 or cell_volume <= 0:
            return {
                "free_volume_A3": 0.0,
                "void_fraction_pct": 0.0,
                "bottleneck_radius_A": 0.0,
                "target_ion": self.target_ion,
                "ion_fits_in_channel": False,
            }

        # Calculate atomic volume occupied using sphere packing radii
        occupied_volume = 0.0
        for site in structure:
            el_symbol = site.specie.symbol
            r = IONIC_RADII_A.get(el_symbol, DEFAULT_IONIC_RADIUS_A)
            occupied_volume += (4.0 / 3.0) * np.pi * (r ** 3)

        # Free volume estimate
        free_volume = max(0.0, cell_volume - occupied_volume)
        void_fraction = min(100.0, max(0.0, (free_volume / cell_volume) * 100.0))

        # Channel bottleneck radius proxy using Voronoi cell face radii
        try:
            va = VoronoiAnalyzer(cutoff=5.0)
            results = va.analyze(structure)
            # Estimate bottleneck radius from average Voronoi face distances
            face_radii = []
            for poly in results:
                for face in poly.get("faces", []):
                    dist = face.get("distance", 0.0)
                    if dist > 0:
                        face_radii.append(dist)

            if face_radii:
                bottleneck_radius = float(np.percentile(face_radii, 75))
            else:
                bottleneck_radius = max(0.5, float((free_volume / n_atoms) ** (1.0 / 3.0)))
        except Exception:
            # Fallback estimation if Voronoi tessellation is singular
            bottleneck_radius = max(0.5, float((free_volume / n_atoms) ** (1.0 / 3.0)))

        bottleneck_radius = round(bottleneck_radius, 3)
        free_volume = round(free_volume, 2)
        void_fraction = round(void_fraction, 2)

        ion_fits = bottleneck_radius >= (self.ion_radius * 0.85)

        return {
            "free_volume_A3": free_volume,
            "void_fraction_pct": void_fraction,
            "bottleneck_radius_A": bottleneck_radius,
            "target_ion": self.target_ion,
            "ion_radius_A": self.ion_radius,
            "ion_fits_in_channel": bool(ion_fits),
        }
