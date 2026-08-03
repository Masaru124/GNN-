"""
CIF File Parser Service for MatScreen AI.
Extracts structure metadata, formula, lattice constants, and 3D atomic coordinates.
"""

from typing import Any, Dict
from pymatgen.core import Structure
import numpy as np


class CIFParserService:
    @staticmethod
    def parse_cif_text(cif_text: str, filename: str = "structure.cif") -> Dict[str, Any]:
        """Parse CIF text into PyMatGen Structure and metadata payload."""
        try:
            structure = Structure.from_str(cif_text, fmt="cif")
        except Exception as e:
            try:
                structure = Structure.from_str(cif_text, fmt="poscar")
            except Exception:
                raise ValueError(f"Failed to parse structure from text: {str(e)}")

        return CIFParserService.extract_structure_info(structure, filename=filename)

    @staticmethod
    def extract_structure_info(structure: Structure, filename: str = "structure.cif") -> Dict[str, Any]:
        """Extract detailed lattice, chemical, and coordinate info from PyMatGen Structure."""
        formula = structure.composition.reduced_formula
        formula_pretty = structure.composition.formula
        num_atoms = len(structure)
        density = float(round(structure.density, 4))
        volume = float(round(structure.volume, 4))

        lattice = structure.lattice
        lattice_params = {
            "a": float(round(lattice.a, 4)),
            "b": float(round(lattice.b, 4)),
            "c": float(round(lattice.c, 4)),
            "alpha": float(round(lattice.alpha, 2)),
            "beta": float(round(lattice.beta, 2)),
            "gamma": float(round(lattice.gamma, 2)),
            "matrix": [[float(round(v, 4)) for v in row] for row in lattice.matrix.tolist()]
        }

        # Elements breakdown
        elements = [str(el) for el in structure.composition.elements]
        composition_dict = {str(el): int(count) for el, count in structure.composition.as_dict().items()}

        # Site list for 3Dmol visualization & backend graphs
        sites = []
        for i, site in enumerate(structure):
            sites.append({
                "index": i,
                "element": str(site.specie),
                "label": f"{site.specie}{i+1}",
                "cartesian": [float(round(c, 4)) for c in site.coords],
                "fractional": [float(round(f, 4)) for f in site.frac_coords]
            })

        return {
            "filename": filename,
            "formula": formula,
            "formula_pretty": formula_pretty,
            "num_atoms": num_atoms,
            "density_g_cm3": density,
            "volume_A3": volume,
            "elements": elements,
            "composition": composition_dict,
            "lattice": lattice_params,
            "sites": sites,
            "cif_string": structure.to(fmt="cif"),
            "pymatgen_structure": structure
        }
