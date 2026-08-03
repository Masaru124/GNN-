"""
Materials Search REST API Router for MatScreen AI.
Integrates Materials Project API search with single-click screening.
"""

from typing import Optional
from fastapi import APIRouter, HTTPException, Query

from app.services.mp_api import MaterialsProjectService
from app.services.cif_parser import CIFParserService
from app.services.predictor import GNNPredictorService

router = APIRouter(prefix="/api", tags=["Search"])


@router.get("/search")
def search_materials(
    formula: Optional[str] = Query("", description="Chemical formula (e.g. TiO2, LiFePO4)"),
    element: Optional[str] = Query("", description="Chemical element (e.g. Ti, Fe)")
):
    """Search crystal materials from database / Materials Project."""
    results = MaterialsProjectService.search_materials(formula=formula, element=element)
    return {
        "query": {"formula": formula, "element": element},
        "count": len(results),
        "materials": results
    }


@router.get("/search/{material_id}/screen")
def screen_searched_material(material_id: str):
    """Perform 1-click GNN property screening on a searched material."""
    mat = MaterialsProjectService.get_material_by_id(material_id)
    if not mat or not mat.get("cif_string"):
        raise HTTPException(status_code=404, detail="Material structure not found")

    parsed = CIFParserService.parse_cif_text(mat["cif_string"], filename=f"{mat['formula']}.cif")
    predictor = GNNPredictorService.get_instance()
    pred_res = predictor.predict(parsed["pymatgen_structure"])

    return {
        "material_info": {
            "material_id": mat["material_id"],
            "filename": f"{mat['formula']}.cif",
            "formula": mat["formula"],
            "formula_pretty": mat["formula_pretty"],
            "num_atoms": parsed["num_atoms"],
            "density_g_cm3": parsed["density_g_cm3"],
            "volume_A3": parsed["volume_A3"],
            "elements": parsed["elements"],
            "lattice": parsed["lattice"],
            "sites": parsed["sites"],
            "cif_string": parsed["cif_string"]
        },
        "prediction": pred_res
    }
