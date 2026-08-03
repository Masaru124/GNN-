"""
Multi-Scale vs Single-Scale Model Comparison REST API Router for MatScreen AI.
"""

from typing import Optional
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.services.cif_parser import CIFParserService
from app.services.predictor import GNNPredictorService

router = APIRouter(prefix="/api", tags=["Model Comparison"])


@router.post("/compare")
async def compare_models(
    file: Optional[UploadFile] = File(None),
    cif_text: Optional[str] = Form(None)
):
    """Compare Multi-Scale GNN (A7) vs Single-Scale GNN baseline for a given CIF structure."""
    content = ""
    filename = "structure.cif"

    if file is not None:
        filename = file.filename or "structure.cif"
        content_bytes = await file.read()
        content = content_bytes.decode("utf-8", errors="ignore")
    elif cif_text is not None and cif_text.strip():
        content = cif_text
    else:
        raise HTTPException(status_code=400, detail="Either a CIF file upload or cif_text must be provided.")

    try:
        parsed = CIFParserService.parse_cif_text(content, filename=filename)
        predictor = GNNPredictorService.get_instance()
        comp_res = predictor.compare_single_vs_multi(parsed["pymatgen_structure"])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Comparison failed: {str(e)}")

    return {
        "material_info": {
            "filename": parsed["filename"],
            "formula": parsed["formula"],
            "formula_pretty": parsed["formula_pretty"],
            "num_atoms": parsed["num_atoms"],
            "density_g_cm3": parsed["density_g_cm3"],
            "volume_A3": parsed["volume_A3"]
        },
        "comparison": comp_res
    }
