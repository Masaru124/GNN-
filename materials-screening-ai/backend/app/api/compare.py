"""
Multi-Scale vs Single-Scale Model Comparison REST API Router for MatScreen AI.
Supports both FormData (file/cif_text) and JSON payload requests.
"""

from typing import Optional
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field

from app.services.cif_parser import CIFParserService
from app.services.predictor import GNNPredictorService

router = APIRouter(prefix="/api", tags=["Model Comparison"])

DEFAULT_SAMPLE_CIF = """# generated using pymatgen
data_TiO2
_symmetry_space_group_name_H-M   'P 1'
_cell_length_a   4.59300000
_cell_length_b   4.59300000
_cell_length_c   2.95900000
_cell_angle_alpha   90.00000000
_cell_angle_beta   90.00000000
_cell_angle_gamma   90.00000000
_symmetry_Int_Tables_number   1
_chemical_formula_structural   TiO2
_chemical_formula_sum   'Ti2 O4'
loop_
 _symmetry_equiv_pos_site_id
 _symmetry_equiv_pos_as_xyz
  1  'x, y, z'
loop_
 _atom_site_type_symbol
 _atom_site_label
 _atom_site_symmetry_multiplicity
 _atom_site_fract_x
 _atom_site_fract_y
 _atom_site_fract_z
 _atom_site_occupancy
  Ti  Ti0  1  0.00000000  0.00000000  0.00000000  1
  Ti  Ti1  1  0.50000000  0.50000000  0.50000000  1
  O  O2  1  0.30500000  0.30500000  0.00000000  1
  O  O3  1  0.69500000  0.69500000  0.00000000  1
  O  O4  1  0.80500000  0.19500000  0.50000000  1
  O  O5  1  0.19500000  0.80500000  0.50000000  1"""


class CompareRequest(BaseModel):
    cif_text: Optional[str] = Field(None, description="CIF structure text for model comparison")


@router.post("/compare")
async def compare_models(
    request: Request,
    file: Optional[UploadFile] = File(None),
    cif_text: Optional[str] = Form(None)
):
    """Compare Multi-Scale GNN (A7) vs Single-Scale GNN baseline for a given CIF structure."""
    content = ""
    filename = "structure.cif"

    # 1. Handle File Upload
    if file is not None:
        filename = file.filename or "structure.cif"
        content_bytes = await file.read()
        content = content_bytes.decode("utf-8", errors="ignore")

    # 2. Handle Form Field
    if not content.strip() and cif_text is not None and cif_text.strip():
        content = cif_text

    # 3. Handle JSON Body Payload
    if not content.strip():
        try:
            body = await request.json()
            if isinstance(body, dict) and body.get("cif_text"):
                content = str(body["cif_text"]).strip()
        except Exception:
            pass

    # 4. Fallback to Sample CIF if empty
    if not content.strip():
        content = DEFAULT_SAMPLE_CIF

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
