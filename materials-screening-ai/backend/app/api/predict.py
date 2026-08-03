"""
Single Prediction REST API Router for MatScreen AI.
Supports CIF file upload or raw CIF text input and returns property predictions with calibrated confidence.
"""

import json
from typing import Optional
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.services.cif_parser import CIFParserService
from app.services.predictor import GNNPredictorService
from app.database.db import get_db
from app.database.models import PredictionRecord

router = APIRouter(prefix="/api", tags=["Prediction"])


class SinglePredictRequest(BaseModel):
    cif_text: str
    filename: Optional[str] = "structure.cif"


@router.post("/predict")
async def predict_single(
    file: Optional[UploadFile] = File(None),
    cif_text: Optional[str] = Form(None),
    db: Session = Depends(get_db)
):
    """Predict material formation energy per atom with evidential uncertainty & conformal calibration."""
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

    if len(content.strip()) < 10:
        raise HTTPException(status_code=400, detail="CIF content is empty or invalid.")

    # Parse CIF structure
    try:
        parsed = CIFParserService.parse_cif_text(content, filename=filename)
        structure = parsed["pymatgen_structure"]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse CIF crystal structure: {str(e)}")

    # Run GNN Model Prediction
    try:
        predictor = GNNPredictorService.get_instance()
        pred_res = predictor.predict(structure)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"GNN Inference failed: {str(e)}")

    # Assemble response
    response_payload = {
        "material_info": {
            "filename": parsed["filename"],
            "formula": parsed["formula"],
            "formula_pretty": parsed["formula_pretty"],
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

    # Save prediction log to SQLite database
    try:
        rec = PredictionRecord(
            filename=parsed["filename"],
            formula=parsed["formula"],
            predicted_energy_eV=pred_res["predicted_formation_energy_per_atom_eV"],
            evidential_std_eV=pred_res["evidential_std_eV"],
            confidence=pred_res["confidence"],
            risk_level=pred_res["risk_level"],
            conformal_interval_json=json.dumps(pred_res["conformal_90_interval_eV"]),
            scale_attention_json=json.dumps(pred_res["scale_attention"]),
            cif_content=parsed["cif_string"]
        )
        db.add(rec)
        db.commit()
    except Exception as e:
        print(f"[predict.py] Database log failed: {e}")

    return response_payload
