"""
Batch Screening REST API Router for MatScreen AI.
Processes multiple CIF files, ranks candidate materials, and supports confidence filtering.
"""

import time
import uuid
from typing import List, Optional
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.services.cif_parser import CIFParserService
from app.services.predictor import GNNPredictorService
from app.database.db import get_db
from app.database.models import ScreeningJob

router = APIRouter(prefix="/api", tags=["Batch Screening"])


@router.post("/screen")
async def batch_screen_materials(
    files: List[UploadFile] = File(...),
    confidence_filter: Optional[str] = Form("all"),  # all, high, medium, low
    db: Session = Depends(get_db)
):
    """Run batch property prediction and uncertainty screening across multiple CIF structure files."""
    if not files or len(files) == 0:
        raise HTTPException(status_code=400, detail="No CIF files uploaded.")

    start_time = time.time()
    job_id = f"job-{uuid.uuid4().hex[:8]}"

    predictor = GNNPredictorService.get_instance()
    results = []
    high_conf_count = 0

    for file in files:
        filename = file.filename or "structure.cif"
        content_bytes = await file.read()
        content = content_bytes.decode("utf-8", errors="ignore")

        if len(content.strip()) < 10:
            continue

        try:
            parsed = CIFParserService.parse_cif_text(content, filename=filename)
            pred_res = predictor.predict(parsed["pymatgen_structure"])

            if pred_res["confidence"] == "High":
                high_conf_count += 1

            candidate_item = {
                "filename": filename,
                "formula": parsed["formula"],
                "formula_pretty": parsed["formula_pretty"],
                "num_atoms": parsed["num_atoms"],
                "density_g_cm3": parsed["density_g_cm3"],
                "volume_A3": parsed["volume_A3"],
                "predicted_formation_energy_per_atom_eV": pred_res["predicted_formation_energy_per_atom_eV"],
                "evidential_std_eV": pred_res["evidential_std_eV"],
                "aleatoric_std_eV": pred_res["aleatoric_std_eV"],
                "epistemic_std_eV": pred_res["epistemic_std_eV"],
                "conformal_90_interval_eV": pred_res["conformal_90_interval_eV"],
                "conformal_width_eV": pred_res["conformal_width_eV"],
                "confidence": pred_res["confidence"],
                "confidence_score_pct": pred_res["confidence_score_pct"],
                "risk_level": pred_res["risk_level"],
                "recommendation": pred_res["recommendation"],
                "badge_color": pred_res["badge_color"],
                "scale_attention": pred_res["scale_attention"],
                "cif_string": parsed["cif_string"]
            }

            # Confidence filtering
            if confidence_filter == "high" and pred_res["confidence"] != "High":
                continue
            elif confidence_filter == "medium" and pred_res["confidence"] not in ["High", "Medium"]:
                continue

            results.append(candidate_item)
        except Exception as e:
            print(f"[screen.py] Skipped file {filename}: {e}")

    # Rank materials by lowest formation energy per atom (thermodynamically most stable)
    results.sort(key=lambda x: x["predicted_formation_energy_per_atom_eV"])

    # Add rank index
    for i, res in enumerate(results):
        res["rank"] = i + 1

    runtime = round(time.time() - start_time, 2)

    # Save ScreeningJob record to database
    try:
        job_rec = ScreeningJob(
            job_id=job_id,
            title=f"Screening of {len(files)} materials",
            status="completed",
            total_materials=len(files),
            high_confidence_count=high_conf_count,
            runtime_seconds=runtime
        )
        db.add(job_rec)
        db.commit()
    except Exception as e:
        print(f"[screen.py] DB job log failed: {e}")

    return {
        "job_id": job_id,
        "total_files": len(files),
        "screened_count": len(results),
        "high_confidence_count": high_conf_count,
        "runtime_seconds": runtime,
        "candidates": results
    }
