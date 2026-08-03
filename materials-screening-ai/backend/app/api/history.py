"""
Prediction History REST API Router for MatScreen AI.
Retrieves past prediction logs and screening jobs.
"""

import json
from typing import List
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.database.db import get_db
from app.database.models import PredictionRecord, ScreeningJob

router = APIRouter(prefix="/api", tags=["History"])


@router.get("/history")
def get_prediction_history(db: Session = Depends(get_db)):
    """Fetch past prediction logs."""
    records = db.query(PredictionRecord).order_by(PredictionRecord.created_at.desc()).limit(50).all()
    results = []

    for r in records:
        try:
            conf_int = json.loads(r.conformal_interval_json)
        except Exception:
            conf_int = [0.0, 0.0]

        try:
            attn = json.loads(r.scale_attention_json)
        except Exception:
            attn = {"4A": 33.3, "6A": 33.3, "8A": 33.4}

        results.append({
            "id": r.id,
            "filename": r.filename,
            "formula": r.formula,
            "predicted_energy_eV": r.predicted_energy_eV,
            "evidential_std_eV": r.evidential_std_eV,
            "confidence": r.confidence,
            "risk_level": r.risk_level,
            "conformal_90_interval_eV": conf_int,
            "scale_attention": attn,
            "created_at": r.created_at.strftime("%Y-%m-%d %H:%M:%S") if r.created_at else ""
        })

    return {"count": len(results), "history": results}


@router.get("/jobs")
def get_screening_jobs(db: Session = Depends(get_db)):
    """Fetch history of batch screening jobs."""
    jobs = db.query(ScreeningJob).order_by(ScreeningJob.created_at.desc()).limit(20).all()
    results = []
    for j in jobs:
        results.append({
            "job_id": j.job_id,
            "title": j.title,
            "status": j.status,
            "total_materials": j.total_materials,
            "high_confidence_count": j.high_confidence_count,
            "runtime_seconds": j.runtime_seconds,
            "created_at": j.created_at.strftime("%Y-%m-%d %H:%M:%S") if j.created_at else ""
        })
    return {"jobs": results}
