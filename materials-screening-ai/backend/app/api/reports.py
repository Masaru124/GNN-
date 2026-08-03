"""
Report Generator & CSV Export REST API Router for MatScreen AI.
"""

from fastapi import APIRouter, Depends, Response
from pydantic import BaseModel
from typing import List, Dict, Any

from app.services.export_service import ExportService

router = APIRouter(prefix="/api", tags=["Reports"])


class ExportCSVRequest(BaseModel):
    candidates: List[Dict[str, Any]]


@router.post("/report/csv")
def download_csv_report(req: ExportCSVRequest):
    """Generate and return CSV spreadsheet for candidate materials."""
    csv_content = ExportService.generate_csv_report(req.candidates)

    return Response(
        content=csv_content,
        media_type="text/csv",
        headers={
            "Content-Disposition": "attachment; filename=MatScreen_AI_Candidates.csv"
        }
    )
