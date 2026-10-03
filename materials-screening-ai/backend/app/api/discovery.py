# -*- coding: utf-8 -*-
"""
FastAPI Router for Discovery Engine Endpoints.

Endpoints:
  - POST /api/discovery/query: Submit discovery query job
  - GET  /api/discovery/query/{job_id}: Poll job status and fetch candidates
  - POST /api/discovery/validate: Trigger Tier 2 MLIP structure relaxation
  - GET  /api/discovery/report/{run_id}: Generate/fetch discovery run report
  - GET  /api/discovery/history: List past discovery runs
"""

import json
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.database.db import get_db
from app.database.models import DiscoveryRun, DiscoveryCandidate
from app.services.job_orchestrator import DiscoveryJobOrchestrator
from app.services.nl_query_parser import NaturalLanguageQueryParser

router = APIRouter(prefix="/api/discovery", tags=["Discovery Engine"])
orchestrator = DiscoveryJobOrchestrator()
nl_parser = NaturalLanguageQueryParser()


class PromptParseRequest(BaseModel):
    prompt: str = Field(..., description="Free-text natural language material specification prompt")


class DiscoveryQueryRequest(BaseModel):
    title: Optional[str] = Field(None, description="Optional title for discovery run")
    scaffold: str = Field("perovskite", description="Scaffold family: perovskite, spinel, layered_oxide, olivine")
    num_candidates: int = Field(12, ge=1, le=50, description="Number of novel candidates to generate")
    generation_mode: str = Field("substitution", description="Generation mode: substitution or mattergen")
    exclude_toxic: bool = Field(True, description="Apply RoHS toxic element exclusion filter")
    max_cost_usd_kg: Optional[float] = Field(None, description="Max raw material cost ceiling ($/kg)")
    max_density_g_cm3: Optional[float] = Field(None, description="Max density ceiling (g/cm³)")
    target_ion: str = Field("Li", description="Target mobile ion for transport proxy (Li, Na, Mg, K)")
    target_property_min: Optional[float] = Field(None, description="Min target formation energy (eV/atom)")
    target_property_max: Optional[float] = Field(None, description="Max target formation energy (eV/atom)")
    excluded_elements: Optional[List[str]] = Field(None, description="Specific elements to exclude")
    allowed_elements: Optional[List[str]] = Field(None, description="Specific elements allowed for substitution")


class ValidateRequest(BaseModel):
    discovery_run_id: int = Field(..., description="ID of discovery run to validate")
    candidate_ids: Optional[List[int]] = Field(None, description="Specific candidate IDs to run Tier 2 MLIP on")


@router.post("/parse-prompt")
def parse_natural_language_prompt(req: PromptParseRequest):
    """Parse free-text natural language specification prompt into structured discovery constraints."""
    if not req.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt string cannot be empty.")
    return nl_parser.parse_prompt(req.prompt)


@router.post("/query", status_code=status.HTTP_202_ACCEPTED)
def submit_discovery_query(req: DiscoveryQueryRequest):
    """Submit a discovery query job to the background orchestrator."""
    query_dict = req.dict()
    res = orchestrator.submit_discovery_query(query=query_dict, title=req.title)
    return res


@router.get("/query/{job_id}")
def get_discovery_job_status(job_id: str, db: Session = Depends(get_db)):
    """Fetch status, metadata, and generated candidate list for a discovery run."""
    run_rec = db.query(DiscoveryRun).filter(DiscoveryRun.job_id == job_id).first()
    if not run_rec:
        raise HTTPException(status_code=404, detail=f"Discovery run with job_id '{job_id}' not found.")

    candidates_rec = db.query(DiscoveryCandidate).filter(
        DiscoveryCandidate.discovery_run_id == run_rec.id
    ).order_by(DiscoveryCandidate.pareto_rank.asc(), DiscoveryCandidate.gnn_prediction.asc()).all()

    candidates = []
    for c in candidates_rec:
        candidates.append({
            "id": c.id,
            "candidate_index": c.candidate_index,
            "formula": c.formula,
            "structure_cif": c.structure_cif,
            "generation_method": c.generation_method,
            "gnn_prediction": c.gnn_prediction,
            "gnn_uncertainty_low": c.gnn_uncertainty_low,
            "gnn_uncertainty_high": c.gnn_uncertainty_high,
            "evidential_std_eV": c.evidential_std_eV,
            "novelty_status": c.novelty_status,
            "known_match_id": c.known_match_id,
            "hard_filter_pass": c.hard_filter_pass,
            "charge_neutral_pass": c.charge_neutral_pass,
            "filter_reasons": json.loads(c.filter_reasons_json or "[]"),
            "density_g_cm3": c.density_g_cm3,
            "estimated_cost_usd_kg": c.estimated_cost_usd_kg,
            "free_volume_A3": c.free_volume_A3,
            "bottleneck_radius_A": c.bottleneck_radius_A,
            "mlip_relaxed_energy_eV": c.mlip_relaxed_energy_eV,
            "mlip_stability_flag": c.mlip_stability_flag,
            "mlip_trajectory": json.loads(c.mlip_trajectory_json or "[]"),
            "relaxed_structure_cif": c.relaxed_structure_cif,
            "mean_displacement_A": c.mean_displacement_A,
            "gnn_model_version": c.gnn_model_version or "v1.0.0-initial",
            "orchestrator_decision": c.orchestrator_decision or "promote_to_tier2",
            "confidence_tier": c.confidence_tier,
            "pareto_rank": c.pareto_rank,
            # Ensemble Disagreement Gate
            "mace_relaxed_energy_eV": c.mace_relaxed_energy_eV,
            "energy_disagreement_eV_per_atom": c.energy_disagreement_eV_per_atom,
            "structural_rmsd_between_mlips_A": c.structural_rmsd_between_mlips_A,
            "ensemble_status": c.ensemble_status,
            # Item 1: Energy Above Hull
            "e_above_hull_eV": getattr(c, "e_above_hull_eV", None),
            "hull_classification": getattr(c, "hull_classification", None),
            # Item 3: Band gap is not screened. Heuristic estimates (median 90%
            # interval half-width 3.35 eV) are excluded; only Tier C DFT / Delta-ML
            # fills predicted_band_gap_eV below.
            "estimated_band_gap_eV": None,
            "bandgap_estimate_source": None,
            "band_gap_status": "unavailable",
            "band_gap_unavailable_reason": (
                "Band-gap heuristic excluded from screening (median 90% interval "
                "half-width 3.35 eV); requires Tier C DFT / Delta-ML."
            ),
            "is_solar_optimal": None,
            # Item 8: Synthesis Feasibility & Route
            "synthesis_route": getattr(c, "synthesis_route", None),
            "synthesis_feasibility": getattr(c, "synthesis_feasibility", None),
            "synthesis_precursors": json.loads(getattr(c, "synthesis_precursors_json", None) or "[]"),
            "synthesis_estimated_temp_c": getattr(c, "synthesis_estimated_temp_c", None),
            # Item 9: Literature Novelty Check
            "literature_exact_known": getattr(c, "literature_exact_known", None),
            "literature_matches_count": getattr(c, "literature_matches_count", 0),
            "literature_top_doi": getattr(c, "literature_top_doi", None),
            "literature_top_title": getattr(c, "literature_top_title", None),
            # Item 10 / Tier 3: DFT & Δ-ML
            "dft_job_id": getattr(c, "id", None),
            "dft_status": getattr(c, "dft_pbe_status", None) or "not_queued",
            "dft_pbe_status": getattr(c, "dft_pbe_status", None),
            "dft_pbe_energy_eV": getattr(c, "dft_pbe_gap_eV", None),
            "dft_pbe_gap_eV": getattr(c, "dft_pbe_gap_eV", None),
            "dft_pbe_gap_type": getattr(c, "dft_pbe_gap_type", None),
            "dft_delta_ml_bandgap_eV": getattr(c, "dft_delta_ml_gap_eV", None),
            "dft_delta_ml_interval_low": getattr(c, "dft_delta_ml_interval_lower", None),
            "dft_delta_ml_interval_high": getattr(c, "dft_delta_ml_interval_upper", None),
            "dft_delta_ml_q_hat": getattr(c, "dft_delta_ml_q_hat", None),
        })

    return {
        "job_id": run_rec.job_id,
        "run_id": run_rec.id,
        "title": run_rec.title,
        "status": run_rec.status,
        "query": json.loads(run_rec.query_json or "{}"),
        "total_generated": run_rec.total_generated,
        "novel_count": run_rec.novel_count,
        "tier2_validated_count": run_rec.tier2_validated_count,
        "runtime_seconds": run_rec.runtime_seconds,
        "created_at": run_rec.created_at.isoformat() if run_rec.created_at else None,
        "completed_at": run_rec.completed_at.isoformat() if run_rec.completed_at else None,
        "candidates": candidates,
    }


@router.post("/validate")
def trigger_tier2_validation(req: ValidateRequest):
    """Trigger Tier 2 MLIP structure relaxation on shortlisted candidates."""
    res = orchestrator.validate_candidates_tier2(
        discovery_run_id=req.discovery_run_id,
        candidate_ids=req.candidate_ids
    )
    if res.get("status") == "error":
        raise HTTPException(status_code=400, detail=res.get("message"))
    return res


@router.get("/report/{run_id}")
def get_discovery_report(run_id: int, db: Session = Depends(get_db)):
    """Generate comprehensive discovery report payload for export."""
    run_rec = db.query(DiscoveryRun).filter(DiscoveryRun.id == run_id).first()
    if not run_rec:
        raise HTTPException(status_code=404, detail=f"Discovery run {run_id} not found.")

    candidates = db.query(DiscoveryCandidate).filter(
        DiscoveryCandidate.discovery_run_id == run_id
    ).order_by(DiscoveryCandidate.pareto_rank.asc()).all()

    top_candidates = [c for c in candidates if c.hard_filter_pass and c.pareto_rank == 1]

    return {
        "report_id": f"REP-DISC-{run_rec.id}",
        "title": run_rec.title,
        "created_at": run_rec.created_at.isoformat() if run_rec.created_at else None,
        "summary": {
            "total_candidates": run_rec.total_generated,
            "novel_count": run_rec.novel_count,
            "pareto_rank_1_count": len(top_candidates),
            "tier2_validated_count": run_rec.tier2_validated_count,
        },
        "query_constraints": json.loads(run_rec.query_json or "{}"),
        "top_pareto_candidates": [
            {
                "formula": c.formula,
                "predicted_E_f": c.gnn_prediction,
                "conformal_interval": [c.gnn_uncertainty_low, c.gnn_uncertainty_high],
                "novelty_status": c.novelty_status,
                "cost_usd_kg": c.estimated_cost_usd_kg,
                "free_volume_A3": c.free_volume_A3,
                "confidence_tier": c.confidence_tier,
                "mlip_relaxed_energy_eV": c.mlip_relaxed_energy_eV,
            }
            for c in top_candidates
        ]
    }


@router.get("/history")
def list_discovery_history(db: Session = Depends(get_db)):
    """List past discovery runs."""
    runs = db.query(DiscoveryRun).order_by(DiscoveryRun.created_at.desc()).limit(30).all()
    out = []
    for r in runs:
        out.append({
            "id": r.id,
            "job_id": r.job_id,
            "title": r.title,
            "status": r.status,
            "total_generated": r.total_generated,
            "novel_count": r.novel_count,
            "tier2_validated_count": r.tier2_validated_count,
            "created_at": r.created_at.isoformat() if r.created_at else None,
        })
    return out


from app.services.active_learning import ActiveLearningService
al_service = ActiveLearningService()


@router.post("/retrain")
def trigger_active_learning_retrain():
    """Trigger Active Learning fine-tuning loop on newly Tier 2 validated candidates."""
    res = al_service.trigger_retrain_loop()
    return res


@router.get("/retrain/history")
def get_retrain_history():
    """Fetch Active Learning retraining event history."""
    return al_service.get_retrain_history()


@router.get("/ensemble-report/{run_id}")
def get_ensemble_disagreement_report(run_id: int, db: Session = Depends(get_db)):
    """Aggregate ensemble-disagreement statistics for a discovery run.

    Reports what fraction of Tier 2-validated candidates fell into each
    ensemble_status bucket, useful for assessing how much the disagreement
    gate changed the overall confidence profile."""
    run_rec = db.query(DiscoveryRun).filter(DiscoveryRun.id == run_id).first()
    if not run_rec:
        raise HTTPException(status_code=404, detail=f"Discovery run {run_id} not found.")

    candidates = db.query(DiscoveryCandidate).filter(
        DiscoveryCandidate.discovery_run_id == run_id,
        DiscoveryCandidate.ensemble_status.isnot(None)
    ).all()

    total = len(candidates)
    buckets = {
        "high_confidence_agreement": 0,
        "moderate_agreement": 0,
        "requires_independent_validation": 0,
        "single_model_only": 0,
    }
    energy_disagreements = []
    rmsds = []

    for c in candidates:
        status = c.ensemble_status or "single_model_only"
        buckets[status] = buckets.get(status, 0) + 1
        if c.energy_disagreement_eV_per_atom is not None:
            energy_disagreements.append(c.energy_disagreement_eV_per_atom)
        if c.structural_rmsd_between_mlips_A is not None and c.structural_rmsd_between_mlips_A >= 0:
            rmsds.append(c.structural_rmsd_between_mlips_A)

    return {
        "run_id": run_id,
        "title": run_rec.title,
        "total_ensemble_evaluated": total,
        "status_distribution": buckets,
        "status_distribution_pct": {
            k: round(v / max(total, 1) * 100, 1) for k, v in buckets.items()
        },
        "energy_disagreement_stats": {
            "mean_eV_per_atom": round(float(sum(energy_disagreements) / max(len(energy_disagreements), 1)), 4) if energy_disagreements else None,
            "max_eV_per_atom": round(max(energy_disagreements), 4) if energy_disagreements else None,
            "min_eV_per_atom": round(min(energy_disagreements), 4) if energy_disagreements else None,
        },
        "structural_rmsd_stats": {
            "mean_A": round(float(sum(rmsds) / max(len(rmsds), 1)), 4) if rmsds else None,
            "max_A": round(max(rmsds), 4) if rmsds else None,
        },
        "note": (
            "Thresholds are first-pass calibration based on CHGNet error-floor literature. "
            "Agreement means 'no evidence of blind spot,' not 'confirmed correct.'"
        ),
    }
