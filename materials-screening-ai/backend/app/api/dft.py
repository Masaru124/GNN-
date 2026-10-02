# -*- coding: utf-8 -*-
"""
DFT Tier 3 API Endpoints.

Provides FastAPI routes for:
  GET  /api/dft/status          — QE installation status
  POST /api/dft/queue/{id}      — Queue PBE DFT job for a candidate
  GET  /api/dft/job/{id}        — Poll DFT job status
  POST /api/dft/r2scan/{id}     — Queue r2SCAN job
  POST /api/dft/hse06/{id}      — Queue HSE06 job (manually triggered)
  POST /api/dft/delta-ml/{id}   — Apply Δ-ML correction on existing PBE result
  POST /api/dft/nac-phonon/{id} — Run NAC-corrected phonon on candidate
  GET  /api/dft/buffer/stats    — Retrain buffer statistics
  POST /api/dft/buffer/retrain  — Trigger GNN fine-tuning on retrain buffer
  GET  /api/dft/literature/{formula} — Literature novelty check for formula
  GET  /api/dft/synthesis/{formula}  — Synthesis route classification
"""

import json
import logging
import threading
import tempfile
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, BackgroundTasks, Query
from pydantic import BaseModel

from app.database.db import get_db
from app.database.models import DiscoveryCandidate
from app.services.dft_validation import get_dft_service
from app.services.delta_ml_corrector import get_delta_ml_corrector
from app.services.retrain_buffer_service import get_retrain_buffer
from app.services.literature_check import get_literature_checker
from app.services.synthesis_service import get_synthesis_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/dft", tags=["dft"])


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------

class RetrainRequest(BaseModel):
    min_samples: int = 10
    learning_rate: float = 1e-4
    epochs: int = 20


class DeltaMLRequest(BaseModel):
    chi_diff: Optional[float] = None
    r_ratio: Optional[float] = None
    Z_avg: Optional[float] = None
    eps_inf: Optional[float] = None


# ---------------------------------------------------------------------------
# DFT Status
# ---------------------------------------------------------------------------

@router.get("/status")
def get_dft_status() -> Dict[str, Any]:
    """Return Quantum ESPRESSO installation status and pseudopotential availability."""
    dft_svc = get_dft_service()
    return dft_svc.get_install_status()


# ---------------------------------------------------------------------------
# DFT Job Queuing (Tier 3a: PBE)
# ---------------------------------------------------------------------------

@router.post("/queue/{candidate_id}")
def queue_pbe_job(
    candidate_id: int,
    background_tasks: BackgroundTasks,
    kpt_dist: float = Query(default=0.35, description="K-point distance in Å⁻¹"),
    fast_mode: bool = Query(default=True, description="Fast mode: use MLIP pre-relaxed structure and direct SCF+NSCF in ~30s"),
) -> Dict[str, Any]:
    """
    Queue a Tier 3a PBE DFT job for a discovery candidate.
    
    fast_mode=True (default): Uses the candidate's Tier 2 MLIP relaxed geometry, 
    running standardized SCF + NSCF in ~30 seconds.
    fast_mode=False: Runs full variable-cell ionic relaxation (vc-relax).
    
    Runs asynchronously: pw.x is spawned in a BackgroundTask.
    Poll /api/dft/job/{candidate_id} to check progress.
    """
    db = next(get_db())
    try:
        cand = db.query(DiscoveryCandidate).filter(DiscoveryCandidate.id == candidate_id).first()
        if not cand:
            raise HTTPException(status_code=404, detail=f"Candidate #{candidate_id} not found")

        # Check QE availability
        dft_svc = get_dft_service()
        if not dft_svc.qe_available:
            return {
                "status": "qe_not_installed",
                "candidate_id": candidate_id,
                "formula": cand.formula,
                "message": dft_svc.get_install_status()["install_instructions"],
            }

        # Mark as queued
        cand.dft_pbe_status = "queued"
        db.commit()

        formula = cand.formula
        cif = cand.relaxed_structure_cif or cand.structure_cif

        # Launch async DFT job
        def _run_pbe():
            from pymatgen.core import Structure
            try:
                _db_init = next(get_db())
                try:
                    _c_init = _db_init.query(DiscoveryCandidate).filter(DiscoveryCandidate.id == candidate_id).first()
                    if _c_init:
                        _c_init.dft_pbe_status = "running"
                        _db_init.commit()
                except Exception:
                    pass
                finally:
                    _db_init.close()

                struct = Structure.from_str(cif, fmt="cif")
                workdir = tempfile.mkdtemp(prefix=f"matscreen_pbe_{candidate_id}_")
                result = dft_svc.run_pbe_pipeline(struct, formula, workdir=workdir, kpt_dist=kpt_dist, fast_mode=fast_mode)

                _db = next(get_db())
                try:
                    _cand = _db.query(DiscoveryCandidate).filter(DiscoveryCandidate.id == candidate_id).first()
                    if _cand:
                        _cand.dft_pbe_status = result.get("status", "done")
                        _cand.dft_pbe_gap_eV = result.get("pbe_gap_eV")
                        _cand.dft_pbe_gap_type = result.get("gap_type")
                        _cand.dft_pbe_convergence_params = json.dumps(result.get("convergence_params", {}))
                        if result.get("relaxed_structure_cif"):
                            _cand.dft_relaxed_structure_cif = result["relaxed_structure_cif"]

                        # Auto-run Δ-ML correction if PBE gap available
                        if result.get("pbe_gap_eV") is not None:
                            corrector = get_delta_ml_corrector()
                            ml_result = corrector.predict_corrected_gap(
                                pbe_gap_ev=result["pbe_gap_eV"],
                                formula=formula,
                            )
                            _cand.dft_delta_ml_gap_eV = ml_result.get("corrected_gap_eV")
                            _cand.dft_delta_ml_interval_lower = ml_result.get("interval_lower")
                            _cand.dft_delta_ml_interval_upper = ml_result.get("interval_upper")
                            _cand.dft_delta_ml_q_hat = ml_result.get("q_hat")
                            _cand.dft_delta_ml_training_provenance = (
                                f"delta_ml_ridge + conformal_calibration (method={ml_result.get('method')})"
                            )

                        # Auto-collect into retrain buffer
                        if result.get("pbe_gap_eV") is not None and _cand.estimated_band_gap_eV is not None:
                            buf = get_retrain_buffer()
                            buf.collect_entry(
                                formula=formula,
                                structure_cif=cif,
                                gnn_pred_eV=_cand.gnn_prediction,
                                verified_label_eV=result["pbe_gap_eV"],
                                label_source="dft_pbe",
                                label_uncertainty_eV=0.1,
                            )

                        _db.commit()
                except Exception as e:
                    logger.error(f"[DFT] DB write-back failed for candidate {candidate_id}: {e}")
                    _db.rollback()
                finally:
                    _db.close()
            except Exception as e:
                logger.error(f"[DFT] PBE job failed for candidate {candidate_id}: {e}")
                _fail_db = next(get_db())
                try:
                    _c = _fail_db.query(DiscoveryCandidate).filter(DiscoveryCandidate.id == candidate_id).first()
                    if _c:
                        _c.dft_pbe_status = "failed"
                        _fail_db.commit()
                finally:
                    _fail_db.close()

        background_tasks.add_task(_run_pbe)

        return {
            "status": "queued",
            "candidate_id": candidate_id,
            "formula": formula,
            "tier": "Tier3a_PBE",
            "message": "DFT PBE job queued. Poll /api/dft/job/{candidate_id} for status.",
        }
    finally:
        db.close()


# ---------------------------------------------------------------------------
# DFT Job Status Poll
# ---------------------------------------------------------------------------

@router.get("/job/{candidate_id}")
def get_dft_job_status(candidate_id: int) -> Dict[str, Any]:
    """Poll DFT job status for a candidate."""
    db = next(get_db())
    try:
        cand = db.query(DiscoveryCandidate).filter(DiscoveryCandidate.id == candidate_id).first()
        if not cand:
            raise HTTPException(status_code=404, detail=f"Candidate #{candidate_id} not found")

        return {
            "candidate_id": candidate_id,
            "formula": cand.formula,
            # Tier 3a: PBE
            "dft_pbe_status": cand.dft_pbe_status,
            "dft_pbe_gap_eV": cand.dft_pbe_gap_eV,
            "dft_pbe_gap_type": cand.dft_pbe_gap_type,
            "dft_pbe_disclosure": "PBE (DFT) — underestimates gaps by ~30-50%",
            # Tier 3b: Δ-ML
            "dft_delta_ml_gap_eV": cand.dft_delta_ml_gap_eV,
            "dft_delta_ml_interval_lower": cand.dft_delta_ml_interval_lower,
            "dft_delta_ml_interval_upper": cand.dft_delta_ml_interval_upper,
            "dft_delta_ml_q_hat": cand.dft_delta_ml_q_hat,
            "dft_delta_ml_training_provenance": cand.dft_delta_ml_training_provenance,
            # Tier 3c: r2SCAN
            "dft_r2scan_status": cand.dft_r2scan_status,
            "dft_r2scan_gap_eV": cand.dft_r2scan_gap_eV,
            # Tier 3d: HSE06
            "dft_hse06_status": cand.dft_hse06_status,
            "dft_hse06_gap_eV": cand.dft_hse06_gap_eV,
            # Phonon
            "phonon_source": cand.phonon_source,
            "phonon_nac_correction_applied": cand.phonon_nac_correction_applied,
            "phonon_instability_flag": cand.phonon_instability_flag,
        }
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Tier 3c: r2SCAN
# ---------------------------------------------------------------------------

@router.post("/r2scan/{candidate_id}")
def queue_r2scan_job(candidate_id: int, background_tasks: BackgroundTasks) -> Dict[str, Any]:
    """Queue Tier 3c r2SCAN meta-GGA SCF + gap job."""
    db = next(get_db())
    try:
        cand = db.query(DiscoveryCandidate).filter(DiscoveryCandidate.id == candidate_id).first()
        if not cand:
            raise HTTPException(status_code=404, detail=f"Candidate #{candidate_id} not found")

        dft_svc = get_dft_service()
        if not dft_svc.qe_available:
            return {"status": "qe_not_installed", "message": dft_svc.get_install_status()["install_instructions"]}

        cand.dft_r2scan_status = "queued"
        db.commit()

        cif = cand.dft_relaxed_structure_cif or cand.relaxed_structure_cif or cand.structure_cif
        formula = cand.formula

        def _run_r2scan():
            from pymatgen.core import Structure
            try:
                struct = Structure.from_str(cif, fmt="cif")
                result = dft_svc.run_r2scan_gap(struct, formula)
                _db = next(get_db())
                try:
                    _c = _db.query(DiscoveryCandidate).filter(DiscoveryCandidate.id == candidate_id).first()
                    if _c:
                        _c.dft_r2scan_status = result.get("status", "done")
                        _c.dft_r2scan_gap_eV = result.get("r2scan_gap_eV")
                        _db.commit()
                finally:
                    _db.close()
            except Exception as e:
                logger.error(f"[DFT] r2SCAN job failed for candidate {candidate_id}: {e}")

        background_tasks.add_task(_run_r2scan)
        return {"status": "queued", "candidate_id": candidate_id, "tier": "Tier3c_r2SCAN"}
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Tier 3d: HSE06 (manually triggered)
# ---------------------------------------------------------------------------

@router.post("/hse06/{candidate_id}")
def queue_hse06_job(candidate_id: int, background_tasks: BackgroundTasks) -> Dict[str, Any]:
    """
    Queue Tier 3d HSE06 gap-only job. MANUALLY TRIGGERED — expensive.
    Requires ONCV norm-conserving pseudopotentials (set SSSP_PP_DIR to ONCV dir).
    """
    db = next(get_db())
    try:
        cand = db.query(DiscoveryCandidate).filter(DiscoveryCandidate.id == candidate_id).first()
        if not cand:
            raise HTTPException(status_code=404, detail=f"Candidate #{candidate_id} not found")

        dft_svc = get_dft_service()
        if not dft_svc.qe_available:
            return {"status": "qe_not_installed", "message": dft_svc.get_install_status()["install_instructions"]}

        cand.dft_hse06_status = "queued"
        db.commit()

        cif = cand.dft_relaxed_structure_cif or cand.relaxed_structure_cif or cand.structure_cif
        formula = cand.formula

        def _run_hse06():
            from pymatgen.core import Structure
            try:
                struct = Structure.from_str(cif, fmt="cif")
                result = dft_svc.run_hse06_gap_only(struct, formula)
                _db = next(get_db())
                try:
                    _c = _db.query(DiscoveryCandidate).filter(DiscoveryCandidate.id == candidate_id).first()
                    if _c:
                        _c.dft_hse06_status = result.get("status", "done")
                        _c.dft_hse06_gap_eV = result.get("hse06_gap_eV")
                        _c.dft_hse06_convergence_params = json.dumps(result.get("convergence_params", {}))
                        # Update retrain buffer with high-fidelity label
                        if result.get("hse06_gap_eV") is not None:
                            buf = get_retrain_buffer()
                            buf.collect_entry(
                                formula=formula,
                                structure_cif=cif,
                                gnn_pred_eV=_c.gnn_prediction,
                                verified_label_eV=result["hse06_gap_eV"],
                                label_source="dft_hse06",
                                label_uncertainty_eV=0.05,
                            )
                        _db.commit()
                finally:
                    _db.close()
            except Exception as e:
                logger.error(f"[DFT] HSE06 job failed for candidate {candidate_id}: {e}")

        background_tasks.add_task(_run_hse06)
        return {
            "status": "queued",
            "candidate_id": candidate_id,
            "tier": "Tier3d_HSE06",
            "warning": "HSE06 is expensive (≥10× PBE). Monitor wall-clock time on first run.",
        }
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Tier 3b: Δ-ML correction (can be applied without running HSE06)
# ---------------------------------------------------------------------------

@router.post("/delta-ml/{candidate_id}")
def apply_delta_ml_correction(candidate_id: int, req: DeltaMLRequest) -> Dict[str, Any]:
    """Apply Δ-ML calibrated gap correction to an existing PBE result."""
    db = next(get_db())
    try:
        cand = db.query(DiscoveryCandidate).filter(DiscoveryCandidate.id == candidate_id).first()
        if not cand:
            raise HTTPException(status_code=404, detail=f"Candidate #{candidate_id} not found")

        pbe_gap = cand.dft_pbe_gap_eV
        if pbe_gap is None:
            raise HTTPException(
                status_code=400,
                detail="No PBE gap available for this candidate. Run /api/dft/queue/{id} first."
            )

        features = {}
        if req.chi_diff is not None:
            features["chi_diff"] = req.chi_diff
        if req.r_ratio is not None:
            features["r_ratio"] = req.r_ratio
        if req.Z_avg is not None:
            features["Z_avg"] = req.Z_avg
        if req.eps_inf is not None:
            features["eps_inf"] = req.eps_inf

        corrector = get_delta_ml_corrector()
        result = corrector.predict_corrected_gap(
            pbe_gap_ev=pbe_gap,
            features=features if features else None,
            formula=cand.formula,
        )

        cand.dft_delta_ml_gap_eV = result["corrected_gap_eV"]
        cand.dft_delta_ml_interval_lower = result["interval_lower"]
        cand.dft_delta_ml_interval_upper = result["interval_upper"]
        q_hat = result.get("q_hat", result.get("q_hat_pooled"))
        cand.dft_delta_ml_q_hat = q_hat
        cand.dft_delta_ml_training_provenance = f"delta_ml_ridge (method={result['method']}, q_hat={q_hat})"
        if req.eps_inf is not None:
            cand.dft_dielectric_const_pbe = req.eps_inf

        db.commit()
        return {
            "candidate_id": candidate_id,
            "formula": cand.formula,
            **result,
        }
    finally:
        db.close()


# ---------------------------------------------------------------------------
# NAC-Corrected Phonon
# ---------------------------------------------------------------------------

@router.post("/nac-phonon/{candidate_id}")
def run_nac_phonon(candidate_id: int, background_tasks: BackgroundTasks) -> Dict[str, Any]:
    """Run NAC-corrected phonon stability check for a candidate."""
    db = next(get_db())
    try:
        cand = db.query(DiscoveryCandidate).filter(DiscoveryCandidate.id == candidate_id).first()
        if not cand:
            raise HTTPException(status_code=404, detail=f"Candidate #{candidate_id} not found")

        cif = cand.relaxed_structure_cif or cand.structure_cif
        formula = cand.formula

        def _run_nac():
            from pymatgen.core import Structure
            from app.services.simulation_service import VirtualLabSimulationService
            try:
                struct = Structure.from_str(cif, fmt="cif")
                svc = VirtualLabSimulationService()
                result = svc.run_nac_corrected_phonon(struct, candidate_id=candidate_id)
                logger.info(
                    f"[NACPhonon] Candidate #{candidate_id} ({formula}): "
                    f"source={result['phonon_source']}, stable={result['is_dynamically_stable']}"
                )
            except Exception as e:
                logger.error(f"[NACPhonon] Failed for candidate {candidate_id}: {e}")

        background_tasks.add_task(_run_nac)
        return {
            "status": "queued",
            "candidate_id": candidate_id,
            "formula": formula,
            "message": "NAC-corrected phonon job queued. Poll /api/dft/job/{id} for phonon_source and phonon_instability_flag.",
        }
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Retrain Buffer
# ---------------------------------------------------------------------------

@router.get("/buffer/stats")
def get_retrain_buffer_stats() -> Dict[str, Any]:
    """Return retrain buffer statistics."""
    buf = get_retrain_buffer()
    return buf.get_buffer_stats()


@router.post("/buffer/retrain")
def trigger_retrain(req: RetrainRequest) -> Dict[str, Any]:
    """
    Trigger GNN fine-tuning on the accumulated retrain buffer.
    
    Fine-tuning runs synchronously (this call will block for the duration).
    For large buffers (>50 samples), consider running via a background task
    or the retrain_from_buffer.py script instead.
    """
    from app.services.active_learning import ActiveLearningService
    al_svc = ActiveLearningService()
    buf = get_retrain_buffer()

    result = buf.run_retrain(
        min_samples=req.min_samples,
        learning_rate=req.learning_rate,
        epochs=req.epochs,
        current_version=al_svc.current_version,
    )

    if result.get("status") == "success":
        al_svc.current_version = result["model_version_after"]

    return result


# ---------------------------------------------------------------------------
# Literature & Synthesis (standalone endpoints)
# ---------------------------------------------------------------------------

@router.get("/literature/{formula}")
def check_literature(formula: str) -> Dict[str, Any]:
    """Check literature for prior publications matching this crystal formula."""
    checker = get_literature_checker()
    return checker.check_formula(formula)


@router.get("/synthesis/{formula}")
def get_synthesis_route(formula: str) -> Dict[str, Any]:
    """Classify synthesis route and enumerate precursors for a crystal formula."""
    svc = get_synthesis_service()
    return svc.classify_route(formula)
