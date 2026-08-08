# -*- coding: utf-8 -*-
"""
Asynchronous Discovery Job Orchestrator.

Orchestrates the multi-stage materials discovery & physics validation pipeline:
  Stage 1: Novel structure generation (pymatgen scaffold substitution / MatterGen)
  Stage 2: Constraint parsing & hard filter evaluation (toxicity, cost, density)
  Stage 3: High-throughput Multi-Scale GNN property prediction + DER & Conformal UQ
  Stage 4: Transport property proxy calculation (Voronoi free volume & channel radius)
  Stage 5: Weisfeiler-Lehman (WL) novelty verification vs Materials Project & training set
  Stage 6: Multi-objective Pareto frontier optimization & ranking
  Stage 7: Tier 2 Physics Validation (ASE + CHGNet MLIP structure relaxation)
"""

import uuid
import time
import json
from datetime import datetime
from typing import Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor

from pymatgen.core import Structure

from app.database.db import get_db
from app.database.models import DiscoveryRun, DiscoveryCandidate
from app.services.constraint_parser import ConstraintParser
from app.services.generation_engine import StructureGenerationEngine
from app.services.predictor import GNNPredictorService
from app.services.transport_proxy import TransportPropertyProxy
from app.services.novelty_checker import NoveltyChecker
from app.services.pareto_ranker import ParetoRanker
from app.services.physics_validation import PhysicsValidationLayer

# Thread pool executor for async background discovery tasks
DISCOVERY_EXECUTOR = ThreadPoolExecutor(max_workers=4)


from app.services.multi_fidelity_orchestrator import MultiFidelityOrchestrator
from app.services.active_learning import ActiveLearningService


class DiscoveryJobOrchestrator:
    """Orchestrates multi-stage materials discovery pipelines and Tier 2 validation."""

    def __init__(self):
        self.predictor = GNNPredictorService()
        self.generator = StructureGenerationEngine()
        self.novelty_checker = NoveltyChecker()
        self.physics_validator = PhysicsValidationLayer()
        self.mf_orchestrator = MultiFidelityOrchestrator()
        self.al_service = ActiveLearningService()

    def run_discovery_pipeline(self, job_id: str, query: Dict[str, Any], user_id: Optional[int] = None):
        """Execute full discovery pipeline in background worker."""
        t0 = time.time()
        db = next(get_db())

        try:
            # 1. Fetch DiscoveryRun DB Record
            run_rec = db.query(DiscoveryRun).filter(DiscoveryRun.job_id == job_id).first()
            if not run_rec:
                return

            run_rec.status = "running"
            db.commit()

            # Parse Query Parameters
            scaffold = query.get("scaffold", "perovskite")
            num_candidates = int(query.get("num_candidates", 12))
            generation_mode = query.get("generation_mode", "substitution")
            exclude_toxic = query.get("exclude_toxic", True)
            max_cost = query.get("max_cost_usd_kg")
            max_density = query.get("max_density_g_cm3")
            target_ion = query.get("target_ion", "Li")

            # 2. Stage 1: Generate Candidate Structures
            raw_candidates = self.generator.generate_candidates(
                scaffold_type=scaffold,
                target_ion=target_ion,
                num_candidates=num_candidates,
                generation_mode=generation_mode,
                excluded_elements=query.get("excluded_elements"),
                allowed_elements=query.get("allowed_elements"),
            )

            # 3. Initialize Services
            parser = ConstraintParser(
                exclude_toxic=exclude_toxic,
                max_cost_usd_kg=float(max_cost) if max_cost else None,
                max_density_g_cm3=float(max_density) if max_density else None,
            )
            transport_analyzer = TransportPropertyProxy(target_ion=target_ion)

            processed_candidates = []

            # 4. Process Each Candidate
            for idx, cand in enumerate(raw_candidates):
                struct: Structure = cand["structure"]

                # GNN Formation Energy Prediction & Tier C Band Gap Flagging
                pred_res = self.predictor.predict(struct)
                pred_val = float(pred_res.get("predicted_formation_energy_per_atom_eV", pred_res.get("predicted_formation_energy_eV", 0.0)))
                # Band Gap requires DFT (Tier C) — explicitly set to None (Uncalculated)
                pred_bg = None
                is_solar_opt = False
                ev_std = float(pred_res.get("evidential_std_eV", 0.1))
                conf_int = pred_res.get("conformal_90_interval_eV", pred_res.get("conformal_90_interval", [pred_val - 0.2, pred_val + 0.2]))
                q_low = float(conf_int[0])
                q_high = float(conf_int[1])

                # Hard Filter Evaluation
                pass_filter, reasons, metrics = parser.evaluate_structure(struct, predicted_property=pred_val)

                # HARD FILTER ENFORCEMENT: Disqualify candidates that fail hard cost / toxicity filters
                if not pass_filter:
                    print(f"[HardFilter] Rejected {cand['formula']}: {', '.join(reasons)}")
                    continue

                # Transport Property Proxy
                transport_res = transport_analyzer.analyze_structure(struct)

                # WL Novelty Verification
                novelty_status, match_id = self.novelty_checker.verify_novelty(struct)

                # Multi-Fidelity Information Gain Decision Routing
                decision = self.mf_orchestrator.decide_next_action(
                    gnn_prediction=pred_val,
                    uncertainty_low=q_low,
                    uncertainty_high=q_high,
                    hard_filter_pass=pass_filter,
                    target_threshold=float(query.get("target_property_max")) if query.get("target_property_max") else -0.20,
                    is_solar_optimal=is_solar_opt
                )

                processed_candidates.append({
                    "candidate_index": idx + 1,
                    "formula": cand["formula"],
                    "structure_cif": cand["cif_content"],
                    "generation_method": cand["generation_method"],
                    "gnn_prediction": pred_val,
                    "predicted_band_gap_eV": None,
                    "is_solar_optimal": False,
                    "gnn_uncertainty_low": q_low,
                    "gnn_uncertainty_high": q_high,
                    "evidential_std_eV": ev_std,
                    "gnn_model_version": self.al_service.current_version,
                    "novelty_status": novelty_status,
                    "known_match_id": match_id,
                    "hard_filter_pass": pass_filter,
                    "filter_reasons_json": json.dumps(reasons),
                    "density_g_cm3": metrics["density_g_cm3"],
                    "estimated_cost_usd_kg": metrics["cost_usd_kg"],
                    "free_volume_A3": transport_res["free_volume_A3"],
                    "bottleneck_radius_A": transport_res["bottleneck_radius_A"],
                    "orchestrator_decision": decision,
                    "confidence_tier": "Tier 1 (Stability & Cost Screened)",
                })

            # 5. Stage 6: Multi-Objective Pareto Optimization Ranking
            ranker = ParetoRanker(
                minimize_property=True,
                minimize_cost=True,
                maximize_free_volume=True
            )
            ranked_candidates = ranker.rank_candidates(processed_candidates)

            # 6. Save Candidates to DB
            novel_count = 0
            for item in ranked_candidates:
                if item["novelty_status"] == "novel":
                    novel_count += 1

                cand_rec = DiscoveryCandidate(
                    discovery_run_id=run_rec.id,
                    candidate_index=item["candidate_index"],
                    formula=item["formula"],
                    structure_cif=item["structure_cif"],
                    generation_method=item["generation_method"],
                    gnn_prediction=item["gnn_prediction"],
                    predicted_band_gap_eV=item["predicted_band_gap_eV"],
                    is_solar_optimal=item["is_solar_optimal"],
                    gnn_uncertainty_low=item["gnn_uncertainty_low"],
                    gnn_uncertainty_high=item["gnn_uncertainty_high"],
                    evidential_std_eV=item["evidential_std_eV"],
                    gnn_model_version=item.get("gnn_model_version", "v1.0.0-initial"),
                    novelty_status=item["novelty_status"],
                    known_match_id=item["known_match_id"],
                    hard_filter_pass=item["hard_filter_pass"],
                    filter_reasons_json=item["filter_reasons_json"],
                    density_g_cm3=item["density_g_cm3"],
                    estimated_cost_usd_kg=item["estimated_cost_usd_kg"],
                    free_volume_A3=item["free_volume_A3"],
                    bottleneck_radius_A=item["bottleneck_radius_A"],
                    orchestrator_decision=item.get("orchestrator_decision", "promote_to_tier2"),
                    confidence_tier=item["confidence_tier"],
                    pareto_rank=item.get("pareto_rank", 1),
                )
                db.add(cand_rec)

            runtime = time.time() - t0
            run_rec.status = "complete"
            run_rec.total_generated = len(ranked_candidates)
            run_rec.novel_count = novel_count
            run_rec.runtime_seconds = round(runtime, 2)
            run_rec.completed_at = datetime.utcnow()

            db.commit()

        except Exception as e:
            import traceback
            traceback.print_exc()
            db.rollback()
            run_rec = db.query(DiscoveryRun).filter(DiscoveryRun.job_id == job_id).first()
            if run_rec:
                run_rec.status = "failed"
                db.commit()
        finally:
            db.close()

    def submit_discovery_query(
        self,
        query: Dict[str, Any],
        title: Optional[str] = None,
        user_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """Submit a discovery query job and launch background worker."""
        job_id = f"disc_{uuid.uuid4().hex[:12]}"
        run_title = title or f"Discovery Run ({query.get('scaffold', 'perovskite').title()})"

        db = next(get_db())
        run_rec = DiscoveryRun(
            job_id=job_id,
            user_id=user_id,
            title=run_title,
            query_json=json.dumps(query),
            status="queued",
            created_at=datetime.utcnow(),
        )
        db.add(run_rec)
        db.commit()
        db.close()

        # Launch background execution thread
        DISCOVERY_EXECUTOR.submit(self.run_discovery_pipeline, job_id, query, user_id)

        return {
            "job_id": job_id,
            "title": run_title,
            "status": "queued",
            "message": "Discovery query submitted successfully. Background generation started."
        }

    def validate_candidates_tier2(
        self,
        discovery_run_id: int,
        candidate_ids: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """Execute Tier 2 Physics Validation (MLIP structure relaxation) on selected candidates."""
        db = next(get_db())
        try:
            query_filter = db.query(DiscoveryCandidate).filter(
                DiscoveryCandidate.discovery_run_id == discovery_run_id
            )
            if candidate_ids:
                query_filter = query_filter.filter(DiscoveryCandidate.id.in_(candidate_ids))

            candidates = query_filter.all()
            if not candidates:
                return {"status": "error", "message": "No matching candidates found for validation."}

            validated_count = 0
            for cand in candidates:
                # Parse CIF to pymatgen Structure
                try:
                    struct = Structure.from_str(cand.structure_cif, fmt="cif")
                    val_res = self.physics_validator.validate_candidate(
                        structure=struct,
                        predicted_gnn_energy=cand.gnn_prediction
                    )

                    cand.mlip_relaxed_energy_eV = val_res["mlip_relaxed_energy_eV"]
                    cand.mlip_stability_flag = val_res["mlip_stability_flag"]
                    cand.mlip_trajectory_json = json.dumps(val_res.get("trajectory", []))
                    cand.relaxed_structure_cif = val_res.get("relaxed_cif", cand.structure_cif)
                    cand.mean_displacement_A = val_res.get("mean_displacement_A", 0.0)
                    cand.confidence_tier = "Tier 2 (Physics Validated)"
                    validated_count += 1
                except Exception:
                    continue

            # Update DiscoveryRun record
            run_rec = db.query(DiscoveryRun).filter(DiscoveryRun.id == discovery_run_id).first()
            if run_rec:
                run_rec.tier2_validated_count = (run_rec.tier2_validated_count or 0) + validated_count

            db.commit()

            return {
                "status": "success",
                "validated_count": validated_count,
                "message": f"Successfully executed Tier 2 Physics Validation on {validated_count} candidates."
            }
        except Exception as e:
            db.rollback()
            return {"status": "error", "message": str(e)}
        finally:
            db.close()
