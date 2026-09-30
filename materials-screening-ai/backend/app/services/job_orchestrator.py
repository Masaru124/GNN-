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
from app.services.stability_analysis import StabilityAnalysisService, DiscoveryRunReportService
from app.services.bandgap_estimator import BandGapEstimatorService
from app.services.generation_engine import SCAFFOLD_TEMPLATES
from app.services.literature_check import get_literature_checker
from app.services.synthesis_service import get_synthesis_service
from app.services.retrain_buffer_service import get_retrain_buffer

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
        self.stability_service = StabilityAnalysisService()
        self.bandgap_estimator = BandGapEstimatorService.get_instance()
        self.literature_checker = get_literature_checker()
        self.synthesis_service = get_synthesis_service()
        self.retrain_buffer = get_retrain_buffer()

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

            # 5. Stage 6.5: Energy Above Hull / Decomposition Stability Check
            for item in processed_candidates:
                try:
                    struct = Structure.from_str(item["structure_cif"], fmt="cif")
                    hull_result = self.stability_service.compute_e_above_hull(
                        composition=struct.composition,
                        predicted_formation_energy_per_atom=item["gnn_prediction"],
                    )
                    item["e_above_hull_eV"] = hull_result.get("e_above_hull_eV")
                    item["hull_classification"] = hull_result.get("hull_classification", "insufficient_reference_data")
                    item["decomposition_products_json"] = json.dumps(hull_result.get("decomposition_products", []))
                except Exception as e:
                    print(f"[HullCheck] Error computing hull for {item['formula']}: {e}")
                    item["e_above_hull_eV"] = None
                    item["hull_classification"] = "hull_construction_failed"
                    item["decomposition_products_json"] = "[]"

            # 5.6. Stage 6.6: Tier B Band Gap Estimation (optical-relevant scaffolds only)
            scaffold_meta = SCAFFOLD_TEMPLATES.get(scaffold, {})
            is_optical_scaffold = scaffold_meta.get("optical_relevant", False)

            if is_optical_scaffold:
                print(f"[BandGapTierB] Scaffold '{scaffold}' is optical-relevant — running Tier B band gap estimation.")
                for item in processed_candidates:
                    try:
                        struct = Structure.from_str(item["structure_cif"], fmt="cif")
                        bg_result = self.bandgap_estimator.estimate_band_gap(struct)
                        item["estimated_band_gap_eV"] = bg_result.get("estimated_band_gap_eV")
                        item["bandgap_estimate_source"] = bg_result.get("source_model")
                        item["bandgap_estimate_tier"] = bg_result.get("tier")
                        item["is_solar_optimal"] = bg_result.get("is_solar_optimal", False)
                        item["predicted_band_gap_eV"] = bg_result.get("estimated_band_gap_eV")
                    except Exception as e:
                        print(f"[BandGapTierB] Error estimating band gap for {item['formula']}: {e}")
                        item["estimated_band_gap_eV"] = None
                        item["bandgap_estimate_source"] = None
                        item["bandgap_estimate_tier"] = None

            # 5.7. Compute S.U.N. Rate for this discovery run
            sun_report = DiscoveryRunReportService.compute_sun_rate(processed_candidates)

            # 5.8. Stage 6.7: Literature Novelty Check (Item 8) — async-safe, cached
            print(f"[LiteratureCheck] Checking {len(processed_candidates)} candidates...")
            for item in processed_candidates:
                try:
                    lit_result = self.literature_checker.check_formula(item["formula"])
                    item["literature_known"] = lit_result.get("known_in_literature")
                    item["literature_confidence"] = lit_result.get("confidence")
                    item["literature_refs_json"] = json.dumps(lit_result.get("references", []))
                    item["literature_n_refs"] = lit_result.get("n_references_found", 0)
                except Exception as e:
                    print(f"[LiteratureCheck] Error for {item['formula']}: {e}")
                    item["literature_known"] = None
                    item["literature_confidence"] = None
                    item["literature_refs_json"] = "[]"
                    item["literature_n_refs"] = 0

            # 5.9. Stage 6.8: Synthesis Route Classification (Item 9)
            print(f"[SynthesisRoute] Classifying synthesis routes...")
            for item in processed_candidates:
                try:
                    syn_result = self.synthesis_service.classify_route(item["formula"])
                    item["synthesis_route"] = syn_result.get("route")
                    item["synthesis_feasibility"] = syn_result.get("feasibility")
                    item["synthesis_temperature_C"] = syn_result.get("temperature_C")
                    item["synthesis_precursors_json"] = json.dumps(syn_result.get("precursors", []))
                    item["synthesis_warnings_json"] = json.dumps(syn_result.get("warnings", []))
                except Exception as e:
                    print(f"[SynthesisRoute] Error for {item['formula']}: {e}")
                    item["synthesis_route"] = None
                    item["synthesis_feasibility"] = None
                    item["synthesis_temperature_C"] = None
                    item["synthesis_precursors_json"] = "[]"
                    item["synthesis_warnings_json"] = "[]"

            # 6. Stage 7: Multi-Objective Pareto Optimization Ranking
            ranker = ParetoRanker(
                minimize_property=True,
                minimize_cost=True,
                maximize_free_volume=True
            )
            ranked_candidates = ranker.rank_candidates(processed_candidates)

            # 7. Save Candidates to DB
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
                    # Item 1: Energy Above Hull
                    e_above_hull_eV=item.get("e_above_hull_eV"),
                    hull_classification=item.get("hull_classification"),
                    decomposition_products_json=item.get("decomposition_products_json", "[]"),
                    # Item 3: Tier B Band Gap (populated later if optical_relevant scaffold)
                    estimated_band_gap_eV=item.get("estimated_band_gap_eV"),
                    bandgap_estimate_source=item.get("bandgap_estimate_source"),
                    bandgap_estimate_tier=item.get("bandgap_estimate_tier"),
                    # Item 8: Literature Check
                    literature_known=item.get("literature_known"),
                    literature_confidence=item.get("literature_confidence"),
                    literature_refs_json=item.get("literature_refs_json", "[]"),
                    literature_n_refs=item.get("literature_n_refs", 0),
                    # Item 9: Synthesis Route
                    synthesis_route=item.get("synthesis_route"),
                    synthesis_feasibility=item.get("synthesis_feasibility"),
                    synthesis_temperature_C=item.get("synthesis_temperature_C"),
                    synthesis_precursors_json=item.get("synthesis_precursors_json", "[]"),
                    synthesis_warnings_json=item.get("synthesis_warnings_json", "[]"),
                )
                db.add(cand_rec)


            runtime = time.time() - t0
            run_rec.status = "complete"
            run_rec.total_generated = len(ranked_candidates)
            run_rec.novel_count = novel_count
            run_rec.runtime_seconds = round(runtime, 2)
            run_rec.completed_at = datetime.utcnow()

            # Store S.U.N. rate metrics on the run record
            run_rec.sun_rate_pct = sun_report.get("SUN_rate_pct")
            run_rec.msun_rate_pct = sun_report.get("MSUN_rate_pct")
            run_rec.sun_count = sun_report.get("SUN_count")
            run_rec.msun_count = sun_report.get("MSUN_count")

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
        """Execute Tier 2 Physics Validation (CHGNet+MACE ensemble relaxation) on selected candidates.

        Candidates where the two MLIPs disagree are held with status
        'requires_independent_validation' rather than silently promoted."""
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
            ensemble_agreed = 0
            ensemble_disagreed = 0

            for cand in candidates:
                # Parse CIF to pymatgen Structure
                try:
                    struct = Structure.from_str(cand.structure_cif, fmt="cif")
                    val_res = self.physics_validator.validate_with_ensemble(
                        structure=struct,
                        predicted_gnn_energy=cand.gnn_prediction
                    )

                    # Backward-compatible CHGNet fields
                    cand.mlip_relaxed_energy_eV = val_res["mlip_relaxed_energy_eV"]
                    cand.mlip_stability_flag = val_res["mlip_stability_flag"]
                    cand.mlip_trajectory_json = json.dumps(val_res.get("trajectory", []))
                    cand.relaxed_structure_cif = val_res.get("relaxed_cif", cand.structure_cif)
                    cand.mean_displacement_A = val_res.get("mean_displacement_A", 0.0)

                    # Ensemble disagreement gate fields
                    cand.mace_relaxed_energy_eV = val_res.get("mace_relaxed_energy_eV")
                    cand.energy_disagreement_eV_per_atom = val_res.get("energy_disagreement_eV_per_atom")
                    cand.structural_rmsd_between_mlips_A = val_res.get("structural_rmsd_between_mlips_A")
                    cand.ensemble_status = val_res.get("ensemble_status", "single_model_only")

                    # Confidence tier: do NOT silently promote disagreement candidates
                    cand.confidence_tier = val_res.get("confidence_tier", "Tier 2 (Physics Validated)")

                    # Track counts
                    validated_count += 1
                    if cand.ensemble_status == "requires_independent_validation":
                        ensemble_disagreed += 1
                    elif cand.ensemble_status in ("high_confidence_agreement", "moderate_agreement"):
                        ensemble_agreed += 1

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
                "ensemble_agreed": ensemble_agreed,
                "ensemble_disagreed": ensemble_disagreed,
                "message": (
                    f"Tier 2 Ensemble Validation complete on {validated_count} candidates: "
                    f"{ensemble_agreed} agreed, {ensemble_disagreed} held for independent validation."
                )
            }
        except Exception as e:
            db.rollback()
            return {"status": "error", "message": str(e)}
        finally:
            db.close()
