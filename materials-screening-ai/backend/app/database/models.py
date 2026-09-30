"""
SQLAlchemy Database Models for MatScreen AI.
Defines User, PredictionRecord, ScreeningJob, DiscoveryRun, and DiscoveryCandidate entities.
"""

from datetime import datetime
import json
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, ForeignKey, Boolean
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(100), unique=True, index=True, nullable=False)
    hashed_password = Column(String(200), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    predictions = relationship("PredictionRecord", back_populates="user")
    discovery_runs = relationship("DiscoveryRun", back_populates="user")


class PredictionRecord(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    filename = Column(String(200), nullable=False)
    formula = Column(String(100), nullable=False)
    predicted_energy_eV = Column(Float, nullable=False)
    evidential_std_eV = Column(Float, nullable=False)
    confidence = Column(String(50), nullable=False)
    risk_level = Column(String(50), nullable=False)
    conformal_interval_json = Column(Text, nullable=False)
    scale_attention_json = Column(Text, nullable=False)
    cif_content = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="predictions")


class ScreeningJob(Base):
    __tablename__ = "screening_jobs"

    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(String(100), unique=True, index=True, nullable=False)
    title = Column(String(200), nullable=False)
    status = Column(String(50), default="completed")
    total_materials = Column(Integer, default=0)
    high_confidence_count = Column(Integer, default=0)
    runtime_seconds = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)


class DiscoveryRun(Base):
    __tablename__ = "discovery_runs"

    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(String(100), unique=True, index=True, nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    title = Column(String(200), nullable=False)
    query_json = Column(Text, nullable=False)
    status = Column(String(50), default="queued")  # queued / running / complete / failed
    total_generated = Column(Integer, default=0)
    novel_count = Column(Integer, default=0)
    tier2_validated_count = Column(Integer, default=0)
    runtime_seconds = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)

    # S.U.N. Rate Metrics (Item 1: Energy Above Hull)
    sun_rate_pct = Column(Float, nullable=True)       # Strict: e_above_hull <= 0
    msun_rate_pct = Column(Float, nullable=True)      # Relaxed: e_above_hull <= 0.1 eV/atom
    sun_count = Column(Integer, nullable=True)
    msun_count = Column(Integer, nullable=True)

    user = relationship("User", back_populates="discovery_runs")
    candidates = relationship("DiscoveryCandidate", back_populates="discovery_run", cascade="all, delete-orphan")


class DiscoveryCandidate(Base):
    __tablename__ = "discovery_candidates"

    id = Column(Integer, primary_key=True, index=True)
    discovery_run_id = Column(Integer, ForeignKey("discovery_runs.id"), nullable=False)
    candidate_index = Column(Integer, nullable=False)
    formula = Column(String(100), nullable=False)
    structure_cif = Column(Text, nullable=False)
    generation_method = Column(String(50), nullable=False)  # 'pymatgen_substitution' | 'mattergen_conditioned'
    
    # Tier 1 GNN Predictions & UQ
    gnn_prediction = Column(Float, nullable=False)
    gnn_uncertainty_low = Column(Float, nullable=False)
    gnn_uncertainty_high = Column(Float, nullable=False)
    evidential_std_eV = Column(Float, nullable=False)
    gnn_model_version = Column(String(50), default="v1.0.0-initial")
    
    # Novelty Verification
    novelty_status = Column(String(50), nullable=False)  # 'novel' | 'known_match'
    known_match_id = Column(String(100), nullable=True)
    
    # Hard Filters & Structural Metrics
    hard_filter_pass = Column(Boolean, default=True)
    charge_neutral_pass = Column(Boolean, default=True)
    filter_reasons_json = Column(Text, default="[]")
    density_g_cm3 = Column(Float, nullable=False)
    estimated_cost_usd_kg = Column(Float, nullable=False)
    free_volume_A3 = Column(Float, nullable=False)
    bottleneck_radius_A = Column(Float, nullable=False)
    predicted_band_gap_eV = Column(Float, nullable=True, default=None)
    is_solar_optimal = Column(Boolean, default=False)

    # Energy Above Hull / Decomposition Stability (Item 1)
    e_above_hull_eV = Column(Float, nullable=True)
    hull_classification = Column(String(100), nullable=True)  # on_hull_stable / likely_synthesizable_metastable / borderline_metastable / likely_unstable_decomposes / insufficient_reference_data
    decomposition_products_json = Column(Text, nullable=True)  # JSON list of predicted decomposition formulas
    e_above_hull_tier2_eV = Column(Float, nullable=True)       # Filled once Tier 2 MLIP relaxation completes

    # Band Gap Tier B Estimate (Item 3)
    estimated_band_gap_eV = Column(Float, nullable=True)       # Tier B ML/heuristic estimate
    bandgap_estimate_source = Column(String(100), nullable=True)  # e.g. "calibrated-electronegativity-heuristic" or "M3GNet-MP-pretrained"
    bandgap_estimate_tier = Column(String(50), nullable=True)     # "Tier B"
    
    # Multi-Fidelity Orchestration Decision
    orchestrator_decision = Column(String(50), default="promote_to_tier2")  # 'promote_to_tier2' | 'hold_for_more_data' | 'reject'
    
    # Tier 2 Physics Validation (MLIP)
    mlip_relaxed_energy_eV = Column(Float, nullable=True)
    mlip_stability_flag = Column(Boolean, nullable=True)
    mlip_trajectory_json = Column(Text, nullable=True)
    relaxed_structure_cif = Column(Text, nullable=True)
    mean_displacement_A = Column(Float, nullable=True)
    confidence_tier = Column(String(50), default="Tier 1 (GNN Screen)")

    # Tier 2 Ensemble Disagreement Gate (CHGNet + MACE)
    mace_relaxed_energy_eV = Column(Float, nullable=True)
    energy_disagreement_eV_per_atom = Column(Float, nullable=True)
    structural_rmsd_between_mlips_A = Column(Float, nullable=True)
    ensemble_status = Column(String(100), nullable=True)  # high_confidence_agreement / moderate_agreement / requires_independent_validation / single_model_only
    
    # Tier 3 Virtual Lab Validation Write-Back
    virtuallab_validated = Column(Boolean, default=False)
    virtuallab_results_json = Column(Text, nullable=True)

    # Literature Novelty Check (Item 8)
    literature_known = Column(Boolean, nullable=True)        # True if prior publications found
    literature_confidence = Column(String(20), nullable=True) # high/medium/low/none
    literature_refs_json = Column(Text, nullable=True)       # JSON list of {doi, title, year, source}
    literature_n_refs = Column(Integer, nullable=True)

    # Synthesis Route Classification (Item 9)
    synthesis_route = Column(String(50), nullable=True)      # wet_chemistry/solid_state/vapor_deposition/mechanochemical
    synthesis_feasibility = Column(String(20), nullable=True) # high/medium/low/very_low
    synthesis_temperature_C = Column(Float, nullable=True)
    synthesis_precursors_json = Column(Text, nullable=True)  # JSON list of precursor dicts
    synthesis_warnings_json = Column(Text, nullable=True)    # JSON list of warning strings

    # NAC-Corrected Phonon (Item 12)
    phonon_source = Column(String(40), nullable=True)        # mlip_raw/mlip_nac_corrected/dfpt_full
    phonon_nac_correction_applied = Column(Boolean, default=False)
    phonon_instability_flag = Column(Boolean, nullable=True) # True = imaginary modes found
    phonon_born_charges_json = Column(Text, nullable=True)   # JSON 3×3 tensors per atom
    phonon_dielectric_tensor_json = Column(Text, nullable=True) # JSON 3×3 dielectric tensor

    # Tier 3a: PBE DFT (Item 10)
    dft_pbe_gap_eV = Column(Float, nullable=True)
    dft_pbe_gap_type = Column(String(20), nullable=True)     # direct/indirect
    dft_pbe_status = Column(String(30), nullable=True)       # queued/running/done/failed
    dft_pbe_convergence_params = Column(Text, nullable=True) # JSON: cutoffs, k-dist, PP-set+version
    dft_relaxed_structure_cif = Column(Text, nullable=True)  # post-vc-relax CIF

    # Tier 3b: Δ-ML Calibrated Gap Correction (Item 11)
    dft_delta_ml_gap_eV = Column(Float, nullable=True)
    dft_delta_ml_interval_lower = Column(Float, nullable=True)
    dft_delta_ml_interval_upper = Column(Float, nullable=True)
    dft_delta_ml_q_hat = Column(Float, nullable=True)
    dft_delta_ml_training_provenance = Column(String(200), nullable=True)
    dft_dielectric_const_pbe = Column(Float, nullable=True)  # ε∞ from SCF, feeds Δ-ML feature

    # Tier 3c: r2SCAN Meta-GGA (Item 13)
    dft_r2scan_gap_eV = Column(Float, nullable=True)
    dft_r2scan_status = Column(String(30), nullable=True)

    # Tier 3d: HSE06 (Item 10 — manually triggered)
    dft_hse06_gap_eV = Column(Float, nullable=True)
    dft_hse06_status = Column(String(30), nullable=True)
    dft_hse06_convergence_params = Column(Text, nullable=True)

    # Multi-Objective Optimization Ranking
    pareto_rank = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    discovery_run = relationship("DiscoveryRun", back_populates="candidates")


class RetrainEvent(Base):
    __tablename__ = "retrain_events"

    id = Column(Integer, primary_key=True, index=True)
    triggered_at = Column(DateTime, default=datetime.utcnow)
    n_samples = Column(Integer, nullable=False)
    model_version_before = Column(String(50), nullable=False)
    model_version_after = Column(String(50), nullable=False)
    val_loss_after = Column(Float, nullable=False)

