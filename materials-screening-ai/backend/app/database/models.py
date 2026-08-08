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
    
    # Multi-Fidelity Orchestration Decision
    orchestrator_decision = Column(String(50), default="promote_to_tier2")  # 'promote_to_tier2' | 'hold_for_more_data' | 'reject'
    
    # Tier 2 Physics Validation (MLIP)
    mlip_relaxed_energy_eV = Column(Float, nullable=True)
    mlip_stability_flag = Column(Boolean, nullable=True)
    mlip_trajectory_json = Column(Text, nullable=True)
    relaxed_structure_cif = Column(Text, nullable=True)
    mean_displacement_A = Column(Float, nullable=True)
    confidence_tier = Column(String(50), default="Tier 1 (GNN Screen)")
    
    # Tier 3 Virtual Lab Validation Write-Back
    virtuallab_validated = Column(Boolean, default=False)
    virtuallab_results_json = Column(Text, nullable=True)

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

