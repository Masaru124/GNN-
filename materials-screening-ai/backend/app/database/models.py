"""
SQLAlchemy Database Models for MatScreen AI.
Defines User, PredictionRecord, ScreeningJob, and Report entities.
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
