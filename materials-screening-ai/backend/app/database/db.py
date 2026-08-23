"""
Database Connection & Initialization for MatScreen AI.
Creates SQLite engine and session dependency.
"""

from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import sys
BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.database.models import Base

DB_PATH = BASE_DIR / "mat_screen.db"
SQLALCHEMY_DATABASE_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db():
    """Initialize database tables and run automatic schema migration for SQLite."""
    Base.metadata.create_all(bind=engine)

    # Automatic column migration for existing SQLite databases
    with engine.connect() as conn:
        # Check discovery_runs columns
        try:
            res = conn.exec_driver_sql("PRAGMA table_info(discovery_runs)").fetchall()
            existing_cols = {row[1] for row in res}

            new_run_cols = [
                ("sun_rate_pct", "FLOAT"),
                ("msun_rate_pct", "FLOAT"),
                ("sun_count", "INTEGER"),
                ("msun_count", "INTEGER"),
            ]
            for col_name, col_type in new_run_cols:
                if col_name not in existing_cols:
                    conn.exec_driver_sql(f"ALTER TABLE discovery_runs ADD COLUMN {col_name} {col_type}")
                    print(f"[DB Migration] Added column '{col_name}' to discovery_runs")

            # Check discovery_candidates columns
            res_c = conn.exec_driver_sql("PRAGMA table_info(discovery_candidates)").fetchall()
            existing_cand_cols = {row[1] for row in res_c}

            new_cand_cols = [
                ("e_above_hull_eV", "FLOAT"),
                ("hull_classification", "VARCHAR(100)"),
                ("decomposition_products_json", "TEXT"),
                ("e_above_hull_tier2_eV", "FLOAT"),
                ("estimated_band_gap_eV", "FLOAT"),
                ("bandgap_estimate_source", "VARCHAR(100)"),
                ("bandgap_estimate_tier", "VARCHAR(50)"),
                # Ensemble Disagreement Gate (CHGNet + MACE)
                ("mace_relaxed_energy_eV", "FLOAT"),
                ("energy_disagreement_eV_per_atom", "FLOAT"),
                ("structural_rmsd_between_mlips_A", "FLOAT"),
                ("ensemble_status", "VARCHAR(100)"),
            ]
            for col_name, col_type in new_cand_cols:
                if col_name not in existing_cand_cols:
                    conn.exec_driver_sql(f"ALTER TABLE discovery_candidates ADD COLUMN {col_name} {col_type}")
                    print(f"[DB Migration] Added column '{col_name}' to discovery_candidates")

            conn.commit()
        except Exception as e:
            print(f"[DB Migration] Notice: {e}")


def get_db():
    """FastAPI Dependency for database sessions."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
