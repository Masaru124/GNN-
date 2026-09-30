"""
MatScreen AI FastAPI Main Application.
Registers CORS middleware, database startup, and all REST API routers.
"""

from pathlib import Path
import sys
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Ensure app package is in python path
APP_DIR = Path(__file__).resolve().parent
BACKEND_DIR = APP_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.database.db import init_db
from app.api.auth import router as auth_router
from app.api.predict import router as predict_router
from app.api.search import router as search_router
from app.api.screen import router as screen_router
from app.api.compare import router as compare_router
from app.api.history import router as history_router
from app.api.reports import router as reports_router
from app.api.discovery import router as discovery_router
from app.api.simulation import router as simulation_router
from app.api.dft import router as dft_router

app = FastAPI(
    title="MatScreen AI API",
    description="AI-powered Materials Screening & Autonomous Discovery Platform with Reliable Property Prediction & Uncertainty Quantification",
    version="1.0.0"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database Startup Event
@app.on_event("startup")
def on_startup():
    init_db()
    print("[MatScreen AI Backend] Database initialized successfully.")


# Include API Routers
app.include_router(auth_router)
app.include_router(predict_router)
app.include_router(search_router)
app.include_router(screen_router)
app.include_router(compare_router)
app.include_router(history_router)
app.include_router(reports_router)
app.include_router(discovery_router)
app.include_router(simulation_router)
app.include_router(dft_router)


@app.get("/")
@app.get("/api/health")
def healthcheck():
    return {
        "status": "online",
        "system": "MatScreen AI",
        "model": "MultiScaleGNN A7 + DER + Conformal 90%",
        "version": "1.0.0"
    }
