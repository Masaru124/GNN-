"""
Backend Server Launcher Script for MatScreen AI.
Run with: python run.py
"""

import sys
from pathlib import Path
import uvicorn

BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

if __name__ == "__main__":
    print("=" * 60)
    print("      LAUNCHING MATSCREEN AI FASTAPI BACKEND SERVER         ")
    print("=" * 60)
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
