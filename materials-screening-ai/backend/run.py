"""
Backend Server Launcher Script for MatScreen AI.
Run with: python run.py
"""

import sys
import os
from pathlib import Path
import uvicorn

BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    print("=" * 60)
    print(f"   LAUNCHING MATSCREEN AI FASTAPI BACKEND SERVER ({host}:{port})   ")
    print("=" * 60)
    uvicorn.run("app.main:app", host=host, port=port, reload=True)
