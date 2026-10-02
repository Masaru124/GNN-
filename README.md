# GNN / MatScreen AI

Materials screening stack: Crystal GNN A7 formation-energy model, MatScreen AI FastAPI backend (Delta-ML calibrated band-gap corrections, Quantum ESPRESSO DFT tiers), and a Next.js frontend.

## 1. Clone

```bash
git clone https://github.com/Masaru124/GNN-.git
cd GNN-
```

## 2. Prerequisites

| Tool | Version | Needed for |
|------|---------|------------|
| Python | 3.11 | backend, models, tests |
| Node.js + npm | 20+ | frontend |
| Git | any | clone |
| Docker Desktop | any (optional) | containerized backend |
| Quantum ESPRESSO | any (optional) | DFT / DFPT tiers only |

The repo ships Windows Quantum ESPRESSO binaries in `qe/bin` (`pw.exe`, `ph.exe`, MKL DLLs), so the DFT tiers work out of the box **on Windows**. On Linux/macOS everything except the DFT tiers runs; install QE natively from https://www.quantum-espresso.org and set `QE_BIN_DIR=/path/to/qe/bin` to enable them.

## 3. Python environment (backend + models)

```powershell
# Windows (repo root)
python -m venv .venv311
.venv311\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU-only torch, optional
pip install -r materials-screening-ai/backend/requirements.txt
```

```bash
# Linux/macOS
python3.11 -m venv .venv311
source .venv311/bin/activate
pip install -r materials-screening-ai/backend/requirements.txt
```

## 4. GNN checkpoint

The inference weights `crystal_gnn/model_inference.pt` (77 MB) are needed by all GNN services. If the file is missing after clone, restore it with:

```bash
python scripts/fetch_model.py [optional-source]
```

With no argument it checks `$MODEL_SOURCE`, then a backup mirror at `../GNN-backup.git`. The script verifies size + SHA256 before installing. Without the checkpoint the API still starts, but GNN-dependent endpoints fail with `FileNotFoundError: A7 Model Checkpoint not found ...`.

## 5. Run the backend (native)

```powershell
# Windows, repo root
$env:PYTHONPATH = "$PWD\materials-screening-ai\backend"
cd materials-screening-ai\backend
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

```bash
# Linux/macOS, repo root
export PYTHONPATH="$PWD/materials-screening-ai/backend"
cd materials-screening-ai/backend
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

SQLite (`mat_screen.db`) is created and migrated automatically on startup. Health check: http://localhost:8000/api/health

If you use the tracked QE binaries on Windows: `$env:QE_BIN_DIR = "$PWD\qe\bin"` before starting (the backend also auto-discovers `qe/bin` from the repo root).

## 6. Run the frontend

```bash
cd materials-screening-ai/frontend
npm install
npm run dev          # http://localhost:3000
npm run typecheck    # tsc --noEmit
npm run build        # production build
```

The Next.js dev server proxies `/api/*` to `http://127.0.0.1:8000/api/*` (see `next.config.ts`). Change that rewrite target if the backend runs elsewhere. Start the backend first.

## 7. Run the backend (Docker)

```bash
cd materials-screening-ai
docker compose up --build
curl http://localhost:8000/api/health     # {"status":"online", ...}
docker compose down
```

Notes:

- Build context is the repo root; the Dockerfile copies `crystal_gnn/` and `materials-screening-ai/backend/`. Keep the checkpoint (step 4) in place before building so the image can serve GNN predictions.
- Container honors `PORT` (default 8000).
- `qe/bin` is Windows-only, so in-container DFT jobs are skipped at runtime. Set `QE_BIN_DIR` to a native QE install mounted into the container to enable them.

## 8. Quantum ESPRESSO (DFT tiers)

1. **Windows**: nothing to download — `qe/bin` is tracked in the repo. Set `QE_BIN_DIR=<repo>\qe\bin` (or rely on auto-discovery from the repo root).
2. **Linux/macOS**: download Quantum ESPRESSO from https://www.quantum-espresso.org, install, and export `QE_BIN_DIR=/path/to/qe/bin`.
3. **No QE**: API and all non-DFT features work. The 7 pw.exe-dependent tests skip; everything else runs.

## 9. Tests

```powershell
# Full suite from repo root (64 tests, includes QE DFT when pw.exe found)
$env:PYTHONPATH = "$PWD\materials-screening-ai\backend"
.venv311\Scripts\python.exe -m pytest materials-screening-ai\backend\tests -q
```

```powershell
# QE subset only (10 tests)
cd materials-screening-ai
..\..\.venv311\Scripts\python.exe -m pytest -m qe backend\tests\test_dft_benchmark_regression.py -q
```

JUnit output, `--junitxml` flags, path sanitization, and provenance headers: see `materials-screening-ai/research/README.md`.

## 10. Verify your setup

```bash
python scripts/fetch_model.py                 # checkpoint sha256 verified
npm --prefix materials-screening-ai/frontend run typecheck
.venv311/Scripts/python -m pytest materials-screening-ai/backend/tests -q   # expect 64 passed
curl http://localhost:8000/api/health         # {"status":"online", ...}
```
