# GNN / MatScreen AI

Materials screening stack: Crystal GNN A7 formation-energy model, MatScreen AI FastAPI backend (Delta-ML calibrated band-gap corrections), and Quantum ESPRESSO DFT tiers.

## Large files kept out of git

History was rewritten with `git filter-repo` on 2026-10-02. Two large binaries are intentionally **not tracked**:

| File | Size (bytes) | SHA256 |
|------|--------------|--------|
| `crystal_gnn/model_inference.pt` | 77,454,219 | `ebd1afcb77f70c58326a71c4d2dba2cf93044bb1e540d4618bf1461ecd1e6489` |
| `qe_download.zip` | 400,430,691 | (QE installer archive; not needed at runtime, `qe/bin` is tracked) |

Restore the checkpoint with:

```bash
python scripts/fetch_model.py [path-or-mirror]
```

The script verifies size + SHA256 and installs `crystal_gnn/model_inference.pt`. With no argument it checks `$MODEL_SOURCE`, then a backup mirror at `../GNN-backup.git`. Without the checkpoint, API startup still works, but GNN-dependent services raise `FileNotFoundError: A7 Model Checkpoint not found ...`.

## Docker

```bash
cd materials-screening-ai
docker compose up --build    # backend: http://localhost:8000/api/health
```

Note: `qe/bin` contains Windows binaries (`pw.exe`), so in-container DFT tiers are skipped at runtime. Set `QE_BIN_DIR` to a native Quantum ESPRESSO install to enable them.

## Tests

See `materials-screening-ai/research/README.md` for the reproduction commands (full suite, QE subset, artifact sanitization).
