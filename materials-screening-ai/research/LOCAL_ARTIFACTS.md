# Local (untracked) artifacts

These files exist on this machine but are deliberately **not** in git: they are
binaries or regenerable runtime state. Documented here so they are not mistaken
for missing evidence.

| Path | What it is | How to regenerate |
| --- | --- | --- |
| `local_backups/mat_screen.db.bak-20261003` | Pre-determinism snapshot of `materials-screening-ai/backend/mat_screen.db` (516 096 B, sha256 recorded below). Captured before the v1.0.1 deterministic re-inference pass, so the 162 candidate predictions and 26 Pareto ranks can be diffed against the stochastic values. | Not reproducible: it holds the old MC-dropout inference output. Keep it until the next full discovery re-run. |
| `materials-screening-ai/backend/mat_screen.db` | Live runtime database (untracked). After v1.0.1 all 8 predictions and 162 discovery candidates hold deterministic values; 26 Pareto ranks changed. | Re-run a discovery job through the API. |

### Model-version provenance of the regenerated rows

The pre-determinism rows carried two model versions that are **not both reproducible from this repository**:

| Stored `gnn_model_version` (backup) | rows | what it was | reproducible here? |
| --- | --- | --- | --- |
| `v1.0.0-initial` | 138 | the shipped A7 checkpoint, run with always-on MC dropout (stochastic) | yes, as deterministic values |
| `v1.24.0-active_learned` | 24 | an active-learned checkpoint that is not committed and not in git history | **no** |

Diagnostic for the largest move (candidate 52, LiCrO2, run 9, −1.1527 → −0.6927 eV/atom, Δ = +0.4600):

```text
20 raw MC-dropout draws (torch seed 42): min -0.8882, max -0.5177
MC mean -0.6813, MC std 0.0927
deterministic value -0.6927
baseline (-1.1527) inside MC range = False      # ~5 sigma below the MC mean
MC mean - baseline = +0.4714 eV/atom
```

So the move is **not** MC-dropout noise. The stored value came from the
active-learned `v1.24.0` model, which is not available here; recomputing with the
shipped A7 model under the deterministic default gives −0.6927. Reproduce with:

```bash
cd materials-screening-ai/backend
python ../../crystal_gnn/scripts/mc_diagnostic.py --formula LiCrO2 --id 52 \
  --samples 20 --backup-db ../../local_backups/mat_screen.db.bak-20261003
```

All 162 regenerated rows now carry `gnn_model_version = 'A7-soap-loco-deterministic'`
so no row claims a model that cannot be reproduced. Rows previously stamped
`v1.24.0-active_learned` hold A7 numbers and are therefore *not* comparable to
the backup for those 24 candidates.
| `crystal_gnn/checkpoints/loco_A7_soap_cluster*/` | Per-cluster LOCO checkpoints trained for the cross-conformal study (one per held-out soap_loco cluster). Gitignored via `crystal_gnn/checkpoints/`. | `python crystal_gnn/scripts/loco_cross_conformal.py run --folds <c>` |
| `crystal_gnn/data/cache/` | Multi-scale graph cache (`mp-*_r{4,6,8}p00.pt`). Gitignored; built once and reused by every fold. | Rebuilt automatically on first use. |

Backup checksum:

```text
353717faaee4aa097661544515a0ebbcf3f2aaf3deb6c13b0b9db621b734950c  local_backups/mat_screen.db.bak-20261003
```
