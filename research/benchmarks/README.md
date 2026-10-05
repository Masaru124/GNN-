# Benchmarks

Committed benchmark artifacts for the materials-screening-ai / crystal_gnn benchmarking round.

- `BENCHMARK_RESULTS.md` — compiled report; all numbers from committed artifacts.
- Do not edit by hand; rerun `scripts/compile_benchmark_results.py` after regenerating artifacts.

## Verification

- Full pytest suite and frontend typecheck before tagging.
- Tag: v1.2.0 (created without force, on top of the current HEAD).

## Sources

- LOCO cross-conformal artifacts: `materials-screening-ai/research/loco_cross_conformal/`
- Ablation + ensemble results: `crystal_gnn/results/`
- Report generator: `scripts/compile_benchmark_results.py`
