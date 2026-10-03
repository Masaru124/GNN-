"""
Explain the LiCrO2 (discovery run 9, candidate 52) move between the stochastic
(stored, MC-dropout) and deterministic regeneration.

Prints 20 MC-dropout samples for the exact stored CIF plus the deterministic
serving value, so the 0.46 eV/atom difference can be attributed to MC-dropout
sampling noise rather than a model or data change.

Run from materials-screening-ai/backend:
    python ../../crystal_gnn/scripts/mc_diagnostic.py --formula LiCrO2 --samples 20
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import statistics
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[2] / "materials-screening-ai" / "backend"
sys.path.insert(0, str(BACKEND))


def load_candidate(db_path: Path, formula: str, candidate_id: int | None = None):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    if candidate_id is not None:
        row = conn.execute(
            "select id, discovery_run_id, formula, structure_cif, relaxed_structure_cif, "
            "gnn_prediction, gnn_uncertainty_low, gnn_uncertainty_high, pareto_rank "
            "from discovery_candidates where id = ?",
            (candidate_id,),
        ).fetchone()
    else:
        row = conn.execute(
            "select id, discovery_run_id, formula, structure_cif, relaxed_structure_cif, "
            "gnn_prediction, gnn_uncertainty_low, gnn_uncertainty_high, pareto_rank "
            "from discovery_candidates where formula = ? order by id limit 1",
            (formula,),
        ).fetchone()
    conn.close()
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--formula", default="LiCrO2")
    ap.add_argument("--id", type=int, default=None, help="candidate id (overrides formula lookup)")
    ap.add_argument("--samples", type=int, default=20)
    ap.add_argument("--db", default=str(BACKEND / "mat_screen.db"))
    ap.add_argument("--backup-db", default=None, help="pre-determinism database for the stored value")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    row = load_candidate(Path(args.db), args.formula, args.id)
    if row is None:
        raise SystemExit(f"no candidate {args.formula} in {args.db}")
    print("candidate id      :", row["id"], "run", row["discovery_run_id"])
    print("stored value (stochastic run):", row["gnn_prediction"], "eV/atom")
    print("stored interval   :", [row["gnn_uncertainty_low"], row["gnn_uncertainty_high"]])
    print("stored pareto rank:", row["pareto_rank"])

    from pymatgen.core import Structure
    import torch

    from app.services.predictor import GNNPredictorService

    struct = Structure.from_str(row["structure_cif"], fmt="cif")
    predictor = GNNPredictorService.get_instance()

    det = predictor.predict(struct)
    print("deterministic     :", det["predicted_formation_energy_per_atom_eV"], "eV/atom")
    print("deterministic sigma:", det["evidential_std_eV"], "eV")
    print("deterministic interval:", det["conformal_90_interval_eV"])

    samples = []
    with GNNPredictorService._PREDICT_LOCK:
        predictor._set_mc_dropout(enabled=True)
        try:
            torch.manual_seed(args.seed)
            g = [predictor._build_graph_for_radius(struct, radius=r).to(predictor.device)
                 for r in (4.0, 6.0, 8.0)]
            from torch_geometric.data import Batch

            batches = [Batch.from_data_list([x]) for x in g]
            with torch.no_grad():
                for _ in range(args.samples):
                    out = predictor.model(*batches)
                    samples.append(float(out[0].reshape(-1).cpu().item()))
        finally:
            predictor._set_mc_dropout(enabled=not predictor.deterministic)
    print(f"\n{args.samples} raw MC-dropout draws (torch seed {args.seed}):")
    print(json.dumps(samples, indent=1))
    mean = statistics.fmean(samples)
    sd = statistics.stdev(samples) if len(samples) > 1 else 0.0
    print(f"MC mean           : {mean:.4f} eV/atom")
    print(f"MC std            : {sd:.4f} eV/atom")
    print(f"MC min/max        : {min(samples):.4f} / {max(samples):.4f}")

    stored = row["gnn_prediction"]
    source = "current mat_screen.db (post-regeneration)"
    if args.backup_db:
        old = load_candidate(Path(args.backup_db), args.formula, args.id)
        if old is not None:
            stored = old["gnn_prediction"]
            source = args.backup_db
    delta = det["predicted_formation_energy_per_atom_eV"] - stored
    print(f"\ncomparison baseline: {source} = {stored:.4f} eV/atom")
    print(f"deterministic - baseline = {delta:+.4f} eV/atom")
    print(f"baseline inside MC range = {min(samples) - 1e-9 <= stored <= max(samples) + 1e-9}")
    print(f"deterministic inside MC range = {min(samples) - 1e-9 <= det['predicted_formation_energy_per_atom_eV'] <= max(samples) + 1e-9}")
    print(f"MC mean - baseline       = {mean - stored:+.4f} eV/atom")


if __name__ == "__main__":
    main()
