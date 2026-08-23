# -*- coding: utf-8 -*-
"""
Full Retroactive CHGNet + MACE Ensemble Evaluation on 177 Collision Pairs.
Computes per-structure and pairwise ensemble disagreement metrics.
"""
import sys
import os
import time
import json
import numpy as np
import pandas as pd
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.stdout.reconfigure(encoding='utf-8')

# Ensure imports work from backend and crystal_gnn
sys.path.insert(0, os.path.abspath("materials-screening-ai/backend"))
sys.path.insert(0, os.path.abspath("crystal_gnn/scripts"))

from pymatgen.core import Structure
from collision_scan_full import load_dataset_subsampled
from app.services.physics_validation import PhysicsValidationLayer

OUT_CSV = "crystal_gnn/results/chgnet_mace_ensemble_177_pairs.csv"
OUT_REPORT = "crystal_gnn/results/ensemble_177_pairs_report.json"

def main():
    print("=" * 80)
    print("   RETROACTIVE CHGNET + MACE ENSEMBLE BENCHMARK ON 177 COLLISION PAIRS")
    print("=" * 80)
    
    t_start = time.time()
    
    # 1. Load the 177 pairs CSV
    input_csv = "crystal_gnn/results/chgnet_collision_audit_177_pairs.csv"
    if not os.path.exists(input_csv):
        print(f"Error: {input_csv} not found!")
        return
    
    df_pairs = pd.read_csv(input_csv)
    print(f"Loaded {len(df_pairs)} colliding pairs from {input_csv}.")
    
    # 2. Collect unique structure indices needed
    idx1_set = set(df_pairs["struct_idx_1"])
    idx2_set = set(df_pairs["struct_idx_2"])
    needed_indices = sorted(list(idx1_set.union(idx2_set)))
    print(f"Unique structure indices across 177 pairs: {len(needed_indices)}")
    
    # 3. Load dataset
    print("\nLoading dataset (50,000 subsampled)...")
    dataset = load_dataset_subsampled(50000)
    print(f"Loaded dataset with {len(dataset)} entries.")
    
    # 4. Initialize PhysicsValidationLayer (CHGNet + MACE float64)
    print("\nInitializing CHGNet + MACE ensemble calculators...")
    pv = PhysicsValidationLayer(use_gpu=False)
    if not (pv.chgnet_calc and pv.mace_calc):
        print("Error: Both CHGNet and MACE calculators must be available!")
        return
    print("Ensemble calculators successfully loaded.")
    
    # 5. Evaluate all unique structures with the ensemble
    # Cache results by structure index: idx -> dict
    eval_cache = {}
    print(f"\nEvaluating {len(needed_indices)} unique structures with CHGNet + MACE ensemble...")
    t_eval_start = time.time()
    
    for count, s_idx in enumerate(needed_indices, 1):
        struct, target_e = dataset[s_idx]
        formula = struct.composition.reduced_formula
        
        t0 = time.time()
        # Fast relaxation: max_steps=40, fmax=0.05
        res = pv.validate_with_ensemble(
            structure=struct,
            predicted_gnn_energy=target_e,
            max_steps=40,
            fmax_threshold=0.05
        )
        elapsed = time.time() - t0
        
        eval_cache[s_idx] = {
            "struct_idx": s_idx,
            "formula": formula,
            "natoms": len(struct),
            "target_energy": target_e,
            "chgnet_relaxed_e": res["mlip_relaxed_energy_eV"],
            "mace_relaxed_e": res["mace_relaxed_energy_eV"],
            "energy_disagreement_eV_per_atom": res["energy_disagreement_eV_per_atom"],
            "structural_rmsd_A": res["structural_rmsd_between_mlips_A"],
            "ensemble_status": res["ensemble_status"],
            "confidence_tier": res["confidence_tier"],
            "runtime_s": elapsed
        }
        
        if count % 20 == 0 or count == len(needed_indices):
            avg_t = (time.time() - t_eval_start) / count
            eta = (len(needed_indices) - count) * avg_t
            print(f"  [{count:3d}/{len(needed_indices)}] Evaluated idx {s_idx:5d} ({formula:<10}) -> Status: {res['ensemble_status']} (avg {avg_t:.2f}s/struct, ETA {eta:.1f}s)")
            
    print(f"\nCompleted evaluation of {len(needed_indices)} structures in {time.time() - t_eval_start:.1f}s.")
    
    # 6. Map ensemble results back to the 177 pairs
    print("\nMapping ensemble evaluations to 177 collision pairs...")
    pair_records = []
    
    status_counts = {"high_confidence_agreement": 0, "moderate_agreement": 0, "requires_independent_validation": 0}
    
    for _, row in df_pairs.iterrows():
        p_idx = int(row["pair_index"])
        s1 = int(row["struct_idx_1"])
        s2 = int(row["struct_idx_2"])
        
        r1 = eval_cache[s1]
        r2 = eval_cache[s2]
        
        # Determine pair-level ensemble status: worst status between the two structures
        if r1["ensemble_status"] == "requires_independent_validation" or r2["ensemble_status"] == "requires_independent_validation":
            pair_status = "requires_independent_validation"
        elif r1["ensemble_status"] == "moderate_agreement" or r2["ensemble_status"] == "moderate_agreement":
            pair_status = "moderate_agreement"
        else:
            pair_status = "high_confidence_agreement"
            
        status_counts[pair_status] += 1
        
        # MACE predicted gap between pair members
        mace_pred_gap = abs(r1["mace_relaxed_e"] - r2["mace_relaxed_e"])
        chgnet_pred_gap = float(row["chgnet_pred_gap"])
        true_gap = float(row["true_energy_gap"])
        
        pair_records.append({
            "pair_index": p_idx,
            "struct_idx_1": s1,
            "struct_idx_2": s2,
            "formula_1": row["formula_1"],
            "formula_2": row["formula_2"],
            "true_energy_gap": true_gap,
            "gnn_classification": row["classification"],
            "chgnet_pred_gap": chgnet_pred_gap,
            "mace_pred_gap": round(mace_pred_gap, 6),
            "s1_chgnet_e": r1["chgnet_relaxed_e"],
            "s1_mace_e": r1["mace_relaxed_e"],
            "s1_delta_e": r1["energy_disagreement_eV_per_atom"],
            "s1_rmsd_A": r1["structural_rmsd_A"],
            "s1_status": r1["ensemble_status"],
            "s2_chgnet_e": r2["chgnet_relaxed_e"],
            "s2_mace_e": r2["mace_relaxed_e"],
            "s2_delta_e": r2["energy_disagreement_eV_per_atom"],
            "s2_rmsd_A": r2["structural_rmsd_A"],
            "s2_status": r2["ensemble_status"],
            "pair_ensemble_status": pair_status
        })
        
    df_out = pd.DataFrame(pair_records)
    df_out.to_csv(OUT_CSV, index=False)
    print(f"Saved full enriched dataset to {OUT_CSV}.")
    
    # 7. Generate aggregate statistics
    n_pairs = len(df_out)
    n_structs = len(needed_indices)
    
    struct_status_counts = pd.Series([r["ensemble_status"] for r in eval_cache.values()]).value_counts().to_dict()
    
    all_deltas = [r["energy_disagreement_eV_per_atom"] for r in eval_cache.values()]
    all_rmsds = [r["structural_rmsd_A"] for r in eval_cache.values() if r["structural_rmsd_A"] is not None and r["structural_rmsd_A"] >= 0]
    
    report = {
        "benchmark_summary": {
            "total_pairs_evaluated": n_pairs,
            "unique_structures_evaluated": n_structs,
            "total_runtime_seconds": round(time.time() - t_start, 2),
        },
        "pair_status_distribution": {
            k: {"count": status_counts.get(k, 0), "pct": round(status_counts.get(k, 0) / n_pairs * 100, 1)}
            for k in ["high_confidence_agreement", "moderate_agreement", "requires_independent_validation"]
        },
        "structure_status_distribution": {
            k: {"count": struct_status_counts.get(k, 0), "pct": round(struct_status_counts.get(k, 0) / n_structs * 100, 1)}
            for k in ["high_confidence_agreement", "moderate_agreement", "requires_independent_validation"]
        },
        "energy_disagreement_stats_eV_per_atom": {
            "mean": round(float(np.mean(all_deltas)), 4),
            "median": round(float(np.median(all_deltas)), 4),
            "std": round(float(np.std(all_deltas)), 4),
            "max": round(float(np.max(all_deltas)), 4),
            "min": round(float(np.min(all_deltas)), 4),
            "p90": round(float(np.percentile(all_deltas, 90)), 4),
        },
        "structural_rmsd_stats_A": {
            "mean": round(float(np.mean(all_rmsds)), 4) if all_rmsds else None,
            "median": round(float(np.median(all_rmsds)), 4) if all_rmsds else None,
            "max": round(float(np.max(all_rmsds)), 4) if all_rmsds else None,
            "p90": round(float(np.percentile(all_rmsds, 90)), 4) if all_rmsds else None,
        },
        "by_collision_category": {}
    }
    
    # Breakdown by original collision resolution category
    for cat in ["Fully Resolved", "Partially Resolved", "Collapsed"]:
        sub_df = df_out[df_out["gnn_classification"] == cat]
        cat_counts = sub_df["pair_ensemble_status"].value_counts().to_dict()
        report["by_collision_category"][cat] = {
            "total": len(sub_df),
            "high_confidence_agreement": cat_counts.get("high_confidence_agreement", 0),
            "moderate_agreement": cat_counts.get("moderate_agreement", 0),
            "requires_independent_validation": cat_counts.get("requires_independent_validation", 0),
            "requires_validation_pct": round(cat_counts.get("requires_independent_validation", 0) / max(len(sub_df), 1) * 100, 1)
        }
        
    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Saved aggregate report to {OUT_REPORT}.")
    
    # Print formatted summary table
    print("\n" + "=" * 80)
    print("                     AGGREGATE ENSEMBLE BENCHMARK REPORT                      ")
    print("=" * 80)
    print(f"Total Colliding Pairs Evaluated:     {n_pairs}")
    print(f"Unique Structures Evaluated:        {n_structs}")
    print(f"Total Runtime:                      {report['benchmark_summary']['total_runtime_seconds']:.1f} s")
    print("\n--- PAIR-LEVEL ENSEMBLE STATUS DISTRIBUTION ---")
    for k, v in report["pair_status_distribution"].items():
        print(f"  {k:<35}: {v['count']:3d} ({v['pct']:5.1f}%)")
        
    print("\n--- ENERGY DISAGREEMENT STATS (Relaxation Drop) ---")
    for k, v in report["energy_disagreement_stats_eV_per_atom"].items():
        print(f"  {k:<10}: {v:.4f} eV/atom")
        
    print("\n--- BY COLLISION RESOLUTION CATEGORY ---")
    print(f"{'Category':<22} {'Total':>6} {'Agreed':>8} {'Moderate':>10} {'Requires Valid':>16} {'Disagr %':>10}")
    print("-" * 76)
    for cat, data in report["by_collision_category"].items():
        print(f"{cat:<22} {data['total']:>6d} {data['high_confidence_agreement']:>8d} {data['moderate_agreement']:>10d} {data['requires_independent_validation']:>16d} {data['requires_validation_pct']:>9.1f}%")

if __name__ == "__main__":
    main()
