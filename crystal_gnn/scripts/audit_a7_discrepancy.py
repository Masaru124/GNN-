"""
Audit script to diagnose the MAE discrepancy (0.0641 vs 0.8640) for A7
and evaluate raw DER vs Conformal calibration coverage.
"""

import sys
import os
import json
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader

sys.path.append(os.path.join(os.getcwd(), "scripts"))
from train import load_data, _build_model, MultiScaleDataset, MultiScaleCollate
from crystal_gnn.uncertainty.conformal import LocallyAdaptiveConformalPredictor
from crystal_gnn.evaluation.metrics import compute_all_metrics

def main():
    ckpt_path = "checkpoints/paper_A7_soap_loco_formation_energy_per_atom/best.pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("=" * 80)
    print("               AUDITING A7 CHECKPOINT & MAE DISCREPANCY")
    print("=" * 80)
    print(f"Loading checkpoint: {ckpt_path} on {device}...")
    checkpoint = torch.load(ckpt_path, map_location=device)
    config = checkpoint["config"]
    
    print(f"Checkpoint Epoch:    {checkpoint.get('epoch')}")
    print(f"Checkpoint Best Val MAE: {checkpoint.get('best_val_mae')}")
    print(f"Run ID:              {checkpoint.get('run_id')}")
    
    print("\n[1] Loading Data & Building Dataset...")
    structures, labels = load_data("data/raw")
    max_structures = config["data"].get("max_structures", 50000)
    if max_structures is not None:
        structures = structures[:int(max_structures)]
        
    dataset = MultiScaleDataset(
        structures=structures,
        labels=labels,
        radii=config["model"]["radii"],
        target=config["data"].get("target", "formation_energy_per_atom"),
        cache_dir=config["data"].get("cache_dir", "./data/cache"),
        max_neighbors=config["data"].get("max_neighbors", 24),
    )
    
    print(f"Dataset Size: {len(dataset)}")
    
    if "split" in checkpoint:
        split = checkpoint["split"]
        print("Split origin: Stored inside checkpoint.")
    else:
        split_file = Path("data/splits/soap_loco.json")
        with open(split_file, "r") as f:
            split_data = json.load(f)
        split = split_data["splits"][0]
        print("Split origin: Loaded from data/splits/soap_loco.json.")
        
    test_indices = split["test"]
    valid_test_indices = [idx for idx in test_indices if idx < len(dataset)]
    print(f"Test Split Count: {len(test_indices)} (Valid: {len(valid_test_indices)})")
    
    test_set = Subset(dataset, valid_test_indices)
    test_loader = DataLoader(
        test_set,
        batch_size=32,
        shuffle=False,
        collate_fn=MultiScaleCollate(),
        num_workers=0
    )
    
    model = _build_model(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    # -------------------------------------------------------------
    # Test Evaluation Mode A: model.eval() (Single Forward Pass)
    # -------------------------------------------------------------
    model.eval()
    all_targets = []
    eval_mus = []
    eval_sigmas = []
    eval_uncs = []
    
    with torch.no_grad():
        for b1, b2, b3, y, _ in test_loader:
            b1, b2, b3, y = b1.to(device), b2.to(device), b3.to(device), y.to(device)
            out = model(b1, b2, b3)
            mu, v, alpha, beta = out
            mu = mu.squeeze()
            v = torch.clamp(v.squeeze(), min=1e-4)
            alpha = torch.clamp(alpha.squeeze(), min=1.0001)
            beta = torch.clamp(beta.squeeze(), min=1e-4)
            
            var_tot = (beta * (1.0 + 1.0 / v)) / (alpha - 1.0)
            sigma = torch.sqrt(torch.clamp(var_tot, min=1e-8))
            
            all_targets.extend(y.cpu().numpy().tolist())
            eval_mus.extend(mu.cpu().numpy().tolist())
            eval_sigmas.extend(sigma.cpu().numpy().tolist())
            
    targets = np.array(all_targets)
    eval_means = np.array(eval_mus)
    eval_sigmas = np.array(eval_sigmas)
    eval_mae = np.abs(targets - eval_means).mean()
    
    print(f"\n[Evaluation Mode A: model.eval() direct forward pass]")
    print(f"MAE: {eval_mae:.6f} eV/atom")
    
    # -------------------------------------------------------------
    # Test Evaluation Mode B: model.predict_with_uncertainty (MC-Dropout T=30)
    # -------------------------------------------------------------
    mc_mus = []
    mc_uncs = []
    with torch.no_grad():
        for b1, b2, b3, y, _ in test_loader:
            b1, b2, b3 = b1.to(device), b2.to(device), b3.to(device)
            uq = model.predict_with_uncertainty(b1, b2, b3, T=30)
            mc_mus.extend(uq["prediction"].cpu().numpy().reshape(-1).tolist())
            mc_uncs.extend(uq["epistemic"].cpu().numpy().reshape(-1).tolist())
            
    mc_means = np.array(mc_mus)
    mc_uncs = np.array(mc_uncs)
    mc_mae = np.abs(targets - mc_means).mean()
    
    print(f"\n[Evaluation Mode B: model.predict_with_uncertainty(T=30)]")
    print(f"MAE: {mc_mae:.6f} eV/atom")
    
    # -------------------------------------------------------------
    # Check 1: Target / Output Normalization Check
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("                     DIAGNOSTIC DATA CHECKS")
    print("=" * 80)
    print(f"Target stats  : min={targets.min():.4f}, max={targets.max():.4f}, mean={targets.mean():.4f}, std={targets.std():.4f}")
    print(f"Eval Preds    : min={eval_means.min():.4f}, max={eval_means.max():.4f}, mean={eval_means.mean():.4f}, std={eval_means.std():.4f}")
    print(f"MC Preds      : min={mc_means.min():.4f}, max={mc_means.max():.4f}, mean={mc_means.mean():.4f}, std={mc_means.std():.4f}")
    
    # -------------------------------------------------------------
    # Check 2: Conformal vs Raw DER Coverage & Pooled Arithmetic
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("             COVERAGE COMPARISON: RAW DER VS CONFORMAL")
    print("=" * 80)
    
    # Raw DER 95% intervals
    z_95 = 1.96
    raw_lower = eval_means - z_95 * eval_sigmas
    raw_upper = eval_means + z_95 * eval_sigmas
    raw_hits = (targets >= raw_lower) & (targets <= raw_upper)
    raw_pooled_coverage = raw_hits.mean()
    print(f"Raw DER 95% Nominal Intervals:")
    print(f"  Pooled Hit Count : {raw_hits.sum()} / {len(targets)}")
    print(f"  Pooled Coverage  : {raw_pooled_coverage:.4f} ({raw_pooled_coverage * 100:.2f}%)")
    print(f"  Mean Interval Width : {(raw_upper - raw_lower).mean():.4f} eV")
    
    # Conformal Calibration (using 25% calibration split as in evaluate.py)
    cp = LocallyAdaptiveConformalPredictor(target_coverage=0.90)
    n_cal = max(50, int(0.25 * len(targets)))
    cp.calibrate(mc_means[:n_cal], mc_uncs[:n_cal], targets[:n_cal])
    intervals = cp.predict(mc_means[n_cal:], mc_uncs[n_cal:])
    
    conf_hits = (targets[n_cal:] >= intervals["lower"]) & (targets[n_cal:] <= intervals["upper"])
    conf_coverage = conf_hits.mean()
    conf_mae = np.abs(targets[n_cal:] - mc_means[n_cal:]).mean()
    
    print(f"\nConformal Calibrated 90% Nominal Intervals (Split test set n_test={len(targets)-n_cal}):")
    print(f"  Pooled Hit Count : {conf_hits.sum()} / {len(targets)-n_cal}")
    print(f"  Pooled Coverage  : {conf_coverage:.4f} ({conf_coverage * 100:.2f}%)")
    print(f"  Mean Interval Width : {(intervals['upper'] - intervals['lower']).mean():.4f} eV")
    print(f"  Conformal Test MAE : {conf_mae:.6f} eV/atom")

if __name__ == "__main__":
    main()
