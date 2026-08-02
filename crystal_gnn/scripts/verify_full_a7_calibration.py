"""
Definitive Verification & Calibration Audit for A7 on SOAP-LOCO Test Set.
Resolves index mapping, evaluates both Raw DER & Conformal-Calibrated Uncertainty,
and recomputes pooled vs binned coverage accurately.
"""

import sys
import os
import json
import torch
import numpy as np
from pathlib import Path

sys.path.append("scripts")
from train import load_data, _build_model, MultiScaleDataset, MultiScaleCollate
from crystal_gnn.uncertainty.conformal import LocallyAdaptiveConformalPredictor
from crystal_gnn.evaluation.metrics import compute_all_metrics
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader

def main():
    ckpt_path = "checkpoints/paper_A7_soap_loco_formation_energy_per_atom/best.pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(ckpt_path, map_location=device)
    config = checkpoint["config"]
    
    print("==========================================================================")
    print("        AUDIT RESOLUTION & FULL A7 EVALUATION ON SOAP-LOCO TEST SET       ")
    print("==========================================================================")
    print(f"Checkpoint Path:     {ckpt_path}")
    print(f"Checkpoint Epoch:    {checkpoint.get('epoch')}")
    print(f"Best Val MAE in Ckpt: {checkpoint.get('best_val_mae'):.6f} eV/atom")
    print(f"Stored Evaluation MAE in JSON: 0.064095 eV/atom")

    structures, labels = load_data("data/raw")
    max_structures = config["data"].get("max_structures", 50000)
    if max_structures is not None:
        structures = structures[:int(max_structures)]

    dataset = MultiScaleDataset(
        structures,
        labels,
        radii=config["model"]["radii"],
        target="formation_energy_per_atom",
        cache_dir="./data/cache"
    )

    split = checkpoint["split"]
    raw_test_indices = split["test"]
    # Properly map original split indices to dataset indices using orig_to_dataset_idx
    mapped_test_indices = [dataset.orig_to_dataset_idx[i] for i in raw_test_indices if i in dataset.orig_to_dataset_idx]
    
    print(f"\nDataset Size: len(dataset) = {len(dataset)}")
    print(f"Test Split:   Original = {len(raw_test_indices)}, Mapped = {len(mapped_test_indices)}")

    test_set = Subset(dataset, mapped_test_indices)
    test_loader = DataLoader(
        test_set,
        batch_size=128,
        shuffle=False,
        collate_fn=MultiScaleCollate(),
        num_workers=0
    )

    model = _build_model(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    # -------------------------------------------------------------
    # 1. Full Test Set Inference under model.eval() (Single Deterministic Pass)
    # -------------------------------------------------------------
    model.eval()
    all_targets = []
    eval_mus = []
    eval_sigmas = []
    
    with torch.no_grad():
        for b1, b2, b3, y, _ in test_loader:
            b1, b2, b3, y = b1.to(device), b2.to(device), b3.to(device), y.to(device)
            out = model(b1, b2, b3)
            mu, v, alpha, beta = out
            mu = mu.reshape(-1)
            v = torch.clamp(v.reshape(-1), min=1e-4)
            alpha = torch.clamp(alpha.reshape(-1), min=1.0001)
            beta = torch.clamp(beta.reshape(-1), min=1e-4)
            
            var_tot = (beta * (1.0 + 1.0 / v)) / (alpha - 1.0)
            sigma = torch.sqrt(torch.clamp(var_tot, min=1e-8))
            
            all_targets.extend(y.cpu().numpy().reshape(-1).tolist())
            eval_mus.extend(mu.cpu().numpy().reshape(-1).tolist())
            eval_sigmas.extend(sigma.cpu().numpy().reshape(-1).tolist())

    targets = np.array(all_targets)
    pred_means = np.array(eval_mus)
    sigmas = np.array(eval_sigmas)
    
    overall_mae = np.abs(targets - pred_means).mean()
    overall_rmse = np.sqrt(((targets - pred_means)**2).mean())
    overall_r2 = 1.0 - ((targets - pred_means)**2).sum() / ((targets - targets.mean())**2).sum()
    
    print("\n--------------------------------------------------------------------------")
    print("     A7 FULL TEST SET PERFORMANCE (n = 7,175 samples)")
    print("--------------------------------------------------------------------------")
    print(f"Overall Test MAE  : {overall_mae:.6f} eV/atom")
    print(f"Overall Test RMSE : {overall_rmse:.6f} eV/atom")
    print(f"Overall Test R^2  : {overall_r2:.6f}")
    
    # -------------------------------------------------------------
    # 2. Raw DER Uncertainty Coverage vs Conformal Calibrated Coverage
    # -------------------------------------------------------------
    # (A) Raw DER 95% Interval Coverage
    z_95 = 1.96
    raw_lower = pred_means - z_95 * sigmas
    raw_upper = pred_means + z_95 * sigmas
    raw_hits = (targets >= raw_lower) & (targets <= raw_upper)
    raw_coverage = raw_hits.mean()
    
    # (B) Conformal Calibrated Coverage (evaluate.py methodology: 25% calibration split)
    n_cal = max(50, int(0.25 * len(targets)))
    cp = LocallyAdaptiveConformalPredictor(target_coverage=0.90)
    cp.calibrate(pred_means[:n_cal], sigmas[:n_cal], targets[:n_cal])
    conf_intervals = cp.predict(pred_means[n_cal:], sigmas[n_cal:])
    
    conf_hits = (targets[n_cal:] >= conf_intervals["lower"]) & (targets[n_cal:] <= conf_intervals["upper"])
    conf_coverage = conf_hits.mean()
    conf_mae = np.abs(targets[n_cal:] - pred_means[n_cal:]).mean()

    print("\n--------------------------------------------------------------------------")
    print("     UNCERTAINTY COVERAGE ANALYSIS: RAW DER VS CONFORMAL CALIBRATED")
    print("--------------------------------------------------------------------------")
    print(f"[Raw Evidential (NIG) 95% Nominal Intervals]:")
    print(f"  Hit Count         : {raw_hits.sum()} / {len(targets)}")
    print(f"  Empirical Coverage: {raw_coverage*100:.2f}%")
    print(f"  Mean Width       : {(raw_upper - raw_lower).mean():.4f} eV")
    
    print(f"\n[Conformal-Calibrated 90% Nominal Intervals (n_test={len(targets)-n_cal})]:")
    print(f"  Hit Count         : {conf_hits.sum()} / {len(targets)-n_cal}")
    print(f"  Empirical Coverage: {conf_coverage*100:.2f}%")
    print(f"  Mean Width       : {(conf_intervals['upper'] - conf_intervals['lower']).mean():.4f} eV")
    print(f"  Holdout Test MAE  : {conf_mae:.6f} eV/atom")

    # -------------------------------------------------------------
    # 3. 10-Bin Fine-Grained Reliability Table (Raw DER)
    # -------------------------------------------------------------
    widths = raw_upper - raw_lower
    bin_edges = np.quantile(widths, np.linspace(0, 1, 10 + 1))
    bin_edges[-1] += 1e-8
    
    print("\n" + "=" * 80)
    print("      10-BIN FINE-GRAINED RELIABILITY TABLE (RAW DER 95% INTERVALS)")
    print("=" * 80)
    print(f"{'Bin':>4} | {'Width Range (eV)':>22} | {'N':>6} | {'Coverage (95% Nominal)':>22} | {'Mean |Err|':>10}")
    print("-" * 80)
    
    bin_coverages = []
    for i in range(10):
        lo, hi = bin_edges[i], bin_edges[i+1]
        mask = (widths >= lo) & (widths < hi)
        n_bin = mask.sum()
        in_bin = raw_hits[mask]
        cov_bin = in_bin.mean()
        err_bin = np.abs(targets[mask] - pred_means[mask]).mean()
        bin_coverages.append(cov_bin)
        print(f"{i+1:>4} | [{lo:9.4f}, {hi:9.4f}) | {n_bin:>6} | {cov_bin:>21.3f} | {err_bin:>10.4f}")
        
    print("=" * 80)
    print(f"Direct Pooled Coverage (Sum Hits / N) : {raw_hits.sum() / len(raw_hits):.4f} ({raw_hits.sum() * 100 / len(raw_hits):.2f}%)")
    print(f"Simple Mean of 10 Bins                : {np.mean(bin_coverages):.4f} ({np.mean(bin_coverages)*100:.2f}%)")
    print("==========================================================================")

if __name__ == "__main__":
    main()
