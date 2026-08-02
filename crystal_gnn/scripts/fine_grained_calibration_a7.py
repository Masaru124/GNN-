"""
Task 1: Fine-Grained Calibration Audit for A7 (10 Bins)
Reprocesses test set predictions from paper_A7 checkpoint.
Computes 10-bin empirical coverage table and reliability statistics.
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

def evaluate_a7_test_set():
    ckpt_path = "checkpoints/paper_A7_soap_loco_formation_energy_per_atom/best.pt"
    if not os.path.exists(ckpt_path):
        ckpt_path = "checkpoints/paper_A7_soap_loco_formation_energy_per_atom/last.pt"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading checkpoint: {ckpt_path} on {device}...", flush=True)
    checkpoint = torch.load(ckpt_path, map_location=device)
    config = checkpoint["config"]

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

    if "split" in checkpoint:
        split = checkpoint["split"]
        print("Using split stored in checkpoint.", flush=True)
    else:
        split_file = Path("data/splits/soap_loco.json")
        with open(split_file, "r") as f:
            split_data = json.load(f)
        split = split_data["splits"][0]
        print("Loaded SOAP-LOCO split from data/splits/soap_loco.json.", flush=True)

    test_indices = split["test"]
    valid_test_indices = [idx for idx in test_indices if idx < len(dataset)]
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
    model.eval()

    all_targets = []
    all_mus = []
    all_sigmas = []

    with torch.no_grad():
        for b1, b2, b3, y, _ in test_loader:
            b1 = b1.to(device)
            b2 = b2.to(device)
            b3 = b3.to(device)
            y = y.to(device)

            out = model(b1, b2, b3)
            if isinstance(out, (tuple, list)):
                gamma, v, alpha, beta = out
            else:
                gamma = out[:, 0]
                v = out[:, 1]
                alpha = out[:, 2]
                beta = out[:, 3]

            mu = gamma.squeeze()
            v = torch.clamp(v.squeeze(), min=1e-4)
            alpha = torch.clamp(alpha.squeeze(), min=1.0001)
            beta = torch.clamp(beta.squeeze(), min=1e-4)

            var_tot = (beta * (1.0 + 1.0 / v)) / (alpha - 1.0)
            sigma = torch.sqrt(torch.clamp(var_tot, min=1e-8))

            all_targets.extend(y.cpu().numpy().tolist())
            all_mus.extend(mu.cpu().numpy().tolist())
            all_sigmas.extend(sigma.cpu().numpy().tolist())

    targets = np.array(all_targets)
    pred_means = np.array(all_mus)
    sigmas = np.array(all_sigmas)

    z = 1.96
    lower_bounds = pred_means - z * sigmas
    upper_bounds = pred_means + z * sigmas
    pred_intervals = np.column_stack([lower_bounds, upper_bounds])

    return targets, pred_means, pred_intervals, sigmas

def per_bin_coverage(targets, pred_means, pred_intervals, n_bins=10):
    widths = pred_intervals[:, 1] - pred_intervals[:, 0]
    bin_edges = np.quantile(widths, np.linspace(0, 1, n_bins + 1))
    bin_edges[-1] += 1e-8

    print("\n==========================================================================", flush=True)
    print("      FINE-GRAINED CALIBRATION AUDIT FOR A7 (10-BIN DEEP EVIDENTIAL UQ)    ", flush=True)
    print("==========================================================================", flush=True)
    print(f"{'Bin':>4} | {'Width Range (eV)':>22} | {'N':>6} | {'Coverage (95% Nominal)':>22} | {'Mean |Err|':>10}")
    print("-" * 75, flush=True)

    coverages = []
    mean_errors = []

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (widths >= lo) & (widths < hi)
        n = mask.sum()
        if n == 0:
            continue
        in_interval = (targets[mask] >= pred_intervals[mask, 0]) & (targets[mask] <= pred_intervals[mask, 1])
        coverage = in_interval.mean()
        mean_err = np.abs(targets[mask] - pred_means[mask]).mean()

        coverages.append(coverage)
        mean_errors.append(mean_err)
        print(f"{i+1:>4} | [{lo:9.4f}, {hi:9.4f}) | {n:>6} | {coverage:>21.3f} | {mean_err:>10.4f}")

    coverages = np.array(coverages)
    total_in_interval = (targets >= pred_intervals[:, 0]) & (targets <= pred_intervals[:, 1])

    print("=" * 75, flush=True)
    print(f"Overall Empirical Coverage (All Bins Pooled): {total_in_interval.mean():.4f} ({total_in_interval.mean()*100:.1f}%)")
    print(f"Coverage Std across 10 Bins:                  {coverages.std():.4f}")
    print(f"Min / Max Bin Coverage:                       {coverages.min():.3f} / {coverages.max():.3f}")
    print(f"Overall Test MAE:                             {np.abs(targets - pred_means).mean():.4f} eV/atom")
    print("==========================================================================", flush=True)

def main():
    targets, pred_means, pred_intervals, sigmas = evaluate_a7_test_set()
    per_bin_coverage(targets, pred_means, pred_intervals, n_bins=10)

if __name__ == "__main__":
    main()
