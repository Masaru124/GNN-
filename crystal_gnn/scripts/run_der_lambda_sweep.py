# -*- coding: utf-8 -*-
"""
DER Evidential Lambda Sensitivity Sweep Runner & Evaluator (Option A - Fast Mode).
Executes fast training for lambda in [0.001, 0.005, 0.01, 0.05, 0.1]
and reports test MAE, RMSE, R^2, Raw DER 95% coverage, and Conformal 90% coverage.
"""

import os
import sys
import subprocess
import torch
import numpy as np
from pathlib import Path

sys.path.append("scripts")
from train import load_data, _build_model, MultiScaleDataset, MultiScaleCollate
from crystal_gnn.uncertainty.conformal import LocallyAdaptiveConformalPredictor
from torch.utils.data import DataLoader

LAMBDAS = [0.001, 0.005, 0.01, 0.05, 0.1]
CHECKPOINT_DIR = Path("checkpoints")
RESULTS_DIR = Path("results")

def train_lambda(lam: float, device: str = "cuda") -> Path:
    if lam == 0.1:
        baseline_ckpt = CHECKPOINT_DIR / "paper_A7_soap_loco_formation_energy_per_atom" / "best.pt"
        if baseline_ckpt.exists():
            print(f"[*] Reusing verified baseline checkpoint for lambda=0.1 at {baseline_ckpt}.")
            return baseline_ckpt

    run_id = f"paper_A7_lambda_{lam}"
    ckpt_path = CHECKPOINT_DIR / run_id / "best.pt"
    
    if ckpt_path.exists():
        print(f"[*] Checkpoint for lambda={lam} already exists at {ckpt_path}. Skipping training.")
        return ckpt_path

    cmd = [
        sys.executable,
        "scripts/train.py",
        "--ablation", "A7",
        "--split", "soap_loco",
        "--target", "formation_energy_per_atom",
        "--evidential_lambda", str(lam),
        "--run_id", run_id,
        "--device", device,
        "--max_structures", "10000",
        "--max_epochs", "15",
        "--fast_mode"
    ]
    print(f"\n[+] Launching fast training for lambda={lam}: {' '.join(cmd)}")
    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"Training for lambda={lam} failed with exit code {proc.returncode}")
    
    return ckpt_path

def evaluate_checkpoint(ckpt_path: Path, device: str = "cuda"):
    checkpoint = torch.load(ckpt_path, map_location=device)
    config = checkpoint["config"]
    
    structures, labels = load_data("data/raw")
    split = checkpoint["split"]
    raw_test_indices = sorted(split["test"])
    
    test_structures = [structures[i] for i in raw_test_indices if i < len(structures)]
    test_dataset = MultiScaleDataset(
        test_structures,
        labels,
        radii=config["model"]["radii"],
        target="formation_energy_per_atom",
        cache_dir="./data/cache"
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=128,
        shuffle=False,
        collate_fn=MultiScaleCollate(),
        num_workers=0
    )

    model = _build_model(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_targets, all_mus, all_sigmas = [], [], []

    with torch.no_grad():
        for b1, b2, b3, y, _ in test_loader:
            b1, b2, b3, y = b1.to(device), b2.to(device), b3.to(device), y.to(device)
            mu, v, alpha, beta = model(b1, b2, b3)
            mu = mu.reshape(-1)
            v = torch.clamp(v.reshape(-1), min=1e-4)
            alpha = torch.clamp(alpha.reshape(-1), min=1.0001)
            beta = torch.clamp(beta.reshape(-1), min=1e-4)
            
            var = (beta * (1.0 + 1.0 / v)) / (alpha - 1.0)
            sigma = torch.sqrt(torch.clamp(var, min=1e-8))
            
            all_targets.extend(y.cpu().numpy().reshape(-1).tolist())
            all_mus.extend(mu.cpu().numpy().reshape(-1).tolist())
            all_sigmas.extend(sigma.cpu().numpy().reshape(-1).tolist())

    targets = np.array(all_targets)
    pred_means = np.array(all_mus)
    sigmas = np.array(all_sigmas)
    n = len(targets)

    mae = float(np.abs(targets - pred_means).mean())
    rmse = float(np.sqrt(((targets - pred_means)**2).mean()))
    r2 = 1.0 - float(((targets - pred_means)**2).sum() / ((targets - targets.mean())**2).sum())

    # Raw DER 95% Coverage
    raw_lo = pred_means - 1.96 * sigmas
    raw_hi = pred_means + 1.96 * sigmas
    raw_hits = (targets >= raw_lo) & (targets <= raw_hi)
    raw_cov = float(raw_hits.mean() * 100)
    raw_width = float((raw_hi - raw_lo).mean())

    # Conformal 90% Coverage
    n_cal = max(50, int(0.25 * n))
    cp = LocallyAdaptiveConformalPredictor(target_coverage=0.90)
    cp.calibrate(pred_means[:n_cal], sigmas[:n_cal], targets[:n_cal])
    conf_int = cp.predict(pred_means[n_cal:], sigmas[n_cal:])
    conf_hits = (targets[n_cal:] >= conf_int["lower"]) & (targets[n_cal:] <= conf_int["upper"])
    conf_cov = float(conf_hits.mean() * 100)
    conf_width = float((conf_int["upper"] - conf_int["lower"]).mean())

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "raw_cov": raw_cov,
        "raw_width": raw_width,
        "conf_cov": conf_cov,
        "conf_width": conf_width
    }

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("==========================================================================")
    print("       DER EVIDENTIAL LAMBDA SENSITIVITY SWEEP (Option A - Fast Mode)     ")
    print("==========================================================================")
    print(f"Device: {device}")
    print(f"Lambdas to evaluate: {LAMBDAS}")

    results = {}
    for lam in LAMBDAS:
        ckpt_path = train_lambda(lam, device=device)
        print(f"[+] Evaluating checkpoint {ckpt_path} ...")
        metrics = evaluate_checkpoint(ckpt_path, device=device)
        results[lam] = metrics

    print("\n" + "=" * 95)
    print("               DER EVIDENTIAL LAMBDA SENSITIVITY SWEEP RESULTS            ")
    print("=" * 95)
    print(f"{'Lambda':>8} | {'Test MAE':>10} | {'Test RMSE':>10} | {'Test R^2':>8} | {'Raw 95% Cov':>12} | {'Raw Width':>10} | {'Conf 90% Cov':>12} | {'Conf Width':>10}")
    print("-" * 95)
    for lam in LAMBDAS:
        res = results[lam]
        print(f"{lam:>8.3f} | {res['mae']:>10.4f} | {res['rmse']:>10.4f} | {res['r2']:>8.4f} | {res['raw_cov']:>11.2f}% | {res['raw_width']:>9.4f} eV | {res['conf_cov']:>11.2f}% | {res['conf_width']:>9.4f} eV")
    print("=" * 95)

if __name__ == "__main__":
    main()
