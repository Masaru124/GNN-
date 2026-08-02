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

    structures, labels = load_data("data/raw")
    structures = structures[:50000]

    dataset = MultiScaleDataset(
        structures,
        labels,
        radii=config["model"]["radii"],
        target="formation_energy_per_atom",
        cache_dir="./data/cache"
    )

    split = checkpoint["split"]
    test_indices = split["test"]
    mapped_test = [dataset.orig_to_dataset_idx[i] for i in test_indices if i in dataset.orig_to_dataset_idx]

    test_set = Subset(dataset, mapped_test)
    test_loader = DataLoader(
        test_set,
        batch_size=128,
        shuffle=False,
        collate_fn=MultiScaleCollate(),
        num_workers=0
    )

    model = _build_model(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # 1. Evaluate with model.predict_with_uncertainty (T=1, Fast)
    model.train()
    ys, mc_preds, mc_uncs = [], [], []
    with torch.no_grad():
        for b1, b2, b3, y, _ in test_loader:
            b1, b2, b3 = b1.to(device), b2.to(device), b3.to(device)
            uq = model.predict_with_uncertainty(b1, b2, b3, T=1)
            ys.append(y.cpu().numpy().reshape(-1))
            mc_preds.append(uq["prediction"].cpu().numpy().reshape(-1))
            mc_uncs.append(uq["epistemic"].cpu().numpy().reshape(-1))

    y_true = np.concatenate(ys)
    y_pred = np.concatenate(mc_preds)
    unc = np.concatenate(mc_uncs)

    # Apply Conformal Predictor calibration (n_cal = 25% of y_true)
    cp = LocallyAdaptiveConformalPredictor(target_coverage=0.90)
    n_cal = max(50, int(0.25 * len(y_true)))
    cp.calibrate(y_pred[:n_cal], unc[:n_cal], y_true[:n_cal])
    intervals = cp.predict(y_pred[n_cal:], unc[n_cal:])

    metrics = compute_all_metrics(
        y_true[n_cal:],
        y_pred[n_cal:],
        unc[n_cal:],
        intervals["lower"],
        intervals["upper"],
    )

    print("\n==========================================================================")
    print("      EXACT FAST AUDIT RESULTS FOR A7 CHECKPOINT ON SOAP-LOCO TEST SET    ")
    print("==========================================================================")
    print(f"Overall MAE (Full test set, n={len(y_true)}):  {np.abs(y_true - y_pred).mean():.6f} eV/atom")
    print(f"Eval Split MAE (n_test={len(y_true)-n_cal}):       {metrics['mae']:.6f} eV/atom")
    print(f"Eval Split RMSE:                             {metrics['rmse']:.6f} eV/atom")
    print(f"Eval Split R2:                               {metrics['r2']:.6f}")
    print(f"Conformal Calibrated 90% Coverage:            {metrics['coverage']*100:.2f}%")
    print(f"Conformal Mean Interval Width:               {metrics['mean_width']:.4f} eV")
    print("==========================================================================")

if __name__ == "__main__":
    main()
