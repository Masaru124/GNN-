import sys
import os
import json
import torch
import numpy as np

sys.path.append("scripts")
from train import load_data, _build_model, MultiScaleDataset, MultiScaleCollate
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader

def main():
    ckpt_path = "checkpoints/paper_A7_soap_loco_formation_energy_per_atom/best.pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(ckpt_path, map_location=device)
    config = checkpoint["config"]
    
    print("=================================================================")
    print("             EXACT A7 MAE DISCREPANCY SOLVER")
    print("=================================================================")
    print(f"Checkpoint Epoch:    {checkpoint.get('epoch')}")
    print(f"Best Val MAE in ckpt: {checkpoint.get('best_val_mae')}")
    
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
    model.eval()
    
    all_y = []
    all_mu = []
    all_v = []
    all_alpha = []
    all_beta = []
    
    with torch.no_grad():
        for b1, b2, b3, y, _ in test_loader:
            b1, b2, b3, y = b1.to(device), b2.to(device), b3.to(device), y.to(device)
            out = model(b1, b2, b3)
            mu, v, alpha, beta = out
            all_y.extend(y.cpu().numpy().reshape(-1).tolist())
            all_mu.extend(mu.cpu().numpy().reshape(-1).tolist())
            all_v.extend(v.cpu().numpy().reshape(-1).tolist())
            all_alpha.extend(alpha.cpu().numpy().reshape(-1).tolist())
            all_beta.extend(beta.cpu().numpy().reshape(-1).tolist())
            
    y_true = np.array(all_y)
    mu_pred = np.array(all_mu)
    v_pred = np.array(all_v)
    alpha_pred = np.array(all_alpha)
    beta_pred = np.array(all_beta)
    
    mae = np.abs(y_true - mu_pred).mean()
    rmse = np.sqrt(((y_true - mu_pred)**2).mean())
    r2 = 1.0 - ((y_true - mu_pred)**2).sum() / ((y_true - y_true.mean())**2).sum()
    
    print(f"\nTest Set Size: {len(y_true)}")
    print(f"Computed MAE:  {mae:.6f} eV/atom")
    print(f"Computed RMSE: {rmse:.6f} eV/atom")
    print(f"Computed R2:   {r2:.6f}")
    
    print("\n--- SAMPLE TARGETS VS PREDICTIONS (First 10) ---")
    for i in range(10):
        print(f"  Target: {y_true[i]:10.4f} | Pred: {mu_pred[i]:10.4f} | Error: {abs(y_true[i] - mu_pred[i]):10.4f}")
        
    print("\n--- DISTRIBUTION STATS ---")
    print(f"y_true  : min={y_true.min():8.4f}, max={y_true.max():8.4f}, mean={y_true.mean():8.4f}, std={y_true.std():8.4f}")
    print(f"mu_pred : min={mu_pred.min():8.4f}, max={mu_pred.max():8.4f}, mean={mu_pred.mean():8.4f}, std={mu_pred.std():8.4f}")
    print("=================================================================")

if __name__ == "__main__":
    main()
