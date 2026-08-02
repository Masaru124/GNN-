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
    
    structures, labels = load_data("data/raw")
    structures = structures[:50000]
    dataset = MultiScaleDataset(structures, labels, radii=config["model"]["radii"], target="formation_energy_per_atom", cache_dir="./data/cache")
    
    split = checkpoint["split"]
    raw_test = split["test"]
    mapped_test = [dataset.orig_to_dataset_idx[i] for i in raw_test if i in dataset.orig_to_dataset_idx]
    
    print(f"Raw test first 5:    {raw_test[:5]}")
    print(f"Mapped test first 5: {mapped_test[:5]}")
    
    # Create models and loaders for raw vs mapped
    model = _build_model(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    # Subsets of first 200 samples
    sub_raw = Subset(dataset, raw_test[:200])
    sub_mapped = Subset(dataset, mapped_test[:200])
    
    loader_raw = DataLoader(sub_raw, batch_size=32, shuffle=False, collate_fn=MultiScaleCollate())
    loader_mapped = DataLoader(sub_mapped, batch_size=32, shuffle=False, collate_fn=MultiScaleCollate())
    
    def get_mae(loader):
        ys, mus = [], []
        with torch.no_grad():
            for b1, b2, b3, y, _ in loader:
                b1, b2, b3, y = b1.to(device), b2.to(device), b3.to(device), y.to(device)
                out = model(b1, b2, b3)
                mu = out[0].squeeze()
                ys.extend(y.cpu().numpy().tolist())
                mus.extend(mu.cpu().numpy().tolist())
        return np.abs(np.array(ys) - np.array(mus)).mean()
        
    mae_raw = get_mae(loader_raw)
    mae_mapped = get_mae(loader_mapped)
    
    print("\n=======================================================")
    print(f"MAE with RAW split indices (Unmapped):    {mae_raw:.6f} eV/atom")
    print(f"MAE with MAPPED split indices (Correct):  {mae_mapped:.6f} eV/atom")
    print("=======================================================")

if __name__ == "__main__":
    main()
