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
    # Limit to 5000 structures for lightning fast loading
    dataset = MultiScaleDataset(structures[:5000], labels, radii=config["model"]["radii"], target="formation_energy_per_atom", cache_dir="./data/cache")

    split = checkpoint["split"]
    test_indices = split["test"]
    mapped_test = [dataset.orig_to_dataset_idx[i] for i in test_indices if i in dataset.orig_to_dataset_idx]
    
    print(f"Mapped test samples in first 5k structures: {len(mapped_test)}")

    test_set = Subset(dataset, mapped_test)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False, collate_fn=MultiScaleCollate())

    model = _build_model(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Test 1: model.eval() mode
    model.eval()
    y_true_eval, y_pred_eval = [], []
    with torch.no_grad():
        for b1, b2, b3, y, _ in test_loader:
            b1, b2, b3, y = b1.to(device), b2.to(device), b3.to(device), y.to(device)
            out = model(b1, b2, b3)
            mu = out[0].squeeze()
            y_true_eval.extend(y.cpu().numpy().reshape(-1).tolist())
            y_pred_eval.extend(mu.cpu().numpy().reshape(-1).tolist())

    y_t_eval = np.array(y_true_eval)
    y_p_eval = np.array(y_pred_eval)
    mae_eval = np.abs(y_t_eval - y_p_eval).mean()

    # Test 2: model.train() mode with predict_with_uncertainty (T=1)
    model.train()
    y_true_train, y_pred_train = [], []
    with torch.no_grad():
        for b1, b2, b3, y, _ in test_loader:
            b1, b2, b3, y = b1.to(device), b2.to(device), b3.to(device), y.to(device)
            uq = model.predict_with_uncertainty(b1, b2, b3, T=1)
            y_true_train.extend(y.cpu().numpy().reshape(-1).tolist())
            y_pred_train.extend(uq["prediction"].cpu().numpy().reshape(-1).tolist())

    y_t_tr = np.array(y_true_train)
    y_p_tr = np.array(y_pred_train)
    mae_train = np.abs(y_t_tr - y_p_tr).mean()

    print("\n=======================================================")
    print(f"A7 MAE under model.eval() (No Dropout):      {mae_eval:.6f} eV/atom")
    print(f"A7 MAE under model.train() (Dropout Active): {mae_train:.6f} eV/atom")
    print("=======================================================")
    print(f"y_true head (first 5): {y_t_eval[:5]}")
    print(f"y_pred_eval (first 5): {y_p_eval[:5]}")
    print(f"y_pred_train (first 5): {y_p_tr[:5]}")

if __name__ == "__main__":
    main()
