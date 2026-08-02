"""
Task 3: Fast Batched ALIGNN Zero-Shot Evaluation on SOAP-LOCO Test Set (Path A)
Evaluates published pretrained `jv_formation_energy_peratom_alignn` model
directly on the SOAP-LOCO holdout test set to benchmark external SOTA transfer MAE.
"""

import os
import sys
import json
import gzip
import csv
import torch
import dgl
import numpy as np
from pathlib import Path
from tqdm import tqdm

from pymatgen.core import Structure
from jarvis.core.atoms import Atoms
from alignn.pretrained import get_figshare_model
from alignn.graphs import Graph

def load_soap_loco_test_set():
    print("[1/3] Loading dataset and SOAP-LOCO split...", flush=True)
    raw_dir = Path("data/raw")
    struct_file = raw_dir / "mp_structures.json.gz"
    label_file = raw_dir / "mp_labels.csv"

    with gzip.open(struct_file, "rt", encoding="utf-8") as f:
        structs_raw = json.load(f)

    labels = {}
    with open(label_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels[row["material_id"]] = float(row["formation_energy_per_atom"])

    split_file = Path("data/splits/soap_loco.json")
    with open(split_file, "r") as f:
        split_data = json.load(f)
    test_indices = split_data["splits"][0]["test"]

    test_samples = []
    for idx in test_indices:
        if idx < len(structs_raw):
            row = structs_raw[idx]
            mid = row["material_id"]
            if mid in labels:
                test_samples.append({
                    "material_id": mid,
                    "structure": row["structure"],
                    "target": labels[mid],
                })

    print(f"Loaded {len(test_samples)} test samples from SOAP-LOCO split 0.", flush=True)
    return test_samples

def evaluate_alignn(test_samples, batch_size=64):
    print("[2/3] Loading pretrained ALIGNN model onto GPU...", flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "jv_formation_energy_peratom_alignn"
    model = get_figshare_model(model_name).to(device)
    model.eval()

    results = []
    targets = []
    preds = []

    print(f"[3/3] Running GPU batched inference loop (batch_size={batch_size})...", flush=True)
    
    # Process samples in batches
    for i in tqdm(range(0, len(test_samples), batch_size), desc="ALIGNN Batched GPU Inference"):
        batch_samples = test_samples[i:i + batch_size]
        g_list = []
        lg_list = []
        valid_batch_samples = []

        for sample in batch_samples:
            try:
                pmg_struct = Structure.from_dict(sample["structure"])
                j_atoms = Atoms(
                    lattice_mat=pmg_struct.lattice.matrix,
                    coords=pmg_struct.cart_coords,
                    elements=[s.symbol for s in pmg_struct.species],
                    cartesian=True
                )
                g, lg = Graph.atom_dgl_multigraph(j_atoms, cutoff=8.0, max_neighbors=12)
                g_list.append(g)
                lg_list.append(lg)
                valid_batch_samples.append(sample)
            except Exception:
                continue

        if not g_list:
            continue

        with torch.no_grad():
            g_batch = dgl.batch(g_list).to(device)
            lg_batch = dgl.batch(lg_list).to(device)
            out = model((g_batch, lg_batch, None))
            batch_preds = out.view(-1).cpu().numpy().tolist()

        for sample, pred_y in zip(valid_batch_samples, batch_preds):
            mid = sample["material_id"]
            true_y = sample["target"]
            targets.append(true_y)
            preds.append(pred_y)
            results.append({
                "material_id": mid,
                "true_formation_energy_per_atom": true_y,
                "alignn_pred_formation_energy_per_atom": pred_y,
                "error": abs(true_y - pred_y)
            })

    targets = np.array(targets)
    preds = np.array(preds)
    mae = np.abs(targets - preds).mean()
    rmse = np.sqrt(((targets - preds) ** 2).mean())
    r2 = 1.0 - ((targets - preds) ** 2).sum() / ((targets - targets.mean()) ** 2).sum()

    return results, mae, rmse, r2

def main():
    test_samples = load_soap_loco_test_set()
    results, mae, rmse, r2 = evaluate_alignn(test_samples, batch_size=64)

    out_csv = Path("results/alignn_soap_loco_zeroshot_eval.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "material_id",
            "true_formation_energy_per_atom",
            "alignn_pred_formation_energy_per_atom",
            "error"
        ])
        writer.writeheader()
        writer.writerows(results)

    print("\n==========================================================================", flush=True)
    print("     ALIGNN ZERO-SHOT TRANSFER BENCHMARK ON SOAP-LOCO TEST SET (PATH A)   ", flush=True)
    print("==========================================================================", flush=True)
    print(f"Pretrained Model Name:        jv_formation_energy_peratom_alignn")
    print(f"Evaluated Test Samples:       {len(results)}")
    print(f"Zero-Shot Test MAE:           {mae:.4f} eV/atom")
    print(f"Zero-Shot Test RMSE:          {rmse:.4f} eV/atom")
    print(f"Zero-Shot Test R^2:           {r2:.4f}")
    print("==========================================================================", flush=True)
    print(f"Saved full predictions CSV to {out_csv}", flush=True)

if __name__ == "__main__":
    main()
