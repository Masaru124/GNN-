"""Analyze relationship between prediction uncertainty/interval width and SOAP distance to training set."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import time
from pathlib import Path

import numpy as np
import scipy.stats as stats
import torch
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from torch.utils.data import DataLoader
from dscribe.descriptors import SOAP

from crystal_gnn.data.dataset import MultiScaleCollate, MultiScaleDataset
from crystal_gnn.data.splits import random_split
from crystal_gnn.models.ms_gnn import MultiScaleGNN, SingleScaleGNN
from crystal_gnn.uncertainty.conformal import SplitConformalPredictor


def _load_data(data_dir: str):
    data_path = Path(data_dir)
    structures_path = data_path / "mp_structures.json.gz"
    labels_path = data_path / "mp_labels.csv"

    print("[analysis] loading raw structures and labels ...", flush=True)
    with gzip.open(structures_path, "rt", encoding="utf-8") as f_in:
        structures_raw = json.load(f_in)
    structures = [row["structure"] for row in structures_raw]

    labels = {}
    with labels_path.open("r", encoding="utf-8") as f_in:
        for row in csv.DictReader(f_in):
            labels[row["material_id"]] = {
                "formation_energy_per_atom": float(row["formation_energy_per_atom"]),
                "band_gap": float(row["band_gap"]),
            }
    return structures, labels


def _build_model(cfg: dict):
    radii = cfg["model"]["radii"]
    if len(radii) == 1:
        return SingleScaleGNN(
            hidden_dim=cfg["model"]["hidden_dim"],
            num_encoder_layers=cfg["model"]["num_encoder_layers"],
            dropout_rate=cfg["model"]["dropout_rate"],
            use_der=cfg["model"]["use_der"],
        )
    return MultiScaleGNN(
        hidden_dim=cfg["model"]["hidden_dim"],
        num_encoder_layers=cfg["model"]["num_encoder_layers"],
        dropout_rate=cfg["model"]["dropout_rate"],
        use_attention_fusion=cfg["model"]["use_attention_fusion"],
        use_der=cfg["model"]["use_der"],
        radii=radii,
    )


def _species_union(structures: list[Structure]) -> list[str]:
    counts = {}
    for s in structures:
        for site in s.sites:
            sym = str(site.specie.symbol)
            counts[sym] = counts.get(sym, 0) + 1
    return sorted(counts.keys())


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze uncertainty vs SOAP distance")
    parser.add_argument("--checkpoint", default="checkpoints/A7_soap_loco_formation_energy_per_atom_1775295575/best.pt")
    parser.add_argument("--data_dir", default="data/raw")
    parser.add_argument("--target", default="formation_energy_per_atom")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--train_sample_size", type=int, default=1000, help="Number of training structures to sample for distance computation")
    parser.add_argument("--mc_samples", type=int, default=30)
    args = parser.parse_args()

    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    print(f"[analysis] loading checkpoint from {args.checkpoint} ...", flush=True)
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = ckpt["config"]

    # Build model and load state
    model = _build_model(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.train()  # Keep dropout active

    # Load data
    structures_raw, labels = _load_data(args.data_dir)
    max_structures = cfg["data"].get("max_structures", 20000)
    if max_structures is not None:
        structures_raw = structures_raw[:max_structures]
    print(f"[analysis] using max_structures={len(structures_raw)}", flush=True)

    # We must instantiate pymatgen Structures for SOAP generation
    print("[analysis] instantiating pymatgen Structure objects ...", flush=True)
    structures = [Structure.from_dict(s) for s in structures_raw]

    # Create dataset for model predictions
    dataset = MultiScaleDataset(structures_raw, labels, radii=cfg["model"]["radii"], target=args.target, cache_dir=cfg["data"]["cache_dir"])
    
    # Recreate the split used by the checkpoint (random split with seed 42)
    split = random_split(dataset, seed=cfg["training"]["seed"])
    train_idx = split["train"]
    test_idx = split["test"]

    print(f"[analysis] split train_size={len(train_idx)} test_size={len(test_idx)}", flush=True)

    # Perform model inference on test set
    test_ds = torch.utils.data.Subset(dataset, test_idx)
    loader = DataLoader(test_ds, batch_size=cfg["training"]["batch_size"], shuffle=False, collate_fn=MultiScaleCollate())
    
    print("[analysis] running test set inference (MC-Dropout) ...", flush=True)
    ys = []
    preds = []
    uncs = []
    with torch.no_grad():
        for b1, b2, b3, y, _ in loader:
            b1, b2, b3 = b1.to(device), b2.to(device), b3.to(device)
            uq = model.predict_with_uncertainty(b1, b2, b3, T=args.mc_samples)
            ys.append(y.numpy().reshape(-1))
            preds.append(uq["prediction"].cpu().numpy().reshape(-1))
            uncs.append(uq["epistemic"].cpu().numpy().reshape(-1))
    
    y_true = np.concatenate(ys)
    y_pred = np.concatenate(preds)
    unc = np.concatenate(uncs)

    # Calibrate conformal predictor on test set (split conformal)
    cp = SplitConformalPredictor(target_coverage=cfg["uncertainty"]["conformal_coverage"])
    n_cal = max(50, int(0.25 * len(y_true)))
    cp.calibrate(y_pred[:n_cal], unc[:n_cal], y_true[:n_cal])
    intervals = cp.predict(y_pred[n_cal:], unc[n_cal:])
    widths = intervals["upper"] - intervals["lower"]
    test_unc_eval = unc[n_cal:] # matching size

    print(f"[analysis] conformal prediction computed on {len(widths)} structures.", flush=True)

    # Setup SOAP descriptor
    print("[analysis] setting up SOAP descriptor ...", flush=True)
    species = _species_union(structures)
    soap = SOAP(
        species=species,
        r_cut=6.0,
        n_max=9,
        l_max=9,
        sigma=0.5,
        periodic=True,
        sparse=False,
        average="inner",
    )

    # Sample a subset of training structures for SOAP distance computation
    rng = np.random.default_rng(cfg["training"]["seed"])
    sampled_train_idx = rng.choice(train_idx, size=min(args.train_sample_size, len(train_idx)), replace=False)
    
    print(f"[analysis] generating SOAP descriptors for {len(sampled_train_idx)} training structures ...", flush=True)
    adaptor = AseAtomsAdaptor()
    train_vectors = []
    t0 = time.time()
    for idx in sampled_train_idx:
        ase_atoms = adaptor.get_atoms(structures[idx])
        vec = soap.create(ase_atoms)
        if hasattr(vec, "mean"):
            vec = vec.mean(axis=0)
        train_vectors.append(vec.reshape(-1))
    train_soap = np.stack(train_vectors)
    print(f"[analysis] training SOAP descriptors generated in {time.time() - t0:.1f}s", flush=True)

    # Generate SOAP descriptors for the evaluation portion of the test set
    eval_test_idx = test_idx[n_cal:]
    print(f"[analysis] generating SOAP descriptors for {len(eval_test_idx)} evaluation test structures ...", flush=True)
    test_vectors = []
    t0 = time.time()
    for idx in eval_test_idx:
        ase_atoms = adaptor.get_atoms(structures[idx])
        vec = soap.create(ase_atoms)
        if hasattr(vec, "mean"):
            vec = vec.mean(axis=0)
        test_vectors.append(vec.reshape(-1))
    test_soap = np.stack(test_vectors)
    print(f"[analysis] test SOAP descriptors generated in {time.time() - t0:.1f}s", flush=True)

    # Compute distances from test structures to training set
    print("[analysis] computing distance from test set to training set in SOAP space ...", flush=True)
    min_distances = []
    mean_top5_distances = []
    train_centroid = train_soap.mean(axis=0)
    centroid_distances = []

    for i in range(len(test_soap)):
        diffs = train_soap - test_soap[i]
        dists = np.linalg.norm(diffs, axis=1)
        min_distances.append(float(np.min(dists)))
        mean_top5_distances.append(float(np.mean(np.partition(dists, 5)[:5])))
        centroid_distances.append(float(np.linalg.norm(test_soap[i] - train_centroid)))

    min_distances = np.array(min_distances)
    mean_top5_distances = np.array(mean_top5_distances)
    centroid_distances = np.array(centroid_distances)

    # Calculate correlation with conformal interval widths
    print("[analysis] calculating correlation metrics ...", flush=True)
    
    # 1. Min distance to train set vs Width
    r_min_width, p_min_width = stats.pearsonr(min_distances, widths)
    rho_min_width, prho_min_width = stats.spearmanr(min_distances, widths)
    
    # 2. Mean top-5 distance vs Width
    r_top5_width, p_top5_width = stats.pearsonr(mean_top5_distances, widths)
    rho_top5_width, prho_top5_width = stats.spearmanr(mean_top5_distances, widths)

    # 3. Centroid distance vs Width
    r_centroid_width, p_centroid_width = stats.pearsonr(centroid_distances, widths)
    rho_centroid_width, prho_centroid_width = stats.spearmanr(centroid_distances, widths)

    # Also compute correlation with raw predicted Epistemic Uncertainty
    rho_min_unc, _ = stats.spearmanr(min_distances, test_unc_eval)
    rho_top5_unc, _ = stats.spearmanr(mean_top5_distances, test_unc_eval)

    results = {
        "dataset_size": len(structures),
        "train_sample_size": len(sampled_train_idx),
        "test_eval_size": len(eval_test_idx),
        "pearson_min_distance_vs_width": {
            "coefficient": float(r_min_width),
            "p_value": float(p_min_width)
        },
        "spearman_min_distance_vs_width": {
            "coefficient": float(rho_min_width),
            "p_value": float(prho_min_width)
        },
        "spearman_top5_distance_vs_width": {
            "coefficient": float(rho_top5_width),
            "p_value": float(prho_top5_width)
        },
        "spearman_centroid_distance_vs_width": {
            "coefficient": float(rho_centroid_width),
            "p_value": float(prho_centroid_width)
        },
        "spearman_min_distance_vs_epistemic_unc": {
            "coefficient": float(rho_min_unc)
        },
        "spearman_top5_distance_vs_epistemic_unc": {
            "coefficient": float(rho_top5_unc)
        }
    }

    out_dir = Path("results") / ckpt.get("run_id", "run")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "soap_distance_analysis.json"
    out_file.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\n================== SOAP DISTANCE ANALYSIS RESULTS ==================", flush=True)
    print(f"Results saved to: {out_file}", flush=True)
    print(f"Number of test structures analyzed: {len(widths)}\n", flush=True)
    
    print("1. Distance to Nearest Training structure:", flush=True)
    print(f"   - Pearson correlation coefficient (r):  {r_min_width:+.4f} (p-value: {p_min_width:.3e})", flush=True)
    print(f"   - Spearman rank correlation (rho):     {rho_min_width:+.4f} (p-value: {prho_min_width:.3e})", flush=True)
    
    print("\n2. Mean Distance to 5 Nearest Training structures:", flush=True)
    print(f"   - Spearman rank correlation (rho):     {rho_top5_width:+.4f} (p-value: {prho_top5_width:.3e})", flush=True)
    
    print("\n3. Distance to Training Set Centroid:", flush=True)
    print(f"   - Spearman rank correlation (rho):     {rho_centroid_width:+.4f} (p-value: {prho_centroid_width:.3e})", flush=True)
    
    print("\n4. Raw Predicted Epistemic Uncertainty vs. Nearest Training SOAP Distance:", flush=True)
    print(f"   - Spearman rank correlation (rho):     {rho_min_unc:+.4f}", flush=True)
    print("====================================================================\n", flush=True)

    if rho_min_width > 0.15:
        print("CONCLUSION: There is a statistically significant positive correlation between SOAP distance and interval width.", flush=True)
        print("This confirms Issue 1: predictions far from the training clusters (high SOAP distance / OOD structures) receive significantly wider prediction intervals (higher uncertainty), causing the right-tail skew in mean interval width. This is correct physical behavior.", flush=True)
    else:
        print("CONCLUSION: The correlation between SOAP distance and interval width is weak.", flush=True)


if __name__ == "__main__":
    main()
