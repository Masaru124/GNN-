"""Quartile analysis of A7 vs A1 predictions and uncertainties on the test set."""

import json
import gzip
import csv
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

from crystal_gnn.data.dataset import MultiScaleDataset, MultiScaleCollate
from crystal_gnn.models.ms_gnn import MultiScaleGNN, SingleScaleGNN
from crystal_gnn.uncertainty.conformal import LocallyAdaptiveConformalPredictor


def _load_data(data_dir: str):
    data_path = Path(data_dir)
    if not data_path.exists():
        script_dir = Path(__file__).resolve().parent
        package_dir = script_dir.parent
        candidates = [
            script_dir / data_dir,
            package_dir / data_dir,
            package_dir / "data" / "raw",
        ]
        for candidate in candidates:
            if candidate.exists():
                data_path = candidate
                break

    structures_path = data_path / "mp_structures.json.gz"
    labels_path = data_path / "mp_labels.csv"

    if not structures_path.exists() or not labels_path.exists():
        raise FileNotFoundError(f"Dataset files not found under {data_path}.")

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


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()
    
    device = args.device
    print(f"Using device: {device}", flush=True)

    cache_file = Path("results/inference_cache.npz")
    conformal_coverage = 0.90

    if cache_file.exists():
        print(f"Loading predictions from cache {cache_file}...", flush=True)
        cache_data = np.load(cache_file)
        y_true = cache_data["y_true"]
        y_pred_a7 = cache_data["y_pred_a7"]
        unc_a7 = cache_data["unc_a7"]
        y_pred_a1 = cache_data["y_pred_a1"]
        unc_a1 = cache_data["unc_a1"]
    else:
        # 1. Load checkpoints
        ckpt_path_a7 = Path("checkpoints/paper_A7_soap_loco_formation_energy_per_atom/best.pt")
        ckpt_path_a1 = Path("checkpoints/paper_A1_soap_loco_formation_energy_per_atom/best.pt")

        print(f"Loading A7 checkpoint from {ckpt_path_a7}...", flush=True)
        ckpt_a7 = torch.load(ckpt_path_a7, map_location=device)
        cfg_a7 = ckpt_a7["config"]
        conformal_coverage = cfg_a7.get("uncertainty", {}).get("conformal_coverage", 0.90)
        model_a7 = _build_model(cfg_a7)
        model_a7.load_state_dict(ckpt_a7["model_state_dict"])
        model_a7 = model_a7.to(device)
        model_a7.train()  # Active dropout

        print(f"Loading A1 checkpoint from {ckpt_path_a1}...", flush=True)
        ckpt_a1 = torch.load(ckpt_path_a1, map_location=device)
        cfg_a1 = ckpt_a1["config"]
        model_a1 = _build_model(cfg_a1)
        model_a1.load_state_dict(ckpt_a1["model_state_dict"])
        model_a1 = model_a1.to(device)
        model_a1.train()  # Active dropout

        # 2. Load dataset and split
        print("Loading structure data...", flush=True)
        structures, labels = _load_data("data/raw")
        structures = structures[:50000]

        # Both models were trained on the same split, so we use the split from the A7 checkpoint
        split = ckpt_a7["split"]
        test_indices = split["test"]

        print("Building dataset...", flush=True)
        dataset = MultiScaleDataset(
            structures=structures,
            labels=labels,
            radii=cfg_a7["model"]["radii"],
            target="formation_energy_per_atom",
            cache_dir=cfg_a7["data"]["cache_dir"],
        )

        test_ds = torch.utils.data.Subset(dataset, test_indices)
        loader = DataLoader(
            test_ds,
            batch_size=cfg_a7["training"]["batch_size"],
            shuffle=False,
            collate_fn=MultiScaleCollate(),
        )

        mc_samples = 30
        print(f"Running inference over {len(test_ds)} test structures in {len(loader)} batches...", flush=True)

        ys = []
        preds_a7 = []
        uncs_a7 = []
        preds_a1 = []
        uncs_a1 = []

        start_time = time.time()
        with torch.no_grad():
            for batch_idx, (b1, b2, b3, y, _) in enumerate(loader, start=1):
                b1, b2, b3 = b1.to(device), b2.to(device), b3.to(device)
                y = y.to(device)

                # A7 predictions and uncertainties
                uq_a7 = model_a7.predict_with_uncertainty(b1, b2, b3, T=mc_samples)
                # A1 predictions and uncertainties
                uq_a1 = model_a1.predict_with_uncertainty(b1, b2, b3, T=mc_samples)

                ys.append(y.cpu().numpy().reshape(-1))
                preds_a7.append(uq_a7["prediction"].cpu().numpy().reshape(-1))
                uncs_a7.append(uq_a7["epistemic"].cpu().numpy().reshape(-1))
                preds_a1.append(uq_a1["prediction"].cpu().numpy().reshape(-1))
                uncs_a1.append(uq_a1["epistemic"].cpu().numpy().reshape(-1))

                if batch_idx % 50 == 0 or batch_idx == len(loader):
                    elapsed = time.time() - start_time
                    rate = batch_idx / max(elapsed, 1e-9)
                    print(f"Batch {batch_idx}/{len(loader)} finished. Elapsed: {elapsed:.1f}s ({rate:.2f} batches/s)", flush=True)

                if device == "cuda" and batch_idx % 10 == 0:
                    torch.cuda.empty_cache()

        y_true = np.concatenate(ys)
        y_pred_a7 = np.concatenate(preds_a7)
        unc_a7 = np.concatenate(uncs_a7)
        y_pred_a1 = np.concatenate(preds_a1)
        unc_a1 = np.concatenate(uncs_a1)

        # Save to cache
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            cache_file,
            y_true=y_true,
            y_pred_a7=y_pred_a7,
            unc_a7=unc_a7,
            y_pred_a1=y_pred_a1,
            unc_a1=unc_a1
        )
        print(f"Saved predictions cache to {cache_file}", flush=True)

    # 3. Conformal Calibration
    n_cal = max(50, int(0.25 * len(y_true)))
    print(f"Using {n_cal} samples for calibration, leaving {len(y_true) - n_cal} for prediction.", flush=True)

    # A7 calibration
    cp_a7 = LocallyAdaptiveConformalPredictor(target_coverage=conformal_coverage)
    cp_a7.calibrate(y_pred_a7[:n_cal], unc_a7[:n_cal], y_true[:n_cal])
    intervals_a7 = cp_a7.predict(y_pred_a7[n_cal:], unc_a7[n_cal:])

    # A1 calibration
    cp_a1 = LocallyAdaptiveConformalPredictor(target_coverage=conformal_coverage)
    cp_a1.calibrate(y_pred_a1[:n_cal], unc_a1[:n_cal], y_true[:n_cal])
    intervals_a1 = cp_a1.predict(y_pred_a1[n_cal:], unc_a1[n_cal:])

    # Slice evaluation subsets
    y_true_eval = y_true[n_cal:]
    y_pred_a7_eval = y_pred_a7[n_cal:]
    unc_a7_eval = unc_a7[n_cal:]
    y_pred_a1_eval = y_pred_a1[n_cal:]
    unc_a1_eval = unc_a1[n_cal:]

    lower_a7 = intervals_a7["lower"]
    upper_a7 = intervals_a7["upper"]
    lower_a1 = intervals_a1["lower"]
    upper_a1 = intervals_a1["upper"]

    # 4. Split by A7 epistemic uncertainty quartiles
    q25 = np.percentile(unc_a7_eval, 25)
    q50 = np.percentile(unc_a7_eval, 50)
    q75 = np.percentile(unc_a7_eval, 75)

    print("\nA7 Uncertainty Quartile Thresholds:", flush=True)
    print(f"  Q1: <= {q25:.6f}", flush=True)
    print(f"  Q2: {q25:.6f} to {q50:.6f}", flush=True)
    print(f"  Q3: {q50:.6f} to {q75:.6f}", flush=True)
    print(f"  Q4: > {q75:.6f}", flush=True)

    quartiles = [
        ("Q1 (Low)", unc_a7_eval <= q25),
        ("Q2 (Mid-Low)", (unc_a7_eval > q25) & (unc_a7_eval <= q50)),
        ("Q3 (Mid-High)", (unc_a7_eval > q50) & (unc_a7_eval <= q75)),
        ("Q4 (High)", unc_a7_eval > q75),
    ]

    results = []
    for name, mask in quartiles:
        n_samples = int(np.sum(mask))

        # A7 metrics
        mae_a7 = float(np.mean(np.abs(y_true_eval[mask] - y_pred_a7_eval[mask])))
        cov_a7 = float(np.mean((y_true_eval[mask] >= lower_a7[mask]) & (y_true_eval[mask] <= upper_a7[mask])))
        width_a7 = float(np.mean(upper_a7[mask] - lower_a7[mask]))

        # A1 metrics
        mae_a1 = float(np.mean(np.abs(y_true_eval[mask] - y_pred_a1_eval[mask])))
        cov_a1 = float(np.mean((y_true_eval[mask] >= lower_a1[mask]) & (y_true_eval[mask] <= upper_a1[mask])))
        width_a1 = float(np.mean(upper_a1[mask] - lower_a1[mask]))

        mean_unc_a7 = float(np.mean(unc_a7_eval[mask]))
        mean_unc_a1 = float(np.mean(unc_a1_eval[mask]))

        results.append({
            "Quartile": name,
            "Count": n_samples,
            "Mean_A7_Unc": mean_unc_a7,
            "Mean_A1_Unc": mean_unc_a1,
            "A7_MAE": mae_a7,
            "A1_MAE": mae_a1,
            "A7_Coverage": cov_a7,
            "A1_Coverage": cov_a1,
            "A7_Width": width_a7,
            "A1_Width": width_a1,
        })

    # Print markdown table (no emojis to prevent encoding issues)
    print("\n### Quartile Analysis Table", flush=True)
    print("| Quartile | Count | Mean A7 Unc | A7 MAE | A1 MAE | A7 Coverage | A1 Coverage | A7 Width | A1 Width |", flush=True)
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- |", flush=True)
    for r in results:
        print(
            f"| {r['Quartile']} | {r['Count']} | {r['Mean_A7_Unc']:.6f} | {r['A7_MAE']:.6f} | {r['A1_MAE']:.6f} | "
            f"{r['A7_Coverage']:.4f} | {r['A1_Coverage']:.4f} | {r['A7_Width']:.4f} | {r['A1_Width']:.4f} |",
            flush=True,
        )

    out_path = Path("results/quartile_analysis.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved quartile analysis results to {out_path}", flush=True)


if __name__ == "__main__":
    main()
