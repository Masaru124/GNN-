"""Evaluate trained Crystal GNN checkpoint."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import time
from pathlib import Path

import numpy as np
import torch
from pymatgen.core import Structure
from torch.utils.data import DataLoader

from crystal_gnn.data.dataset import MultiScaleCollate, MultiScaleDataset
from crystal_gnn.data.splits import random_split
from crystal_gnn.evaluation.metrics import compute_all_metrics
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
        raise FileNotFoundError(
            f"Dataset files not found under {data_path}. "
            "Run `python -m crystal_gnn.scripts.download_data --api_key <MP_API_KEY>` "
            "or set --data_dir to the folder containing mp_structures.json.gz and mp_labels.csv."
        )

    with gzip.open(structures_path, "rt", encoding="utf-8") as f_in:
        structures_raw = json.load(f_in)
    # Keep raw structure dicts; conversion to Structure will happen inside the dataset worker.
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_dir", default="data/raw")
    parser.add_argument("--target", default="formation_energy_per_atom")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--mc_samples",
        type=int,
        default=None,
        help="Override MC-dropout sample count for faster/slower evaluation.",
    )
    parser.add_argument(
        "--log_every_batches",
        type=int,
        default=0,
        help="Print progress every N batches (0 disables periodic logs).",
    )
    parser.add_argument(
        "--max_test_samples",
        type=int,
        default=None,
        help="Optional cap on test samples for quick evaluation.",
    )
    parser.add_argument(
        "--max_structures",
        type=int,
        default=None,
        help="Optional cap on loaded structures before dataset construction.",
    )
    args = parser.parse_args()

    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    print(f"[eval] loading checkpoint={args.checkpoint} device={device}", flush=True)
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = ckpt["config"]

    print("[eval] building model", flush=True)
    model = _build_model(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)

    print(f"[eval] loading data from {args.data_dir}", flush=True)
    structures, labels = _load_data(args.data_dir)
    if args.max_structures is not None:
        max_structures = max(1, int(args.max_structures))
        structures = structures[:max_structures]
        print(f"[eval] limiting loaded structures to {len(structures)}", flush=True)
    print("[eval] building dataset", flush=True)
    dataset = MultiScaleDataset(structures, labels, radii=cfg["model"]["radii"], target=args.target, cache_dir=cfg["data"]["cache_dir"])
    print("[eval] building split", flush=True)
    split = None
    if "split" in ckpt:
        print("[eval] using split from checkpoint", flush=True)
        split = ckpt["split"]
    else:
        split_type = cfg.get("splits", {}).get("type", "random")
        split_dir = Path("data/splits")
        split_file = split_dir / f"{split_type}.json"
        if split_file.exists():
            print(f"[eval] loading split from {split_file} ...", flush=True)
            try:
                with open(split_file, "r") as f:
                    split_data = json.load(f)
                if split_type == "random":
                    split = split_data
                elif split_type == "soap_loco":
                    split = split_data["splits"][0]
                elif split_type == "crystal_system":
                    split = split_data[list(split_data.keys())[0]]
                elif split_type == "composition":
                    split = split_data
                
                if split and "train" not in split and "train_idx" in split:
                    split = {
                        "train": split["train_idx"],
                        "val": split["val_idx"],
                        "test": split["test_idx"]
                    }
                
                if split and args.max_structures is not None:
                    max_n = int(args.max_structures)
                    split = {
                        "train": [i for i in split["train"] if i < max_n],
                        "val": [i for i in split["val"] if i < max_n],
                        "test": [i for i in split["test"] if i < max_n]
                    }
                
                if split:
                    mapping = dataset.orig_to_dataset_idx
                    split = {
                        "train": [mapping[i] for i in split["train"] if i in mapping],
                        "val": [mapping[i] for i in split["val"] if i in mapping],
                        "test": [mapping[i] for i in split["test"] if i in mapping]
                    }
            except Exception as e:
                print(f"[eval warning] failed to load split file {split_file}: {e}", flush=True)
                split = None

        if split is None:
            print("[eval] generating fallback random split ...", flush=True)
            split = random_split(dataset, seed=cfg["training"]["seed"])

    test_indices = split["test"]
    if args.max_test_samples is not None:
        test_indices = test_indices[: max(1, int(args.max_test_samples))]
        print(f"[eval] limiting test split to {len(test_indices)} samples", flush=True)
    test_ds = torch.utils.data.Subset(dataset, test_indices)
    loader = DataLoader(test_ds, batch_size=cfg["training"]["batch_size"], shuffle=False, collate_fn=MultiScaleCollate())
    mc_samples = int(args.mc_samples) if args.mc_samples is not None else int(cfg["uncertainty"]["mc_dropout_T"])

    print(
        f"[eval] device={device} test_size={len(test_ds)} batches={len(loader)} mc_samples={mc_samples}",
        flush=True,
    )

    ys = []
    preds = []
    uncs = []

    model.train()
    start_time = time.time()
    log_every = max(0, int(args.log_every_batches))
    print("[eval] starting batch loop", flush=True)
    with torch.no_grad():
        for batch_idx, (b1, b2, b3, y, _) in enumerate(loader, start=1):
            b1, b2, b3 = b1.to(device), b2.to(device), b3.to(device)
            y = y.to(device)
            uq = model.predict_with_uncertainty(b1, b2, b3, T=mc_samples)
            ys.append(y.cpu().numpy().reshape(-1))
            preds.append(uq["prediction"].cpu().numpy().reshape(-1))
            uncs.append(uq["epistemic"].cpu().numpy().reshape(-1))

            if log_every > 0 and (batch_idx % log_every == 0 or batch_idx == len(loader)):
                elapsed = time.time() - start_time
                rate = batch_idx / max(elapsed, 1e-9)
                print(
                    f"[eval] batch={batch_idx}/{len(loader)} elapsed={elapsed:.1f}s rate={rate:.2f} batches/s",
                    flush=True,
                )

    y_true = np.concatenate(ys)
    y_pred = np.concatenate(preds)
    unc = np.concatenate(uncs)

    cp = LocallyAdaptiveConformalPredictor(target_coverage=cfg["uncertainty"]["conformal_coverage"])
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

    out_path = Path("results") / ckpt.get("run_id", "run") / "evaluation.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
