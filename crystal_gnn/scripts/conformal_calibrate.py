"""Calibrate split conformal intervals from checkpoint predictions."""

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
from crystal_gnn.models.ms_gnn import MultiScaleGNN, SingleScaleGNN
from crystal_gnn.uncertainty.conformal import SplitConformalPredictor


def _load_data(data_dir: str):
    with gzip.open(Path(data_dir) / "mp_structures.json.gz", "rt", encoding="utf-8") as f_in:
        structures_raw = json.load(f_in)
    structures = [Structure.from_dict(row["structure"]) for row in structures_raw]

    labels = {}
    with (Path(data_dir) / "mp_labels.csv").open("r", encoding="utf-8") as f_in:
        for row in csv.DictReader(f_in):
            labels[row["material_id"]] = {
                "formation_energy_per_atom": float(row["formation_energy_per_atom"]),
                "band_gap": float(row["band_gap"]),
            }
    return structures, labels


def _build_model(cfg):
    if len(cfg["model"]["radii"]) == 1:
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
        radii=cfg["model"]["radii"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Conformal calibration from trained model")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_dir", default="data/raw")
    parser.add_argument("--target", default="formation_energy_per_atom")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mc_samples", type=int, default=None, help="Override MC-dropout sample count.")
    parser.add_argument("--max_structures", type=int, default=None, help="Cap loaded structures before dataset build.")
    parser.add_argument("--max_val_samples", type=int, default=None, help="Cap validation samples for quick calibration.")
    parser.add_argument("--max_test_samples", type=int, default=None, help="Cap test samples for quick calibration.")
    parser.add_argument("--log_every_batches", type=int, default=0, help="Print progress every N batches (0 disables periodic logs).")
    args = parser.parse_args()

    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    print(f"[calib] loading checkpoint={args.checkpoint} device={device}", flush=True)
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = ckpt["config"]

    print("[calib] building model", flush=True)
    model = _build_model(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    print(f"[calib] loading data from {args.data_dir}", flush=True)
    structures, labels = _load_data(args.data_dir)
    if args.max_structures is not None:
        structures = structures[: max(1, int(args.max_structures))]
        print(f"[calib] limiting loaded structures to {len(structures)}", flush=True)
    print("[calib] building dataset", flush=True)
    dataset = MultiScaleDataset(structures, labels, radii=cfg["model"]["radii"], target=args.target, cache_dir=cfg["data"]["cache_dir"])
    print("[calib] building split", flush=True)
    split = random_split(dataset, seed=cfg["training"]["seed"])

    val_indices = split["val"]
    test_indices = split["test"]
    if args.max_val_samples is not None:
        val_indices = val_indices[: max(1, int(args.max_val_samples))]
        print(f"[calib] limiting val split to {len(val_indices)} samples", flush=True)
    if args.max_test_samples is not None:
        test_indices = test_indices[: max(1, int(args.max_test_samples))]
        print(f"[calib] limiting test split to {len(test_indices)} samples", flush=True)

    val_loader = DataLoader(torch.utils.data.Subset(dataset, val_indices), batch_size=cfg["training"]["batch_size"], shuffle=False, collate_fn=MultiScaleCollate())
    test_loader = DataLoader(torch.utils.data.Subset(dataset, test_indices), batch_size=cfg["training"]["batch_size"], shuffle=False, collate_fn=MultiScaleCollate())
    mc_samples = int(args.mc_samples) if args.mc_samples is not None else int(cfg["uncertainty"]["mc_dropout_T"])

    def collect(loader):
        ys, preds, sig = [], [], []
        model.train()
        start_time = time.time()
        log_every = max(0, int(args.log_every_batches))
        with torch.no_grad():
            for batch_idx, (b1, b2, b3, y, _) in enumerate(loader, start=1):
                b1, b2, b3 = b1.to(device), b2.to(device), b3.to(device)
                y = y.to(device)
                uq = model.predict_with_uncertainty(b1, b2, b3, T=mc_samples)
                ys.append(y.cpu().numpy().reshape(-1))
                preds.append(uq["prediction"].cpu().numpy().reshape(-1))
                sig.append(uq["epistemic"].cpu().numpy().reshape(-1))
                if log_every > 0 and (batch_idx % log_every == 0 or batch_idx == len(loader)):
                    elapsed = time.time() - start_time
                    rate = batch_idx / max(elapsed, 1e-9)
                    print(f"[calib] batch={batch_idx}/{len(loader)} elapsed={elapsed:.1f}s rate={rate:.2f} batches/s", flush=True)
        return np.concatenate(ys), np.concatenate(preds), np.concatenate(sig)

    y_cal, p_cal, s_cal = collect(val_loader)
    y_tst, p_tst, s_tst = collect(test_loader)

    cp = SplitConformalPredictor(target_coverage=cfg["uncertainty"]["conformal_coverage"])
    cp.calibrate(p_cal, s_cal, y_cal)
    intervals = cp.predict(p_tst, s_tst)
    coverage = cp.evaluate_coverage(intervals, y_tst)

    out_dir = Path("results") / ckpt.get("run_id", "run")
    out_dir.mkdir(parents=True, exist_ok=True)
    cp.save(str(out_dir / "conformal.json"))
    (out_dir / "conformal_eval.json").write_text(json.dumps(coverage, indent=2), encoding="utf-8")
    print(json.dumps(coverage, indent=2))


if __name__ == "__main__":
    main()
