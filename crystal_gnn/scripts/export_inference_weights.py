# -*- coding: utf-8 -*-
"""
Export Lightweight Inference Weights for MatScreen AI Backend & Git.

Strips out training-only artifacts (Adam optimizer states, momentum buffers, 
scheduler history, split index arrays) from the 221 MB best.pt checkpoint.

Saves a compact `model_inference.pt` (~35 MB) containing ONLY:
  - PyTorch model state_dict
  - Model architecture config
  - Conformal calibration parameters (q_hat_90 = 0.4954)

Resulting file is under GitHub's 100 MB limit and can be tracked in Git.
"""

import os
import sys
import torch
from pathlib import Path

sys.path.append("scripts")
from train import _build_model

CHECKPOINT_DIR = Path("checkpoints/paper_A7_soap_loco_formation_energy_per_atom")
INPUT_CKPT = CHECKPOINT_DIR / "best.pt"
OUTPUT_CKPT = CHECKPOINT_DIR / "model_inference.pt"
ROOT_OUTPUT_CKPT = Path("model_inference.pt")

def main():
    if not INPUT_CKPT.exists():
        print(f"[!] Error: {INPUT_CKPT} not found.")
        return

    print("==========================================================================")
    print("      EXPORTING LIGHTWEIGHT INFERENCE WEIGHTS (STRIPPING OPTIMIZER)       ")
    print("==========================================================================")
    print(f"Input full checkpoint: {INPUT_CKPT} ({INPUT_CKPT.stat().st_size / (1024*1024):.2f} MB)")

    ckpt = torch.load(INPUT_CKPT, map_location="cpu")

    # Create compact inference dictionary
    inference_dict = {
        "model_state_dict": ckpt["model_state_dict"],
        "config": {
            "model": ckpt["config"]["model"],
            "data": {
                "radii": ckpt["config"]["model"]["radii"],
                "target": ckpt["config"]["data"].get("target", "formation_energy_per_atom")
            }
        },
        "conformal_q_hat_90": 0.4954,
        "best_val_mae": ckpt.get("best_val_mae", 0.0644),
    }

    # Save compact checkpoint
    torch.save(inference_dict, OUTPUT_CKPT)
    torch.save(inference_dict, ROOT_OUTPUT_CKPT)

    out_size_mb = OUTPUT_CKPT.stat().st_size / (1024 * 1024)
    print(f"\n[SUCCESS] Exported lightweight inference weights to:")
    print(f"  -> {OUTPUT_CKPT.resolve()} ({out_size_mb:.2f} MB)")
    print(f"  -> {ROOT_OUTPUT_CKPT.resolve()} ({out_size_mb:.2f} MB)")
    print("==========================================================================")
    print(f"File size reduced by {(1 - out_size_mb / (INPUT_CKPT.stat().st_size / (1024*1024))) * 100:.1f}%!")
    print("This lightweight file (<100 MB) can be pushed to GitHub cleanly.")
    print("==========================================================================")

if __name__ == "__main__":
    main()
