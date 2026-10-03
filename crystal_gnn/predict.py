# -*- coding: utf-8 -*-
"""
Zero-Training Inference API & CLI for Crystal GNN (A7 Model).

Allows anyone to load pre-trained model weights and predict formation energy 
and calibrated uncertainty for any crystal structure (CIF file or PyMatGen Structure)
WITHOUT retraining.

Usage (CLI):
    python predict.py --cif path/to/crystal.cif

Usage (Python API):
    from predict import CrystalPredictor
    predictor = CrystalPredictor()
    result = predictor.predict_cif("path/to/crystal.cif")
    print(result)
"""

import sys
import os
import argparse
import torch
import numpy as np
from pathlib import Path
from pymatgen.core import Structure

# Include local package paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts"))

from train import _build_model, MultiScaleDataset, MultiScaleCollate
from torch.utils.data import DataLoader

def _find_default_ckpt() -> Path:
    candidates = [
        Path(__file__).parent / "model_inference.pt",
        Path(__file__).parent / "checkpoints" / "paper_A7_soap_loco_formation_energy_per_atom" / "model_inference.pt",
        Path(__file__).parent / "checkpoints" / "paper_A7_soap_loco_formation_energy_per_atom" / "best.pt",
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    return candidates[0]


DEFAULT_CKPT = _find_default_ckpt()


class CrystalPredictor:
    """Out-of-the-box predictor using pretrained A7 model weights."""

    def __init__(self, checkpoint_path: str | Path = DEFAULT_CKPT, device: str | None = None):
        self.ckpt_path = Path(checkpoint_path)
        if not self.ckpt_path.exists():
            raise FileNotFoundError(
                f"Pretrained checkpoint not found at {self.ckpt_path}. "
                "Ensure the repository includes the pretrained best.pt checkpoint."
            )

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[CrystalPredictor] Loading model weights from: {self.ckpt_path}")
        print(f"[CrystalPredictor] Running on device: {self.device}")

        self.checkpoint = torch.load(self.ckpt_path, map_location=self.device)
        self.config = self.checkpoint["config"]

        self.model = _build_model(self.config).to(self.device)
        self.model.load_state_dict(self.checkpoint["model_state_dict"])
        self.model.eval()

        # Split-conformal quantile for the 90% interval (mu ± q·sigma, sigma = DER total
        # std), calibrated on the held-out val split (seed=42, n=2000) with the deterministic
        # forward; test coverage 0.897 at target 0.90 (n=2000). Old value 0.4954 -> 0.624.
        self.q_hat_conformal = 1.0002

    def predict_structure(self, structure: Structure) -> dict:
        """Predict formation energy and uncertainty for a PyMatGen Structure."""
        fake_label = {
            "dummy_id": {"formation_energy_per_atom": 0.0}
        }
        setattr(structure, "material_id", "dummy_id")

        ds = MultiScaleDataset(
            [structure],
            fake_label,
            radii=self.config["model"]["radii"],
            target="formation_energy_per_atom",
            cache_dir="./data/cache"
        )

        if len(ds) == 0:
            raise ValueError("Structure has zero neighbors within smallest cutoff radius (4.0 Å). Cannot construct multiscale graph.")

        loader = DataLoader(ds, batch_size=1, collate_fn=MultiScaleCollate())
        b1, b2, b3, _, _ = next(iter(loader))

        b1 = b1.to(self.device)
        b2 = b2.to(self.device)
        b3 = b3.to(self.device)

        with torch.no_grad():
            out = self.model(b1, b2, b3)
            mu, v, alpha, beta = out
            mu = float(mu.reshape(-1).cpu().item())
            v = float(torch.clamp(v.reshape(-1), min=1e-4).cpu().item())
            alpha = float(torch.clamp(alpha.reshape(-1), min=1.0001).cpu().item())
            beta = float(torch.clamp(beta.reshape(-1), min=1e-4).cpu().item())

            var_tot = (beta * (1.0 + 1.0 / v)) / (alpha - 1.0)
            sigma = float(np.sqrt(max(1e-8, var_tot)))

        # Conformal 90% Calibrated Interval
        half_width_conf = self.q_hat_conformal * sigma
        conf_lower = mu - half_width_conf
        conf_upper = mu + half_width_conf

        # Raw DER 95% Interval
        half_width_raw = 1.96 * sigma

        return {
            "predicted_formation_energy_per_atom_eV": round(mu, 6),
            "evidential_std_eV": round(sigma, 6),
            "conformal_90_interval_eV": [round(conf_lower, 6), round(conf_upper, 6)],
            "conformal_width_eV": round(2 * half_width_conf, 6),
            "raw_der_95_interval_eV": [round(mu - half_width_raw, 6), round(mu + half_width_raw, 6)],
            "raw_der_width_eV": round(2 * half_width_raw, 6),
        }

    def predict_cif(self, cif_path: str | Path) -> dict:
        """Predict formation energy and uncertainty for a CIF file."""
        cif_p = Path(cif_path)
        if not cif_p.exists():
            raise FileNotFoundError(f"CIF file not found: {cif_p}")
        structure = Structure.from_file(cif_p)
        res = self.predict_structure(structure)
        res["cif_file"] = str(cif_p)
        return res


def main():
    parser = argparse.ArgumentParser(description="Zero-training prediction for crystal structures.")
    parser.add_argument("--cif", type=str, required=True, help="Path to CIF crystal structure file")
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CKPT), help="Path to best.pt checkpoint")
    args = parser.parse_args()

    predictor = CrystalPredictor(checkpoint_path=args.checkpoint)
    res = predictor.predict_cif(args.cif)

    print("\n" + "=" * 65)
    print("           CRYSTAL GNN FORMATION ENERGY PREDICTION           ")
    print("=" * 65)
    print(f"CIF File                : {res['cif_file']}")
    print(f"Predicted E_f           : {res['predicted_formation_energy_per_atom_eV']:.6f} eV/atom")
    print(f"Evidential Std (sigma)  : {res['evidential_std_eV']:.6f} eV/atom")
    print(f"Conformal 90% Interval  : [{res['conformal_90_interval_eV'][0]:.6f}, {res['conformal_90_interval_eV'][1]:.6f}] eV/atom")
    print(f"Conformal Width         : {res['conformal_width_eV']:.6f} eV")
    print(f"Raw DER 95% Interval    : [{res['raw_der_95_interval_eV'][0]:.6f}, {res['raw_der_95_interval_eV'][1]:.6f}] eV/atom")
    print("=" * 65)


if __name__ == "__main__":
    main()
