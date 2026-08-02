"""Run all ablations A1-A7 sequentially."""

from __future__ import annotations

import argparse
import subprocess


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ablations A1-A7")
    parser.add_argument("--split", default="soap_loco")
    parser.add_argument("--target", default="formation_energy_per_atom")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    for ab in ["A1", "A2", "A3", "A4", "A5", "A6", "A7"]:
        cmd = [
            "python",
            "scripts/train.py",
            "--ablation",
            ab,
            "--split",
            args.split,
            "--target",
            args.target,
            "--device",
            args.device,
            "--seed",
            str(args.seed),
        ]
        print("Running", " ".join(cmd))
        proc = subprocess.run(cmd, check=False)
        if proc.returncode != 0:
            raise RuntimeError(f"Ablation {ab} failed with code {proc.returncode}")


if __name__ == "__main__":
    main()
