"""
Confirms A4 and A5 configs are truly identical after resolving OmegaConf merges.

Usage:
    python scripts/diff_a4_a5_resolved.py
"""

import sys
import os
from omegaconf import OmegaConf

sys.path.append(os.path.join(os.getcwd(), "scripts"))
from train import _merge_cfg

def main():
    cfg_a4 = _merge_cfg("configs/default.yaml", ablation="A4")
    cfg_a5 = _merge_cfg("configs/default.yaml", ablation="A5")

    print("--- COMPARING RESOLVED CONFIGURATIONS FOR A4 AND A5 ---")
    if cfg_a4 == cfg_a5:
        print("\nMATCH: A4 and A5 resolve to identical configs.")
        print("The shortcut (reusing A4's metrics for A5) is fully justified and mathematically identical.")
        return

    print("\nMISMATCH: A4 and A5 do NOT resolve to identical configs.")
    print("A5's metrics need to be re-run for real -- they cannot be assumed from A4.\n")

    keys = set(cfg_a4.keys()) | set(cfg_a5.keys())
    for k in sorted(keys):
        v4 = cfg_a4.get(k, "<MISSING>")
        v5 = cfg_a5.get(k, "<MISSING>")
        if v4 != v5:
            print(f"  DIFF at top-level key '{k}':")
            print(f"    A4: {v4}")
            print(f"    A5: {v5}")

if __name__ == "__main__":
    main()
