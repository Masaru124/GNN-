"""
Independent Spot-Check Verification Script for 5 Selected Colliding Pairs.
Loads exported CIF files directly and queries CHGNet outside the main dataset loop.
"""

import os
from pymatgen.core import Structure
from chgnet.model import CHGNet

SPOTCHECK_PAIRS = [
    {
        "category": "Fully Resolved #1",
        "pair_id": 1,
        "formula": "Zr16 N16 O8",
        "cif_1": "results/spotcheck_cifs/spot_0_1_idx_143.cif",
        "cif_2": "results/spotcheck_cifs/spot_0_2_idx_33995.cif",
        "true_target_1": -2.334245, "true_target_2": -2.241597, "true_gap": 0.092648,
    },
    {
        "category": "Fully Resolved #2",
        "pair_id": 84,
        "formula": "Ca4 Bi4 O12",
        "cif_1": "results/spotcheck_cifs/spot_83_1_idx_12139.cif",
        "cif_2": "results/spotcheck_cifs/spot_83_2_idx_48945.cif",
        "true_target_1": -2.164672, "true_target_2": -2.264223, "true_gap": 0.099551,
    },
    {
        "category": "Collapsed #1",
        "pair_id": 5,
        "formula": "Ba2 Nb4 O12",
        "cif_1": "results/spotcheck_cifs/spot_4_1_idx_206.cif",
        "cif_2": "results/spotcheck_cifs/spot_4_2_idx_35021.cif",
        "true_target_1": -2.986412, "true_target_2": -3.216099, "true_gap": 0.229687,
    },
    {
        "category": "Collapsed #2",
        "pair_id": 102,
        "formula": "Bi4 O8",
        "cif_1": "results/spotcheck_cifs/spot_101_1_idx_15667.cif",
        "cif_2": "results/spotcheck_cifs/spot_101_2_idx_45854.cif",
        "true_target_1": -1.387370, "true_target_2": -1.439044, "true_gap": 0.051674,
    },
    {
        "category": "Partially Resolved #1",
        "pair_id": 11,
        "formula": "Zn10 S10",
        "cif_1": "results/spotcheck_cifs/spot_10_1_idx_1072.cif",
        "cif_2": "results/spotcheck_cifs/spot_10_2_idx_24101.cif",
        "true_target_1": -0.911231, "true_target_2": -0.962353, "true_gap": 0.051122,
    },
]

def main():
    print("==========================================================================", flush=True)
    print("      INDEPENDENT SPOT-CHECK VERIFICATION OF 5 SELECTED PAIRS            ", flush=True)
    print("==========================================================================", flush=True)

    print("Loading pretrained CHGNet model...", flush=True)
    chgnet = CHGNet.load()

    print("\n" + "=" * 95)
    print(f"{'Category':<20} | {'Formula':<12} | {'True Gap':<12} | {'CHGNet Pred Gap':<16} | {'Status':<15}")
    print("-" * 95)

    for item in SPOTCHECK_PAIRS:
        s1 = Structure.from_file(item["cif_1"])
        s2 = Structure.from_file(item["cif_2"])

        res1 = float(chgnet.predict_structure(s1)["e"])
        res2 = float(chgnet.predict_structure(s2)["e"])

        pred_gap = abs(res1 - res2)
        true_gap = item["true_gap"]
        ratio = pred_gap / true_gap

        if ratio >= 0.50 or pred_gap >= 0.025:
            status = "Fully Resolved"
        elif ratio >= 0.10:
            status = "Partially Resolved"
        else:
            status = "Collapsed"

        print(f"{item['category']:<20} | {item['formula']:<12} | {true_gap:<12.6f} | {pred_gap:<16.6f} | {status:<15}")
        print(f"  -> CIF 1 ({os.path.basename(item['cif_1'])}): Pred = {res1:.6f} eV/atom (Target: {item['true_target_1']:.6f})")
        print(f"  -> CIF 2 ({os.path.basename(item['cif_2'])}): Pred = {res2:.6f} eV/atom (Target: {item['true_target_2']:.6f})")
        print("-" * 95)

if __name__ == "__main__":
    main()
