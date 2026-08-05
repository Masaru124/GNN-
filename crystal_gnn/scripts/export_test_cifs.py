# -*- coding: utf-8 -*-
"""
Export Representative Test CIF Files for Hands-On Model Testing.

Creates a clean `sample_cifs/` directory containing well-formatted CIF files 
spanning different structural families, collision categories, and benchmark cases.
"""

import os
import shutil
from pathlib import Path
from pymatgen.core import Structure

ROOT_DIR = Path(__file__).parent.parent
TARGET_DIR = ROOT_DIR / "sample_cifs"

# List of source CIFs to organize cleanly
SOURCE_CIFS = [
    # CHGNet Audit Spotcheck Cases
    ("results/spotcheck_cifs/spot_0_1_idx_143.cif", "Zr16_N16_O8_Fully_Resolved_1.cif"),
    ("results/spotcheck_cifs/spot_0_2_idx_33995.cif", "Zr16_N16_O8_Fully_Resolved_2.cif"),
    ("results/spotcheck_cifs/spot_83_1_idx_12139.cif", "Ca4_Bi4_O12_Fully_Resolved_1.cif"),
    ("results/spotcheck_cifs/spot_83_2_idx_48945.cif", "Ca4_Bi4_O12_Fully_Resolved_2.cif"),
    ("results/spotcheck_cifs/spot_4_1_idx_206.cif", "Ba2_Nb4_O12_Collapsed_1.cif"),
    ("results/spotcheck_cifs/spot_4_2_idx_35021.cif", "Ba2_Nb4_O12_Collapsed_2.cif"),
    ("results/spotcheck_cifs/spot_101_1_idx_15667.cif", "Bi4_O8_Collapsed_1.cif"),
    ("results/spotcheck_cifs/spot_101_2_idx_45854.cif", "Bi4_O8_Collapsed_2.cif"),
    ("results/spotcheck_cifs/spot_10_1_idx_1072.cif", "Zn10_S10_Partially_Resolved_1.cif"),
    ("results/spotcheck_cifs/spot_10_2_idx_24101.cif", "Zn10_S10_Partially_Resolved_2.cif"),

    # Benchmark Test Cases
    ("comformer_check_structures/mp-754388.cif", "mp-754388_Perovskite.cif"),
    ("tests/fixtures/mp-1341203.cif", "mp-1341203_Fixture1.cif"),
    ("tests/fixtures/mp-2901430.cif", "mp-2901430_Fixture2.cif"),
]

def main():
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    print("==========================================================================")
    print("             EXPORTING SAMPLE CIF FILES FOR PREDICTOR TESTING             ")
    print("==========================================================================")
    print(f"Target Directory: {TARGET_DIR.resolve()}\n")

    exported = []
    for rel_src, dst_name in SOURCE_CIFS:
        src_path = ROOT_DIR / rel_src
        dst_path = TARGET_DIR / dst_name

        if src_path.exists():
            shutil.copy(src_path, dst_path)
            # Read formula from pymatgen to verify structure validity
            try:
                struct = Structure.from_file(dst_path)
                formula = struct.composition.reduced_formula
                n_atoms = len(struct)
                exported.append((dst_name, formula, n_atoms, str(dst_path)))
                print(f"  [OK] Exported {dst_name:<38} | Formula: {formula:<10} | Atoms: {n_atoms}")
            except Exception as e:
                print(f"  [OK] Exported {dst_name:<38} (raw file copy)")
        else:
            print(f"  [SKIP] Source missing: {src_path}")

    # Write a clean README.md inside sample_cifs/
    readme_content = """# Sample CIF Files for Model Prediction Testing

This directory contains representative CIF files from the benchmark test sets and CHGNet collision audit.

## How to Test

Run prediction on any of these CIF files using the zero-training predictor:

```bash
python predict.py --cif sample_cifs/Zr16_N16_O8_Fully_Resolved_1.cif
```

Or in Python:

```python
from predict import CrystalPredictor

predictor = CrystalPredictor()
result = predictor.predict_cif("sample_cifs/mp-754388_Perovskite.cif")
print(result)
```

## Available CIF Files

| CIF Filename | Formula | Description / Category |
|---|---|---|
| `Zr16_N16_O8_Fully_Resolved_1.cif` | Zr2NO | CHGNet Fully Resolved Pair #1 (Struct A) |
| `Zr16_N16_O8_Fully_Resolved_2.cif` | Zr2NO | CHGNet Fully Resolved Pair #1 (Struct B) |
| `Ca4_Bi4_O12_Fully_Resolved_1.cif` | CaBiO3 | CHGNet Fully Resolved Pair #2 (Struct A) |
| `Ca4_Bi4_O12_Fully_Resolved_2.cif` | CaBiO3 | CHGNet Fully Resolved Pair #2 (Struct B) |
| `Ba2_Nb4_O12_Collapsed_1.cif` | BaNb2O6 | CHGNet Collapsed Pair #1 (Struct A) |
| `Ba2_Nb4_O12_Collapsed_2.cif` | BaNb2O6 | CHGNet Collapsed Pair #1 (Struct B) |
| `Bi4_O8_Collapsed_1.cif` | BiO2 | CHGNet Collapsed Pair #2 (Struct A) |
| `Bi4_O8_Collapsed_2.cif` | BiO2 | CHGNet Collapsed Pair #2 (Struct B) |
| `Zn10_S10_Partially_Resolved_1.cif` | ZnS | CHGNet Partially Resolved Pair (Struct A) |
| `Zn10_S10_Partially_Resolved_2.cif` | ZnS | CHGNet Partially Resolved Pair (Struct B) |
| `mp-754388_Perovskite.cif` | CsPbI3 | Benchmark Perovskite Structure |
| `mp-1341203_Fixture1.cif` | FeO | Standard Test Fixture 1 |
| `mp-2901430_Fixture2.cif` | TiO2 | Standard Test Fixture 2 |
"""
    with open(TARGET_DIR / "README.md", "w", encoding="utf-8") as f:
        f.write(readme_content)

    print("\n" + "=" * 80)
    print(f"Successfully exported {len(exported)} CIF files to {TARGET_DIR}")
    print("=" * 80)

if __name__ == "__main__":
    main()
