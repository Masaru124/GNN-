# Sample CIF Files for Model Prediction Testing

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
