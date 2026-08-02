# Crystal GNN

Crystal GNN is a publication-focused, structure-aware multi-scale crystal graph neural network that combines three periodic graph radii with attention fusion and calibrated uncertainty using MC Dropout, Deep Evidential Regression, and split-conformal prediction for robust OOD evaluation under realistic structural distribution shifts.

## Installation

```bash
git clone <your-repo-url>
cd crystal_gnn
pip install -e .
# Linux/macOS
export MP_API_KEY="your_key_here"
# Windows PowerShell
$env:MP_API_KEY="your_key_here"
```

## Quick Start

```bash
python scripts/download_data.py --api_key $MP_API_KEY
python scripts/train.py --ablation A7 --split random --seed 42
python scripts/evaluate.py --checkpoint checkpoints/<run_id>/best.pt
```

## 3D Web Demo

Launch a browser app that accepts crystal input (CIF/POSCAR), predicts target energy with uncertainty, and visualizes both the 3D crystal and converted graph:

```bash
python -m uvicorn webapp.app:app --host 0.0.0.0 --port 8000
```

Then open `http://localhost:8000`.

Notes:

- The app auto-selects the newest `checkpoints/*/best.pt` if you leave checkpoint blank.
- You can paste either CIF or POSCAR, or set format to auto-detect.

## Reproduce Paper Results

### 1. Download Data

python scripts/download_data.py --api_key $MP_API_KEY

### 2. Build All Splits

python scripts/build_splits.py

### 3. Reproduce Expressivity Results (Theorem 1)

python scripts/run_wl_check.py --chemsys O-Ti --max_materials 40

### 4. Train Proposed Model (A7)

python scripts/train.py --ablation A7 --split soap_loco --seed 42

### 5. Run All Ablations

python scripts/run_ablations.py --split soap_loco

### 6. Evaluate with Conformal Prediction

python scripts/conformal_calibrate.py --checkpoint checkpoints/A7/best.pt

### 7. Full Evaluation Report

python scripts/evaluate.py --checkpoint checkpoints/A7/best.pt

## Project Structure

```text
crystal_gnn/
├── README.md                         # Project overview and reproduction commands
├── pyproject.toml                    # Build system and package metadata
├── setup.py                          # Editable install support
├── requirements.txt                  # Runtime and research dependencies
├── .env.example                      # Environment variable template
├── .gitignore                        # Ignore datasets, checkpoints and caches
├── configs/                          # Default and ablation configurations
├── data/                             # Downloading, preprocessing, dataset, split builders
├── models/                           # Encoder, fusion, DER head, full model assembly
├── losses/                           # Evidential and warm-up losses
├── uncertainty/                      # MC Dropout and conformal calibration
├── expressivity/                     # WL collision/resolution theorem checks
├── evaluation/                       # Metrics and plotting utilities
├── scripts/                          # CLI scripts for full workflow execution
└── tests/                            # Pytest fixtures and module coverage
```

## Theorem 1 (Expressivity Separation)

Informally, multi-scale periodic crystal graphs are strictly more expressive than any fixed-radius graph in this setting; the implementation and reproducibility checks are in `expressivity/wl_check.py` and include collision-resolution scans and theorem-pair verification.

## Configuration

All experiments are driven by YAML files in `configs/`. Use `configs/default.yaml` for the proposed model and `configs/ablation_A1.yaml` through `configs/ablation_A7.yaml` for ablations. Command-line arguments can override split type, target property, seed, device, and resume checkpoint.

## Hardware Requirements

Target hardware is an RTX 4050 (6GB VRAM). Typical training uses batch size 32 with gradient accumulation 4 (effective 128). For memory pressure, reduce batch size and increase accumulation. Depending on dataset size and split, expect minutes to tens of minutes per epoch, and multi-hour full ablation runs.

## Citation

```bibtex
@article{crystal_gnn_2026,
  title={Multi-Scale Crystal Graph Neural Network with Provably Calibrated Uncertainty under Structural Distribution Shift},
  author={Anonymous},
  journal={npj Computational Materials},
  year={2026}
}
```
