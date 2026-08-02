# Crystal GNN Run Summary

## What Was Done

This workspace was used to fix GPU training, stabilize the training pipeline, speed up runs, and validate the model with fast evaluation and conformal calibration.

## Environment Fixes

- Confirmed the original problem was CPU-only PyTorch in the `.venv311` environment.
- Installed CUDA-enabled PyTorch and matching PyG wheels.
- Verified CUDA availability on the RTX 4050 system.

## Training Stability Fixes

- Fixed CUDA OOM issues by reducing batch size and neighbor count.
- Forced validation loading to use `num_workers=0` on Windows to avoid multiprocessing memory errors.
- Reduced noisy zero-neighbor logging in the dataset path.
- Added a safer fast-mode configuration for training runs.
- Added gradient clipping and NaN/Inf sanitization in the model path to avoid crashes from unstable evidential outputs.

## Evaluation Improvements

- Added progress logs to `scripts/evaluate.py`.
- Added fast-eval controls:
  - `--mc_samples` to reduce MC dropout passes
  - `--log_every_batches` to print progress during evaluation
  - `--max_structures` to cap structures before dataset build
  - `--max_test_samples` to cap test samples
- Added progress logs and fast caps to `scripts/conformal_calibrate.py` as well.
- Fixed the model path so `predict_with_uncertainty()` supports `T=1` for fast evaluation.

## Successful Training Runs

### 1. Tiny smoke run

Command:

```powershell
$env:PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"
Push-Location C:\Users\User\Desktop\GNN\crystal_gnn
C:\Users\User\Desktop\GNN\.venv311\Scripts\python.exe -u scripts/train.py --ablation A7 --split soap_loco --seed 42 --device cuda --fast_mode --num_workers 0 --batch_size 8 --accumulate_grad_batches 8 --max_neighbors 12 --max_structures 1000 --max_epochs 3 --val_every_epochs 3 --log_every_steps 100
```

Result:

- Completed successfully.
- No OOM or NaN crash.
- Demonstrated fast throughput on CUDA.

### 2. Medium fast run

Command:

```powershell
$env:PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"
Push-Location C:\Users\User\Desktop\GNN\crystal_gnn
C:\Users\User\Desktop\GNN\.venv311\Scripts\python.exe -u scripts/train.py --ablation A7 --split soap_loco --seed 42 --device cuda --fast_mode --num_workers 0 --batch_size 8 --accumulate_grad_batches 8 --max_neighbors 12 --max_structures 5000 --max_epochs 8 --val_every_epochs 4 --log_every_steps 100
```

Result:

- Completed successfully.
- No OOM or NaN crash.
- Train loss improved from about `0.365` to `0.058`.
- Validation MAE improved from about `0.385` to `0.161`.

### 3. Final fast run with end-of-run validation

Command:

```powershell
$env:PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"
Push-Location C:\Users\User\Desktop\GNN\crystal_gnn
C:\Users\User\Desktop\GNN\.venv311\Scripts\python.exe -u scripts/train.py --ablation A7 --split soap_loco --seed 42 --device cuda --fast_mode --num_workers 0 --batch_size 8 --accumulate_grad_batches 8 --max_neighbors 12 --max_structures 5000 --max_epochs 9 --val_every_epochs 4 --log_every_steps 50
```

Result:

- Completed successfully.
- No OOM or NaN crash.
- Final train loss about `0.058`.
- Final validation MAE about `0.1616`.
- This became the best fast baseline run.

## Evaluation Results

### Fast eval on capped subset

Command:

```powershell
C:\Users\User\Desktop\GNN\.venv311\Scripts\python.exe -u scripts/evaluate.py --checkpoint checkpoints\A7_soap_loco_formation_energy_per_atom_1775286844\best.pt --device cuda --mc_samples 1 --log_every_batches 1 --max_structures 1000 --max_test_samples 100
```

Result:

```json
{
  "mae": 0.175008844435215,
  "rmse": 0.2356847247375051,
  "r2": 0.9115591421596949,
  "mape": 0.21593356164214936,
  "coverage": 0.88,
  "mean_width": 0.7219664214917763,
  "median_width": 0.6749721419524648,
  "ece": 0.3518460834608891,
  "spearman_rho": 0.09589435774309724,
  "p_value": 0.5076801308216786
}
```

Notes:

- This run confirmed evaluation worked end to end.
- `mc_samples=1` required a code fix in `predict_with_uncertainty()`.
- The capped eval was used for speed and visibility.

## Conformal Calibration Results

### Fast calibration on capped subset

Command:

```powershell
C:\Users\User\Desktop\GNN\.venv311\Scripts\python.exe -u scripts/conformal_calibrate.py --checkpoint checkpoints\A7_soap_loco_formation_energy_per_atom_1775286844\best.pt --device cuda --mc_samples 1 --max_structures 1000 --max_val_samples 100 --max_test_samples 100 --log_every_batches 1
```

Result:

```json
{
  "empirical_coverage": 0.87,
  "target_coverage": 0.9,
  "coverage_gap": -0.030000000000000027,
  "mean_interval_width": 0.681292981435712,
  "median_interval_width": 0.6543303400915355,
  "coverage_passed": false
}
```

Notes:

- Calibration completed successfully.
- Coverage was slightly below target on the capped quick subset.
- A larger calibration subset or more MC samples may improve coverage.

## Important Commands Used

- Training fast baseline:
  - `--num_workers 0`
  - `--batch_size 8`
  - `--accumulate_grad_batches 8`
  - `--max_neighbors 12`
  - `--max_structures 5000`
  - `--max_epochs 9`
  - `--val_every_epochs 4`
- Fast evaluation:
  - `--mc_samples 1`
  - `--log_every_batches 1`
  - `--max_structures 1000`
  - `--max_test_samples 100`
- Fast calibration:
  - `--mc_samples 1`
  - `--max_structures 1000`
  - `--max_val_samples 100`
  - `--max_test_samples 100`

## Best Fast Baseline

The best quick baseline from this session is:

- `max_structures=5000`
- `max_epochs=9`
- `max_neighbors=12`
- `num_workers=0`
- `batch_size=8`
- `accumulate_grad_batches=8`

This setting completed cleanly and gave the strongest fast-run validation result.

## Next Recommended Step

If you want better quality without going back to the very long full run, scale up gradually:

1. `max_structures=8000`
2. `max_epochs=10`
3. Keep `num_workers=0` and `max_neighbors=12`

## Stronger Dev-Check (Completed)

### Training run: 20k structures, 20 epochs, smaller model

Command:

```powershell
$env:PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"
Push-Location C:\Users\User\Desktop\GNN\crystal_gnn
C:\Users\User\Desktop\GNN\.venv311\Scripts\python.exe -u scripts/train.py --ablation A7 --split soap_loco --seed 42 --device cuda --fast_mode --num_workers 0 --batch_size 8 --accumulate_grad_batches 4 --max_neighbors 24 --max_structures 20000 --max_epochs 20 --hidden_dim 64 --num_encoder_layers 2 --val_every_epochs 5 --log_every_steps 200
```

Observed result highlights:

- Completed successfully.
- Throughput stabilized around ~30 steps/s after warm-up.
- Training loss improved from `0.208353` at epoch 0 to `0.006575` at epoch 19.
- Validation MAE improved from `0.246222` (epoch 0) to `0.064878` (epoch 15), and `0.070658` at epoch 10.

### Evaluation for 60-Epoch Run (20k, mc_samples=10)

Command:

```powershell
Push-Location C:\Users\User\Desktop\GNN\crystal_gnn
C:\Users\User\Desktop\GNN\.venv311\Scripts\python.exe -u scripts/evaluate.py --checkpoint checkpoints\A7_soap_loco_formation_energy_per_atom_1775292506\best.pt --device cuda --mc_samples 10 --log_every_batches 25 --max_structures 20000
```

Result:

```json
{
  "mae": 0.06240024799058059,
  "rmse": 0.08221553449555087,
  "r2": 0.9871221102366561,
  "coverage": 0.8693333333333333,
  "mean_width": 0.3805170014323333,
  "ece": 0.3191305770476384,
  "spearman_rho": -0.18132849157946915,
  "p_value": 1.4869508843958286e-12
}
```

### Conformal calibration for 60-Epoch Run (20k, mc_samples=10)

Command:

```powershell
Push-Location C:\Users\User\Desktop\GNN\crystal_gnn
C:\Users\User\Desktop\GNN\.venv311\Scripts\python.exe -u scripts/conformal_calibrate.py --checkpoint checkpoints\A7_soap_loco_formation_energy_per_atom_1775292506\best.pt --device cuda --mc_samples 10 --max_structures 20000 --log_every_batches 25
```

Result:

```json
{
  "empirical_coverage": 0.896,
  "target_coverage": 0.9,
  "coverage_gap": -0.0040000000000000036,
  "mean_interval_width": 0.43903428421068325,
  "median_interval_width": 0.4527272316626163,
  "coverage_passed": true
}
```

Interpretation:

- This stronger dev-check closed the coverage gap substantially (`0.87 -> 0.896`) and passed conformal target coverage.
- Point metrics improved significantly vs the earlier 5k fast baseline.
- Raw ECE remains relatively high, so evidential calibration still needs improvement before paper claims.

## 60-Epoch DER Fix Run (Completed)

### Training run: 20k structures, 60 epochs, short warm-up

Command:

```powershell
$env:PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"
Push-Location C:\Users\User\Desktop\GNN\crystal_gnn
C:\Users\User\Desktop\GNN\.venv311\Scripts\python.exe -u scripts/train.py --ablation A7 --split soap_loco --seed 42 --device cuda --fast_mode --num_workers 0 --batch_size 8 --accumulate_grad_batches 4 --max_neighbors 24 --max_structures 20000 --max_epochs 60 --hidden_dim 64 --num_encoder_layers 2 --warm_up_epochs 5 --val_every_epochs 5 --log_every_steps 200
```

Observed result highlights:

- Completed successfully.
- Epoch 59 finished cleanly with train loss about `-1.802709`.
- Throughput stabilized around `28-29` steps/s late in training.

### Evaluation (20k, mc_samples=10)

Command:

```powershell
Push-Location C:\Users\User\Desktop\GNN\crystal_gnn
C:\Users\User\Desktop\GNN\.venv311\Scripts\python.exe -u scripts/evaluate.py --checkpoint checkpoints\A7_soap_loco_formation_energy_per_atom_1775295575\best.pt --device cuda --mc_samples 10 --log_every_batches 25 --max_structures 20000
```

Result:

```json
{
  "mae": 0.03402937544583498,
  "rmse": 0.04856639334651327,
  "r2": 0.9955062527707884,
  "mape": 0.3522019157685815,
  "coverage": 0.922,
  "mean_width": 0.8210745878561001,
  "median_width": 0.18134859962333152,
  "ece": 0.08376686869242726,
  "spearman_rho": 0.28847521646762975,
  "p_value": 3.874180302536628e-30
}
```

### Conformal calibration (20k, mc_samples=10)

Command:

```powershell
Push-Location C:\Users\User\Desktop\GNN\crystal_gnn
C:\Users\User\Desktop\GNN\.venv311\Scripts\python.exe -u scripts/conformal_calibrate.py --checkpoint checkpoints\A7_soap_loco_formation_energy_per_atom_1775295575\best.pt --device cuda --mc_samples 10 --max_structures 20000 --log_every_batches 25
```

Result:

```json
{
  "empirical_coverage": 0.9055,
  "target_coverage": 0.9,
  "coverage_gap": 0.005499999999999949,
  "mean_interval_width": 0.7492680077492322,
  "median_interval_width": 0.16687695344500242,
  "coverage_passed": true
}
```

Interpretation:

- This run is the best result in the session so far.
- MAE dropped to `0.0340`, R² reached `0.9955`, and Spearman rho turned positive at `0.2885`.
- ECE dropped sharply to `0.0838`, which clears the earlier calibration concern.
- Conformal coverage passed at `0.9055`, so both point accuracy and uncertainty quality are now in a paper-credible range for a dev-scale check.
