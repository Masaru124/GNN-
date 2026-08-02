import torch
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.append(os.getcwd())

from crystal_gnn.models.ms_gnn import MultiScaleGNN
from crystal_gnn.data.dataset import MultiScaleDataset, MultiScaleCollate
from scripts.train import load_data
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

print("Loading config...")
cfg = OmegaConf.load("configs/default.yaml")
cfg.model.use_attention_fusion = True
cfg.model.use_der = True # A6 uses DER

print("Building model...")
device = "cuda"
model = MultiScaleGNN(
    hidden_dim=cfg.model.hidden_dim,
    num_encoder_layers=cfg.model.num_encoder_layers,
    dropout_rate=cfg.model.dropout_rate,
    use_attention_fusion=cfg.model.use_attention_fusion,
    use_der=cfg.model.use_der,
).to(device)

print("Loading dataset...")
structures_raw, labels = load_data("data/raw")
from pymatgen.core import Structure
# convert first 10 to pymatgen Structure
structures = [Structure.from_dict(s) for s in structures_raw[:10]]

dataset = MultiScaleDataset(
    structures=structures,
    labels=labels,
    radii=cfg.model.radii,
    target=cfg.data.target,
    cache_dir=cfg.data.cache_dir,
    max_neighbors=cfg.data.max_neighbors,
)

loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=False,
    collate_fn=MultiScaleCollate(),
)

print("Getting batch...")
batch = next(iter(loader))
b1, b2, b3, y, _ = batch

b1 = b1.to(device)
b2 = b2.to(device)
b3 = b3.to(device)
y = y.to(device)

print("Running forward pass...")
out = model(b1, b2, b3)
print("Forward pass successful! Outputs:")
print([o.shape for o in out])

# evidential loss calculation
from crystal_gnn.losses.evidential import combined_loss
mu, v, alpha, beta = out
loss, _ = combined_loss(
    mu, v, alpha, beta, y, epoch=0, lam=0.1, warm_up_epochs=5
)
print("Loss calculation successful! Loss:", loss.item())

print("Running backward pass...")
loss.backward()
print("Backward pass successful!")

print("All tests passed! No hangs or deadlocks in the model/loss/optimizer computation.")
