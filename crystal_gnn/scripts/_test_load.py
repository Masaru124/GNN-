import torch
from pathlib import Path
import sys
import os

# Add parent directory to path
sys.path.append(os.getcwd())

from crystal_gnn.models.ms_gnn import MultiScaleGNN
from omegaconf import OmegaConf

print("Loading config...")
cfg = OmegaConf.load("configs/default.yaml")
# override for A5
cfg.model.use_attention_fusion = True
cfg.model.use_der = False

print("Building model...")
device = "cuda"
model = MultiScaleGNN(
    hidden_dim=cfg.model.hidden_dim,
    num_encoder_layers=cfg.model.num_encoder_layers,
    dropout_rate=cfg.model.dropout_rate,
    use_attention_fusion=cfg.model.use_attention_fusion,
    use_der=cfg.model.use_der,
).to(device)

print("Building optimizer...")
from torch.optim import AdamW
optimizer = AdamW(
    model.parameters(),
    lr=cfg.training.learning_rate,
    weight_decay=cfg.training.weight_decay,
    fused=True
)

print("Loading checkpoint...")
ckpt_path = "checkpoints/paper_A5_soap_loco_formation_energy_per_atom/last.pt"
ckpt = torch.load(ckpt_path, map_location=device)

print("Loading model state dict...")
model.load_state_dict(ckpt["model_state_dict"])
print("Model state dict loaded.")

print("Loading optimizer state dict...")
optimizer.load_state_dict(ckpt["optimizer_state_dict"])
print("Optimizer state dict loaded.")

print("Done! No hang detected in load_state_dict.")
