import torch
ckpt = torch.load("checkpoints/paper_A5_soap_loco_formation_energy_per_atom/last.pt", weights_only=False, map_location="cpu")
print(f"epoch in ckpt: {ckpt.get('epoch')}")
print(f"max_epochs would resume from: {int(ckpt.get('epoch', 0)) + 1}")
print(f"run_id: {ckpt.get('run_id')}")
print(f"best_val_mae: {ckpt.get('best_val_mae')}")
print(f"history length: {len(ckpt.get('history', []))}")
