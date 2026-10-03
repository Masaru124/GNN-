"""Training entrypoint for Crystal GNN ablations."""
from __future__ import annotations

# MLflow experiment tracking (Item 7) — gracefully disabled if not installed
try:
    import sys as _sys
    _sys.path.insert(0, str(__import__('pathlib').Path(__file__).parents[2]))
    from crystal_gnn.mlflow_config import mlflow_context, log_training_params, log_epoch_metrics, log_final_results
    _HAS_MLFLOW_CONFIG = True
except ImportError:
    _HAS_MLFLOW_CONFIG = False
    def mlflow_context(*a, **kw):
        import contextlib
        return contextlib.nullcontext()
    def log_training_params(*a, **kw): pass
    def log_epoch_metrics(*a, **kw): pass
    def log_final_results(*a, **kw): pass

import argparse
import csv
import gzip
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from pymatgen.core import Structure
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import threading
import queue


class ThreadedPrefetcher:
    """Prefetches batches in a background thread to overlap I/O with GPU compute.
    
    Uses threading (NOT multiprocessing), so:
    - No pickling/serialization (shared memory)
    - No pipes that can deadlock  
    - No Windows spawn issues
    - Daemon thread auto-dies if main thread exits
    
    torch.load releases the GIL during file I/O, enabling true I/O concurrency.
    """

    def __init__(self, loader, device='cuda'):
        self.loader = loader
        self.device = device
        self._queue = queue.Queue(maxsize=2)
        self._len = len(loader)

    def __len__(self):
        return self._len

    def __iter__(self):
        def _produce():
            try:
                for batch in self.loader:
                    b1, b2, b3, y, ids = batch
                    b1 = b1.to(self.device, non_blocking=True)
                    b2 = b2.to(self.device, non_blocking=True)
                    b3 = b3.to(self.device, non_blocking=True)
                    y = y.to(self.device, non_blocking=True)
                    self._queue.put((b1, b2, b3, y, ids))
                self._queue.put(None)  # sentinel: epoch done
            except Exception as exc:
                self._queue.put(exc)  # propagate error to main thread

        t = threading.Thread(target=_produce, daemon=True)
        t.start()

        while True:
            item = self._queue.get(timeout=300)  # 5-min safety timeout
            if item is None:
                break
            if isinstance(item, Exception):
                raise item  # re-raise in main thread
            yield item

from crystal_gnn.data.dataset import MultiScaleCollate, MultiScaleDataset
from crystal_gnn.data.splits import random_split, load_splits
from crystal_gnn.losses.evidential import combined_loss
from crystal_gnn.models.ms_gnn import MultiScaleGNN, SingleScaleGNN


def set_seed(seed: int) -> None:
    """Set deterministic seeds for reproducible training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_data(data_dir: str):
    data_path = Path(data_dir)
    if not data_path.exists():
        script_dir = Path(__file__).resolve().parent
        package_dir = script_dir.parent
        candidates = [
            script_dir / data_dir,
            package_dir / data_dir,
            package_dir / "data" / "raw",
        ]
        for candidate in candidates:
            if candidate.exists():
                data_path = candidate
                break

    structures_path = data_path / "mp_structures.json.gz"
    labels_path = data_path / "mp_labels.csv"

    if not structures_path.exists() or not labels_path.exists():
        raise FileNotFoundError(
            f"Dataset files not found under {data_path}. "
            "Run `python -m crystal_gnn.scripts.download_data --api_key <MP_API_KEY>` "
            "or set --data_dir to the folder containing mp_structures.json.gz and mp_labels.csv."
        )

    with gzip.open(structures_path, "rt", encoding="utf-8") as f_in:
        structures_raw = json.load(f_in)
    # Keep raw structure dicts to avoid creating heavy pymatgen.Structure objects in the main process.
    structures = [row["structure"] for row in structures_raw]

    labels = {}
    with labels_path.open("r", encoding="utf-8") as f_in:
        reader = csv.DictReader(f_in)
        for row in reader:
            labels[row["material_id"]] = {
                "formation_energy_per_atom": float(row["formation_energy_per_atom"]),
                "band_gap": float(row["band_gap"]),
                "crystal_system": row.get("crystal_system", "triclinic"),
                "nelements": int(row.get("nelements", 0)),
            }
    return structures, labels


def parse_radii_string(radii_str: str) -> list[float]:
    """Parse a comma-separated radii string into a list of floats.

    Examples: '4.0,6.0,8.0' -> [4.0, 6.0, 8.0]
    """
    if not radii_str:
        return []
    parts = [p.strip() for p in radii_str.split(",") if p.strip()]
    try:
        return [float(p) for p in parts]
    except ValueError as exc:
        raise ValueError(f"Invalid radii list: {radii_str}") from exc


def _resolve_cfg_path(cfg_path: str) -> Path:
    path = Path(cfg_path)
    if path.exists():
        return path

    script_dir = Path(__file__).resolve().parent
    package_dir = script_dir.parent
    candidates = [
        script_dir / cfg_path,
        package_dir / cfg_path,
        package_dir / "configs" / path.name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"Config file not found: {cfg_path}")


def _merge_cfg(default_cfg_path: str, ablation: str | None) -> dict:
    resolved_cfg_path = _resolve_cfg_path(default_cfg_path)
    base = OmegaConf.load(resolved_cfg_path)
    if ablation:
        ab_path = resolved_cfg_path.parent / f"ablation_{ablation}.yaml"
        if not ab_path.exists():
            raise FileNotFoundError(f"Ablation config not found: {ab_path}")
        ab_cfg = OmegaConf.load(ab_path)
        merged = OmegaConf.merge(base, ab_cfg)
    else:
        merged = base
    return OmegaConf.to_container(merged, resolve=True)


def _build_model(cfg: dict):
    mcfg = cfg["model"]
    radii = mcfg.get("radii", [4.0, 6.0, 8.0])
    if len(radii) == 1:
        return SingleScaleGNN(
            hidden_dim=mcfg["hidden_dim"],
            num_encoder_layers=mcfg["num_encoder_layers"],
            dropout_rate=mcfg["dropout_rate"],
            use_der=mcfg["use_der"],
        )
    return MultiScaleGNN(
        hidden_dim=mcfg["hidden_dim"],
        num_encoder_layers=mcfg["num_encoder_layers"],
        dropout_rate=mcfg["dropout_rate"],
        use_attention_fusion=mcfg["use_attention_fusion"],
        use_der=mcfg["use_der"],
        radii=radii,
    )


def _make_loader(subset, batch_size: int, shuffle: bool, num_workers: int, pin_memory: bool):
    persistent_workers = num_workers > 0 and shuffle
    prefetch_factor = 4 if num_workers > 0 else None
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=MultiScaleCollate(),
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )


def _eval_mae(model, loader, device: str, use_der: bool) -> float:
    model.train()  # Keep MCDropout active by design.
    all_abs = []
    with torch.no_grad():
        for b1, b2, b3, y, _ in loader:
            b1 = b1.to(device)
            b2 = b2.to(device)
            b3 = b3.to(device)
            y = y.to(device)
            out = model(b1, b2, b3)
            mu = out[0] if use_der else out
            all_abs.append(torch.abs(mu - y).mean().item())
    return float(np.mean(all_abs)) if all_abs else float("inf")


def main() -> None:
    start_time = time.time()
    parser = argparse.ArgumentParser(description="Train Crystal GNN model")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--ablation", type=str, default=None, choices=[None, "A1", "A2", "A3", "A4", "A5", "A6", "A7"])
    parser.add_argument("--split", type=str, default="random", choices=["random", "soap_loco", "crystal_system", "composition"])
    parser.add_argument("--target", type=str, default="formation_energy_per_atom", choices=["formation_energy_per_atom", "band_gap"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--log_every_steps", type=int, default=200)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--accumulate_grad_batches", type=int, default=None)
    parser.add_argument("--max_neighbors", type=int, default=None)
    parser.add_argument("--max_structures", type=int, default=None)
    parser.add_argument("--max_epochs", type=int, default=None)
    parser.add_argument("--hidden_dim", type=int, default=None)
    parser.add_argument("--num_encoder_layers", type=int, default=None)
    parser.add_argument("--warm_up_epochs", type=int, default=None)
    parser.add_argument("--evidential_lambda", type=float, default=None)
    parser.add_argument("--val_every_epochs", type=int, default=1)
    parser.add_argument("--fast_mode", action="store_true")
    parser.add_argument("--radii", type=str, default=None, help="Comma-separated radii list, e.g. '4.0,6.0,8.0'")
    parser.add_argument("--run_id", type=str, default=None, help="Custom run ID for deterministic checkpoint paths")
    parser.add_argument("--auto_resume", action="store_true", help="Automatically resume from last.pt in checkpoint directory")
    parser.add_argument("--max_hours", type=float, default=None, help="Maximum training time in hours before saving and exiting")
    parser.add_argument("--soap_loco_idx", type=int, default=0, help="Index of the SOAP-LOCO split to use (0-9)")
    parser.add_argument("--crystal_system_test", type=str, default=None, help="Crystal system to hold out for test")
    args = parser.parse_args()

    cfg = _merge_cfg(args.config, args.ablation)
    # Allow overriding radii from the CLI using --radii
    if args.radii:
        parsed = parse_radii_string(args.radii)
        if parsed:
            cfg["model"]["radii"] = parsed
    cfg["data"]["target"] = args.target
    cfg["splits"]["type"] = args.split
    cfg["training"]["seed"] = args.seed
    if args.batch_size is not None:
        cfg["training"]["batch_size"] = int(args.batch_size)
    if args.accumulate_grad_batches is not None:
        cfg["training"]["accumulate_grad_batches"] = int(args.accumulate_grad_batches)
    if args.max_neighbors is not None:
        cfg["data"]["max_neighbors"] = int(args.max_neighbors)
    if args.max_structures is not None:
        cfg["data"]["max_structures"] = int(args.max_structures)
    if args.max_epochs is not None:
        cfg["training"]["max_epochs"] = int(args.max_epochs)
    if args.hidden_dim is not None:
        cfg["model"]["hidden_dim"] = int(args.hidden_dim)
    if args.num_encoder_layers is not None:
        cfg["model"]["num_encoder_layers"] = int(args.num_encoder_layers)
    if args.warm_up_epochs is not None:
        cfg["training"]["warm_up_epochs"] = int(args.warm_up_epochs)
    if args.evidential_lambda is not None:
        cfg["training"]["evidential_lambda"] = float(args.evidential_lambda)

    set_seed(args.seed)

    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    if args.device == "cuda" and device == "cpu":
        print("CUDA requested but not available in this environment; falling back to CPU.", flush=True)
    if args.fast_mode and device == "cuda":
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = True
    if args.run_id:
        run_id = args.run_id
    else:
        run_id = f"{args.ablation or 'default'}_{args.split}_{args.target}_{int(time.time())}"

    ckpt_dir = Path(os.getenv("CHECKPOINT_DIR", "checkpoints")) / run_id
    res_dir = Path(os.getenv("RESULTS_DIR", "results")) / run_id

    if args.auto_resume and (ckpt_dir / "last.pt").exists() and not args.resume:
        args.resume = str(ckpt_dir / "last.pt")
        print(f"[resume] Automatically resuming from {args.resume}", flush=True)

    ckpt = None
    if args.resume:
        print(f"[resume] Loading checkpoint metadata from {args.resume} ...", flush=True)
        ckpt = torch.load(args.resume, map_location="cpu")

    print(f"[startup] run_id={run_id} device={device}", flush=True)
    print("[startup] loading data from data/raw ...", flush=True)

    structures, labels = load_data("data/raw")
    max_structures = cfg["data"].get("max_structures")
    if max_structures is not None:
        max_n = int(max_structures)
        structures = structures[:max_n]
    print(f"[startup] loaded structures={len(structures)} labels={len(labels)}", flush=True)
    print("[startup] building MultiScaleDataset ...", flush=True)
    dataset = MultiScaleDataset(
        structures=structures,
        labels=labels,
        radii=cfg["model"]["radii"],
        target=args.target,
        cache_dir=cfg["data"]["cache_dir"],
        max_neighbors=cfg["data"]["max_neighbors"],
    )
    print(f"[startup] dataset_size={len(dataset)}", flush=True)

    print(f"[startup] building split type={args.split} ...", flush=True)
    split = None
    if ckpt and "split" in ckpt:
        print("[startup] using split from checkpoint", flush=True)
        split = ckpt["split"]
    else:
        split_type = args.split
        split_dir = Path("data/splits")
        split_file = split_dir / f"{split_type}.json"
        if split_file.exists():
            print(f"[startup] loading split from {split_file} ...", flush=True)
            try:
                with open(split_file, "r") as f:
                    split_data = json.load(f)
                
                if split_type == "random":
                    split = split_data
                elif split_type == "soap_loco":
                    soap_idx = getattr(args, "soap_loco_idx", 0)
                    split = split_data["splits"][soap_idx]
                elif split_type == "crystal_system":
                    sys_key = getattr(args, "crystal_system_test", None)
                    if sys_key is None:
                        sys_key = list(split_data.keys())[0]
                    sys_split = split_data[sys_key]
                    split = {
                        "train": sys_split["train_idx"],
                        "val": sys_split["val_idx"],
                        "test": sys_split["test_idx"]
                    }
                elif split_type == "composition":
                    split = {
                        "train": split_data["train_idx"],
                        "val": split_data["val_idx"],
                        "test": split_data["test_idx"]
                    }
                
                # Normalize keys to 'train', 'val', 'test'
                if split and "train" not in split and "train_idx" in split:
                    split = {
                        "train": split["train_idx"],
                        "val": split["val_idx"],
                        "test": split["test_idx"]
                    }
                
                # Filter indices for max_structures
                if split and max_structures is not None:
                    max_n = int(max_structures)
                    split = {
                        "train": [i for i in split["train"] if i < max_n],
                        "val": [i for i in split["val"] if i < max_n],
                        "test": [i for i in split["test"] if i < max_n]
                    }

                # Map original indices to dataset indices
                if split:
                    mapping = dataset.orig_to_dataset_idx
                    split = {
                        "train": [mapping[i] for i in split["train"] if i in mapping],
                        "val": [mapping[i] for i in split["val"] if i in mapping],
                        "test": [mapping[i] for i in split["test"] if i in mapping]
                    }
            except Exception as e:
                print(f"[warning] failed to load split file {split_file}: {e}. Falling back to random_split.", flush=True)
                split = None
        
        if split is None:
            print("[startup] generating random split ...", flush=True)
            split = random_split(dataset, ratios=tuple(cfg["splits"].get("random_ratios", [0.8, 0.1, 0.1])), seed=args.seed)

    train_ds = torch.utils.data.Subset(dataset, split["train"])
    val_ds = torch.utils.data.Subset(dataset, split["val"])
    print(f"[startup] split sizes train={len(train_ds)} val={len(val_ds)} test={len(split.get('test', []))}", flush=True)

    num_workers = int(args.num_workers) if args.num_workers is not None else int(os.getenv("NUM_WORKERS", "0"))
    # Disable pin_memory to prevent CUDA out of memory errors in PyTorch Geometric's host-to-device pinning
    pin_memory = False
    train_loader = _make_loader(
        train_ds,
        cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = _make_loader(
        val_ds,
        cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
    )

    print(
        f"run_id={run_id} device={device} train_batches={len(train_loader)} val_batches={len(val_loader)} "
        f"batch_size={cfg['training']['batch_size']} accum={cfg['training'].get('accumulate_grad_batches', 4)} "
        f"train_workers={num_workers} val_workers=0 pin_memory={pin_memory} max_neighbors={cfg['data']['max_neighbors']} "
        f"fast_mode={args.fast_mode}",
        flush=True,
    )

    model = _build_model(cfg).to(device)
    if device == "cuda":
        try:
            optimizer = AdamW(
                model.parameters(),
                lr=cfg["training"]["learning_rate"],
                weight_decay=cfg["training"]["weight_decay"],
                fused=True,
            )
        except TypeError:
            optimizer = AdamW(model.parameters(), lr=cfg["training"]["learning_rate"], weight_decay=cfg["training"]["weight_decay"])
    else:
        optimizer = AdamW(model.parameters(), lr=cfg["training"]["learning_rate"], weight_decay=cfg["training"]["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-5)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    start_epoch = 0
    best_val_mae = float("inf")
    history = []
    if ckpt:
        print("[resume] loading model, optimizer, scheduler state dicts ...", flush=True)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = int(ckpt["epoch"]) + 1
        best_val_mae = float(ckpt["best_val_mae"])
        run_id = ckpt.get("run_id", run_id)
        history = ckpt.get("history", [])

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)

    patience = cfg["training"]["patience"]
    bad_epochs = 0
    history = []
    accum = int(cfg["training"].get("accumulate_grad_batches", 4))
    grad_clip_norm = float(cfg["training"].get("grad_clip_norm", 1.0))

    val_every = max(1, int(args.val_every_epochs))

    # MLflow experiment setup & params (Item 7)
    try:
        from crystal_gnn.mlflow_config import setup_mlflow
        setup_mlflow()
    except Exception:
        pass
    log_training_params(
        lr=float(cfg["training"]["learning_rate"]),
        epochs=int(cfg["training"]["max_epochs"]),
        model_arch=str(cfg["model"].get("name", "MultiScaleGNN")),
        batch_size=int(cfg["training"]["batch_size"]),
        n_train=len(train_ds),
        n_val=len(val_ds),
        extra_params={"ablation": args.ablation or "default", "target": args.target, "split": args.split, "run_id": run_id},
    )

    for epoch in range(start_epoch, cfg["training"]["max_epochs"]):
        model.train()
        running = 0.0
        optimizer.zero_grad(set_to_none=True)
        epoch_start = time.time()
        log_every = max(1, int(args.log_every_steps))

        for step, (b1, b2, b3, y, _) in enumerate(train_loader, start=1):
            b1 = b1.to(device, non_blocking=True)
            b2 = b2.to(device, non_blocking=True)
            b3 = b3.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            try:
                with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                    out = model(b1, b2, b3)
                    if cfg["model"]["use_der"]:
                        mu, v, alpha, beta = out
                        loss, _ = combined_loss(
                            mu,
                            v,
                            alpha,
                            beta,
                            y,
                            epoch=epoch,
                            lam=cfg["training"]["evidential_lambda"],
                            warm_up_epochs=cfg["training"]["warm_up_epochs"],
                        )
                    else:
                        mu = out
                        loss = torch.nn.functional.mse_loss(mu, y)
            except torch.OutOfMemoryError as exc:
                if device == "cuda":
                    torch.cuda.empty_cache()
                raise RuntimeError(
                    "CUDA OOM during forward pass. Try: --batch_size 8 --accumulate_grad_batches 8 --max_neighbors 24 --num_workers 2"
                ) from exc

            loss = loss / accum
            scaler.scale(loss).backward()
            running += loss.item() * accum

            if step % accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            if step % log_every == 0 or step == len(train_loader):
                elapsed = time.time() - epoch_start
                avg_loss = running / max(step, 1)
                steps_per_sec = step / max(elapsed, 1e-9)
                print(
                    f"epoch={epoch} step={step}/{len(train_loader)} avg_train_loss={avg_loss:.6f} "
                    f"elapsed={elapsed:.1f}s rate={steps_per_sec:.2f} steps/s",
                    flush=True,
                )

            # Defragment CUDA memory every 1000 steps to prevent rate degradation
            # from variable-sized graph batch allocations
            if device == "cuda" and step % 1000 == 0:
                torch.cuda.empty_cache()

        if len(train_loader) % accum != 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        scheduler.step()
        if device == "cuda":
            torch.cuda.empty_cache()
        do_val = (epoch % val_every) == 0
        val_mae = _eval_mae(model, val_loader, device=device, use_der=cfg["model"]["use_der"]) if do_val else float("nan")
        train_loss = running / max(len(train_loader), 1)
        val_loss = val_mae

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_mae": val_mae})
        if do_val:
            print(
                f"epoch={epoch} train_loss={train_loss:.6f} val_loss={val_loss:.6f} val_mae={val_mae:.6f}",
                flush=True,
            )
        else:
            print(
                f"epoch={epoch} train_loss={train_loss:.6f} val=skipped (val_every_epochs={val_every})",
                flush=True,
            )

        # Write history.csv inside the loop at the end of each epoch
        with (res_dir / "history.csv").open("w", encoding="utf-8", newline="") as f_out:
            writer = csv.DictWriter(f_out, fieldnames=["epoch", "train_loss", "val_loss", "val_mae"])
            writer.writeheader()
            writer.writerows(history)

        # MLflow per-epoch logging (Item 7)
        log_epoch_metrics(
            epoch=epoch,
            train_loss=train_loss,
            val_mae=val_mae if do_val and not np.isnan(val_mae) else None,
        )

        if do_val and val_mae < best_val_mae:
            best_val_mae = val_mae
            bad_epochs = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_val_mae": best_val_mae,
                    "config": cfg,
                    "run_id": run_id,
                    "history": history,
                    "split": split,
                },
                ckpt_dir / "best.pt",
            )
        elif do_val:
            bad_epochs += 1
            if bad_epochs >= patience:
                print("Early stopping triggered", flush=True)
                break

        # Save last checkpoint at the end of every epoch
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_mae": best_val_mae,
                "config": cfg,
                "run_id": run_id,
                "history": history,
                "split": split,
            },
            ckpt_dir / "last.pt",
        )

        # Check time limit
        if args.max_hours is not None:
            elapsed_hours = (time.time() - start_time) / 3600.0
            if elapsed_hours >= args.max_hours:
                print(f"Time limit of {args.max_hours} hours reached (elapsed: {elapsed_hours:.2f} hours). Saving last.pt and exiting cleanly.", flush=True)
                break

    # MLflow final results logging (Item 7)
    best_ckpt = str(ckpt_dir / "best.pt") if (ckpt_dir / "best.pt").exists() else str(ckpt_dir / "last.pt")
    log_final_results(
        val_mae=best_val_mae,
        checkpoint_path=best_ckpt if os.path.exists(best_ckpt) else None,
    )


if __name__ == "__main__":
    main()
