"""Orchestrator script to sequentially run and resume Crystal GNN ablations under a time budget."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def load_state(state_path: Path, ablations: list[str]) -> dict:
    if state_path.exists():
        try:
            with open(state_path, "r", encoding="utf-8") as f:
                state = json.load(f)
            # Validate and merge in any missing ablations
            for ab in ablations:
                if ab not in state:
                    state[ab] = {"status": "pending", "run_id": "", "current_epoch": -1, "metrics": {}}
            return state
        except Exception as e:
            print(f"[orchestrator warning] failed to load state file: {e}. Starting fresh.", flush=True)

    # Initialize fresh state
    state = {}
    for ab in ablations:
        state[ab] = {
            "status": "pending",
            "run_id": "",
            "current_epoch": -1,
            "metrics": {}
        }
    return state


def save_state(state: dict, state_path: Path) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def print_summary_table(state: dict, ablations: list[str]) -> None:
    print("\n" + "=" * 80)
    print("                      CRYSTAL GNN RUNS SUMMARY TABLE")
    print("=" * 80)
    print(f"{'Ablation':<10} | {'Status':<10} | {'Epochs':<8} | {'MAE (eV/a)':<11} | {'R2':<6} | {'ECE':<6} | {'Spearman':<8} | {'Coverage':<8}")
    print("-" * 80)
    
    for ab in ablations:
        info = state.get(ab, {})
        status = info.get("status", "pending")
        metrics = info.get("metrics", {})
        
        # Determine number of completed epochs
        run_id = info.get("run_id")
        epochs_str = "-"
        if run_id:
            ckpt_path = Path("checkpoints") / run_id / "last.pt"
            if ckpt_path.exists():
                try:
                    ckpt = torch_load_meta(ckpt_path)
                    epochs_str = f"{ckpt.get('epoch', 0) + 1}"
                except Exception:
                    pass
        
        if status == "completed" and metrics:
            mae = f"{metrics.get('mae', float('nan')):.4f}"
            r2 = f"{metrics.get('r2', float('nan')):.3f}"
            ece = f"{metrics.get('ece', float('nan')):.3f}"
            spearman = f"{metrics.get('spearman_rho', float('nan')):.3f}"
            coverage = f"{metrics.get('coverage', float('nan')):.3f}"
        else:
            mae, r2, ece, spearman, coverage = "-", "-", "-", "-", "-"
            
        print(f"{ab:<10} | {status:<10} | {epochs_str:<8} | {mae:<11} | {r2:<6} | {ece:<6} | {spearman:<8} | {coverage:<8}")
    print("=" * 80 + "\n")


def torch_load_meta(path: Path) -> dict:
    """Helper to load PyTorch checkpoint metadata using a light sub-process to avoid importing torch in main process."""
    # We do a simple python call to print epoch
    script = f"import torch; ckpt=torch.load(r'{path}', map_location='cpu'); print(ckpt.get('epoch', 0))"
    proc = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True)
    epoch = int(proc.stdout.strip())
    return {"epoch": epoch}


def main() -> None:
    parser = argparse.ArgumentParser(description="Managed orchestrator for Crystal GNN ablations")
    parser.add_argument("--max_hours", type=float, default=None, help="Time limit in hours for this run session (e.g. 8.0)")
    parser.add_argument("--split", default="soap_loco", choices=["random", "soap_loco", "crystal_system", "composition"])
    parser.add_argument("--target", default="formation_energy_per_atom", choices=["formation_energy_per_atom", "band_gap"])
    parser.add_argument("--max_structures", type=int, default=50000)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_encoder_layers", type=int, default=3)
    parser.add_argument("--warm_up_epochs", type=int, default=5)
    parser.add_argument("--val_every_epochs", type=int, default=5)
    parser.add_argument("--log_every_steps", type=int, default=500)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ablations", default="A7,A1,A2,A3,A4,A5,A6", help="Comma-separated list of ablations to run")
    parser.add_argument("--run_prefix", default="paper", help="Prefix for run IDs to separate runs")
    parser.add_argument("--status", action="store_true", help="Print summary table of current state and exit")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of dataloader worker processes")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--accumulate_grad_batches", type=int, default=4, help="Gradient accumulation steps")
    args = parser.parse_args()

    ab_list = [a.strip() for a in args.ablations.split(",") if a.strip()]
    state_path = Path("checkpoints") / f"{args.run_prefix}_managed_state.json"
    
    state = load_state(state_path, ab_list)

    if args.status:
        print_summary_table(state, ab_list)
        return

    session_start = time.time()
    
    # Pre-paper runs logic loop
    for ab in ab_list:
        info = state[ab]
        status = info["status"]
        
        if status == "completed":
            print(f"[orchestrator] Ablation {ab} is already completed. Skipping.", flush=True)
            continue
            
        run_id = f"{args.run_prefix}_{ab}_{args.split}_{args.target}"
        info["run_id"] = run_id
        save_state(state, state_path)

        # Calculate time budget
        elapsed_hours = (time.time() - session_start) / 3600.0
        if args.max_hours is not None:
            remaining_hours = args.max_hours - elapsed_hours
            if remaining_hours <= 0.05: # Less than 3 minutes left
                print(f"[orchestrator] Session time limit ({args.max_hours} hours) reached. Exiting.", flush=True)
                break
            print(f"\n[orchestrator] Starting/Resuming ablation {ab} (Remaining session time: {remaining_hours:.2f} hours)...", flush=True)
        else:
            remaining_hours = None
            print(f"\n[orchestrator] Starting/Resuming ablation {ab} (No session time limit)...", flush=True)

        # Build training command
        cmd = [
            sys.executable, "-u", "scripts/train.py",
            "--ablation", ab,
            "--split", args.split,
            "--target", args.target,
            "--seed", str(args.seed),
            "--device", args.device,
            "--num_workers", str(args.num_workers),
            "--batch_size", str(args.batch_size),
            "--accumulate_grad_batches", str(args.accumulate_grad_batches),
            "--max_neighbors", "24",
            "--max_structures", str(args.max_structures),
            "--max_epochs", str(args.max_epochs),
            "--hidden_dim", str(args.hidden_dim),
            "--num_encoder_layers", str(args.num_encoder_layers),
            "--warm_up_epochs", str(args.warm_up_epochs),
            "--val_every_epochs", str(args.val_every_epochs),
            "--log_every_steps", str(args.log_every_steps),
            "--run_id", run_id,
            "--auto_resume",
            "--fast_mode"
        ]
        
        # If running A1 or A4, use evidential lambda = 0 since they don't have UQ
        if ab in ["A1", "A4"]:
            # Note: train.py automatically toggles use_der based on config/ablation, but we can be explicit
            pass

        if remaining_hours is not None:
            cmd.extend(["--max_hours", f"{remaining_hours:.4f}"])

        print(f"[orchestrator] Running command: {' '.join(cmd)}", flush=True)
        
        info["status"] = "training"
        save_state(state, state_path)

        # Launch training
        # We set PyTorch memory settings as requested by the user
        env = os.environ.copy()
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"
        
        proc = subprocess.run(cmd, env=env, check=False)
        
        if proc.returncode != 0:
            print(f"[orchestrator error] training failed for ablation {ab} with code {proc.returncode}.", flush=True)
            # Exit session immediately on training crash
            sys.exit(proc.returncode)

        # Check if training completed all epochs
        ckpt_path = Path("checkpoints") / run_id / "last.pt"
        if ckpt_path.exists():
            try:
                ckpt = torch_load_meta(ckpt_path)
                last_epoch = ckpt.get("epoch", 0)
                info["current_epoch"] = last_epoch
                save_state(state, state_path)
                
                # Check if it reached max_epochs
                if last_epoch >= args.max_epochs - 1:
                    print(f"[orchestrator] Ablation {ab} completed all {args.max_epochs} epochs! Starting evaluation...", flush=True)
                    
                    # Check if best.pt exists; fall back to last.pt if not
                    best_path = Path("checkpoints") / run_id / "best.pt"
                    last_path = Path("checkpoints") / run_id / "last.pt"
                    ckpt_to_eval = best_path if best_path.exists() else last_path

                    # Launch evaluation
                    eval_cmd = [
                        sys.executable, "scripts/evaluate.py",
                        "--checkpoint", str(ckpt_to_eval),
                        "--device", args.device,
                        "--max_structures", str(args.max_structures)
                    ]
                    print(f"[orchestrator] Evaluating: {' '.join(eval_cmd)}", flush=True)
                    eval_proc = subprocess.run(eval_cmd, check=False)
                    
                    if eval_proc.returncode == 0:
                        eval_file = Path("results") / run_id / "evaluation.json"
                        if eval_file.exists():
                            with open(eval_file, "r") as f:
                                metrics = json.load(f)
                            info["metrics"] = metrics
                            info["status"] = "completed"
                            print(f"[orchestrator] Ablation {ab} finished and evaluated successfully!", flush=True)
                        else:
                            print(f"[orchestrator error] evaluation.json not found for {ab} after evaluation run.", flush=True)
                            info["status"] = "pending"
                    else:
                        print(f"[orchestrator error] evaluation script failed for {ab}.", flush=True)
                        info["status"] = "pending"
                else:
                    print(f"[orchestrator] Ablation {ab} stopped at epoch {last_epoch + 1}/{args.max_epochs} due to time limit.", flush=True)
                    # Exit session since time limit was reached
                    break
            except Exception as e:
                print(f"[orchestrator error] failed to inspect checkpoint for {ab}: {e}", flush=True)
                break
        else:
            print(f"[orchestrator error] checkpoint last.pt not found for {ab}.", flush=True)
            break
            
        save_state(state, state_path)

    print_summary_table(state, ab_list)


if __name__ == "__main__":
    main()
