#!/usr/bin/env python3
"""Resumable, pausable LOCO cross-conformal scheduler (train + score + report).

Runs folds one at a time. For each fold in order:
  1. if `checkpoints/loco_A7_soap_cluster<F>_seed42/best.pt` is missing, train it
     via `train.py --auto_resume` (mid-epoch restart survives power cuts)
  2. if `research/loco_cross_conformal/fold<F>_scores.npz` is missing, score it
     via `loco_cross_conformal.py score --folds <F>`
  3. when all 10 folds are scored, regenerate the summary with `loco_report.py`

Durability model (power-cut safe):
  * ground truth is ON DISK ARTIFACTS (TRAIN_DONE marker / fold*_scores.npz),
    never the log: on every restart the scheduler re-derives progress from them,
    so the state file can be lost, stale or hand-edited without re-training
    anything. best.pt alone is deliberately insufficient — it is written every
    improved epoch, long before the run finishes.
  * the state JSON is written atomically (tmp + os.replace) after each transition
    and is only a human-readable mirror for the monitor script.
  * a single-instance lock (PID file) prevents the double-scheduler bug that
    launched two trainers for one fold; before training a fold we also adopt any
    already-running trainer for that fold instead of starting a second one.
  * pause  = Ctrl+C / kill this process (the child train.py is terminated too);
    resume = relaunch the exact same command. Nothing already trained is redone.

Usage (from the repo root or crystal_gnn/):
    python crystal_gnn/scripts/loco_scheduler.py

State file: crystal_gnn/.loco_scheduler_state.json
Log files:  crystal_gnn/loco_scheduler.log (scheduler + child output, appended)
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

CG = Path(__file__).resolve().parents[1]          # crystal_gnn/
ROOT = CG.parent                                   # repo root
OUT_DIR = ROOT / "materials-screening-ai" / "research" / "loco_cross_conformal"
STATE_FILE = CG / ".loco_scheduler_state.json"
LOCK_FILE = CG / ".loco_scheduler.lock"
LOG_FILE = CG / "loco_scheduler.log"

# Fold 0 is the production checkpoint (scored, never retrained); the other nine
# are trained here if their checkpoint is missing.  Trained folds come first so
# the run scores 1,2,3,9 immediately and only then pays for 4,5,6,7,8.
FOLDS = [1, 2, 3, 9, 4, 5, 6, 7, 8]
SEED = 42
MAX_ATTEMPTS_PER_FOLD = 3
RETRY_COOLDOWN_S = 30

PY = ROOT / ".venv311" / "Scripts" / "python.exe"
if not PY.exists():
    PY = Path(sys.executable)

child: subprocess.Popen | None = None
state: dict = {}


def log(msg: str) -> None:
    line = f"[loco_sched {time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def ckpt_path(fold: int) -> Path:
    return CG / "checkpoints" / f"loco_A7_soap_cluster{fold}_seed{SEED}" / "best.pt"


def marker_path(fold: int) -> Path:
    return ckpt_path(fold).parent / "TRAIN_DONE"


def npz_path(fold: int) -> Path:
    return OUT_DIR / f"fold{fold}_scores.npz"


def trained(fold: int) -> bool:
    """Trained iff a clean-finish marker exists, or the fold was already scored
    by an earlier (pre-marker) run.

    best.pt alone is NOT enough: train.py writes it after every improved epoch,
    so mid-training it exists long before the run finishes — treating that as
    "trained" is what scored fold 4 from its epoch-0 checkpoint.
    """
    if scored(fold):
        return True
    return ckpt_path(fold).exists() and marker_path(fold).exists()


def scored(fold: int) -> bool:
    return npz_path(fold).exists()


# ---------------------------------------------------------------- state / lock
def save_state() -> None:
    state["trained_folds"] = [f for f in FOLDS if trained(f)]
    state["scored_folds"] = [f for f in [0] + FOLDS if scored(f)]
    state["pending_folds"] = [f for f in FOLDS if not trained(f) or not scored(f)]
    state["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    tmp = STATE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    os.replace(tmp, STATE_FILE)


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        if os.name == "nt":
            out = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}"],
                capture_output=True, text=True, timeout=15,
            ).stdout
            return str(pid) in out
        os.kill(pid, 0)
        return True
    except Exception:
        return False


def acquire_lock() -> None:
    if LOCK_FILE.exists():
        try:
            old_pid = int(LOCK_FILE.read_text().strip() or "0")
        except ValueError:
            old_pid = 0
        if _pid_alive(old_pid) and old_pid != os.getpid():
            sys.exit(f"[loco_sched] already running as PID {old_pid}; refusing a second instance")
        log(f"stale lock from dead PID {old_pid}; taking over")
    LOCK_FILE.write_text(str(os.getpid()), encoding="utf-8")


def release_lock() -> None:
    try:
        if LOCK_FILE.exists() and LOCK_FILE.read_text().strip() == str(os.getpid()):
            LOCK_FILE.unlink()
    except Exception:
        pass


# ------------------------------------------------------------------- children
def find_orphan_trainer(fold: int) -> int | None:
    """PID of an already-running train.py for this fold, if any (adoption).

    The negative filters are load-bearing: without them the query process's own
    command line (which contains this very pattern) matches itself and the
    scheduler "adopts" a dying shell instead of launching a real trainer.
    """
    if os.name != "nt":
        return None
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match "
             f"'scripts.train.py' -and $_.CommandLine -match 'soap_loco_idx {fold} ' "
             "-and $_.CommandLine -notmatch "
             "'Where-Object|Get-CimInstance|loco_scheduler|grep' } | "
             "Select-Object -ExpandProperty ProcessId"],
            capture_output=True, text=True, timeout=30,
        ).stdout
        for tok in out.split():
            tok = tok.strip()
            if tok.isdigit() and int(tok) != os.getpid():
                return int(tok)
    except Exception:
        pass
    return None


def run_child(cmd: list[str], cwd: Path, tag: str) -> int:
    global child
    log(f"{tag}: {' '.join(str(c) for c in cmd)}")
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(CG), str(CG / "scripts")])
    with open(LOG_FILE, "a", encoding="utf-8") as fh:
        child = subprocess.Popen(cmd, cwd=str(cwd), stdout=fh, stderr=subprocess.STDOUT, env=env)
        rc = child.wait()
    child = None
    log(f"{tag}: exit={rc}")
    return rc


def train_fold(fold: int) -> None:
    """Train fold until best.pt exists. Artifact existence, not rc, decides."""
    pid = find_orphan_trainer(fold)
    if pid:
        log(f"fold {fold}: adopting already-running trainer PID {pid}")
        while not trained(fold):
            time.sleep(30)
            if not _pid_alive(pid):
                break
        return
    cmd = [
        str(PY), str(CG / "scripts" / "train.py"),
        "--ablation", "A7",
        "--split", "soap_loco",
        "--soap_loco_idx", str(fold),
        "--max_structures", "50000",
        "--max_neighbors", "24",
        "--batch_size", "8",
        "--accumulate_grad_batches", "4",
        "--max_epochs", "100",
        "--warm_up_epochs", "5",
        "--seed", str(SEED),
        "--num_workers", "2",
        "--max_hours", "3.0",
        "--auto_resume",
        "--run_id", f"loco_A7_soap_cluster{fold}_seed{SEED}",
    ]
    run_child(cmd, CG, f"train fold {fold}")


def score_fold(fold: int) -> None:
    cmd = [
        str(PY), str(CG / "scripts" / "loco_cross_conformal.py"),
        "score", "--folds", str(fold), "--seed", str(SEED),
    ]
    run_child(cmd, CG, f"score fold {fold}")


def regenerate_report() -> int:
    return run_child([str(PY), str(CG / "scripts" / "loco_report.py")],
                     CG, "loco_report")


# ----------------------------------------------------------------------- main
def handle_sig(signum, _frame) -> None:
    log(f"signal {signum}: pausing (child={'running' if child else 'idle'}); "
        "relaunch the same command to resume")
    if child is not None and child.poll() is None:
        try:
            child.terminate()
        except Exception:
            pass
    save_state()
    release_lock()
    sys.exit(128 + signum)


def main() -> None:
    global state
    signal.signal(signal.SIGINT, handle_sig)
    signal.signal(signal.SIGTERM, handle_sig)
    if hasattr(signal, "SIGBREAK"):
        try:
            signal.signal(signal.SIGBREAK, handle_sig)
        except Exception:
            pass

    acquire_lock()
    state = {"pid": os.getpid(), "attempts": {}, "failed": [],
             "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    log(f"start pid={os.getpid()} py={PY}")

    # Loop until every fold is scored; failed folds are retried in later passes.
    while True:
        done = True
        for fold in FOLDS:
            if not trained(fold):
                done = False
                attempts = state["attempts"].get(str(fold), 0)
                if attempts >= MAX_ATTEMPTS_PER_FOLD:
                    if fold not in state["failed"]:
                        state["failed"].append(fold)
                        log(f"fold {fold}: giving up after {attempts} attempts")
                    continue
                state["attempts"][str(fold)] = attempts + 1
                save_state()
                log(f"fold {fold}: train attempt {attempts + 1}/{MAX_ATTEMPTS_PER_FOLD}")
                train_fold(fold)
                if trained(fold):
                    log(f"fold {fold}: trained "
                        f"({ckpt_path(fold).stat().st_size} bytes, marker ok)")
                else:
                    log(f"fold {fold}: train failed; cooling down {RETRY_COOLDOWN_S}s")
                    time.sleep(RETRY_COOLDOWN_S)
                save_state()
                break  # re-evaluate all folds from the top (state on disk wins)

            if not scored(fold):
                done = False
                save_state()
                score_fold(fold)
                if scored(fold):
                    log(f"fold {fold}: scored ({npz_path(fold).stat().st_size} bytes)")
                else:
                    log(f"fold {fold}: scoring produced no npz; will retry")
                save_state()
                break

        if done:
            break
        if state["failed"] and all(
            f in state["failed"] for f in FOLDS if not trained(f)
        ):
            log(f"aborting: folds permanently failed: {state['failed']}")
            save_state()
            release_lock()
            sys.exit(1)

    log("all folds scored; regenerating report")
    rc = regenerate_report()
    save_state()
    release_lock()
    log(f"scheduler finished report_rc={rc}")
    sys.exit(rc)


if __name__ == "__main__":
    main()
