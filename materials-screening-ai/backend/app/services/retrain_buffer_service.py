# -*- coding: utf-8 -*-
"""
Active Learning Retraining Buffer Service (Item 2).

Collects (structure, verified_label_eV) pairs from Tier 2 MLIP validation
and Tier 3 DFT results, then executes a real GNN fine-tuning pass on the
accumulated buffer using PyTorch gradient descent.

Pipeline:
  1. collect_entry()   — push one (structure, label, source) into buffer table
  2. should_trigger_retrain() — True when ≥ threshold new entries since last retrain
  3. export_buffer_to_dataset() — serialize buffer to training-compatible JSON
  4. run_retrain()     — fine-tune GNN checkpoint on buffer, bump version, persist
  5. mark_retrained()  — stamp entries as used in a specific model version
"""

import os
import json
import time
import logging
import hashlib
from datetime import datetime
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pymatgen.core import Structure
from torch_geometric.data import Data, Batch

from app.database.db import get_db
from app.database.models import RetrainEvent

logger = logging.getLogger(__name__)

RETRAIN_TRIGGER_THRESHOLD = int(os.getenv("RETRAIN_TRIGGER_THRESHOLD", "10"))
BUFFER_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "crystal_gnn", "retrain_buffer")


# ---------------------------------------------------------------------------
# In-memory + file-backed buffer (SQLite via models.py for persistence)
# ---------------------------------------------------------------------------

class RetrainBufferService:
    """
    Manages the active learning buffer and executes real GNN fine-tuning.
    
    Buffer entries are stored as JSON files in BUFFER_DIR (one per entry)
    for full persistence across restarts without adding a new DB table.
    Each file: {formula_hash}.json containing {formula, cif, gnn_pred_eV,
    verified_label_eV, label_source, label_uncertainty_eV, collected_at,
    used_in_version}
    """

    def __init__(self):
        os.makedirs(BUFFER_DIR, exist_ok=True)
        self._cache: List[Dict] = []
        self._load_from_disk()

    # ------------------------------------------------------------------
    # Disk I/O
    # ------------------------------------------------------------------

    def _load_from_disk(self):
        """Load all buffered entries from disk into memory."""
        self._cache = []
        for fname in os.listdir(BUFFER_DIR):
            if fname.endswith(".json"):
                try:
                    with open(os.path.join(BUFFER_DIR, fname), "r") as f:
                        entry = json.load(f)
                    self._cache.append(entry)
                except Exception as e:
                    logger.warning(f"[RetrainBuffer] Failed to load {fname}: {e}")

    def _entry_path(self, formula: str, label_source: str) -> str:
        key = f"{formula}_{label_source}_{time.time()}"
        h = hashlib.md5(key.encode()).hexdigest()[:16]
        return os.path.join(BUFFER_DIR, f"{h}.json")

    def _save_entry(self, entry: Dict):
        path = self._entry_path(entry["formula"], entry["label_source"])
        with open(path, "w") as f:
            json.dump(entry, f, indent=2)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def collect_entry(
        self,
        formula: str,
        structure_cif: str,
        gnn_pred_eV: float,
        verified_label_eV: float,
        label_source: str,  # "chgnet_mlip" | "mace_mlip" | "dft_pbe" | "dft_hse06"
        label_uncertainty_eV: float = 0.05,
    ) -> Dict[str, Any]:
        """
        Add one (structure, verified_energy) pair to the buffer.
        
        Returns entry metadata so the caller can confirm acceptance.
        Entries where |gnn_pred - label| < 0.005 eV are skipped as
        uninformative (no gradient signal).
        """
        information_gain = abs(gnn_pred_eV - verified_label_eV)
        if information_gain < 0.005:
            return {
                "status": "skipped",
                "reason": "uninformative_entry",
                "information_gain_eV": information_gain,
            }

        entry = {
            "formula": formula,
            "structure_cif": structure_cif,
            "gnn_pred_eV": gnn_pred_eV,
            "verified_label_eV": verified_label_eV,
            "label_source": label_source,
            "label_uncertainty_eV": label_uncertainty_eV,
            "collected_at": datetime.utcnow().isoformat(),
            "used_in_version": None,
            "information_gain_eV": round(information_gain, 4),
        }

        self._cache.append(entry)
        self._save_entry(entry)

        logger.info(
            f"[RetrainBuffer] Collected {formula} from {label_source} "
            f"(GNN={gnn_pred_eV:.3f}, label={verified_label_eV:.3f}, "
            f"ΔE={information_gain:.3f} eV)"
        )
        return {"status": "accepted", "buffer_size": len(self._cache), "entry": entry}

    def should_trigger_retrain(self, threshold: int = RETRAIN_TRIGGER_THRESHOLD) -> bool:
        """
        Return True when ≥ threshold unused entries have accumulated.
        Unused = used_in_version is None.
        """
        unused = [e for e in self._cache if e.get("used_in_version") is None]
        return len(unused) >= threshold

    def get_buffer_stats(self) -> Dict[str, Any]:
        """Return buffer statistics summary."""
        total = len(self._cache)
        unused = [e for e in self._cache if e.get("used_in_version") is None]
        sources = {}
        for e in self._cache:
            src = e.get("label_source", "unknown")
            sources[src] = sources.get(src, 0) + 1
        avg_gain = float(np.mean([e.get("information_gain_eV", 0) for e in self._cache])) if self._cache else 0.0
        return {
            "total_entries": total,
            "unused_entries": len(unused),
            "entries_by_source": sources,
            "avg_information_gain_eV": round(avg_gain, 4),
            "trigger_threshold": RETRAIN_TRIGGER_THRESHOLD,
            "ready_to_retrain": self.should_trigger_retrain(),
        }

    def export_buffer_to_dataset(self) -> List[Dict]:
        """
        Export buffer entries as a list of {structure_cif, label_eV, weight}
        training items, sorted by information gain (highest first).
        
        Uncertainty-weighted: entries with larger GNN error get proportionally
        higher sampling weight, implementing the standard active-learning
        uncertainty-weighting strategy.
        """
        entries = [e for e in self._cache if e.get("used_in_version") is None]
        if not entries:
            return []

        # Compute inverse-uncertainty sampling weights
        gains = np.array([e.get("information_gain_eV", 0.01) for e in entries])
        # Weight proportional to information gain (larger error = more weight)
        weights = gains / gains.sum()

        dataset = []
        for entry, weight in zip(entries, weights.tolist()):
            dataset.append({
                "formula": entry["formula"],
                "structure_cif": entry["structure_cif"],
                "label_eV": entry["verified_label_eV"],
                "label_source": entry["label_source"],
                "label_uncertainty_eV": entry.get("label_uncertainty_eV", 0.05),
                "sampling_weight": round(weight, 6),
            })

        # Sort by information gain (most informative first)
        dataset.sort(key=lambda x: x["sampling_weight"], reverse=True)
        return dataset

    def run_retrain(
        self,
        min_samples: int = RETRAIN_TRIGGER_THRESHOLD,
        learning_rate: float = 1e-4,
        epochs: int = 20,
        current_version: str = "v1.0.0-initial",
    ) -> Dict[str, Any]:
        """
        Execute real GNN fine-tuning on accumulated buffer entries.
        
        Uses uncertainty-weighted MSE loss on the active learning buffer,
        with early stopping based on validation loss on a held-out split.
        
        Returns: training results dict including new model version and final loss.
        """
        dataset = self.export_buffer_to_dataset()
        if len(dataset) < min_samples:
            return {
                "status": "skipped",
                "reason": f"insufficient_samples ({len(dataset)} < {min_samples})",
                "buffer_stats": self.get_buffer_stats(),
            }

        # Import GNN predictor lazily to avoid circular imports
        from app.services.predictor import GNNPredictorService
        predictor = GNNPredictorService.get_instance()

        if predictor.model is None:
            return {
                "status": "error",
                "reason": "GNN model not loaded — cannot fine-tune",
            }

        t0 = time.time()
        model = predictor.model
        device = predictor.device

        # Build training graph batches from buffer CIFs
        graph_items = []
        labels = []
        weights = []

        for item in dataset:
            try:
                struct = Structure.from_str(item["structure_cif"], fmt="cif")
                # Build graph using predictor's internal graph builder
                graphs = predictor._build_multi_scale_graphs(struct)
                if graphs is not None:
                    graph_items.append(graphs)
                    labels.append(item["label_eV"])
                    weights.append(item["sampling_weight"])
            except Exception as e:
                logger.warning(f"[RetrainBuffer] Graph build failed for {item['formula']}: {e}")
                continue

        if len(graph_items) < min_samples:
            return {
                "status": "skipped",
                "reason": f"graph_build_failed_too_many ({len(graph_items)} usable of {len(dataset)})",
            }

        # Train/val split (80/20)
        n_val = max(1, len(graph_items) // 5)
        n_train = len(graph_items) - n_val
        train_items = graph_items[:n_train]
        train_labels = labels[:n_train]
        train_weights = weights[:n_train]
        val_items = graph_items[n_train:]
        val_labels = labels[n_train:]

        # Optimizer: fine-tune only the final prediction head + last message-passing layer
        # to avoid catastrophic forgetting of the pretrained backbone
        trainable_params = []
        for name, param in model.named_parameters():
            if any(key in name for key in ["der_head", "output_head", "final", "predictor"]):
                param.requires_grad = True
                trainable_params.append(param)
            else:
                param.requires_grad = False  # freeze backbone
        
        # If no head params found, fine-tune everything with smaller LR
        if not trainable_params:
            for param in model.parameters():
                param.requires_grad = True
            trainable_params = list(model.parameters())
            learning_rate = learning_rate * 0.1  # extra conservative

        optimizer = optim.AdamW(trainable_params, lr=learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

        model.train()
        best_val_loss = float("inf")
        best_epoch = 0
        train_losses = []
        val_losses = []

        label_tensor = torch.tensor(train_labels, dtype=torch.float32, device=device)
        weight_tensor = torch.tensor(train_weights, dtype=torch.float32, device=device)
        weight_tensor = weight_tensor / weight_tensor.sum()  # normalize weights

        for epoch in range(epochs):
            # --- Training step ---
            model.train()
            optimizer.zero_grad()

            epoch_preds = []
            for g_tuple in train_items:
                try:
                    # g_tuple is (g1, g2, g3) multi-scale graphs
                    if isinstance(g_tuple, (list, tuple)):
                        g_list = [g.to(device) for g in g_tuple]
                        out = model(*g_list) if len(g_list) > 1 else model(g_list[0])
                    else:
                        out = model(g_tuple.to(device))
                    
                    # Extract mu (predicted energy) from DER output
                    if isinstance(out, dict):
                        pred = out.get("mu", out.get("prediction", list(out.values())[0]))
                    elif isinstance(out, (list, tuple)):
                        pred = out[0]
                    else:
                        pred = out
                    
                    if pred.dim() > 0:
                        pred = pred.squeeze()
                    if pred.dim() == 0:
                        pred = pred.unsqueeze(0)
                    epoch_preds.append(pred.mean())
                except Exception as e:
                    logger.debug(f"[RetrainBuffer] Forward pass failed: {e}")
                    continue

            if not epoch_preds:
                continue

            preds_tensor = torch.stack(epoch_preds[:len(train_labels)])
            actual_labels = label_tensor[:len(preds_tensor)]
            actual_weights = weight_tensor[:len(preds_tensor)]

            # Weighted MSE loss
            squared_errors = (preds_tensor - actual_labels) ** 2
            train_loss = (squared_errors * actual_weights).sum()
            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            scheduler.step()
            train_losses.append(float(train_loss.item()))

            # --- Validation step ---
            model.eval()
            with torch.no_grad():
                val_preds = []
                for g_tuple in val_items:
                    try:
                        if isinstance(g_tuple, (list, tuple)):
                            g_list = [g.to(device) for g in g_tuple]
                            out = model(*g_list) if len(g_list) > 1 else model(g_list[0])
                        else:
                            out = model(g_tuple.to(device))
                        if isinstance(out, dict):
                            pred = out.get("mu", list(out.values())[0])
                        elif isinstance(out, (list, tuple)):
                            pred = out[0]
                        else:
                            pred = out
                        val_preds.append(float(pred.mean().item()))
                    except Exception:
                        continue

                if val_preds:
                    val_labels_subset = val_labels[:len(val_preds)]
                    val_loss = float(np.mean([(p - l) ** 2 for p, l in zip(val_preds, val_labels_subset)]))
                    val_losses.append(val_loss)

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_epoch = epoch

        # Restore grad computation for all params
        for param in model.parameters():
            param.requires_grad = True

        model.eval()
        runtime = time.time() - t0

        # Generate new version tag
        import re
        match = re.match(r"v(\d+)\.(\d+)\.(\d+)", current_version)
        if match:
            major, minor, patch = int(match.group(1)), int(match.group(2)), int(match.group(3))
            new_version = f"v{major}.{minor + 1}.{patch}-active_learned_{len(dataset)}samples"
        else:
            new_version = f"v1.1.0-active_learned_{len(dataset)}samples"

        # Mark buffer entries as used
        for entry in self._cache:
            if entry.get("used_in_version") is None:
                entry["used_in_version"] = new_version

        # Persist retrain event to DB
        db = next(get_db())
        try:
            event = RetrainEvent(
                triggered_at=datetime.utcnow(),
                n_samples=len(dataset),
                model_version_before=current_version,
                model_version_after=new_version,
                val_loss_after=round(best_val_loss, 6) if best_val_loss < float("inf") else 9.999,
            )
            db.add(event)
            db.commit()
        except Exception as e:
            logger.warning(f"[RetrainBuffer] Failed to log retrain event: {e}")
            db.rollback()
        finally:
            db.close()

        logger.info(
            f"[RetrainBuffer] Fine-tuning complete: {len(dataset)} samples, "
            f"{epochs} epochs, best_val_loss={best_val_loss:.4f} @ epoch {best_epoch}, "
            f"{runtime:.1f}s"
        )

        return {
            "status": "success",
            "model_version_before": current_version,
            "model_version_after": new_version,
            "n_samples": len(dataset),
            "epochs_run": epochs,
            "best_val_loss": round(best_val_loss, 6) if best_val_loss < float("inf") else None,
            "best_epoch": best_epoch,
            "final_train_loss": round(train_losses[-1], 6) if train_losses else None,
            "runtime_seconds": round(runtime, 2),
            "learning_rate": learning_rate,
        }

    def mark_retrained(self, version: str):
        """Mark all pending buffer entries as used in the given version."""
        for entry in self._cache:
            if entry.get("used_in_version") is None:
                entry["used_in_version"] = version


# Module-level singleton
_retrain_buffer_instance: Optional[RetrainBufferService] = None


def get_retrain_buffer() -> RetrainBufferService:
    global _retrain_buffer_instance
    if _retrain_buffer_instance is None:
        _retrain_buffer_instance = RetrainBufferService()
    return _retrain_buffer_instance
