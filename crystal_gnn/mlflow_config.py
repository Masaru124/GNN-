# -*- coding: utf-8 -*-
"""
MLflow Experiment Tracking Configuration (Item 7).

Central configuration for all MatScreen AI / CrystalGNN experiment tracking.
Import this module in any training script to get a consistent experiment
context, run naming, and parameter logging interface.

Usage in training scripts:
    from crystal_gnn.mlflow_config import mlflow_context, log_training_params

    with mlflow_context(run_name="ablation_A7_no_der"):
        log_training_params(lr=1e-3, epochs=100, model_arch="MultiScaleGNN_A7")
        for epoch in range(epochs):
            ...
            mlflow.log_metrics({"val_mae": val_mae, "train_loss": loss}, step=epoch)
        mlflow.log_artifact(ckpt_path, artifact_path="checkpoints")
"""

import os
import contextlib
import logging
from typing import Any, Dict, Optional

try:
    import mlflow
    import mlflow.pytorch
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "sqlite:///" + os.path.join(os.path.dirname(__file__), "..", "crystal_gnn", "mlflow.db")
)
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT", "CrystalGNN-MatScreen")

# Standard tags applied to every run
STANDARD_TAGS = {
    "project": "MatScreen-AI",
    "pipeline": "GNN+DER+Conformal",
    "dataset": "Materials-Project-50k",
}


def setup_mlflow() -> bool:
    """
    Initialize MLflow tracking URI and experiment.
    Returns True if MLflow is available, False if not installed.
    """
    if not HAS_MLFLOW:
        logger.warning("[MLflow] mlflow not installed. Run: pip install mlflow")
        return False

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    try:
        experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
        if experiment is None:
            experiment_id = mlflow.create_experiment(
                EXPERIMENT_NAME,
                tags=STANDARD_TAGS,
            )
            logger.info(f"[MLflow] Created experiment '{EXPERIMENT_NAME}' (id={experiment_id})")
        else:
            mlflow.set_experiment(EXPERIMENT_NAME)
            logger.info(f"[MLflow] Using experiment '{EXPERIMENT_NAME}'")
    except Exception as e:
        logger.warning(f"[MLflow] Experiment setup failed: {e}")
        return False

    return True


@contextlib.contextmanager
def mlflow_context(
    run_name: Optional[str] = None,
    tags: Optional[Dict[str, str]] = None,
    nested: bool = False,
):
    """
    Context manager for an MLflow run. Safe to use even if mlflow is not installed
    (degrades gracefully to a no-op context).

    Args:
        run_name: Human-readable run name (e.g. "ablation_A7_no_DER")
        tags: Additional tags to set on the run
        nested: If True, creates a nested run inside an active parent run

    Yields:
        The active MLflow run (or None if mlflow not available)
    """
    if not HAS_MLFLOW:
        logger.warning("[MLflow] mlflow not installed — tracking disabled for this run")
        yield None
        return

    setup_mlflow()
    all_tags = dict(STANDARD_TAGS)
    if tags:
        all_tags.update(tags)

    with mlflow.start_run(run_name=run_name, tags=all_tags, nested=nested) as run:
        logger.info(f"[MLflow] Started run '{run_name}' (id={run.info.run_id})")
        yield run
        logger.info(f"[MLflow] Finished run '{run_name}'")


def log_training_params(
    lr: float,
    epochs: int,
    model_arch: str,
    batch_size: int = 32,
    optimizer: str = "AdamW",
    scheduler: str = "CosineAnnealing",
    n_train: Optional[int] = None,
    n_val: Optional[int] = None,
    extra_params: Optional[Dict[str, Any]] = None,
):
    """Log standard training hyperparameters to the active MLflow run."""
    if not HAS_MLFLOW:
        return
    params = {
        "lr": lr,
        "epochs": epochs,
        "model_arch": model_arch,
        "batch_size": batch_size,
        "optimizer": optimizer,
        "scheduler": scheduler,
    }
    if n_train is not None:
        params["n_train"] = n_train
    if n_val is not None:
        params["n_val"] = n_val
    if extra_params:
        params.update(extra_params)
    try:
        mlflow.log_params(params)
    except Exception as e:
        logger.warning(f"[MLflow] log_params failed: {e}")


def log_epoch_metrics(
    epoch: int,
    train_loss: Optional[float] = None,
    val_mae: Optional[float] = None,
    val_coverage: Optional[float] = None,
    conformal_q_hat: Optional[float] = None,
    extra_metrics: Optional[Dict[str, float]] = None,
):
    """Log per-epoch training metrics to the active MLflow run."""
    if not HAS_MLFLOW:
        return
    metrics = {}
    if train_loss is not None:
        metrics["train_loss"] = train_loss
    if val_mae is not None:
        metrics["val_mae"] = val_mae
    if val_coverage is not None:
        metrics["val_coverage"] = val_coverage
    if conformal_q_hat is not None:
        metrics["conformal_q_hat"] = conformal_q_hat
    if extra_metrics:
        metrics.update(extra_metrics)
    try:
        mlflow.log_metrics(metrics, step=epoch)
    except Exception as e:
        logger.warning(f"[MLflow] log_metrics failed at epoch {epoch}: {e}")


def log_final_results(
    val_mae: float,
    test_mae: Optional[float] = None,
    conformal_q_hat: Optional[float] = None,
    empirical_coverage: Optional[float] = None,
    checkpoint_path: Optional[str] = None,
):
    """Log final training results and model checkpoint artifact."""
    if not HAS_MLFLOW:
        return
    final_metrics = {"final_val_mae": val_mae}
    if test_mae is not None:
        final_metrics["final_test_mae"] = test_mae
    if conformal_q_hat is not None:
        final_metrics["final_conformal_q_hat"] = conformal_q_hat
    if empirical_coverage is not None:
        final_metrics["final_empirical_coverage"] = empirical_coverage

    try:
        mlflow.log_metrics(final_metrics)
        if checkpoint_path and os.path.exists(checkpoint_path):
            mlflow.log_artifact(checkpoint_path, artifact_path="checkpoints")
            logger.info(f"[MLflow] Logged checkpoint artifact: {checkpoint_path}")
    except Exception as e:
        logger.warning(f"[MLflow] log_final_results failed: {e}")


def get_tracking_uri() -> str:
    return MLFLOW_TRACKING_URI


def get_experiment_name() -> str:
    return EXPERIMENT_NAME
