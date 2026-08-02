"""Visualization helpers for uncertainty and interpretability outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


sns.set_theme(style="whitegrid")


def plot_reliability_diagram(y_true, y_pred, uncertainties, output_path: str) -> None:
    """Plot uncertainty vs absolute error calibration diagram."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    unc = np.asarray(uncertainties, dtype=float)
    abs_err = np.abs(y_true - y_pred)

    bins = np.quantile(unc, np.linspace(0, 1, 11))
    centers = []
    avg_err = []
    avg_unc = []
    for i in range(10):
        mask = (unc >= bins[i]) & (unc <= bins[i + 1])
        if np.any(mask):
            centers.append((bins[i] + bins[i + 1]) * 0.5)
            avg_err.append(float(abs_err[mask].mean()))
            avg_unc.append(float(unc[mask].mean()))

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(avg_unc, avg_err, marker="o", label="Empirical")
    lim = max(max(avg_unc, default=1.0), max(avg_err, default=1.0))
    ax.plot([0, lim], [0, lim], linestyle="--", color="black", label="Ideal")
    ax.set_xlabel("Predicted uncertainty")
    ax.set_ylabel("Observed absolute error")
    ax.legend()

    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_attention_heatmap(attn_weights: np.ndarray, output_path: str) -> None:
    """Plot scale-attention heatmap for a batch."""
    arr = np.asarray(attn_weights, dtype=float)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(arr, cmap="mako", cbar=True, ax=ax)
    ax.set_xlabel("Scale index (r1, r2, r3)")
    ax.set_ylabel("Sample")

    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_uncertainty_error_scatter(uncertainties, abs_errors, output_path: str) -> None:
    """Scatter uncertainty against error."""
    u = np.asarray(uncertainties, dtype=float).reshape(-1)
    e = np.asarray(abs_errors, dtype=float).reshape(-1)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(u, e, alpha=0.6, s=14)
    ax.set_xlabel("Uncertainty")
    ax.set_ylabel("Absolute error")

    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
