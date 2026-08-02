"""Evaluation metrics for regression, uncertainty and OOD detection."""

from __future__ import annotations

import warnings

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score, roc_curve


def _validate_array(name: str, arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr, dtype=float).reshape(-1)
    if a.size == 0:
        raise ValueError(f"{name}: empty input")
    if np.isnan(a).any() or np.isinf(a).any():
        raise ValueError(f"{name}: NaN/Inf input with shape {a.shape}")
    return a


def compute_regression_metrics(y_true, y_pred) -> dict[str, float]:
    """Return MAE/RMSE/R2/MAPE for regression outputs."""
    y_t = _validate_array("compute_regression_metrics.y_true", y_true)
    y_p = _validate_array("compute_regression_metrics.y_pred", y_pred)
    if y_t.size != y_p.size:
        raise ValueError("compute_regression_metrics: shape mismatch")
    if y_t.size == 1:
        warnings.warn("Single sample metrics are not reliable.", UserWarning)
        return {"mae": float("nan"), "rmse": float("nan"), "r2": float("nan"), "mape": float("nan")}

    mae = float(mean_absolute_error(y_t, y_p))
    rmse = float(np.sqrt(mean_squared_error(y_t, y_p)))
    r2 = float("nan") if np.allclose(y_t, y_t[0]) else float(r2_score(y_t, y_p))
    denom = np.where(np.abs(y_t) < 1e-8, 1e-8, np.abs(y_t))
    mape = float(np.mean(np.abs((y_t - y_p) / denom)))
    return {"mae": mae, "rmse": rmse, "r2": r2, "mape": mape}


def compute_coverage(intervals_lower, intervals_upper, y_true) -> dict[str, float]:
    """Return interval coverage and width statistics."""
    lower = _validate_array("compute_coverage.lower", intervals_lower)
    upper = _validate_array("compute_coverage.upper", intervals_upper)
    y_t = _validate_array("compute_coverage.y_true", y_true)
    if not (lower.size == upper.size == y_t.size):
        raise ValueError("compute_coverage: shape mismatch")
    covered = (y_t >= lower) & (y_t <= upper)
    widths = upper - lower
    return {
        "coverage": float(np.mean(covered)),
        "mean_width": float(np.mean(widths)),
        "median_width": float(np.median(widths)),
    }


def compute_ece(y_true, y_pred, uncertainties, n_bins: int = 15) -> dict[str, float | list[float]]:
    """Compute regression ECE by comparing binned normalized uncertainty vs error."""
    y_t = _validate_array("compute_ece.y_true", y_true)
    y_p = _validate_array("compute_ece.y_pred", y_pred)
    u = _validate_array("compute_ece.uncertainties", uncertainties)
    if not (y_t.size == y_p.size == u.size):
        raise ValueError("compute_ece: shape mismatch")
    if y_t.size == 1:
        warnings.warn("Single sample ECE undefined.", UserWarning)
        return {"ece": float("nan"), "bin_coverages": [], "bin_widths": []}

    errors = np.abs(y_t - y_p)
    if np.allclose(errors, 0.0):
        return {"ece": 0.0, "bin_coverages": [0.0] * n_bins, "bin_widths": [0.0] * n_bins}

    # Normalize to [0, 1] scale so ECE captures relative miscalibration.
    u_norm = u / (np.max(u) + 1e-12)
    e_norm = errors / (np.max(errors) + 1e-12)

    bins = np.quantile(u_norm, np.linspace(0.0, 1.0, n_bins + 1))
    bins[0] -= 1e-12

    bin_coverages = []
    bin_widths = []
    ece = 0.0
    for i in range(n_bins):
        mask = (u_norm > bins[i]) & (u_norm <= bins[i + 1])
        if not np.any(mask):
            bin_coverages.append(0.0)
            bin_widths.append(0.0)
            continue
        avg_u = float(np.mean(u_norm[mask]))
        avg_e = float(np.mean(e_norm[mask]))
        w = float(np.mean(mask))
        ece += w * abs(avg_u - avg_e)
        bin_coverages.append(avg_e)
        bin_widths.append(avg_u)

    return {"ece": float(ece), "bin_coverages": bin_coverages, "bin_widths": bin_widths}


def compute_ood_auroc(uncertainties_id, uncertainties_ood) -> dict[str, float]:
    """Compute AUROC separating ID/OOD by uncertainty score."""
    u_id = _validate_array("compute_ood_auroc.uncertainties_id", uncertainties_id)
    u_ood = _validate_array("compute_ood_auroc.uncertainties_ood", uncertainties_ood)
    labels = np.concatenate([np.zeros_like(u_id), np.ones_like(u_ood)])
    scores = np.concatenate([u_id, u_ood])
    if labels.size == 1:
        warnings.warn("Single sample AUROC undefined.", UserWarning)
        return {"auroc": float("nan"), "threshold": float("nan")}
    auc = float(roc_auc_score(labels, scores))
    fpr, tpr, thresholds = roc_curve(labels, scores)
    j = np.argmax(tpr - fpr)
    thr = float(thresholds[j])
    return {"auroc": auc, "threshold": thr}


def compute_spearman_uq_error(uncertainties, abs_errors) -> dict[str, float]:
    """Spearman correlation between uncertainty and absolute error."""
    u = _validate_array("compute_spearman_uq_error.uncertainties", uncertainties)
    e = _validate_array("compute_spearman_uq_error.abs_errors", abs_errors)
    if u.size != e.size:
        raise ValueError("compute_spearman_uq_error: shape mismatch")
    if u.size == 1:
        warnings.warn("Single sample Spearman undefined.", UserWarning)
        return {"spearman_rho": float("nan"), "p_value": float("nan")}
    rho, p = spearmanr(u, e)
    return {"spearman_rho": float(rho), "p_value": float(p)}


def compute_all_metrics(
    y_true,
    y_pred,
    uncertainties,
    intervals_lower,
    intervals_upper,
    uncertainties_ood=None,
) -> dict[str, float | list[float]]:
    """Compute complete evaluation metric suite."""
    reg = compute_regression_metrics(y_true, y_pred)
    cov = compute_coverage(intervals_lower, intervals_upper, y_true)
    ece = compute_ece(y_true, y_pred, uncertainties)
    abs_err = np.abs(np.asarray(y_true) - np.asarray(y_pred))
    spr = compute_spearman_uq_error(uncertainties, abs_err)

    out: dict[str, float | list[float]] = {}
    out.update(reg)
    out.update(cov)
    out.update(ece)
    out.update(spr)

    if uncertainties_ood is not None:
        out.update(compute_ood_auroc(uncertainties, uncertainties_ood))
    return out
