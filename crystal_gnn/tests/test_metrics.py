from __future__ import annotations

import numpy as np
import pytest

from crystal_gnn.evaluation.metrics import (
    compute_coverage,
    compute_ece,
    compute_ood_auroc,
    compute_regression_metrics,
    compute_spearman_uq_error,
)


def test_mae_correct():
    assert compute_regression_metrics([1, 2, 3], [1, 2, 3])["mae"] == 0
    assert compute_regression_metrics([0, 0], [1, 1])["mae"] == 1


def test_coverage_correct():
    c1 = compute_coverage([0, 0, 0], [2, 2, 2], [1, 1, 1])
    c2 = compute_coverage([0, 0, 0], [0.5, 0.5, 0.5], [1, 1, 1])
    assert c1["coverage"] == 1.0
    assert c2["coverage"] == 0.0


def test_ece_perfect_calibration():
    y = np.array([1, 2, 3, 4], dtype=float)
    p = y.copy()
    u = np.array([0.1, 0.1, 0.1, 0.1])
    assert compute_ece(y, p, u)["ece"] < 0.05


def test_auroc_perfect_separation():
    out = compute_ood_auroc(np.array([0.01, 0.02]), np.array([0.98, 0.99]))
    assert np.isclose(out["auroc"], 1.0)


def test_auroc_no_separation():
    rng = np.random.default_rng(42)
    id_u = rng.normal(0.5, 0.1, 100)
    ood_u = rng.normal(0.5, 0.1, 100)
    out = compute_ood_auroc(id_u, ood_u)
    assert 0.3 <= out["auroc"] <= 0.7


def test_spearman_positive_correlation():
    u = np.linspace(0, 1, 100)
    e = u + np.random.default_rng(42).normal(0, 0.05, 100)
    out = compute_spearman_uq_error(u, e)
    assert out["spearman_rho"] > 0.5


def test_nan_input_raises():
    with pytest.raises(ValueError):
        compute_regression_metrics([1, np.nan], [1, 2])


def test_empty_input_raises():
    with pytest.raises(ValueError):
        compute_regression_metrics([], [])
