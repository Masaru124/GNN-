from __future__ import annotations

import numpy as np
import pytest

from crystal_gnn.uncertainty.conformal import SplitConformalPredictor


def _make_data(n=1000, seed=42):
    rng = np.random.default_rng(seed)
    y_pred = rng.normal(0, 1, n)
    sigma = np.abs(rng.normal(0.5, 0.2, n))
    y_true = y_pred + rng.normal(0, sigma)
    return y_pred, sigma, y_true


def test_calibration_achieves_coverage():
    p, s, y = _make_data(700)
    cp = SplitConformalPredictor(target_coverage=0.9)
    cp.calibrate(p[:200], s[:200], y[:200])
    cov = cp.evaluate_coverage(cp.predict(p[200:], s[200:]), y[200:])
    assert abs(cov["empirical_coverage"] - 0.9) < 0.05


def test_coverage_90_percent():
    p, s, y = _make_data(700)
    cp = SplitConformalPredictor(0.90)
    cp.calibrate(p[:200], s[:200], y[:200])
    c = cp.evaluate_coverage(cp.predict(p[200:], s[200:]), y[200:])["empirical_coverage"]
    assert 0.87 <= c <= 0.93


def test_coverage_95_percent():
    p, s, y = _make_data(700)
    cp = SplitConformalPredictor(0.95)
    cp.calibrate(p[:200], s[:200], y[:200])
    c = cp.evaluate_coverage(cp.predict(p[200:], s[200:]), y[200:])["empirical_coverage"]
    assert 0.92 <= c <= 0.98


def test_intervals_are_finite():
    p, s, y = _make_data(100)
    cp = SplitConformalPredictor(0.9)
    cp.calibrate(p, s, y)
    ints = cp.predict(p, s)
    assert np.all(ints["lower"] < ints["upper"])
    assert np.isfinite(ints["lower"]).all() and np.isfinite(ints["upper"]).all()


def test_small_calibration_warns():
    p, s, y = _make_data(80)
    cp = SplitConformalPredictor(0.9)
    with pytest.warns(UserWarning):
        cp.calibrate(p[:30], s[:30], y[:30])


def test_zero_sigma_fallback():
    p, _, y = _make_data(200)
    cp = SplitConformalPredictor(0.9)
    with pytest.warns(UserWarning):
        cp.calibrate(p[:100], np.zeros(100), y[:100])


def test_save_load_predictor(tmp_path):
    p, s, y = _make_data(200)
    cp = SplitConformalPredictor(0.9)
    cp.calibrate(p[:100], s[:100], y[:100])
    path = tmp_path / "cp.json"
    cp.save(str(path))
    cp2 = SplitConformalPredictor.load(str(path))
    assert cp2.q_hat == cp.q_hat


def test_coverage_gap_reported():
    p, s, y = _make_data(300)
    cp = SplitConformalPredictor(0.9)
    cp.calibrate(p[:100], s[:100], y[:100])
    cov = cp.evaluate_coverage(cp.predict(p[100:], s[100:]), y[100:])
    assert np.isclose(cov["coverage_gap"], cov["empirical_coverage"] - cov["target_coverage"])
    assert isinstance(cov["coverage_passed"], bool)
