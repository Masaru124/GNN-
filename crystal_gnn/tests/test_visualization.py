from __future__ import annotations

import numpy as np

from crystal_gnn.evaluation.visualization import plot_attention_heatmap, plot_reliability_diagram, plot_uncertainty_error_scatter


def test_reliability_diagram(tmp_path):
    y = np.array([1.0, 2.0, 3.0])
    p = np.array([1.1, 1.9, 3.2])
    u = np.array([0.1, 0.2, 0.3])
    out = tmp_path / "rel.png"
    plot_reliability_diagram(y, p, u, str(out))
    assert out.exists()


def test_attention_heatmap(tmp_path):
    att = np.random.default_rng(42).random((4, 3))
    out = tmp_path / "att.png"
    plot_attention_heatmap(att, str(out))
    assert out.exists()


def test_uncertainty_error_scatter(tmp_path):
    u = np.random.default_rng(1).random(50)
    e = np.random.default_rng(2).random(50)
    out = tmp_path / "scat.png"
    plot_uncertainty_error_scatter(u, e, str(out))
    assert out.exists()
