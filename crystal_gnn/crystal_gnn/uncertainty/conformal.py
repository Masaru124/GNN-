"""Split conformal prediction for uncertainty-calibrated intervals."""

from __future__ import annotations

import json
import math
import warnings
from pathlib import Path

import numpy as np


class SplitConformalPredictor:
    """Calibrate and evaluate split-conformal prediction intervals."""

    def __init__(self, target_coverage: float = 0.90) -> None:
        if target_coverage >= 1.0:
            raise ValueError("target_coverage must be < 1.0")
        self.target_coverage = float(target_coverage)
        self.q_hat: float | None = None
        self.calibration_size: int = 0
        self.achieved_coverage: float | None = None

    def calibrate(self, predictions: np.ndarray, uncertainties: np.ndarray, true_labels: np.ndarray) -> float:
        """Fit nonconformity quantile on calibration data."""
        pred = np.asarray(predictions, dtype=float).reshape(-1)
        unc = np.asarray(uncertainties, dtype=float).reshape(-1)
        y = np.asarray(true_labels, dtype=float).reshape(-1)

        if pred.size != unc.size or pred.size != y.size:
            raise ValueError("predictions, uncertainties, true_labels must have the same length")
        if pred.size < 50:
            warnings.warn("Calibration set has <50 samples; quantile estimate may be unreliable.", UserWarning)

        if np.allclose(unc, 0.0):
            warnings.warn("All sigma_hat are zero; using absolute residual score fallback.", UserWarning)
            scores = np.abs(y - pred)
            sigma = np.ones_like(scores)
        else:
            sigma = unc + 1e-8
            scores = np.abs(y - pred) / sigma

        n = len(scores)
        q_level = math.ceil((n + 1) * self.target_coverage) / n
        q_level = min(max(q_level, 0.0), 1.0)
        self.q_hat = float(np.quantile(scores, q_level, method="higher"))
        self.calibration_size = int(n)

        intervals = self.predict(pred, unc)
        self.achieved_coverage = float(np.mean((y >= intervals["lower"]) & (y <= intervals["upper"])))
        return self.q_hat

    def predict(self, predictions: np.ndarray, uncertainties: np.ndarray) -> dict[str, np.ndarray | float]:
        """Return conformal intervals for supplied predictions/uncertainties."""
        if self.q_hat is None:
            raise RuntimeError("Predictor must be calibrated before predict().")
        pred = np.asarray(predictions, dtype=float).reshape(-1)
        unc = np.asarray(uncertainties, dtype=float).reshape(-1)
        lower = pred - self.q_hat * unc
        upper = pred + self.q_hat * unc
        width = 2.0 * self.q_hat * unc
        return {"lower": lower, "upper": upper, "width": width, "q_hat": self.q_hat}

    def evaluate_coverage(self, intervals: dict[str, np.ndarray | float], true_labels: np.ndarray) -> dict[str, float | bool]:
        """Compute empirical coverage and interval statistics."""
        lower = np.asarray(intervals["lower"], dtype=float)
        upper = np.asarray(intervals["upper"], dtype=float)
        y = np.asarray(true_labels, dtype=float).reshape(-1)
        covered = (y >= lower) & (y <= upper)
        empirical = float(np.mean(covered))
        gap = empirical - self.target_coverage
        width = upper - lower
        return {
            "empirical_coverage": empirical,
            "target_coverage": float(self.target_coverage),
            "coverage_gap": float(gap),
            "mean_interval_width": float(np.mean(width)),
            "median_interval_width": float(np.median(width)),
            "coverage_passed": bool(abs(gap) < 0.02),
        }

    def save(self, path: str) -> None:
        """Save calibrated predictor state to JSON."""
        if self.q_hat is None:
            raise RuntimeError("Cannot save uncalibrated predictor")
        out = {
            "target_coverage": self.target_coverage,
            "q_hat": self.q_hat,
            "calibration_size": self.calibration_size,
            "achieved_coverage": self.achieved_coverage,
        }
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str) -> "SplitConformalPredictor":
        """Load predictor state from JSON."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        inst = cls(target_coverage=float(payload["target_coverage"]))
        inst.q_hat = float(payload["q_hat"])
        inst.calibration_size = int(payload.get("calibration_size", 0))
        inst.achieved_coverage = payload.get("achieved_coverage", None)
        return inst


class LocallyAdaptiveConformalPredictor:
    """Calibrate and evaluate locally adaptive split-conformal prediction intervals by uncertainty quartiles."""

    def __init__(self, target_coverage: float = 0.90) -> None:
        if target_coverage >= 1.0:
            raise ValueError("target_coverage must be < 1.0")
        self.target_coverage = float(target_coverage)
        self.q_hats: list[float] | None = None
        self.thresholds: list[float] | None = None  # q25, q50, q75
        self.calibration_size: int = 0
        self.achieved_coverage: float | None = None

    def calibrate(self, predictions: np.ndarray, uncertainties: np.ndarray, true_labels: np.ndarray) -> list[float]:
        """Fit nonconformity quantiles on calibration data grouped by uncertainty quartiles."""
        pred = np.asarray(predictions, dtype=float).reshape(-1)
        unc = np.asarray(uncertainties, dtype=float).reshape(-1)
        y = np.asarray(true_labels, dtype=float).reshape(-1)

        if pred.size != unc.size or pred.size != y.size:
            raise ValueError("predictions, uncertainties, true_labels must have the same length")
        
        n = len(pred)
        self.calibration_size = n

        # Absolute residual scaled by uncertainty (standard conformal score)
        sigma = unc + 1e-8
        scores = np.abs(y - pred) / sigma

        # Split calibration uncertainties into 4 quartiles
        q25 = float(np.percentile(unc, 25))
        q50 = float(np.percentile(unc, 50))
        q75 = float(np.percentile(unc, 75))
        self.thresholds = [q25, q50, q75]

        # Define masks for each quartile in calibration data
        masks = [
            unc <= q25,
            (unc > q25) & (unc <= q50),
            (unc > q50) & (unc <= q75),
            unc > q75
        ]

        self.q_hats = []
        for i, mask in enumerate(masks):
            scores_k = scores[mask]
            n_k = len(scores_k)
            if n_k == 0:
                # Fallback if a quartile is empty
                self.q_hats.append(1.96)
                continue

            q_level = math.ceil((n_k + 1) * self.target_coverage) / n_k
            q_level = min(max(q_level, 0.0), 1.0)
            q_hat_k = float(np.quantile(scores_k, q_level, method="higher"))
            self.q_hats.append(q_hat_k)

        # Compute achieved overall coverage on calibration data
        intervals = self.predict(pred, unc)
        self.achieved_coverage = float(np.mean((y >= intervals["lower"]) & (y <= intervals["upper"])))
        return self.q_hats

    def predict(self, predictions: np.ndarray, uncertainties: np.ndarray) -> dict[str, np.ndarray]:
        """Return conformal intervals using local quartile q_hats."""
        if self.q_hats is None or self.thresholds is None:
            raise RuntimeError("Predictor must be calibrated before predict().")
        pred = np.asarray(predictions, dtype=float).reshape(-1)
        unc = np.asarray(uncertainties, dtype=float).reshape(-1)

        q25, q50, q75 = self.thresholds
        
        # Build masks for prediction data based on calibration thresholds
        masks = [
            unc <= q25,
            (unc > q25) & (unc <= q50),
            (unc > q50) & (unc <= q75),
            unc > q75
        ]

        lower = np.zeros_like(pred)
        upper = np.zeros_like(pred)
        width = np.zeros_like(pred)
        q_hat_assigned = np.zeros_like(pred)

        for k, mask in enumerate(masks):
            if np.any(mask):
                q_hat_k = self.q_hats[k]
                lower[mask] = pred[mask] - q_hat_k * unc[mask]
                upper[mask] = pred[mask] + q_hat_k * unc[mask]
                width[mask] = 2.0 * q_hat_k * unc[mask]
                q_hat_assigned[mask] = q_hat_k

        return {
            "lower": lower,
            "upper": upper,
            "width": width,
            "q_hat": q_hat_assigned
        }

    def evaluate_coverage(self, intervals: dict[str, np.ndarray], true_labels: np.ndarray) -> dict[str, float | bool]:
        """Compute empirical coverage and interval statistics."""
        lower = np.asarray(intervals["lower"], dtype=float)
        upper = np.asarray(intervals["upper"], dtype=float)
        y = np.asarray(true_labels, dtype=float).reshape(-1)
        covered = (y >= lower) & (y <= upper)
        empirical = float(np.mean(covered))
        gap = empirical - self.target_coverage
        width = upper - lower
        return {
            "empirical_coverage": empirical,
            "target_coverage": float(self.target_coverage),
            "coverage_gap": float(gap),
            "mean_interval_width": float(np.mean(width)),
            "median_interval_width": float(np.median(width)),
            "coverage_passed": bool(abs(gap) < 0.05),
        }

    def save(self, path: str) -> None:
        """Save calibrated predictor state to JSON."""
        if self.q_hats is None:
            raise RuntimeError("Cannot save uncalibrated predictor")
        out = {
            "target_coverage": self.target_coverage,
            "q_hats": self.q_hats,
            "thresholds": self.thresholds,
            "calibration_size": self.calibration_size,
            "achieved_coverage": self.achieved_coverage,
        }
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str) -> "LocallyAdaptiveConformalPredictor":
        """Load predictor state from JSON."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        inst = cls(target_coverage=float(payload["target_coverage"]))
        inst.q_hats = [float(q) for q in payload["q_hats"]]
        inst.thresholds = [float(t) for t in payload["thresholds"]]
        inst.calibration_size = int(payload.get("calibration_size", 0))
        inst.achieved_coverage = payload.get("achieved_coverage", None)
        return inst
