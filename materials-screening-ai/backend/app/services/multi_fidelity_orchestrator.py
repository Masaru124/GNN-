# -*- coding: utf-8 -*-
"""
Multi-Fidelity Information-Gain Orchestrator.

Intelligently routes discovery candidates between Tier 1 GNN screening and Tier 2 MLIP Physics Validation
based on Expected Information Gain (Expected Improvement heuristic):
  - Evaluates GNN Uncertainty Interval Width (conformal 90% width, LOCO 10-fold, nested: 89.6% pooled / 92.2% macro at q_LOFO; 71.2% at shipped q; worst fold 7)
  - Evaluates proximity to target constraint boundaries
  - Returns decision policy:
      * 'promote_to_tier2': Promoted for expensive Tier 2 MLIP physics relaxation
      * 'hold_for_more_data': Retained in Tier 1 GNN screen
      * 'reject': Disqualified by hard filters (cost, toxicity)
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class MultiFidelityOrchestrator:
    """Multi-fidelity expected information-gain decision orchestrator."""

    def __init__(
        self,
        high_uncertainty_threshold: float = 0.10,
        boundary_margin: float = 0.40
    ):
        self.high_uncertainty_threshold = high_uncertainty_threshold
        self.boundary_margin = boundary_margin

    def decide_next_action(
        self,
        gnn_prediction: float,
        uncertainty_low: float,
        uncertainty_high: float,
        hard_filter_pass: bool,
        target_threshold: Optional[float] = None,
        is_solar_optimal: bool = False
    ) -> str:
        """
        Evaluate candidate expected information gain and return routing decision.

        Returns:
            decision (str): 'promote_to_tier2' | 'hold_for_more_data' | 'reject'
        """
        if not hard_filter_pass:
            return "reject"

        interval_width = float(uncertainty_high - uncertainty_low)
        target = target_threshold if target_threshold is not None else -0.20

        # Log decision inputs for full audit transparency
        logger.info(
            f"[MultiFidelityOrchestrator] E_f={gnn_prediction:.3f}, UQ_width={interval_width:.3f}, "
            f"hard_pass={hard_filter_pass}, target={target:.3f}, solar_opt={is_solar_optimal}"
        )

        # 1. Promote candidates that are Shockley-Queisser Solar Optimal or thermodynamically stable (E_f < -0.20 eV/atom)
        if is_solar_optimal or gnn_prediction < -0.20:
            return "promote_to_tier2"

        # 2. Promote candidates near decision boundary or with higher uncertainty
        if interval_width >= self.high_uncertainty_threshold or abs(gnn_prediction - target) < self.boundary_margin:
            return "promote_to_tier2"

        return "hold_for_more_data"

    def prioritize_for_dft(
        self,
        uncertainty_low: float,
        uncertainty_high: float,
        energy_disagreement_eV: Optional[float] = None,
        ensemble_status: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Compute DFT priority score combining Tier 1 UQ and Tier 2 ensemble disagreement.

        A candidate with high Tier 1 uncertainty AND high Tier 2 ensemble disagreement
        is the highest-priority DFT candidate — this extends the "spend expensive compute
        where uncertainty is highest" philosophy one tier further.

        Returns:
            dict with 'dft_priority_score' (0.0-1.0), 'dft_priority_tier' (str),
            and 'rationale' (str)
        """
        interval_width = float(uncertainty_high - uncertainty_low)

        # Normalize Tier 1 UQ signal (0-1 scale)
        uq_signal = min(interval_width / 0.5, 1.0)

        # Normalize Tier 2 ensemble disagreement signal (0-1 scale)
        if energy_disagreement_eV is not None:
            ensemble_signal = min(energy_disagreement_eV / 0.15, 1.0)
        elif ensemble_status == "requires_independent_validation":
            ensemble_signal = 1.0
        elif ensemble_status == "moderate_agreement":
            ensemble_signal = 0.4
        elif ensemble_status == "high_confidence_agreement":
            ensemble_signal = 0.0
        else:
            # No ensemble data yet — use only UQ signal
            ensemble_signal = 0.0

        # Combined score: weighted average (ensemble disagreement is the stronger signal
        # since it represents actual physics-level disagreement, not just statistical uncertainty)
        dft_priority_score = 0.35 * uq_signal + 0.65 * ensemble_signal

        # Tier classification
        if dft_priority_score >= 0.7:
            tier = "critical_dft_priority"
            rationale = "High Tier 1 uncertainty AND high ensemble disagreement — strongest DFT candidate"
        elif dft_priority_score >= 0.4:
            tier = "elevated_dft_priority"
            rationale = "Moderate uncertainty or disagreement signal — DFT recommended"
        elif dft_priority_score >= 0.15:
            tier = "low_dft_priority"
            rationale = "Some uncertainty signal but models mostly agree — DFT optional"
        else:
            tier = "no_dft_needed"
            rationale = "Low uncertainty and strong ensemble agreement — no DFT required"

        logger.info(
            f"[MultiFidelityOrchestrator] DFT priority: score={dft_priority_score:.3f}, "
            f"tier={tier}, UQ={uq_signal:.3f}, ensemble={ensemble_signal:.3f}"
        )

        return {
            "dft_priority_score": round(dft_priority_score, 4),
            "dft_priority_tier": tier,
            "rationale": rationale,
        }

