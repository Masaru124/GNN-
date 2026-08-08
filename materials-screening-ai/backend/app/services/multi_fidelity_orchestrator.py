# -*- coding: utf-8 -*-
"""
Multi-Fidelity Information-Gain Orchestrator.

Intelligently routes discovery candidates between Tier 1 GNN screening and Tier 2 MLIP Physics Validation
based on Expected Information Gain (Expected Improvement heuristic):
  - Evaluates GNN Uncertainty Interval Width (conformal 90% width)
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
