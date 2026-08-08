# -*- coding: utf-8 -*-
"""
Multi-Objective Pareto Frontier Optimization & Ranking Service.

Ranks discovery candidates into Pareto Frontiers across competing axes:
  1. Target Property (Formation Energy Ef minimization)
  2. Raw Material Cost ($/kg minimization)
  3. Structural Density / Stability
  4. Transport Free Volume (Å³ maximization)
"""

from typing import Any, Dict, List


class ParetoRanker:
    """Non-dominated sorting algorithm for multi-objective candidate ranking."""

    def __init__(
        self,
        minimize_property: bool = True,
        minimize_cost: bool = True,
        maximize_free_volume: bool = True
    ):
        self.minimize_property = minimize_property
        self.minimize_cost = minimize_cost
        self.maximize_free_volume = maximize_free_volume

    def _dominates(self, p1: Dict[str, Any], p2: Dict[str, Any]) -> bool:
        """
        Check if candidate p1 strictly dominates candidate p2.
        p1 dominates p2 iff p1 is no worse than p2 in all objectives and strictly better in at least one.
        """
        # Objective 1: Predicted Formation Energy
        e1 = p1.get("gnn_prediction", 0.0)
        e2 = p2.get("gnn_prediction", 0.0)
        obj1_better = (e1 < e2) if self.minimize_property else (e1 > e2)
        obj1_worse = (e1 > e2) if self.minimize_property else (e1 < e2)

        # Objective 2: Cost ($/kg)
        c1 = p1.get("estimated_cost_usd_kg", 0.0)
        c2 = p2.get("estimated_cost_usd_kg", 0.0)
        obj2_better = (c1 < c2) if self.minimize_cost else (c1 > c2)
        obj2_worse = (c1 > c2) if self.minimize_cost else (c1 < c2)

        # Objective 3: Transport Free Volume (Å³)
        v1 = p1.get("free_volume_A3", 0.0)
        v2 = p2.get("free_volume_A3", 0.0)
        obj3_better = (v1 > v2) if self.maximize_free_volume else (v1 < v2)
        obj3_worse = (v1 < v2) if self.maximize_free_volume else (v1 > v2)

        at_least_one_better = obj1_better or obj2_better or obj3_better
        no_worse = (not obj1_worse) and (not obj2_worse) and (not obj3_worse)

        return at_least_one_better and no_worse

    def rank_candidates(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Perform fast non-dominated sorting on a list of candidate dictionaries.

        Assigns `pareto_rank` (1 = non-dominated frontier, 2 = second frontier, etc.).
        """
        if not candidates:
            return []

        remaining = list(range(len(candidates)))
        current_rank = 1

        while remaining:
            frontier = []
            for i in remaining:
                dominated = False
                for j in remaining:
                    if i != j and self._dominates(candidates[j], candidates[i]):
                        dominated = True
                        break
                if not dominated:
                    frontier.append(i)

            if not frontier:
                # Fallback if remaining candidates form a dominance cycle
                for i in remaining:
                    candidates[i]["pareto_rank"] = current_rank
                break

            for i in frontier:
                candidates[i]["pareto_rank"] = current_rank
                remaining.remove(i)

            current_rank += 1

        # Sort candidate list by pareto_rank ascending
        candidates.sort(key=lambda x: (x.get("pareto_rank", 999), x.get("gnn_prediction", 0.0)))
        return candidates
