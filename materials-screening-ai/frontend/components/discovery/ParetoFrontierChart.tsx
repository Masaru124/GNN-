"use client";

import { useMemo } from "react";
import { Layers, Sparkles, AlertCircle } from "lucide-react";

export interface CandidateItem {
  id: number;
  candidate_index: number;
  formula: string;
  structure_cif: string;
  generation_method: string;
  gnn_prediction: number;
  gnn_uncertainty_low: number;
  gnn_uncertainty_high: number;
  novelty_status: string;
  known_match_id?: string;
  hard_filter_pass: boolean;
  density_g_cm3: number;
  estimated_cost_usd_kg: number;
  free_volume_A3: number;
  bottleneck_radius_A: number;
  confidence_tier: string;
  pareto_rank: number;
  charge_neutral_pass?: boolean;
  mlip_relaxed_energy_eV?: number;
  mlip_stability_flag?: boolean;
  mlip_trajectory?: { step: number; energy_per_atom: number }[];
  relaxed_structure_cif?: string;
  mean_displacement_A?: number;
}

interface Props {
  candidates: CandidateItem[];
  onSelectCandidate?: (candidate: CandidateItem) => void;
}

export function ParetoFrontierChart({ candidates, onSelectCandidate }: Props) {
  const validCandidates = useMemo(
    () => candidates.filter((c) => c.hard_filter_pass),
    [candidates]
  );

  if (validCandidates.length === 0) {
    return (
      <div className="rounded-xl border border-border bg-card p-6 text-center text-muted-foreground">
        <AlertCircle className="mx-auto h-8 w-8 text-amber-500 mb-2" />
        <p className="text-sm font-medium">No valid candidates passing hard filters found for Pareto visualization.</p>
      </div>
    );
  }

  // Calculate Bounds for SVG Plotting
  const costs = validCandidates.map((c) => c.estimated_cost_usd_kg);
  const energies = validCandidates.map((c) => c.gnn_prediction);

  const minCost = Math.min(...costs, 0);
  const maxCost = Math.max(...costs, 100);
  const minEnergy = Math.min(...energies, -3.5);
  const maxEnergy = Math.max(...energies, 0.5);

  const width = 600;
  const height = 320;
  const padding = 50;

  const scaleX = (cost: number) => {
    const range = maxCost - minCost || 1;
    return padding + ((cost - minCost) / range) * (width - padding * 2);
  };

  const scaleY = (energy: number) => {
    const range = maxEnergy - minEnergy || 1;
    return height - padding - ((energy - minEnergy) / range) * (height - padding * 2);
  };

  // Pareto Rank 1 Frontier Points sorted by cost
  const rank1Points = validCandidates
    .filter((c) => c.pareto_rank === 1)
    .sort((a, b) => a.estimated_cost_usd_kg - b.estimated_cost_usd_kg);

  const frontierPath = rank1Points
    .map((pt, idx) => `${idx === 0 ? "M" : "L"} ${scaleX(pt.estimated_cost_usd_kg)} ${scaleY(pt.gnn_prediction)}`)
    .join(" ");

  return (
    <div className="rounded-xl border border-border bg-card p-5 shadow-sm">
      <div className="flex items-center justify-between mb-4 pb-3 border-b border-border">
        <div className="flex items-center gap-2">
          <Layers className="h-5 w-5 text-primary" />
          <h3 className="text-base font-bold text-foreground">Multi-Objective Pareto Frontier</h3>
        </div>
        <div className="flex items-center gap-3 text-xs">
          <span className="flex items-center gap-1">
            <span className="h-2.5 w-2.5 rounded-full bg-emerald-500 inline-block" />
            Rank 1 Frontier (Non-dominated)
          </span>
          <span className="flex items-center gap-1">
            <span className="h-2.5 w-2.5 rounded-full bg-blue-500 inline-block" />
            Rank 2+ Sub-Frontier
          </span>
        </div>
      </div>

      <div className="relative w-full overflow-x-auto">
        <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-auto max-h-[340px] select-none">
          {/* Axis Grid lines */}
          <line x1={padding} y1={height - padding} x2={width - padding} y2={height - padding} stroke="currentColor" className="text-border" strokeWidth="1.5" />
          <line x1={padding} y1={padding} x2={padding} y2={height - padding} stroke="currentColor" className="text-border" strokeWidth="1.5" />

          {/* Axis Labels */}
          <text x={width / 2} y={height - 12} textAnchor="middle" className="fill-muted-foreground text-[11px] font-medium">
            Estimated Raw Material Cost ($/kg) → (Lower is Better)
          </text>
          <text x={15} y={height / 2} textAnchor="middle" transform={`rotate(-90 15 ${height / 2})`} className="fill-muted-foreground text-[11px] font-medium">
            Formation Energy E_f (eV/atom) → (Lower is Better)
          </text>

          {/* Pareto Frontier Line */}
          {frontierPath && (
            <path
              d={frontierPath}
              fill="none"
              stroke="#10b981"
              strokeWidth="2"
              strokeDasharray="4 4"
              className="opacity-80"
            />
          )}

          {/* Candidate Scatter Plot Nodes */}
          {validCandidates.map((c) => {
            const cx = scaleX(c.estimated_cost_usd_kg);
            const cy = scaleY(c.gnn_prediction);
            const isRank1 = c.pareto_rank === 1;

            return (
              <g
                key={c.id}
                onClick={() => onSelectCandidate && onSelectCandidate(c)}
                className="cursor-pointer group"
              >
                <circle
                  cx={cx}
                  cy={cy}
                  r={isRank1 ? 7 : 5}
                  className={
                    isRank1
                      ? "fill-emerald-500 stroke-background stroke-2 group-hover:scale-125 transition-transform"
                      : "fill-blue-500 stroke-background stroke-2 group-hover:scale-125 transition-transform opacity-85"
                  }
                />
                <text
                  x={cx + 10}
                  y={cy + 4}
                  className="fill-foreground text-[10px] font-semibold opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none"
                >
                  {c.formula} ({c.gnn_prediction.toFixed(2)} eV, ${c.estimated_cost_usd_kg}/kg)
                </text>
              </g>
            );
          })}
        </svg>
      </div>

      <div className="mt-3 flex items-center justify-between text-xs text-muted-foreground border-t border-border/60 pt-3">
        <span>Click any candidate point to highlight in candidate table</span>
        <span className="font-semibold text-emerald-600 dark:text-emerald-400">
          {rank1Points.length} candidates on optimal Pareto Frontier
        </span>
      </div>
    </div>
  );
}
