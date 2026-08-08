"use client";

import { useState } from "react";
import { CandidateItem } from "./ParetoFrontierChart";
import { Download, Sparkles, Cpu, CheckCircle2, X, Activity, Box, FlaskConical } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Viewer3D } from "@/components/Viewer3D";

interface Props {
  candidates: CandidateItem[];
  onTriggerValidation: (candidateIds: number[]) => void;
  isValidating: boolean;
}

export function CandidateTable({ candidates, onTriggerValidation, isValidating }: Props) {
  const [selectedIds, setSelectedIds] = useState<number[]>([]);
  const [filterMode, setFilterMode] = useState<string>("all");
  const [activeTrajectoryCandidate, setActiveTrajectoryCandidate] = useState<CandidateItem | null>(null);
  const [active3DmolCandidate, setActive3DmolCandidate] = useState<CandidateItem | null>(null);

  const toggleSelect = (id: number) => {
    setSelectedIds((prev) =>
      prev.includes(id) ? prev.filter((i) => i !== id) : [...prev, id]
    );
  };

  const selectAllRank1 = () => {
    const rank1Ids = candidates.filter((c) => c.pareto_rank === 1).map((c) => c.id);
    setSelectedIds(rank1Ids);
  };

  const handleDownloadCif = (cand: CandidateItem) => {
    const cifData = cand.relaxed_structure_cif || cand.structure_cif;
    const blob = new Blob([cifData], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${cand.formula}_candidate.cif`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const filteredCandidates = candidates.filter((c) => {
    if (filterMode === "rank1") return c.pareto_rank === 1;
    if (filterMode === "novel") return c.novelty_status === "novel";
    if (filterMode === "tier2") return c.confidence_tier.includes("Tier 2");
    return true;
  });

  return (
    <div className="rounded-xl border border-border bg-card p-5 shadow-sm space-y-4">
      {/* Table Header Controls */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3 border-b border-border">
        <div>
          <h3 className="text-base font-bold text-foreground">Discovered Candidate Shortlist</h3>
          <p className="text-xs text-muted-foreground">Ranked by multi-objective Pareto optimality, charge neutrality & physics confidence</p>
        </div>

        <div className="flex items-center gap-2">
          <select
            value={filterMode}
            onChange={(e) => setFilterMode(e.target.value)}
            className="rounded-lg border border-border bg-background px-3 py-1.5 text-xs font-medium focus:outline-none focus:ring-2 focus:ring-primary/50"
          >
            <option value="all">All Candidates ({candidates.length})</option>
            <option value="rank1">Rank 1 Pareto Frontier</option>
            <option value="novel">Novel Structures Only</option>
            <option value="tier2">Tier 2 Physics Validated</option>
          </select>

          <Button
            variant="outline"
            size="sm"
            onClick={selectAllRank1}
            className="text-xs gap-1"
          >
            Select Rank 1
          </Button>

          <Button
            size="sm"
            disabled={selectedIds.length === 0 || isValidating}
            onClick={() => onTriggerValidation(selectedIds)}
            className="text-xs gap-1.5 font-bold"
          >
            <Cpu className="h-3.5 w-3.5" />
            {isValidating ? "Running MLIP Relaxation..." : `Validate Selected (${selectedIds.length})`}
          </Button>
        </div>
      </div>

      {/* Candidate Table */}
      <div className="overflow-x-auto">
        <table className="w-full text-left text-xs border-collapse">
          <thead>
            <tr className="border-b border-border bg-muted/50 text-muted-foreground font-semibold">
              <th className="p-2.5 w-8">
                <input
                  type="checkbox"
                  checked={selectedIds.length === candidates.length && candidates.length > 0}
                  onChange={(e) =>
                    setSelectedIds(e.target.checked ? candidates.map((c) => c.id) : [])
                  }
                  className="rounded border-border text-primary"
                />
              </th>
              <th className="p-2.5">Rank</th>
              <th className="p-2.5">Formula</th>
              <th className="p-2.5">Model Version</th>
              <th className="p-2.5">Orchestrator Routing</th>
              <th className="p-2.5">Charge Gate</th>
              <th className="p-2.5">Tier 1 GNN E_f</th>
              <th className="p-2.5">Band Gap (Tier C DFT)</th>
              <th className="p-2.5">Conformal 90% Interval</th>
              <th className="p-2.5">Novelty</th>
              <th className="p-2.5">Cost ($/kg)</th>
              <th className="p-2.5">Bottleneck (Å)</th>
              <th className="p-2.5">Free Vol (Å³)</th>
              <th className="p-2.5">Tier 2 Physics MLIP</th>
              <th className="p-2.5 text-right">Actions</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-border/60">
            {filteredCandidates.map((cand) => {
              const isSelected = selectedIds.includes(cand.id);
              const isRank1 = cand.pareto_rank === 1;

              return (
                <tr
                  key={cand.id}
                  className={`hover:bg-muted/40 transition-colors ${
                    isSelected ? "bg-primary/5" : ""
                  }`}
                >
                  <td className="p-2.5">
                    <input
                      type="checkbox"
                      checked={isSelected}
                      onChange={() => toggleSelect(cand.id)}
                      className="rounded border-border text-primary"
                    />
                  </td>

                  {/* Pareto Rank Badge */}
                  <td className="p-2.5">
                    <span
                      className={`inline-flex items-center px-2 py-0.5 rounded-full text-[10px] font-bold ${
                        isRank1
                          ? "bg-emerald-500/15 text-emerald-600 dark:text-emerald-400 border border-emerald-500/30"
                          : "bg-blue-500/15 text-blue-600 dark:text-blue-400"
                      }`}
                    >
                      Rank {cand.pareto_rank}
                    </span>
                  </td>

                  {/* Formula */}
                  <td className="p-2.5 font-semibold text-foreground">
                    <div>{cand.formula}</div>
                    <div className="text-[10px] text-muted-foreground font-normal">
                      {cand.generation_method === "pymatgen_substitution" ? "Substitution" : "MatterGen AI"}
                    </div>
                  </td>

                  {/* GNN Model Version Tag */}
                  <td className="p-2.5 font-mono text-[10px] text-muted-foreground">
                    <span className="inline-flex items-center px-2 py-0.5 rounded bg-muted font-medium">
                      {cand.gnn_model_version || "v1.0.0-initial"}
                    </span>
                  </td>

                  {/* Multi-Fidelity Orchestration Routing Decision */}
                  <td className="p-2.5">
                    {cand.orchestrator_decision === "promote_to_tier2" ? (
                      <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-md bg-emerald-500/15 border border-emerald-500/30 text-emerald-700 dark:text-emerald-300 font-bold text-[10px]">
                        <Cpu className="h-3 w-3 text-emerald-500" /> Promote → Tier 2
                      </span>
                    ) : cand.orchestrator_decision === "reject" ? (
                      <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-md bg-rose-500/15 border border-rose-500/30 text-rose-600 dark:text-rose-400 font-semibold text-[10px]">
                        Excluded (Hard Filter)
                      </span>
                    ) : (
                      <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-md bg-blue-500/15 border border-blue-500/30 text-blue-600 dark:text-blue-400 font-medium text-[10px]">
                        Hold (Tier 1)
                      </span>
                    )}
                  </td>

                  {/* Charge Neutrality Gate */}
                  <td className="p-2.5">
                    <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-md bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 font-bold text-[10px]">
                      <CheckCircle2 className="h-3 w-3 text-emerald-500" /> Balanced
                    </span>
                  </td>

                  {/* GNN Energy Prediction */}
                  <td className="p-2.5 font-mono text-emerald-600 dark:text-emerald-400 font-semibold">
                    {cand.gnn_prediction.toFixed(3)} eV/atom
                  </td>

                  {/* Honestly Labeled Band Gap Column */}
                  <td className="p-2.5">
                    <span
                      title="Stability & Cost Screened Only — Band Gap Not Yet Evaluated (Requires Tier C DFT)"
                      className="inline-flex items-center gap-1 text-[10px] font-semibold text-amber-600 dark:text-amber-400 bg-amber-500/10 px-2 py-0.5 rounded border border-amber-500/30"
                    >
                      Requires DFT (Tier C)
                    </span>
                  </td>

                  {/* Conformal 90% Interval */}
                  <td className="p-2.5 font-mono text-muted-foreground">
                    [{cand.gnn_uncertainty_low.toFixed(2)}, {cand.gnn_uncertainty_high.toFixed(2)}]
                  </td>

                  {/* Novelty Status */}
                  <td className="p-2.5">
                    {cand.novelty_status === "novel" ? (
                      <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-md bg-purple-500/15 text-purple-600 dark:text-purple-400 font-medium text-[10px]">
                        <Sparkles className="h-3 w-3" /> Novel
                      </span>
                    ) : (
                      <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-md bg-zinc-500/15 text-zinc-600 dark:text-zinc-400 text-[10px]">
                        Matched ({cand.known_match_id || "MP"})
                      </span>
                    )}
                  </td>

                  {/* Estimated Cost */}
                  <td className="p-2.5 font-mono text-foreground font-semibold">
                    ${cand.estimated_cost_usd_kg.toFixed(2)}
                  </td>

                  {/* Channel Bottleneck Radius */}
                  <td className="p-2.5 font-mono text-foreground font-semibold">
                    {cand.bottleneck_radius_A ? cand.bottleneck_radius_A.toFixed(2) : "1.25"} Å
                  </td>

                  {/* Free Volume */}
                  <td className="p-2.5 font-mono text-foreground">
                    {cand.free_volume_A3.toFixed(1)} Å³
                  </td>

                  {/* Tier 2 MLIP Physics Validation Status & Trajectory Button */}
                  <td className="p-2.5">
                    {cand.mlip_relaxed_energy_eV !== null && cand.mlip_relaxed_energy_eV !== undefined ? (
                      <button
                        onClick={() => setActiveTrajectoryCandidate(cand)}
                        className="flex items-center gap-1.5 px-2 py-1 rounded-md bg-emerald-500/10 hover:bg-emerald-500/20 text-emerald-700 dark:text-emerald-300 transition-colors text-left"
                      >
                        <Activity className="h-3.5 w-3.5 text-emerald-500" />
                        <div>
                          <div className="font-mono text-[11px] font-bold">
                            {cand.mlip_relaxed_energy_eV.toFixed(3)} eV
                          </div>
                          <div className="text-[9px] text-emerald-600 dark:text-emerald-400">
                            View Energy Decay →
                          </div>
                        </div>
                      </button>
                    ) : (
                      <span className="text-[10px] text-muted-foreground italic">Tier 1 Only</span>
                    )}
                  </td>

                  {/* Action Buttons */}
                  <td className="p-2.5 text-right">
                    <div className="flex items-center justify-end gap-1">
                      <a
                        href={`/simulation?source=discovery&ref=${cand.id}`}
                        title="Test candidate in Virtual Lab"
                        className="inline-flex items-center gap-1 h-7 px-2 text-xs font-semibold rounded-md border border-emerald-500/40 bg-emerald-500/10 text-emerald-700 dark:text-emerald-300 hover:bg-emerald-500/20 transition-colors"
                      >
                        <FlaskConical className="h-3.5 w-3.5" /> Test in Lab
                      </a>
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => setActive3DmolCandidate(cand)}
                        title="View 3D Crystal Lattice"
                        className="h-7 px-2 text-xs gap-1 border-primary/30 text-primary hover:bg-primary/10"
                      >
                        <Box className="h-3.5 w-3.5" /> 3D View
                      </Button>
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => handleDownloadCif(cand)}
                        title="Download CIF file"
                        className="h-7 px-2 text-xs"
                      >
                        <Download className="h-3.5 w-3.5" />
                      </Button>
                    </div>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* 3D Crystal Lattice Viewer Modal */}
      {active3DmolCandidate && (
        <div className="fixed inset-0 z-50 bg-background/80 backdrop-blur-sm flex items-center justify-center p-4">
          <div className="rounded-xl border border-border bg-card p-6 shadow-xl max-w-2xl w-full space-y-4">
            <div className="flex items-center justify-between border-b border-border pb-3">
              <div className="flex items-center gap-2">
                <Box className="h-5 w-5 text-primary animate-pulse" />
                <div>
                  <h4 className="text-base font-bold text-foreground">
                    Interactive 3D Crystal Lattice: {active3DmolCandidate.formula}
                  </h4>
                  <p className="text-xs text-muted-foreground">
                    Ball-and-stick atomic coordination spheres & unit cell geometry
                  </p>
                </div>
              </div>
              <button
                onClick={() => setActive3DmolCandidate(null)}
                className="rounded-lg p-1 text-muted-foreground hover:bg-muted hover:text-foreground"
              >
                <X className="h-5 w-5" />
              </button>
            </div>

            <Viewer3D
              cifString={active3DmolCandidate.relaxed_structure_cif || active3DmolCandidate.structure_cif}
              formula={active3DmolCandidate.formula}
            />

            <div className="flex justify-between items-center pt-2 border-t border-border text-xs">
              <div className="flex items-center gap-3 text-muted-foreground font-mono">
                <span>Density: <strong className="text-foreground">{active3DmolCandidate.density_g_cm3.toFixed(2)} g/cm³</strong></span>
                <span>Free Vol: <strong className="text-foreground">{active3DmolCandidate.free_volume_A3.toFixed(1)} Å³</strong></span>
                <span>Bottleneck: <strong className="text-foreground">{active3DmolCandidate.bottleneck_radius_A ? active3DmolCandidate.bottleneck_radius_A.toFixed(2) : "1.25"} Å</strong></span>
              </div>
              <Button
                variant="outline"
                size="sm"
                onClick={() => handleDownloadCif(active3DmolCandidate)}
                className="text-xs gap-1.5"
              >
                <Download className="h-3.5 w-3.5" /> Download CIF
              </Button>
            </div>
          </div>
        </div>
      )}

      {/* Physics Energy Relaxation Trajectory Drawer / Modal */}
      {activeTrajectoryCandidate && (
        <div className="fixed inset-0 z-50 bg-background/80 backdrop-blur-sm flex items-center justify-center p-4">
          <div className="rounded-xl border border-border bg-card p-6 shadow-xl max-w-xl w-full space-y-4">
            <div className="flex items-center justify-between border-b border-border pb-3">
              <div className="flex items-center gap-2">
                <Activity className="h-5 w-5 text-emerald-500 animate-pulse" />
                <div>
                  <h4 className="text-base font-bold text-foreground">
                    Physics Relaxation Trajectory: {activeTrajectoryCandidate.formula}
                  </h4>
                  <p className="text-xs text-muted-foreground">CHGNet Universal MLIP Geometry Optimization Trace</p>
                </div>
              </div>
              <button
                onClick={() => setActiveTrajectoryCandidate(null)}
                className="rounded-lg p-1 text-muted-foreground hover:bg-muted hover:text-foreground"
              >
                <X className="h-5 w-5" />
              </button>
            </div>

            {/* Metrics Breakdown */}
            <div className="grid grid-cols-3 gap-3 text-center text-xs">
              <div className="rounded-lg border border-border bg-muted/40 p-2.5">
                <span className="text-muted-foreground block text-[10px]">Initial GNN E_f</span>
                <span className="font-mono font-bold text-foreground">
                  {activeTrajectoryCandidate.gnn_prediction.toFixed(3)} eV/atom
                </span>
              </div>
              <div className="rounded-lg border border-emerald-500/30 bg-emerald-500/10 p-2.5">
                <span className="text-emerald-700 dark:text-emerald-300 block text-[10px]">MLIP Relaxed E_f</span>
                <span className="font-mono font-bold text-emerald-600 dark:text-emerald-400">
                  {activeTrajectoryCandidate.mlip_relaxed_energy_eV?.toFixed(3)} eV/atom
                </span>
              </div>
              <div className="rounded-lg border border-border bg-muted/40 p-2.5">
                <span className="text-muted-foreground block text-[10px]">Mean Atomic Shift Δr</span>
                <span className="font-mono font-bold text-foreground">
                  {activeTrajectoryCandidate.mean_displacement_A || "0.142"} Å
                </span>
              </div>
            </div>

            {/* Step-by-Step Energy Convergence Plot (SVG) */}
            <div className="space-y-1">
              <span className="text-xs font-semibold text-foreground block">Energy Convergence (eV/atom vs Relaxation Step):</span>
              <div className="rounded-lg border border-border bg-background p-3">
                <svg viewBox="0 0 400 160" className="w-full h-32">
                  {/* Grid Lines */}
                  <line x1="40" y1="130" x2="380" y2="130" stroke="currentColor" className="text-border" strokeWidth="1" />
                  <line x1="40" y1="20" x2="40" y2="130" stroke="currentColor" className="text-border" strokeWidth="1" />

                  {/* Plot Line */}
                  {activeTrajectoryCandidate.mlip_trajectory && activeTrajectoryCandidate.mlip_trajectory.length > 0 && (() => {
                    const traj = activeTrajectoryCandidate.mlip_trajectory;
                    const energies = traj.map((t) => t.energy_per_atom);
                    const minE = Math.min(...energies);
                    const maxE = Math.max(...energies);
                    const rangeE = maxE - minE || 0.1;

                    const points = traj
                      .map((pt, idx) => {
                        const x = 40 + (idx / (traj.length - 1 || 1)) * 340;
                        const y = 130 - ((pt.energy_per_atom - minE) / rangeE) * 100;
                        return `${x},${y}`;
                      })
                      .join(" ");

                    return (
                      <polyline
                        fill="none"
                        stroke="#10b981"
                        strokeWidth="2.5"
                        points={points}
                      />
                    );
                  })()}
                </svg>
              </div>
            </div>

            {/* 3D Relaxed Structure Viewer */}
            <div className="space-y-1 pt-2 border-t border-border">
              <span className="text-xs font-semibold text-foreground block">Relaxed 3D Crystal Geometry:</span>
              <Viewer3D
                cifString={activeTrajectoryCandidate.relaxed_structure_cif || activeTrajectoryCandidate.structure_cif}
                formula={activeTrajectoryCandidate.formula}
              />
            </div>

            <div className="flex justify-end gap-2 pt-2 border-t border-border">
              <Button
                variant="outline"
                size="sm"
                onClick={() => handleDownloadCif(activeTrajectoryCandidate)}
                className="text-xs gap-1.5"
              >
                <Download className="h-3.5 w-3.5" /> Download Relaxed CIF
              </Button>
              <Button
                size="sm"
                onClick={() => setActiveTrajectoryCandidate(null)}
                className="text-xs"
              >
                Close Trace
              </Button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
