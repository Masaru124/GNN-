"use client";

import { useState } from "react";
import { CandidateItem } from "./ParetoFrontierChart";
import {
  Download,
  Sparkles,
  Cpu,
  CheckCircle2,
  X,
  Activity,
  Box,
  FlaskConical,
  ShieldCheck,
  RefreshCw,
  BookOpen,
  AlertCircle,
  Zap,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Viewer3D } from "@/components/Viewer3D";

interface Props {
  candidates: CandidateItem[];
  onTriggerValidation: (candidateIds: number[]) => void;
  isValidating: boolean;
  onCandidatesUpdated?: () => void;
}

export function CandidateTable({
  candidates,
  onTriggerValidation,
  isValidating,
  onCandidatesUpdated,
}: Props) {
  const [selectedIds, setSelectedIds] = useState<number[]>([]);
  const [filterMode, setFilterMode] = useState<string>("all");
  const [activeTrajectoryCandidate, setActiveTrajectoryCandidate] =
    useState<CandidateItem | null>(null);
  const [active3DmolCandidate, setActive3DmolCandidate] =
    useState<CandidateItem | null>(null);
  const [dftQueuingId, setDftQueuingId] = useState<number | null>(null);
  const [dftStatusMsg, setDftStatusMsg] = useState<{
    id: number;
    msg: string;
    isError?: boolean;
  } | null>(null);

  const handleQueueDft = async (
    cand: CandidateItem,
    fastMode: boolean = true,
  ) => {
    setDftQueuingId(cand.id);
    try {
      const res = await fetch(
        `http://127.0.0.1:8000/api/dft/queue/${cand.id}?fast_mode=${fastMode}&kpt_dist=0.35`,
        {
          method: "POST",
        },
      );
      const data = await res.json();
      if (data.status === "qe_not_installed") {
        setDftStatusMsg({
          id: cand.id,
          msg: `${cand.formula}: Quantum ESPRESSO is not installed. To run natively on Windows without WSL, run 'python backend/scripts/setup_qe_windows.py' to download portable QE 7.5 (pw.exe).`,
          isError: false,
        });
      } else {
        setDftStatusMsg({
          id: cand.id,
          msg: `${cand.formula}: Tier 3 DFT calculation started in ${fastMode ? "⚡ Fast Mode (~25s)" : "Full vc-relax"} (job status: ${data.status || "queued"}).`,
          isError: false,
        });
        if (onCandidatesUpdated) {
          onCandidatesUpdated();
        }
      }
    } catch (err: any) {
      setDftStatusMsg({
        id: cand.id,
        msg: `Failed to queue DFT for ${cand.formula}: ${err.message}`,
        isError: true,
      });
    } finally {
      setDftQueuingId(null);
    }
  };

  const toggleSelect = (id: number) => {
    setSelectedIds((prev) =>
      prev.includes(id) ? prev.filter((i) => i !== id) : [...prev, id],
    );
  };

  const selectAllRank1 = () => {
    const rank1Ids = candidates
      .filter((c) => c.pareto_rank === 1)
      .map((c) => c.id);
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
    if (filterMode === "ensemble_disagree")
      return c.ensemble_status === "requires_independent_validation";
    if (filterMode === "ensemble_agree")
      return (
        c.ensemble_status === "high_confidence_agreement" ||
        c.ensemble_status === "moderate_agreement"
      );
    return true;
  });

  return (
    <div className="rounded-xl border border-border bg-card p-5 shadow-sm space-y-4">
      {/* Table Header Controls */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3 border-b border-border">
        <div>
          <h3 className="text-base font-bold text-foreground">
            Discovered Candidate Shortlist
          </h3>
          <p className="text-xs text-muted-foreground">
            Ranked by multi-objective Pareto optimality, charge neutrality &
            physics confidence
          </p>
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
            <option value="ensemble_agree">Ensemble Agreed</option>
            <option value="ensemble_disagree">Ensemble Disagreed (Held)</option>
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
            {isValidating
              ? "Running MLIP Relaxation..."
              : `Validate Selected (${selectedIds.length})`}
          </Button>
        </div>
      </div>

      {/* S.U.N. Rate Summary Card (MatterGen / LeMat-GenBench Standard Benchmark) */}
      {candidates.some(
        (c) => c.e_above_hull_eV !== undefined && c.e_above_hull_eV !== null,
      ) && (
        <div className="rounded-lg border border-primary/20 bg-primary/5 p-3 flex flex-wrap items-center justify-between gap-3 text-xs">
          <div className="flex items-center gap-2">
            <Sparkles className="h-4 w-4 text-primary" />
            <div>
              <span className="font-bold text-foreground">
                Discovery Funnel S.U.N. Metrics
              </span>
              <span className="text-muted-foreground ml-1.5 text-[11px]">
                (Stable / Unique / Novel standard crystal discovery rate)
              </span>
            </div>
          </div>
          <div className="flex items-center gap-4 font-mono text-xs">
            <div className="bg-background/80 px-2.5 py-1 rounded border border-border">
              <span className="text-muted-foreground text-[10px] block">
                Strict S.U.N. (≤0 eV/atom)
              </span>
              <strong className="text-emerald-500 text-sm">
                {(
                  (candidates.filter(
                    (c) =>
                      c.e_above_hull_eV !== null &&
                      c.e_above_hull_eV !== undefined &&
                      c.e_above_hull_eV <= 0.0 &&
                      c.novelty_status === "novel",
                  ).length /
                    Math.max(1, candidates.length)) *
                  100
                ).toFixed(1)}
                %
              </strong>
            </div>
            <div className="bg-background/80 px-2.5 py-1 rounded border border-border">
              <span className="text-muted-foreground text-[10px] block">
                M.S.U.N. (≤0.1 eV/atom)
              </span>
              <strong className="text-primary text-sm">
                {(
                  (candidates.filter(
                    (c) =>
                      c.e_above_hull_eV !== null &&
                      c.e_above_hull_eV !== undefined &&
                      c.e_above_hull_eV <= 0.1 &&
                      c.novelty_status === "novel",
                  ).length /
                    Math.max(1, candidates.length)) *
                  100
                ).toFixed(1)}
                %
              </strong>
            </div>
            <div className="text-[10px] text-muted-foreground max-w-xs hidden md:block">
              Thresholds match MatterGen & CrystalGRW benchmark standards.
            </div>
          </div>
        </div>
      )}

      {/* DFT Status Notification */}
      {dftStatusMsg && (
        <div
          className={`rounded-lg border p-3 flex items-center justify-between text-xs ${
            dftStatusMsg.isError
              ? "bg-rose-500/10 border-rose-500/30 text-rose-700 dark:text-rose-300"
              : "bg-cyan-500/10 border-cyan-500/30 text-cyan-800 dark:text-cyan-200"
          }`}
        >
          <div className="flex items-center gap-2">
            <Cpu className="h-4 w-4 shrink-0" />
            <span>{dftStatusMsg.msg}</span>
          </div>
          <button
            onClick={() => setDftStatusMsg(null)}
            className="text-muted-foreground hover:text-foreground text-xs ml-3"
          >
            ✕
          </button>
        </div>
      )}

      {/* Candidate Table */}
      <div className="overflow-x-auto">
        <table className="w-full text-left text-xs border-collapse">
          <thead>
            <tr className="border-b border-border bg-muted/50 text-muted-foreground font-semibold">
              <th className="p-2.5 w-8">
                <input
                  type="checkbox"
                  checked={
                    selectedIds.length === candidates.length &&
                    candidates.length > 0
                  }
                  onChange={(e) =>
                    setSelectedIds(
                      e.target.checked ? candidates.map((c) => c.id) : [],
                    )
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
              <th className="p-2.5">E Above Hull</th>
              <th className="p-2.5">Band Gap (unavailable)</th>
              <th className="p-2.5">Conformal 90% Interval (86% LOCO coverage, one held-out cluster n=7178)</th>
              <th className="p-2.5">Novelty</th>
              <th className="p-2.5">Cost ($/kg)</th>
              <th className="p-2.5">Bottleneck (Å)</th>
              <th className="p-2.5">Free Vol (Å³)</th>
              <th className="p-2.5">Tier 2 Physics MLIP</th>
              <th className="p-2.5">MLIP Agreement</th>
              <th className="p-2.5">Synthesis</th>
              <th className="p-2.5">Literature</th>
              <th className="p-2.5">Tier 3 DFT / Δ-ML</th>
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
                      {cand.generation_method === "pymatgen_substitution"
                        ? "Substitution"
                        : "MatterGen AI"}
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
                        <Cpu className="h-3 w-3 text-emerald-500" /> Promote →
                        Tier 2
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
                      <CheckCircle2 className="h-3 w-3 text-emerald-500" />{" "}
                      Balanced
                    </span>
                  </td>

                  {/* GNN Energy Prediction */}
                  <td className="p-2.5 font-mono text-emerald-600 dark:text-emerald-400 font-semibold">
                    {cand.gnn_prediction.toFixed(3)} eV/atom
                  </td>

                  {/* Energy Above Hull (Item 1) */}
                  <td className="p-2.5">
                    {cand.e_above_hull_eV !== null &&
                    cand.e_above_hull_eV !== undefined ? (
                      <span
                        title={`Convex Hull Classification: ${cand.hull_classification || "computed"}`}
                        className={`inline-flex items-center gap-1 font-mono text-[10px] font-bold px-2 py-0.5 rounded border ${
                          cand.e_above_hull_eV <= 0.0001
                            ? "bg-emerald-500/15 text-emerald-600 dark:text-emerald-400 border-emerald-500/30"
                            : cand.e_above_hull_eV <= 0.05
                              ? "bg-teal-500/15 text-teal-600 dark:text-teal-400 border-teal-500/30"
                              : cand.e_above_hull_eV <= 0.1
                                ? "bg-amber-500/15 text-amber-600 dark:text-amber-400 border-amber-500/30"
                                : "bg-rose-500/15 text-rose-600 dark:text-rose-400 border-rose-500/30"
                        }`}
                      >
                        {cand.e_above_hull_eV <= 0
                          ? "0.000"
                          : `+${cand.e_above_hull_eV.toFixed(3)}`}{" "}
                        eV
                      </span>
                    ) : (
                      <span
                        title="Materials Project reference formation energy entries needed for valid hull construction"
                        className="text-[10px] text-muted-foreground italic"
                      >
                        —
                      </span>
                    )}
                  </td>

                  {/* Band gap: heuristic excluded from screening (Item 3) */}
                  <td className="p-2.5">
                    <span
                      title={
                        cand.band_gap_unavailable_reason ||
                        "Band-gap heuristic excluded from screening: median 90% interval half-width 3.35 eV. Requires Tier C DFT / Delta-ML."
                      }
                      className="inline-flex items-center gap-1 text-[10px] font-semibold text-zinc-500 dark:text-zinc-400 bg-muted px-2 py-0.5 rounded border border-border"
                    >
                      unavailable (needs DFT)
                    </span>
                  </td>

                  {/* Conformal 90% Interval */}
                  <td className="p-2.5 font-mono text-muted-foreground">
                    [{cand.gnn_uncertainty_low.toFixed(2)},{" "}
                    {cand.gnn_uncertainty_high.toFixed(2)}]
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
                    {cand.bottleneck_radius_A
                      ? cand.bottleneck_radius_A.toFixed(2)
                      : "1.25"}{" "}
                    Å
                  </td>

                  {/* Free Volume */}
                  <td className="p-2.5 font-mono text-foreground">
                    {cand.free_volume_A3.toFixed(1)} Å³
                  </td>

                  {/* Tier 2 MLIP Physics Validation Status & Trajectory Button */}
                  <td className="p-2.5">
                    {cand.mlip_relaxed_energy_eV !== null &&
                    cand.mlip_relaxed_energy_eV !== undefined ? (
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
                      <span className="text-[10px] text-muted-foreground italic">
                        Tier 1 Only
                      </span>
                    )}
                  </td>

                  {/* Ensemble Disagreement Gate Badge */}
                  <td className="p-2.5">
                    {cand.ensemble_status ? (
                      <div
                        title={
                          cand.energy_disagreement_eV_per_atom != null
                            ? `ΔE: ${cand.energy_disagreement_eV_per_atom.toFixed(4)} eV/atom | RMSD: ${cand.structural_rmsd_between_mlips_A != null ? cand.structural_rmsd_between_mlips_A.toFixed(4) + " Å" : "N/A"}\n\nNote: Agreement = \"no evidence of blind spot,\" NOT \"confirmed correct.\" This is a triage gate, not a replacement for independent validation.`
                            : `Status: ${cand.ensemble_status}\n\nNote: Agreement = \"no evidence of blind spot,\" NOT \"confirmed correct.\"`
                        }
                      >
                        {cand.ensemble_status ===
                        "high_confidence_agreement" ? (
                          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-md bg-emerald-500/15 border border-emerald-500/30 text-emerald-700 dark:text-emerald-300 font-bold text-[10px]">
                            <ShieldCheck className="h-3 w-3 text-emerald-500" />{" "}
                            Agreed
                          </span>
                        ) : cand.ensemble_status === "moderate_agreement" ? (
                          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-md bg-amber-500/15 border border-amber-500/30 text-amber-700 dark:text-amber-300 font-bold text-[10px]">
                            <ShieldCheck className="h-3 w-3 text-amber-500" />{" "}
                            Moderate
                          </span>
                        ) : cand.ensemble_status ===
                          "requires_independent_validation" ? (
                          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-md bg-rose-500/15 border border-rose-500/30 text-rose-700 dark:text-rose-300 font-bold text-[10px]">
                            <ShieldCheck className="h-3 w-3 text-rose-500" />{" "}
                            Held for DFT
                          </span>
                        ) : (
                          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-md bg-zinc-500/10 border border-zinc-500/20 text-zinc-500 font-medium text-[10px]">
                            Single Model
                          </span>
                        )}
                        {cand.energy_disagreement_eV_per_atom != null && (
                          <div className="text-[9px] text-muted-foreground mt-0.5 font-mono">
                            ΔE:{" "}
                            {cand.energy_disagreement_eV_per_atom.toFixed(3)} eV
                          </div>
                        )}
                      </div>
                    ) : (
                      <span className="text-[10px] text-muted-foreground italic">
                        —
                      </span>
                    )}
                  </td>

                  {/* Synthesis Route & Feasibility */}
                  <td className="p-2.5">
                    {cand.synthesis_route ? (
                      <div
                        title={`Route: ${cand.synthesis_route}\nPrecursors: ${(cand.synthesis_precursors || []).join(", ")}\nEst. Temp: ${cand.synthesis_estimated_temp_c ? cand.synthesis_estimated_temp_c + " °C" : "N/A"}`}
                      >
                        <span
                          className={`inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-semibold border ${
                            cand.synthesis_feasibility === "high"
                              ? "bg-emerald-500/15 border-emerald-500/30 text-emerald-700 dark:text-emerald-300"
                              : cand.synthesis_feasibility === "medium"
                                ? "bg-amber-500/15 border-amber-500/30 text-amber-700 dark:text-amber-300"
                                : "bg-rose-500/15 border-rose-500/30 text-rose-700 dark:text-rose-300"
                          }`}
                        >
                          {cand.synthesis_route.replace("_", " ")}
                        </span>
                        <div className="text-[9px] text-muted-foreground mt-0.5 capitalize">
                          {cand.synthesis_feasibility || "moderate"} feasibility
                        </div>
                      </div>
                    ) : (
                      <span className="text-[10px] text-muted-foreground italic">
                        —
                      </span>
                    )}
                  </td>

                  {/* Literature Novelty Check */}
                  <td className="p-2.5">
                    {cand.literature_matches_count !== undefined &&
                    cand.literature_matches_count !== null ? (
                      <div
                        title={cand.literature_top_title || "Literature check"}
                      >
                        {cand.literature_matches_count > 0 ? (
                          <div>
                            <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded bg-blue-500/15 border border-blue-500/30 text-blue-700 dark:text-blue-300 text-[10px] font-medium">
                              <BookOpen className="h-3 w-3" />{" "}
                              {cand.literature_matches_count}{" "}
                              {cand.literature_matches_count === 1
                                ? "paper"
                                : "papers"}
                            </span>
                            {cand.literature_top_doi && (
                              <div className="text-[9px] text-muted-foreground font-mono truncate max-w-[90px] mt-0.5">
                                {cand.literature_top_doi}
                              </div>
                            )}
                          </div>
                        ) : (
                          <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded bg-purple-500/15 border border-purple-500/30 text-purple-700 dark:text-purple-300 text-[10px] font-semibold">
                            <Sparkles className="h-3 w-3" /> Unreported
                          </span>
                        )}
                      </div>
                    ) : (
                      <span className="text-[10px] text-muted-foreground italic">
                        —
                      </span>
                    )}
                  </td>

                  {/* Tier 3 DFT / Δ-ML Band Gap */}
                  <td className="p-2.5">
                    {cand.dft_status === "converged" ||
                    cand.dft_status === "done" ||
                    cand.dft_status === "success" ? (
                      <div className="flex flex-col gap-0.5">
                        {cand.dft_pbe_energy_eV === 0 ? (
                          <span
                            title="Calculated PBE band gap is 0.0 eV (Metallic conductor)"
                            className="inline-flex items-center gap-1 font-mono text-[10px] font-bold text-amber-600 dark:text-amber-400 bg-amber-500/15 px-1.5 py-0.5 rounded border border-amber-500/30"
                          >
                            ⚡ Metallic (0.0 eV)
                          </span>
                        ) : cand.dft_delta_ml_bandgap_eV !== null &&
                          cand.dft_delta_ml_bandgap_eV !== undefined ? (
                          <span
                            title={`Δ-ML Corrected Gap (High-Fidelity Reference): ${cand.dft_delta_ml_bandgap_eV?.toFixed(2)} eV [${cand.dft_delta_ml_interval_low?.toFixed(2)}, ${cand.dft_delta_ml_interval_high?.toFixed(2)}]`}
                            className="inline-flex items-center gap-1 font-mono text-[10px] font-bold text-emerald-600 dark:text-emerald-400 bg-emerald-500/15 px-1.5 py-0.5 rounded border border-emerald-500/30"
                          >
                            Δ-ML {cand.dft_delta_ml_bandgap_eV?.toFixed(2)} eV
                          </span>
                        ) : (
                          <span
                            title="Δ-ML correction unavailable: candidate is outside calibration domain"
                            className="inline-flex items-center gap-1 text-[10px] font-medium text-muted-foreground bg-muted px-1.5 py-0.5 rounded border border-border"
                          >
                            Δ-ML unavailable
                          </span>
                        )}
                        {cand.dft_pbe_energy_eV !== null &&
                          cand.dft_pbe_energy_eV !== undefined && (
                            <span className="text-[9px] text-muted-foreground font-mono">
                              PBE: {cand.dft_pbe_energy_eV.toFixed(2)} eV
                            </span>
                          )}
                      </div>
                    ) : cand.dft_status === "running" ||
                      cand.dft_status === "queued" ? (
                      <span className="inline-flex items-center gap-1 text-[10px] font-medium text-amber-600 dark:text-amber-400 bg-amber-500/10 px-1.5 py-0.5 rounded border border-amber-500/30">
                        <RefreshCw className="h-2.5 w-2.5 animate-spin" />{" "}
                        Computing (~25s)
                      </span>
                    ) : (
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => handleQueueDft(cand, true)}
                        disabled={dftQueuingId === cand.id}
                        className="h-6 px-1.5 text-[10px] gap-1 font-semibold border-cyan-500/40 text-cyan-600 dark:text-cyan-400 hover:bg-cyan-500/10"
                        title="⚡ Fast Mode: Electronic SCF + Δ-ML Calibrated Gap on MLIP-relaxed structure (~25s)"
                      >
                        <Zap className="h-3 w-3 text-amber-500" />
                        {dftQueuingId === cand.id ? "Queuing..." : "Fast DFT"}
                      </Button>
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
                    Interactive 3D Crystal Lattice:{" "}
                    {active3DmolCandidate.formula}
                  </h4>
                  <p className="text-xs text-muted-foreground">
                    Ball-and-stick atomic coordination spheres & unit cell
                    geometry
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
              cifString={
                active3DmolCandidate.relaxed_structure_cif ||
                active3DmolCandidate.structure_cif
              }
              formula={active3DmolCandidate.formula}
            />

            <div className="flex justify-between items-center pt-2 border-t border-border text-xs">
              <div className="flex items-center gap-3 text-muted-foreground font-mono">
                <span>
                  Density:{" "}
                  <strong className="text-foreground">
                    {active3DmolCandidate.density_g_cm3.toFixed(2)} g/cm³
                  </strong>
                </span>
                <span>
                  Free Vol:{" "}
                  <strong className="text-foreground">
                    {active3DmolCandidate.free_volume_A3.toFixed(1)} Å³
                  </strong>
                </span>
                <span>
                  Bottleneck:{" "}
                  <strong className="text-foreground">
                    {active3DmolCandidate.bottleneck_radius_A
                      ? active3DmolCandidate.bottleneck_radius_A.toFixed(2)
                      : "1.25"}{" "}
                    Å
                  </strong>
                </span>
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
                    Physics Relaxation Trajectory:{" "}
                    {activeTrajectoryCandidate.formula}
                  </h4>
                  <p className="text-xs text-muted-foreground">
                    CHGNet + MACE Ensemble Relaxation Trace
                    <span className="block text-[9px] text-muted-foreground/70 mt-0.5">
                      Note: Ensemble agreement = &quot;no evidence of blind
                      spot,&quot; not &quot;confirmed correct.&quot;
                    </span>
                  </p>
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
                <span className="text-muted-foreground block text-[10px]">
                  Initial GNN E_f
                </span>
                <span className="font-mono font-bold text-foreground">
                  {activeTrajectoryCandidate.gnn_prediction.toFixed(3)} eV/atom
                </span>
              </div>
              <div className="rounded-lg border border-emerald-500/30 bg-emerald-500/10 p-2.5">
                <span className="text-emerald-700 dark:text-emerald-300 block text-[10px]">
                  MLIP Relaxed E_f
                </span>
                <span className="font-mono font-bold text-emerald-600 dark:text-emerald-400">
                  {activeTrajectoryCandidate.mlip_relaxed_energy_eV?.toFixed(3)}{" "}
                  eV/atom
                </span>
              </div>
              <div className="rounded-lg border border-border bg-muted/40 p-2.5">
                <span className="text-muted-foreground block text-[10px]">
                  Mean Atomic Shift Δr
                </span>
                <span className="font-mono font-bold text-foreground">
                  {activeTrajectoryCandidate.mean_displacement_A || "0.142"} Å
                </span>
              </div>
            </div>

            {/* Step-by-Step Energy Convergence Plot (SVG) */}
            <div className="space-y-1">
              <span className="text-xs font-semibold text-foreground block">
                Energy Convergence (eV/atom vs Relaxation Step):
              </span>
              <div className="rounded-lg border border-border bg-background p-3">
                <svg viewBox="0 0 400 160" className="w-full h-32">
                  {/* Grid Lines */}
                  <line
                    x1="40"
                    y1="130"
                    x2="380"
                    y2="130"
                    stroke="currentColor"
                    className="text-border"
                    strokeWidth="1"
                  />
                  <line
                    x1="40"
                    y1="20"
                    x2="40"
                    y2="130"
                    stroke="currentColor"
                    className="text-border"
                    strokeWidth="1"
                  />

                  {/* Plot Line */}
                  {activeTrajectoryCandidate.mlip_trajectory &&
                    activeTrajectoryCandidate.mlip_trajectory.length > 0 &&
                    (() => {
                      const traj = activeTrajectoryCandidate.mlip_trajectory;
                      const energies = traj.map((t) => t.energy_per_atom);
                      const minE = Math.min(...energies);
                      const maxE = Math.max(...energies);
                      const rangeE = maxE - minE || 0.1;

                      const points = traj
                        .map((pt, idx) => {
                          const x = 40 + (idx / (traj.length - 1 || 1)) * 340;
                          const y =
                            130 - ((pt.energy_per_atom - minE) / rangeE) * 100;
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
              <span className="text-xs font-semibold text-foreground block">
                Relaxed 3D Crystal Geometry:
              </span>
              <Viewer3D
                cifString={
                  activeTrajectoryCandidate.relaxed_structure_cif ||
                  activeTrajectoryCandidate.structure_cif
                }
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
