"use client";

import { useState, useEffect, Suspense } from "react";
import { useSearchParams } from "next/navigation";
import { Box, Play, Activity, Thermometer, Zap, Cpu, Bot, Send, Sparkles, RefreshCw, FileText, Search, Database, CheckCircle2, ArrowRight, ShieldCheck, AlertTriangle, Layers } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Textarea } from "@/components/ui/textarea";
import { PageHeader } from "@/components/layout/PageHeader";
import { Viewer3D } from "@/components/Viewer3D";

const API_BASE_URL = "http://127.0.0.1:8000";

function VirtualLabContent() {
  const searchParams = useSearchParams();
  const initialSource = searchParams.get("source") || "cif";
  const initialRef = searchParams.get("ref") || "";

  const [cifText, setCifText] = useState("");
  const [activeFormula, setActiveFormula] = useState("Loaded Crystal");
  const [activeCandidateId, setActiveCandidateId] = useState<number | null>(null);

  // Search / Load state
  const [mpQuery, setMpQuery] = useState("TiO2");
  const [isLoadingStructure, setIsLoadingStructure] = useState(false);

  // Test Selection & Sweeps
  const [selectedTest, setSelectedTest] = useState<"strain" | "neb" | "nvt" | "mutation" | "phonon">("strain");
  const [isRunningTest, setIsRunningTest] = useState(false);
  const [simResult, setSimResult] = useState<any>(null);
  const [trajectoryFrames, setTrajectoryFrames] = useState<string[]>([]);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  // Test Parameters
  const [strainAxis, setStrainAxis] = useState("a");
  const [mobileIon, setMobileIon] = useState("Li");
  const [temperatureK, setTemperatureK] = useState(600);
  const [mutationTargetElement, setMutationTargetElement] = useState("Co");
  const [mutationMatrix, setMutationMatrix] = useState<any>(null);

  // Write-Back State
  const [isWritingBack, setIsWritingBack] = useState(false);
  const [writeBackSuccess, setWriteBackSuccess] = useState<string | null>(null);

  // Tool-Grounded AI Assistant Chat State
  const [chatMessages, setChatMessages] = useState<Array<{ sender: "user" | "ai"; text: string; toolGrounded?: boolean }>>([
    {
      sender: "ai",
      text: "Welcome to the Virtual Lab! I am your Tool-Grounded AI Scientist. Every numeric prediction or physical barrier I cite is retrieved directly from tool calls to our PyTorch GNN/MLIP physics engine.",
      toolGrounded: true
    }
  ]);
  const [inputMessage, setInputMessage] = useState("");
  const [isAiThinking, setIsAiThinking] = useState(false);

  // Load structure on initial query params
  useEffect(() => {
    if (initialSource && initialRef) {
      handleLoadStructure(initialSource, initialRef);
    } else {
      handleLoadStructure("mp", "LiCoO2");
    }
  }, [initialSource, initialRef]);

  const handleLoadStructure = async (source: string, ref: string) => {
    setIsLoadingStructure(true);
    setErrorMsg(null);
    setSimResult(null);
    setTrajectoryFrames([]);
    setWriteBackSuccess(null);

    try {
      const res = await fetch(`${API_BASE_URL}/api/simulation/load?source=${source}&ref=${encodeURIComponent(ref)}`);
      if (!res.ok) {
        throw new Error("Failed to load crystal structure.");
      }
      const data = await res.json();
      setCifText(data.cif_text);
      setActiveFormula(data.formula);
      if (data.candidate_id) {
        setActiveCandidateId(data.candidate_id);
      }
    } catch (err: any) {
      setErrorMsg(err.message || "Failed to load structure.");
    } finally {
      setIsLoadingStructure(false);
    }
  };

  const handleRunSimulation = async () => {
    setIsRunningTest(true);
    setErrorMsg(null);
    setWriteBackSuccess(null);

    let endpoint = "/api/simulation/strain-sweep";
    let body: any = { cif_text: cifText };

    if (selectedTest === "strain") {
      endpoint = "/api/simulation/strain-sweep";
      body.axis = strainAxis;
    } else if (selectedTest === "neb") {
      endpoint = "/api/simulation/neb-barrier";
      body.mobile_ion = mobileIon;
      body.n_images = 5;
    } else if (selectedTest === "nvt") {
      endpoint = "/api/simulation/nvt-md";
      body.temperature_K = temperatureK;
      body.n_steps = 50;
    } else if (selectedTest === "phonon") {
      endpoint = "/api/simulation/phonon-check";
    }

    try {
      const res = await fetch(`${API_BASE_URL}${endpoint}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body)
      });

      if (!res.ok) {
        const errData = await res.json();
        throw new Error(errData.detail || "Simulation execution failed.");
      }

      const data = await res.json();
      setSimResult(data);
      setTrajectoryFrames(data.simulation_result?.trajectory_frames || []);

      setChatMessages((prev) => [
        ...prev,
        {
          sender: "ai",
          text: `Executed **${data.simulation_result?.test_name}** on **${data.material_info?.formula_pretty}**. Results updated with full 3D trajectory animation!`,
          toolGrounded: true
        }
      ]);
    } catch (err: any) {
      setErrorMsg(err.message || "Simulation execution error.");
    } finally {
      setIsRunningTest(false);
    }
  };

  const handleFetchMutationMatrix = async () => {
    try {
      const res = await fetch(`${API_BASE_URL}/api/simulation/mutation-matrix?target_element=${mutationTargetElement}&cif_text=${encodeURIComponent(cifText)}`);
      if (res.ok) {
        const data = await res.json();
        setMutationMatrix(data);
      }
    } catch (err) {
      console.error("Mutation matrix error:", err);
    }
  };

  const handleWriteBackToDiscovery = async () => {
    if (!activeCandidateId || !simResult) return;

    setIsWritingBack(true);
    try {
      const res = await fetch(`${API_BASE_URL}/api/simulation/writeback`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          candidate_id: activeCandidateId,
          test_results: simResult.simulation_result
        })
      });

      if (res.ok) {
        const data = await res.json();
        setWriteBackSuccess(data.message);
      }
    } catch (err: any) {
      console.error("Write back error:", err);
    } finally {
      setIsWritingBack(false);
    }
  };

  const handleSendChatMessage = async () => {
    if (!inputMessage.trim()) return;

    const userText = inputMessage;
    setInputMessage("");
    setChatMessages((prev) => [...prev, { sender: "user", text: userText }]);
    setIsAiThinking(true);

    try {
      const res = await fetch(`${API_BASE_URL}/api/simulation/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          user_message: userText,
          cif_text: cifText,
          last_test_result: simResult
        })
      });

      if (res.ok) {
        const data = await res.json();
        setChatMessages((prev) => [...prev, { sender: "ai", text: data.reply, toolGrounded: data.tool_grounded }]);
      }
    } catch (err) {
      console.error("AI chat error:", err);
    } finally {
      setIsAiThinking(false);
    }
  };

  return (
    <div className="space-y-8">
      <PageHeader
        title="Interactive Physics Simulation & Feasibility Tiered Lab"
        description="Run strain sweeps, NEB ion migration barriers, NVT MD annealing, Phonon dynamical stability checks, and charge-gated mutations."
      />

      {/* Universal Structure Loading Bar */}
      <Card className="p-4 bg-card border-border space-y-3">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
          <div className="flex items-center gap-2">
            <Database className="h-4 w-4 text-primary shrink-0" />
            <span className="text-xs font-bold text-foreground">Load Crystal Structure Source:</span>
          </div>

          <div className="flex items-center gap-2 flex-1 max-w-md">
            <input
              type="text"
              value={mpQuery}
              onChange={(e) => setMpQuery(e.target.value)}
              placeholder="Enter Formula (e.g. TiO2, LiFePO4) or MP-ID..."
              className="flex-1 rounded-lg border border-border bg-background px-3 py-1.5 text-xs font-mono"
            />
            <Button
              size="sm"
              onClick={() => handleLoadStructure("mp", mpQuery)}
              disabled={isLoadingStructure}
              className="text-xs gap-1.5 font-bold"
            >
              <Search className="h-3.5 w-3.5" /> Load MP
            </Button>
          </div>

          {activeCandidateId && (
            <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 text-xs font-bold">
              <Sparkles className="h-3.5 w-3.5" /> Loaded from Discovery Candidate #{activeCandidateId}
            </div>
          )}
        </div>
      </Card>

      {/* Explicit Feasibility Tier System Badge Bar */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <Card className="p-3.5 border-emerald-500/30 bg-emerald-500/5 space-y-1">
          <div className="flex items-center justify-between">
            <span className="text-xs font-bold text-emerald-600 dark:text-emerald-400 flex items-center gap-1.5">
              <CheckCircle2 className="h-3.5 w-3.5" /> Tier A — Fast MLIP (Seconds)
            </span>
            <span className="text-[10px] font-mono uppercase bg-emerald-500/20 text-emerald-700 dark:text-emerald-300 px-1.5 py-0.5 rounded font-bold">MLIP Ready</span>
          </div>
          <p className="text-[11px] text-muted-foreground">Formation Energy, Elastic Tensor, Hardness, Phonon Dynamical Stability (Tool #6)</p>
        </Card>

        <Card className="p-3.5 border-blue-500/30 bg-blue-500/5 space-y-1">
          <div className="flex items-center justify-between">
            <span className="text-xs font-bold text-blue-600 dark:text-blue-400 flex items-center gap-1.5">
              <Activity className="h-3.5 w-3.5" /> Tier B — MLIP MD / Sweeps (Minutes)
            </span>
            <span className="text-[10px] font-mono uppercase bg-blue-500/20 text-blue-700 dark:text-blue-300 px-1.5 py-0.5 rounded font-bold">MLIP Sweeps</span>
          </div>
          <p className="text-[11px] text-muted-foreground">NEB Ion Migration Path, Surface Energy, NVT Molecular Dynamics Annealing, Defect Supercell</p>
        </Card>

        <Card className="p-3.5 border-amber-500/30 bg-amber-500/5 space-y-1">
          <div className="flex items-center justify-between">
            <span className="text-xs font-bold text-amber-600 dark:text-amber-400 flex items-center gap-1.5">
              <AlertTriangle className="h-3.5 w-3.5" /> Tier C — DFT Required (External)
            </span>
            <span className="text-[10px] font-mono uppercase bg-amber-500/20 text-amber-700 dark:text-amber-300 px-1.5 py-0.5 rounded font-bold">Needs DFT</span>
          </div>
          <p className="text-[11px] text-muted-foreground">Band Gap, Dielectric Constant, Carrier Mobility, Optical Spectra (Explicitly labeled uncalculated)</p>
        </Card>
      </div>

      {writeBackSuccess && (
        <div className="rounded-xl border border-emerald-500/30 bg-emerald-500/10 p-4 text-xs text-emerald-700 dark:text-emerald-300 font-bold flex items-center justify-between">
          <div className="flex items-center gap-2">
            <CheckCircle2 className="h-4 w-4 text-emerald-500" />
            <span>{writeBackSuccess}</span>
          </div>
          <span className="font-mono text-[11px] uppercase">Tier 3 Validated</span>
        </div>
      )}

      {errorMsg && (
        <div className="rounded-xl border border-rose-500/30 bg-rose-500/10 p-4 text-xs text-rose-600 dark:text-rose-400 font-semibold">
          {errorMsg}
        </div>
      )}

      {/* Main Grid: Left Structure & Editor, Right Simulation Suite */}
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-8">
        {/* Left Column: 3D Crystal Viewer with Trajectory Scrubber */}
        <div className="space-y-6">
          <Viewer3D
            cifString={cifText}
            formula={activeFormula}
            trajectoryFrames={trajectoryFrames}
          />

          {/* Tier C Uncalculated Properties Panel */}
          <Card className="p-4 border-amber-500/30 bg-amber-500/5 space-y-2 text-xs">
            <div className="flex items-center justify-between text-amber-700 dark:text-amber-300 font-bold">
              <span className="flex items-center gap-1.5">
                <Layers className="h-4 w-4" /> Tier C Electronic Structure Properties:
              </span>
              <span className="text-[10px] font-mono uppercase">Requires DFT</span>
            </div>
            <div className="grid grid-cols-3 gap-2 text-[11px]">
              <div className="rounded-md border border-amber-500/20 bg-background p-2">
                <span className="text-muted-foreground block text-[10px]">Band Gap:</span>
                <span className="font-bold italic text-amber-600 dark:text-amber-400">Requires DFT Job</span>
              </div>
              <div className="rounded-md border border-amber-500/20 bg-background p-2">
                <span className="text-muted-foreground block text-[10px]">Dielectric Constant:</span>
                <span className="font-bold italic text-amber-600 dark:text-amber-400">Requires DFPT</span>
              </div>
              <div className="rounded-md border border-amber-500/20 bg-background p-2">
                <span className="text-muted-foreground block text-[10px]">Carrier Mobility:</span>
                <span className="font-bold italic text-amber-600 dark:text-amber-400">Requires DFT Band</span>
              </div>
            </div>
          </Card>

          <Card className="p-5 space-y-3">
            <div className="flex items-center justify-between">
              <label htmlFor="cif-editor" className="text-sm font-semibold text-foreground flex items-center gap-1.5">
                <FileText className="h-4 w-4 text-primary" /> CIF Crystallographic Text Editor
              </label>
              <span className="text-[10px] text-muted-foreground font-mono">P1 / Spacegroup CIF</span>
            </div>
            <Textarea
              id="cif-editor"
              rows={6}
              value={cifText}
              onChange={(e) => setCifText(e.target.value)}
              className="font-mono text-xs leading-relaxed"
            />
          </Card>
        </div>

        {/* Right Column: Parameter Sweeps, NEB, NVT MD & Controls */}
        <div className="space-y-6">
          <Card className="p-6 space-y-6">
            <div className="flex items-center justify-between border-b border-border pb-3">
              <span className="text-base font-bold text-foreground flex items-center gap-2">
                <Activity className="h-5 w-5 text-primary" /> Parameter Sweeps & Physics Probes
              </span>
              <Button onClick={handleRunSimulation} disabled={isRunningTest} className="gap-2 font-bold">
                <Play className={`h-4 w-4 ${isRunningTest ? "animate-spin" : ""}`} />
                {isRunningTest ? "Running Physics..." : "Execute Simulation"}
              </Button>
            </div>

            {/* Test Selection Tabs */}
            <div className="grid grid-cols-2 xl:grid-cols-5 gap-1.5">
              <button
                onClick={() => setSelectedTest("strain")}
                className={`flex flex-col items-center justify-center p-2.5 rounded-xl border text-[11px] font-semibold transition-colors ${
                  selectedTest === "strain" ? "border-primary bg-primary/10 text-primary" : "border-border bg-card hover:bg-muted text-muted-foreground"
                }`}
              >
                <Cpu className="h-4 w-4 mb-1" /> Strain Sweep
                <span className="text-[9px] font-mono text-emerald-600 dark:text-emerald-400">Tier A</span>
              </button>
              <button
                onClick={() => setSelectedTest("phonon")}
                className={`flex flex-col items-center justify-center p-2.5 rounded-xl border text-[11px] font-semibold transition-colors ${
                  selectedTest === "phonon" ? "border-primary bg-primary/10 text-primary" : "border-border bg-card hover:bg-muted text-muted-foreground"
                }`}
              >
                <Activity className="h-4 w-4 mb-1 text-emerald-500" /> Tool #6 Phonon
                <span className="text-[9px] font-mono text-emerald-600 dark:text-emerald-400">Tier A</span>
              </button>
              <button
                onClick={() => setSelectedTest("neb")}
                className={`flex flex-col items-center justify-center p-2.5 rounded-xl border text-[11px] font-semibold transition-colors ${
                  selectedTest === "neb" ? "border-primary bg-primary/10 text-primary" : "border-border bg-card hover:bg-muted text-muted-foreground"
                }`}
              >
                <Zap className="h-4 w-4 mb-1" /> NEB Barrier
                <span className="text-[9px] font-mono text-blue-600 dark:text-blue-400">Tier B</span>
              </button>
              <button
                onClick={() => setSelectedTest("nvt")}
                className={`flex flex-col items-center justify-center p-2.5 rounded-xl border text-[11px] font-semibold transition-colors ${
                  selectedTest === "nvt" ? "border-primary bg-primary/10 text-primary" : "border-border bg-card hover:bg-muted text-muted-foreground"
                }`}
              >
                <Thermometer className="h-4 w-4 mb-1" /> NVT MD Anneal
                <span className="text-[9px] font-mono text-blue-600 dark:text-blue-400">Tier B</span>
              </button>
              <button
                onClick={() => {
                  setSelectedTest("mutation");
                  handleFetchMutationMatrix();
                }}
                className={`flex flex-col items-center justify-center p-2.5 rounded-xl border text-[11px] font-semibold transition-colors ${
                  selectedTest === "mutation" ? "border-primary bg-primary/10 text-primary" : "border-border bg-card hover:bg-muted text-muted-foreground"
                }`}
              >
                <Sparkles className="h-4 w-4 mb-1" /> Gated Mutation
                <span className="text-[9px] font-mono text-emerald-600 dark:text-emerald-400">Tier A</span>
              </button>
            </div>

            {/* Parameter Controls */}
            <div className="rounded-xl border border-border bg-muted/30 p-4 space-y-4 text-xs">
              {selectedTest === "strain" && (
                <div className="space-y-2">
                  <label className="font-semibold text-foreground block">Lattice Perturbation Axis:</label>
                  <select
                    value={strainAxis}
                    onChange={(e) => setStrainAxis(e.target.value)}
                    className="w-full rounded-lg border border-border bg-background p-2 font-semibold text-xs"
                  >
                    <option value="a">a-axis (-5% to +5% sweep)</option>
                    <option value="b">b-axis (-5% to +5% sweep)</option>
                    <option value="c">c-axis (-5% to +5% sweep)</option>
                  </select>
                  <p className="text-[11px] text-muted-foreground">Sweeps strain values and fits parabolic energy curvature to derive Bulk Elastic Stiffness Modulus (GPa).</p>
                </div>
              )}

              {selectedTest === "phonon" && (
                <div className="space-y-2">
                  <span className="font-bold text-emerald-600 dark:text-emerald-400 block">Tool #6: Phonon Dynamical Stability Check (Tier A)</span>
                  <p className="text-[11px] text-muted-foreground">Calculates finite-difference mass-weighted dynamical matrix eigenvalues to detect imaginary (negative) frequency saddle-point instabilities.</p>
                </div>
              )}

              {selectedTest === "neb" && (
                <div className="space-y-2">
                  <label className="font-semibold text-foreground block">Target Mobile Ion Species:</label>
                  <select
                    value={mobileIon}
                    onChange={(e) => setMobileIon(e.target.value)}
                    className="w-full rounded-lg border border-border bg-background p-2 text-xs font-semibold"
                  >
                    <option value="Li">Lithium (Li+)</option>
                    <option value="Na">Sodium (Na+)</option>
                    <option value="Mg">Magnesium (Mg2+)</option>
                    <option value="K">Potassium (K+)</option>
                  </select>
                  <p className="text-[11px] text-muted-foreground">Interpolates 5 intermediate image frames and computes Nudged Elastic Band activation energy barrier Ea (eV).</p>
                </div>
              )}

              {selectedTest === "nvt" && (
                <div className="space-y-2">
                  <label className="font-semibold text-foreground block">Target Annealing Temperature: {temperatureK} K ({temperatureK - 273} °C)</label>
                  <input
                    type="range"
                    min={300}
                    max={1200}
                    step={50}
                    value={temperatureK}
                    onChange={(e) => setTemperatureK(Number(e.target.value))}
                    className="w-full accent-primary"
                  />
                  <p className="text-[11px] text-muted-foreground">Runs 50-step Langevin NVT Molecular Dynamics trajectory tracking energy drift over time.</p>
                </div>
              )}

              {selectedTest === "mutation" && mutationMatrix && (
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <label className="font-semibold text-foreground block">Charge-Gated Element Mutation Matrix:</label>
                    <span className="text-[10px] text-muted-foreground">oxi_state_guesses Gating</span>
                  </div>

                  <div className="space-y-2">
                    <span className="text-[10px] text-emerald-600 dark:text-emerald-400 font-bold block">Charge-Balanced Substitutions (Click to Swap):</span>
                    <div className="flex flex-wrap gap-1.5">
                      {mutationMatrix.valid_substitutions?.map((item: any) => (
                        <button
                          key={item.element}
                          onClick={() => setCifText(cifText.replaceAll(mutationTargetElement, item.element))}
                          className="px-2.5 py-1 rounded-md bg-emerald-500/15 border border-emerald-500/30 text-emerald-700 dark:text-emerald-300 font-bold text-xs hover:bg-emerald-500/25 transition-colors"
                        >
                          {item.element}
                        </button>
                      ))}
                    </div>
                  </div>

                  <div className="space-y-2">
                    <span className="text-[10px] text-rose-500 font-bold block">Unphysical / Charge-Imbalanced Substitutions:</span>
                    <div className="flex flex-wrap gap-1.5">
                      {mutationMatrix.invalid_substitutions?.map((item: any) => (
                        <span
                          key={item.element}
                          title={item.reason}
                          className="px-2 py-1 rounded-md bg-muted text-muted-foreground/60 text-xs border border-border line-through cursor-not-allowed"
                        >
                          {item.element}
                        </span>
                      ))}
                    </div>
                  </div>
                </div>
              )}
            </div>

            {/* Simulation Test Results & Plot Panel */}
            {simResult && (
              <div className="rounded-xl border border-primary/30 bg-primary/5 p-4 space-y-4 animate-fade-up">
                <div className="flex items-center justify-between font-bold text-xs text-primary border-b border-primary/20 pb-2">
                  <span>{simResult.simulation_result?.test_name}</span>
                  <span className="px-2 py-0.5 rounded bg-primary/10 text-[10px] font-mono">{simResult.simulation_result?.tier || "Tier A"}</span>
                </div>

                {/* Phonon Bandstructure DOS Spectrum Plot */}
                {simResult.simulation_result?.phonon_dos_spectrum && (
                  <div className="space-y-2">
                    <div className="flex items-center justify-between text-xs">
                      <span className="font-semibold text-foreground">Phonon Density of States (DOS) Spectrum:</span>
                      <span className={`font-mono font-bold ${simResult.simulation_result.is_dynamically_stable ? "text-emerald-500" : "text-rose-500"}`}>
                        {simResult.simulation_result.stability_status}
                      </span>
                    </div>

                    <div className="rounded-lg border border-border bg-background p-3">
                      <svg viewBox="0 0 400 140" className="w-full h-32">
                        <line x1="30" y1="110" x2="380" y2="110" stroke="currentColor" className="text-border" strokeWidth="1" />
                        <line x1="30" y1="10" x2="30" y2="110" stroke="currentColor" className="text-border" strokeWidth="1" />
                        {(() => {
                          const pts = simResult.simulation_result.phonon_dos_spectrum;
                          const freqs = pts.map((p: any) => p.frequency_THz);
                          const minF = Math.min(...freqs);
                          const maxF = Math.max(...freqs);
                          const rangeF = maxF - minF || 1.0;
                          const maxDos = Math.max(...pts.map((p: any) => p.dos_density)) || 1;

                          const polyPoints = pts.map((pt: any, idx: number) => {
                            const x = 30 + ((pt.frequency_THz - minF) / rangeF) * 350;
                            const y = 110 - (pt.dos_density / maxDos) * 90;
                            return `${x},${y}`;
                          }).join(" ");

                          return <polyline fill="none" stroke={simResult.simulation_result.is_dynamically_stable ? "#10b981" : "#ef4444"} strokeWidth="2.5" points={polyPoints} />;
                        })()}
                      </svg>
                    </div>
                  </div>
                )}

                {/* Strain Sweep Plot */}
                {simResult.simulation_result?.sweep_points && (
                  <div className="space-y-2">
                    <div className="flex items-center justify-between text-xs">
                      <span className="font-semibold text-foreground">Stress-Strain Energy Curve (E vs Strain):</span>
                      <span className="font-mono font-bold text-primary">K = {simResult.simulation_result.elastic_modulus_GPa} GPa ({simResult.simulation_result.framework_stiffness})</span>
                    </div>

                    <div className="rounded-lg border border-border bg-background p-3">
                      <svg viewBox="0 0 400 140" className="w-full h-32">
                        <line x1="30" y1="110" x2="380" y2="110" stroke="currentColor" className="text-border" strokeWidth="1" />
                        <line x1="30" y1="10" x2="30" y2="110" stroke="currentColor" className="text-border" strokeWidth="1" />
                        {(() => {
                          const pts = simResult.simulation_result.sweep_points;
                          const energies = pts.map((p: any) => p.energy_per_atom);
                          const minE = Math.min(...energies);
                          const maxE = Math.max(...energies);
                          const rangeE = maxE - minE || 0.1;

                          const polyPoints = pts.map((pt: any, idx: number) => {
                            const x = 30 + (idx / (pts.length - 1)) * 350;
                            const y = 110 - ((pt.energy_per_atom - minE) / rangeE) * 90;
                            return `${x},${y}`;
                          }).join(" ");

                          return <polyline fill="none" stroke="#3b82f6" strokeWidth="2.5" points={polyPoints} />;
                        })()}
                      </svg>
                    </div>
                  </div>
                )}

                {/* NEB Barrier Profile */}
                {simResult.simulation_result?.images_profile && (
                  <div className="space-y-2">
                    <div className="flex items-center justify-between text-xs">
                      <span className="font-semibold text-foreground">NEB Ion Migration Barrier Profile:</span>
                      <span className="font-mono font-bold text-emerald-600 dark:text-emerald-400">Ea = {simResult.simulation_result.activation_barrier_eV} eV</span>
                    </div>

                    <div className="rounded-lg border border-border bg-background p-3">
                      <svg viewBox="0 0 400 140" className="w-full h-32">
                        <line x1="30" y1="110" x2="380" y2="110" stroke="currentColor" className="text-border" strokeWidth="1" />
                        <line x1="30" y1="10" x2="30" y2="110" stroke="currentColor" className="text-border" strokeWidth="1" />
                        {(() => {
                          const pts = simResult.simulation_result.images_profile;
                          const energies = pts.map((p: any) => p.relative_barrier_eV);
                          const maxE = Math.max(...energies) || 0.3;

                          const polyPoints = pts.map((pt: any, idx: number) => {
                            const x = 30 + (idx / (pts.length - 1)) * 350;
                            const y = 110 - (pt.relative_barrier_eV / maxE) * 90;
                            return `${x},${y}`;
                          }).join(" ");

                          return <polyline fill="none" stroke="#10b981" strokeWidth="2.5" points={polyPoints} />;
                        })()}
                      </svg>
                    </div>
                  </div>
                )}

                {/* Write-Back to Discovery Button */}
                {activeCandidateId && (
                  <div className="pt-2 border-t border-primary/20 flex justify-end">
                    <Button
                      size="sm"
                      onClick={handleWriteBackToDiscovery}
                      disabled={isWritingBack}
                      className="text-xs gap-1.5 bg-emerald-600 hover:bg-emerald-700 font-bold"
                    >
                      <CheckCircle2 className="h-3.5 w-3.5" />
                      {isWritingBack ? "Writing Back..." : "Push Validation to Discovery (Tier 3)"}
                    </Button>
                  </div>
                )}
              </div>
            )}
          </Card>

          {/* Tool-Grounded AI Assistant Chat Box */}
          <Card className="p-5 space-y-4">
            <div className="flex items-center justify-between border-b border-border pb-3">
              <div className="flex items-center gap-2">
                <Bot className="h-5 w-5 text-primary" />
                <div>
                  <h4 className="text-sm font-bold text-foreground">Tool-Grounded AI Lab Scientist</h4>
                  <p className="text-[11px] text-muted-foreground">Queries GNN predictor & physics simulation tools directly</p>
                </div>
              </div>
              <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full bg-primary/10 text-primary text-[10px] font-bold">
                <ShieldCheck className="h-3 w-3" /> Tool Grounded
              </span>
            </div>

            <div className="h-44 overflow-y-auto space-y-3 pr-2 text-xs">
              {chatMessages.map((msg, idx) => (
                <div
                  key={idx}
                  className={`p-3 rounded-xl max-w-[90%] leading-relaxed ${
                    msg.sender === "user"
                      ? "ml-auto bg-primary text-primary-foreground font-medium"
                      : "mr-auto bg-muted/60 text-foreground border border-border space-y-1"
                  }`}
                >
                  <p>{msg.text}</p>
                  {msg.toolGrounded && msg.sender === "ai" && (
                    <span className="text-[9px] text-muted-foreground font-mono block">✓ Verified via GNN/MLIP Tool Execution</span>
                  )}
                </div>
              ))}
              {isAiThinking && (
                <div className="mr-auto bg-muted/40 p-2.5 rounded-xl text-xs text-muted-foreground animate-pulse">
                  Executing GNN Tool Call & Retrieving Calibrated Physics Outputs...
                </div>
              )}
            </div>

            <div className="flex items-center gap-2 pt-2 border-t border-border">
              <input
                type="text"
                value={inputMessage}
                onChange={(e) => setInputMessage(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && handleSendChatMessage()}
                placeholder="Ask Tool-Grounded AI (e.g. Is this candidate formation energy stable?)..."
                className="flex-1 rounded-lg border border-border bg-background p-2 text-xs"
              />
              <Button size="sm" onClick={handleSendChatMessage} disabled={isAiThinking} className="h-9 px-3">
                <Send className="h-3.5 w-3.5" />
              </Button>
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
}

export default function VirtualLabPage() {
  return (
    <Suspense fallback={<div className="p-8 text-center text-xs text-muted-foreground">Loading Virtual Lab Sandbox...</div>}>
      <VirtualLabContent />
    </Suspense>
  );
}
