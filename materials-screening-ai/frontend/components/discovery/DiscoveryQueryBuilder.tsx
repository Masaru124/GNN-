"use client";

import { useState } from "react";
import { Sparkles, DollarSign, ShieldAlert, Cpu, Zap, MessageSquareText, ArrowRight, Check } from "lucide-react";
import { Button } from "@/components/ui/button";

const API_BASE_URL = "http://127.0.0.1:8000";

export interface DiscoveryQueryPayload {
  title?: string;
  scaffold: string;
  num_candidates: number;
  generation_mode: string;
  exclude_toxic: boolean;
  max_cost_usd_kg?: number;
  max_density_g_cm3?: number;
  target_ion: string;
  target_property_min?: number;
  target_property_max?: number;
}

interface ParsedConstraint {
  term: string;
  mapped_to: string;
}

interface Props {
  onSubmit: (query: DiscoveryQueryPayload) => void;
  isSubmitting: boolean;
}

const SAMPLE_PROMPTS = [
  "I need a lightweight battery cathode with conductivity > 1e-3, low toxicity, stable above 600°C and inexpensive",
  "Cheap non-toxic solid electrolyte with fast lithium transport",
  "High-stability perovskite solar absorber under $20/kg",
];

export function DiscoveryQueryBuilder({ onSubmit, isSubmitting }: Props) {
  const [promptText, setPromptText] = useState(SAMPLE_PROMPTS[0]);
  const [isParsing, setIsParsing] = useState(false);
  const [parsedConstraints, setParsedConstraints] = useState<ParsedConstraint[]>([]);

  const [title, setTitle] = useState("Lightweight Cathode Discovery Run");
  const [scaffold, setScaffold] = useState("layered_oxide");
  const [numCandidates, setNumCandidates] = useState(12);
  const [generationMode, setGenerationMode] = useState("substitution");
  const [excludeToxic, setExcludeToxic] = useState(true);
  const [maxCost, setMaxCost] = useState<string>("30");
  const [maxDensity, setMaxDensity] = useState<string>("4.5");
  const [targetIon, setTargetIon] = useState("Li");
  const [targetMaxEf, setTargetMaxEf] = useState<string>("-1.85");

  const handleParsePrompt = async (textToParse: string) => {
    if (!textToParse.trim()) return;
    setIsParsing(true);

    try {
      const res = await fetch(`${API_BASE_URL}/api/discovery/parse-prompt`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt: textToParse }),
      });

      if (res.ok) {
        const data = await res.json();
        if (data.scaffold) setScaffold(data.scaffold);
        if (data.target_ion) setTargetIon(data.target_ion);
        if (data.exclude_toxic !== undefined) setExcludeToxic(data.exclude_toxic);
        if (data.max_cost_usd_kg !== null && data.max_cost_usd_kg !== undefined) {
          setMaxCost(data.max_cost_usd_kg.toString());
        }
        if (data.max_density_g_cm3 !== null && data.max_density_g_cm3 !== undefined) {
          setMaxDensity(data.max_density_g_cm3.toString());
        }
        if (data.target_property_max !== null && data.target_property_max !== undefined) {
          setTargetMaxEf(data.target_property_max.toString());
        }
        if (data.title) setTitle(data.title);
        if (data.parsed_constraints) setParsedConstraints(data.parsed_constraints);
      }
    } catch (err) {
      console.error("Failed to parse prompt:", err);
    } finally {
      setIsParsing(false);
    }
  };

  const handleSelectSample = (sample: string) => {
    setPromptText(sample);
    handleParsePrompt(sample);
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSubmit({
      title,
      scaffold,
      num_candidates: numCandidates,
      generation_mode: generationMode,
      exclude_toxic: excludeToxic,
      max_cost_usd_kg: maxCost ? parseFloat(maxCost) : undefined,
      max_density_g_cm3: maxDensity ? parseFloat(maxDensity) : undefined,
      target_ion: targetIon,
      target_property_max: targetMaxEf ? parseFloat(targetMaxEf) : undefined,
    });
  };

  return (
    <div className="rounded-xl border border-border bg-card p-5 sm:p-6 shadow-sm space-y-5">
      {/* Header */}
      <div className="flex items-center gap-2.5 pb-4 border-b border-border">
        <Sparkles className="h-5 w-5 text-primary animate-pulse" />
        <div>
          <h2 className="text-lg font-bold tracking-tight text-foreground">Discovery Query Builder</h2>
          <p className="text-xs text-muted-foreground">Natural Language AI Specification & Constraint Controls</p>
        </div>
      </div>

      {/* AI Natural Language Prompt Input Bar */}
      <div className="rounded-xl border border-primary/30 bg-primary/5 p-4 space-y-3">
        <label className="flex items-center gap-1.5 text-xs font-bold text-foreground">
          <MessageSquareText className="h-4 w-4 text-primary" />
          Natural Language Specification Prompt
        </label>

        <div className="flex gap-2">
          <textarea
            rows={2}
            value={promptText}
            onChange={(e) => setPromptText(e.target.value)}
            placeholder="Describe your material in plain English e.g. I need a lightweight battery cathode with conductivity > 1e-3..."
            className="flex-1 rounded-lg border border-border bg-background px-3 py-2 text-xs text-foreground focus:outline-none focus:ring-2 focus:ring-primary/50 resize-none font-sans"
          />
          <Button
            type="button"
            size="sm"
            onClick={() => handleParsePrompt(promptText)}
            disabled={isParsing}
            className="self-end text-xs font-semibold gap-1 shrink-0"
          >
            {isParsing ? "Parsing..." : "Parse AI Constraints"}
          </Button>
        </div>

        {/* Sample Prompt Chips */}
        <div className="space-y-1.5 pt-1">
          <span className="text-[11px] font-semibold text-muted-foreground block">Preset Natural Language Prompts:</span>
          <div className="flex flex-wrap gap-1.5">
            {SAMPLE_PROMPTS.map((sample, idx) => (
              <button
                key={idx}
                type="button"
                onClick={() => handleSelectSample(sample)}
                className="text-left text-[11px] rounded-md border border-border/80 bg-background/80 px-2.5 py-1 text-muted-foreground hover:border-primary/50 hover:text-foreground transition-colors"
              >
                "{sample}"
              </button>
            ))}
          </div>
        </div>

        {/* Parsed Constraint Badges */}
        {parsedConstraints.length > 0 && (
          <div className="pt-2 border-t border-primary/20 space-y-1">
            <span className="text-[11px] font-bold text-primary flex items-center gap-1">
              <Check className="h-3.5 w-3.5" /> AI Parsed Constraints ({parsedConstraints.length}):
            </span>
            <div className="flex flex-wrap gap-1.5">
              {parsedConstraints.map((item, idx) => (
                <span
                  key={idx}
                  className="inline-flex items-center gap-1 rounded-md bg-emerald-500/10 border border-emerald-500/30 px-2 py-0.5 text-[10px] font-semibold text-emerald-700 dark:text-emerald-300"
                >
                  <span className="font-bold">{item.term}</span> → {item.mapped_to}
                </span>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Main Query Form */}
      <form onSubmit={handleSubmit} className="space-y-4">
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <div>
            <label className="block text-xs font-semibold text-muted-foreground mb-1">Run Title</label>
            <input
              type="text"
              value={title}
              onChange={(e) => setTitle(e.target.value)}
              className="w-full rounded-lg border border-border bg-background px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary/50"
              required
            />
          </div>

          <div>
            <label className="block text-xs font-semibold text-muted-foreground mb-1">Scaffold Family Archetype</label>
            <select
              value={scaffold}
              onChange={(e) => setScaffold(e.target.value)}
              className="w-full rounded-lg border border-border bg-background px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary/50"
            >
              <option value="layered_oxide">Layered Oxide AxMO2 (Trigonal Cathode)</option>
              <option value="perovskite">Perovskite ABX3 (Cubic/Orthorhombic)</option>
              <option value="spinel">Spinel AB2X4 (Cubic Cathode/Electrolyte)</option>
              <option value="olivine">Olivine AxMPO4 (Orthorhombic Cathode)</option>
            </select>
          </div>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
          <div>
            <label className="block text-xs font-semibold text-muted-foreground mb-1">Generation Engine</label>
            <select
              value={generationMode}
              onChange={(e) => setGenerationMode(e.target.value)}
              className="w-full rounded-lg border border-border bg-background px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary/50"
            >
              <option value="substitution">pymatgen Substitution</option>
              <option value="mattergen">MatterGen Conditioned AI</option>
            </select>
          </div>

          <div>
            <label className="block text-xs font-semibold text-muted-foreground mb-1">Candidate Batch Size</label>
            <input
              type="number"
              min={4}
              max={30}
              value={numCandidates}
              onChange={(e) => setNumCandidates(parseInt(e.target.value) || 12)}
              className="w-full rounded-lg border border-border bg-background px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary/50"
            />
          </div>

          <div>
            <label className="block text-xs font-semibold text-muted-foreground mb-1">Mobile Ion Species</label>
            <select
              value={targetIon}
              onChange={(e) => setTargetIon(e.target.value)}
              className="w-full rounded-lg border border-border bg-background px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary/50"
            >
              <option value="Li">Lithium (Li+)</option>
              <option value="Na">Sodium (Na+)</option>
              <option value="Mg">Magnesium (Mg2+)</option>
              <option value="K">Potassium (K+)</option>
            </select>
          </div>
        </div>

        {/* Hard Constraint Sliders / Thresholds */}
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 pt-2 border-t border-border/60">
          <div>
            <label className="flex items-center gap-1.5 text-xs font-semibold text-muted-foreground mb-1">
              <DollarSign className="h-3.5 w-3.5 text-amber-500" />
              Max Cost Limit ($/kg)
            </label>
            <input
              type="number"
              step="5"
              value={maxCost}
              onChange={(e) => setMaxCost(e.target.value)}
              className="w-full rounded-lg border border-border bg-background px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary/50 font-mono"
            />
          </div>

          <div>
            <label className="flex items-center gap-1.5 text-xs font-semibold text-muted-foreground mb-1">
              <Cpu className="h-3.5 w-3.5 text-blue-500" />
              Max Density (g/cm³)
            </label>
            <input
              type="number"
              step="0.5"
              value={maxDensity}
              onChange={(e) => setMaxDensity(e.target.value)}
              className="w-full rounded-lg border border-border bg-background px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary/50 font-mono"
            />
          </div>

          <div>
            <label className="flex items-center gap-1.5 text-xs font-semibold text-muted-foreground mb-1">
              <Zap className="h-3.5 w-3.5 text-emerald-500" />
              Target E_f Ceiling (eV/atom)
            </label>
            <input
              type="number"
              step="0.05"
              value={targetMaxEf}
              onChange={(e) => setTargetMaxEf(e.target.value)}
              className="w-full rounded-lg border border-border bg-background px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary/50 font-mono"
            />
          </div>
        </div>

        {/* Toxicity Exclusion Checkbox */}
        <div className="flex items-center justify-between rounded-lg border border-border/80 bg-muted/40 p-3">
          <div className="flex items-center gap-2.5">
            <ShieldAlert className="h-4 w-4 text-rose-500 shrink-0" />
            <div>
              <span className="text-xs font-semibold text-foreground block">RoHS Toxicity Exclusion Filter</span>
              <span className="text-[11px] text-muted-foreground block">Excludes hazardous elements (Pb, Cd, As, Hg, Tl, Be)</span>
            </div>
          </div>
          <input
            type="checkbox"
            checked={excludeToxic}
            onChange={(e) => setExcludeToxic(e.target.checked)}
            className="h-4 w-4 rounded border-border text-primary focus:ring-primary"
          />
        </div>

        {/* Submit Button */}
        <Button
          type="submit"
          disabled={isSubmitting}
          className="w-full font-bold gap-2 py-3 mt-2 text-sm"
        >
          <Sparkles className="h-4 w-4" />
          {isSubmitting ? "Generating & Screening Candidates..." : "Launch Autonomous Discovery Pipeline"}
        </Button>
      </form>
    </div>
  );
}
