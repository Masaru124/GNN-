"use client";

import { useState, useEffect } from "react";
import { Sparkles, RefreshCw, CheckCircle2, AlertCircle, Cpu, GitBranch, ShieldCheck } from "lucide-react";
import { DiscoveryQueryBuilder, DiscoveryQueryPayload } from "@/components/discovery/DiscoveryQueryBuilder";
import { ParetoFrontierChart, CandidateItem } from "@/components/discovery/ParetoFrontierChart";
import { CandidateTable } from "@/components/discovery/CandidateTable";
import { Button } from "@/components/ui/button";

const API_BASE_URL = "http://127.0.0.1:8000";

export default function DiscoveryPage() {
  const [jobId, setJobId] = useState<string | null>(null);
  const [runId, setRunId] = useState<number | null>(null);
  const [status, setStatus] = useState<string | null>(null);
  const [queryMeta, setQueryMeta] = useState<any>(null);
  const [candidates, setCandidates] = useState<CandidateItem[]>([]);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isValidating, setIsValidating] = useState(false);
  const [isRetraining, setIsRetraining] = useState(false);
  const [retrainResult, setRetrainResult] = useState<any>(null);
  const [retrainHistory, setRetrainHistory] = useState<any[]>([]);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  // Poll Discovery Job Status
  useEffect(() => {
    if (!jobId || status === "complete" || status === "failed") return;

    const interval = setInterval(async () => {
      try {
        const res = await fetch(`${API_BASE_URL}/api/discovery/query/${jobId}`);
        if (!res.ok) return;

        const data = await res.json();
        setStatus(data.status);
        setRunId(data.run_id);
        setQueryMeta(data.query);
        setCandidates(data.candidates || []);

        if (data.status === "complete" || data.status === "failed") {
          setIsSubmitting(false);
        }
      } catch (err) {
        console.error("Job polling error:", err);
      }
    }, 2000);

    return () => clearInterval(interval);
  }, [jobId, status]);

  // Fetch Retrain History
  const fetchRetrainHistory = async () => {
    try {
      const res = await fetch(`${API_BASE_URL}/api/discovery/retrain/history`);
      if (res.ok) {
        const history = await res.json();
        setRetrainHistory(history || []);
      }
    } catch (err) {
      console.error("Retrain history error:", err);
    }
  };

  useEffect(() => {
    fetchRetrainHistory();
  }, []);

  const handleRunDiscovery = async (query: DiscoveryQueryPayload) => {
    setIsSubmitting(true);
    setErrorMsg(null);
    setStatus("queued");
    setCandidates([]);

    try {
      const res = await fetch(`${API_BASE_URL}/api/discovery/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(query),
      });

      if (!res.ok) {
        throw new Error(`Server returned status ${res.status}`);
      }

      const data = await res.json();
      setJobId(data.job_id);
    } catch (err: any) {
      setErrorMsg(err.message || "Failed to launch discovery job.");
      setIsSubmitting(false);
    }
  };

  const handleTriggerValidation = async (candidateIds: number[]) => {
    if (!runId || candidateIds.length === 0) return;

    setIsValidating(true);
    try {
      const res = await fetch(`${API_BASE_URL}/api/discovery/validate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          discovery_run_id: runId,
          candidate_ids: candidateIds,
        }),
      });

      if (!res.ok) {
        throw new Error("Tier 2 physics validation failed.");
      }

      // Re-fetch updated candidates
      if (jobId) {
        const statusRes = await fetch(`${API_BASE_URL}/api/discovery/query/${jobId}`);
        if (statusRes.ok) {
          const statusData = await statusRes.json();
          setCandidates(statusData.candidates || []);
        }
      }
    } catch (err: any) {
      console.error("Validation error:", err);
    } finally {
      setIsValidating(false);
    }
  };

  const handleTriggerRetrain = async () => {
    setIsRetraining(true);
    try {
      const res = await fetch(`${API_BASE_URL}/api/discovery/retrain`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
      });

      if (res.ok) {
        const data = await res.json();
        setRetrainResult(data);
        fetchRetrainHistory();
      }
    } catch (err) {
      console.error("Retrain trigger error:", err);
    } finally {
      setIsRetraining(false);
    }
  };

  const rank1Count = candidates.filter((c) => c.pareto_rank === 1).length;
  const novelCount = candidates.filter((c) => c.novelty_status === "novel").length;
  const tier2Count = candidates.filter((c) => c.confidence_tier.includes("Tier 2")).length;
  const currentModelVersion = candidates[0]?.gnn_model_version || retrainHistory[0]?.model_version_after || "v1.0.0-initial";

  return (
    <div className="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8 space-y-8">
      {/* Header Banner */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-border pb-6">
        <div>
          <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-primary/10 text-primary text-xs font-bold mb-2">
            <Sparkles className="h-3.5 w-3.5" /> Discovery & Validation Engine
          </div>
          <h1 className="text-3xl font-extrabold tracking-tight text-foreground">Autonomous Crystal Discovery</h1>
          <p className="text-sm text-muted-foreground mt-1 max-w-2xl">
            Generates novel crystal compositions, enforces charge neutrality, predicts calibrated properties via Multi-Scale GNN, verifies structural novelty, and performs Active Learning & Tier 2 MLIP physics validation.
          </p>
        </div>

        {/* Model Version & Active Learning Retrain Controls */}
        <div className="flex flex-col sm:flex-row items-start sm:items-center gap-3">
          <div className="flex items-center gap-2 rounded-xl border border-primary/30 bg-primary/5 px-3.5 py-2">
            <GitBranch className="h-4 w-4 text-primary shrink-0" />
            <div>
              <span className="text-[10px] text-muted-foreground block font-semibold uppercase">Active GNN Model Version</span>
              <span className="font-mono text-xs font-bold text-foreground">{currentModelVersion}</span>
            </div>
          </div>

          <Button
            variant="outline"
            size="sm"
            disabled={isRetraining}
            onClick={handleTriggerRetrain}
            className="text-xs gap-1.5 border-primary/40 font-semibold h-10"
          >
            <RefreshCw className={`h-3.5 w-3.5 ${isRetraining ? "animate-spin" : ""}`} />
            {isRetraining ? "Retraining GNN Loop..." : "Trigger Active Learning Loop"}
          </Button>
        </div>
      </div>

      {retrainResult && (
        <div className="rounded-xl border border-emerald-500/30 bg-emerald-500/10 p-4 text-xs text-emerald-700 dark:text-emerald-300 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <CheckCircle2 className="h-4 w-4 text-emerald-500" />
            <span className="font-semibold">{retrainResult.message}</span>
          </div>
          <span className="font-mono text-[11px] font-bold">New Model Version: {retrainResult.model_version_after}</span>
        </div>
      )}

      {errorMsg && (
        <div className="rounded-xl border border-rose-500/30 bg-rose-500/10 p-4 text-sm text-rose-600 dark:text-rose-400 flex items-center gap-2">
          <AlertCircle className="h-5 w-5 shrink-0" />
          <span>{errorMsg}</span>
        </div>
      )}

      {/* Main Grid: Query Builder & Results Summary */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        <div className="lg:col-span-1 space-y-6">
          <DiscoveryQueryBuilder onSubmit={handleRunDiscovery} isSubmitting={isSubmitting} />

          {/* Active Learning Retrain History Logs */}
          {retrainHistory.length > 0 && (
            <div className="rounded-xl border border-border bg-card p-5 space-y-3">
              <div className="flex items-center justify-between pb-2 border-b border-border">
                <span className="text-xs font-bold text-foreground flex items-center gap-1.5">
                  <GitBranch className="h-4 w-4 text-primary" /> Active Learning Retrain Logs
                </span>
                <span className="text-[10px] text-muted-foreground">{retrainHistory.length} Event(s)</span>
              </div>
              <div className="space-y-2 text-xs">
                {retrainHistory.map((item) => (
                  <div key={item.id} className="rounded-lg border border-border/80 bg-muted/40 p-2.5 space-y-1">
                    <div className="flex items-center justify-between font-mono text-[11px] font-bold text-foreground">
                      <span>{item.model_version_after}</span>
                      <span className="text-emerald-600 dark:text-emerald-400">N = {item.n_samples} Samples</span>
                    </div>
                    <div className="flex items-center justify-between text-[10px] text-muted-foreground">
                      <span>{item.triggered_at ? new Date(item.triggered_at).toLocaleString() : ""}</span>
                      <span>Loss: {item.val_loss_after}</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        <div className="lg:col-span-2 space-y-6">
          {/* Summary Stat Cards */}
          {candidates.length > 0 && (
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
              <div className="rounded-xl border border-border bg-card p-4 text-center">
                <span className="text-2xl font-black text-foreground block">{candidates.length}</span>
                <span className="text-xs text-muted-foreground font-medium">Generated Batch</span>
              </div>
              <div className="rounded-xl border border-emerald-500/30 bg-emerald-500/10 p-4 text-center">
                <span className="text-2xl font-black text-emerald-600 dark:text-emerald-400 block">{rank1Count}</span>
                <span className="text-xs text-emerald-700 dark:text-emerald-300 font-medium">Pareto Frontier (Rank 1)</span>
              </div>
              <div className="rounded-xl border border-purple-500/30 bg-purple-500/10 p-4 text-center">
                <span className="text-2xl font-black text-purple-600 dark:text-purple-400 block">{novelCount}</span>
                <span className="text-xs text-purple-700 dark:text-purple-300 font-medium">Novel Structures</span>
              </div>
              <div className="rounded-xl border border-blue-500/30 bg-blue-500/10 p-4 text-center">
                <span className="text-2xl font-black text-blue-600 dark:text-blue-400 block">{tier2Count}</span>
                <span className="text-xs text-blue-700 dark:text-blue-300 font-medium">Tier 2 Validated</span>
              </div>
            </div>
          )}

          {/* Pareto Frontier Chart */}
          {candidates.length > 0 && (
            <ParetoFrontierChart candidates={candidates} />
          )}

          {/* Candidate Table */}
          {candidates.length > 0 && (
            <CandidateTable
              candidates={candidates}
              onTriggerValidation={handleTriggerValidation}
              isValidating={isValidating}
            />
          )}

          {candidates.length === 0 && !isSubmitting && (
            <div className="rounded-xl border border-dashed border-border bg-card/50 p-12 text-center text-muted-foreground">
              <Sparkles className="mx-auto h-12 w-12 text-muted-foreground/50 mb-3" />
              <h3 className="text-base font-bold text-foreground mb-1">Ready for Discovery</h3>
              <p className="text-xs text-muted-foreground max-w-md mx-auto">
                Configure your target scaffold family, cost limits, and property bounds on the left, then click "Launch Autonomous Discovery Pipeline".
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
