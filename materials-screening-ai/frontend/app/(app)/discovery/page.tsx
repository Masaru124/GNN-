"use client";

import { useState, useEffect } from "react";
import {
  Sparkles,
  RefreshCw,
  CheckCircle2,
  AlertCircle,
  Cpu,
  GitBranch,
  ShieldCheck,
} from "lucide-react";
import {
  DiscoveryQueryBuilder,
  DiscoveryQueryPayload,
} from "@/components/discovery/DiscoveryQueryBuilder";
import {
  ParetoFrontierChart,
  CandidateItem,
} from "@/components/discovery/ParetoFrontierChart";
import { CandidateTable } from "@/components/discovery/CandidateTable";
import { Button } from "@/components/ui/button";
import { PageHeader } from "@/components/layout/PageHeader";
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
        const statusRes = await fetch(
          `${API_BASE_URL}/api/discovery/query/${jobId}`,
        );
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
  const novelCount = candidates.filter(
    (c) => c.novelty_status === "novel",
  ).length;
  const tier2Count = candidates.filter((c) =>
    c.confidence_tier.includes("Tier 2"),
  ).length;
  const currentModelVersion =
    candidates[0]?.gnn_model_version ||
    retrainHistory[0]?.model_version_after ||
    "v1.0.0-initial";

  return (
    <div className="space-y-8">
      {/* Page Header */}
      <PageHeader
        title="Autonomous Crystal Discovery"
        description="Generate novel crystal compositions, predict material properties, verify structural novelty, and validate promising candidates."
      />

      {/* Discovery Controls */}
      <div className="border-b border-border pb-6">
        <div className="flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
          <div>
            <p className="text-sm font-semibold text-foreground">
              Discovery Configuration
            </p>

            <p className="mt-1 text-xs text-muted-foreground">
              Define the search space and launch the autonomous discovery pipeline.
            </p>
          </div>

          <div className="flex items-center gap-3">
            <div className="flex items-center gap-2 text-xs text-muted-foreground">
              <GitBranch className="h-3.5 w-3.5 text-primary" />

              <span>GNN</span>

              <span className="font-mono font-semibold text-foreground">
                {currentModelVersion}
              </span>
            </div>

            <Button
              variant="outline"
              size="sm"
              disabled={isRetraining}
              onClick={handleTriggerRetrain}
              className="gap-2"
            >
              <RefreshCw
                className={`h-3.5 w-3.5 ${
                  isRetraining ? "animate-spin" : ""
                }`}
              />

              {isRetraining
                ? "Retraining..."
                : "Run Active Learning"}
            </Button>
          </div>
        </div>

        <div className="mt-5">
          <DiscoveryQueryBuilder
            onSubmit={handleRunDiscovery}
            isSubmitting={isSubmitting}
          />
        </div>
      </div>

      {/* Notifications */}
      {retrainResult && (
        <div className="flex items-center justify-between border-b border-emerald-500/30 pb-4 text-xs text-emerald-600 dark:text-emerald-400">
          <div className="flex items-center gap-2">
            <CheckCircle2 className="h-4 w-4" />

            <span className="font-medium">
              {retrainResult.message}
            </span>
          </div>

          <span className="font-mono">
            {retrainResult.model_version_after}
          </span>
        </div>
      )}

      {errorMsg && (
        <div className="flex items-center gap-2 border-b border-destructive/30 pb-4 text-xs text-destructive">
          <AlertCircle className="h-4 w-4" />
          <span>{errorMsg}</span>
        </div>
      )}

      {/* Main Workspace */}
      <div className="grid grid-cols-1 gap-10 xl:grid-cols-[280px_minmax(0,1fr)]">

        {/* Sidebar */}
        <aside className="space-y-7">
          {/* Pipeline Status */}
          <div>
            <div className="mb-3 flex items-center justify-between">
              <div>
                <p className="text-sm font-semibold">
                  Pipeline
                </p>

                <p className="text-xs text-muted-foreground">
                  Current discovery run
                </p>
              </div>

              {status && (
                <span className="text-xs font-medium capitalize text-muted-foreground">
                  {status}
                </span>
              )}
            </div>

            <div className="space-y-2 border-l border-border pl-4">
              <PipelineStep
                label="Candidate Generation"
                active={
                  status === "queued" ||
                  status === "running"
                }
                complete={candidates.length > 0}
              />

              <PipelineStep
                label="GNN Property Prediction"
                active={candidates.length > 0}
                complete={candidates.length > 0}
              />

              <PipelineStep
                label="Novelty Verification"
                active={candidates.length > 0}
                complete={novelCount > 0}
              />

              <PipelineStep
                label="Tier 2 Physics Validation"
                active={tier2Count > 0}
                complete={tier2Count > 0}
              />
            </div>
          </div>

          {/* Active Learning */}
          {retrainHistory.length > 0 && (
            <div className="border-t border-border pt-5">
              <div className="mb-3 flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <GitBranch className="h-3.5 w-3.5 text-primary" />

                  <span className="text-sm font-semibold">
                    Retrain History
                  </span>
                </div>

                <span className="text-[10px] text-muted-foreground">
                  {retrainHistory.length}
                </span>
              </div>

              <div className="divide-y divide-border">
                {retrainHistory.map((item) => (
                  <div
                    key={item.id}
                    className="flex items-center justify-between py-2.5"
                  >
                    <div>
                      <p className="font-mono text-[11px] font-semibold">
                        {item.model_version_after}
                      </p>

                      <p className="text-[10px] text-muted-foreground">
                        {item.triggered_at
                          ? new Date(
                              item.triggered_at,
                            ).toLocaleDateString()
                          : ""}
                      </p>
                    </div>

                    <div className="text-right">
                      <p className="text-[10px] text-emerald-600 dark:text-emerald-400">
                        N={item.n_samples}
                      </p>

                      <p className="text-[10px] text-muted-foreground">
                        Loss {item.val_loss_after}
                      </p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </aside>

        {/* Results */}
        <section className="min-w-0">

          {/* Results Header */}
          <div className="flex flex-col gap-4 border-b border-border pb-4 sm:flex-row sm:items-end sm:justify-between">
            <div>
              <p className="text-sm font-semibold">
                Discovery Results
              </p>

              <p className="mt-1 text-xs text-muted-foreground">
                Candidates generated from the current search configuration.
              </p>
            </div>

            {candidates.length > 0 && (
              <div className="flex items-center divide-x divide-border">
                <ResultStat
                  value={candidates.length}
                  label="Generated"
                />

                <ResultStat
                  value={rank1Count}
                  label="Pareto"
                  className="text-emerald-600 dark:text-emerald-400"
                />

                <ResultStat
                  value={novelCount}
                  label="Novel"
                  className="text-purple-600 dark:text-purple-400"
                />

                <ResultStat
                  value={tier2Count}
                  label="Tier 2"
                  className="text-blue-600 dark:text-blue-400"
                />
              </div>
            )}
          </div>

          {/* Loading */}
          {isSubmitting && candidates.length === 0 && (
            <div className="flex min-h-[420px] items-center justify-center">
              <div className="text-center">
                <RefreshCw className="mx-auto h-6 w-6 animate-spin text-primary" />

                <p className="mt-4 text-sm font-semibold">
                  Running discovery pipeline
                </p>

                <p className="mt-1 text-xs text-muted-foreground">
                  Generating and evaluating candidate materials...
                </p>

                <p className="mt-3 font-mono text-[10px] uppercase tracking-wider text-muted-foreground">
                  {status || "processing"}
                </p>
              </div>
            </div>
          )}

          {/* Empty */}
          {candidates.length === 0 &&
            !isSubmitting &&
            !status && (
              <div className="flex min-h-[420px] items-center justify-center border-y border-dashed border-border">
                <div className="max-w-md text-center">
                  <Sparkles className="mx-auto h-8 w-8 text-muted-foreground/50" />

                  <h3 className="mt-4 text-sm font-semibold">
                    Ready for Discovery
                  </h3>

                  <p className="mt-2 text-xs leading-5 text-muted-foreground">
                    Configure your target scaffold family, cost limits,
                    and property bounds above, then launch the autonomous
                    discovery pipeline.
                  </p>
                </div>
              </div>
            )}

          {/* Results */}
          {candidates.length > 0 && (
            <div className="mt-6 space-y-8">

              {/* Pareto */}
              <div>
                <div className="mb-4 flex items-center justify-between">
                  <div>
                    <h3 className="text-sm font-semibold">
                      Pareto Frontier
                    </h3>

                    <p className="mt-1 text-xs text-muted-foreground">
                      Multi-objective ranking of generated candidates.
                    </p>
                  </div>

                  <span className="flex items-center gap-1.5 text-xs text-emerald-600 dark:text-emerald-400">
                    <ShieldCheck className="h-3.5 w-3.5" />
                    {rank1Count} frontier candidates
                  </span>
                </div>

                <div className="border-y border-border py-5">
                  <ParetoFrontierChart
                    candidates={candidates}
                  />
                </div>
              </div>

              {/* Candidates */}
              <div>
                <div className="mb-4 flex items-end justify-between">
                  <div>
                    <h3 className="text-sm font-semibold">
                      Candidate Materials
                    </h3>

                    <p className="mt-1 text-xs text-muted-foreground">
                      Review candidates and trigger Tier 2 validation.
                    </p>
                  </div>

                  <span className="text-xs text-muted-foreground">
                    {candidates.length} candidates
                  </span>
                </div>

                <CandidateTable
                  candidates={candidates}
                  onTriggerValidation={handleTriggerValidation}
                  isValidating={isValidating}
                />
              </div>
            </div>
          )}
        </section>
      </div>
    </div>
  );
}


function ResultStat({
  value,
  label,
  className = "",
}: {
  value: number;
  label: string;
  className?: string;
}) {
  return (
    <div className="px-4 first:pl-0">
      <p className={`text-lg font-bold leading-none ${className}`}>
        {value}
      </p>

      <p className="mt-1 text-[10px] uppercase tracking-wider text-muted-foreground">
        {label}
      </p>
    </div>
  );
}

function PipelineStep({
  label,
  active,
  complete,
}: {
  label: string;
  active: boolean;
  complete: boolean;
}) {
  return (
    <div className="relative flex items-center gap-2.5 py-1.5">
      <span
        className={`h-1.5 w-1.5 rounded-full ${
          complete
            ? "bg-emerald-500"
            : active
              ? "bg-primary animate-pulse"
              : "bg-muted-foreground/30"
        }`}
      />

      <span
        className={`text-xs ${
          complete || active
            ? "text-foreground"
            : "text-muted-foreground"
        }`}
      >
        {label}
      </span>
    </div>
  );
}
