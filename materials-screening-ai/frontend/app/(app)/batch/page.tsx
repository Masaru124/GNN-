"use client";

import { useState } from "react";
import { UploadCloud, Layers, FileText, Play, Fingerprint, ShieldCheck, Timer } from "lucide-react";
import { BatchTable } from "@/components/BatchTable";
import { BatchScreenResponse } from "@/lib/types";
import { api } from "@/lib/api";
import { getErrorMessage } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { StatCard } from "@/components/ui/stat-card";
import { PageHeader } from "@/components/layout/PageHeader";

export default function Page() {
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [batchResult, setBatchResult] = useState<BatchScreenResponse | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const handleFilesSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      const fileList = Array.from(e.target.files);
      setSelectedFiles(fileList);
    }
  };

  const handleRunBatch = async () => {
    if (selectedFiles.length === 0) return;
    setIsLoading(true);
    setErrorMsg(null);

    try {
      const res = await api.screenBatch(selectedFiles);
      setBatchResult(res);
    } catch (err) {
      setErrorMsg(getErrorMessage(err, "Batch screening failed."));
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="space-y-8">
      <PageHeader
        title="Batch Candidate Screening"
        description="Upload multiple CIF files to rank candidate materials by predicted stability and filter by confidence thresholds."
        actions={
          <Button onClick={handleRunBatch} disabled={selectedFiles.length === 0 || isLoading}>
            <Play className="h-4 w-4 fill-current" />
            {isLoading ? "Screening..." : "Start Batch Screening"}
          </Button>
        }
      />

      <Card className="p-6">
        <div className="flex flex-col gap-5 sm:flex-row sm:items-center sm:justify-between">
          <div className="flex items-center gap-4">
            <div className="flex h-11 w-11 shrink-0 items-center justify-center rounded-xl bg-primary-muted text-primary">
              <Layers className="h-5 w-5" />
            </div>
            <div>
              <p className="text-sm font-semibold text-foreground">
                {selectedFiles.length > 0
                  ? `${selectedFiles.length} CIF files selected`
                  : "Select multiple CIF files"}
              </p>
              <p className="text-xs text-muted-foreground">Supported formats: .cif, .poscar</p>
            </div>
          </div>

          <label className="inline-flex h-10 cursor-pointer items-center justify-center gap-2 whitespace-nowrap rounded-lg border border-border bg-background/60 px-4 text-sm font-semibold text-foreground shadow-xs transition-all duration-150 ease-out hover:border-muted-foreground/30 hover:bg-muted focus-ring">
            <UploadCloud className="h-4 w-4 text-accent" />
            Select Files
            <input
              type="file"
              multiple
              accept=".cif,.poscar,.txt"
              onChange={handleFilesSelect}
              className="hidden"
            />
          </label>
        </div>

        {selectedFiles.length > 0 && (
          <div className="mt-5 flex flex-wrap gap-2 rounded-lg border border-border bg-muted/40 p-3">
            {selectedFiles.map((f, i) => (
              <span
                key={i}
                className="inline-flex items-center gap-1.5 rounded-md border border-border bg-card px-2.5 py-1 text-xs text-foreground"
              >
                <FileText className="h-3 w-3 text-accent" />
                {f.name}
              </span>
            ))}
          </div>
        )}
      </Card>

      {isLoading && (
        <Card className="p-8">
          <div className="flex items-center gap-3">
            <div className="h-6 w-6 shrink-0 animate-spin rounded-full border-2 border-border border-t-primary" />
            <div className="space-y-0.5">
              <p className="text-sm font-semibold text-foreground">
                Screening {selectedFiles.length} Material Candidates...
              </p>
              <p className="text-xs text-muted-foreground">
                Executing parallel Multi-Scale GNN graph inference
              </p>
            </div>
          </div>
          <div className="mt-5 space-y-3">
            <Skeleton className="h-4 w-full" />
            <Skeleton className="h-4 w-4/5" />
            <Skeleton className="h-4 w-3/5" />
          </div>
        </Card>
      )}

      {errorMsg && (
        <div
          role="alert"
          className="rounded-lg border border-destructive/25 bg-destructive-muted px-4 py-3 text-sm text-destructive"
        >
          <strong>Error:</strong> {errorMsg}
        </div>
      )}

      {batchResult && !isLoading && (
        <div className="animate-fade-up space-y-6">
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
            <StatCard
              label="Job ID"
              value={batchResult.job_id}
              icon={Fingerprint}
              iconClass="bg-primary-muted text-primary"
            />
            <StatCard
              label="Total Screened"
              value={batchResult.screened_count}
              icon={Layers}
              iconClass="bg-primary-muted text-primary"
            />
            <StatCard
              label="High Trust Candidates"
              value={batchResult.high_confidence_count}
              icon={ShieldCheck}
              iconClass="bg-success-muted text-success"
            />
            <StatCard
              label="Runtime"
              value={`${batchResult.runtime_seconds}s`}
              icon={Timer}
            />
          </div>

          <BatchTable candidates={batchResult.candidates} />
        </div>
      )}
    </div>
  );
}
