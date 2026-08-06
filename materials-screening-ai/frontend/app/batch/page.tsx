"use client";

import { useState } from "react";
import { UploadCloud, Layers, FileText, Play } from "lucide-react";
import { BatchTable } from "@/components/BatchTable";
import { BatchScreenResponse } from "@/lib/types";
import { api } from "@/lib/api";
import { getErrorMessage } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";

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
    <div className="space-y-6">
      <div className="space-y-1.5">
        <p className="micro-label">High-Throughput Screening</p>
        <h1 className="font-display text-2xl font-bold tracking-tight sm:text-3xl">
          Batch Candidate Screening
        </h1>
        <p className="text-sm text-muted-foreground leading-relaxed">
          Upload multiple CIF files to rank candidate materials by predicted stability and filter by
          confidence thresholds.
        </p>
      </div>

      <Card className="p-5">
        <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary/10 text-primary">
              <Layers className="h-5 w-5" />
            </div>
            <div>
              <p className="text-sm font-semibold text-foreground">
                {selectedFiles.length > 0
                  ? `${selectedFiles.length} CIF files selected`
                  : "Select Multiple CIF Files"}
              </p>
              <p className="text-xs text-muted-foreground">Supported formats: .cif, .poscar</p>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <label className="inline-flex h-10 cursor-pointer items-center gap-2 rounded-lg border border-border bg-background px-4 text-sm font-medium text-foreground shadow-sm transition-colors hover:bg-muted focus-ring">
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

            <Button onClick={handleRunBatch} disabled={selectedFiles.length === 0 || isLoading}>
              <Play className="h-4 w-4 fill-current" />
              {isLoading ? "Screening..." : "Start Batch Screening"}
            </Button>
          </div>
        </div>

        {selectedFiles.length > 0 && (
          <div className="mt-4 flex flex-wrap gap-2 rounded-lg border border-border bg-muted/60 p-3">
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
        <Card className="space-y-4 p-10">
          <div className="mx-auto h-8 w-8 animate-spin rounded-full border-2 border-primary border-t-transparent" />
          <div className="space-y-1 text-center">
            <p className="text-sm font-semibold text-foreground">
              Screening {selectedFiles.length} Material Candidates...
            </p>
            <p className="text-xs text-muted-foreground">
              Executing parallel Multi-Scale GNN graph inference
            </p>
          </div>
        </Card>
      )}

      {errorMsg && (
        <div className="rounded-lg border border-destructive/30 bg-destructive/10 px-4 py-3 text-sm text-destructive">
          <strong>Error:</strong> {errorMsg}
        </div>
      )}

      {batchResult && !isLoading && (
        <div className="space-y-6">
          <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
            <div className="rounded-xl border border-border bg-card p-5 shadow-sm">
              <p className="micro-label">Job ID</p>
              <p className="mt-1 font-mono text-lg font-bold text-primary">{batchResult.job_id}</p>
            </div>
            <div className="rounded-xl border border-border bg-card p-5 shadow-sm">
              <p className="micro-label">Total Screened</p>
              <p className="mt-1 font-mono text-2xl font-bold">{batchResult.screened_count}</p>
            </div>
            <div className="rounded-xl border border-border bg-card p-5 shadow-sm">
              <p className="micro-label">High Trust Candidates</p>
              <p className="mt-1 font-mono text-2xl font-bold text-success">
                {batchResult.high_confidence_count}
              </p>
            </div>
            <div className="rounded-xl border border-border bg-card p-5 shadow-sm">
              <p className="micro-label">Runtime</p>
              <p className="mt-1 font-mono text-2xl font-bold">{batchResult.runtime_seconds}s</p>
            </div>
          </div>

          <BatchTable candidates={batchResult.candidates} />
        </div>
      )}
    </div>
  );
}
