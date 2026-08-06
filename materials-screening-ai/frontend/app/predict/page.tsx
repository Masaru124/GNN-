"use client";

import { useState } from "react";
import { UploadBox } from "@/components/UploadBox";
import { PredictionCard } from "@/components/PredictionCard";
import { ConfidenceGauge } from "@/components/ConfidenceGauge";
import { AttentionChart } from "@/components/AttentionChart";
import { Viewer3D } from "@/components/Viewer3D";
import { PredictResponsePayload } from "@/lib/types";
import { api } from "@/lib/api";
import { getErrorMessage } from "@/lib/utils";
import { Card } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { PageHeader } from "@/components/layout/PageHeader";

export default function Page() {
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<PredictResponsePayload | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const handleFileSelected = async (file: File) => {
    setIsLoading(true);
    setErrorMsg(null);
    try {
      const res = await api.predictSingle(undefined, file);
      setResult(res);
    } catch (err) {
      setErrorMsg(getErrorMessage(err, "Prediction failed."));
    } finally {
      setIsLoading(false);
    }
  };

  const handleSampleSelected = async (cifText: string) => {
    setIsLoading(true);
    setErrorMsg(null);
    try {
      const res = await api.predictSingle(cifText);
      setResult(res);
    } catch (err) {
      setErrorMsg(getErrorMessage(err, "Prediction failed."));
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="space-y-8">
      <PageHeader
        eyebrow="Single Structure Prediction"
        title="Single Crystal Structure Prediction"
        description="Upload a CIF structure file to calculate formation energy (E_f) and calibrated confidence."
      />

      <UploadBox
        onFileSelected={handleFileSelected}
        onSampleSelected={(cifText) => handleSampleSelected(cifText)}
        isLoading={isLoading}
      />

      {isLoading && (
        <Card className="p-8">
          <div className="flex items-center gap-3">
            <div className="h-6 w-6 shrink-0 animate-spin rounded-full border-2 border-border border-t-primary" />
            <div className="space-y-0.5">
              <p className="text-sm font-semibold text-foreground">
                Running Multi-Scale GNN Inference...
              </p>
              <p className="text-xs text-muted-foreground">
                Constructing 4Å, 6Å, 8Å graphs & DER uncertainty estimation
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

      {result && !isLoading && (
        <div className="grid animate-fade-up grid-cols-1 gap-6 pt-4 lg:grid-cols-3">
          <div className="space-y-6 lg:col-span-2">
            <PredictionCard materialInfo={result.material_info} prediction={result.prediction} />

            <Viewer3D
              cifString={result.material_info.cif_string}
              formula={result.material_info.formula_pretty || result.material_info.formula}
            />
          </div>

          <div className="space-y-6">
            <ConfidenceGauge prediction={result.prediction} />
            <AttentionChart scaleAttention={result.prediction.scale_attention} />
          </div>
        </div>
      )}
    </div>
  );
}
