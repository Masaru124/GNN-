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
    <div className="space-y-6">
      <div className="space-y-1.5">
        <p className="micro-label">Single Structure Prediction</p>
        <h1 className="font-display text-2xl font-bold tracking-tight sm:text-3xl">
          Single Crystal Structure Prediction
        </h1>
        <p className="text-sm text-muted-foreground leading-relaxed">
          Upload a CIF structure file to calculate formation energy (E_f) and calibrated confidence.
        </p>
      </div>

      <UploadBox
        onFileSelected={handleFileSelected}
        onSampleSelected={(cifText) => handleSampleSelected(cifText)}
        isLoading={isLoading}
      />

      {isLoading && (
        <Card className="space-y-4 p-10">
          <div className="mx-auto h-8 w-8 animate-spin rounded-full border-2 border-primary border-t-transparent" />
          <div className="space-y-1 text-center">
            <p className="text-sm font-semibold text-foreground">
              Running Multi-Scale GNN Inference...
            </p>
            <p className="text-xs text-muted-foreground">
              Constructing 4Å, 6Å, 8Å graphs & DER uncertainty estimation
            </p>
          </div>
        </Card>
      )}

      {errorMsg && (
        <div className="rounded-lg border border-destructive/30 bg-destructive/10 px-4 py-3 text-sm text-destructive">
          <strong>Error:</strong> {errorMsg}
        </div>
      )}

      {result && !isLoading && (
        <div className="grid grid-cols-1 gap-6 pt-2 lg:grid-cols-3">
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
