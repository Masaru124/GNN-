import React, { useState } from 'react';
import { UploadBox } from '../components/UploadBox';
import { PredictionCard } from '../components/PredictionCard';
import { ConfidenceGauge } from '../components/ConfidenceGauge';
import { AttentionChart } from '../components/AttentionChart';
import { Viewer3D } from '../components/Viewer3D';
import { PredictResponsePayload } from '../types';
import { api } from '../services/api';

export const SinglePrediction: React.FC = () => {
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<PredictResponsePayload | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const handleFileSelected = async (file: File) => {
    setIsLoading(true);
    setErrorMsg(null);
    try {
      const res = await api.predictSingle(undefined, file);
      setResult(res);
    } catch (err: any) {
      setErrorMsg(err.response?.data?.detail || err.message || 'Prediction failed.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleSampleSelected = async (cifText: string, formula: string) => {
    setIsLoading(true);
    setErrorMsg(null);
    try {
      const res = await api.predictSingle(cifText);
      setResult(res);
    } catch (err: any) {
      setErrorMsg(err.response?.data?.detail || err.message || 'Prediction failed.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-slate-100">Single Crystal Structure Prediction</h1>
        <p className="text-xs text-slate-400">Upload a CIF structure file to calculate formation energy ($E_f$) and calibrated confidence.</p>
      </div>

      <UploadBox
        onFileSelected={handleFileSelected}
        onSampleSelected={handleSampleSelected}
        isLoading={isLoading}
      />

      {isLoading && (
        <div className="glass-card rounded-2xl p-12 text-center border border-slate-800 space-y-3">
          <div className="w-10 h-10 border-4 border-blue-500 border-t-transparent rounded-full animate-spin mx-auto"></div>
          <p className="text-sm font-semibold text-slate-200">Running Multi-Scale GNN Inference...</p>
          <p className="text-xs text-slate-400">Constructing 4Å, 6Å, 8Å graphs & DER uncertainty estimation</p>
        </div>
      )}

      {errorMsg && (
        <div className="p-4 rounded-2xl bg-rose-500/10 border border-rose-500/30 text-rose-400 text-xs">
          <strong>Error:</strong> {errorMsg}
        </div>
      )}

      {result && !isLoading && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 pt-2">
          <div className="lg:col-span-2 space-y-6">
            <PredictionCard
              materialInfo={result.material_info}
              prediction={result.prediction}
            />

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
};
