import React, { useState } from 'react';
import { UploadCloud, Layers, FileText, CheckCircle2, Play } from 'lucide-react';
import { BatchTable } from '../components/BatchTable';
import { BatchScreenResponse } from '../types';
import { api } from '../services/api';

export const BatchScreening: React.FC = () => {
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
    } catch (err: any) {
      setErrorMsg(err.response?.data?.detail || err.message || 'Batch screening failed.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-slate-100">Batch Candidate Screening</h1>
        <p className="text-xs text-slate-400">Upload multiple CIF files to rank candidate materials by predicted stability and filter by confidence thresholds.</p>
      </div>

      <div className="glass-card rounded-2xl p-6 border border-slate-800 space-y-4">
        <div className="flex flex-col sm:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-blue-600/15 border border-blue-500/30 flex items-center justify-center text-blue-400">
              <Layers className="w-5 h-5" />
            </div>
            <div>
              <p className="text-sm font-semibold text-slate-200">
                {selectedFiles.length > 0 ? `${selectedFiles.length} CIF files selected` : 'Select Multiple CIF Files'}
              </p>
              <p className="text-xs text-slate-400">Supported formats: .cif, .poscar</p>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <label className="cursor-pointer inline-flex items-center gap-2 px-4 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-semibold border border-slate-700/60 transition-all">
              <UploadCloud className="w-4 h-4 text-cyan-400" />
              <span>Select Files</span>
              <input
                type="file"
                multiple
                accept=".cif,.poscar,.txt"
                onChange={handleFilesSelect}
                className="hidden"
              />
            </label>

            <button
              onClick={handleRunBatch}
              disabled={selectedFiles.length === 0 || isLoading}
              className="inline-flex items-center gap-2 px-5 py-2 rounded-xl bg-blue-600 hover:bg-blue-500 disabled:opacity-50 text-white text-xs font-semibold shadow-lg shadow-blue-600/25 transition-all"
            >
              <Play className="w-4 h-4 fill-white" />
              <span>{isLoading ? 'Screening...' : 'Start Batch Screening'}</span>
            </button>
          </div>
        </div>

        {selectedFiles.length > 0 && (
          <div className="p-3 rounded-xl bg-slate-900/60 border border-slate-800 flex flex-wrap gap-2 text-xs text-slate-300">
            {selectedFiles.map((f, i) => (
              <span key={i} className="px-2.5 py-1 rounded-md bg-slate-800 border border-slate-700/50 flex items-center gap-1.5">
                <FileText className="w-3 h-3 text-cyan-400" />
                <span>{f.name}</span>
              </span>
            ))}
          </div>
        )}
      </div>

      {isLoading && (
        <div className="glass-card rounded-2xl p-12 text-center border border-slate-800 space-y-3">
          <div className="w-10 h-10 border-4 border-blue-500 border-t-transparent rounded-full animate-spin mx-auto"></div>
          <p className="text-sm font-semibold text-slate-200">Screening {selectedFiles.length} Material Candidates...</p>
          <p className="text-xs text-slate-400">Executing parallel Multi-Scale GNN graph inference</p>
        </div>
      )}

      {errorMsg && (
        <div className="p-4 rounded-2xl bg-rose-500/10 border border-rose-500/30 text-rose-400 text-xs">
          <strong>Error:</strong> {errorMsg}
        </div>
      )}

      {batchResult && !isLoading && (
        <div className="space-y-6">
          <div className="grid grid-cols-4 gap-4 text-center">
            <div className="p-4 rounded-2xl glass-card border border-slate-800">
              <p className="text-xs text-slate-400 font-medium">Job ID</p>
              <p className="text-sm font-bold text-blue-400 font-mono mt-0.5">{batchResult.job_id}</p>
            </div>
            <div className="p-4 rounded-2xl glass-card border border-slate-800">
              <p className="text-xs text-slate-400 font-medium">Total Screened</p>
              <p className="text-2xl font-black text-slate-100 font-mono mt-0.5">{batchResult.screened_count}</p>
            </div>
            <div className="p-4 rounded-2xl glass-card border border-slate-800">
              <p className="text-xs text-slate-400 font-medium">High Trust Candidates</p>
              <p className="text-2xl font-black text-emerald-400 font-mono mt-0.5">{batchResult.high_confidence_count}</p>
            </div>
            <div className="p-4 rounded-2xl glass-card border border-slate-800">
              <p className="text-xs text-slate-400 font-medium">Runtime</p>
              <p className="text-2xl font-black text-slate-100 font-mono mt-0.5">{batchResult.runtime_seconds}s</p>
            </div>
          </div>

          <BatchTable candidates={batchResult.candidates} />
        </div>
      )}
    </div>
  );
};
