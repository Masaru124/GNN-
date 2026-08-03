import React, { useEffect, useState } from 'react';
import { 
  TrendingUp, 
  ShieldCheck, 
  Layers, 
  Award, 
  ArrowRight, 
  Search, 
  UploadCloud, 
  Sparkles,
  CheckCircle2
} from 'lucide-react';
import { api } from '../services/api';

interface DashboardProps {
  onNavigate: (page: string) => void;
}

export const Dashboard: React.FC<DashboardProps> = ({ onNavigate }) => {
  const [stats, setStats] = useState({
    totalPredictions: 48,
    avgConfidencePct: 92.4,
    screeningsToday: 12,
    topCandidate: 'TiO2 (Rutile)'
  });

  return (
    <div className="space-y-8">
      {/* Welcome Banner */}
      <div className="relative glass-card rounded-3xl p-8 border border-slate-800 overflow-hidden bg-gradient-to-r from-blue-950/40 via-dark-900 to-indigo-950/30">
        <div className="relative z-10 max-w-3xl space-y-3">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-blue-500/10 border border-blue-500/20 text-blue-400 text-xs font-semibold">
            <Sparkles className="w-3.5 h-3.5" />
            <span>AI-Powered Materials Screening Assistant</span>
          </div>

          <h1 className="text-3xl font-extrabold text-slate-100 tracking-tight">
            Accelerate Materials Discovery with <span className="bg-gradient-to-r from-blue-400 via-cyan-300 to-emerald-400 bg-clip-text text-transparent">Calibrated GNN Confidence</span>
          </h1>

          <p className="text-sm text-slate-400 leading-relaxed">
            Screen promising crystal structures <strong className="text-slate-200">before expensive DFT simulations</strong>. Get predicted formation energy ($E_f$) coupled with 90% Conformal Calibration intervals and Evidential Uncertainty decomposition.
          </p>

          <div className="flex flex-wrap items-center gap-4 pt-2">
            <button
              onClick={() => onNavigate('upload')}
              className="flex items-center gap-2 px-5 py-2.5 rounded-xl bg-blue-600 hover:bg-blue-500 text-white font-semibold text-sm shadow-lg shadow-blue-600/25 transition-all"
            >
              <UploadCloud className="w-4 h-4" />
              <span>Predict Single CIF</span>
            </button>

            <button
              onClick={() => onNavigate('batch')}
              className="flex items-center gap-2 px-5 py-2.5 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 font-semibold text-sm border border-slate-700/60 transition-all"
            >
              <Layers className="w-4 h-4 text-cyan-400" />
              <span>Batch Screening</span>
            </button>
          </div>
        </div>
      </div>

      {/* Analytics KPI Metric Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="glass-card rounded-2xl p-5 border border-slate-800 space-y-2">
          <div className="flex items-center justify-between text-slate-400 text-xs font-medium">
            <span>Total Predictions</span>
            <TrendingUp className="w-4 h-4 text-blue-400" />
          </div>
          <p className="text-3xl font-black text-slate-100 font-mono">{stats.totalPredictions}</p>
          <p className="text-[11px] text-emerald-400 font-medium">↑ +14% this week</p>
        </div>

        <div className="glass-card rounded-2xl p-5 border border-slate-800 space-y-2">
          <div className="flex items-center justify-between text-slate-400 text-xs font-medium">
            <span>Average Confidence</span>
            <ShieldCheck className="w-4 h-4 text-emerald-400" />
          </div>
          <p className="text-3xl font-black text-emerald-400 font-mono">{stats.avgConfidencePct}%</p>
          <p className="text-[11px] text-slate-400">High Trust Predictions</p>
        </div>

        <div className="glass-card rounded-2xl p-5 border border-slate-800 space-y-2">
          <div className="flex items-center justify-between text-slate-400 text-xs font-medium">
            <span>Today's Screenings</span>
            <Layers className="w-4 h-4 text-cyan-400" />
          </div>
          <p className="text-3xl font-black text-slate-100 font-mono">{stats.screeningsToday}</p>
          <p className="text-[11px] text-slate-400">Batches executed</p>
        </div>

        <div className="glass-card rounded-2xl p-5 border border-slate-800 space-y-2">
          <div className="flex items-center justify-between text-slate-400 text-xs font-medium">
            <span>Top Candidate</span>
            <Award className="w-4 h-4 text-amber-400" />
          </div>
          <p className="text-lg font-bold text-amber-400 truncate">{stats.topCandidate}</p>
          <p className="text-[11px] text-slate-400">$E_f$ = -2.85 eV/atom</p>
        </div>
      </div>

      {/* Featured Quick Actions Grid */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div 
          onClick={() => onNavigate('search')}
          className="glass-card glass-card-hover rounded-2xl p-6 border border-slate-800 cursor-pointer space-y-3"
        >
          <div className="w-12 h-12 rounded-xl bg-cyan-500/10 border border-cyan-500/20 flex items-center justify-center text-cyan-400">
            <Search className="w-6 h-6" />
          </div>
          <h3 className="text-base font-bold text-slate-100">Search Materials Project</h3>
          <p className="text-xs text-slate-400 leading-relaxed">
            Query over 150,000+ crystal structures by chemical formula or element with 1-click GNN property screening.
          </p>
          <div className="flex items-center gap-1.5 text-xs font-semibold text-cyan-400 pt-1">
            <span>Search Materials</span>
            <ArrowRight className="w-3.5 h-3.5" />
          </div>
        </div>

        <div 
          onClick={() => onNavigate('upload')}
          className="glass-card glass-card-hover rounded-2xl p-6 border border-slate-800 cursor-pointer space-y-3"
        >
          <div className="w-12 h-12 rounded-xl bg-blue-500/10 border border-blue-500/20 flex items-center justify-center text-blue-400">
            <UploadCloud className="w-6 h-6" />
          </div>
          <h3 className="text-base font-bold text-slate-100">Single CIF Prediction</h3>
          <p className="text-xs text-slate-400 leading-relaxed">
            Upload custom CIF crystal files to get instant property estimates, 3D WebGL lattice rendering, and scale attention charts.
          </p>
          <div className="flex items-center gap-1.5 text-xs font-semibold text-blue-400 pt-1">
            <span>Upload Structure</span>
            <ArrowRight className="w-3.5 h-3.5" />
          </div>
        </div>

        <div 
          onClick={() => onNavigate('batch')}
          className="glass-card glass-card-hover rounded-2xl p-6 border border-slate-800 cursor-pointer space-y-3"
        >
          <div className="w-12 h-12 rounded-xl bg-emerald-500/10 border border-emerald-500/20 flex items-center justify-center text-emerald-400">
            <Layers className="w-6 h-6" />
          </div>
          <h3 className="text-base font-bold text-slate-100">Batch Candidate Screening</h3>
          <p className="text-xs text-slate-400 leading-relaxed">
            Screen 50+ structures simultaneously, filter by confidence thresholds, rank candidates, and export CSV reports.
          </p>
          <div className="flex items-center gap-1.5 text-xs font-semibold text-emerald-400 pt-1">
            <span>Run Batch Screening</span>
            <ArrowRight className="w-3.5 h-3.5" />
          </div>
        </div>
      </div>
    </div>
  );
};
