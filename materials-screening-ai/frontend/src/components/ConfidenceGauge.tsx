import React from 'react';
import { ShieldCheck, AlertTriangle, AlertOctagon } from 'lucide-react';
import { PredictionResult } from '../types';

interface ConfidenceGaugeProps {
  prediction: PredictionResult;
}

export const ConfidenceGauge: React.FC<ConfidenceGaugeProps> = ({ prediction }) => {
  const {
    confidence,
    confidence_score_pct,
    evidential_std_eV,
    conformal_90_interval_eV,
    conformal_width_eV,
    risk_level,
    recommendation,
    badge_color,
  } = prediction;

  const colorStyles = {
    green: {
      bg: 'bg-emerald-500/10',
      border: 'border-emerald-500/30',
      text: 'text-emerald-400',
      glow: 'glow-green',
      icon: ShieldCheck,
    },
    yellow: {
      bg: 'bg-amber-500/10',
      border: 'border-amber-500/30',
      text: 'text-amber-400',
      glow: 'glow-yellow',
      icon: AlertTriangle,
    },
    red: {
      bg: 'bg-rose-500/10',
      border: 'border-rose-500/30',
      text: 'text-rose-400',
      glow: 'glow-red',
      icon: AlertOctagon,
    },
  }[badge_color || 'green'];

  const IconComponent = colorStyles.icon;

  return (
    <div className="glass-card rounded-2xl p-5 border border-slate-800 space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold text-slate-200">Uncertainty & Calibration Gauge</h3>
        <div className={`flex items-center gap-1.5 px-3 py-1 rounded-full border text-xs font-semibold ${colorStyles.bg} ${colorStyles.border} ${colorStyles.text} ${colorStyles.glow}`}>
          <IconComponent className="w-3.5 h-3.5" />
          <span>{confidence} Confidence</span>
        </div>
      </div>

      {/* Progress Bar for Confidence Score */}
      <div className="space-y-1.5">
        <div className="flex justify-between text-xs font-medium">
          <span className="text-slate-400">Calibrated Coverage Score</span>
          <span className="text-slate-200">{confidence_score_pct}%</span>
        </div>
        <div className="w-full h-2.5 bg-slate-800 rounded-full overflow-hidden">
          <div
            className={`h-full transition-all duration-500 ${
              badge_color === 'green' ? 'bg-emerald-500' : badge_color === 'yellow' ? 'bg-amber-500' : 'bg-rose-500'
            }`}
            style={{ width: `${Math.max(5, confidence_score_pct)}%` }}
          />
        </div>
      </div>

      {/* Key Calibration Metrics Grid */}
      <div className="grid grid-cols-2 gap-3 pt-2">
        <div className="p-3 rounded-xl bg-slate-900/60 border border-slate-800">
          <p className="text-[11px] text-slate-400 font-medium">Evidential Std ($\sigma$)</p>
          <p className="text-base font-bold text-slate-100 font-mono mt-0.5">
            ±{evidential_std_eV} <span className="text-xs font-normal text-slate-400">eV/atom</span>
          </p>
        </div>

        <div className="p-3 rounded-xl bg-slate-900/60 border border-slate-800">
          <p className="text-[11px] text-slate-400 font-medium">Conformal 90% Interval</p>
          <p className="text-xs font-bold text-blue-400 font-mono mt-1">
            [{conformal_90_interval_eV[0]}, {conformal_90_interval_eV[1]}] eV
          </p>
        </div>
      </div>

      {/* Recommendation Card */}
      <div className={`p-3.5 rounded-xl border ${colorStyles.bg} ${colorStyles.border} text-xs ${colorStyles.text}`}>
        <p className="font-semibold mb-1">{risk_level}</p>
        <p className="opacity-90">{recommendation}</p>
      </div>
    </div>
  );
};
