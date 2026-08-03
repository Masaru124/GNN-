import React from 'react';
import { Zap, ShieldCheck, Cpu, Flame, Target } from 'lucide-react';
import { PredictionResult, MaterialInfo } from '../types';

interface PredictionCardProps {
  materialInfo: MaterialInfo;
  prediction: PredictionResult;
}

export const PredictionCard: React.FC<PredictionCardProps> = ({ materialInfo, prediction }) => {
  const {
    predicted_formation_energy_per_atom_eV,
    evidential_std_eV,
    aleatoric_std_eV,
    epistemic_std_eV,
    conformal_90_interval_eV,
    conformal_width_eV,
    confidence,
    badge_color,
  } = prediction;

  return (
    <div className="glass-card rounded-2xl p-6 border border-slate-800 space-y-6">
      {/* Header Info */}
      <div className="flex items-start justify-between border-b border-slate-800 pb-4">
        <div>
          <div className="flex items-center gap-2">
            <h2 className="text-2xl font-extrabold text-slate-100">{materialInfo.formula_pretty || materialInfo.formula}</h2>
            <span className="text-xs px-2.5 py-1 rounded-md bg-slate-800 text-slate-300 font-mono">
              {materialInfo.filename}
            </span>
          </div>
          <p className="text-xs text-slate-400 mt-1">
            {materialInfo.num_atoms} atoms | Density: {materialInfo.density_g_cm3} g/cm³ | Volume: {materialInfo.volume_A3} Å³
          </p>
        </div>

        <div className="text-right">
          <p className="text-xs text-slate-400 font-medium uppercase tracking-wider">Formation Energy ($E_f$)</p>
          <p className="text-3xl font-black text-blue-400 font-mono tracking-tight mt-0.5">
            {predicted_formation_energy_per_atom_eV} <span className="text-base font-normal text-slate-300">eV/atom</span>
          </p>
        </div>
      </div>

      {/* Primary Value Proposition Banner */}
      <div className="p-4 rounded-xl bg-gradient-to-r from-blue-900/30 via-slate-900 to-indigo-900/30 border border-blue-500/20 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-blue-500/20 border border-blue-400/30 flex items-center justify-center text-blue-400">
            <Target className="w-5 h-5" />
          </div>
          <div>
            <p className="text-xs font-semibold text-slate-300">Calibrated Property Answer</p>
            <p className="text-sm font-bold text-slate-100 font-mono">
              $E_f$ = {predicted_formation_energy_per_atom_eV} eV ±{evidential_std_eV} eV ({confidence} Confidence)
            </p>
          </div>
        </div>

        <div className="text-right font-mono text-xs text-slate-300">
          <p className="text-[10px] text-slate-400 uppercase">Conformal 90% Bounds</p>
          <p className="font-bold text-cyan-400">[{conformal_90_interval_eV[0]}, {conformal_90_interval_eV[1]}] eV</p>
        </div>
      </div>

      {/* Uncertainty Breakdown Cards */}
      <div className="grid grid-cols-3 gap-3">
        <div className="p-3.5 rounded-xl bg-slate-900/60 border border-slate-800">
          <div className="flex items-center gap-1.5 text-xs text-slate-400 mb-1">
            <Zap className="w-3.5 h-3.5 text-amber-400" />
            <span>Evidential Std ($\sigma$)</span>
          </div>
          <p className="text-lg font-bold text-slate-100 font-mono">±{evidential_std_eV} eV</p>
          <p className="text-[10px] text-slate-400 mt-1">Total Calibrated Variance</p>
        </div>

        <div className="p-3.5 rounded-xl bg-slate-900/60 border border-slate-800">
          <div className="flex items-center gap-1.5 text-xs text-slate-400 mb-1">
            <Flame className="w-3.5 h-3.5 text-rose-400" />
            <span>Aleatoric Uncertainty</span>
          </div>
          <p className="text-lg font-bold text-slate-100 font-mono">±{aleatoric_std_eV} eV</p>
          <p className="text-[10px] text-slate-400 mt-1">Data / Structural Noise</p>
        </div>

        <div className="p-3.5 rounded-xl bg-slate-900/60 border border-slate-800">
          <div className="flex items-center gap-1.5 text-xs text-slate-400 mb-1">
            <Cpu className="w-3.5 h-3.5 text-cyan-400" />
            <span>Epistemic Uncertainty</span>
          </div>
          <p className="text-lg font-bold text-slate-100 font-mono">±{epistemic_std_eV} eV</p>
          <p className="text-[10px] text-slate-400 mt-1">Model Parameter Uncertainty</p>
        </div>
      </div>
    </div>
  );
};
