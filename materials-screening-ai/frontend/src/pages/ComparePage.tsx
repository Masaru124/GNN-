import React, { useState } from 'react';
import { GitCompare, ArrowRight, UploadCloud } from 'lucide-react';
import { api } from '../services/api';

export const ComparePage: React.FC = () => {
  const [cifText, setCifText] = useState(`data_TiO2
_symmetry_space_group_name_H-M   'P 42/m n m'
_cell_length_a   4.593
_cell_length_b   4.593
_cell_length_c   2.959
_cell_angle_alpha   90.0
_cell_angle_beta    90.0
_cell_angle_gamma   90.0
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Ti1 Ti 0.0000 0.0000 0.0000
Ti2 Ti 0.5000 0.5000 0.5000
O1 O 0.3050 0.3050 0.0000
O2 O 0.6950 0.6950 0.0000
O3 O 0.8050 0.1950 0.5000
O4 O 0.1950 0.8050 0.5000`);

  const [isLoading, setIsLoading] = useState(false);
  const [comparison, setComparison] = useState<any | null>(null);

  const handleRunComparison = async () => {
    setIsLoading(true);
    try {
      const res = await api.compareModels(cifText);
      setComparison(res);
    } catch (err) {
      console.error(err);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-slate-100">Multi-Scale GNN vs Single-Scale GNN</h1>
        <p className="text-xs text-slate-400">Evaluate model explainability by comparing multi-scale fusion predictions against single-scale (4.0Å) baseline graph predictions.</p>
      </div>

      <div className="glass-card rounded-2xl p-6 border border-slate-800 space-y-4">
        <div className="flex items-center justify-between">
          <label className="text-xs font-semibold text-slate-300">Input CIF Crystal Structure Text:</label>
          <button
            onClick={handleRunComparison}
            disabled={isLoading}
            className="flex items-center gap-2 px-4 py-2 rounded-xl bg-blue-600 hover:bg-blue-500 text-white text-xs font-semibold shadow-md shadow-blue-600/20 transition-all"
          >
            <GitCompare className="w-4 h-4" />
            <span>{isLoading ? 'Running...' : 'Run Model Comparison'}</span>
          </button>
        </div>

        <textarea
          rows={8}
          value={cifText}
          onChange={(e) => setCifText(e.target.value)}
          className="w-full bg-slate-900 border border-slate-700/80 rounded-xl p-3 text-xs font-mono text-slate-200 focus:outline-none focus:border-blue-500"
        />
      </div>

      {comparison && (
        <div className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Multi Scale GNN Card */}
            <div className="glass-card rounded-2xl p-6 border border-blue-500/30 bg-blue-950/10 space-y-3">
              <span className="text-xs font-bold uppercase tracking-wider text-blue-400">Proposed Architecture</span>
              <h3 className="text-xl font-extrabold text-slate-100">Multi-Scale GNN (A7)</h3>
              <p className="text-xs text-slate-400">Incorporates 4.0Å, 6.0Å & 8.0Å coordination graph fusion with DER Uncertainty.</p>

              <div className="pt-2">
                <p className="text-xs text-slate-400">Predicted $E_f$</p>
                <p className="text-3xl font-black text-blue-400 font-mono">
                  {comparison.comparison.multi_scale.predicted_formation_energy_per_atom_eV} <span className="text-sm font-normal text-slate-300">eV/atom</span>
                </p>
              </div>

              <div className="p-3 rounded-xl bg-slate-900/60 border border-slate-800 text-xs text-slate-300">
                <span>Evidential Std: <strong>±{comparison.comparison.multi_scale.evidential_std_eV} eV</strong></span>
              </div>
            </div>

            {/* Single Scale GNN Card */}
            <div className="glass-card rounded-2xl p-6 border border-slate-800 space-y-3">
              <span className="text-xs font-bold uppercase tracking-wider text-slate-400">Baseline Architecture</span>
              <h3 className="text-xl font-extrabold text-slate-100">Single-Scale GNN (4.0Å)</h3>
              <p className="text-xs text-slate-400">Restricted to immediate 4.0Å nearest neighbor coordination sphere.</p>

              <div className="pt-2">
                <p className="text-xs text-slate-400">Predicted $E_f$</p>
                <p className="text-3xl font-black text-slate-300 font-mono">
                  {comparison.comparison.single_scale.predicted_formation_energy_per_atom_eV} <span className="text-sm font-normal text-slate-300">eV/atom</span>
                </p>
              </div>

              <div className="p-3 rounded-xl bg-slate-900/60 border border-slate-800 text-xs text-slate-400">
                <span>Refinement Difference: <strong className="text-cyan-400">{comparison.comparison.difference_eV} eV/atom</strong></span>
              </div>
            </div>
          </div>

          <div className="glass-card rounded-2xl p-5 border border-slate-800 text-xs text-slate-300 space-y-1">
            <p className="font-bold text-slate-100">Explainability Summary:</p>
            <p className="text-slate-400">{comparison.comparison.explanation}</p>
          </div>
        </div>
      )}
    </div>
  );
};
