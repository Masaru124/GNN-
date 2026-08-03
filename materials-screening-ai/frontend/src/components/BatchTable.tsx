import React, { useState } from 'react';
import { Download, ShieldCheck, AlertTriangle, AlertOctagon, Filter, ArrowUpDown } from 'lucide-react';
import { CandidateItem } from '../types';
import { api } from '../services/api';

interface BatchTableProps {
  candidates: CandidateItem[];
}

export const BatchTable: React.FC<BatchTableProps> = ({ candidates }) => {
  const [filterConfidence, setFilterConfidence] = useState<string>('all');
  const [sortField, setSortField] = useState<'rank' | 'energy' | 'confidence'>('rank');

  const filteredCandidates = candidates.filter((c) => {
    if (filterConfidence === 'high') return c.confidence === 'High';
    if (filterConfidence === 'medium') return c.confidence === 'High' || c.confidence === 'Medium';
    return true;
  });

  const handleDownloadCSV = async () => {
    try {
      await api.downloadCSV(filteredCandidates);
    } catch (err) {
      console.error('CSV Export failed', err);
    }
  };

  return (
    <div className="glass-card rounded-2xl p-6 border border-slate-800 space-y-4">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h3 className="text-lg font-bold text-slate-100">Ranked Screening Candidates</h3>
          <p className="text-xs text-slate-400">Sorted by lowest predicted formation energy ($E_f$)</p>
        </div>

        <div className="flex items-center gap-3">
          {/* Filter Pills */}
          <div className="flex items-center gap-1 p-1 bg-slate-900 rounded-xl border border-slate-800 text-xs">
            <button
              onClick={() => setFilterConfidence('all')}
              className={`px-3 py-1 rounded-lg font-medium transition-colors ${
                filterConfidence === 'all' ? 'bg-blue-600 text-white' : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              All ({candidates.length})
            </button>
            <button
              onClick={() => setFilterConfidence('high')}
              className={`px-3 py-1 rounded-lg font-medium transition-colors ${
                filterConfidence === 'high' ? 'bg-emerald-600 text-white' : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              High Trust
            </button>
          </div>

          <button
            onClick={handleDownloadCSV}
            className="flex items-center gap-2 px-4 py-2 rounded-xl bg-blue-600 hover:bg-blue-500 text-white text-xs font-semibold shadow-md shadow-blue-600/20 transition-all"
          >
            <Download className="w-4 h-4" />
            <span>Export CSV</span>
          </button>
        </div>
      </div>

      {/* Candidates Data Table */}
      <div className="overflow-x-auto rounded-xl border border-slate-800">
        <table className="w-full text-left text-xs text-slate-300">
          <thead className="bg-slate-900/80 text-slate-400 font-semibold uppercase tracking-wider border-b border-slate-800">
            <tr>
              <th className="p-3.5 text-center">Rank</th>
              <th className="p-3.5">Material</th>
              <th className="p-3.5 text-right">Predicted $E_f$</th>
              <th className="p-3.5 text-right">Evidential Std ($\sigma$)</th>
              <th className="p-3.5 text-center">Conformal 90% Bounds</th>
              <th className="p-3.5 text-center">Confidence</th>
              <th className="p-3.5 text-center">Action</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800/60">
            {filteredCandidates.map((item, idx) => (
              <tr key={idx} className="hover:bg-slate-800/40 transition-colors">
                <td className="p-3.5 text-center font-bold text-blue-400 font-mono">
                  #{item.rank || idx + 1}
                </td>
                <td className="p-3.5">
                  <div className="font-bold text-slate-100 text-sm">{item.formula_pretty || item.formula}</div>
                  <div className="text-[11px] text-slate-400">{item.filename} ({item.num_atoms} atoms)</div>
                </td>
                <td className="p-3.5 text-right font-mono font-bold text-slate-100">
                  {item.predicted_formation_energy_per_atom_eV} <span className="text-slate-400 text-[10px]">eV/atom</span>
                </td>
                <td className="p-3.5 text-right font-mono text-slate-300">
                  ±{item.evidential_std_eV} eV
                </td>
                <td className="p-3.5 text-center font-mono text-cyan-400 text-[11px]">
                  [{item.conformal_90_interval_eV[0]}, {item.conformal_90_interval_eV[1]}]
                </td>
                <td className="p-3.5 text-center">
                  <span
                    className={`inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-[11px] font-semibold border ${
                      item.confidence === 'High'
                        ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30'
                        : item.confidence === 'Medium'
                        ? 'bg-amber-500/10 text-amber-400 border-amber-500/30'
                        : 'bg-rose-500/10 text-rose-400 border-rose-500/30'
                    }`}
                  >
                    {item.confidence}
                  </span>
                </td>
                <td className="p-3.5 text-center">
                  <span className="text-[11px] text-slate-400 italic">
                    {item.confidence === 'High' ? 'Screened' : 'DFT Recommended'}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};
