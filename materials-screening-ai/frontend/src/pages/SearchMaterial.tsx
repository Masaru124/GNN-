import React, { useState, useEffect } from 'react';
import { Search, Sparkles, ArrowRight, CheckCircle2 } from 'lucide-react';
import { api } from '../services/api';
import { PredictionCard } from '../components/PredictionCard';
import { ConfidenceGauge } from '../components/ConfidenceGauge';
import { Viewer3D } from '../components/Viewer3D';
import { PredictResponsePayload } from '../types';

export const SearchMaterial: React.FC = () => {
  const [formulaQuery, setFormulaQuery] = useState('');
  const [elementQuery, setElementQuery] = useState('');
  const [results, setResults] = useState<any[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [screenedResult, setScreenedResult] = useState<PredictResponsePayload | null>(null);
  const [screeningId, setScreeningId] = useState<string | null>(null);

  const handleSearch = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    setIsLoading(true);
    try {
      const res = await api.searchMaterials(formulaQuery, elementQuery);
      setResults(res.materials || []);
    } catch (err) {
      console.error(err);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    handleSearch();
  }, []);

  const handleScreenMaterial = async (matId: string) => {
    setScreeningId(matId);
    try {
      const res = await api.screenSearchedMaterial(matId);
      setScreenedResult(res);
    } catch (err) {
      console.error(err);
    } finally {
      setScreeningId(null);
    }
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-slate-100">Search Materials Database</h1>
        <p className="text-xs text-slate-400">Search materials by formula (e.g. TiO2, LiFePO4, BaTiO3) or element with single-click GNN property screening.</p>
      </div>

      {/* Search Bar */}
      <form onSubmit={handleSearch} className="glass-card rounded-2xl p-4 border border-slate-800 flex flex-col sm:flex-row gap-3">
        <div className="relative flex-1">
          <Search className="w-4 h-4 text-slate-400 absolute left-3.5 top-1/2 -translate-y-1/2" />
          <input
            type="text"
            placeholder="Chemical Formula (e.g. TiO2, LiFePO4, Si, MoS2)"
            value={formulaQuery}
            onChange={(e) => setFormulaQuery(e.target.value)}
            className="w-full bg-slate-900 border border-slate-700/80 rounded-xl pl-10 pr-4 py-2 text-xs text-slate-100 placeholder:text-slate-500 focus:outline-none focus:border-blue-500"
          />
        </div>

        <div className="w-full sm:w-48">
          <input
            type="text"
            placeholder="Element (e.g. Ti, Fe)"
            value={elementQuery}
            onChange={(e) => setElementQuery(e.target.value)}
            className="w-full bg-slate-900 border border-slate-700/80 rounded-xl px-4 py-2 text-xs text-slate-100 placeholder:text-slate-500 focus:outline-none focus:border-blue-500"
          />
        </div>

        <button
          type="submit"
          disabled={isLoading}
          className="px-5 py-2 rounded-xl bg-blue-600 hover:bg-blue-500 text-white text-xs font-semibold shadow-md shadow-blue-600/20 transition-all flex items-center justify-center gap-2"
        >
          <Search className="w-4 h-4" />
          <span>Search</span>
        </button>
      </form>

      {/* Material Grid Results */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {results.map((mat) => (
          <div key={mat.material_id} className="glass-card rounded-2xl p-5 border border-slate-800 space-y-3 flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between">
                <span className="text-xs font-mono text-cyan-400 font-semibold">{mat.material_id}</span>
                <span className="text-[10px] px-2 py-0.5 rounded bg-slate-800 text-slate-300 font-medium">{mat.crystal_system}</span>
              </div>
              <h3 className="text-xl font-extrabold text-slate-100 mt-1">{mat.formula_pretty || mat.formula}</h3>
              <p className="text-xs text-slate-400">{mat.name}</p>
            </div>

            <div className="text-xs text-slate-400 space-y-1 bg-slate-900/60 p-3 rounded-xl border border-slate-800/80">
              <div className="flex justify-between"><span>Spacegroup:</span><strong className="text-slate-200">{mat.spacegroup}</strong></div>
              <div className="flex justify-between"><span>Volume:</span><strong className="text-slate-200">{mat.volume_A3} Å³</strong></div>
              <div className="flex justify-between"><span>Density:</span><strong className="text-slate-200">{mat.density_g_cm3} g/cm³</strong></div>
            </div>

            <button
              onClick={() => handleScreenMaterial(mat.material_id)}
              disabled={screeningId === mat.material_id}
              className="w-full flex items-center justify-center gap-2 py-2 rounded-xl bg-blue-600/20 hover:bg-blue-600 text-blue-400 hover:text-white text-xs font-semibold border border-blue-500/30 transition-all"
            >
              <Sparkles className="w-3.5 h-3.5" />
              <span>{screeningId === mat.material_id ? 'Screening...' : '1-Click Screen Material'}</span>
            </button>
          </div>
        ))}
      </div>

      {/* Screened Result Drawer / Modal section */}
      {screenedResult && (
        <div className="pt-6 space-y-6 border-t border-slate-800">
          <div className="flex items-center justify-between">
            <h2 className="text-xl font-bold text-slate-100">Screening Result for {screenedResult.material_info.formula}</h2>
            <button
              onClick={() => setScreenedResult(null)}
              className="text-xs text-slate-400 hover:text-slate-200"
            >
              Close
            </button>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div className="lg:col-span-2 space-y-6">
              <PredictionCard materialInfo={screenedResult.material_info} prediction={screenedResult.prediction} />
              <Viewer3D cifString={screenedResult.material_info.cif_string} formula={screenedResult.material_info.formula} />
            </div>

            <div>
              <ConfidenceGauge prediction={screenedResult.prediction} />
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
