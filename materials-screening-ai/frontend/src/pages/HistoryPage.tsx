import React, { useEffect, useState } from 'react';
import { History, FileText, Download } from 'lucide-react';
import { api } from '../services/api';

export const HistoryPage: React.FC = () => {
  const [historyLogs, setHistoryLogs] = useState<any[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const loadHistory = async () => {
      try {
        const res = await api.fetchHistory();
        setHistoryLogs(res.history || []);
      } catch (err) {
        console.error(err);
      } finally {
        setIsLoading(false);
      }
    };
    loadHistory();
  }, []);

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-slate-100">Prediction & Screening History</h1>
        <p className="text-xs text-slate-400">Log of past material property predictions and evidential uncertainty assessments.</p>
      </div>

      <div className="glass-card rounded-2xl p-6 border border-slate-800 space-y-4">
        {isLoading ? (
          <div className="text-center py-8 text-xs text-slate-400">Loading history logs...</div>
        ) : historyLogs.length === 0 ? (
          <div className="text-center py-8 text-xs text-slate-400">No predictions recorded yet. Run a single prediction or batch screening task!</div>
        ) : (
          <div className="overflow-x-auto rounded-xl border border-slate-800">
            <table className="w-full text-left text-xs text-slate-300">
              <thead className="bg-slate-900/80 text-slate-400 font-semibold uppercase tracking-wider border-b border-slate-800">
                <tr>
                  <th className="p-3.5">Timestamp</th>
                  <th className="p-3.5">Filename</th>
                  <th className="p-3.5">Formula</th>
                  <th className="p-3.5 text-right">Predicted $E_f$</th>
                  <th className="p-3.5 text-right">Evidential Std</th>
                  <th className="p-3.5 text-center">Confidence</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60">
                {historyLogs.map((log) => (
                  <tr key={log.id} className="hover:bg-slate-800/40 transition-colors">
                    <td className="p-3.5 font-mono text-slate-400 text-[11px]">{log.created_at}</td>
                    <td className="p-3.5 font-medium text-slate-200">{log.filename}</td>
                    <td className="p-3.5 font-bold text-blue-400">{log.formula}</td>
                    <td className="p-3.5 text-right font-mono font-bold text-slate-100">{log.predicted_energy_eV} eV/atom</td>
                    <td className="p-3.5 text-right font-mono text-slate-300">±{log.evidential_std_eV} eV</td>
                    <td className="p-3.5 text-center">
                      <span className="px-2.5 py-0.5 rounded-full text-[11px] font-semibold bg-blue-500/10 text-blue-400 border border-blue-500/30">
                        {log.confidence}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
};
