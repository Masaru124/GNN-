import React from 'react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import { Sparkles } from 'lucide-react';
import { ScaleAttention } from '../types';

interface AttentionChartProps {
  scaleAttention?: ScaleAttention;
}

export const AttentionChart: React.FC<AttentionChartProps> = ({ scaleAttention }) => {
  const data = [
    { scale: '4.0 Å (First Shell)', weight: scaleAttention?.['4A'] ?? 33.3, color: '#3b82f6' },
    { scale: '6.0 Å (Second Shell)', weight: scaleAttention?.['6A'] ?? 33.3, color: '#06b6d4' },
    { scale: '8.0 Å (Long Range)', weight: scaleAttention?.['8A'] ?? 33.4, color: '#10b981' },
  ];

  return (
    <div className="glass-card rounded-2xl p-5 border border-slate-800 space-y-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Sparkles className="w-4 h-4 text-cyan-400" />
          <h3 className="text-sm font-semibold text-slate-200">Scale Attention Explainability</h3>
        </div>
        <span className="text-xs text-slate-400">Multi-Scale Fusion Weights</span>
      </div>

      <div className="h-[200px] w-full pt-2">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
            <XAxis dataKey="scale" stroke="#64748b" fontSize={11} tickLine={false} />
            <YAxis stroke="#64748b" fontSize={11} tickFormatter={(v) => `${v}%`} domain={[0, 100]} />
            <Tooltip
              contentStyle={{ backgroundColor: '#0b0f17', borderColor: '#334155', borderRadius: '8px', color: '#f8fafc' }}
              formatter={(value: any) => [`${value}%`, 'Contribution Weight']}
            />
            <Bar dataKey="weight" radius={[6, 6, 0, 0]}>
              {data.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={entry.color} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div className="grid grid-cols-3 gap-2 text-center text-xs pt-1">
        <div className="p-2 rounded-lg bg-blue-500/10 border border-blue-500/20 text-blue-400">
          <p className="font-bold text-sm">{scaleAttention?.['4A'] ?? 33.3}%</p>
          <p className="text-[10px] text-slate-400 mt-0.5">4.0 Å Local</p>
        </div>
        <div className="p-2 rounded-lg bg-cyan-500/10 border border-cyan-500/20 text-cyan-400">
          <p className="font-bold text-sm">{scaleAttention?.['6A'] ?? 33.3}%</p>
          <p className="text-[10px] text-slate-400 mt-0.5">6.0 Å Medium</p>
        </div>
        <div className="p-2 rounded-lg bg-emerald-500/10 border border-emerald-500/20 text-emerald-400">
          <p className="font-bold text-sm">{scaleAttention?.['8A'] ?? 33.4}%</p>
          <p className="text-[10px] text-slate-400 mt-0.5">8.0 Å Extended</p>
        </div>
      </div>
    </div>
  );
};
