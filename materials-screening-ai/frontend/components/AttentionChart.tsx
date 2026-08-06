"use client";

import React from "react";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from "recharts";
import { Sparkles } from "lucide-react";
import { ScaleAttention } from "@/lib/types";
import { Card } from "@/components/ui/card";

interface AttentionChartProps {
  scaleAttention?: ScaleAttention;
}

export const AttentionChart: React.FC<AttentionChartProps> = ({ scaleAttention }) => {
  const data = [
    { scale: '4.0 Å (First Shell)', weight: scaleAttention?.['4A'] ?? 33.3, color: 'var(--chart-1)' },
    { scale: '6.0 Å (Second Shell)', weight: scaleAttention?.['6A'] ?? 33.3, color: 'var(--chart-2)' },
    { scale: '8.0 Å (Long Range)', weight: scaleAttention?.['8A'] ?? 33.4, color: 'var(--chart-3)' },
  ];

  return (
    <Card className="space-y-4 p-5">
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <Sparkles className="h-4 w-4 text-accent" />
          <h3 className="font-display text-lg font-bold tracking-tight text-foreground">
            Scale Attention Explainability
          </h3>
        </div>
        <span className="micro-label">Multi-Scale Fusion Weights</span>
      </div>

      <div className="h-[200px] w-full pt-2">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
            <XAxis dataKey="scale" stroke="var(--muted-foreground)" fontSize={11} tickLine={false} />
            <YAxis stroke="var(--muted-foreground)" fontSize={11} tickFormatter={(v) => `${v}%`} domain={[0, 100]} />
            <Tooltip
              contentStyle={{ backgroundColor: 'var(--card)', borderColor: 'var(--border)', borderRadius: '8px', color: 'var(--foreground)' }}
              formatter={(value) => [`${value}%`, 'Contribution Weight']}
            />
            <Bar dataKey="weight" radius={[6, 6, 0, 0]}>
              {data.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={entry.color} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div className="grid grid-cols-1 gap-2 text-center text-xs pt-1 sm:grid-cols-3">
        <div className="rounded-lg border border-chart-1/30 bg-chart-1/10 text-chart-1 p-2">
          <p className="font-mono text-sm font-bold">{scaleAttention?.['4A'] ?? 33.3}%</p>
          <p className="mt-0.5 text-[10px] text-muted-foreground">4.0 Å Local</p>
        </div>
        <div className="rounded-lg border border-chart-2/30 bg-chart-2/10 text-chart-2 p-2">
          <p className="font-mono text-sm font-bold">{scaleAttention?.['6A'] ?? 33.3}%</p>
          <p className="mt-0.5 text-[10px] text-muted-foreground">6.0 Å Medium</p>
        </div>
        <div className="rounded-lg border border-chart-3/30 bg-chart-3/10 text-chart-3 p-2">
          <p className="font-mono text-sm font-bold">{scaleAttention?.['8A'] ?? 33.4}%</p>
          <p className="mt-0.5 text-[10px] text-muted-foreground">8.0 Å Extended</p>
        </div>
      </div>
    </Card>
  );
};
