"use client";

import React from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";
import { Sparkles } from "lucide-react";
import { ScaleAttention } from "@/lib/types";
import { Card } from "@/components/ui/card";

interface AttentionChartProps {
  scaleAttention?: ScaleAttention;
}

type ScaleKey = "4A" | "6A" | "8A";

const chartColors: Record<ScaleKey, string> = {
  "4A": "var(--chart-1)",
  "6A": "var(--chart-2)",
  "8A": "var(--chart-3)",
};

const chartTints: Record<ScaleKey, string> = {
  "4A": "text-chart-1 bg-chart-1/10 border-chart-1/20",
  "6A": "text-chart-2 bg-chart-2/10 border-chart-2/20",
  "8A": "text-chart-3 bg-chart-3/10 border-chart-3/20",
};

export const AttentionChart: React.FC<AttentionChartProps> = ({ scaleAttention }) => {
  const weights = {
    "4A": scaleAttention?.["4A"] ?? 33.3,
    "6A": scaleAttention?.["6A"] ?? 33.3,
    "8A": scaleAttention?.["8A"] ?? 33.4,
  };

  const data: {
    key: ScaleKey;
    scale: string;
    label: string;
    weight: number;
  }[] = [
    { key: "4A", scale: "4.0 Å", label: "First shell", weight: weights["4A"] },
    { key: "6A", scale: "6.0 Å", label: "Second shell", weight: weights["6A"] },
    { key: "8A", scale: "8.0 Å", label: "Long range", weight: weights["8A"] },
  ];

  return (
    <Card className="space-y-5 p-5">
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <Sparkles className="h-4 w-4 text-accent" />
          <h3 className="text-base font-semibold tracking-tight text-foreground">
            Scale Attention
          </h3>
        </div>
        <span className="micro-label">Fusion weights</span>
      </div>

      <div className="h-[200px] w-full">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} margin={{ top: 10, right: 8, left: -24, bottom: 0 }} barCategoryGap="28%">
            <XAxis
              dataKey="scale"
              tick={{ fill: "var(--muted-foreground)", fontSize: 12, fontFamily: "var(--font-plex-mono)" }}
              tickLine={false}
              axisLine={false}
            />
            <YAxis
              tick={{ fill: "var(--muted-foreground)", fontSize: 11, fontFamily: "var(--font-plex-mono)" }}
              tickFormatter={(v: number) => `${v}%`}
              domain={[0, 100]}
              tickLine={false}
              axisLine={false}
            />
            <Tooltip
              cursor={{ fill: "var(--muted)", opacity: 0.6 }}
              contentStyle={{
                backgroundColor: "var(--card)",
                border: "1px solid var(--border)",
                borderRadius: "10px",
                color: "var(--foreground)",
                fontSize: "12px",
                fontFamily: "var(--font-plex-mono)",
                boxShadow: "0 8px 16px -4px oklch(0.24 0.01 260 / 0.1)",
              }}
              formatter={(value: number) => [`${value}%`, "Contribution"]}
              labelFormatter={(label: string) => `${label} coordination shell`}
            />
            <Bar dataKey="weight" radius={[6, 6, 0, 0]} maxBarSize={56}>
              {data.map((entry) => (
                <Cell key={entry.key} fill={chartColors[entry.key]} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div className="grid grid-cols-1 gap-2 sm:grid-cols-3">
        {data.map((entry) => (
          <div
            key={entry.key}
            className={`rounded-lg border p-2.5 text-center ${chartTints[entry.key]}`}
          >
            <p className="font-mono text-sm font-semibold">
              {weights[entry.key]}%
            </p>
            <p className="mt-0.5 text-[11px] text-muted-foreground">
              {entry.scale} · {entry.label}
            </p>
          </div>
        ))}
      </div>
    </Card>
  );
};
