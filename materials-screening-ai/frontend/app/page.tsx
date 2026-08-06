"use client";

import { useState } from "react";
import Link from "next/link";
import {
  TrendingUp,
  ShieldCheck,
  Layers,
  Award,
  ArrowRight,
  Search,
  UploadCloud,
  Sparkles,
} from "lucide-react";
import { cn } from "@/lib/utils";

export default function Page() {
  const [stats] = useState({
    totalPredictions: 48,
    avgConfidencePct: 92.4,
    screeningsToday: 12,
    topCandidate: "TiO2 (Rutile)",
  });

  const kpis = [
    {
      label: "Total Predictions",
      icon: TrendingUp,
      value: String(stats.totalPredictions),
      valueClass: "text-2xl",
      iconClass: "text-chart-1",
      sub: "↑ +14% this week",
    },
    {
      label: "Average Confidence",
      icon: ShieldCheck,
      value: `${stats.avgConfidencePct}%`,
      valueClass: "text-2xl text-success",
      iconClass: "text-chart-5",
      sub: "High Trust Predictions",
    },
    {
      label: "Today's Screenings",
      icon: Layers,
      value: String(stats.screeningsToday),
      valueClass: "text-2xl",
      iconClass: "text-chart-2",
      sub: "Batches executed",
    },
    {
      label: "Top Candidate",
      icon: Award,
      value: stats.topCandidate,
      valueClass: "text-lg text-warning truncate",
      iconClass: "text-warning",
      sub: "E_f = -2.85 eV/atom",
    },
  ];

  const quickActions = [
    {
      href: "/search",
      icon: Search,
      iconClass: "bg-accent/10 text-accent border-accent/30",
      title: "Search Materials Project",
      description:
        "Query over 150,000+ crystal structures by chemical formula or element with 1-click GNN property screening.",
      cta: "Search Materials",
    },
    {
      href: "/predict",
      icon: UploadCloud,
      iconClass: "bg-primary/10 text-primary border-primary/30",
      title: "Single CIF Prediction",
      description:
        "Upload custom CIF crystal files to get instant property estimates, 3D WebGL lattice rendering, and scale attention charts.",
      cta: "Upload Structure",
    },
    {
      href: "/batch",
      icon: Layers,
      iconClass: "bg-success/10 text-success border-success/30",
      title: "Batch Candidate Screening",
      description:
        "Screen 50+ structures simultaneously, filter by confidence thresholds, rank candidates, and export CSV reports.",
      cta: "Run Batch Screening",
    },
  ];

  return (
    <div className="space-y-8">
      <div className="relative overflow-hidden rounded-xl border border-border bg-card bg-gradient-to-b from-muted/50 to-transparent shadow-sm">
        <div className="relative z-10 max-w-3xl space-y-5 p-6 sm:p-10">
          <div className="inline-flex items-center gap-2 rounded-full border border-primary/30 bg-primary/10 px-3 py-1">
            <Sparkles className="h-3.5 w-3.5 text-primary" />
            <span className="micro-label">AI-Powered Materials Screening Assistant</span>
          </div>

          <h1 className="font-display text-3xl font-bold tracking-tight sm:text-4xl">
            Accelerate Materials Discovery with{" "}
            <span className="text-primary">Calibrated GNN Confidence</span>
          </h1>

          <p className="text-sm text-muted-foreground leading-relaxed">
            Screen promising crystal structures{" "}
            <strong className="text-foreground">before expensive DFT simulations</strong>. Get
            predicted formation energy (E_f) coupled with 90% Conformal Calibration intervals and
            Evidential Uncertainty decomposition.
          </p>

          <div className="flex flex-wrap items-center gap-3 pt-1">
            <Link
              href="/predict"
              className={cn(
                "inline-flex h-11 items-center justify-center gap-2 rounded-lg bg-primary px-6 text-base font-medium text-primary-foreground shadow-sm transition-all hover:-translate-y-0.5 hover:bg-primary/90 focus-ring",
              )}
            >
              <UploadCloud className="h-4 w-4" />
              Predict Single CIF
            </Link>
            <Link
              href="/batch"
              className={cn(
                "inline-flex h-11 items-center justify-center gap-2 rounded-lg border border-border bg-background px-6 text-base font-medium text-foreground shadow-sm transition-all hover:-translate-y-0.5 hover:bg-muted focus-ring",
              )}
            >
              <Layers className="h-4 w-4 text-accent" />
              Batch Screening
            </Link>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
        {kpis.map((kpi) => {
          const Icon = kpi.icon;
          return (
            <div key={kpi.label} className="rounded-xl border border-border bg-card p-5 shadow-sm">
              <div className="flex items-center justify-between">
                <span className="micro-label">{kpi.label}</span>
                <Icon className={cn("h-4 w-4", kpi.iconClass)} />
              </div>
              <p className={cn("mt-2 font-mono font-bold tracking-tight", kpi.valueClass)}>
                {kpi.value}
              </p>
              <p className="mt-1 text-xs text-muted-foreground">{kpi.sub}</p>
            </div>
          );
        })}
      </div>

      <div className="grid grid-cols-1 gap-6 md:grid-cols-3">
        {quickActions.map((action) => {
          const Icon = action.icon;
          return (
            <Link
              key={action.href}
              href={action.href}
              className="group flex flex-col gap-3 rounded-xl border border-border bg-card p-6 shadow-sm transition-all hover:-translate-y-0.5 hover:border-primary/40 hover:bg-muted/40"
            >
              <div
                className={cn(
                  "flex h-12 w-12 items-center justify-center rounded-xl border",
                  action.iconClass,
                )}
              >
                <Icon className="h-6 w-6" />
              </div>
              <h3 className="font-display text-base font-bold tracking-tight">{action.title}</h3>
              <p className="text-sm text-muted-foreground leading-relaxed">{action.description}</p>
              <div className="mt-auto flex items-center gap-1.5 pt-1 text-xs font-semibold text-foreground">
                {action.cta}
                <ArrowRight className="h-3.5 w-3.5 transition-transform group-hover:translate-x-0.5" />
              </div>
            </Link>
          );
        })}
      </div>
    </div>
  );
}
