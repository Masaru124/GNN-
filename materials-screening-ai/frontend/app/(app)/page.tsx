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
} from "lucide-react";
import { cn } from "@/lib/utils";
import { StatCard } from "@/components/ui/stat-card";

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
      iconClass: "bg-primary-muted text-primary",
      sub: "↑ +14% this week",
    },
    {
      label: "Average Confidence",
      icon: ShieldCheck,
      value: `${stats.avgConfidencePct}%`,
      valueClass: "text-2xl text-success",
      iconClass: "bg-success-muted text-success",
      sub: "High Trust Predictions",
    },
    {
      label: "Today's Screenings",
      icon: Layers,
      value: String(stats.screeningsToday),
      valueClass: "text-2xl",
      iconClass: "bg-accent-muted text-accent",
      sub: "Batches executed",
    },
    {
      label: "Top Candidate",
      icon: Award,
      value: stats.topCandidate,
      valueClass: "text-lg text-warning truncate",
      iconClass: "bg-warning-muted text-warning",
      sub: "E_f = -2.85 eV/atom",
    },
  ];

  const quickActions = [
    {
      href: "/search",
      icon: Search,
      iconClass: "bg-accent-muted text-accent border-accent/20",
      title: "Search Materials Project",
      description:
        "Query over 150,000+ crystal structures by chemical formula or element with 1-click GNN property screening.",
      cta: "Search Materials",
    },
    {
      href: "/predict",
      icon: UploadCloud,
      iconClass: "bg-primary-muted text-primary border-primary/20",
      title: "Single CIF Prediction",
      description:
        "Upload custom CIF crystal files to get instant property estimates, 3D WebGL lattice rendering, and scale attention charts.",
      cta: "Upload Structure",
    },
    {
      href: "/batch",
      icon: Layers,
      iconClass: "bg-success-muted text-success border-success/20",
      title: "Batch Candidate Screening",
      description:
        "Screen 50+ structures simultaneously, filter by confidence thresholds, rank candidates, and export CSV reports.",
      cta: "Run Batch Screening",
    },
  ];

  return (
    <div className="space-y-10 sm:space-y-12">
      <section className="animate-fade-up">
        <div className="max-w-3xl space-y-5">
          <h1 className="text-3xl font-bold tracking-tight sm:text-4xl">
            Accelerate Materials Discovery with{" "}
            <span className="text-primary">Calibrated GNN Confidence</span>
          </h1>

          <p className="max-w-2xl text-[15px] leading-relaxed text-muted-foreground">
            Screen promising crystal structures{" "}
            <strong className="text-foreground">
              before expensive DFT simulations
            </strong>
            . Get predicted formation energy (E_f) coupled with 90% Conformal
            Calibration intervals (86% LOCO coverage, one held-out cluster n=7178) and Evidential Uncertainty decomposition.
          </p>

          <div className="flex flex-wrap items-center gap-3 pt-1">
            <Link
              href="/predict"
              className="inline-flex h-11 items-center justify-center gap-2 rounded-lg bg-primary px-6 text-[15px] font-semibold text-primary-foreground shadow-xs transition-colors duration-150 hover:bg-primary-hover active:translate-y-px focus-ring"
            >
              <UploadCloud className="h-4 w-4" />
              Predict Single CIF
            </Link>
            <Link
              href="/batch"
              className="inline-flex h-11 items-center justify-center gap-2 rounded-lg border border-border bg-background/60 px-6 text-[15px] font-semibold text-foreground shadow-xs transition-colors duration-150 hover:bg-muted active:translate-y-px focus-ring"
            >
              <Layers className="h-4 w-4 text-accent" />
              Batch Screening
            </Link>
          </div>
        </div>
      </section>

      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 2xl:grid-cols-4">
        {kpis.map((kpi) => (
          <StatCard
            key={kpi.label}
            label={kpi.label}
            value={kpi.value}
            sub={kpi.sub}
            icon={kpi.icon}
            iconClass={kpi.iconClass}
          />
        ))}
      </div>

      <section className="space-y-6">
        <div>
          <p className="micro-label">Workspace</p>
          <h2 className="mt-2 text-xl font-bold tracking-tight sm:text-2xl">
            Quick Actions
          </h2>
        </div>

        <div className="grid grid-cols-1 gap-6 md:grid-cols-3">
          {quickActions.map((action) => {
            const Icon = action.icon;
            return (
              <Link
                key={action.href}
                href={action.href}
                className="group card-hover flex flex-col gap-3 rounded-xl border border-border bg-card p-6 focus-ring hover:border-accent/20"
              >
                <div
                  className={cn(
                    "flex h-12 w-12 items-center justify-center rounded-xl border",
                    action.iconClass,
                  )}
                >
                  <Icon className="h-6 w-6" />
                </div>
                <h3 className="text-base font-bold tracking-tight">
                  {action.title}
                </h3>
                <p className="text-sm leading-relaxed text-muted-foreground">
                  {action.description}
                </p>
                <div className="mt-auto flex items-center gap-1.5 pt-1 text-sm font-medium text-foreground">
                  {action.cta}
                  <ArrowRight className="h-4 w-4 text-muted-foreground transition-all duration-200 group-hover:translate-x-0.5 group-hover:text-primary" />
                </div>
              </Link>
            );
          })}
        </div>
      </section>
    </div>
  );
}
