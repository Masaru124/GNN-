"use client";

import React from "react";
import { ShieldCheck, AlertTriangle, AlertOctagon } from "lucide-react";
import { PredictionResult } from "@/lib/types";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";

interface ConfidenceGaugeProps {
  prediction: PredictionResult;
}

type BadgeVariant = "default" | "success" | "warning" | "destructive" | "outline";

const colorStyles: Record<
  string,
  {
    badge: BadgeVariant;
    bar: string;
    card: string;
    icon: React.ComponentType<{ className?: string }>;
  }
> = {
  green: {
    badge: "success",
    bar: "bg-success",
    card: "border-success/30 bg-success/10 text-success",
    icon: ShieldCheck,
  },
  yellow: {
    badge: "warning",
    bar: "bg-warning",
    card: "border-warning/30 bg-warning/10 text-warning",
    icon: AlertTriangle,
  },
  red: {
    badge: "destructive",
    bar: "bg-destructive",
    card: "border-destructive/30 bg-destructive/10 text-destructive",
    icon: AlertOctagon,
  },
};

export const ConfidenceGauge: React.FC<ConfidenceGaugeProps> = ({ prediction }) => {
  const {
    confidence,
    confidence_score_pct,
    evidential_std_eV,
    conformal_90_interval_eV,
    risk_level,
    recommendation,
    badge_color,
  } = prediction;

  const styles = colorStyles[badge_color || 'green'];
  const IconComponent = styles.icon;

  return (
    <Card className="space-y-4 p-5">
      <div className="flex items-center justify-between gap-2">
        <h3 className="font-display text-lg font-bold tracking-tight text-foreground">
          Uncertainty & Calibration Gauge
        </h3>
        <Badge variant={styles.badge} className="whitespace-nowrap font-mono">
          <IconComponent className="h-3.5 w-3.5" />
          <span>{confidence} Confidence</span>
        </Badge>
      </div>

      <div className="space-y-1.5">
        <div className="flex justify-between text-xs font-medium">
          <span className="text-muted-foreground">Calibrated Coverage Score</span>
          <span className="font-mono text-foreground">{confidence_score_pct}%</span>
        </div>
        <div className="h-2.5 w-full overflow-hidden rounded-full bg-muted">
          <div
            className={cn("h-full rounded-full transition-all duration-500", styles.bar)}
            style={{ width: `${Math.max(5, confidence_score_pct)}%` }}
          />
        </div>
      </div>

      <div className="grid grid-cols-1 gap-3 pt-2 sm:grid-cols-2">
        <div className="rounded-lg border border-border bg-muted/60 p-3">
          <p className="text-[11px] font-medium text-muted-foreground">Evidential Std ($\sigma$)</p>
          <p className="mt-0.5 font-mono text-base font-bold text-foreground">
            ±{evidential_std_eV} <span className="text-xs font-normal text-muted-foreground">eV/atom</span>
          </p>
        </div>

        <div className="rounded-lg border border-border bg-muted/60 p-3">
          <p className="text-[11px] font-medium text-muted-foreground">Conformal 90% Interval</p>
          <p className="mt-1 font-mono text-xs font-bold text-accent">
            [{conformal_90_interval_eV[0]}, {conformal_90_interval_eV[1]}] eV
          </p>
        </div>
      </div>

      <div className={cn("rounded-lg border p-3.5 text-xs", styles.card)}>
        <p className="mb-1 font-semibold">{risk_level}</p>
        <p className="opacity-90">{recommendation}</p>
      </div>
    </Card>
  );
};
