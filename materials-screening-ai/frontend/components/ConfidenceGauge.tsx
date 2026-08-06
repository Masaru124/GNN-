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

type BadgeVariant = "default" | "secondary" | "success" | "warning" | "destructive" | "outline";

const colorStyles: Record<
  string,
  {
    badge: BadgeVariant;
    bar: string;
    accentText: string;
    icon: React.ComponentType<{ className?: string }>;
  }
> = {
  green: {
    badge: "success",
    bar: "bg-success",
    accentText: "text-success",
    icon: ShieldCheck,
  },
  yellow: {
    badge: "warning",
    bar: "bg-warning",
    accentText: "text-warning",
    icon: AlertTriangle,
  },
  red: {
    badge: "destructive",
    bar: "bg-destructive",
    accentText: "text-destructive",
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

  const styles = colorStyles[badge_color || "green"];
  const IconComponent = styles.icon;

  return (
    <Card className="space-y-5 p-5">
      <div className="flex items-center justify-between gap-2">
        <h3 className="text-base font-semibold tracking-tight text-foreground">
          Confidence Gauge
        </h3>
        <Badge variant={styles.badge} className="font-mono">
          <IconComponent className="h-3 w-3" />
          {confidence} Confidence
        </Badge>
      </div>

      <div className="space-y-2">
        <div className="flex items-baseline justify-between gap-2">
          <span className="text-[13px] text-muted-foreground">
            Calibrated coverage score
          </span>
          <span className={cn("font-mono text-sm font-semibold", styles.accentText)}>
            {confidence_score_pct}%
          </span>
        </div>
        <div
          className="h-2 w-full overflow-hidden rounded-full bg-muted"
          role="progressbar"
          aria-valuenow={confidence_score_pct}
          aria-valuemin={0}
          aria-valuemax={100}
          aria-label={`Confidence score ${confidence_score_pct} percent`}
        >
          <div
            className={cn("h-full rounded-full transition-all duration-500 ease-out", styles.bar)}
            style={{ width: `${Math.max(5, confidence_score_pct)}%` }}
          />
        </div>
      </div>

      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
        <div className="rounded-lg border border-border bg-muted/50 p-3.5">
          <p className="text-[11px] font-medium text-muted-foreground">
            Evidential std σ
          </p>
          <p className="mt-0.5 font-mono text-base font-semibold text-foreground">
            ±{evidential_std_eV}{" "}
            <span className="text-xs font-normal text-muted-foreground">eV/atom</span>
          </p>
        </div>

        <div className="rounded-lg border border-border bg-muted/50 p-3.5">
          <p className="text-[11px] font-medium text-muted-foreground">
            Conformal 90% interval
          </p>
          <p className="mt-1 font-mono text-xs font-semibold text-accent">
            [{conformal_90_interval_eV[0]}, {conformal_90_interval_eV[1]}] eV
          </p>
        </div>
      </div>

      <div
        className={cn(
          "rounded-lg border border-border bg-muted/50 p-4",
        )}
      >
        <p className={cn("flex items-center gap-1.5 text-[13px] font-semibold", styles.accentText)}>
          <IconComponent className="h-3.5 w-3.5" />
          {risk_level}
        </p>
        <p className="mt-1.5 text-[13px] leading-relaxed text-muted-foreground">
          {recommendation}
        </p>
      </div>
    </Card>
  );
};
