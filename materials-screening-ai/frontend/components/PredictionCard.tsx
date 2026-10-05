"use client";

import React from "react";
import { Zap, Cpu, Flame, Target } from "lucide-react";
import { PredictionResult, MaterialInfo, ConformalCoverageGate } from "@/lib/types";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

interface PredictionCardProps {
  materialInfo: MaterialInfo;
  prediction: PredictionResult;
  conformalCoverageGate?: ConformalCoverageGate;
}

export const PredictionCard: React.FC<PredictionCardProps> = ({
  materialInfo,
  prediction,
  conformalCoverageGate,
}) => {
  const {
    predicted_formation_energy_per_atom_eV,
    evidential_std_eV,
    aleatoric_std_eV,
    epistemic_std_eV,
    conformal_90_interval_eV,
    confidence,
  } = prediction;

  return (
    <Card className="space-y-6 p-6">
      <div className="flex flex-col gap-4 border-b border-border/70 pb-6 sm:flex-row sm:items-start sm:justify-between">
        <div className="min-w-0">
          <div className="flex flex-wrap items-center gap-2.5">
            <h2 className="text-2xl font-bold tracking-tight text-foreground">
              {materialInfo.formula_pretty || materialInfo.formula}
            </h2>
            <Badge variant="secondary" className="font-mono">
              {materialInfo.filename}
            </Badge>
          </div>
          <p className="mt-2 text-[13px] text-muted-foreground">
            {materialInfo.num_atoms} atoms · Density: {materialInfo.density_g_cm3}{" "}
            g/cm³ · Volume: {materialInfo.volume_A3} Å³
          </p>
        </div>

        <div className="text-left sm:text-right">
          <p className="micro-label">Formation Energy E_f</p>
          <p className="mt-1.5 font-mono text-3xl font-semibold tracking-tight text-primary">
            {predicted_formation_energy_per_atom_eV}{" "}
            <span className="text-base font-normal text-muted-foreground">
              eV/atom
            </span>
          </p>
        </div>
      </div>

      <div className="flex flex-col gap-4 rounded-lg border border-primary/20 bg-primary-muted p-4 sm:flex-row sm:items-center sm:justify-between">
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl border border-primary/25 bg-card text-primary">
            <Target className="h-5 w-5" />
          </div>
          <div>
            <p className="micro-label">Calibrated Property Answer</p>
            <p className="mt-0.5 font-mono text-sm font-semibold text-foreground">
              E_f = {predicted_formation_energy_per_atom_eV} eV ±
              {evidential_std_eV} eV · {confidence} Confidence
            </p>
          </div>
        </div>

        <div className="shrink-0">
          <p className="micro-label">Conformal 90% Bounds (LOCO 10-fold, nested: 89.6% pooled / 92.2% macro at q_LOFO; 71.2% at shipped q; worst fold 7)</p>
          <p className="mt-0.5 font-mono text-sm font-semibold text-accent">
            [{conformal_90_interval_eV[0]}, {conformal_90_interval_eV[1]}] eV
          </p>
          {conformalCoverageGate && (
            <p
              className="mt-1 text-[11px] text-muted-foreground"
              title={`Class-level coverage claims need at least ${conformalCoverageGate.min_n_cal} calibration points. This class has n_cal = ${conformalCoverageGate.n_cal}.`}
            >
              Class: {conformalCoverageGate.chemistry_class} · n_cal ={" "}
              {conformalCoverageGate.n_cal} ·{" "}
              {conformalCoverageGate.status === "calibrated"
                ? "class-level coverage permitted"
                : "marginal coverage only"}
            </p>
          )}
        </div>
      </div>

      <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
        <div className="rounded-lg border border-border bg-muted/50 p-4">
          <div className="mb-2 flex items-center gap-1.5 text-xs font-medium text-muted-foreground">
            <Zap className="h-3.5 w-3.5 text-warning" />
            <span>Evidential Std σ</span>
          </div>
          <p className="font-mono text-xl font-semibold text-foreground">
            ±{evidential_std_eV}{" "}
            <span className="text-xs font-normal text-muted-foreground">eV</span>
          </p>
          <p className="mt-1 text-[11px] text-muted-foreground">
            Total calibrated variance
          </p>
        </div>

        <div className="rounded-lg border border-border bg-muted/50 p-4">
          <div className="mb-2 flex items-center gap-1.5 text-xs font-medium text-muted-foreground">
            <Flame className="h-3.5 w-3.5 text-destructive" />
            <span>Aleatoric Uncertainty</span>
          </div>
          <p className="font-mono text-xl font-semibold text-foreground">
            ±{aleatoric_std_eV}{" "}
            <span className="text-xs font-normal text-muted-foreground">eV</span>
          </p>
          <p className="mt-1 text-[11px] text-muted-foreground">
            Data / structural noise
          </p>
        </div>

        <div className="rounded-lg border border-border bg-muted/50 p-4">
          <div className="mb-2 flex items-center gap-1.5 text-xs font-medium text-muted-foreground">
            <Cpu className="h-3.5 w-3.5 text-accent" />
            <span>Epistemic Uncertainty</span>
          </div>
          <p className="font-mono text-xl font-semibold text-foreground">
            ±{epistemic_std_eV}{" "}
            <span className="text-xs font-normal text-muted-foreground">eV</span>
          </p>
          <p className="mt-1 text-[11px] text-muted-foreground">
            Model parameter uncertainty
          </p>
        </div>
      </div>
    </Card>
  );
};
