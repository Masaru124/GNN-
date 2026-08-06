"use client";

import React from "react";
import { Zap, Cpu, Flame, Target } from "lucide-react";
import { PredictionResult, MaterialInfo } from "@/lib/types";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

interface PredictionCardProps {
  materialInfo: MaterialInfo;
  prediction: PredictionResult;
}

export const PredictionCard: React.FC<PredictionCardProps> = ({
  materialInfo,
  prediction,
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
      <div className="flex flex-col gap-4 border-b border-border pb-4 sm:flex-row sm:items-start sm:justify-between">
        <div>
          <div className="flex flex-wrap items-center gap-2">
            <h2 className="font-display text-2xl font-bold tracking-tight text-foreground">
              {materialInfo.formula_pretty || materialInfo.formula}
            </h2>
            <Badge variant="outline" className="font-mono">
              {materialInfo.filename}
            </Badge>
          </div>
          <p className="mt-1 text-xs text-muted-foreground">
            {materialInfo.num_atoms} atoms | Density: {materialInfo.density_g_cm3} g/cm³ | Volume: {materialInfo.volume_A3} Å³
          </p>
        </div>

        <div className="text-left sm:text-right">
          <p className="micro-label">Formation Energy ($E_f$)</p>
          <p className="mt-0.5 font-mono text-3xl font-bold tracking-tight text-primary">
            {predicted_formation_energy_per_atom_eV}{' '}
            <span className="text-base font-normal text-muted-foreground">eV/atom</span>
          </p>
        </div>
      </div>

      <div className="flex flex-col gap-4 rounded-lg border border-primary/30 bg-primary/10 p-4 sm:flex-row sm:items-center sm:justify-between">
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-xl border border-primary/30 bg-primary/10 text-primary">
            <Target className="h-5 w-5" />
          </div>
          <div>
            <p className="micro-label">Calibrated Property Answer</p>
            <p className="font-mono text-sm font-semibold text-foreground">
              $E_f$ = {predicted_formation_energy_per_atom_eV} eV ±{evidential_std_eV} eV ({confidence} Confidence)
            </p>
          </div>
        </div>

        <div className="font-mono text-xs text-muted-foreground">
          <p className="micro-label">Conformal 90% Bounds</p>
          <p className="font-semibold text-accent">
            [{conformal_90_interval_eV[0]}, {conformal_90_interval_eV[1]}] eV
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
        <div className="rounded-lg border border-border bg-muted/60 p-3.5">
          <div className="mb-1 flex items-center gap-1.5 text-xs text-muted-foreground">
            <Zap className="h-3.5 w-3.5 text-warning" />
            <span>Evidential Std ($\sigma$)</span>
          </div>
          <p className="font-mono text-lg font-bold text-foreground">±{evidential_std_eV} eV</p>
          <p className="mt-1 text-[10px] text-muted-foreground">Total Calibrated Variance</p>
        </div>

        <div className="rounded-lg border border-border bg-muted/60 p-3.5">
          <div className="mb-1 flex items-center gap-1.5 text-xs text-muted-foreground">
            <Flame className="h-3.5 w-3.5 text-destructive" />
            <span>Aleatoric Uncertainty</span>
          </div>
          <p className="font-mono text-lg font-bold text-foreground">±{aleatoric_std_eV} eV</p>
          <p className="mt-1 text-[10px] text-muted-foreground">Data / Structural Noise</p>
        </div>

        <div className="rounded-lg border border-border bg-muted/60 p-3.5">
          <div className="mb-1 flex items-center gap-1.5 text-xs text-muted-foreground">
            <Cpu className="h-3.5 w-3.5 text-accent" />
            <span>Epistemic Uncertainty</span>
          </div>
          <p className="font-mono text-lg font-bold text-foreground">±{epistemic_std_eV} eV</p>
          <p className="mt-1 text-[10px] text-muted-foreground">Model Parameter Uncertainty</p>
        </div>
      </div>
    </Card>
  );
};
