"use client";

import { useState } from "react";
import { GitCompare } from "lucide-react";
import { api } from "@/lib/api";
import { ModelComparison } from "@/lib/types";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Card } from "@/components/ui/card";
import { PageHeader } from "@/components/layout/PageHeader";

const DEFAULT_CIF = `# generated using pymatgen
data_TiO2
_symmetry_space_group_name_H-M   'P 1'
_cell_length_a   4.59300000
_cell_length_b   4.59300000
_cell_length_c   2.95900000
_cell_angle_alpha   90.00000000
_cell_angle_beta   90.00000000
_cell_angle_gamma   90.00000000
_symmetry_Int_Tables_number   1
_chemical_formula_structural   TiO2
_chemical_formula_sum   'Ti2 O4'
loop_
 _symmetry_equiv_pos_site_id
 _symmetry_equiv_pos_as_xyz
  1  'x, y, z'
loop_
 _atom_site_type_symbol
 _atom_site_label
 _atom_site_symmetry_multiplicity
 _atom_site_fract_x
 _atom_site_fract_y
 _atom_site_fract_z
 _atom_site_occupancy
  Ti  Ti0  1  0.00000000  0.00000000  0.00000000  1
  Ti  Ti1  1  0.50000000  0.50000000  0.50000000  1
  O  O2  1  0.30500000  0.30500000  0.00000000  1
  O  O3  1  0.69500000  0.69500000  0.00000000  1
  O  O4  1  0.80500000  0.19500000  0.50000000  1
  O  O5  1  0.19500000  0.80500000  0.50000000  1`;

export default function Page() {
  const [cifText, setCifText] = useState(DEFAULT_CIF);
  const [isLoading, setIsLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [comparison, setComparison] = useState<ModelComparison | null>(null);

  const handleRunComparison = async () => {
    setIsLoading(true);
    setErrorMsg(null);
    try {
      const res = await api.compareModels(cifText);
      setComparison(res);
    } catch (err: any) {
      const detail = err.response?.data?.detail || err.message || "Model comparison failed.";
      setErrorMsg(detail);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="space-y-8">
      <PageHeader
        eyebrow="Architecture Ablation"
        title="Multi-Scale GNN vs Single-Scale GNN"
        description="Evaluate model explainability by comparing multi-scale fusion predictions against single-scale (4.0Å) baseline graph predictions."
        actions={
          <Button onClick={handleRunComparison} disabled={isLoading}>
            <GitCompare className="h-4 w-4" />
            {isLoading ? "Running…" : "Run Model Comparison"}
          </Button>
        }
      />

      {errorMsg && (
        <div className="rounded-xl border border-rose-500/30 bg-rose-500/10 p-4 text-xs text-rose-600 dark:text-rose-400 font-semibold">
          {errorMsg}
        </div>
      )}

      <Card className="space-y-4 p-5">
        <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
          <label htmlFor="cif-input" className="text-sm font-semibold text-foreground">
            Input CIF Crystal Structure Text
          </label>
        </div>

        <Textarea
          id="cif-input"
          rows={10}
          value={cifText}
          onChange={(e) => setCifText(e.target.value)}
          className="font-mono text-xs leading-relaxed"
        />

        <p className="text-xs text-muted-foreground">Paste a full CIF crystallographic file</p>
      </Card>

      {comparison && (
        <div className="animate-fade-up space-y-6">
          <div className="grid grid-cols-1 gap-6 md:grid-cols-2">
            <Card className="space-y-4 border-primary/20 bg-primary-muted p-6">
              <p className="micro-label text-primary">Proposed Architecture</p>
              <h3 className="text-xl font-bold tracking-tight">Multi-Scale GNN (A7)</h3>
              <p className="text-sm leading-relaxed text-muted-foreground">
                Incorporates 4.0Å, 6.0Å & 8.0Å coordination graph fusion with DER Uncertainty.
              </p>

              <div className="pt-1">
                <p className="text-xs text-muted-foreground">Predicted E_f</p>
                <p className="font-mono text-3xl font-semibold text-primary">
                  {comparison.comparison.multi_scale.predicted_formation_energy_per_atom_eV}{" "}
                  <span className="text-sm font-normal text-muted-foreground">eV/atom</span>
                </p>
              </div>

              <div className="rounded-lg border border-border bg-card/60 p-3.5 text-xs text-muted-foreground">
                Evidential std:{" "}
                <strong className="font-mono text-foreground">
                  ±{comparison.comparison.multi_scale.evidential_std_eV} eV
                </strong>
              </div>
            </Card>

            <Card className="space-y-4 p-6">
              <p className="micro-label">Baseline Architecture</p>
              <h3 className="text-xl font-bold tracking-tight">Single-Scale GNN (4.0Å)</h3>
              <p className="text-sm leading-relaxed text-muted-foreground">
                Restricted to immediate 4.0Å nearest neighbor coordination sphere.
              </p>

              <div className="pt-1">
                <p className="text-xs text-muted-foreground">Predicted E_f</p>
                <p className="font-mono text-3xl font-semibold text-foreground">
                  {comparison.comparison.single_scale.predicted_formation_energy_per_atom_eV}{" "}
                  <span className="text-sm font-normal text-muted-foreground">eV/atom</span>
                </p>
              </div>

              <div className="rounded-lg border border-border bg-muted/60 p-3.5 text-xs text-muted-foreground">
                Refinement Difference:{" "}
                <strong className="font-mono text-accent">
                  {comparison.comparison.difference_eV} eV/atom
                </strong>
              </div>
            </Card>
          </div>

          <Card className="space-y-2 p-6">
            <p className="text-sm font-semibold text-foreground">Explainability Summary:</p>
            <p className="leading-relaxed text-muted-foreground">
              {comparison.comparison.explanation}
            </p>
          </Card>
        </div>
      )}
    </div>
  );
}
