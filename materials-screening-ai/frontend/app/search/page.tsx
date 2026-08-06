"use client";

import { useState, useEffect } from "react";
import { Search, Sparkles, X } from "lucide-react";
import { api } from "@/lib/api";
import { PredictionCard } from "@/components/PredictionCard";
import { ConfidenceGauge } from "@/components/ConfidenceGauge";
import { Viewer3D } from "@/components/Viewer3D";
import { PredictResponsePayload, SearchResultItem } from "@/lib/types";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Card } from "@/components/ui/card";

export default function Page() {
  const [formulaQuery, setFormulaQuery] = useState("");
  const [elementQuery, setElementQuery] = useState("");
  const [results, setResults] = useState<SearchResultItem[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [screenedResult, setScreenedResult] = useState<PredictResponsePayload | null>(null);
  const [screeningId, setScreeningId] = useState<string | null>(null);

  const handleSearch = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    setIsLoading(true);
    try {
      const res = await api.searchMaterials(formulaQuery, elementQuery);
      setResults(res.materials || []);
    } catch (err) {
      console.error(err);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    let cancelled = false;
    api
      .searchMaterials("", "")
      .then((res) => {
        if (!cancelled) setResults(res.materials || []);
      })
      .catch((err) => console.error(err))
      .finally(() => {
        if (!cancelled) setIsLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const handleScreenMaterial = async (matId: string) => {
    setScreeningId(matId);
    try {
      const res = await api.screenSearchedMaterial(matId);
      setScreenedResult(res);
    } catch (err) {
      console.error(err);
    } finally {
      setScreeningId(null);
    }
  };

  return (
    <div className="space-y-6">
      <div className="space-y-1.5">
        <p className="micro-label">Structure Database</p>
        <h1 className="font-display text-2xl font-bold tracking-tight sm:text-3xl">
          Search Materials Database
        </h1>
        <p className="text-sm text-muted-foreground leading-relaxed">
          Search materials by formula (e.g. TiO2, LiFePO4, BaTiO3) or element with single-click GNN
          property screening.
        </p>
      </div>

      <form
        onSubmit={handleSearch}
        className="flex flex-col gap-3 rounded-xl border border-border bg-card p-4 shadow-sm sm:flex-row"
      >
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
          <Input
            type="text"
            placeholder="Chemical Formula (e.g. TiO2, LiFePO4, Si, MoS2)"
            value={formulaQuery}
            onChange={(e) => setFormulaQuery(e.target.value)}
            className="pl-9"
          />
        </div>

        <div className="w-full sm:w-48">
          <Input
            type="text"
            placeholder="Element (e.g. Ti, Fe)"
            value={elementQuery}
            onChange={(e) => setElementQuery(e.target.value)}
          />
        </div>

        <Button type="submit" disabled={isLoading}>
          <Search className="h-4 w-4" />
          Search
        </Button>
      </form>

      <div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-3">
        {results.map((mat) => (
          <Card key={mat.material_id} className="flex flex-col gap-3 p-5">
            <div className="flex items-center justify-between">
              <span className="font-mono text-xs font-semibold text-accent">
                {mat.material_id}
              </span>
              <Badge variant="outline">{mat.crystal_system}</Badge>
            </div>

            <div>
              <h3 className="font-display text-xl font-bold tracking-tight">
                {mat.formula_pretty || mat.formula}
              </h3>
              <p className="text-xs text-muted-foreground">{mat.name}</p>
            </div>

            <div className="space-y-1.5 rounded-lg border border-border bg-muted/60 p-3 text-xs text-muted-foreground">
              <div className="flex justify-between">
                <span>Spacegroup:</span>
                <strong className="text-foreground">{mat.spacegroup}</strong>
              </div>
              <div className="flex justify-between">
                <span>Volume:</span>
                <strong className="text-foreground">{mat.volume_A3} Å³</strong>
              </div>
              <div className="flex justify-between">
                <span>Density:</span>
                <strong className="text-foreground">{mat.density_g_cm3} g/cm³</strong>
              </div>
            </div>

            <Button
              variant="secondary"
              onClick={() => handleScreenMaterial(mat.material_id)}
              disabled={screeningId === mat.material_id}
              className="w-full"
            >
              <Sparkles className="h-4 w-4 text-primary" />
              {screeningId === mat.material_id ? "Screening..." : "1-Click Screen Material"}
            </Button>
          </Card>
        ))}
      </div>

      {screenedResult && (
        <div className="space-y-6 border-t border-border pt-6">
          <div className="flex items-center justify-between">
            <h2 className="font-display text-xl font-bold tracking-tight">
              Screening Result for {screenedResult.material_info.formula}
            </h2>
            <Button variant="ghost" size="sm" onClick={() => setScreenedResult(null)}>
              <X className="h-4 w-4" />
              Close
            </Button>
          </div>

          <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
            <div className="space-y-6 lg:col-span-2">
              <PredictionCard
                materialInfo={screenedResult.material_info}
                prediction={screenedResult.prediction}
              />
              <Viewer3D
                cifString={screenedResult.material_info.cif_string}
                formula={screenedResult.material_info.formula}
              />
            </div>

            <div>
              <ConfidenceGauge prediction={screenedResult.prediction} />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
