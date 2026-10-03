"use client";

import React, { useState } from "react";
import { Download } from "lucide-react";
import { CandidateItem } from "@/lib/types";
import { api } from "@/lib/api";
import { cn } from "@/lib/utils";
import {
  Card,
  CardHeader,
  CardTitle,
  CardDescription,
  CardContent,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  Table,
  TableHeader,
  TableBody,
  TableRow,
  TableHead,
  TableCell,
} from "@/components/ui/table";

interface BatchTableProps {
  candidates: CandidateItem[];
}

const confidenceBadge = (confidence: string) => {
  if (confidence === 'High') return 'success' as const;
  if (confidence === 'Medium') return 'warning' as const;
  return 'destructive' as const;
};

const FILTERS = [
  { id: "all", label: "All", match: () => true },
  {
    id: "high",
    label: "High Trust",
    match: (c: CandidateItem) => c.confidence === "High",
  },
  {
    id: "medium",
    label: "Medium+",
    match: (c: CandidateItem) =>
      c.confidence === "High" || c.confidence === "Medium",
  },
] as const;

export const BatchTable: React.FC<BatchTableProps> = ({ candidates }) => {
  const [filterConfidence, setFilterConfidence] = useState<string>("all");

  const activeFilter =
    FILTERS.find((f) => f.id === filterConfidence) ?? FILTERS[0];
  const filteredCandidates = candidates.filter(activeFilter.match);

  const handleDownloadCSV = async () => {
    try {
      await api.downloadCSV(filteredCandidates);
    } catch (err) {
      console.error('CSV Export failed', err);
    }
  };

  return (
    <Card>
      <CardHeader className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <CardTitle>Ranked Screening Candidates</CardTitle>
          <CardDescription className="mt-1">
            Sorted by lowest predicted formation energy E_f
          </CardDescription>
        </div>

        <div className="flex flex-wrap items-center gap-3">
          <div
            role="group"
            aria-label="Filter candidates by confidence"
            className="inline-flex items-center gap-0.5 rounded-lg border border-border bg-muted/60 p-1"
          >
            {FILTERS.map((filter) => (
              <button
                key={filter.id}
                onClick={() => setFilterConfidence(filter.id)}
                aria-pressed={filterConfidence === filter.id}
                className={cn(
                  "rounded-md px-3 py-1.5 text-xs font-medium transition-colors duration-150 focus-ring",
                  filterConfidence === filter.id
                    ? "bg-card text-foreground shadow-xs"
                    : "text-muted-foreground hover:text-foreground",
                )}
              >
                {filter.label}
              </button>
            ))}
          </div>

          <Button onClick={handleDownloadCSV} variant="secondary" size="sm">
            <Download className="h-4 w-4" />
            Export CSV
          </Button>
        </div>
      </CardHeader>

      <CardContent>
        <div className="overflow-x-auto rounded-lg border border-border/80">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead className="text-center">Rank</TableHead>
                <TableHead>Material</TableHead>
                <TableHead className="text-right">Predicted E_f</TableHead>
                <TableHead className="text-right">Evidential σ</TableHead>
                <TableHead className="text-center">Conformal 90% Bounds (86% LOCO coverage, one held-out cluster n=7178)</TableHead>
                <TableHead className="text-center">Confidence</TableHead>
                <TableHead className="text-center">Status</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {filteredCandidates.map((item, idx) => (
                <TableRow key={idx}>
                  <TableCell className="text-center font-mono text-sm font-semibold text-primary">
                    #{item.rank || idx + 1}
                  </TableCell>
                  <TableCell>
                    <div className="text-sm font-semibold tracking-tight text-foreground">
                      {item.formula_pretty || item.formula}
                    </div>
                    <div className="text-[11px] text-muted-foreground">
                      {item.filename} · {item.num_atoms} atoms
                    </div>
                  </TableCell>
                  <TableCell className="text-right font-mono font-semibold text-foreground">
                    {item.predicted_formation_energy_per_atom_eV}{" "}
                    <span className="text-[10px] font-normal text-muted-foreground">
                      eV/atom
                    </span>
                  </TableCell>
                  <TableCell className="text-right font-mono text-muted-foreground">
                    ±{item.evidential_std_eV} eV
                  </TableCell>
                  <TableCell className="text-center font-mono text-xs text-accent">
                    [{item.conformal_90_interval_eV[0]},{" "}
                    {item.conformal_90_interval_eV[1]}]
                  </TableCell>
                  <TableCell className="text-center">
                    <Badge variant={confidenceBadge(item.confidence)} className="font-mono">
                      {item.confidence}
                    </Badge>
                  </TableCell>
                  <TableCell className="text-center text-[11px] text-muted-foreground">
                    {item.confidence === 'High' ? 'Screened' : 'DFT recommended'}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>
      </CardContent>
    </Card>
  );
};
