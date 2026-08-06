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

export const BatchTable: React.FC<BatchTableProps> = ({ candidates }) => {
  const [filterConfidence, setFilterConfidence] = useState<string>('all');

  const filteredCandidates = candidates.filter((c) => {
    if (filterConfidence === 'high') return c.confidence === 'High';
    if (filterConfidence === 'medium') return c.confidence === 'High' || c.confidence === 'Medium';
    return true;
  });

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
            Sorted by lowest predicted formation energy ($E_f$)
          </CardDescription>
        </div>

        <div className="flex items-center gap-3">
          <div className="flex items-center gap-1 rounded-lg border border-border bg-muted/60 p-1 text-xs">
            <button
              onClick={() => setFilterConfidence('all')}
              className={cn(
                "rounded-md px-3 py-1 font-medium transition-colors",
                filterConfidence === 'all'
                  ? 'bg-primary text-primary-foreground'
                  : 'text-muted-foreground hover:text-foreground',
              )}
            >
              All ({candidates.length})
            </button>
            <button
              onClick={() => setFilterConfidence('high')}
              className={cn(
                "rounded-md px-3 py-1 font-medium transition-colors",
                filterConfidence === 'high'
                  ? 'bg-success text-success-foreground'
                  : 'text-muted-foreground hover:text-foreground',
              )}
            >
              High Trust
            </button>
          </div>

          <Button onClick={handleDownloadCSV} variant="secondary" size="sm">
            <Download className="h-4 w-4" />
            <span>Export CSV</span>
          </Button>
        </div>
      </CardHeader>

      <CardContent>
        <div className="overflow-x-auto rounded-lg border border-border">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead className="text-center">Rank</TableHead>
                <TableHead>Material</TableHead>
                <TableHead className="text-right">Predicted $E_f$</TableHead>
                <TableHead className="text-right">Evidential Std ($\sigma$)</TableHead>
                <TableHead className="text-center">Conformal 90% Bounds</TableHead>
                <TableHead className="text-center">Confidence</TableHead>
                <TableHead className="text-center">Action</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {filteredCandidates.map((item, idx) => (
                <TableRow key={idx}>
                  <TableCell className="text-center font-mono font-bold text-primary">
                    #{item.rank || idx + 1}
                  </TableCell>
                  <TableCell>
                    <div className="font-display text-sm font-semibold tracking-tight text-foreground">
                      {item.formula_pretty || item.formula}
                    </div>
                    <div className="text-[11px] text-muted-foreground">
                      {item.filename} ({item.num_atoms} atoms)
                    </div>
                  </TableCell>
                  <TableCell className="text-right font-mono font-bold text-foreground">
                    {item.predicted_formation_energy_per_atom_eV}{' '}
                    <span className="text-[10px] font-normal text-muted-foreground">eV/atom</span>
                  </TableCell>
                  <TableCell className="text-right font-mono text-muted-foreground">
                    ±{item.evidential_std_eV} eV
                  </TableCell>
                  <TableCell className="text-center font-mono text-[11px] text-accent">
                    [{item.conformal_90_interval_eV[0]}, {item.conformal_90_interval_eV[1]}]
                  </TableCell>
                  <TableCell className="text-center">
                    <Badge variant={confidenceBadge(item.confidence)} className="font-mono">
                      {item.confidence}
                    </Badge>
                  </TableCell>
                  <TableCell className="text-center text-[11px] italic text-muted-foreground">
                    {item.confidence === 'High' ? 'Screened' : 'DFT Recommended'}
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
