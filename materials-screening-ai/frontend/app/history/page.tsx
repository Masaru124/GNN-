"use client";

import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import { HistoryLog } from "@/lib/types";
import {
  Table,
  TableHeader,
  TableBody,
  TableRow,
  TableHead,
  TableCell,
} from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import { Card } from "@/components/ui/card";

function confidenceVariant(confidence: string) {
  if (confidence === "High") return "success" as const;
  if (confidence === "Medium") return "warning" as const;
  return "destructive" as const;
}

export default function Page() {
  const [historyLogs, setHistoryLogs] = useState<HistoryLog[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const loadHistory = async () => {
      try {
        const res = await api.fetchHistory();
        setHistoryLogs(res.history || []);
      } catch (err) {
        console.error(err);
      } finally {
        setIsLoading(false);
      }
    };
    loadHistory();
  }, []);

  return (
    <div className="space-y-6">
      <div className="space-y-1.5">
        <p className="micro-label">Activity Log</p>
        <h1 className="font-display text-2xl font-bold tracking-tight sm:text-3xl">
          Prediction & Screening History
        </h1>
        <p className="text-sm text-muted-foreground leading-relaxed">
          Log of past material property predictions and evidential uncertainty assessments.
        </p>
      </div>

      <Card className="p-5">
        {isLoading ? (
          <div className="space-y-3 py-10 text-center">
            <div className="mx-auto h-8 w-8 animate-spin rounded-full border-2 border-primary border-t-transparent" />
            <p className="text-sm text-muted-foreground">Loading history logs...</p>
          </div>
        ) : historyLogs.length === 0 ? (
          <div className="py-10 text-center text-sm text-muted-foreground">
            No predictions recorded yet. Run a single prediction or batch screening task!
          </div>
        ) : (
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Timestamp</TableHead>
                <TableHead>Filename</TableHead>
                <TableHead>Formula</TableHead>
                <TableHead className="text-right">Predicted E_f</TableHead>
                <TableHead className="text-right">Evidential Std</TableHead>
                <TableHead className="text-center">Confidence</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {historyLogs.map((log) => (
                <TableRow key={log.id}>
                  <TableCell className="font-mono text-[11px] text-muted-foreground">
                    {log.created_at}
                  </TableCell>
                  <TableCell className="font-medium text-foreground">{log.filename}</TableCell>
                  <TableCell className="font-bold text-primary">{log.formula}</TableCell>
                  <TableCell className="text-right font-mono font-bold text-foreground">
                    {log.predicted_energy_eV} eV/atom
                  </TableCell>
                  <TableCell className="text-right font-mono text-muted-foreground">
                    ±{log.evidential_std_eV} eV
                  </TableCell>
                  <TableCell className="text-center">
                    <Badge variant={confidenceVariant(log.confidence)}>{log.confidence}</Badge>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        )}
      </Card>
    </div>
  );
}
