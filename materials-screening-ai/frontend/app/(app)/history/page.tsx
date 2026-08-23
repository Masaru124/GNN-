"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { History, ArrowRight } from "lucide-react";
import { api } from "@/lib/api";
import { HistoryLog } from "@/lib/types";
import {
  Table,
  TableHeader,
  TableBody,
  TableRow,
  TableHead,
  TableCell,
  TableCaption,
} from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { EmptyState } from "@/components/ui/empty-state";
import { PageHeader } from "@/components/layout/PageHeader";

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
    <div className="space-y-8">
      <PageHeader
        title="Prediction & Screening History"
        description="Log of past material property predictions and evidential uncertainty assessments."
      />

      <Card className="overflow-hidden">
        {isLoading ? (
          <div className="space-y-4 p-6">
            <Skeleton className="h-8 w-full" />
            {Array.from({ length: 5 }).map((_, i) => (
              <Skeleton key={i} className="h-12 w-full" />
            ))}
            <p className="text-xs text-muted-foreground">Loading history…</p>
          </div>
        ) : historyLogs.length === 0 ? (
          <EmptyState
            icon={History}
            title="No predictions recorded yet"
            description="Run a single prediction or batch screening task to start building your history."
            action={
              <Link href="/predict">
                <Button>
                  <ArrowRight className="h-4 w-4" />
                  Run a Prediction
                </Button>
              </Link>
            }
          />
        ) : (
          <div className="p-2">
            <div className="overflow-x-auto">
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
                      <TableCell className="whitespace-nowrap font-mono text-xs text-muted-foreground">
                        {log.created_at}
                      </TableCell>
                      <TableCell className="font-medium text-foreground">
                        {log.filename}
                      </TableCell>
                      <TableCell className="font-semibold text-primary">
                        {log.formula}
                      </TableCell>
                      <TableCell className="text-right font-mono font-semibold text-foreground">
                        {log.predicted_energy_eV} eV/atom
                      </TableCell>
                      <TableCell className="text-right font-mono text-muted-foreground">
                        ±{log.evidential_std_eV} eV
                      </TableCell>
                      <TableCell className="text-center">
                        <Badge variant={confidenceVariant(log.confidence)}>
                          {log.confidence}
                        </Badge>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
                <TableCaption>Most recent predictions appear first.</TableCaption>
              </Table>
            </div>
          </div>
        )}
      </Card>
    </div>
  );
}
