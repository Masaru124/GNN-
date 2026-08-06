import * as React from "react";
import type { LucideIcon } from "lucide-react";
import { cn } from "@/lib/utils";
import { Card } from "@/components/ui/card";

interface StatCardProps extends React.HTMLAttributes<HTMLDivElement> {
  label: string;
  value: React.ReactNode;
  sub?: React.ReactNode;
  icon?: LucideIcon;
  iconClass?: string;
}

export const StatCard = React.forwardRef<HTMLDivElement, StatCardProps>(
  ({ label, value, sub, icon: Icon, iconClass, className, ...props }, ref) => (
    <Card
      ref={ref}
      className={cn("p-5 transition-shadow duration-200 hover:shadow-md", className)}
      {...props}
    >
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <p className="micro-label truncate">{label}</p>
          <p className="mt-2 font-mono text-2xl font-semibold tracking-tight text-foreground">
            {value}
          </p>
          {sub && <p className="mt-1 text-xs text-muted-foreground">{sub}</p>}
        </div>
        {Icon && (
          <div
            className={cn(
              "flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-muted text-muted-foreground",
              iconClass,
            )}
          >
            <Icon className="h-[18px] w-[18px]" />
          </div>
        )}
      </div>
    </Card>
  ),
);
StatCard.displayName = "StatCard";
