import * as React from "react";
import { cn } from "@/lib/utils";

interface PageHeaderProps {
  eyebrow?: string;
  title: React.ReactNode;
  description?: string;
  actions?: React.ReactNode;
  className?: string;
}

export function PageHeader({
  eyebrow,
  title,
  description,
  actions,
  className,
}: PageHeaderProps) {
  return (
    <div
      className={cn(
        "flex flex-col gap-6 lg:flex-row lg:items-end lg:justify-between",
        className,
      )}
    >
      <div className="max-w-2xl space-y-2">
        {eyebrow && <p className="micro-label">{eyebrow}</p>}
        <h1 className="text-2xl font-bold tracking-tight text-foreground sm:text-3xl lg:text-[2rem]">
          {title}
        </h1>
        {description && (
          <p className="text-[15px] leading-relaxed text-muted-foreground">
            {description}
          </p>
        )}
      </div>
      {actions && (
        <div className="flex shrink-0 flex-wrap items-center gap-3">
          {actions}
        </div>
      )}
    </div>
  );
}
