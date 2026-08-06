import * as React from "react";
import { cn } from "@/lib/utils";

type BadgeVariant =
  | "default"
  | "secondary"
  | "success"
  | "warning"
  | "destructive"
  | "outline";

const variantClasses: Record<BadgeVariant, string> = {
  default: "border-primary/20 bg-primary-muted text-primary",
  secondary: "border-border bg-muted text-muted-foreground",
  success: "border-success/20 bg-success-muted text-success",
  warning: "border-warning/25 bg-warning-muted text-warning",
  destructive: "border-destructive/20 bg-destructive-muted text-destructive",
  outline: "border-border bg-transparent text-muted-foreground",
};

export interface BadgeProps extends React.HTMLAttributes<HTMLSpanElement> {
  variant?: BadgeVariant;
}

export const Badge = React.forwardRef<HTMLSpanElement, BadgeProps>(
  ({ className, variant = "default", ...props }, ref) => (
    <span
      ref={ref}
      className={cn(
        "inline-flex items-center gap-1.5 whitespace-nowrap rounded-full border px-2.5 py-0.5 text-[11px] font-semibold leading-4",
        variantClasses[variant],
        className,
      )}
      {...props}
    />
  ),
);
Badge.displayName = "Badge";
