import { cn } from "@/lib/utils";
import type { ReactNode } from "react";

interface CardProps {
  title?: string;
  className?: string;
  children: ReactNode;
}

export function Card({ title, className, children }: CardProps) {
  return (
    <div className={cn("glass-card p-4", className)}>
      {title && (
        <h3 className="mb-3 text-sm font-medium text-muted-foreground">{title}</h3>
      )}
      {children}
    </div>
  );
}

interface StatCardProps {
  label: string;
  value: string | number;
  icon?: ReactNode;
  trend?: "up" | "down" | "neutral";
}

export function StatCard({ label, value, icon }: StatCardProps) {
  return (
    <div className="glass-card flex items-start gap-3 p-4 transition-all hover:border-[rgba(0,240,255,0.2)] hover:shadow-[0_0_10px_rgba(0,240,255,0.2)]">
      {icon && (
        <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-md bg-accent text-accent-foreground">
          {icon}
        </div>
      )}
      <div>
        <p className="text-sm text-muted-foreground">{label}</p>
        <p className="text-2xl font-semibold">{value}</p>
      </div>
    </div>
  );
}
