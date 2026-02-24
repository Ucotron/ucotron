"use client";

import { Sun, Moon, Monitor } from "lucide-react";
import { useTheme } from "./theme-provider";
import { cn } from "@/lib/utils";

const tenants = ["default", "production", "staging"];

interface HeaderProps {
  namespace: string;
  onNamespaceChange: (ns: string) => void;
}

export function Header({ namespace, onNamespaceChange }: HeaderProps) {
  const { theme, setTheme } = useTheme();

  return (
    <header className="flex h-14 items-center justify-between border-b border-border bg-card px-6">
      <div className="flex items-center gap-4">
        <label className="text-sm text-muted-foreground">Namespace</label>
        <select
          value={namespace}
          onChange={(e) => onNamespaceChange(e.target.value)}
          className="rounded-md border border-border bg-background px-3 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-primary"
        >
          {tenants.map((t) => (
            <option key={t} value={t}>{t}</option>
          ))}
        </select>
      </div>

      <div className="flex items-center gap-3">
        <div className="flex items-center gap-1 rounded-lg border border-border p-0.5">
          {[
            { value: "light" as const, icon: Sun },
            { value: "dark" as const, icon: Moon },
            { value: "system" as const, icon: Monitor },
          ].map(({ value, icon: Icon }) => (
            <button
              key={value}
              onClick={() => setTheme(value)}
              className={cn(
                "rounded-md p-1.5 transition-colors",
                theme === value
                  ? "bg-accent text-accent-foreground"
                  : "text-muted-foreground hover:text-foreground"
              )}
              title={value}
            >
              <Icon className="h-4 w-4" />
            </button>
          ))}
        </div>
      </div>
    </header>
  );
}
