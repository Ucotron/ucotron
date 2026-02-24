"use client";

import Link from "next/link";
import Image from "next/image";
import { usePathname } from "next/navigation";
import {
  LayoutDashboard,
  Brain,
  Users,
  GitFork,
  Settings,
  Bot,
  Plug,
  MessageSquare,
  MessagesSquare,
  Activity,
} from "lucide-react";
import { cn } from "@/lib/utils";

const navItems = [
  { href: "/overview", label: "Overview", icon: LayoutDashboard },
  { href: "/memories", label: "Memories", icon: Brain },
  { href: "/entities", label: "Entities", icon: Users },
  { href: "/graph", label: "Graph", icon: GitFork },
  { href: "/memory-chat", label: "Memory Chat", icon: MessageSquare },
  { href: "/agents", label: "Agents", icon: Bot },
  { href: "/connectors", label: "Connectors", icon: Plug },
  { href: "/conversations", label: "Conversations", icon: MessagesSquare },
  { href: "/tracing", label: "Tracing", icon: Activity },
  { href: "/settings", label: "Settings", icon: Settings },
];

export function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="flex h-full w-[260px] flex-col border-r border-border bg-muted">
      <div className="flex items-center gap-3 border-b border-border px-5 py-4">
        <Image
          src="/ucotron_logo.png"
          alt="Ucotron"
          width={28}
          height={28}
          className="rounded"
        />
        <span className="font-display text-lg font-bold uppercase tracking-wider text-primary">
          Ucotron
        </span>
      </div>

      <nav className="flex-1 space-y-0.5 p-3">
        {navItems.map((item) => {
          const active = pathname === item.href || pathname.startsWith(item.href + "/");
          return (
            <Link
              key={item.href}
              href={item.href}
              className={cn(
                "flex items-center gap-3 rounded-md px-3 py-2 text-sm font-medium transition-all",
                active
                  ? "border-l-2 border-primary bg-accent text-accent-foreground"
                  : "text-muted-foreground hover:bg-accent/50 hover:text-foreground"
              )}
            >
              <item.icon className="h-4 w-4" />
              {item.label}
            </Link>
          );
        })}
      </nav>

      <div className="border-t border-border p-4">
        <p className="text-xs text-muted-foreground">Ucotron Dashboard v0.1</p>
      </div>
    </aside>
  );
}
