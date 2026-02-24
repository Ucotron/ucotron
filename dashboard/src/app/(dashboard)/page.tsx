"use client";

import { useEffect, useState, useCallback } from "react";
import Link from "next/link";
import {
  Brain,
  GitFork,
  Search,
  MessageSquare,
  Database,
  Plus,
  ArrowRight,
  RefreshCw,
  Clock,
  FileText,
  Users,
  Zap,
} from "lucide-react";
import { Card, StatCard } from "@/components/card";
import { getMetrics, listNamespaces, getRecentMemories } from "@/lib/api";
import type { MetricsResponse, NamespaceListResponse, MemoryResponse } from "@/lib/api";

function Skeleton() {
  return (
    <div className="animate-pulse">
      <div className="h-4 w-24 rounded bg-muted" />
      <div className="mt-2 h-8 w-16 rounded bg-muted" />
    </div>
  );
}

function StatCardSkeleton() {
  return (
    <div className="glass-card flex items-start gap-3 p-4">
      <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-md bg-muted" />
      <div className="space-y-2">
        <div className="h-3 w-20 rounded bg-muted" />
        <div className="h-6 w-12 rounded bg-muted" />
      </div>
    </div>
  );
}

function ActivityItemSkeleton() {
  return (
    <div className="flex items-start gap-3 py-3">
      <div className="h-8 w-8 shrink-0 rounded-full bg-muted" />
      <div className="flex-1 space-y-2">
        <div className="h-4 w-3/4 rounded bg-muted" />
        <div className="h-3 w-1/4 rounded bg-muted" />
      </div>
    </div>
  );
}

function QuickActionSkeleton() {
  return (
    <div className="glass-card flex items-center gap-3 p-4">
      <div className="h-10 w-10 shrink-0 rounded-md bg-muted" />
      <div className="flex-1 space-y-2">
        <div className="h-4 w-24 rounded bg-muted" />
        <div className="h-3 w-32 rounded bg-muted" />
      </div>
    </div>
  );
}

export default function DashboardPage() {
  const [metrics, setMetrics] = useState<MetricsResponse | null>(null);
  const [namespaceList, setNamespaceList] = useState<NamespaceListResponse | null>(null);
  const [recentMemories, setRecentMemories] = useState<MemoryResponse[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string>("");

  const fetchData = useCallback(async () => {
    setLoading(true);
    try {
      const [m, ns, memories] = await Promise.all([
        getMetrics().catch(() => null),
        listNamespaces().catch(() => null),
        getRecentMemories(5).catch(() => []),
      ]);
      setMetrics(m);
      setNamespaceList(ns);
      setRecentMemories(memories);
      setError("");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load data");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 30000);
    return () => clearInterval(interval);
  }, [fetchData]);

  const formatTimeAgo = (timestamp: number) => {
    const secs = Math.floor(Date.now() / 1000 - timestamp);
    if (secs < 60) return `${secs}s ago`;
    const mins = Math.floor(secs / 60);
    if (mins < 60) return `${mins}m ago`;
    const hours = Math.floor(mins / 60);
    if (hours < 24) return `${hours}h ago`;
    const days = Math.floor(hours / 24);
    return `${days}d ago`;
  };

  const getActivityIcon = (nodeType: string) => {
    switch (nodeType) {
      case "memory":
        return <Brain className="h-4 w-4" />;
      case "entity":
        return <Users className="h-4 w-4" />;
      default:
        return <FileText className="h-4 w-4" />;
    }
  };

  const quickActions = [
    {
      href: "/search",
      icon: Search,
      label: "Search",
      description: "Query your knowledge graph",
      color: "text-cyan-400",
    },
    {
      href: "/memories?new=true",
      icon: Plus,
      label: "Add Memory",
      description: "Ingest new information",
      color: "text-green-400",
    },
    {
      href: "/graph",
      icon: GitFork,
      label: "Explore Graph",
      description: "Visualize connections",
      color: "text-purple-400",
    },
    {
      href: "/conversations",
      icon: MessageSquare,
      label: "Chat",
      description: "Start a conversation",
      color: "text-blue-400",
    },
  ];

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">Dashboard</h1>
          <p className="text-sm text-muted-foreground">
            Welcome back! Here&apos;s your knowledge graph overview.
          </p>
        </div>
        <button
          onClick={fetchData}
          disabled={loading}
          className="flex items-center gap-2 rounded-md border border-border px-3 py-1.5 text-sm text-muted-foreground transition-colors hover:bg-muted hover:text-foreground disabled:opacity-50"
        >
          <RefreshCw className={`h-3.5 w-3.5 ${loading ? "animate-spin" : ""}`} />
          Refresh
        </button>
      </div>

      {error && (
        <div className="rounded-lg border border-destructive/30 bg-destructive/5 p-4 text-sm text-destructive">
          {error}
        </div>
      )}

      {/* Stats Grid */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {loading ? (
          <>
            <StatCardSkeleton />
            <StatCardSkeleton />
            <StatCardSkeleton />
            <StatCardSkeleton />
          </>
        ) : (
          <>
            <StatCard
              label="Total Requests"
              value={metrics?.total_requests.toLocaleString() ?? "0"}
              icon={<Zap className="h-5 w-5" />}
            />
            <StatCard
              label="Ingestions"
              value={metrics?.total_ingestions.toLocaleString() ?? "0"}
              icon={<Brain className="h-5 w-5" />}
            />
            <StatCard
              label="Searches"
              value={metrics?.total_searches.toLocaleString() ?? "0"}
              icon={<Search className="h-5 w-5" />}
            />
            <StatCard
              label="Namespaces"
              value={namespaceList?.namespaces.length ?? 0}
              icon={<Database className="h-5 w-5" />}
            />
          </>
        )}
      </div>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        {/* Recent Activity */}
        <div className="lg:col-span-2">
          <Card title="Recent Activity">
            {loading ? (
              <div className="space-y-1">
                <ActivityItemSkeleton />
                <ActivityItemSkeleton />
                <ActivityItemSkeleton />
              </div>
            ) : recentMemories.length > 0 ? (
              <div className="divide-y divide-border">
                {recentMemories.map((memory) => (
                  <div
                    key={memory.id}
                    className="flex items-start gap-3 py-3"
                  >
                    <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-accent text-accent-foreground">
                      {getActivityIcon(memory.node_type)}
                    </div>
                    <div className="flex-1 min-w-0">
                      <p className="truncate text-sm">
                        {memory.content.substring(0, 80)}
                        {memory.content.length > 80 ? "..." : ""}
                      </p>
                      <div className="flex items-center gap-2 text-xs text-muted-foreground">
                        <Clock className="h-3 w-3" />
                        {formatTimeAgo(memory.timestamp)}
                        <span className="capitalize">• {memory.node_type}</span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="py-8 text-center text-sm text-muted-foreground">
                No recent activity. Start by adding memories or searching!
              </div>
            )}
            <div className="mt-4 pt-3 border-t border-border">
              <Link
                href="/memories"
                className="flex items-center justify-center gap-2 text-sm text-primary hover:underline"
              >
                View all memories
                <ArrowRight className="h-3.5 w-3.5" />
              </Link>
            </div>
          </Card>
        </div>

        {/* Quick Actions */}
        <div>
          <Card title="Quick Actions">
            {loading ? (
              <div className="space-y-3">
                <QuickActionSkeleton />
                <QuickActionSkeleton />
                <QuickActionSkeleton />
                <QuickActionSkeleton />
              </div>
            ) : (
              <div className="space-y-3">
                {quickActions.map((action) => (
                  <Link
                    key={action.href}
                    href={action.href}
                    className="glass-card flex items-center gap-3 p-3 transition-all hover:border-[rgba(0,240,255,0.2)] hover:shadow-[0_0_10px_rgba(0,240,255,0.2)]"
                  >
                    <div className={`flex h-10 w-10 shrink-0 items-center justify-center rounded-md bg-accent ${action.color}`}>
                      <action.icon className="h-5 w-5" />
                    </div>
                    <div>
                      <p className="text-sm font-medium">{action.label}</p>
                      <p className="text-xs text-muted-foreground">
                        {action.description}
                      </p>
                    </div>
                  </Link>
                ))}
              </div>
            )}
          </Card>
        </div>
      </div>

      {/* Namespaces Overview */}
      {namespaceList && namespaceList.namespaces.length > 0 && (
        <Card title="Namespaces">
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {namespaceList.namespaces.slice(0, 6).map((ns) => (
              <Link
                key={ns.name}
                href={`/memories?namespace=${ns.name}`}
                className="glass-card flex items-center justify-between p-3 transition-all hover:border-[rgba(0,240,255,0.2)] hover:shadow-[0_0_10px_rgba(0,240,255,0.2)]"
              >
                <div>
                  <p className="font-mono text-sm font-medium">{ns.name}</p>
                  <p className="text-xs text-muted-foreground">
                    {ns.memory_count} memories • {ns.entity_count} entities
                  </p>
                </div>
                <ArrowRight className="h-4 w-4 text-muted-foreground" />
              </Link>
            ))}
          </div>
        </Card>
      )}
    </div>
  );
}
