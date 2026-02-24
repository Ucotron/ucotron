"use client";

import { useEffect, useState, useCallback } from "react";
import {
  Activity,
  Brain,
  Search,
  Clock,
  Server,
  Database,
  Cpu,
  CheckCircle,
  XCircle,
  RefreshCw,
  GitFork,
  HardDrive,
  Layers,
} from "lucide-react";
import { Card, StatCard } from "@/components/card";
import { getHealth, getMetrics, getSystemInfo, listNamespaces } from "@/lib/api";
import type {
  HealthResponse,
  MetricsResponse,
  SystemInfoResponse,
  NamespaceListResponse,
} from "@/lib/api";

function formatUptime(secs: number): string {
  const d = Math.floor(secs / 86400);
  const h = Math.floor((secs % 86400) / 3600);
  const m = Math.floor((secs % 3600) / 60);
  if (d > 0) return `${d}d ${h}h ${m}m`;
  if (h > 0) return `${h}h ${m}m`;
  return `${m}m ${secs % 60}s`;
}

type Status = "loading" | "connected" | "error";

export default function OverviewPage() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [metrics, setMetrics] = useState<MetricsResponse | null>(null);
  const [systemInfo, setSystemInfo] = useState<SystemInfoResponse | null>(null);
  const [namespaceList, setNamespaceList] = useState<NamespaceListResponse | null>(null);
  const [status, setStatus] = useState<Status>("loading");
  const [error, setError] = useState<string>("");

  const fetchData = useCallback(async () => {
    setStatus("loading");
    try {
      const [h, m, sys, ns] = await Promise.all([
        getHealth(),
        getMetrics(),
        getSystemInfo().catch(() => null),
        listNamespaces().catch(() => null),
      ]);
      setHealth(h);
      setMetrics(m);
      setSystemInfo(sys);
      setNamespaceList(ns);
      setStatus("connected");
      setError("");
    } catch (err) {
      setStatus("error");
      setError(err instanceof Error ? err.message : "Connection failed");
    }
  }, []);

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 10000);
    return () => clearInterval(interval);
  }, [fetchData]);

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold">Overview</h1>
        <button
          onClick={fetchData}
          className="flex items-center gap-2 rounded-md border border-border px-3 py-1.5 text-sm text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
        >
          <RefreshCw className="h-3.5 w-3.5" />
          Refresh
        </button>
      </div>

      {/* Connection Status Banner */}
      <div className={`flex items-center gap-3 rounded-lg border p-4 ${
        status === "connected"
          ? "border-success/30 bg-success/5"
          : status === "error"
          ? "border-destructive/30 bg-destructive/5"
          : "border-border bg-muted"
      }`}>
        {status === "connected" ? (
          <CheckCircle className="h-5 w-5 text-success" />
        ) : status === "error" ? (
          <XCircle className="h-5 w-5 text-destructive" />
        ) : (
          <RefreshCw className="h-5 w-5 animate-spin text-muted-foreground" />
        )}
        <div>
          <p className="text-sm font-medium">
            {status === "connected"
              ? "Connected to Ucotron Server"
              : status === "error"
              ? "Cannot reach Ucotron Server"
              : "Connecting..."}
          </p>
          {status === "error" && (
            <p className="text-xs text-muted-foreground">{error}</p>
          )}
          {status === "connected" && health && (
            <p className="text-xs text-muted-foreground">
              v{health.version} &middot; {health.instance_role} &middot; {health.storage_mode}
            </p>
          )}
        </div>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <StatCard
          label="Total Requests"
          value={metrics?.total_requests ?? "-"}
          icon={<Activity className="h-5 w-5" />}
        />
        <StatCard
          label="Ingestions"
          value={metrics?.total_ingestions ?? "-"}
          icon={<Brain className="h-5 w-5" />}
        />
        <StatCard
          label="Searches"
          value={metrics?.total_searches ?? "-"}
          icon={<Search className="h-5 w-5" />}
        />
        <StatCard
          label="Uptime"
          value={metrics ? formatUptime(metrics.uptime_secs) : "-"}
          icon={<Clock className="h-5 w-5" />}
        />
      </div>

      {/* Server Details */}
      {health && (
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
          <Card title="Server Information">
            <dl className="space-y-3 text-sm">
              {[
                { icon: Server, label: "Instance ID", value: health.instance_id },
                { icon: Server, label: "Role", value: health.instance_role },
                { icon: Database, label: "Storage Mode", value: health.storage_mode },
                { icon: Database, label: "Vector Backend", value: health.vector_backend },
                { icon: Database, label: "Graph Backend", value: health.graph_backend },
              ].map((row) => (
                <div key={row.label} className="flex items-center justify-between">
                  <dt className="flex items-center gap-2 text-muted-foreground">
                    <row.icon className="h-3.5 w-3.5" />
                    {row.label}
                  </dt>
                  <dd className="font-mono text-xs">{row.value}</dd>
                </div>
              ))}
            </dl>
          </Card>

          <Card title="Models">
            <dl className="space-y-3 text-sm">
              <div className="flex items-center justify-between">
                <dt className="flex items-center gap-2 text-muted-foreground">
                  <Cpu className="h-3.5 w-3.5" />
                  Embedding Model
                </dt>
                <dd className="flex items-center gap-2">
                  <span className="font-mono text-xs">{health.models.embedding_model}</span>
                  {health.models.embedder_loaded ? (
                    <CheckCircle className="h-3.5 w-3.5 text-success" />
                  ) : (
                    <XCircle className="h-3.5 w-3.5 text-destructive" />
                  )}
                </dd>
              </div>
              <div className="flex items-center justify-between">
                <dt className="flex items-center gap-2 text-muted-foreground">
                  <Cpu className="h-3.5 w-3.5" />
                  NER Model
                </dt>
                <dd>
                  {health.models.ner_loaded ? (
                    <CheckCircle className="h-3.5 w-3.5 text-success" />
                  ) : (
                    <XCircle className="h-3.5 w-3.5 text-muted-foreground" />
                  )}
                </dd>
              </div>
              <div className="flex items-center justify-between">
                <dt className="flex items-center gap-2 text-muted-foreground">
                  <Cpu className="h-3.5 w-3.5" />
                  Relation Extractor
                </dt>
                <dd>
                  {health.models.relation_extractor_loaded ? (
                    <CheckCircle className="h-3.5 w-3.5 text-success" />
                  ) : (
                    <XCircle className="h-3.5 w-3.5 text-muted-foreground" />
                  )}
                </dd>
              </div>
            </dl>
          </Card>
        </div>
      )}

      {/* System Info Stats */}
      {systemInfo && (
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <StatCard
            label="Total Nodes"
            value={systemInfo.total_nodes.toLocaleString()}
            icon={<Brain className="h-5 w-5" />}
          />
          <StatCard
            label="Total Edges"
            value={systemInfo.total_edges.toLocaleString()}
            icon={<GitFork className="h-5 w-5" />}
          />
          <StatCard
            label="Memory RSS (MB)"
            value={(systemInfo.memory_rss_bytes / 1024 / 1024).toFixed(1)}
            icon={<HardDrive className="h-5 w-5" />}
          />
          <StatCard
            label="Namespaces"
            value={namespaceList?.total ?? "-"}
            icon={<Layers className="h-5 w-5" />}
          />
        </div>
      )}

      {/* Namespaces Table */}
      {namespaceList && namespaceList.namespaces.length > 0 && (
        <Card title="Namespaces">
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-border text-left text-xs text-muted-foreground">
                  <th className="pb-2 pr-4 font-medium">Name</th>
                  <th className="pb-2 pr-4 text-right font-medium">Memories</th>
                  <th className="pb-2 pr-4 text-right font-medium">Entities</th>
                  <th className="pb-2 text-right font-medium">Total Nodes</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border">
                {namespaceList.namespaces.map((ns) => (
                  <tr key={ns.name} className="hover:bg-muted/50">
                    <td className="py-2 pr-4 font-mono text-xs font-medium">
                      {ns.name}
                    </td>
                    <td className="py-2 pr-4 text-right tabular-nums">
                      {ns.memory_count.toLocaleString()}
                    </td>
                    <td className="py-2 pr-4 text-right tabular-nums">
                      {ns.entity_count.toLocaleString()}
                    </td>
                    <td className="py-2 text-right tabular-nums">
                      {ns.total_nodes.toLocaleString()}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}
    </div>
  );
}
