"use client";

import { useState, useEffect, useCallback } from "react";
import { GitFork, Filter, X, ChevronRight, Loader2, AlertCircle } from "lucide-react";
import { Card } from "@/components/card";
import { ForceGraph } from "@/components/force-graph";
import { getGraph, getEntity } from "@/lib/api";
import type { GraphNode, GraphEdge, EntityResponse, NeighborResponse } from "@/lib/api";
import { cn } from "@/lib/utils";

type Status = "loading" | "loaded" | "error" | "empty";

const NODE_TYPE_OPTIONS = ["", "entity", "event", "fact", "skill"];
const NODE_TYPE_LABELS: Record<string, string> = {
  "": "All Types",
  entity: "Entity",
  event: "Event",
  fact: "Fact",
  skill: "Skill",
};

const NODE_COLORS: Record<string, string> = {
  Entity: "bg-[#00F0FF]",
  Event: "bg-[#00B8CC]",
  Fact: "bg-[#00FF94]",
  Skill: "bg-[#FFB800]",
};

export default function GraphPage() {
  const [status, setStatus] = useState<Status>("loading");
  const [error, setError] = useState<string | null>(null);
  const [nodes, setNodes] = useState<GraphNode[]>([]);
  const [edges, setEdges] = useState<GraphEdge[]>([]);
  const [totalNodes, setTotalNodes] = useState(0);
  const [totalEdges, setTotalEdges] = useState(0);

  // Filters.
  const [nodeTypeFilter, setNodeTypeFilter] = useState("");
  const [limitFilter, setLimitFilter] = useState(200);

  // Selection.
  const [selectedNodeId, setSelectedNodeId] = useState<number | null>(null);
  const [selectedEntity, setSelectedEntity] = useState<EntityResponse | null>(null);
  const [panelLoading, setPanelLoading] = useState(false);

  const fetchGraph = useCallback(async () => {
    setStatus("loading");
    setError(null);
    try {
      const data = await getGraph({
        limit: limitFilter,
        node_type: nodeTypeFilter || undefined,
      });
      setNodes(data.nodes);
      setEdges(data.edges);
      setTotalNodes(data.total_nodes);
      setTotalEdges(data.total_edges);
      setStatus(data.nodes.length === 0 ? "empty" : "loaded");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to fetch graph");
      setStatus("error");
    }
  }, [limitFilter, nodeTypeFilter]);

  useEffect(() => {
    fetchGraph();
  }, [fetchGraph]);

  // When a node is selected, fetch its entity details.
  useEffect(() => {
    if (selectedNodeId === null) {
      setSelectedEntity(null);
      return;
    }
    let cancelled = false;
    setPanelLoading(true);
    getEntity(selectedNodeId)
      .then((entity) => {
        if (!cancelled) {
          setSelectedEntity(entity);
          setPanelLoading(false);
        }
      })
      .catch(() => {
        if (!cancelled) {
          // Fall back to basic info from graph data.
          const node = nodes.find((n) => n.id === selectedNodeId);
          if (node) {
            setSelectedEntity({
              id: node.id,
              content: node.content,
              node_type: node.node_type,
              timestamp: node.timestamp,
              metadata: {},
              neighbors: undefined,
            });
          }
          setPanelLoading(false);
        }
      });
    return () => {
      cancelled = true;
    };
  }, [selectedNodeId, nodes]);

  return (
    <div className="flex h-full gap-4">
      {/* Main graph area */}
      <div className="flex flex-1 flex-col gap-4 min-w-0">
        {/* Header + filters */}
        <div className="flex items-center justify-between gap-4">
          <div className="flex items-center gap-2">
            <GitFork className="h-5 w-5 text-primary" />
            <h1 className="text-2xl font-bold">Graph Visualization</h1>
          </div>
          {status === "loaded" && (
            <span className="text-sm text-muted-foreground">
              {totalNodes} nodes, {totalEdges} edges
            </span>
          )}
        </div>

        {/* Filter bar */}
        <div className="flex items-center gap-3">
          <Filter className="h-4 w-4 text-muted-foreground" />

          <select
            className="rounded-md border border-border bg-card px-3 py-1.5 text-sm"
            value={nodeTypeFilter}
            onChange={(e) => setNodeTypeFilter(e.target.value)}
          >
            {NODE_TYPE_OPTIONS.map((opt) => (
              <option key={opt} value={opt}>
                {NODE_TYPE_LABELS[opt]}
              </option>
            ))}
          </select>

          <label className="flex items-center gap-2 text-sm text-muted-foreground">
            Limit:
            <select
              className="rounded-md border border-border bg-card px-2 py-1.5 text-sm"
              value={limitFilter}
              onChange={(e) => setLimitFilter(Number(e.target.value))}
            >
              {[50, 100, 200, 300, 500].map((n) => (
                <option key={n} value={n}>
                  {n}
                </option>
              ))}
            </select>
          </label>

          {/* Legend */}
          <div className="ml-auto flex items-center gap-3 text-xs text-muted-foreground">
            {Object.entries(NODE_COLORS).map(([type, colorClass]) => (
              <span key={type} className="flex items-center gap-1">
                <span className={cn("inline-block h-2.5 w-2.5 rounded-full", colorClass)} />
                {type}
              </span>
            ))}
          </div>
        </div>

        {/* Graph canvas */}
        <div className="flex-1 min-h-0">
          {status === "loading" && (
            <Card className="flex h-full items-center justify-center">
              <div className="flex flex-col items-center gap-3">
                <Loader2 className="h-8 w-8 animate-spin text-primary" />
                <p className="text-sm text-muted-foreground">Loading graph data...</p>
              </div>
            </Card>
          )}
          {status === "error" && (
            <Card className="flex h-full items-center justify-center">
              <div className="flex flex-col items-center gap-3 text-center">
                <AlertCircle className="h-8 w-8 text-destructive" />
                <p className="text-sm text-destructive">{error}</p>
                <button
                  className="rounded-md bg-primary px-4 py-1.5 text-sm text-white"
                  onClick={fetchGraph}
                >
                  Retry
                </button>
              </div>
            </Card>
          )}
          {status === "empty" && (
            <Card className="flex h-full items-center justify-center">
              <div className="flex flex-col items-center gap-3">
                <GitFork className="h-10 w-10 text-muted-foreground" />
                <p className="text-sm text-muted-foreground">
                  No graph data available. Ingest some memories first.
                </p>
              </div>
            </Card>
          )}
          {status === "loaded" && (
            <ForceGraph
              nodes={nodes}
              edges={edges}
              selectedNodeId={selectedNodeId}
              onSelectNode={setSelectedNodeId}
            />
          )}
        </div>
      </div>

      {/* Side panel — node details */}
      {selectedNodeId !== null && (
        <div className="w-80 shrink-0 overflow-y-auto rounded-lg border border-border bg-card">
          {/* Panel header */}
          <div className="flex items-center justify-between border-b border-border p-4">
            <h2 className="text-sm font-medium">Node Details</h2>
            <button
              className="rounded p-1 hover:bg-accent"
              onClick={() => setSelectedNodeId(null)}
            >
              <X className="h-4 w-4" />
            </button>
          </div>

          {panelLoading ? (
            <div className="flex items-center justify-center p-8">
              <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
            </div>
          ) : selectedEntity ? (
            <div className="space-y-4 p-4">
              {/* Identity */}
              <div>
                <span className="text-xs text-muted-foreground">ID</span>
                <p className="font-mono text-sm">{selectedEntity.id}</p>
              </div>

              <div>
                <span className="text-xs text-muted-foreground">Type</span>
                <div className="mt-0.5 flex items-center gap-1.5">
                  <span
                    className={cn(
                      "inline-block h-2.5 w-2.5 rounded-full",
                      NODE_COLORS[selectedEntity.node_type] || "bg-zinc-500"
                    )}
                  />
                  <span className="text-sm">{selectedEntity.node_type}</span>
                </div>
              </div>

              <div>
                <span className="text-xs text-muted-foreground">Content</span>
                <p className="mt-0.5 text-sm leading-relaxed">{selectedEntity.content}</p>
              </div>

              <div>
                <span className="text-xs text-muted-foreground">Timestamp</span>
                <p className="text-sm">
                  {selectedEntity.timestamp > 0
                    ? new Date(selectedEntity.timestamp * 1000).toLocaleString()
                    : "—"}
                </p>
              </div>

              {/* Metadata */}
              {Object.keys(selectedEntity.metadata).length > 0 && (
                <div>
                  <span className="text-xs text-muted-foreground">Metadata</span>
                  <div className="mt-1 space-y-1">
                    {Object.entries(selectedEntity.metadata).map(([k, v]) => (
                      <div key={k} className="flex gap-2 text-xs">
                        <span className="font-mono text-muted-foreground">{k}:</span>
                        <span className="break-all">{String(v)}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Neighbors */}
              {selectedEntity.neighbors && selectedEntity.neighbors.length > 0 && (
                <div>
                  <span className="text-xs text-muted-foreground">
                    Neighbors ({selectedEntity.neighbors.length})
                  </span>
                  <div className="mt-1 space-y-1">
                    {selectedEntity.neighbors.map((n: NeighborResponse) => (
                      <button
                        key={n.node_id}
                        className="flex w-full items-center gap-2 rounded-md px-2 py-1.5 text-left text-xs hover:bg-accent"
                        onClick={() => setSelectedNodeId(n.node_id)}
                      >
                        <ChevronRight className="h-3 w-3 shrink-0 text-muted-foreground" />
                        <span className="min-w-0 truncate">{n.content}</span>
                        <span className="ml-auto shrink-0 text-[10px] text-muted-foreground">
                          {n.edge_type}
                        </span>
                      </button>
                    ))}
                  </div>
                </div>
              )}
            </div>
          ) : null}
        </div>
      )}
    </div>
  );
}
