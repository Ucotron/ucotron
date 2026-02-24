"use client";

import { useState } from "react";
import {
  Activity,
  Search,
  Clock,
  Zap,
  Hash,
  X,
  ChevronDown,
  ChevronRight,
  Copy,
  Check,
  Loader2,
} from "lucide-react";
import { Card, StatCard } from "@/components/card";
import { augmentQueryDebug } from "@/lib/api";
import type {
  AugmentResponse,
  AugmentDebugInfo,
  PipelineTimings,
  ScoreBreakdown,
  SearchResultItem,
} from "@/lib/api";
import { useNamespace } from "@/components/namespace-context";

type Status = "idle" | "loading" | "done" | "error";

const TIMING_LABELS: Record<keyof Omit<PipelineTimings, "total_us">, string> = {
  query_embedding_us: "Query Embedding",
  vector_search_us: "Vector Search",
  entity_extraction_us: "Entity Extraction",
  graph_expansion_us: "Graph Expansion",
  community_selection_us: "Community Selection",
  reranking_us: "Re-ranking",
  context_assembly_us: "Context Assembly",
};

const TIMING_COLORS: Record<keyof Omit<PipelineTimings, "total_us">, string> = {
  query_embedding_us: "bg-[#00F0FF]",
  vector_search_us: "bg-[#00B8CC]",
  entity_extraction_us: "bg-[#00FF94]",
  graph_expansion_us: "bg-[#FFB800]",
  community_selection_us: "bg-[#00F0FF]/60",
  reranking_us: "bg-[#00B8CC]/60",
  context_assembly_us: "bg-[#94A3B8]",
};

function usToMs(us: number): string {
  return (us / 1000).toFixed(2);
}

function PipelineChart({ timings }: { timings: PipelineTimings }) {
  const total = timings.total_us || 1;
  const steps = (Object.keys(TIMING_LABELS) as (keyof typeof TIMING_LABELS)[]).map(
    (key) => ({
      key,
      label: TIMING_LABELS[key],
      value: timings[key],
      color: TIMING_COLORS[key],
      pct: (timings[key] / total) * 100,
    })
  );

  return (
    <div className="space-y-2">
      {/* Stacked horizontal bar */}
      <div className="flex h-8 w-full overflow-hidden rounded-md">
        {steps.map((step) =>
          step.pct > 0 ? (
            <div
              key={step.key}
              className={`${step.color} relative transition-all`}
              style={{ width: `${Math.max(step.pct, 0.5)}%` }}
              title={`${step.label}: ${usToMs(step.value)}ms (${step.pct.toFixed(1)}%)`}
            />
          ) : null
        )}
      </div>

      {/* Legend + individual bars */}
      <div className="space-y-1.5">
        {steps.map((step) => (
          <div key={step.key} className="flex items-center gap-3 text-sm">
            <span
              className={`inline-block h-2.5 w-2.5 shrink-0 rounded-sm ${step.color}`}
            />
            <span className="w-40 shrink-0 text-muted-foreground">
              {step.label}
            </span>
            <div className="flex-1">
              <div className="h-4 w-full rounded-sm bg-muted">
                <div
                  className={`h-full rounded-sm ${step.color} opacity-70 transition-all`}
                  style={{ width: `${Math.max(step.pct, 0.5)}%` }}
                />
              </div>
            </div>
            <span className="w-20 shrink-0 text-right font-mono text-xs tabular-nums">
              {usToMs(step.value)}ms
            </span>
            <span className="w-14 shrink-0 text-right text-xs text-muted-foreground tabular-nums">
              {step.pct.toFixed(1)}%
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

function ScoreTable({ scores }: { scores: ScoreBreakdown[] }) {
  if (scores.length === 0) {
    return (
      <p className="text-sm text-muted-foreground">No score breakdown available.</p>
    );
  }

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-border text-left text-xs text-muted-foreground">
            <th className="pb-2 pr-4 font-medium">ID</th>
            <th className="pb-2 pr-4 text-right font-medium">Final Score</th>
            <th className="pb-2 pr-4 text-right font-medium">Vector Sim</th>
            <th className="pb-2 pr-4 text-right font-medium">Graph Centrality</th>
            <th className="pb-2 pr-4 text-right font-medium">Recency</th>
            <th className="pb-2 pr-4 text-right font-medium">Mindset</th>
            <th className="pb-2 text-right font-medium">Path Reward</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-border">
          {scores.map((s) => (
            <tr key={s.id} className="hover:bg-muted/50">
              <td className="py-2 pr-4 font-mono text-xs">{s.id}</td>
              <td className="py-2 pr-4 text-right font-mono text-xs tabular-nums">
                <span className="rounded-full bg-primary/10 px-2 py-0.5 text-primary">
                  {s.final_score.toFixed(4)}
                </span>
              </td>
              <td className="py-2 pr-4 text-right font-mono text-xs tabular-nums">
                {s.vector_sim.toFixed(4)}
              </td>
              <td className="py-2 pr-4 text-right font-mono text-xs tabular-nums">
                {s.graph_centrality.toFixed(4)}
              </td>
              <td className="py-2 pr-4 text-right font-mono text-xs tabular-nums">
                {s.recency.toFixed(4)}
              </td>
              <td className="py-2 pr-4 text-right font-mono text-xs tabular-nums">
                {s.mindset_score.toFixed(4)}
              </td>
              <td className="py-2 text-right font-mono text-xs tabular-nums">
                {s.path_reward.toFixed(4)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function MemoryCard({ memory }: { memory: SearchResultItem }) {
  const [expanded, setExpanded] = useState(false);

  return (
    <div className="rounded-md border border-border bg-card p-3">
      <div
        className="flex cursor-pointer items-start gap-2"
        onClick={() => setExpanded(!expanded)}
      >
        {expanded ? (
          <ChevronDown className="mt-0.5 h-4 w-4 shrink-0 text-muted-foreground" />
        ) : (
          <ChevronRight className="mt-0.5 h-4 w-4 shrink-0 text-muted-foreground" />
        )}
        <div className="min-w-0 flex-1">
          <p className={`text-sm ${expanded ? "" : "line-clamp-2"}`}>
            {memory.content}
          </p>
          <div className="mt-1.5 flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
            <span className="rounded-full bg-accent px-2 py-0.5 text-accent-foreground">
              {memory.node_type}
            </span>
            <span>ID: {memory.id}</span>
            <span className="rounded-full bg-primary/10 px-2 py-0.5 text-primary">
              Score: {memory.score.toFixed(4)}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}

export default function TracingPage() {
  const { namespace } = useNamespace();
  const [query, setQuery] = useState("");
  const [limit, setLimit] = useState(10);
  const [status, setStatus] = useState<Status>("idle");
  const [error, setError] = useState("");
  const [response, setResponse] = useState<AugmentResponse | null>(null);
  const [copied, setCopied] = useState(false);
  const [contextExpanded, setContextExpanded] = useState(false);

  async function handleTrace(e: React.FormEvent) {
    e.preventDefault();
    if (!query.trim()) return;

    setStatus("loading");
    setError("");
    setResponse(null);

    try {
      const res = await augmentQueryDebug(query.trim(), namespace, limit);
      setResponse(res);
      setStatus("done");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Augment request failed");
      setStatus("error");
    }
  }

  function handleCopyContext() {
    if (!response?.context_text) return;
    navigator.clipboard.writeText(response.context_text).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  }

  const debug = response?.debug;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center gap-2">
        <Activity className="h-5 w-5 text-primary" />
        <h1 className="text-2xl font-bold">Augmentation Tracing</h1>
      </div>

      <p className="text-sm text-muted-foreground">
        Trace the full retrieval pipeline for a given query. Sends a debug
        augment request and displays pipeline timings, score breakdowns, and
        returned memories.
      </p>

      {/* Query Form */}
      <Card>
        <form onSubmit={handleTrace} className="space-y-4">
          <div>
            <label className="mb-1.5 block text-sm font-medium">Query</label>
            <div className="relative">
              <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
              <input
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Enter a query to trace through the augmentation pipeline..."
                className="w-full rounded-md border border-border bg-background py-2.5 pl-10 pr-4 text-sm focus:outline-none focus:ring-2 focus:ring-primary"
              />
            </div>
          </div>
          <div className="flex items-end gap-4">
            <div>
              <label className="mb-1.5 block text-xs text-muted-foreground">
                Max memories
              </label>
              <select
                value={limit}
                onChange={(e) => setLimit(Number(e.target.value))}
                className="rounded-md border border-border bg-background px-3 py-2 text-sm"
              >
                {[5, 10, 15, 20, 50].map((n) => (
                  <option key={n} value={n}>
                    {n}
                  </option>
                ))}
              </select>
            </div>
            <div className="flex items-center gap-2 text-xs text-muted-foreground">
              <span>
                Namespace:{" "}
                <span className="font-mono font-medium text-foreground">
                  {namespace}
                </span>
              </span>
            </div>
            <div className="ml-auto">
              <button
                type="submit"
                disabled={status === "loading" || !query.trim()}
                className="flex items-center gap-2 rounded-md bg-primary px-5 py-2 text-sm font-medium text-primary-foreground transition-colors hover:bg-primary/90 disabled:opacity-50"
              >
                {status === "loading" ? (
                  <>
                    <Loader2 className="h-4 w-4 animate-spin" />
                    Tracing...
                  </>
                ) : (
                  <>
                    <Zap className="h-4 w-4" />
                    Trace
                  </>
                )}
              </button>
            </div>
          </div>
        </form>
      </Card>

      {/* Error */}
      {status === "error" && (
        <div className="flex items-center justify-between rounded-md border border-destructive/30 bg-destructive/5 p-3 text-sm text-destructive">
          <span>{error}</span>
          <button onClick={() => setError("")}>
            <X className="h-4 w-4" />
          </button>
        </div>
      )}

      {/* Loading */}
      {status === "loading" && (
        <Card className="flex items-center justify-center py-12">
          <div className="flex flex-col items-center gap-3">
            <Loader2 className="h-8 w-8 animate-spin text-primary" />
            <p className="text-sm text-muted-foreground">
              Running augmentation pipeline...
            </p>
          </div>
        </Card>
      )}

      {/* Results */}
      {status === "done" && response && (
        <>
          {/* Summary Stats */}
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
            <StatCard
              label="Pipeline Duration"
              value={
                debug
                  ? `${debug.pipeline_duration_ms.toFixed(2)}ms`
                  : "-"
              }
              icon={<Clock className="h-5 w-5" />}
            />
            <StatCard
              label="Memories Returned"
              value={response.memories.length}
              icon={<Search className="h-5 w-5" />}
            />
            <StatCard
              label="Vector Results"
              value={debug?.vector_results_count ?? "-"}
              icon={<Zap className="h-5 w-5" />}
            />
            <StatCard
              label="Query Entities"
              value={debug?.query_entities_count ?? "-"}
              icon={<Hash className="h-5 w-5" />}
            />
          </div>

          {/* Pipeline Timings */}
          {debug && (
            <Card title="Pipeline Timings">
              <PipelineChart timings={debug.pipeline_timings} />
            </Card>
          )}

          {/* Score Breakdown */}
          {debug && debug.score_breakdown.length > 0 && (
            <Card title="Score Breakdown">
              <ScoreTable scores={debug.score_breakdown} />
            </Card>
          )}

          {/* Returned Memories */}
          {response.memories.length > 0 && (
            <Card title={`Returned Memories (${response.memories.length})`}>
              <div className="space-y-2">
                {response.memories.map((mem) => (
                  <MemoryCard key={mem.id} memory={mem} />
                ))}
              </div>
            </Card>
          )}

          {/* Context Text */}
          {response.context_text && (
            <Card title="Assembled Context">
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <button
                    onClick={() => setContextExpanded(!contextExpanded)}
                    className="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground"
                  >
                    {contextExpanded ? (
                      <ChevronDown className="h-3.5 w-3.5" />
                    ) : (
                      <ChevronRight className="h-3.5 w-3.5" />
                    )}
                    {contextExpanded ? "Collapse" : "Expand"} (
                    {response.context_text.length} chars)
                  </button>
                  <button
                    onClick={handleCopyContext}
                    className="flex items-center gap-1 rounded-md border border-border px-2.5 py-1 text-xs text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
                  >
                    {copied ? (
                      <>
                        <Check className="h-3 w-3" />
                        Copied
                      </>
                    ) : (
                      <>
                        <Copy className="h-3 w-3" />
                        Copy
                      </>
                    )}
                  </button>
                </div>
                {contextExpanded ? (
                  <pre className="max-h-96 overflow-auto whitespace-pre-wrap rounded-md bg-muted p-3 text-xs leading-relaxed">
                    {response.context_text}
                  </pre>
                ) : (
                  <pre className="line-clamp-4 overflow-hidden whitespace-pre-wrap rounded-md bg-muted p-3 text-xs leading-relaxed">
                    {response.context_text}
                  </pre>
                )}
              </div>
            </Card>
          )}

          {/* Extracted Entities */}
          {response.entities.length > 0 && (
            <Card title={`Query Entities (${response.entities.length})`}>
              <div className="flex flex-wrap gap-2">
                {response.entities.map((ent) => (
                  <div
                    key={ent.id}
                    className="flex items-center gap-1.5 rounded-md bg-muted px-2.5 py-1.5 text-xs"
                  >
                    <span className="rounded-full bg-accent px-1.5 py-0.5 text-[10px] text-accent-foreground">
                      {ent.node_type}
                    </span>
                    <span>{ent.content}</span>
                  </div>
                ))}
              </div>
            </Card>
          )}
        </>
      )}

      {/* Idle state */}
      {status === "idle" && (
        <Card className="flex flex-col items-center gap-3 py-12">
          <Activity className="h-10 w-10 text-muted-foreground" />
          <p className="text-sm text-muted-foreground">
            Enter a query above and click Trace to inspect the augmentation
            pipeline.
          </p>
        </Card>
      )}
    </div>
  );
}
