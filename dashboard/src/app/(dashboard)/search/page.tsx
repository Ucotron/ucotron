"use client";

import { useState, useCallback, useEffect } from "react";
import { useRouter } from "next/navigation";
import {
  Search,
  Copy,
  Check,
  FileText,
  Brain,
  Sparkles,
  Loader2,
  Clock,
  X,
  ChevronRight,
  Network,
  Layers,
  Zap,
  ArrowRight,
} from "lucide-react";
import { Card, StatCard } from "@ucotron/ui";
import { Input, Textarea } from "@ucotron/ui";
import { Button } from "@ucotron/ui";
import {
  augmentQuery,
  searchMemories,
  type SearchResponse,
  type AugmentResponse,
  type SearchResultItem,
  type ScoreBreakdown,
} from "@/lib/api";

const SUGGESTED_QUERIES = [
  "What is my team's quarterly goals?",
  "Show me recent project updates",
  "Find information about the new product launch",
  "What decisions were made last week?",
  "Summarize customer feedback",
];

interface ExplainabilityPanelProps {
  memories: SearchResultItem[];
  scoreBreakdown: ScoreBreakdown[];
  debug?: AugmentResponse["debug"];
  onNodeClick: (nodeId: number) => void;
}

function ExplainabilityPanel({
  memories,
  scoreBreakdown,
  debug,
  onNodeClick,
}: ExplainabilityPanelProps) {
  const [activeTab, setActiveTab] = useState<"scores" | "path" | "context">("scores");

  const getScoreColor = (score: number) => {
    if (score >= 0.8) return "bg-green-500";
    if (score >= 0.6) return "bg-amber-500";
    return "bg-red-500";
  };

  const formatMicroseconds = (us: number) => {
    if (us >= 1000000) return `${(us / 1000000).toFixed(2)}s`;
    if (us >= 1000) return `${(us / 1000).toFixed(1)}ms`;
    return `${us}μs`;
  };

  return (
    <Card title="Query Explainability">
      <div className="space-y-4">
        <div className="flex rounded-md bg-muted p-1">
          <button
            onClick={() => setActiveTab("scores")}
            className={`flex-1 rounded-sm px-3 py-1.5 text-sm transition-all ${
              activeTab === "scores"
                ? "bg-background shadow-sm"
                : "text-muted-foreground hover:text-foreground"
            }`}
          >
            <Brain className="mr-1.5 inline h-4 w-4" />
            Relevance Scores
          </button>
          <button
            onClick={() => setActiveTab("path")}
            className={`flex-1 rounded-sm px-3 py-1.5 text-sm transition-all ${
              activeTab === "path"
                ? "bg-background shadow-sm"
                : "text-muted-foreground hover:text-foreground"
            }`}
          >
            <Network className="mr-1.5 inline h-4 w-4" />
            Retrieval Path
          </button>
          <button
            onClick={() => setActiveTab("context")}
            className={`flex-1 rounded-sm px-3 py-1.5 text-sm transition-all ${
              activeTab === "context"
                ? "bg-background shadow-sm"
                : "text-muted-foreground hover:text-foreground"
            }`}
          >
            <Layers className="mr-1.5 inline h-4 w-4" />
            Context
          </button>
        </div>

        {activeTab === "scores" && (
          <div className="space-y-3">
            <p className="text-xs text-muted-foreground">
              Nodes ranked by relevance score (vector similarity + graph centrality + recency)
            </p>
            {scoreBreakdown.map((item, index) => {
              const memory = memories.find((m) => m.id === item.id);
              return (
                <div
                  key={item.id}
                  className="group flex items-center gap-3 rounded-lg border border-border p-3 transition-colors hover:bg-muted/50"
                >
                  <div className="flex h-6 w-6 items-center justify-center rounded-full bg-muted text-xs font-medium">
                    {index + 1}
                  </div>
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center justify-between">
                      <span className="truncate text-sm font-medium">
                        {memory?.node_type || "memory"} #{item.id}
                      </span>
                      <button
                        onClick={() => onNodeClick(item.id)}
                        className="flex items-center gap-1 text-xs text-primary opacity-0 transition-opacity group-hover:opacity-100"
                      >
                        View in Graph
                        <ArrowRight className="h-3 w-3" />
                      </button>
                    </div>
                    <p className="mt-1 line-clamp-2 text-xs text-muted-foreground">
                      {memory?.content.substring(0, 100)}...
                    </p>
                  </div>
                  <div className="flex flex-col items-end gap-1">
                    <div className="flex items-center gap-2">
                      <span className="text-sm font-semibold">
                        {(item.final_score * 100).toFixed(0)}%
                      </span>
                      <div
                        className={`h-2 w-2 rounded-full ${getScoreColor(item.final_score)}`}
                      />
                    </div>
                    <div className="flex gap-2 text-xs text-muted-foreground">
                      <span title="Vector Similarity">V: {(item.vector_sim * 100).toFixed(0)}%</span>
                      <span title="Graph Centrality">G: {(item.graph_centrality * 100).toFixed(0)}%</span>
                      <span title="Recency">R: {(item.recency * 100).toFixed(0)}%</span>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        )}

        {activeTab === "path" && debug && (
          <div className="space-y-4">
            <p className="text-xs text-muted-foreground">
              Retrieval pipeline: query → embedding → vector search → graph expansion → reranking → context
            </p>
            <div className="space-y-2">
              <div className="flex items-center gap-2 rounded-lg bg-muted p-3">
                <div className="flex h-8 w-8 items-center justify-center rounded-full bg-primary/20">
                  <Zap className="h-4 w-4 text-primary" />
                </div>
                <div className="flex-1">
                  <p className="text-sm font-medium">Query Embedding</p>
                  <p className="text-xs text-muted-foreground">Convert query to vector</p>
                </div>
                <span className="text-sm font-mono text-muted-foreground">
                  {formatMicroseconds(debug.pipeline_timings.query_embedding_us)}
                </span>
              </div>

              <div className="flex items-center gap-2 rounded-lg bg-muted p-3">
                <div className="flex h-8 w-8 items-center justify-center rounded-full bg-primary/20">
                  <Search className="h-4 w-4 text-primary" />
                </div>
                <div className="flex-1">
                  <p className="text-sm font-medium">Vector Search</p>
                  <p className="text-xs text-muted-foreground">
                    Found {debug.vector_results_count} candidates
                  </p>
                </div>
                <span className="text-sm font-mono text-muted-foreground">
                  {formatMicroseconds(debug.pipeline_timings.vector_search_us)}
                </span>
              </div>

              <div className="flex items-center gap-2 rounded-lg bg-muted p-3">
                <div className="flex h-8 w-8 items-center justify-center rounded-full bg-primary/20">
                  <Brain className="h-4 w-4 text-primary" />
                </div>
                <div className="flex-1">
                  <p className="text-sm font-medium">Graph Expansion</p>
                  <p className="text-xs text-muted-foreground">
                    Extracted {debug.query_entities_count} entities
                  </p>
                </div>
                <span className="text-sm font-mono text-muted-foreground">
                  {formatMicroseconds(debug.pipeline_timings.graph_expansion_us)}
                </span>
              </div>

              <div className="flex items-center gap-2 rounded-lg bg-muted p-3">
                <div className="flex h-8 w-8 items-center justify-center rounded-full bg-primary/20">
                  <ChevronRight className="h-4 w-4 text-primary" />
                </div>
                <div className="flex-1">
                  <p className="text-sm font-medium">Community Selection</p>
                  <p className="text-xs text-muted-foreground">Select relevant graph communities</p>
                </div>
                <span className="text-sm font-mono text-muted-foreground">
                  {formatMicroseconds(debug.pipeline_timings.community_selection_us)}
                </span>
              </div>

              <div className="flex items-center gap-2 rounded-lg bg-muted p-3">
                <div className="flex h-8 w-8 items-center justify-center rounded-full bg-primary/20">
                  <Sparkles className="h-4 w-4 text-primary" />
                </div>
                <div className="flex-1">
                  <p className="text-sm font-medium">Reranking</p>
                  <p className="text-xs text-muted-foreground">
                    Re-rank {debug.vector_results_count} → {scoreBreakdown.length} results
                  </p>
                </div>
                <span className="text-sm font-mono text-muted-foreground">
                  {formatMicroseconds(debug.pipeline_timings.reranking_us)}
                </span>
              </div>

              <div className="flex items-center gap-2 rounded-lg bg-muted p-3">
                <div className="flex h-8 w-8 items-center justify-center rounded-full bg-green-500/20">
                  <Layers className="h-4 w-4 text-green-500" />
                </div>
                <div className="flex-1">
                  <p className="text-sm font-medium">Context Assembly</p>
                  <p className="text-xs text-muted-foreground">Build final context window</p>
                </div>
                <span className="text-sm font-mono text-muted-foreground">
                  {formatMicroseconds(debug.pipeline_timings.context_assembly_us)}
                </span>
              </div>
            </div>

            <div className="rounded-lg border border-border p-3">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">Total Pipeline Time</span>
                <span className="font-mono text-lg font-semibold">
                  {formatMicroseconds(debug.pipeline_timings.total_us)}
                </span>
              </div>
            </div>
          </div>
        )}

        {activeTab === "context" && (
          <div className="space-y-3">
            <p className="text-xs text-muted-foreground">
              Context window composition: how the augmented context is built from source nodes
            </p>
            <div className="space-y-2">
              {memories.map((memory, index) => (
                <div
                  key={memory.id}
                  className="flex items-start gap-2 rounded-lg border border-border p-2"
                >
                  <div className="flex h-5 w-5 shrink-0 items-center justify-center rounded bg-muted text-xs">
                    {index + 1}
                  </div>
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center justify-between">
                      <span className="text-xs font-medium capitalize text-muted-foreground">
                        {memory.node_type} #{memory.id}
                      </span>
                      <span className="text-xs text-muted-foreground">
                        {memory.content.length} chars
                      </span>
                    </div>
                    <p className="mt-1 line-clamp-2 text-xs">{memory.content}</p>
                  </div>
                </div>
              ))}
            </div>
            <div className="rounded-lg bg-muted p-3">
              <div className="flex items-center justify-between text-sm">
                <span>Total Context Size</span>
                <span className="font-mono font-medium">
                  {memories.reduce((acc, m) => acc + m.content.length, 0)} characters
                </span>
              </div>
            </div>
          </div>
        )}
      </div>
    </Card>
  );
}


function LoadingSkeleton() {
  return (
    <div className="space-y-4">
      {[1, 2, 3].map((i) => (
        <div key={i} className="animate-pulse">
          <div className="h-4 w-3/4 rounded bg-muted" />
          <div className="mt-2 h-3 w-1/2 rounded bg-muted" />
          <div className="mt-3 h-20 w-full rounded bg-muted" />
        </div>
      ))}
    </div>
  );
}

function ResultCard({
  result,
  onCopy,
}: {
  result: SearchResultItem;
  onCopy: (text: string) => void;
}) {
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    onCopy(result.content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const getNodeTypeIcon = (nodeType: string) => {
    switch (nodeType) {
      case "memory":
        return <Brain className="h-4 w-4" />;
      case "entity":
        return <FileText className="h-4 w-4" />;
      default:
        return <FileText className="h-4 w-4" />;
    }
  };

  return (
    <div className="glass-card p-4 transition-all hover:border-[rgba(0,240,255,0.2)]">
      <div className="flex items-start justify-between gap-3">
        <div className="flex items-center gap-2">
          <div className="flex h-6 w-6 shrink-0 items-center justify-center rounded bg-accent text-accent-foreground">
            {getNodeTypeIcon(result.node_type)}
          </div>
          <span className="text-xs capitalize text-muted-foreground">
            {result.node_type} #{result.id}
          </span>
        </div>
        <div className="flex items-center gap-2">
          <span className="rounded bg-primary/10 px-2 py-0.5 text-xs font-medium text-primary">
            {(result.score * 100).toFixed(0)}% match
          </span>
          <button
            onClick={handleCopy}
            className="flex h-6 w-6 items-center justify-center rounded text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
            title="Copy to clipboard"
          >
            {copied ? <Check className="h-3.5 w-3.5 text-green-500" /> : <Copy className="h-3.5 w-3.5" />}
          </button>
        </div>
      </div>
      <p className="mt-3 text-sm">{result.content}</p>
      <div className="mt-2 flex flex-wrap gap-3 text-xs text-muted-foreground">
        <span className="flex items-center gap-1">
          <Sparkles className="h-3 w-3" />
          Vector: {(result.vector_sim * 100).toFixed(0)}%
        </span>
        <span>Centrality: {(result.graph_centrality * 100).toFixed(0)}%</span>
        <span>Recency: {(result.recency * 100).toFixed(0)}%</span>
      </div>
    </div>
  );
}

function ContextDisplay({
  context,
  onCopy,
}: {
  context: string;
  onCopy: (text: string) => void;
}) {
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    onCopy(context);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="relative">
      <div className="absolute right-2 top-2">
        <button
          onClick={handleCopy}
          className="flex items-center gap-1.5 rounded-md border border-border px-2 py-1 text-xs text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
        >
          {copied ? (
            <>
              <Check className="h-3.5 w-3.5 text-green-500" />
              Copied!
            </>
          ) : (
            <>
              <Copy className="h-3.5 w-3.5" />
              Copy Context
            </>
          )}
        </button>
      </div>
      <div className="mt-3 rounded-md bg-accent/50 p-4 text-sm leading-relaxed whitespace-pre-wrap">
        {context}
      </div>
    </div>
  );
}

export default function SearchPage() {
  const router = useRouter();
  const [query, setQuery] = useState("");
  const [searchMode, setSearchMode] = useState<"basic" | "augment">("augment");
  const [loading, setLoading] = useState(false);
  const [searchResults, setSearchResults] = useState<SearchResultItem[]>([]);
  const [augmentResult, setAugmentResult] = useState<AugmentResponse | null>(null);
  const [error, setError] = useState("");
  const [hasSearched, setHasSearched] = useState(false);
  const [copiedText, setCopiedText] = useState("");

  const handleSearch = useCallback(async () => {
    if (!query.trim()) return;

    setLoading(true);
    setError("");
    setHasSearched(true);

    try {
      if (searchMode === "augment") {
        const result = await augmentQuery(
          { query: query.trim(), max_memories: 10, max_hops: 3 },
          undefined
        );
        setAugmentResult(result);
        setSearchResults(result.memories);
      } else {
        const result = await searchMemories(
          { query: query.trim(), limit: 10 },
          undefined
        );
        setSearchResults(result.results);
        setAugmentResult(null);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Search failed");
      setSearchResults([]);
      setAugmentResult(null);
    } finally {
      setLoading(false);
    }
  }, [query, searchMode]);

  const handleCopyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
    setCopiedText(text);
    setTimeout(() => setCopiedText(""), 2000);
  };

  const handleNodeClick = (nodeId: number) => {
    router.push(`/graph?highlight=${nodeId}`);
  };

  const handleSuggestionClick = (suggestion: string) => {
    setQuery(suggestion);
  };

  const clearSearch = () => {
    setQuery("");
    setSearchResults([]);
    setAugmentResult(null);
    setHasSearched(false);
    setError("");
  };

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
        handleSearch();
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [handleSearch]);

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">Search</h1>
          <p className="text-sm text-muted-foreground">
            Query your knowledge graph
          </p>
        </div>
      </div>

      {/* Search Input */}
      <Card>
        <div className="space-y-4">
          <div className="flex gap-2">
            <div className="relative flex-1">
              <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
              <Input
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Ask a question or search..."
                className="pl-10 pr-10"
                onKeyDown={(e) => {
                  if (e.key === "Enter") {
                    handleSearch();
                  }
                }}
              />
              {query && (
                <button
                  onClick={clearSearch}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
                >
                  <X className="h-4 w-4" />
                </button>
              )}
            </div>
            <Button onClick={handleSearch} disabled={loading || !query.trim()}>
              {loading ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <>
                  <Search className="mr-2 h-4 w-4" />
                  Search
                </>
              )}
            </Button>
          </div>

          {/* Search Mode Toggle */}
          <div className="flex items-center gap-4">
            <div className="flex rounded-md bg-muted p-1">
              <button
                onClick={() => setSearchMode("basic")}
                className={`rounded-sm px-3 py-1.5 text-sm transition-all ${
                  searchMode === "basic"
                    ? "bg-background shadow-sm"
                    : "text-muted-foreground hover:text-foreground"
                }`}
              >
                Basic Search
              </button>
              <button
                onClick={() => setSearchMode("augment")}
                className={`rounded-sm px-3 py-1.5 text-sm transition-all ${
                  searchMode === "augment"
                    ? "bg-background shadow-sm"
                    : "text-muted-foreground hover:text-foreground"
                }`}
              >
                Augment (AI)
              </button>
            </div>
            <span className="text-xs text-muted-foreground">
              {searchMode === "augment"
                ? "Uses AI to find relevant context"
                : "Direct keyword search"}
            </span>
          </div>

          {/* Suggestions */}
          {!hasSearched && !query && (
            <div className="pt-2">
              <p className="mb-2 text-xs text-muted-foreground">
                Try asking:
              </p>
              <div className="flex flex-wrap gap-2">
                {SUGGESTED_QUERIES.map((suggestion) => (
                  <button
                    key={suggestion}
                    onClick={() => handleSuggestionClick(suggestion)}
                    className="rounded-full border border-border px-3 py-1 text-xs text-muted-foreground transition-colors hover:border-primary/50 hover:text-foreground"
                  >
                    {suggestion}
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>
      </Card>

      {/* Error Display */}
      {error && (
        <div className="rounded-lg border border-destructive/30 bg-destructive/5 p-4 text-sm text-destructive">
          {error}
        </div>
      )}

      {/* Results */}
      {hasSearched && !loading && !error && searchResults.length === 0 && (
        <div className="py-12 text-center">
          <Search className="mx-auto h-12 w-12 text-muted-foreground" />
          <p className="mt-4 text-lg font-medium">No results found</p>
          <p className="mt-1 text-sm text-muted-foreground">
            Try a different query or add more memories to your knowledge graph.
          </p>
        </div>
      )}

      {/* Context Display (Augment Mode) */}
      {hasSearched && augmentResult && augmentResult.context_text && (
        <Card title="Generated Context">
          <ContextDisplay
            context={augmentResult.context_text}
            onCopy={handleCopyToClipboard}
          />
          {augmentResult.debug && (
            <div className="mt-4 pt-3 border-t border-border">
              <p className="mb-2 text-xs text-muted-foreground">
                Debug Info:
              </p>
              <div className="grid grid-cols-2 gap-2 text-xs md:grid-cols-4">
                <div className="rounded bg-muted p-2">
                  <span className="text-muted-foreground">Vector Search</span>
                  <p className="font-mono">
                    {(
                      (augmentResult.debug.pipeline_timings.vector_search_us ||
                        0) / 1000
                    ).toFixed(1)}
                    ms
                  </p>
                </div>
                <div className="rounded bg-muted p-2">
                  <span className="text-muted-foreground">Graph Expansion</span>
                  <p className="font-mono">
                    {(
                      (augmentResult.debug.pipeline_timings.graph_expansion_us ||
                        0) / 1000
                    ).toFixed(1)}
                    ms
                  </p>
                </div>
                <div className="rounded bg-muted p-2">
                  <span className="text-muted-foreground">Reranking</span>
                  <p className="font-mono">
                    {(
                      (augmentResult.debug.pipeline_timings.reranking_us || 0) /
                      1000
                    ).toFixed(1)}
                    ms
                  </p>
                </div>
                <div className="rounded bg-muted p-2">
                  <span className="text-muted-foreground">Total</span>
                  <p className="font-mono">
                    {(
                      augmentResult.debug.pipeline_timings.total_us / 1000
                    ).toFixed(1)}
                    ms
                  </p>
                </div>
              </div>
            </div>
          )}
        </Card>
      )}

      {/* Search Results */}
      {hasSearched && (loading || searchResults.length > 0) && (
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-lg font-semibold">
              {loading ? "Searching..." : `${searchResults.length} Results`}
            </h2>
            {searchResults.length > 0 && (
              <button
                onClick={() =>
                  handleCopyToClipboard(
                    searchResults.map((r) => r.content).join("\n\n")
                  )
                }
                className="flex items-center gap-1.5 text-sm text-muted-foreground hover:text-foreground"
              >
                {copiedText ? (
                  <Check className="h-4 w-4 text-green-500" />
                ) : (
                  <Copy className="h-4 w-4" />
                )}
                Copy All
              </button>
            )}
          </div>
          {loading ? (
            <LoadingSkeleton />
          ) : (
            <div className="space-y-3">
              {searchResults.map((result) => (
                <ResultCard
                  key={result.id}
                  result={result}
                  onCopy={handleCopyToClipboard}
                />
              ))}
            </div>
          )}
        </div>
      )}

      {/* Query Explainability Panel */}
      {hasSearched && !loading && augmentResult && augmentResult.debug && (
        <ExplainabilityPanel
          memories={searchResults}
          scoreBreakdown={augmentResult.debug.score_breakdown}
          debug={augmentResult.debug}
          onNodeClick={handleNodeClick}
        />
      )}
    </div>
  );
}
