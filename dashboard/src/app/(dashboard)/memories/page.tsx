"use client";

import { useEffect, useState, useCallback } from "react";
import {
  Brain,
  Search,
  ChevronLeft,
  ChevronRight,
  Download,
  X,
  Pencil,
  Trash2,
  Eye,
  Check,
  Calendar,
  Filter,
} from "lucide-react";
import { Card } from "@/components/card";
import {
  listMemories,
  searchMemories,
  getMemory,
  updateMemory,
  deleteMemory,
  getEntity,
} from "@/lib/api";
import type {
  MemoryResponse,
  SearchResultItem,
  EntityResponse,
  NeighborResponse,
} from "@/lib/api";

const PAGE_SIZES = [10, 25, 50];
const NODE_TYPES = ["", "entity", "event", "fact", "skill"];

type DisplayItem = MemoryResponse | SearchResultItem;

function isSearchResult(item: DisplayItem): item is SearchResultItem {
  return "score" in item;
}

function formatTimestamp(ts: number): string {
  if (!ts) return "—";
  const d = new Date(ts * 1000);
  return d.toLocaleDateString(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function dateToUnix(dateStr: string): number | undefined {
  if (!dateStr) return undefined;
  return Math.floor(new Date(dateStr).getTime() / 1000);
}

export default function MemoriesPage() {
  // List state
  const [memories, setMemories] = useState<MemoryResponse[]>([]);
  const [searchResults, setSearchResults] = useState<SearchResultItem[] | null>(
    null
  );
  const [query, setQuery] = useState("");
  const [searching, setSearching] = useState(false);
  const [nodeType, setNodeType] = useState("");
  const [pageSize, setPageSize] = useState(25);
  const [offset, setOffset] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  // Filter state
  const [dateFrom, setDateFrom] = useState("");
  const [dateTo, setDateTo] = useState("");
  const [minScore, setMinScore] = useState("");
  const [showFilters, setShowFilters] = useState(false);

  // Detail panel state
  const [selectedMemory, setSelectedMemory] = useState<MemoryResponse | null>(
    null
  );
  const [detailLoading, setDetailLoading] = useState(false);
  const [relatedEntities, setRelatedEntities] = useState<EntityResponse[]>([]);

  // Edit state
  const [editing, setEditing] = useState(false);
  const [editContent, setEditContent] = useState("");
  const [editMetadata, setEditMetadata] = useState("");
  const [saving, setSaving] = useState(false);

  // Delete state
  const [confirmDelete, setConfirmDelete] = useState<number | null>(null);
  const [deleting, setDeleting] = useState(false);

  // Fetch memories list
  const fetchMemories = useCallback(() => {
    setSearchResults(null);
    setLoading(true);
    setError("");
    listMemories({ node_type: nodeType || undefined, limit: pageSize, offset })
      .then((items) => {
        // Client-side date filter
        let filtered = items;
        const fromTs = dateToUnix(dateFrom);
        const toTs = dateTo ? dateToUnix(dateTo + "T23:59:59") : undefined;
        if (fromTs) filtered = filtered.filter((m) => m.timestamp >= fromTs);
        if (toTs) filtered = filtered.filter((m) => m.timestamp <= toTs);
        setMemories(filtered);
      })
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, [nodeType, pageSize, offset, dateFrom, dateTo]);

  useEffect(() => {
    fetchMemories();
  }, [fetchMemories]);

  // Search
  async function handleSearch(e: React.FormEvent) {
    e.preventDefault();
    if (!query.trim()) {
      setSearchResults(null);
      return;
    }
    setSearching(true);
    setError("");
    try {
      const res = await searchMemories({
        query: query.trim(),
        limit: pageSize,
        node_type: nodeType || undefined,
      });
      let results = res.results;

      // Client-side date filter on search results
      const fromTs = dateToUnix(dateFrom);
      const toTs = dateTo ? dateToUnix(dateTo + "T23:59:59") : undefined;

      // Search results don't have timestamp field from server, use score filter instead
      const minScoreVal = minScore ? parseFloat(minScore) : 0;
      if (minScoreVal > 0) {
        results = results.filter((r) => r.score >= minScoreVal);
      }

      // If date filter is active, re-fetch individual memories for timestamp filtering
      if (fromTs || toTs) {
        const filtered: SearchResultItem[] = [];
        for (const r of results) {
          try {
            const mem = await getMemory(r.id);
            const ts = mem.timestamp;
            if (fromTs && ts < fromTs) continue;
            if (toTs && ts > toTs) continue;
            filtered.push(r);
          } catch {
            filtered.push(r); // include if we can't verify
          }
        }
        results = filtered;
      }

      setSearchResults(results);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Search failed");
    } finally {
      setSearching(false);
    }
  }

  // Open detail panel
  async function openDetail(item: DisplayItem) {
    setDetailLoading(true);
    setEditing(false);
    setConfirmDelete(null);
    try {
      const mem = await getMemory(item.id);
      setSelectedMemory(mem);
      setEditContent(mem.content);
      setEditMetadata(JSON.stringify(mem.metadata, null, 2));

      // Try to find related entities (nodes connected to this memory)
      try {
        const entity = await getEntity(item.id);
        setRelatedEntities(
          entity.neighbors
            ? [
                entity,
                ...((await Promise.allSettled(
                  entity.neighbors.map((n) => getEntity(n.node_id))
                ).then((results) =>
                  results
                    .filter(
                      (r): r is PromiseFulfilledResult<EntityResponse> =>
                        r.status === "fulfilled"
                    )
                    .map((r) => r.value)
                )) ?? []),
              ]
            : [entity]
        );
      } catch {
        setRelatedEntities([]);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load memory");
    } finally {
      setDetailLoading(false);
    }
  }

  // Save edited memory
  async function handleSave() {
    if (!selectedMemory) return;
    setSaving(true);
    setError("");
    try {
      let parsedMeta: Record<string, unknown> = {};
      try {
        parsedMeta = JSON.parse(editMetadata);
      } catch {
        setError("Invalid JSON in metadata");
        setSaving(false);
        return;
      }
      const updated = await updateMemory(selectedMemory.id, {
        content: editContent,
        metadata: parsedMeta,
      });
      setSelectedMemory(updated);
      setEditing(false);
      fetchMemories(); // Refresh list
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to update");
    } finally {
      setSaving(false);
    }
  }

  // Delete memory
  async function handleDelete(id: number) {
    setDeleting(true);
    setError("");
    try {
      await deleteMemory(id);
      setSelectedMemory(null);
      setConfirmDelete(null);
      fetchMemories(); // Refresh list
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to delete");
    } finally {
      setDeleting(false);
    }
  }

  // Export results to JSON
  function handleExport() {
    const data = searchResults ?? memories;
    const blob = new Blob([JSON.stringify(data, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = searchResults
      ? `ucotron-search-${query.replace(/\s+/g, "_")}.json`
      : `ucotron-memories-export.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }

  const displayItems: DisplayItem[] = searchResults ?? memories;

  return (
    <div className="flex h-full gap-4">
      {/* Main list panel */}
      <div className={`flex-1 space-y-4 overflow-auto ${selectedMemory ? "max-w-[60%]" : ""}`}>
        <div className="flex items-center justify-between">
          <h1 className="text-2xl font-bold">Memories</h1>
          <div className="flex gap-2">
            <button
              onClick={() => setShowFilters(!showFilters)}
              className={`flex items-center gap-1 rounded-md border px-3 py-1.5 text-sm transition-colors ${
                showFilters || dateFrom || dateTo || minScore
                  ? "border-primary bg-primary/10 text-primary"
                  : "border-border"
              }`}
            >
              <Filter className="h-3.5 w-3.5" />
              Filters
              {(dateFrom || dateTo || minScore) && (
                <span className="ml-1 flex h-4 w-4 items-center justify-center rounded-full bg-primary text-[10px] text-primary-foreground">
                  {[dateFrom, dateTo, minScore].filter(Boolean).length}
                </span>
              )}
            </button>
            <button
              onClick={handleExport}
              disabled={displayItems.length === 0}
              className="flex items-center gap-1 rounded-md border border-border px-3 py-1.5 text-sm transition-colors hover:bg-muted disabled:opacity-50"
            >
              <Download className="h-3.5 w-3.5" /> Export JSON
            </button>
          </div>
        </div>

        {/* Search Bar */}
        <form onSubmit={handleSearch} className="flex gap-3">
          <div className="relative flex-1">
            <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Semantic search..."
              className="w-full rounded-md border border-border bg-background py-2 pl-10 pr-4 text-sm focus:outline-none focus:ring-2 focus:ring-primary"
            />
          </div>
          <select
            value={nodeType}
            onChange={(e) => {
              setNodeType(e.target.value);
              setOffset(0);
            }}
            className="rounded-md border border-border bg-background px-3 py-2 text-sm"
          >
            <option value="">All types</option>
            {NODE_TYPES.filter(Boolean).map((t) => (
              <option key={t} value={t}>
                {t}
              </option>
            ))}
          </select>
          <select
            value={pageSize}
            onChange={(e) => {
              setPageSize(Number(e.target.value));
              setOffset(0);
            }}
            className="rounded-md border border-border bg-background px-3 py-2 text-sm"
          >
            {PAGE_SIZES.map((s) => (
              <option key={s} value={s}>
                {s} per page
              </option>
            ))}
          </select>
          <button
            type="submit"
            disabled={searching}
            className="rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground transition-colors hover:bg-primary/90 disabled:opacity-50"
          >
            {searching ? "Searching..." : "Search"}
          </button>
        </form>

        {/* Advanced Filters */}
        {showFilters && (
          <Card className="space-y-3">
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-medium">Advanced Filters</h3>
              {(dateFrom || dateTo || minScore) && (
                <button
                  onClick={() => {
                    setDateFrom("");
                    setDateTo("");
                    setMinScore("");
                    setOffset(0);
                  }}
                  className="text-xs text-primary hover:underline"
                >
                  Clear all
                </button>
              )}
            </div>
            <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
              <div>
                <label className="mb-1 block text-xs text-muted-foreground">
                  <Calendar className="mr-1 inline h-3 w-3" />
                  Date from
                </label>
                <input
                  type="date"
                  value={dateFrom}
                  onChange={(e) => {
                    setDateFrom(e.target.value);
                    setOffset(0);
                  }}
                  className="w-full rounded-md border border-border bg-background px-3 py-1.5 text-sm"
                />
              </div>
              <div>
                <label className="mb-1 block text-xs text-muted-foreground">
                  <Calendar className="mr-1 inline h-3 w-3" />
                  Date to
                </label>
                <input
                  type="date"
                  value={dateTo}
                  onChange={(e) => {
                    setDateTo(e.target.value);
                    setOffset(0);
                  }}
                  className="w-full rounded-md border border-border bg-background px-3 py-1.5 text-sm"
                />
              </div>
              <div>
                <label className="mb-1 block text-xs text-muted-foreground">
                  Minimum score
                </label>
                <input
                  type="number"
                  step="0.01"
                  min="0"
                  max="1"
                  value={minScore}
                  onChange={(e) => setMinScore(e.target.value)}
                  placeholder="0.00"
                  className="w-full rounded-md border border-border bg-background px-3 py-1.5 text-sm"
                />
              </div>
            </div>
          </Card>
        )}

        {/* Search results header */}
        {searchResults && (
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <span>{searchResults.length} results</span>
            <button
              onClick={() => {
                setSearchResults(null);
                setQuery("");
              }}
              className="text-primary hover:underline"
            >
              Clear search
            </button>
          </div>
        )}

        {error && (
          <div className="flex items-center justify-between rounded-md border border-destructive/30 bg-destructive/5 p-3 text-sm text-destructive">
            <span>{error}</span>
            <button onClick={() => setError("")}>
              <X className="h-4 w-4" />
            </button>
          </div>
        )}

        {/* Memory List */}
        <div className="space-y-2">
          {loading ? (
            <Card>
              <p className="text-sm text-muted-foreground">Loading...</p>
            </Card>
          ) : displayItems.length === 0 ? (
            <Card>
              <div className="flex flex-col items-center gap-2 py-8">
                <Brain className="h-8 w-8 text-muted-foreground" />
                <p className="text-sm text-muted-foreground">
                  No memories found
                </p>
              </div>
            </Card>
          ) : (
            displayItems.map((item) => (
              <Card
                key={item.id}
                className={`cursor-pointer transition-colors hover:border-primary/30 ${
                  selectedMemory?.id === item.id
                    ? "border-primary bg-primary/5"
                    : ""
                }`}
              >
                <div
                  className="flex items-start justify-between gap-4"
                  onClick={() => openDetail(item)}
                >
                  <div className="min-w-0 flex-1">
                    <p className="line-clamp-2 text-sm">{item.content}</p>
                    <div className="mt-2 flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
                      <span className="rounded-full bg-accent px-2 py-0.5 text-accent-foreground">
                        {item.node_type}
                      </span>
                      <span>ID: {item.id}</span>
                      {"timestamp" in item && (
                        <span>{formatTimestamp(item.timestamp)}</span>
                      )}
                      {isSearchResult(item) && (
                        <span className="rounded-full bg-primary/10 px-2 py-0.5 text-primary">
                          Score: {item.score.toFixed(3)}
                        </span>
                      )}
                    </div>
                  </div>
                  <button
                    className="flex-shrink-0 rounded-md p-1 text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
                    onClick={(e) => {
                      e.stopPropagation();
                      openDetail(item);
                    }}
                  >
                    <Eye className="h-4 w-4" />
                  </button>
                </div>
              </Card>
            ))
          )}
        </div>

        {/* Pagination */}
        {!searchResults && (
          <div className="flex items-center justify-between">
            <p className="text-sm text-muted-foreground">
              Showing {offset + 1}&ndash;{offset + memories.length}
            </p>
            <div className="flex gap-2">
              <button
                onClick={() => setOffset(Math.max(0, offset - pageSize))}
                disabled={offset === 0}
                className="flex items-center gap-1 rounded-md border border-border px-3 py-1.5 text-sm disabled:opacity-50"
              >
                <ChevronLeft className="h-3.5 w-3.5" /> Prev
              </button>
              <button
                onClick={() => setOffset(offset + pageSize)}
                disabled={memories.length < pageSize}
                className="flex items-center gap-1 rounded-md border border-border px-3 py-1.5 text-sm disabled:opacity-50"
              >
                Next <ChevronRight className="h-3.5 w-3.5" />
              </button>
            </div>
          </div>
        )}
      </div>

      {/* Detail Panel */}
      {selectedMemory && (
        <div className="w-[40%] min-w-[320px] space-y-4 overflow-auto rounded-lg border border-border bg-card p-4">
          {/* Detail header */}
          <div className="flex items-center justify-between">
            <h2 className="text-lg font-semibold">Memory Detail</h2>
            <button
              onClick={() => {
                setSelectedMemory(null);
                setEditing(false);
                setConfirmDelete(null);
              }}
              className="rounded-md p-1 text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
            >
              <X className="h-4 w-4" />
            </button>
          </div>

          {detailLoading ? (
            <p className="text-sm text-muted-foreground">Loading details...</p>
          ) : (
            <>
              {/* Actions */}
              <div className="flex gap-2">
                {!editing && (
                  <button
                    onClick={() => setEditing(true)}
                    className="flex items-center gap-1 rounded-md border border-border px-3 py-1.5 text-sm transition-colors hover:bg-muted"
                  >
                    <Pencil className="h-3.5 w-3.5" /> Edit
                  </button>
                )}
                {editing && (
                  <>
                    <button
                      onClick={handleSave}
                      disabled={saving}
                      className="flex items-center gap-1 rounded-md bg-primary px-3 py-1.5 text-sm font-medium text-primary-foreground transition-colors hover:bg-primary/90 disabled:opacity-50"
                    >
                      <Check className="h-3.5 w-3.5" />
                      {saving ? "Saving..." : "Save"}
                    </button>
                    <button
                      onClick={() => {
                        setEditing(false);
                        setEditContent(selectedMemory.content);
                        setEditMetadata(
                          JSON.stringify(selectedMemory.metadata, null, 2)
                        );
                      }}
                      className="flex items-center gap-1 rounded-md border border-border px-3 py-1.5 text-sm transition-colors hover:bg-muted"
                    >
                      Cancel
                    </button>
                  </>
                )}
                {confirmDelete === selectedMemory.id ? (
                  <div className="flex items-center gap-2">
                    <span className="text-xs text-destructive">
                      Are you sure?
                    </span>
                    <button
                      onClick={() => handleDelete(selectedMemory.id)}
                      disabled={deleting}
                      className="rounded-md bg-destructive px-3 py-1.5 text-sm font-medium text-white transition-colors hover:bg-destructive/90 disabled:opacity-50"
                    >
                      {deleting ? "Deleting..." : "Confirm"}
                    </button>
                    <button
                      onClick={() => setConfirmDelete(null)}
                      className="rounded-md border border-border px-3 py-1.5 text-sm transition-colors hover:bg-muted"
                    >
                      Cancel
                    </button>
                  </div>
                ) : (
                  <button
                    onClick={() => setConfirmDelete(selectedMemory.id)}
                    className="flex items-center gap-1 rounded-md border border-destructive/30 px-3 py-1.5 text-sm text-destructive transition-colors hover:bg-destructive/5"
                  >
                    <Trash2 className="h-3.5 w-3.5" /> Delete
                  </button>
                )}
              </div>

              {/* Content */}
              <div>
                <h3 className="mb-1 text-xs font-medium uppercase text-muted-foreground">
                  Content
                </h3>
                {editing ? (
                  <textarea
                    value={editContent}
                    onChange={(e) => setEditContent(e.target.value)}
                    rows={4}
                    className="w-full rounded-md border border-border bg-background p-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary"
                  />
                ) : (
                  <p className="whitespace-pre-wrap text-sm">
                    {selectedMemory.content}
                  </p>
                )}
              </div>

              {/* Properties */}
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <span className="text-xs text-muted-foreground">ID</span>
                  <p className="text-sm font-mono">{selectedMemory.id}</p>
                </div>
                <div>
                  <span className="text-xs text-muted-foreground">Type</span>
                  <p className="text-sm">
                    <span className="rounded-full bg-accent px-2 py-0.5 text-accent-foreground">
                      {selectedMemory.node_type}
                    </span>
                  </p>
                </div>
                <div>
                  <span className="text-xs text-muted-foreground">
                    Timestamp
                  </span>
                  <p className="text-sm">
                    {formatTimestamp(selectedMemory.timestamp)}
                  </p>
                </div>
                <div>
                  <span className="text-xs text-muted-foreground">
                    Unix Time
                  </span>
                  <p className="text-sm font-mono">
                    {selectedMemory.timestamp}
                  </p>
                </div>
              </div>

              {/* Metadata */}
              <div>
                <h3 className="mb-1 text-xs font-medium uppercase text-muted-foreground">
                  Metadata
                </h3>
                {editing ? (
                  <textarea
                    value={editMetadata}
                    onChange={(e) => setEditMetadata(e.target.value)}
                    rows={6}
                    className="w-full rounded-md border border-border bg-background p-2 font-mono text-xs focus:outline-none focus:ring-2 focus:ring-primary"
                  />
                ) : Object.keys(selectedMemory.metadata).length > 0 ? (
                  <pre className="overflow-x-auto rounded-md bg-muted p-2 text-xs">
                    {JSON.stringify(selectedMemory.metadata, null, 2)}
                  </pre>
                ) : (
                  <p className="text-sm text-muted-foreground">No metadata</p>
                )}
              </div>

              {/* Related entities */}
              {relatedEntities.length > 0 && (
                <div>
                  <h3 className="mb-1 text-xs font-medium uppercase text-muted-foreground">
                    Related Entities
                  </h3>
                  <div className="space-y-1">
                    {relatedEntities.map((ent) => (
                      <div
                        key={ent.id}
                        className="flex items-center gap-2 rounded-md bg-muted p-2 text-sm"
                      >
                        <span className="rounded-full bg-accent px-2 py-0.5 text-xs text-accent-foreground">
                          {ent.node_type}
                        </span>
                        <span className="truncate">{ent.content}</span>
                        {ent.neighbors && ent.neighbors.length > 0 && (
                          <span className="ml-auto text-xs text-muted-foreground">
                            {ent.neighbors.length} relations
                          </span>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Relations from the first entity that has neighbors */}
              {relatedEntities.some(
                (e) => e.neighbors && e.neighbors.length > 0
              ) && (
                <div>
                  <h3 className="mb-1 text-xs font-medium uppercase text-muted-foreground">
                    Relations
                  </h3>
                  <div className="space-y-1">
                    {relatedEntities
                      .flatMap((e) => e.neighbors ?? [])
                      .slice(0, 20)
                      .map((n: NeighborResponse, i: number) => (
                        <div
                          key={`${n.node_id}-${i}`}
                          className="flex items-center gap-2 rounded-md bg-muted/50 p-2 text-xs"
                        >
                          <span className="rounded bg-primary/10 px-1.5 py-0.5 font-medium text-primary">
                            {n.edge_type}
                          </span>
                          <span className="truncate">{n.content}</span>
                          <span className="ml-auto font-mono text-muted-foreground">
                            w={n.weight.toFixed(2)}
                          </span>
                        </div>
                      ))}
                  </div>
                </div>
              )}
            </>
          )}
        </div>
      )}
    </div>
  );
}
