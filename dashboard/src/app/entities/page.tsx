"use client";

import { useEffect, useState } from "react";
import { Users, ChevronLeft, ChevronRight, ArrowRight } from "lucide-react";
import { Card } from "@/components/card";
import { listEntities, getEntity } from "@/lib/api";
import type { EntityResponse } from "@/lib/api";

export default function EntitiesPage() {
  const [entities, setEntities] = useState<EntityResponse[]>([]);
  const [selected, setSelected] = useState<EntityResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [detailLoading, setDetailLoading] = useState(false);
  const [error, setError] = useState("");
  const [offset, setOffset] = useState(0);
  const pageSize = 25;

  useEffect(() => {
    setLoading(true);
    listEntities({ limit: pageSize, offset })
      .then(setEntities)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, [offset]);

  async function handleSelect(id: number) {
    setDetailLoading(true);
    try {
      const entity = await getEntity(id);
      setSelected(entity);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load entity");
    } finally {
      setDetailLoading(false);
    }
  }

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold">Entities</h1>

      {error && (
        <div className="rounded-md border border-destructive/30 bg-destructive/5 p-3 text-sm text-destructive">
          {error}
        </div>
      )}

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        {/* Entity List */}
        <div className="space-y-2 lg:col-span-2">
          {loading ? (
            <Card><p className="text-sm text-muted-foreground">Loading...</p></Card>
          ) : entities.length === 0 ? (
            <Card>
              <div className="flex flex-col items-center gap-2 py-8">
                <Users className="h-8 w-8 text-muted-foreground" />
                <p className="text-sm text-muted-foreground">No entities found</p>
              </div>
            </Card>
          ) : (
            entities.map((e) => (
              <button
                key={e.id}
                onClick={() => handleSelect(e.id)}
                className={`w-full text-left rounded-lg border p-3 transition-colors ${
                  selected?.id === e.id
                    ? "border-primary bg-accent"
                    : "border-border bg-card hover:border-primary/30"
                }`}
              >
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium">{e.content}</p>
                    <p className="text-xs text-muted-foreground">
                      {e.node_type} &middot; ID: {e.id}
                    </p>
                  </div>
                  <ArrowRight className="h-4 w-4 text-muted-foreground" />
                </div>
              </button>
            ))
          )}

          <div className="flex items-center justify-between pt-2">
            <p className="text-sm text-muted-foreground">
              Showing {offset + 1}–{offset + entities.length}
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
                disabled={entities.length < pageSize}
                className="flex items-center gap-1 rounded-md border border-border px-3 py-1.5 text-sm disabled:opacity-50"
              >
                Next <ChevronRight className="h-3.5 w-3.5" />
              </button>
            </div>
          </div>
        </div>

        {/* Detail Panel */}
        <div>
          {detailLoading ? (
            <Card><p className="text-sm text-muted-foreground">Loading...</p></Card>
          ) : selected ? (
            <Card title={`Entity: ${selected.content}`}>
              <dl className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <dt className="text-muted-foreground">ID</dt>
                  <dd className="font-mono">{selected.id}</dd>
                </div>
                <div className="flex justify-between">
                  <dt className="text-muted-foreground">Type</dt>
                  <dd>{selected.node_type}</dd>
                </div>
                <div className="flex justify-between">
                  <dt className="text-muted-foreground">Timestamp</dt>
                  <dd className="font-mono text-xs">
                    {new Date(selected.timestamp * 1000).toLocaleString()}
                  </dd>
                </div>
              </dl>

              {selected.neighbors && selected.neighbors.length > 0 && (
                <div className="mt-4 border-t border-border pt-4">
                  <h4 className="mb-2 text-sm font-medium text-muted-foreground">
                    Neighbors ({selected.neighbors.length})
                  </h4>
                  <div className="space-y-1.5">
                    {selected.neighbors.map((n) => (
                      <div
                        key={n.node_id}
                        className="flex items-center justify-between rounded-md bg-muted px-3 py-1.5 text-xs"
                      >
                        <span>{n.content}</span>
                        <span className="rounded-full bg-accent px-2 py-0.5 text-accent-foreground">
                          {n.edge_type}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </Card>
          ) : (
            <Card>
              <p className="text-sm text-muted-foreground">
                Select an entity to view details
              </p>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}
