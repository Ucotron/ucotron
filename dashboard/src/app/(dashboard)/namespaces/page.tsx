"use client";

import { useEffect, useState, useCallback } from "react";
import {
  Folder,
  Plus,
  RefreshCw,
  X,
  CheckCircle2,
  AlertCircle,
  Trash2,
  Clock,
  Database,
  FileText,
  Network,
} from "lucide-react";
import { Card } from "@/components/card";
import {
  listNamespaces,
  createNamespace,
  deleteNamespace,
  type NamespaceInfo,
  type NamespaceListResponse,
} from "@/lib/api";
import { useNamespace } from "@/components/namespace-context";

function formatTimestamp(ts: number): string {
  if (ts === 0) return "Never";
  return new Date(ts * 1000).toLocaleString();
}

function formatNumber(n: number): string {
  if (n >= 1000000) return `${(n / 1000000).toFixed(1)}M`;
  if (n >= 1000) return `${(n / 1000).toFixed(1)}K`;
  return String(n);
}

export default function NamespacesPage() {
  const { namespace: currentNamespace, setNamespace } = useNamespace();
  const [namespaces, setNamespaces] = useState<NamespaceInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");

  // Create modal state
  const [createOpen, setCreateOpen] = useState(false);
  const [createName, setCreateName] = useState("");
  const [creating, setCreating] = useState(false);

  // Delete modal state
  const [deleteTarget, setDeleteTarget] = useState<NamespaceInfo | null>(null);
  const [deleting, setDeleting] = useState(false);

  // Switch modal state
  const [switchTarget, setSwitchTarget] = useState<NamespaceInfo | null>(null);

  const fetchNamespaces = useCallback(async () => {
    setLoading(true);
    setError("");
    try {
      const result: NamespaceListResponse = await listNamespaces();
      setNamespaces(result.namespaces);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load namespaces");
    }
    setLoading(false);
  }, []);

  useEffect(() => {
    fetchNamespaces();
  }, [fetchNamespaces]);

  const handleCreate = async () => {
    if (!createName.trim()) return;
    setCreating(true);
    setError("");
    try {
      const result = await createNamespace(createName.trim());
      setSuccess(`Namespace "${result.name}" created successfully`);
      setCreateOpen(false);
      setCreateName("");
      fetchNamespaces();
      setTimeout(() => setSuccess(""), 3000);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to create namespace");
    }
    setCreating(false);
  };

  const handleDelete = async () => {
    if (!deleteTarget) return;
    setDeleting(true);
    setError("");
    try {
      const result = await deleteNamespace(deleteTarget.name);
      setSuccess(`Namespace "${deleteTarget.name}" deleted (${result.nodes_deleted} nodes removed)`);
      setDeleteTarget(null);
      fetchNamespaces();
      if (currentNamespace === deleteTarget.name) {
        setNamespace("default");
      }
      setTimeout(() => setSuccess(""), 3000);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to delete namespace");
    }
    setDeleting(false);
  };

  const handleSwitch = (ns: NamespaceInfo) => {
    setNamespace(ns.name);
    setSwitchTarget(ns);
    setTimeout(() => setSwitchTarget(null), 2000);
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <Folder className="h-6 w-6 text-primary" />
          <h1 className="text-2xl font-bold">Namespaces</h1>
          <span className="rounded bg-muted px-2 py-0.5 text-xs text-muted-foreground">
            {namespaces.length}
          </span>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={fetchNamespaces}
            className="flex items-center gap-2 rounded-md border border-border px-3 py-1.5 text-sm text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
          >
            <RefreshCw className="h-3.5 w-3.5" />
            Refresh
          </button>
          <button
            onClick={() => setCreateOpen(true)}
            className="flex items-center gap-2 rounded-md bg-primary px-4 py-1.5 text-sm font-medium text-primary-foreground transition-colors hover:bg-primary/90"
          >
            <Plus className="h-4 w-4" />
            New Namespace
          </button>
        </div>
      </div>

      {/* Success message */}
      {success && (
        <div className="flex items-center gap-2 rounded-lg border border-green-500/30 bg-green-500/5 p-3 text-sm text-green-700 dark:text-green-400">
          <CheckCircle2 className="h-4 w-4 flex-shrink-0" />
          {success}
          <button
            onClick={() => setSuccess("")}
            className="ml-auto rounded p-0.5 hover:bg-green-500/10"
          >
            <X className="h-3.5 w-3.5" />
          </button>
        </div>
      )}

      {/* Error banner */}
      {error && (
        <div className="rounded-lg border border-destructive/30 bg-destructive/5 p-3 text-sm text-destructive">
          {error}
        </div>
      )}

      {/* Loading */}
      {loading && namespaces.length === 0 && (
        <div className="flex items-center gap-2 text-muted-foreground">
          <RefreshCw className="h-4 w-4 animate-spin" />
          Loading namespaces...
        </div>
      )}

      {/* Empty state */}
      {!loading && namespaces.length === 0 && (
        <Card>
          <div className="flex flex-col items-center gap-3 py-8 text-center">
            <Folder className="h-12 w-12 text-muted-foreground/30" />
            <p className="text-sm text-muted-foreground">
              No namespaces found. Create a namespace to organize your data.
            </p>
            <button
              onClick={() => setCreateOpen(true)}
              className="flex items-center gap-2 rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground transition-colors hover:bg-primary/90"
            >
              <Plus className="h-4 w-4" />
              Create Namespace
            </button>
          </div>
        </Card>
      )}

      {/* Namespace list */}
      {namespaces.length > 0 && (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {namespaces.map((ns) => (
            <Card key={ns.name}>
              <div className="flex items-start justify-between">
                <div className="flex items-center gap-2">
                  <Folder className="h-5 w-5 text-primary" />
                  <h3 className="font-mono font-medium">{ns.name}</h3>
                  {ns.name === currentNamespace && (
                    <span className="rounded bg-primary/10 px-1.5 py-0.5 text-[10px] font-medium text-primary">
                      Active
                    </span>
                  )}
                </div>
                <div className="flex items-center gap-1">
                  {ns.name !== currentNamespace && (
                    <button
                      onClick={() => handleSwitch(ns)}
                      className="rounded-md border border-border px-2 py-1 text-xs text-muted-foreground transition-colors hover:bg-accent hover:text-foreground"
                    >
                      Switch
                    </button>
                  )}
                  {ns.name !== "default" && (
                    <button
                      onClick={() => setDeleteTarget(ns)}
                      className="rounded-md border border-border p-1 text-muted-foreground transition-colors hover:bg-accent hover:text-destructive"
                      title="Delete namespace"
                    >
                      <Trash2 className="h-3.5 w-3.5" />
                    </button>
                  )}
                </div>
              </div>

              {/* Stats */}
              <div className="mt-4 grid grid-cols-2 gap-3">
                <div className="flex items-center gap-2 rounded-md bg-muted/30 p-2">
                  <Database className="h-4 w-4 text-muted-foreground" />
                  <div>
                    <p className="text-xs text-muted-foreground">Total Nodes</p>
                    <p className="font-semibold">{formatNumber(ns.total_nodes)}</p>
                  </div>
                </div>
                <div className="flex items-center gap-2 rounded-md bg-muted/30 p-2">
                  <FileText className="h-4 w-4 text-muted-foreground" />
                  <div>
                    <p className="text-xs text-muted-foreground">Memories</p>
                    <p className="font-semibold">{formatNumber(ns.memory_count)}</p>
                  </div>
                </div>
                <div className="flex items-center gap-2 rounded-md bg-muted/30 p-2">
                  <Network className="h-4 w-4 text-muted-foreground" />
                  <div>
                    <p className="text-xs text-muted-foreground">Entities</p>
                    <p className="font-semibold">{formatNumber(ns.entity_count)}</p>
                  </div>
                </div>
                <div className="flex items-center gap-2 rounded-md bg-muted/30 p-2">
                  <Clock className="h-4 w-4 text-muted-foreground" />
                  <div>
                    <p className="text-xs text-muted-foreground">Last Activity</p>
                    <p className="text-xs">{formatTimestamp(ns.last_activity)}</p>
                  </div>
                </div>
              </div>
            </Card>
          ))}
        </div>
      )}

      {/* Create Namespace Modal */}
      {createOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
          <div className="w-full max-w-md rounded-lg border border-border bg-card p-6 shadow-xl">
            <div className="mb-4 flex items-center justify-between">
              <h2 className="text-lg font-semibold">Create Namespace</h2>
              <button
                onClick={() => {
                  setCreateOpen(false);
                  setCreateName("");
                  setError("");
                }}
                className="rounded-md p-1 text-muted-foreground hover:text-foreground"
              >
                <X className="h-5 w-5" />
              </button>
            </div>

            <div className="space-y-4">
              <div>
                <label className="mb-1.5 block text-sm text-muted-foreground">
                  Namespace Name
                </label>
                <input
                  type="text"
                  value={createName}
                  onChange={(e) => setCreateName(e.target.value)}
                  placeholder="my-namespace"
                  className="w-full rounded-md border border-border bg-background px-3 py-2 font-mono text-sm focus:outline-none focus:ring-2 focus:ring-primary"
                  autoFocus
                />
                <p className="mt-1 text-[11px] text-muted-foreground">
                  Use lowercase letters, numbers, and hyphens. Cannot be changed after creation.
                </p>
              </div>

              <div className="flex justify-end gap-2 pt-2">
                <button
                  onClick={() => {
                    setCreateOpen(false);
                    setCreateName("");
                    setError("");
                  }}
                  className="rounded-md border border-border px-4 py-2 text-sm transition-colors hover:bg-muted"
                >
                  Cancel
                </button>
                <button
                  onClick={handleCreate}
                  disabled={!createName.trim() || creating}
                  className="flex items-center gap-2 rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground transition-colors hover:bg-primary/90 disabled:opacity-50"
                >
                  {creating ? (
                    <>
                      <RefreshCw className="h-4 w-4 animate-spin" />
                      Creating...
                    </>
                  ) : (
                    <>
                      <Plus className="h-4 w-4" />
                      Create
                    </>
                  )}
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Delete Confirmation Modal */}
      {deleteTarget && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
          <div className="w-full max-w-md rounded-lg border border-border bg-card p-6 shadow-xl">
            <div className="mb-4 flex items-center gap-3">
              <div className="flex h-10 w-10 items-center justify-center rounded-full bg-destructive/10">
                <AlertCircle className="h-5 w-5 text-destructive" />
              </div>
              <div>
                <h2 className="text-lg font-semibold">Delete Namespace</h2>
                <p className="text-sm text-muted-foreground">This action cannot be undone</p>
              </div>
            </div>

            <div className="mb-6 rounded-md bg-muted/50 p-4">
              <p className="text-sm">
                You are about to delete namespace{" "}
                <span className="font-mono font-medium">{deleteTarget.name}</span>.
              </p>
              <p className="mt-2 text-sm text-muted-foreground">
                This will permanently remove {deleteTarget.total_nodes} nodes,{" "}
                {deleteTarget.memory_count} memories, and {deleteTarget.entity_count} entities.
              </p>
            </div>

            <div className="flex justify-end gap-2">
              <button
                onClick={() => setDeleteTarget(null)}
                className="rounded-md border border-border px-4 py-2 text-sm transition-colors hover:bg-muted"
              >
                Cancel
              </button>
              <button
                onClick={handleDelete}
                disabled={deleting}
                className="flex items-center gap-2 rounded-md bg-destructive px-4 py-2 text-sm font-medium text-destructive-foreground transition-colors hover:bg-destructive/90 disabled:opacity-50"
              >
                {deleting ? (
                  <>
                    <RefreshCw className="h-4 w-4 animate-spin" />
                    Deleting...
                  </>
                ) : (
                  <>
                    <Trash2 className="h-4 w-4" />
                    Delete Namespace
                  </>
                )}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Switch Confirmation Toast */}
      {switchTarget && (
        <div className="fixed bottom-6 right-6 z-50 flex items-center gap-2 rounded-lg border border-green-500/30 bg-card p-4 shadow-lg">
          <CheckCircle2 className="h-5 w-5 text-green-500" />
          <span className="text-sm font-medium">
            Switched to namespace <span className="font-mono">{switchTarget.name}</span>
          </span>
        </div>
      )}
    </div>
  );
}
