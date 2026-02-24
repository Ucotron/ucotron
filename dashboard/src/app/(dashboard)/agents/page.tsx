"use client";

import { useEffect, useState, useCallback } from "react";
import {
  Bot,
  Plus,
  Trash2,
  Copy,
  Merge,
  Share2,
  RefreshCw,
  ChevronDown,
  ChevronUp,
  X,
  Database,
} from "lucide-react";
import { Card } from "@/components/card";
import {
  listAgents,
  createAgent,
  deleteAgent,
  cloneAgent,
  listShares,
  createShare,
  deleteShare,
} from "@/lib/api";
import type {
  AgentResponse,
  ShareResponse,
} from "@/lib/api";

function formatTimestamp(ts: number): string {
  if (ts === 0) return "Never";
  return new Date(ts * 1000).toLocaleString();
}

type ModalType = "create" | "clone" | "share" | null;

export default function AgentsPage() {
  const [agents, setAgents] = useState<AgentResponse[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [expandedAgent, setExpandedAgent] = useState<string | null>(null);
  const [shares, setShares] = useState<ShareResponse[]>([]);
  const [sharesLoading, setSharesLoading] = useState(false);

  // Modal state
  const [modal, setModal] = useState<ModalType>(null);
  const [modalAgentId, setModalAgentId] = useState("");

  // Create form
  const [newName, setNewName] = useState("");
  const [creating, setCreating] = useState(false);

  // Clone form
  const [cloneTargetNs, setCloneTargetNs] = useState("");
  const [cloning, setCloning] = useState(false);
  const [cloneResult, setCloneResult] = useState("");

  // Share form
  const [shareTargetId, setShareTargetId] = useState("");
  const [sharePermission, setSharePermission] = useState("read");
  const [sharing, setSharing] = useState(false);

  // Delete state
  const [deleting, setDeleting] = useState<string | null>(null);

  // Pagination
  const [offset, setOffset] = useState(0);
  const limit = 20;

  const fetchAgents = useCallback(async () => {
    setLoading(true);
    setError("");
    try {
      const result = await listAgents({ limit, offset });
      setAgents(result.agents);
      setTotal(result.total);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load agents");
    }
    setLoading(false);
  }, [offset]);

  useEffect(() => {
    fetchAgents();
  }, [fetchAgents]);

  const fetchShares = async (agentId: string) => {
    setSharesLoading(true);
    try {
      const result = await listShares(agentId);
      setShares(result.shares);
    } catch {
      setShares([]);
    }
    setSharesLoading(false);
  };

  const handleExpand = (agentId: string) => {
    if (expandedAgent === agentId) {
      setExpandedAgent(null);
      setShares([]);
    } else {
      setExpandedAgent(agentId);
      fetchShares(agentId);
    }
  };

  const handleCreate = async () => {
    if (!newName.trim()) return;
    setCreating(true);
    setError("");
    try {
      await createAgent({ name: newName.trim() });
      setNewName("");
      setModal(null);
      await fetchAgents();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to create agent");
    }
    setCreating(false);
  };

  const handleDelete = async (id: string) => {
    if (!confirm(`Delete agent "${id}" and all its data? This cannot be undone.`)) return;
    setDeleting(id);
    setError("");
    try {
      await deleteAgent(id);
      if (expandedAgent === id) {
        setExpandedAgent(null);
        setShares([]);
      }
      await fetchAgents();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to delete agent");
    }
    setDeleting(null);
  };

  const handleClone = async () => {
    setCloning(true);
    setError("");
    setCloneResult("");
    try {
      const result = await cloneAgent(modalAgentId, {
        target_namespace: cloneTargetNs.trim() || undefined,
      });
      setCloneResult(
        `Cloned ${result.nodes_copied} nodes and ${result.edges_copied} edges to "${result.target_namespace}"`
      );
      await fetchAgents();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to clone agent");
    }
    setCloning(false);
  };

  const handleShare = async () => {
    if (!shareTargetId.trim()) return;
    setSharing(true);
    setError("");
    try {
      await createShare(modalAgentId, {
        target_agent_id: shareTargetId.trim(),
        permission: sharePermission,
      });
      setShareTargetId("");
      setModal(null);
      if (expandedAgent === modalAgentId) {
        await fetchShares(modalAgentId);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to share agent");
    }
    setSharing(false);
  };

  const handleDeleteShare = async (agentId: string, targetId: string) => {
    try {
      await deleteShare(agentId, targetId);
      await fetchShares(agentId);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to revoke share");
    }
  };

  const openCloneModal = (agentId: string) => {
    setModalAgentId(agentId);
    setCloneTargetNs("");
    setCloneResult("");
    setModal("clone");
  };

  const openShareModal = (agentId: string) => {
    setModalAgentId(agentId);
    setShareTargetId("");
    setSharePermission("read");
    setModal("share");
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <Bot className="h-6 w-6 text-primary" />
          <h1 className="text-2xl font-bold">Agents</h1>
          <span className="rounded bg-muted px-2 py-0.5 text-xs text-muted-foreground">
            {total}
          </span>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={fetchAgents}
            className="flex items-center gap-2 rounded-md border border-border px-3 py-1.5 text-sm text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
          >
            <RefreshCw className="h-3.5 w-3.5" />
            Refresh
          </button>
          <button
            onClick={() => {
              setNewName("");
              setModal("create");
            }}
            className="flex items-center gap-2 rounded-md bg-primary px-4 py-1.5 text-sm font-medium text-primary-foreground transition-colors hover:bg-primary/90"
          >
            <Plus className="h-4 w-4" />
            Create Agent
          </button>
        </div>
      </div>

      {/* Error banner */}
      {error && (
        <div className="rounded-lg border border-destructive/30 bg-destructive/5 p-3 text-sm text-destructive">
          {error}
        </div>
      )}

      {/* Loading */}
      {loading && agents.length === 0 && (
        <div className="flex items-center gap-2 text-muted-foreground">
          <RefreshCw className="h-4 w-4 animate-spin" />
          Loading agents...
        </div>
      )}

      {/* Empty state */}
      {!loading && agents.length === 0 && (
        <Card>
          <div className="flex flex-col items-center gap-3 py-8 text-center">
            <Bot className="h-12 w-12 text-muted-foreground/30" />
            <p className="text-sm text-muted-foreground">
              No agents found. Create your first agent to get started.
            </p>
          </div>
        </Card>
      )}

      {/* Agent list */}
      {agents.length > 0 && (
        <Card>
          <div className="divide-y divide-border">
            {agents.map((agent) => (
              <div key={agent.id} className="py-3 first:pt-0 last:pb-0">
                {/* Agent row */}
                <div className="flex items-center justify-between">
                  <button
                    onClick={() => handleExpand(agent.id)}
                    className="flex items-center gap-2 text-sm font-medium hover:text-primary"
                  >
                    {expandedAgent === agent.id ? (
                      <ChevronUp className="h-3.5 w-3.5" />
                    ) : (
                      <ChevronDown className="h-3.5 w-3.5" />
                    )}
                    <Bot className="h-3.5 w-3.5 text-muted-foreground" />
                    <span>{agent.name}</span>
                    <span className="font-mono text-xs text-muted-foreground">
                      {agent.id}
                    </span>
                  </button>
                  <div className="flex items-center gap-2">
                    <span className="text-xs text-muted-foreground">
                      {agent.owner}
                    </span>
                    <button
                      onClick={() => openShareModal(agent.id)}
                      className="rounded-md p-1 text-muted-foreground transition-colors hover:bg-accent hover:text-accent-foreground"
                      title="Share"
                    >
                      <Share2 className="h-4 w-4" />
                    </button>
                    <button
                      onClick={() => openCloneModal(agent.id)}
                      className="rounded-md p-1 text-muted-foreground transition-colors hover:bg-accent hover:text-accent-foreground"
                      title="Clone"
                    >
                      <Copy className="h-4 w-4" />
                    </button>
                    <button
                      onClick={() => handleDelete(agent.id)}
                      disabled={deleting === agent.id}
                      className="rounded-md p-1 text-muted-foreground transition-colors hover:bg-destructive/10 hover:text-destructive"
                      title="Delete"
                    >
                      <Trash2 className="h-4 w-4" />
                    </button>
                  </div>
                </div>

                {/* Expanded details */}
                {expandedAgent === agent.id && (
                  <div className="mt-3 space-y-4 pl-8">
                    {/* Agent details */}
                    <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
                      <div>
                        <p className="text-xs text-muted-foreground">Namespace</p>
                        <p className="font-mono text-sm">{agent.namespace}</p>
                      </div>
                      <div>
                        <p className="text-xs text-muted-foreground">Owner</p>
                        <p className="text-sm font-semibold">{agent.owner}</p>
                      </div>
                      <div>
                        <p className="text-xs text-muted-foreground">Created</p>
                        <p className="text-sm">{formatTimestamp(agent.created_at)}</p>
                      </div>
                      <div>
                        <p className="text-xs text-muted-foreground">Config Keys</p>
                        <p className="text-sm font-semibold">
                          {Object.keys(agent.config).length}
                        </p>
                      </div>
                    </div>

                    {/* Config details */}
                    {Object.keys(agent.config).length > 0 && (
                      <div>
                        <p className="mb-1 text-xs text-muted-foreground">Configuration</p>
                        <div className="rounded-md bg-muted/50 p-2">
                          <pre className="text-xs text-foreground/80">
                            {JSON.stringify(agent.config, null, 2)}
                          </pre>
                        </div>
                      </div>
                    )}

                    {/* Shares */}
                    <div>
                      <p className="mb-2 text-xs text-muted-foreground">
                        Shares
                        {sharesLoading && " (loading...)"}
                      </p>
                      {shares.length === 0 && !sharesLoading ? (
                        <p className="text-xs text-muted-foreground/60">
                          No shares configured.
                        </p>
                      ) : (
                        <div className="space-y-1">
                          {shares.map((share) => (
                            <div
                              key={`${share.agent_id}-${share.target_agent_id}`}
                              className="flex items-center justify-between rounded-md bg-muted/30 px-3 py-1.5"
                            >
                              <div className="flex items-center gap-2 text-xs">
                                <Share2 className="h-3 w-3 text-muted-foreground" />
                                <span className="font-mono">{share.target_agent_id}</span>
                                <span
                                  className={`rounded px-1.5 py-0.5 text-[10px] font-medium ${
                                    share.permission === "read_write"
                                      ? "bg-amber-500/10 text-amber-600 dark:text-amber-400"
                                      : "bg-blue-500/10 text-blue-600 dark:text-blue-400"
                                  }`}
                                >
                                  {share.permission}
                                </span>
                              </div>
                              <button
                                onClick={() =>
                                  handleDeleteShare(agent.id, share.target_agent_id)
                                }
                                className="rounded p-0.5 text-muted-foreground transition-colors hover:text-destructive"
                                title="Revoke share"
                              >
                                <X className="h-3 w-3" />
                              </button>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>
        </Card>
      )}

      {/* Pagination */}
      {total > limit && (
        <div className="flex items-center justify-between">
          <span className="text-sm text-muted-foreground">
            Showing {offset + 1}-{Math.min(offset + limit, total)} of {total}
          </span>
          <div className="flex gap-2">
            <button
              onClick={() => setOffset(Math.max(0, offset - limit))}
              disabled={offset === 0}
              className="rounded-md border border-border px-3 py-1.5 text-sm transition-colors hover:bg-muted disabled:opacity-50"
            >
              Previous
            </button>
            <button
              onClick={() => setOffset(offset + limit)}
              disabled={offset + limit >= total}
              className="rounded-md border border-border px-3 py-1.5 text-sm transition-colors hover:bg-muted disabled:opacity-50"
            >
              Next
            </button>
          </div>
        </div>
      )}

      {/* Create Modal */}
      {modal === "create" && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
          <div className="w-full max-w-md rounded-lg border border-border bg-card p-6 shadow-xl">
            <div className="mb-4 flex items-center justify-between">
              <h2 className="text-lg font-semibold">Create Agent</h2>
              <button
                onClick={() => setModal(null)}
                className="rounded-md p-1 text-muted-foreground hover:text-foreground"
              >
                <X className="h-5 w-5" />
              </button>
            </div>
            <div className="space-y-4">
              <div>
                <label className="mb-1 block text-sm text-muted-foreground">
                  Agent Name
                </label>
                <input
                  type="text"
                  value={newName}
                  onChange={(e) => setNewName(e.target.value)}
                  placeholder="my-agent"
                  className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary"
                  onKeyDown={(e) => {
                    if (e.key === "Enter") handleCreate();
                  }}
                  autoFocus
                />
              </div>
              <div className="flex justify-end gap-2">
                <button
                  onClick={() => setModal(null)}
                  className="rounded-md border border-border px-4 py-2 text-sm transition-colors hover:bg-muted"
                >
                  Cancel
                </button>
                <button
                  onClick={handleCreate}
                  disabled={creating || !newName.trim()}
                  className="flex items-center gap-2 rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground transition-colors hover:bg-primary/90 disabled:opacity-50"
                >
                  <Plus className="h-4 w-4" />
                  {creating ? "Creating..." : "Create"}
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Clone Modal */}
      {modal === "clone" && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
          <div className="w-full max-w-md rounded-lg border border-border bg-card p-6 shadow-xl">
            <div className="mb-4 flex items-center justify-between">
              <h2 className="text-lg font-semibold">
                Clone Agent
                <span className="ml-2 font-mono text-sm text-muted-foreground">
                  {modalAgentId}
                </span>
              </h2>
              <button
                onClick={() => setModal(null)}
                className="rounded-md p-1 text-muted-foreground hover:text-foreground"
              >
                <X className="h-5 w-5" />
              </button>
            </div>
            <div className="space-y-4">
              <div>
                <label className="mb-1 block text-sm text-muted-foreground">
                  Target Namespace (optional)
                </label>
                <input
                  type="text"
                  value={cloneTargetNs}
                  onChange={(e) => setCloneTargetNs(e.target.value)}
                  placeholder="Auto-generated if empty"
                  className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary"
                />
              </div>
              {cloneResult && (
                <div className="rounded-md border border-green-500/30 bg-green-500/5 p-3 text-sm text-green-700 dark:text-green-400">
                  {cloneResult}
                </div>
              )}
              <div className="flex justify-end gap-2">
                <button
                  onClick={() => setModal(null)}
                  className="rounded-md border border-border px-4 py-2 text-sm transition-colors hover:bg-muted"
                >
                  Close
                </button>
                <button
                  onClick={handleClone}
                  disabled={cloning}
                  className="flex items-center gap-2 rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground transition-colors hover:bg-primary/90 disabled:opacity-50"
                >
                  <Copy className="h-4 w-4" />
                  {cloning ? "Cloning..." : "Clone"}
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Share Modal */}
      {modal === "share" && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
          <div className="w-full max-w-md rounded-lg border border-border bg-card p-6 shadow-xl">
            <div className="mb-4 flex items-center justify-between">
              <h2 className="text-lg font-semibold">
                Share Agent
                <span className="ml-2 font-mono text-sm text-muted-foreground">
                  {modalAgentId}
                </span>
              </h2>
              <button
                onClick={() => setModal(null)}
                className="rounded-md p-1 text-muted-foreground hover:text-foreground"
              >
                <X className="h-5 w-5" />
              </button>
            </div>
            <div className="space-y-4">
              <div>
                <label className="mb-1 block text-sm text-muted-foreground">
                  Target Agent ID
                </label>
                <input
                  type="text"
                  value={shareTargetId}
                  onChange={(e) => setShareTargetId(e.target.value)}
                  placeholder="agent_xyz789"
                  className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary"
                  autoFocus
                />
              </div>
              <div>
                <label className="mb-1 block text-sm text-muted-foreground">
                  Permission
                </label>
                <div className="flex gap-2">
                  {(["read", "read_write"] as const).map((perm) => (
                    <button
                      key={perm}
                      onClick={() => setSharePermission(perm)}
                      className={`rounded-md px-4 py-2 text-sm font-medium transition-colors ${
                        sharePermission === perm
                          ? "bg-accent text-accent-foreground"
                          : "border border-border text-muted-foreground hover:bg-muted"
                      }`}
                    >
                      {perm === "read" ? "Read Only" : "Read & Write"}
                    </button>
                  ))}
                </div>
              </div>
              <div className="flex justify-end gap-2">
                <button
                  onClick={() => setModal(null)}
                  className="rounded-md border border-border px-4 py-2 text-sm transition-colors hover:bg-muted"
                >
                  Cancel
                </button>
                <button
                  onClick={handleShare}
                  disabled={sharing || !shareTargetId.trim()}
                  className="flex items-center gap-2 rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground transition-colors hover:bg-primary/90 disabled:opacity-50"
                >
                  <Share2 className="h-4 w-4" />
                  {sharing ? "Sharing..." : "Share"}
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
