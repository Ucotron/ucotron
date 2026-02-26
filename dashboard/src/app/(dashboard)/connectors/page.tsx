"use client";

import { useEffect, useState, useCallback } from "react";
import {
  Plug,
  Plus,
  Play,
  RefreshCw,
  ChevronDown,
  ChevronUp,
  X,
  Clock,
  AlertCircle,
  CheckCircle2,
  History,
  Trash2,
} from "lucide-react";
import { Card } from "@/components/card";
import {
  listConnectorSchedules,
  getConnectorHistory,
  triggerConnectorSync,
  deleteConnector,
} from "@/lib/api";
import type {
  ConnectorScheduleResponse,
  ConnectorSyncRecordResponse,
} from "@/lib/api";

// Available connector types for the wizard
const CONNECTOR_TYPES = [
  { id: "slack", name: "Slack", description: "Import messages from Slack channels", auth: "bot_token" },
  { id: "github", name: "GitHub", description: "Import issues and pull requests", auth: "token" },
  { id: "gitlab", name: "GitLab", description: "Import issues from GitLab projects", auth: "token" },
  { id: "notion", name: "Notion", description: "Import pages from Notion workspaces", auth: "token" },
  { id: "google_docs", name: "Google Docs", description: "Import Google Docs documents", auth: "oauth2" },
  { id: "gdrive", name: "Google Drive", description: "Import files from Google Drive", auth: "oauth2" },
  { id: "discord", name: "Discord", description: "Import messages from Discord channels", auth: "bot_token" },
  { id: "telegram", name: "Telegram", description: "Import messages from Telegram chats", auth: "bot_token" },
  { id: "mongodb", name: "MongoDB", description: "Import documents from MongoDB collections", auth: "connection_string" },
  { id: "postgres", name: "PostgreSQL", description: "Import data from PostgreSQL tables", auth: "connection_string" },
  { id: "bitbucket", name: "Bitbucket", description: "Import from Bitbucket repositories", auth: "token" },
  { id: "spotify", name: "Spotify", description: "Import podcast and music metadata", auth: "oauth2" },
  { id: "obsidian", name: "Obsidian", description: "Import notes from an Obsidian vault", auth: "none" },
] as const;

function formatTimestamp(ts: number): string {
  if (ts === 0) return "Never";
  return new Date(ts * 1000).toLocaleString();
}

function formatDuration(startSecs: number, endSecs: number | null): string {
  if (!endSecs) return "In progress...";
  const dur = endSecs - startSecs;
  if (dur < 60) return `${dur}s`;
  return `${Math.floor(dur / 60)}m ${dur % 60}s`;
}

type WizardStep = "select_type" | "configure" | "review";

export default function ConnectorsPage() {
  const [connectors, setConnectors] = useState<ConnectorScheduleResponse[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const [history, setHistory] = useState<ConnectorSyncRecordResponse[]>([]);
  const [historyLoading, setHistoryLoading] = useState(false);

  // Sync trigger state
  const [syncing, setSyncing] = useState<string | null>(null);
  const [syncMessage, setSyncMessage] = useState("");

  // Delete state
  const [deleting, setDeleting] = useState<string | null>(null);
  const [deleteConfirm, setDeleteConfirm] = useState<string | null>(null);

  // Wizard state
  const [wizardOpen, setWizardOpen] = useState(false);
  const [wizardStep, setWizardStep] = useState<WizardStep>("select_type");
  const [selectedType, setSelectedType] = useState("");
  const [wizardConfig, setWizardConfig] = useState({
    name: "",
    namespace: "default",
    authValue: "",
    cronExpression: "0 */6 * * *",
    enabled: true,
  });

  const [schedulingDisabled, setSchedulingDisabled] = useState(false);

  const fetchConnectors = useCallback(async () => {
    setLoading(true);
    setError("");
    setSchedulingDisabled(false);
    try {
      const result = await listConnectorSchedules();
      setConnectors(result);
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Failed to load connectors";
      if (msg.includes("scheduling is not enabled")) {
        setSchedulingDisabled(true);
      } else {
        setError(msg);
      }
    }
    setLoading(false);
  }, []);

  useEffect(() => {
    fetchConnectors();
  }, [fetchConnectors]);

  const fetchHistory = async (connectorId: string) => {
    setHistoryLoading(true);
    try {
      const result = await getConnectorHistory(connectorId);
      setHistory(result.records);
    } catch {
      setHistory([]);
    }
    setHistoryLoading(false);
  };

  const handleExpand = (connectorId: string) => {
    if (expandedId === connectorId) {
      setExpandedId(null);
      setHistory([]);
    } else {
      setExpandedId(connectorId);
      fetchHistory(connectorId);
    }
  };

  const handleSync = async (connectorId: string) => {
    setSyncing(connectorId);
    setSyncMessage("");
    setError("");
    try {
      const result = await triggerConnectorSync(connectorId);
      setSyncMessage(result.message);
      if (expandedId === connectorId) {
        await fetchHistory(connectorId);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to trigger sync");
    }
    setSyncing(null);
  };

  const handleDelete = async (connectorId: string) => {
    setDeleting(connectorId);
    setError("");
    try {
      await deleteConnector(connectorId);
      setDeleteConfirm(null);
      fetchConnectors();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to delete connector");
    }
    setDeleting(null);
  };

  const openWizard = () => {
    setWizardOpen(true);
    setWizardStep("select_type");
    setSelectedType("");
    setWizardConfig({
      name: "",
      namespace: "default",
      authValue: "",
      cronExpression: "0 */6 * * *",
      enabled: true,
    });
  };

  const closeWizard = () => {
    setWizardOpen(false);
  };

  const selectConnectorType = (typeId: string) => {
    setSelectedType(typeId);
    const ct = CONNECTOR_TYPES.find((c) => c.id === typeId);
    setWizardConfig((prev) => ({
      ...prev,
      name: ct ? `My ${ct.name} Connector` : "",
    }));
    setWizardStep("configure");
  };

  const getAuthLabel = (): string => {
    const ct = CONNECTOR_TYPES.find((c) => c.id === selectedType);
    if (!ct) return "Credential";
    switch (ct.auth) {
      case "bot_token": return "Bot Token";
      case "token": return "Access Token";
      case "oauth2": return "OAuth2 Client Secret";
      case "connection_string": return "Connection String";
      case "none": return "No authentication needed";
    }
  };

  const getAuthPlaceholder = (): string => {
    const ct = CONNECTOR_TYPES.find((c) => c.id === selectedType);
    if (!ct) return "";
    switch (ct.auth) {
      case "bot_token": return "xoxb-...";
      case "token": return "ghp_... or glpat-...";
      case "oauth2": return "client-secret-value";
      case "connection_string": return "postgresql://user:pass@host:5432/db";
      case "none": return "";
    }
  };

  const selectedTypeInfo = CONNECTOR_TYPES.find((c) => c.id === selectedType);
  const needsAuth = selectedTypeInfo?.auth !== "none";

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <Plug className="h-6 w-6 text-primary" />
          <h1 className="text-2xl font-bold">Connectors</h1>
          <span className="rounded bg-muted px-2 py-0.5 text-xs text-muted-foreground">
            {connectors.length}
          </span>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={fetchConnectors}
            className="flex items-center gap-2 rounded-md border border-border px-3 py-1.5 text-sm text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
          >
            <RefreshCw className="h-3.5 w-3.5" />
            Refresh
          </button>
          <button
            onClick={openWizard}
            disabled={schedulingDisabled}
            className="flex items-center gap-2 rounded-md bg-primary px-4 py-1.5 text-sm font-medium text-primary-foreground transition-colors hover:bg-primary/90 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <Plus className="h-4 w-4" />
            Add Connector
          </button>
        </div>
      </div>

      {/* Sync success message */}
      {syncMessage && (
        <div className="flex items-center gap-2 rounded-lg border border-green-500/30 bg-green-500/5 p-3 text-sm text-green-700 dark:text-green-400">
          <CheckCircle2 className="h-4 w-4 flex-shrink-0" />
          {syncMessage}
          <button
            onClick={() => setSyncMessage("")}
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
      {loading && connectors.length === 0 && (
        <div className="flex items-center gap-2 text-muted-foreground">
          <RefreshCw className="h-4 w-4 animate-spin" />
          Loading connectors...
        </div>
      )}

      {/* Scheduling not enabled */}
      {schedulingDisabled && (
        <Card>
          <div className="flex flex-col items-center gap-3 py-8 text-center">
            <Plug className="h-12 w-12 text-muted-foreground/30" />
            <p className="text-sm font-medium text-muted-foreground">
              Connector Scheduling Not Available
            </p>
            <p className="max-w-md text-xs text-muted-foreground/70">
              The server does not have connector scheduling enabled. Enable it in your server configuration to use connectors.
            </p>
          </div>
        </Card>
      )}

      {/* Empty state */}
      {!loading && !schedulingDisabled && connectors.length === 0 && (
        <Card>
          <div className="flex flex-col items-center gap-3 py-8 text-center">
            <Plug className="h-12 w-12 text-muted-foreground/30" />
            <p className="text-sm text-muted-foreground">
              No connectors configured. Add a connector to import data from external sources.
            </p>
            <button
              onClick={openWizard}
              className="flex items-center gap-2 rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground transition-colors hover:bg-primary/90"
            >
              <Plus className="h-4 w-4" />
              Add Connector
            </button>
          </div>
        </Card>
      )}

      {/* Connector list */}
      {connectors.length > 0 && (
        <Card>
          <div className="divide-y divide-border">
            {connectors.map((connector) => (
              <div key={connector.connector_id} className="py-3 first:pt-0 last:pb-0">
                {/* Connector row */}
                <div className="flex items-center justify-between">
                  <button
                    onClick={() => handleExpand(connector.connector_id)}
                    className="flex items-center gap-2 text-sm font-medium hover:text-primary"
                  >
                    {expandedId === connector.connector_id ? (
                      <ChevronUp className="h-3.5 w-3.5" />
                    ) : (
                      <ChevronDown className="h-3.5 w-3.5" />
                    )}
                    <Plug className="h-3.5 w-3.5 text-muted-foreground" />
                    <span>{connector.connector_id}</span>
                    <span
                      className={`rounded px-1.5 py-0.5 text-[10px] font-medium ${
                        connector.enabled
                          ? "bg-green-500/10 text-green-600 dark:text-green-400"
                          : "bg-muted text-muted-foreground"
                      }`}
                    >
                      {connector.enabled ? "Active" : "Disabled"}
                    </span>
                  </button>
                  <div className="flex items-center gap-2">
                    {connector.cron_expression && (
                      <span className="flex items-center gap-1 text-xs text-muted-foreground" title="Schedule">
                        <Clock className="h-3 w-3" />
                        {connector.cron_expression}
                      </span>
                    )}
                    <button
                      onClick={() => handleSync(connector.connector_id)}
                      disabled={syncing === connector.connector_id}
                      className="flex items-center gap-1.5 rounded-md border border-border px-2.5 py-1 text-xs text-muted-foreground transition-colors hover:bg-accent hover:text-accent-foreground disabled:opacity-50"
                      title="Trigger sync"
                    >
                      {syncing === connector.connector_id ? (
                        <RefreshCw className="h-3 w-3 animate-spin" />
                      ) : (
                        <Play className="h-3 w-3" />
                      )}
                      Sync
                    </button>
                    {deleteConfirm === connector.connector_id ? (
                      <div className="flex items-center gap-1">
                        <button
                          onClick={() => handleDelete(connector.connector_id)}
                          disabled={deleting === connector.connector_id}
                          className="flex items-center gap-1 rounded-md bg-destructive px-2.5 py-1 text-xs text-destructive-foreground transition-colors hover:bg-destructive/90 disabled:opacity-50"
                        >
                          {deleting === connector.connector_id ? (
                            <RefreshCw className="h-3 w-3 animate-spin" />
                          ) : (
                            <CheckCircle2 className="h-3 w-3" />
                          )}
                          Confirm
                        </button>
                        <button
                          onClick={() => setDeleteConfirm(null)}
                          className="rounded-md border border-border px-2.5 py-1 text-xs text-muted-foreground transition-colors hover:bg-muted"
                        >
                          Cancel
                        </button>
                      </div>
                    ) : (
                      <button
                        onClick={() => setDeleteConfirm(connector.connector_id)}
                        className="flex items-center gap-1.5 rounded-md border border-border px-2.5 py-1 text-xs text-muted-foreground transition-colors hover:bg-accent hover:text-destructive"
                        title="Delete connector"
                      >
                        <Trash2 className="h-3 w-3" />
                        Delete
                      </button>
                    )}
                  </div>
                </div>

                {/* Expanded details */}
                {expandedId === connector.connector_id && (
                  <div className="mt-3 space-y-4 pl-8">
                    {/* Schedule details */}
                    <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
                      <div>
                        <p className="text-xs text-muted-foreground">Schedule</p>
                        <p className="font-mono text-sm">
                          {connector.cron_expression || "Manual only"}
                        </p>
                      </div>
                      <div>
                        <p className="text-xs text-muted-foreground">Timeout</p>
                        <p className="text-sm">{connector.timeout_secs}s</p>
                      </div>
                      <div>
                        <p className="text-xs text-muted-foreground">Max Retries</p>
                        <p className="text-sm font-semibold">{connector.max_retries}</p>
                      </div>
                      <div>
                        <p className="text-xs text-muted-foreground">Next Run</p>
                        <p className="text-sm">
                          {connector.next_fire_time || "Not scheduled"}
                        </p>
                      </div>
                    </div>

                    {/* Sync history */}
                    <div>
                      <div className="mb-2 flex items-center gap-2">
                        <History className="h-3.5 w-3.5 text-muted-foreground" />
                        <p className="text-xs text-muted-foreground">
                          Sync History
                          {historyLoading && " (loading...)"}
                        </p>
                      </div>
                      {history.length === 0 && !historyLoading ? (
                        <p className="text-xs text-muted-foreground/60">
                          No sync records yet.
                        </p>
                      ) : (
                        <div className="space-y-1">
                          {history.slice(0, 10).map((record, i) => (
                            <div
                              key={`${record.started_at}-${i}`}
                              className="flex items-center justify-between rounded-md bg-muted/30 px-3 py-1.5"
                            >
                              <div className="flex items-center gap-2 text-xs">
                                {record.error ? (
                                  <AlertCircle className="h-3 w-3 text-destructive" />
                                ) : record.finished_at ? (
                                  <CheckCircle2 className="h-3 w-3 text-green-500" />
                                ) : (
                                  <RefreshCw className="h-3 w-3 animate-spin text-amber-500" />
                                )}
                                <span>{formatTimestamp(record.started_at)}</span>
                                <span className="text-muted-foreground">
                                  {formatDuration(record.started_at, record.finished_at)}
                                </span>
                              </div>
                              <div className="flex items-center gap-3 text-xs">
                                <span className="text-muted-foreground">
                                  {record.items_fetched} fetched
                                </span>
                                {record.items_skipped > 0 && (
                                  <span className="text-amber-500">
                                    {record.items_skipped} skipped
                                  </span>
                                )}
                                {record.error && (
                                  <span
                                    className="max-w-[200px] truncate text-destructive"
                                    title={record.error}
                                  >
                                    {record.error}
                                  </span>
                                )}
                              </div>
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

      {/* Wizard Modal */}
      {wizardOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
          <div className="w-full max-w-lg rounded-lg border border-border bg-card p-6 shadow-xl">
            <div className="mb-4 flex items-center justify-between">
              <h2 className="text-lg font-semibold">
                {wizardStep === "select_type" && "Select Connector Type"}
                {wizardStep === "configure" && `Configure ${selectedTypeInfo?.name}`}
                {wizardStep === "review" && "Review Configuration"}
              </h2>
              <button
                onClick={closeWizard}
                className="rounded-md p-1 text-muted-foreground hover:text-foreground"
              >
                <X className="h-5 w-5" />
              </button>
            </div>

            {/* Step indicator */}
            <div className="mb-6 flex items-center gap-2">
              {(["select_type", "configure", "review"] as WizardStep[]).map(
                (step, i) => (
                  <div key={step} className="flex items-center gap-2">
                    {i > 0 && (
                      <div className="h-px w-6 bg-border" />
                    )}
                    <div
                      className={`flex h-6 w-6 items-center justify-center rounded-full text-xs font-medium ${
                        wizardStep === step
                          ? "bg-primary text-primary-foreground"
                          : ["select_type", "configure", "review"].indexOf(wizardStep) >
                            i
                          ? "bg-green-500/20 text-green-600 dark:text-green-400"
                          : "bg-muted text-muted-foreground"
                      }`}
                    >
                      {i + 1}
                    </div>
                  </div>
                )
              )}
              <span className="ml-2 text-xs text-muted-foreground">
                Step{" "}
                {["select_type", "configure", "review"].indexOf(wizardStep) + 1}{" "}
                of 3
              </span>
            </div>

            {/* Step 1: Select type */}
            {wizardStep === "select_type" && (
              <div className="max-h-[400px] space-y-2 overflow-y-auto">
                {CONNECTOR_TYPES.map((ct) => (
                  <button
                    key={ct.id}
                    onClick={() => selectConnectorType(ct.id)}
                    className="flex w-full items-center gap-3 rounded-md border border-border p-3 text-left transition-colors hover:border-primary/50 hover:bg-accent/50"
                  >
                    <Plug className="h-5 w-5 flex-shrink-0 text-primary" />
                    <div>
                      <p className="text-sm font-medium">{ct.name}</p>
                      <p className="text-xs text-muted-foreground">
                        {ct.description}
                      </p>
                    </div>
                  </button>
                ))}
              </div>
            )}

            {/* Step 2: Configure */}
            {wizardStep === "configure" && (
              <div className="space-y-4">
                <div>
                  <label className="mb-1 block text-sm text-muted-foreground">
                    Connector Name
                  </label>
                  <input
                    type="text"
                    value={wizardConfig.name}
                    onChange={(e) =>
                      setWizardConfig((prev) => ({ ...prev, name: e.target.value }))
                    }
                    placeholder="My Slack Connector"
                    className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary"
                    autoFocus
                  />
                </div>
                <div>
                  <label className="mb-1 block text-sm text-muted-foreground">
                    Target Namespace
                  </label>
                  <input
                    type="text"
                    value={wizardConfig.namespace}
                    onChange={(e) =>
                      setWizardConfig((prev) => ({
                        ...prev,
                        namespace: e.target.value,
                      }))
                    }
                    placeholder="default"
                    className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary"
                  />
                </div>
                {needsAuth && (
                  <div>
                    <label className="mb-1 block text-sm text-muted-foreground">
                      {getAuthLabel()}
                    </label>
                    <input
                      type="password"
                      value={wizardConfig.authValue}
                      onChange={(e) =>
                        setWizardConfig((prev) => ({
                          ...prev,
                          authValue: e.target.value,
                        }))
                      }
                      placeholder={getAuthPlaceholder()}
                      className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm font-mono focus:outline-none focus:ring-2 focus:ring-primary"
                    />
                  </div>
                )}
                <div>
                  <label className="mb-1 block text-sm text-muted-foreground">
                    Sync Schedule (cron)
                  </label>
                  <input
                    type="text"
                    value={wizardConfig.cronExpression}
                    onChange={(e) =>
                      setWizardConfig((prev) => ({
                        ...prev,
                        cronExpression: e.target.value,
                      }))
                    }
                    placeholder="0 */6 * * *"
                    className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm font-mono focus:outline-none focus:ring-2 focus:ring-primary"
                  />
                  <p className="mt-1 text-[11px] text-muted-foreground">
                    Default: every 6 hours. Leave empty for manual sync only.
                  </p>
                </div>
                <div className="flex items-center gap-2">
                  <input
                    type="checkbox"
                    id="wizard-enabled"
                    checked={wizardConfig.enabled}
                    onChange={(e) =>
                      setWizardConfig((prev) => ({
                        ...prev,
                        enabled: e.target.checked,
                      }))
                    }
                    className="h-4 w-4 rounded border-border"
                  />
                  <label
                    htmlFor="wizard-enabled"
                    className="text-sm text-muted-foreground"
                  >
                    Enable immediately after creation
                  </label>
                </div>
                <div className="flex justify-between pt-2">
                  <button
                    onClick={() => setWizardStep("select_type")}
                    className="rounded-md border border-border px-4 py-2 text-sm transition-colors hover:bg-muted"
                  >
                    Back
                  </button>
                  <button
                    onClick={() => setWizardStep("review")}
                    disabled={!wizardConfig.name.trim() || (needsAuth && !wizardConfig.authValue.trim())}
                    className="rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground transition-colors hover:bg-primary/90 disabled:opacity-50"
                  >
                    Review
                  </button>
                </div>
              </div>
            )}

            {/* Step 3: Review */}
            {wizardStep === "review" && (
              <div className="space-y-4">
                <div className="rounded-md bg-muted/50 p-4">
                  <div className="space-y-3">
                    <div className="flex items-center justify-between">
                      <span className="text-xs text-muted-foreground">Type</span>
                      <span className="text-sm font-medium">
                        {selectedTypeInfo?.name}
                      </span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-xs text-muted-foreground">Name</span>
                      <span className="text-sm">{wizardConfig.name}</span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-xs text-muted-foreground">
                        Namespace
                      </span>
                      <span className="font-mono text-sm">
                        {wizardConfig.namespace || "default"}
                      </span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-xs text-muted-foreground">Auth</span>
                      <span className="text-sm">
                        {needsAuth ? getAuthLabel() + " provided" : "None required"}
                      </span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-xs text-muted-foreground">
                        Schedule
                      </span>
                      <span className="font-mono text-sm">
                        {wizardConfig.cronExpression || "Manual only"}
                      </span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-xs text-muted-foreground">
                        Enabled
                      </span>
                      <span
                        className={`rounded px-1.5 py-0.5 text-xs font-medium ${
                          wizardConfig.enabled
                            ? "bg-green-500/10 text-green-600 dark:text-green-400"
                            : "bg-muted text-muted-foreground"
                        }`}
                      >
                        {wizardConfig.enabled ? "Yes" : "No"}
                      </span>
                    </div>
                  </div>
                </div>
                <p className="text-xs text-muted-foreground">
                  The connector will be added to your server configuration. After
                  creation, you can trigger an initial sync from the connectors
                  list.
                </p>
                <div className="flex justify-between pt-2">
                  <button
                    onClick={() => setWizardStep("configure")}
                    className="rounded-md border border-border px-4 py-2 text-sm transition-colors hover:bg-muted"
                  >
                    Back
                  </button>
                  <button
                    onClick={() => {
                      // In a full implementation, this would POST to the API
                      // to create the connector configuration on the server.
                      // For now, we close the wizard and refresh the list.
                      closeWizard();
                      fetchConnectors();
                    }}
                    className="flex items-center gap-2 rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground transition-colors hover:bg-primary/90"
                  >
                    <Plus className="h-4 w-4" />
                    Create Connector
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
