"use client";

import { useState, useEffect, useCallback } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import {
  Sparkles,
  Database,
  Plug,
  Search,
  GitBranch,
  ArrowRight,
  ArrowLeft,
  CheckCircle2,
  Circle,
  X,
  Loader2,
  Copy,
  Check,
  AlertCircle,
  Zap,
} from "lucide-react";
import { Card } from "@/components/card";
import { Button, Input, Textarea } from "@ucotron/ui";
import { createNamespace, listConnectorSchedules, searchMemories, getGraph } from "@/lib/api";
import type { ConnectorScheduleResponse } from "@/lib/api";

const ONBOARDING_COMPLETE_KEY = "ucotron_onboarding_complete";
const ONBOARDING_STEP_KEY = "ucotron_onboarding_step";

interface ConnectorType {
  id: string;
  name: string;
  description: string;
  icon: string;
}

const TOP_CONNECTORS: ConnectorType[] = [
  { id: "slack", name: "Slack", description: "Import messages from Slack channels", icon: "💬" },
  { id: "github", name: "GitHub", description: "Import issues and pull requests", icon: "🐙" },
  { id: "notion", name: "Notion", description: "Import pages from Notion workspaces", icon: "📝" },
  { id: "google_docs", name: "Google Docs", description: "Import Google Docs documents", icon: "📄" },
  { id: "discord", name: "Discord", description: "Import messages from Discord servers", icon: "🎮" },
];

export default function OnboardingPage() {
  const router = useRouter();
  const [currentStep, setCurrentStep] = useState(1);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [skipped, setSkipped] = useState(false);

  const [namespaceName, setNamespaceName] = useState("");
  const [namespaceCreated, setNamespaceCreated] = useState(false);
  const [creatingNamespace, setCreatingNamespace] = useState(false);

  const [selectedConnector, setSelectedConnector] = useState<string | null>(null);
  const [connectorName, setConnectorName] = useState("");
  const [connectors, setConnectors] = useState<ConnectorScheduleResponse[]>([]);

  const [queryText, setQueryText] = useState("");
  const [queryResults, setQueryResults] = useState<{ content: string; score: number }[]>([]);
  const [queryLoading, setQueryLoading] = useState(false);
  const [queryCopied, setQueryCopied] = useState(false);

  const [graphNodes, setGraphNodes] = useState<{ id: number; content: string }[]>([]);
  const [graphLoading, setGraphLoading] = useState(false);

  const checkOnboardingStatus = useCallback(() => {
    if (typeof window !== "undefined") {
      const complete = localStorage.getItem(ONBOARDING_COMPLETE_KEY);
      if (complete === "true") {
        router.push("/");
      }
    }
  }, [router]);

  useEffect(() => {
    checkOnboardingStatus();
  }, [checkOnboardingStatus]);

  useEffect(() => {
    const fetchConnectors = async () => {
      try {
        const result = await listConnectorSchedules();
        setConnectors(result);
      } catch {
        setConnectors([]);
      }
    };
    fetchConnectors();
  }, []);

  const handleSkip = () => {
    if (typeof window !== "undefined") {
      localStorage.setItem(ONBOARDING_COMPLETE_KEY, "true");
    }
    setSkipped(true);
    router.push("/");
  };

  const handleCreateNamespace = async () => {
    if (!namespaceName.trim()) return;
    setCreatingNamespace(true);
    setError("");
    try {
      await createNamespace(namespaceName.trim());
      setNamespaceCreated(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to create namespace");
    } finally {
      setCreatingNamespace(false);
    }
  };

  const handleConnectorConnect = () => {
    if (typeof window !== "undefined") {
      localStorage.setItem(ONBOARDING_STEP_KEY, "2");
    }
    router.push("/connectors");
  };

  const handleRunQuery = async () => {
    if (!queryText.trim()) return;
    setQueryLoading(true);
    setError("");
    try {
      const result = await searchMemories({ query: queryText.trim(), limit: 5 });
      setQueryResults(
        result.results.map((r) => ({
          content: r.content,
          score: r.score,
        }))
      );
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to search");
    } finally {
      setQueryLoading(false);
    }
  };

  const handleExploreGraph = async () => {
    setGraphLoading(true);
    try {
      const result = await getGraph({ limit: 20 });
      setGraphNodes(
        result.nodes.map((n) => ({
          id: n.id,
          content: n.content.substring(0, 50) + (n.content.length > 50 ? "..." : ""),
        }))
      );
    } catch {
      setGraphNodes([]);
    } finally {
      setGraphLoading(false);
    }
  };

  const handleComplete = () => {
    if (typeof window !== "undefined") {
      localStorage.setItem(ONBOARDING_COMPLETE_KEY, "true");
    }
    router.push("/");
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
    setQueryCopied(true);
    setTimeout(() => setQueryCopied(false), 2000);
  };

  const steps = [
    { number: 1, title: "Welcome", icon: Sparkles },
    { number: 2, title: "Connect Data", icon: Database },
    { number: 3, title: "Search", icon: Search },
    { number: 4, title: "Explore", icon: GitBranch },
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-background via-background to-primary/5 p-6">
      <div className="mx-auto max-w-3xl">
        {/* Header */}
        <div className="mb-8 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-primary/10">
              <Zap className="h-6 w-6 text-primary" />
            </div>
            <div>
              <h1 className="text-2xl font-bold">Welcome to Ucotron</h1>
              <p className="text-sm text-muted-foreground">Let&apos;s get you set up</p>
            </div>
          </div>
          <button
            onClick={handleSkip}
            className="flex items-center gap-2 rounded-md border border-border px-3 py-1.5 text-sm text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
          >
            <X className="h-4 w-4" />
            Skip
          </button>
        </div>

        {/* Progress Indicator */}
        <div className="mb-8 flex items-center justify-center gap-2">
          {steps.map((step, index) => (
            <div key={step.number} className="flex items-center">
              {index > 0 && (
                <div
                  className={`h-0.5 w-12 ${
                    currentStep > step.number - 1
                      ? "bg-primary"
                      : "bg-border"
                  }`}
                />
              )}
              <button
                onClick={() => step.number < currentStep && setCurrentStep(step.number)}
                disabled={step.number > currentStep}
                className={`flex flex-col items-center gap-1 rounded-lg p-2 transition-colors ${
                  step.number === currentStep
                    ? "bg-primary/10"
                    : step.number < currentStep
                    ? "hover:bg-muted cursor-pointer"
                    : "cursor-not-allowed opacity-50"
                }`}
              >
                {step.number < currentStep ? (
                  <CheckCircle2 className="h-6 w-6 text-primary" />
                ) : step.number === currentStep ? (
                  <div className="flex h-6 w-6 items-center justify-center rounded-full bg-primary text-primary-foreground">
                    <span className="text-xs font-medium">{step.number}</span>
                  </div>
                ) : (
                  <Circle className="h-6 w-6 text-muted-foreground" />
                )}
                <span
                  className={`text-xs ${
                    step.number === currentStep
                      ? "font-medium text-primary"
                      : "text-muted-foreground"
                  }`}
                >
                  {step.title}
                </span>
              </button>
            </div>
          ))}
        </div>

        {/* Error Banner */}
        {error && (
          <div className="mb-4 flex items-center gap-2 rounded-lg border border-destructive/30 bg-destructive/5 p-3 text-sm text-destructive">
            <AlertCircle className="h-4 w-4" />
            {error}
            <button onClick={() => setError("")} className="ml-auto rounded p-0.5 hover:bg-destructive/10">
              <X className="h-3.5 w-3.5" />
            </button>
          </div>
        )}

        {/* Step Content */}
        <Card className="p-6">
          {/* Step 1: Welcome + Namespace Creation */}
          {currentStep === 1 && (
            <div className="space-y-6">
              <div className="text-center">
                <div className="mx-auto mb-4 flex h-16 w-16 items-center justify-center rounded-full bg-primary/10">
                  <Sparkles className="h-8 w-8 text-primary" />
                </div>
                <h2 className="text-xl font-semibold">Create Your First Namespace</h2>
                <p className="mt-2 text-sm text-muted-foreground">
                  Namespaces help you organize different knowledge bases. Create one to get started.
                </p>
              </div>

              <div className="mx-auto max-w-md space-y-4">
                <div>
                  <label className="mb-1 block text-sm text-muted-foreground">
                    Namespace Name
                  </label>
                  <Input
                    type="text"
                    value={namespaceName}
                                        onChange={(e: React.ChangeEvent<HTMLInputElement>) => setNamespaceName(e.target.value)}
                    placeholder="e.g., work, personal, research"
                    disabled={namespaceCreated}
                  />
                </div>

                {namespaceCreated ? (
                  <div className="flex items-center gap-2 rounded-lg border border-green-500/30 bg-green-500/5 p-3">
                    <CheckCircle2 className="h-5 w-5 text-green-500" />
                    <span className="text-sm text-green-700 dark:text-green-400">
                      Namespace &quot;{namespaceName}&quot; created successfully!
                    </span>
                  </div>
                ) : (
                  <Button
                    onClick={handleCreateNamespace}
                    disabled={!namespaceName.trim() || creatingNamespace}
                    className="w-full"
                  >
                    {creatingNamespace ? (
                      <>
                        <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                        Creating...
                      </>
                    ) : (
                      <>
                        <Database className="mr-2 h-4 w-4" />
                        Create Namespace
                      </>
                    )}
                  </Button>
                )}
              </div>

              <div className="flex justify-end pt-4">
                <Button
                  onClick={() => setCurrentStep(2)}
                  disabled={!namespaceCreated && !namespaceName.trim()}
                  className="flex items-center gap-2"
                >
                  Next
                  <ArrowRight className="h-4 w-4" />
                </Button>
              </div>
            </div>
          )}

          {/* Step 2: Connect First Connector */}
          {currentStep === 2 && (
            <div className="space-y-6">
              <div className="text-center">
                <div className="mx-auto mb-4 flex h-16 w-16 items-center justify-center rounded-full bg-primary/10">
                  <Plug className="h-8 w-8 text-primary" />
                </div>
                <h2 className="text-xl font-semibold">Connect Your Data Sources</h2>
                <p className="mt-2 text-sm text-muted-foreground">
                  Import data from your favorite tools to build your knowledge graph.
                </p>
              </div>

              {connectors.length > 0 ? (
                <div className="rounded-lg border border-green-500/30 bg-green-500/5 p-4">
                  <div className="flex items-center gap-2">
                    <CheckCircle2 className="h-5 w-5 text-green-500" />
                    <span className="text-sm text-green-700 dark:text-green-400">
                      You have {connectors.length} connector{connectors.length > 1 ? "s" : ""} configured!
                    </span>
                  </div>
                </div>
              ) : (
                <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
                  {TOP_CONNECTORS.map((connector) => (
                    <button
                      key={connector.id}
                      onClick={() => {
                        setSelectedConnector(connector.id);
                        setConnectorName(`My ${connector.name} Connector`);
                      }}
                      className={`flex items-center gap-3 rounded-lg border p-4 text-left transition-all hover:border-primary/50 hover:bg-accent/50 ${
                        selectedConnector === connector.id
                          ? "border-primary bg-primary/5"
                          : "border-border"
                      }`}
                    >
                      <span className="text-2xl">{connector.icon}</span>
                      <div>
                        <p className="font-medium">{connector.name}</p>
                        <p className="text-xs text-muted-foreground">
                          {connector.description}
                        </p>
                      </div>
                    </button>
                  ))}
                </div>
              )}

              <div className="flex justify-between pt-4">
                <Button
                  onClick={() => setCurrentStep(1)}
                  variant="outline"
                  className="flex items-center gap-2"
                >
                  <ArrowLeft className="h-4 w-4" />
                  Back
                </Button>
                <Button
                  onClick={handleConnectorConnect}
                  className="flex items-center gap-2"
                >
                  Go to Connectors
                  <ArrowRight className="h-4 w-4" />
                </Button>
              </div>
            </div>
          )}

          {/* Step 3: Run First Query */}
          {currentStep === 3 && (
            <div className="space-y-6">
              <div className="text-center">
                <div className="mx-auto mb-4 flex h-16 w-16 items-center justify-center rounded-full bg-primary/10">
                  <Search className="h-8 w-8 text-primary" />
                </div>
                <h2 className="text-xl font-semibold">Try Your First Search</h2>
                <p className="mt-2 text-sm text-muted-foreground">
                  Query your knowledge graph to find relevant information.
                </p>
              </div>

              <div className="space-y-4">
                    <div className="flex gap-2">
                  <Input
                    type="text"
                    value={queryText}
                    onChange={(e: React.ChangeEvent<HTMLInputElement>) => setQueryText(e.target.value)}
                    placeholder="Ask a question about your data..."
                    onKeyDown={(e: React.KeyboardEvent<HTMLInputElement>) => e.key === "Enter" && handleRunQuery()}
                    className="flex-1"
                  />
                  <Button
                    onClick={handleRunQuery}
                    disabled={!queryText.trim() || queryLoading}
                  >
                    {queryLoading ? (
                      <Loader2 className="h-4 w-4 animate-spin" />
                    ) : (
                      "Search"
                    )}
                  </Button>
                </div>

                {queryResults.length > 0 && (
                  <div className="space-y-2">
                    <p className="text-sm font-medium">
                      Found {queryResults.length} result{queryResults.length > 1 ? "s" : ""}
                    </p>
                    {queryResults.map((result, index) => (
                      <div
                        key={index}
                        className="flex items-start gap-3 rounded-lg border border-border bg-muted/30 p-3"
                      >
                        <div className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-primary/10 text-xs font-medium text-primary">
                          {index + 1}
                        </div>
                        <div className="flex-1 min-w-0">
                          <p className="text-sm">{result.content}</p>
                          <div className="mt-1 flex items-center gap-2">
                            <span className="text-xs text-muted-foreground">
                              Score: {result.score.toFixed(2)}
                            </span>
                            <button
                              onClick={() => copyToClipboard(result.content)}
                              className="flex items-center gap-1 text-xs text-muted-foreground hover:text-primary"
                            >
                              {queryCopied ? (
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
                        </div>
                      </div>
                    ))}
                  </div>
                )}

                {queryResults.length === 0 && !queryLoading && queryText && (
                  <div className="rounded-lg border border-amber-500/30 bg-amber-500/5 p-4 text-center">
                    <p className="text-sm text-amber-700 dark:text-amber-400">
                      No results found. Try a different query or add more data first.
                    </p>
                  </div>
                )}
              </div>

              <div className="flex justify-between pt-4">
                <Button
                  onClick={() => setCurrentStep(2)}
                  variant="outline"
                  className="flex items-center gap-2"
                >
                  <ArrowLeft className="h-4 w-4" />
                  Back
                </Button>
                <Button
                  onClick={() => setCurrentStep(4)}
                  className="flex items-center gap-2"
                >
                  Next
                  <ArrowRight className="h-4 w-4" />
                </Button>
              </div>
            </div>
          )}

          {/* Step 4: Explore Graph */}
          {currentStep === 4 && (
            <div className="space-y-6">
              <div className="text-center">
                <div className="mx-auto mb-4 flex h-16 w-16 items-center justify-center rounded-full bg-primary/10">
                  <GitBranch className="h-8 w-8 text-primary" />
                </div>
                <h2 className="text-xl font-semibold">Explore Your Knowledge Graph</h2>
                <p className="mt-2 text-sm text-muted-foreground">
                  Visualize how your knowledge is connected.
                </p>
              </div>

              <div className="flex justify-center">
                <Button
                  onClick={handleExploreGraph}
                  disabled={graphLoading}
                  variant="outline"
                  className="flex items-center gap-2"
                >
                  {graphLoading ? (
                    <Loader2 className="h-4 w-4 animate-spin" />
                  ) : (
                    <GitBranch className="h-4 w-4" />
                  )}
                  {graphLoading ? "Loading..." : "Preview Graph"}
                </Button>
              </div>

              {graphNodes.length > 0 && (
                <div className="rounded-lg border border-border bg-muted/30 p-4">
                  <p className="mb-3 text-sm font-medium">
                    Found {graphNodes.length} nodes in your graph
                  </p>
                  <div className="flex flex-wrap gap-2">
                    {graphNodes.slice(0, 10).map((node) => (
                      <span
                        key={node.id}
                        className="rounded-full bg-primary/10 px-3 py-1 text-xs text-primary"
                        title={node.content}
                      >
                        {node.content.substring(0, 20)}...
                      </span>
                    ))}
                    {graphNodes.length > 10 && (
                      <span className="rounded-full bg-muted px-3 py-1 text-xs text-muted-foreground">
                        +{graphNodes.length - 10} more
                      </span>
                    )}
                  </div>
                </div>
              )}

              {graphNodes.length === 0 && !graphLoading && (
                <div className="rounded-lg border border-amber-500/30 bg-amber-500/5 p-4 text-center">
                  <p className="text-sm text-amber-700 dark:text-amber-400">
                    Your graph is empty. Add data to see connections!
                  </p>
                </div>
              )}

              <div className="flex justify-between pt-4">
                <Button
                  onClick={() => setCurrentStep(3)}
                  variant="outline"
                  className="flex items-center gap-2"
                >
                  <ArrowLeft className="h-4 w-4" />
                  Back
                </Button>
                <Button
                  onClick={handleComplete}
                  className="flex items-center gap-2"
                >
                  <CheckCircle2 className="h-4 w-4" />
                  Complete Setup
                </Button>
              </div>
            </div>
          )}
        </Card>
      </div>
    </div>
  );
}
