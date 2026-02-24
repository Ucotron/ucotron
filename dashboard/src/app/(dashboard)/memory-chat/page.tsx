"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import {
  Send,
  Settings2,
  Loader2,
  Brain,
  Sparkles,
  X,
} from "lucide-react";
import { Card } from "@/components/card";
import { useNamespace } from "@/components/namespace-context";
import { augmentQuery, learnText } from "@/lib/api";
import type { SearchResultItem, EntityResponse } from "@/lib/api";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface LLMConfig {
  apiUrl: string;
  apiKey: string;
  model: string;
}

interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  memoriesUsed?: SearchResultItem[];
  entitiesUsed?: EntityResponse[];
  memoriesExpanded?: boolean;
}

const DEFAULT_LLM_CONFIG: LLMConfig = {
  apiUrl: "http://localhost:3002/api/chat",
  apiKey: "",
  model: "gpt-4o-mini",
};

const LS_KEY = "ucotron_llm_config";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function loadConfig(): LLMConfig {
  if (typeof window === "undefined") return DEFAULT_LLM_CONFIG;
  try {
    const raw = localStorage.getItem(LS_KEY);
    if (raw) return { ...DEFAULT_LLM_CONFIG, ...JSON.parse(raw) };
  } catch {
    // ignore
  }
  return DEFAULT_LLM_CONFIG;
}

function saveConfig(cfg: LLMConfig) {
  try {
    localStorage.setItem(LS_KEY, JSON.stringify(cfg));
  } catch {
    // ignore
  }
}

function uid() {
  return Math.random().toString(36).slice(2);
}

/**
 * Parse Vercel AI SDK streaming format lines (0:"text").
 * Returns the accumulated text chunk.
 */
function parseStreamChunk(line: string): string {
  const match = line.match(/^0:"((?:[^"\\]|\\.)*)"/);
  if (!match) return "";
  try {
    return JSON.parse(`"${match[1]}"`);
  } catch {
    return match[1];
  }
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export default function MemoryChatPage() {
  const { namespace } = useNamespace();

  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [llmConfig, setLlmConfig] = useState<LLMConfig>(DEFAULT_LLM_CONFIG);

  const bottomRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Load config from localStorage on mount
  useEffect(() => {
    setLlmConfig(loadConfig());
  }, []);

  // Scroll to bottom when messages change
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Auto-resize textarea
  useEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 160)}px`;
  }, [input]);

  const handleConfigChange = useCallback(
    (key: keyof LLMConfig, value: string) => {
      setLlmConfig((prev) => {
        const next = { ...prev, [key]: value };
        saveConfig(next);
        return next;
      });
    },
    []
  );

  const toggleMemories = useCallback((id: string) => {
    setMessages((prev) =>
      prev.map((m) =>
        m.id === id ? { ...m, memoriesExpanded: !m.memoriesExpanded } : m
      )
    );
  }, []);

  const sendMessage = useCallback(async () => {
    const text = input.trim();
    if (!text || sending) return;

    setInput("");
    setSending(true);

    // Add user message immediately
    const userMsg: ChatMessage = {
      id: uid(),
      role: "user",
      content: text,
    };

    // Capture current messages for history before state update
    const historySnapshot = messages
      .filter((m) => m.role === "user" || m.role === "assistant")
      .map((m) => ({ role: m.role, content: m.content }));

    setMessages((prev) => [...prev, userMsg]);

    // Placeholder assistant message for streaming
    const assistantId = uid();
    setMessages((prev) => [
      ...prev,
      {
        id: assistantId,
        role: "assistant",
        content: "",
        memoriesUsed: [],
        entitiesUsed: [],
        memoriesExpanded: false,
      },
    ]);

    let augmentedMemories: SearchResultItem[] = [];
    let augmentedEntities: EntityResponse[] = [];
    let contextText = "";

    // 1. Augment: fetch relevant memories from Ucotron
    try {
      const augResult = await augmentQuery(
        { query: text, max_memories: 5, max_hops: 2 },
        namespace
      );
      augmentedMemories = augResult.memories;
      augmentedEntities = augResult.entities;
      contextText = augResult.context_text;
    } catch (err) {
      console.warn("augmentQuery failed:", err);
    }

    // 2. Ingest user message into Ucotron (fire and forget)
    learnText({ text }, namespace).catch((err) =>
      console.warn("learnText failed:", err)
    );

    // 3. Build messages for LLM
    const systemContent = contextText
      ? `You are a helpful assistant with access to the user's memory.\n\nRelevant memories:\n${contextText}`
      : "You are a helpful assistant.";

    const llmMessages = [
      { role: "system" as const, content: systemContent },
      ...historySnapshot,
      { role: "user" as const, content: text },
    ];

    // 4. Stream LLM response
    let fullResponse = "";
    try {
      const headers: Record<string, string> = {
        "Content-Type": "application/json",
      };
      if (llmConfig.apiKey) {
        headers["Authorization"] = `Bearer ${llmConfig.apiKey}`;
      }

      const res = await fetch(llmConfig.apiUrl, {
        method: "POST",
        headers,
        body: JSON.stringify({
          messages: llmMessages,
          model: llmConfig.model,
          namespace,
          stream: true,
        }),
      });

      if (!res.ok || !res.body) {
        const errText = await res.text();
        throw new Error(`LLM error ${res.status}: ${errText}`);
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        const lines = buffer.split("\n");
        buffer = lines.pop() ?? "";

        for (const line of lines) {
          const chunk = parseStreamChunk(line);
          if (chunk) {
            fullResponse += chunk;
            setMessages((prev) =>
              prev.map((m) =>
                m.id === assistantId
                  ? {
                      ...m,
                      content: fullResponse,
                      memoriesUsed: augmentedMemories,
                      entitiesUsed: augmentedEntities,
                    }
                  : m
              )
            );
          }
        }
      }

      // Handle any remaining buffer content
      if (buffer) {
        const chunk = parseStreamChunk(buffer);
        if (chunk) {
          fullResponse += chunk;
        }
      }

      // Finalize assistant message
      setMessages((prev) =>
        prev.map((m) =>
          m.id === assistantId
            ? {
                ...m,
                content: fullResponse || "(no response)",
                memoriesUsed: augmentedMemories,
                entitiesUsed: augmentedEntities,
              }
            : m
        )
      );

      // 5. Ingest assistant response into Ucotron (fire and forget)
      if (fullResponse) {
        learnText(
          { text: fullResponse, metadata: { source: "assistant" } },
          namespace
        ).catch((err) => console.warn("learnText (assistant) failed:", err));
      }
    } catch (err) {
      const errMsg = err instanceof Error ? err.message : "Unknown error";
      setMessages((prev) =>
        prev.map((m) =>
          m.id === assistantId ? { ...m, content: `Error: ${errMsg}` } : m
        )
      );
    } finally {
      setSending(false);
    }
  }, [input, sending, messages, namespace, llmConfig]);

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
      }
    },
    [sendMessage]
  );

  // ---------------------------------------------------------------------------
  // Render
  // ---------------------------------------------------------------------------

  return (
    <div className="flex h-full flex-col gap-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Brain className="h-5 w-5 text-primary" />
          <h1 className="text-2xl font-bold">Memory Chat</h1>
        </div>
        <button
          onClick={() => setShowSettings((v) => !v)}
          className="flex items-center gap-2 rounded-md border border-border px-3 py-1.5 text-sm text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
          title="LLM Settings"
        >
          <Settings2 className="h-4 w-4" />
          {showSettings ? (
            <>
              <X className="h-3.5 w-3.5" />
              Close
            </>
          ) : (
            "LLM Settings"
          )}
        </button>
      </div>

      {/* LLM Config Panel */}
      {showSettings && (
        <Card title="LLM Configuration">
          <div className="space-y-3">
            <div className="flex items-center gap-3">
              <label className="w-24 shrink-0 text-sm text-muted-foreground">
                API URL
              </label>
              <input
                type="text"
                value={llmConfig.apiUrl}
                onChange={(e) => handleConfigChange("apiUrl", e.target.value)}
                placeholder="http://localhost:3002/api/chat"
                className="flex-1 rounded-md border border-border bg-background px-3 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-primary"
              />
            </div>
            <div className="flex items-center gap-3">
              <label className="w-24 shrink-0 text-sm text-muted-foreground">
                API Key
              </label>
              <input
                type="password"
                value={llmConfig.apiKey}
                onChange={(e) => handleConfigChange("apiKey", e.target.value)}
                placeholder="sk-... (optional)"
                className="flex-1 rounded-md border border-border bg-background px-3 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-primary"
              />
            </div>
            <div className="flex items-center gap-3">
              <label className="w-24 shrink-0 text-sm text-muted-foreground">
                Model
              </label>
              <input
                type="text"
                value={llmConfig.model}
                onChange={(e) => handleConfigChange("model", e.target.value)}
                placeholder="gpt-4o-mini"
                className="flex-1 rounded-md border border-border bg-background px-3 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-primary"
              />
            </div>
            <p className="text-xs text-muted-foreground">
              Settings are saved to localStorage. Active namespace:{" "}
              <span className="font-mono font-medium">{namespace}</span>
            </p>
          </div>
        </Card>
      )}

      {/* Chat area */}
      <div className="flex flex-1 flex-col overflow-hidden rounded-lg border border-border bg-card">
        {/* Messages */}
        <div className="flex-1 space-y-4 overflow-y-auto p-4">
          {messages.length === 0 && (
            <div className="flex h-full flex-col items-center justify-center gap-3 text-muted-foreground">
              <Sparkles className="h-10 w-10 opacity-40" />
              <p className="text-sm">
                Start a conversation. Ucotron will augment responses with
                relevant memories.
              </p>
            </div>
          )}

          {messages.map((msg) => (
            <div
              key={msg.id}
              className={`flex gap-3 ${msg.role === "user" ? "justify-end" : "justify-start"}`}
            >
              {msg.role === "assistant" && (
                <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-primary text-primary-foreground">
                  <Brain className="h-4 w-4" />
                </div>
              )}

              <div
                className={`max-w-[75%] space-y-1 ${msg.role === "user" ? "items-end" : "items-start"} flex flex-col`}
              >
                <div
                  className={`rounded-lg px-4 py-2.5 text-sm leading-relaxed ${
                    msg.role === "user"
                      ? "bg-primary text-primary-foreground"
                      : "bg-muted text-foreground"
                  }`}
                >
                  {msg.content === "" && msg.role === "assistant" ? (
                    <Loader2 className="h-4 w-4 animate-spin" />
                  ) : (
                    <span className="whitespace-pre-wrap">{msg.content}</span>
                  )}
                </div>

                {/* Memories used badge */}
                {msg.role === "assistant" &&
                  msg.memoriesUsed &&
                  msg.memoriesUsed.length > 0 && (
                    <div className="w-full">
                      <button
                        onClick={() => toggleMemories(msg.id)}
                        className="flex items-center gap-1.5 rounded-md px-2 py-1 text-xs text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
                      >
                        <Sparkles className="h-3 w-3 text-primary" />
                        {msg.memoriesUsed.length} memor
                        {msg.memoriesUsed.length === 1 ? "y" : "ies"} used
                        <span className="ml-1 text-muted-foreground/60">
                          {msg.memoriesExpanded ? "▲" : "▼"}
                        </span>
                      </button>

                      {msg.memoriesExpanded && (
                        <div className="mt-1 space-y-1.5 rounded-md border border-border bg-background p-2">
                          {msg.memoriesUsed.map((mem) => (
                            <div
                              key={mem.id}
                              className="flex items-start justify-between gap-2 text-xs"
                            >
                              <span className="line-clamp-2 text-muted-foreground">
                                {mem.content}
                              </span>
                              <span className="shrink-0 rounded bg-accent px-1.5 py-0.5 font-mono text-accent-foreground">
                                {(mem.score * 100).toFixed(0)}%
                              </span>
                            </div>
                          ))}
                          {msg.entitiesUsed && msg.entitiesUsed.length > 0 && (
                            <div className="mt-1 border-t border-border pt-1.5">
                              <p className="mb-1 text-xs font-medium text-muted-foreground">
                                Entities
                              </p>
                              <div className="flex flex-wrap gap-1">
                                {msg.entitiesUsed.map((ent) => (
                                  <span
                                    key={ent.id}
                                    className="rounded bg-primary/10 px-1.5 py-0.5 text-xs text-primary"
                                  >
                                    {ent.content}
                                  </span>
                                ))}
                              </div>
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  )}
              </div>

              {msg.role === "user" && (
                <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-muted text-muted-foreground">
                  <span className="text-xs font-medium">You</span>
                </div>
              )}
            </div>
          ))}

          <div ref={bottomRef} />
        </div>

        {/* Input area */}
        <div className="border-t border-border p-4">
          <div className="flex items-end gap-2">
            <div className="flex-1 rounded-lg border border-border bg-background focus-within:ring-2 focus-within:ring-primary">
              <textarea
                ref={textareaRef}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Type a message… (Enter to send, Shift+Enter for newline)"
                rows={1}
                disabled={sending}
                className="w-full resize-none bg-transparent px-3 py-2.5 text-sm focus:outline-none disabled:opacity-50"
              />
            </div>
            <button
              onClick={sendMessage}
              disabled={!input.trim() || sending}
              className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-primary text-primary-foreground transition-colors hover:bg-primary/90 disabled:cursor-not-allowed disabled:opacity-50"
              title="Send message"
            >
              {sending ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <Send className="h-4 w-4" />
              )}
            </button>
          </div>
          <p className="mt-1.5 text-xs text-muted-foreground">
            Augmenting with Ucotron namespace:{" "}
            <span className="font-mono font-medium">{namespace}</span>
          </p>
        </div>
      </div>
    </div>
  );
}
