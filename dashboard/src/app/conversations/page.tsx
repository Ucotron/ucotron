"use client";

import { useEffect, useState, useCallback } from "react";
import {
  MessageSquare,
  ChevronDown,
  ChevronRight,
  Clock,
  Hash,
  X,
  Loader2,
  Tag,
} from "lucide-react";
import { Card } from "@/components/card";
import {
  listConversations,
  getConversationMessages,
} from "@/lib/api";
import type {
  ConversationSummary,
  ConversationDetail,
  ConversationMessage,
} from "@/lib/api";
import { useNamespace } from "@/components/namespace-context";

function formatDate(dateStr: string): string {
  if (!dateStr) return "\u2014";
  const d = new Date(dateStr);
  if (isNaN(d.getTime())) return dateStr;
  return d.toLocaleDateString(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function relativeTime(dateStr: string): string {
  if (!dateStr) return "";
  const d = new Date(dateStr);
  if (isNaN(d.getTime())) return "";
  const now = Date.now();
  const diff = now - d.getTime();
  const mins = Math.floor(diff / 60000);
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins}m ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs}h ago`;
  const days = Math.floor(hrs / 24);
  if (days < 30) return `${days}d ago`;
  return formatDate(dateStr);
}

function MessageBubble({ message }: { message: ConversationMessage }) {
  return (
    <div className="rounded-md border border-border bg-muted/30 p-3">
      <p className="whitespace-pre-wrap text-sm leading-relaxed">
        {message.content}
      </p>
      <div className="mt-2 flex flex-wrap items-center gap-2">
        <span className="flex items-center gap-1 text-xs text-muted-foreground">
          <Clock className="h-3 w-3" />
          {formatDate(message.created_at)}
        </span>
        {message.id && (
          <span className="font-mono text-xs text-muted-foreground">
            {message.id.slice(0, 8)}
          </span>
        )}
      </div>
      {message.entities.length > 0 && (
        <div className="mt-2 flex flex-wrap gap-1.5">
          {message.entities.map((ent, i) => (
            <span
              key={`${ent.content}-${i}`}
              className="flex items-center gap-1 rounded-full bg-accent px-2 py-0.5 text-xs text-accent-foreground"
            >
              <Tag className="h-2.5 w-2.5" />
              {ent.node_type && (
                <span className="font-medium">{ent.node_type}:</span>
              )}
              {ent.content}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

export default function ConversationsPage() {
  const { namespace } = useNamespace();
  const [conversations, setConversations] = useState<ConversationSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  // Expanded conversation state
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const [detail, setDetail] = useState<ConversationDetail | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);

  const fetchConversations = useCallback(() => {
    setLoading(true);
    setError("");
    listConversations(namespace)
      .then((items) => setConversations(items))
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, [namespace]);

  useEffect(() => {
    fetchConversations();
  }, [fetchConversations]);

  async function toggleExpand(conversationId: string) {
    if (expandedId === conversationId) {
      setExpandedId(null);
      setDetail(null);
      return;
    }

    setExpandedId(conversationId);
    setDetail(null);
    setDetailLoading(true);

    try {
      const data = await getConversationMessages(conversationId, namespace);
      setDetail(data);
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Failed to load messages"
      );
      setExpandedId(null);
    } finally {
      setDetailLoading(false);
    }
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <MessageSquare className="h-5 w-5 text-primary" />
          <h1 className="text-2xl font-bold">Conversations</h1>
        </div>
        <span className="text-xs text-muted-foreground">
          Namespace:{" "}
          <span className="font-mono font-medium text-foreground">
            {namespace}
          </span>
        </span>
      </div>

      <p className="text-sm text-muted-foreground">
        Browse conversation history tracked by Ucotron. Click a conversation to
        view its full message timeline with extracted entities.
      </p>

      {/* Error */}
      {error && (
        <div className="flex items-center justify-between rounded-md border border-destructive/30 bg-destructive/5 p-3 text-sm text-destructive">
          <span>{error}</span>
          <button onClick={() => setError("")}>
            <X className="h-4 w-4" />
          </button>
        </div>
      )}

      {/* Loading */}
      {loading && (
        <Card className="flex items-center justify-center py-12">
          <div className="flex flex-col items-center gap-3">
            <Loader2 className="h-8 w-8 animate-spin text-primary" />
            <p className="text-sm text-muted-foreground">
              Loading conversations...
            </p>
          </div>
        </Card>
      )}

      {/* Empty state */}
      {!loading && conversations.length === 0 && !error && (
        <Card className="flex flex-col items-center gap-3 py-12">
          <MessageSquare className="h-10 w-10 text-muted-foreground" />
          <p className="text-sm text-muted-foreground">
            No conversations found in this namespace.
          </p>
        </Card>
      )}

      {/* Conversation list */}
      {!loading && conversations.length > 0 && (
        <div className="space-y-3">
          {conversations.map((conv) => {
            const isExpanded = expandedId === conv.conversation_id;

            return (
              <Card key={conv.conversation_id} className="p-0 overflow-hidden">
                {/* Conversation header / card */}
                <button
                  onClick={() => toggleExpand(conv.conversation_id)}
                  className="flex w-full items-start gap-3 p-4 text-left transition-colors hover:bg-muted/50"
                >
                  <div className="mt-0.5 shrink-0 text-muted-foreground">
                    {isExpanded ? (
                      <ChevronDown className="h-4 w-4" />
                    ) : (
                      <ChevronRight className="h-4 w-4" />
                    )}
                  </div>

                  <div className="min-w-0 flex-1">
                    {/* Preview */}
                    <p
                      className={`text-sm ${
                        isExpanded ? "" : "line-clamp-2"
                      }`}
                    >
                      {conv.preview || "(no preview)"}
                    </p>

                    {/* Meta row */}
                    <div className="mt-2 flex flex-wrap items-center gap-3 text-xs text-muted-foreground">
                      <span
                        className="font-mono"
                        title={conv.conversation_id}
                      >
                        {conv.conversation_id.length > 12
                          ? `${conv.conversation_id.slice(0, 12)}...`
                          : conv.conversation_id}
                      </span>

                      <span className="flex items-center gap-1">
                        <Hash className="h-3 w-3" />
                        {conv.message_count}{" "}
                        {conv.message_count === 1 ? "message" : "messages"}
                      </span>

                      <span className="flex items-center gap-1">
                        <Clock className="h-3 w-3" />
                        {relativeTime(conv.last_message_at)}
                      </span>

                      <span className="hidden text-muted-foreground/60 sm:inline">
                        Started {formatDate(conv.first_message_at)}
                      </span>
                    </div>
                  </div>
                </button>

                {/* Expanded message list */}
                {isExpanded && (
                  <div className="border-t border-border bg-background px-4 py-4">
                    {detailLoading && (
                      <div className="flex items-center justify-center py-8">
                        <Loader2 className="h-5 w-5 animate-spin text-primary" />
                        <span className="ml-2 text-sm text-muted-foreground">
                          Loading messages...
                        </span>
                      </div>
                    )}

                    {!detailLoading && detail && (
                      <div className="space-y-3">
                        <div className="flex items-center justify-between text-xs text-muted-foreground">
                          <span>
                            {detail.messages.length}{" "}
                            {detail.messages.length === 1
                              ? "message"
                              : "messages"}{" "}
                            in conversation
                          </span>
                          <span className="font-mono">
                            {detail.conversation_id}
                          </span>
                        </div>

                        <div className="space-y-2">
                          {detail.messages.map((msg) => (
                            <MessageBubble key={msg.id} message={msg} />
                          ))}
                        </div>

                        {detail.messages.length === 0 && (
                          <p className="py-4 text-center text-sm text-muted-foreground">
                            No messages in this conversation.
                          </p>
                        )}
                      </div>
                    )}
                  </div>
                )}
              </Card>
            );
          })}
        </div>
      )}

      {/* Summary footer */}
      {!loading && conversations.length > 0 && (
        <p className="text-xs text-muted-foreground">
          Showing {conversations.length}{" "}
          {conversations.length === 1 ? "conversation" : "conversations"}
        </p>
      )}
    </div>
  );
}
