const API_URL = process.env.NEXT_PUBLIC_UCOTRON_API_URL || "http://localhost:8420";

export interface HealthResponse {
  status: string;
  version: string;
  instance_id: string;
  instance_role: string;
  storage_mode: string;
  vector_backend: string;
  graph_backend: string;
  models: {
    embedder_loaded: boolean;
    embedding_model: string;
    ner_loaded: boolean;
    relation_extractor_loaded: boolean;
  };
}

export interface MetricsResponse {
  instance_id: string;
  total_requests: number;
  total_ingestions: number;
  total_searches: number;
  uptime_secs: number;
}

export interface MemoryResponse {
  id: number;
  content: string;
  node_type: string;
  timestamp: number;
  metadata: Record<string, unknown>;
}

export interface SearchResultItem {
  id: number;
  content: string;
  node_type: string;
  score: number;
  vector_sim: number;
  graph_centrality: number;
  recency: number;
}

export interface SearchResponse {
  results: SearchResultItem[];
  total: number;
  query: string;
}

export interface EntityResponse {
  id: number;
  content: string;
  node_type: string;
  timestamp: number;
  metadata: Record<string, unknown>;
  neighbors?: NeighborResponse[];
}

export interface NeighborResponse {
  node_id: number;
  content: string;
  edge_type: string;
  weight: number;
}

async function apiFetch<T>(path: string, options?: RequestInit): Promise<T> {
  const url = `${API_URL}/api/v1${path}`;
  const res = await fetch(url, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...options?.headers,
    },
  });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`API error ${res.status}: ${body}`);
  }
  return res.json() as Promise<T>;
}

export async function getHealth(namespace?: string): Promise<HealthResponse> {
  const headers: Record<string, string> = {};
  if (namespace) headers["X-Ucotron-Namespace"] = namespace;
  return apiFetch("/health", { headers });
}

export async function getMetrics(): Promise<MetricsResponse> {
  return apiFetch("/metrics");
}

export async function listMemories(
  params?: { node_type?: string; limit?: number; offset?: number },
  namespace?: string
): Promise<MemoryResponse[]> {
  const query = new URLSearchParams();
  if (params?.node_type) query.set("node_type", params.node_type);
  if (params?.limit) query.set("limit", String(params.limit));
  if (params?.offset) query.set("offset", String(params.offset));
  const qs = query.toString();
  const headers: Record<string, string> = {};
  if (namespace) headers["X-Ucotron-Namespace"] = namespace;
  return apiFetch(`/memories${qs ? `?${qs}` : ""}`, { headers });
}

export async function searchMemories(
  body: { query: string; limit?: number; node_type?: string },
  namespace?: string
): Promise<SearchResponse> {
  const headers: Record<string, string> = {};
  if (namespace) headers["X-Ucotron-Namespace"] = namespace;
  return apiFetch("/memories/search", {
    method: "POST",
    body: JSON.stringify(body),
    headers,
  });
}

export async function listEntities(
  params?: { limit?: number; offset?: number },
  namespace?: string
): Promise<EntityResponse[]> {
  const query = new URLSearchParams();
  if (params?.limit) query.set("limit", String(params.limit));
  if (params?.offset) query.set("offset", String(params.offset));
  const qs = query.toString();
  const headers: Record<string, string> = {};
  if (namespace) headers["X-Ucotron-Namespace"] = namespace;
  return apiFetch(`/entities${qs ? `?${qs}` : ""}`, { headers });
}

export async function getEntity(
  id: number,
  namespace?: string
): Promise<EntityResponse> {
  const headers: Record<string, string> = {};
  if (namespace) headers["X-Ucotron-Namespace"] = namespace;
  return apiFetch(`/entities/${id}`, { headers });
}

// ---------------------------------------------------------------------------
// Memory CRUD
// ---------------------------------------------------------------------------

export async function getMemory(
  id: number,
  namespace?: string
): Promise<MemoryResponse> {
  const headers: Record<string, string> = {};
  if (namespace) headers["X-Ucotron-Namespace"] = namespace;
  return apiFetch(`/memories/${id}`, { headers });
}

export async function updateMemory(
  id: number,
  body: { content?: string; metadata?: Record<string, unknown> },
  namespace?: string
): Promise<MemoryResponse> {
  const headers: Record<string, string> = {};
  if (namespace) headers["X-Ucotron-Namespace"] = namespace;
  return apiFetch(`/memories/${id}`, {
    method: "PUT",
    body: JSON.stringify(body),
    headers,
  });
}

export async function deleteMemory(
  id: number,
  namespace?: string
): Promise<void> {
  const url = `${API_URL}/api/v1/memories/${id}`;
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  if (namespace) headers["X-Ucotron-Namespace"] = namespace;
  const res = await fetch(url, { method: "DELETE", headers });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`API error ${res.status}: ${body}`);
  }
}

export async function getRecentMemories(
  limitOrParams?: number | { limit?: number },
  namespace?: string
): Promise<MemoryResponse[]> {
  const limit = typeof limitOrParams === "number" ? limitOrParams : limitOrParams?.limit;
  const query = new URLSearchParams();
  if (limit) query.set("limit", String(limit));
  query.set("sort", "recent");
  const qs = query.toString();
  const headers: Record<string, string> = {};
  if (namespace) headers["X-Ucotron-Namespace"] = namespace;
  return apiFetch(`/memories${qs ? `?${qs}` : ""}`, { headers });
}

// ---------------------------------------------------------------------------
// Graph Visualization
// ---------------------------------------------------------------------------

export interface GraphNode {
  id: number;
  content: string;
  node_type: string;
  timestamp: number;
  community_id: number | null;
}

export interface GraphEdge {
  source: number;
  target: number;
  weight: number;
}

export interface GraphResponse {
  nodes: GraphNode[];
  edges: GraphEdge[];
  total_nodes: number;
  total_edges: number;
}

export async function getGraph(
  params?: { limit?: number; community_id?: number; node_type?: string },
  namespace?: string
): Promise<GraphResponse> {
  const query = new URLSearchParams();
  if (params?.limit) query.set("limit", String(params.limit));
  if (params?.community_id != null) query.set("community_id", String(params.community_id));
  if (params?.node_type) query.set("node_type", params.node_type);
  const qs = query.toString();
  const headers: Record<string, string> = {};
  if (namespace) headers["X-Ucotron-Namespace"] = namespace;
  return apiFetch(`/graph${qs ? `?${qs}` : ""}`, { headers });
}

// ---------------------------------------------------------------------------
// Admin: Namespace Management
// ---------------------------------------------------------------------------

export interface NamespaceInfo {
  name: string;
  memory_count: number;
  entity_count: number;
  total_nodes: number;
  last_activity: number;
}

export interface NamespaceListResponse {
  namespaces: NamespaceInfo[];
  total: number;
}

export interface ConfigSummaryResponse {
  server: { host: string; port: number };
  storage: {
    mode: string;
    vector_backend: string;
    graph_backend: string;
    vector_data_dir: string;
    graph_data_dir: string;
  };
  models: { models_dir: string; embedding_model: string };
  instance: {
    instance_id: string;
    role: string;
    id_range_start: number;
    id_range_size: number;
  };
  namespaces: {
    default_namespace: string;
    allowed_namespaces: string[];
    max_namespaces: number;
  };
}

export interface SystemInfoResponse {
  memory_rss_bytes: number;
  cpu_count: number;
  next_node_id: number;
  id_range_end: number;
  total_nodes: number;
  total_edges: number;
  uptime_secs: number;
}

export async function listNamespaces(): Promise<NamespaceListResponse> {
  return apiFetch("/admin/namespaces");
}

export async function getNamespace(name: string): Promise<NamespaceInfo> {
  return apiFetch(`/admin/namespaces/${encodeURIComponent(name)}`);
}

export async function createNamespace(name: string): Promise<{ name: string; created: boolean }> {
  return apiFetch("/admin/namespaces", {
    method: "POST",
    body: JSON.stringify({ name }),
  });
}

export async function deleteNamespace(name: string): Promise<{ name: string; nodes_deleted: number }> {
  const url = `${API_URL}/api/v1/admin/namespaces/${encodeURIComponent(name)}`;
  const res = await fetch(url, {
    method: "DELETE",
    headers: { "Content-Type": "application/json" },
  });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`API error ${res.status}: ${body}`);
  }
  return res.json();
}

export async function getAdminConfig(): Promise<ConfigSummaryResponse> {
  return apiFetch("/admin/config");
}

export async function getSystemInfo(): Promise<SystemInfoResponse> {
  return apiFetch("/admin/system");
}

// ---------------------------------------------------------------------------
// Agent Management
// ---------------------------------------------------------------------------

export interface AgentResponse {
  id: string;
  name: string;
  namespace: string;
  owner: string;
  created_at: number;
  config: Record<string, unknown>;
}

export interface ListAgentsResponse {
  agents: AgentResponse[];
  total: number;
  limit: number;
  offset: number;
}

export interface CreateAgentRequest {
  name: string;
  config?: Record<string, unknown>;
}

export interface CreateAgentResponse {
  id: string;
  name: string;
  namespace: string;
  owner: string;
  created_at: number;
}

export interface CloneAgentRequest {
  target_namespace?: string;
  node_types?: string[];
  time_range_start?: number;
  time_range_end?: number;
}

export interface CloneAgentResponse {
  source_agent_id: string;
  source_namespace: string;
  target_namespace: string;
  nodes_copied: number;
  edges_copied: number;
}

export interface MergeAgentRequest {
  source_agent_id: string;
}

export interface MergeAgentResponse {
  source_namespace: string;
  target_namespace: string;
  nodes_copied: number;
  edges_copied: number;
  nodes_deduplicated: number;
}

export interface ShareResponse {
  agent_id: string;
  target_agent_id: string;
  permission: string;
  created_at: number;
}

export interface ListSharesResponse {
  shares: ShareResponse[];
  total: number;
}

export async function listAgents(
  params?: { owner?: string; limit?: number; offset?: number }
): Promise<ListAgentsResponse> {
  const query = new URLSearchParams();
  if (params?.owner) query.set("owner", params.owner);
  if (params?.limit) query.set("limit", String(params.limit));
  if (params?.offset) query.set("offset", String(params.offset));
  const qs = query.toString();
  return apiFetch(`/agents${qs ? `?${qs}` : ""}`);
}

export async function createAgent(body: CreateAgentRequest): Promise<CreateAgentResponse> {
  return apiFetch("/agents", {
    method: "POST",
    body: JSON.stringify(body),
  });
}

export async function deleteAgent(id: string): Promise<{ id: string; deleted: boolean; nodes_deleted: number }> {
  const url = `${API_URL}/api/v1/agents/${encodeURIComponent(id)}`;
  const res = await fetch(url, {
    method: "DELETE",
    headers: { "Content-Type": "application/json" },
  });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`API error ${res.status}: ${body}`);
  }
  return res.json();
}

export async function cloneAgent(
  id: string,
  body: CloneAgentRequest
): Promise<CloneAgentResponse> {
  return apiFetch(`/agents/${encodeURIComponent(id)}/clone`, {
    method: "POST",
    body: JSON.stringify(body),
  });
}

export async function mergeAgent(
  id: string,
  body: MergeAgentRequest
): Promise<MergeAgentResponse> {
  return apiFetch(`/agents/${encodeURIComponent(id)}/merge`, {
    method: "POST",
    body: JSON.stringify(body),
  });
}

export async function createShare(
  agentId: string,
  body: { target_agent_id: string; permission?: string }
): Promise<ShareResponse> {
  return apiFetch(`/agents/${encodeURIComponent(agentId)}/share`, {
    method: "POST",
    body: JSON.stringify(body),
  });
}

export async function listShares(agentId: string): Promise<ListSharesResponse> {
  return apiFetch(`/agents/${encodeURIComponent(agentId)}/share`);
}

export async function deleteShare(agentId: string, targetId: string): Promise<void> {
  const url = `${API_URL}/api/v1/agents/${encodeURIComponent(agentId)}/share/${encodeURIComponent(targetId)}`;
  const res = await fetch(url, {
    method: "DELETE",
    headers: { "Content-Type": "application/json" },
  });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`API error ${res.status}: ${body}`);
  }
}

// ---------------------------------------------------------------------------
// Connector Management
// ---------------------------------------------------------------------------

export interface ConnectorScheduleResponse {
  connector_id: string;
  cron_expression: string | null;
  enabled: boolean;
  timeout_secs: number;
  max_retries: number;
  next_fire_time: string | null;
}

export interface ConnectorSyncRecordResponse {
  started_at: number;
  finished_at: number | null;
  items_fetched: number;
  items_skipped: number;
  error: string | null;
}

export interface ConnectorSyncHistoryResponse {
  connector_id: string;
  records: ConnectorSyncRecordResponse[];
}

export interface ConnectorSyncTriggerResponse {
  triggered: boolean;
  connector_id: string;
  message: string;
}

export async function listConnectorSchedules(): Promise<ConnectorScheduleResponse[]> {
  return apiFetch("/connectors/schedules");
}

export async function getConnectorHistory(
  id: string
): Promise<ConnectorSyncHistoryResponse> {
  return apiFetch(`/connectors/${encodeURIComponent(id)}/history`);
}

export async function triggerConnectorSync(
  id: string
): Promise<ConnectorSyncTriggerResponse> {
  return apiFetch(`/connectors/${encodeURIComponent(id)}/sync`, {
    method: "POST",
  });
}

export async function deleteConnector(
  id: string
): Promise<{ connector_id: string; deleted: boolean }> {
  const url = `${API_URL}/api/v1/connectors/${encodeURIComponent(id)}`;
  const res = await fetch(url, {
    method: "DELETE",
    headers: { "Content-Type": "application/json" },
  });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`API error ${res.status}: ${body}`);
  }
  return res.json();
}

// ---------------------------------------------------------------------------
// Augment & Learn (Memory Chat integration)
// ---------------------------------------------------------------------------

export interface AugmentRequest {
  query: string;
  max_memories?: number;
  max_hops?: number;
  debug?: boolean;
}

export interface AugmentResponse {
  memories: SearchResultItem[];
  entities: EntityResponse[];
  context_text: string;
  debug?: AugmentDebugInfo;
}

export interface AugmentDebugInfo {
  pipeline_timings: PipelineTimings;
  pipeline_duration_ms: number;
  vector_results_count: number;
  query_entities_count: number;
  score_breakdown: ScoreBreakdown[];
}

export interface PipelineTimings {
  query_embedding_us: number;
  vector_search_us: number;
  entity_extraction_us: number;
  graph_expansion_us: number;
  community_selection_us: number;
  reranking_us: number;
  context_assembly_us: number;
  total_us: number;
}

export interface ScoreBreakdown {
  id: number;
  final_score: number;
  vector_sim: number;
  graph_centrality: number;
  recency: number;
  mindset_score: number;
  path_reward: number;
}

export interface LearnRequest {
  text: string;
  metadata?: Record<string, unknown>;
}

export interface LearnResponse {
  nodes_created: number;
  entities_extracted: number;
  relations_created: number;
}

export async function augmentQuery(
  body: AugmentRequest,
  namespace?: string
): Promise<AugmentResponse> {
  const headers: Record<string, string> = {};
  if (namespace) headers["X-Ucotron-Namespace"] = namespace;
  // Server expects { context, limit, max_hops, debug } — map from our interface
  const serverBody: Record<string, unknown> = {
    context: body.query,
    debug: body.debug ?? true,
  };
  if (body.max_memories != null) serverBody.limit = body.max_memories;
  if (body.max_hops != null) serverBody.max_hops = body.max_hops;
  return apiFetch("/augment", {
    method: "POST",
    body: JSON.stringify(serverBody),
    headers,
  });
}

export async function augmentQueryDebug(
  query: string,
  namespace?: string,
  limit?: number
): Promise<AugmentResponse> {
  return augmentQuery(
    { query, max_memories: limit ?? 10, debug: true },
    namespace
  );
}

export async function learnText(
  body: LearnRequest,
  namespace?: string
): Promise<LearnResponse> {
  const headers: Record<string, string> = {};
  if (namespace) headers["X-Ucotron-Namespace"] = namespace;
  return apiFetch("/learn", {
    method: "POST",
    body: JSON.stringify(body),
    headers,
  });
}

// ---------------------------------------------------------------------------
// Conversations
// ---------------------------------------------------------------------------

export interface ConversationSummary {
  conversation_id: string;
  namespace: string;
  message_count: number;
  first_message_at: string;
  last_message_at: string;
  preview: string;
}

export interface ConversationMessageEntity {
  id?: number;
  content: string;
  node_type?: string;
}

export interface ConversationMessage {
  id: string;
  content: string;
  created_at: string;
  entities: ConversationMessageEntity[];
}

export interface ConversationDetail {
  conversation_id: string;
  namespace: string;
  messages: ConversationMessage[];
}

export async function listConversations(
  namespace?: string
): Promise<ConversationSummary[]> {
  const headers: Record<string, string> = {};
  if (namespace) headers["X-Ucotron-Namespace"] = namespace;
  return apiFetch("/conversations", { headers });
}

export async function getConversationMessages(
  conversationId: string,
  namespace?: string
): Promise<ConversationDetail> {
  const headers: Record<string, string> = {};
  if (namespace) headers["X-Ucotron-Namespace"] = namespace;
  return apiFetch(`/conversations/${encodeURIComponent(conversationId)}/messages`, { headers });
}

// End of OSS API client
