//! JSON request/response types for the Ucotron REST API.

use serde::{Deserialize, Deserializer, Serialize};
use std::collections::HashMap;
use utoipa::ToSchema;

/// Deserializes `media_filter` accepting either a single string or an array of strings.
/// Examples: `"image"` → `Some(vec!["image"])`, `["text","image"]` → `Some(vec!["text","image"])`.
fn deserialize_media_filter<'de, D>(deserializer: D) -> Result<Option<Vec<String>>, D::Error>
where
    D: Deserializer<'de>,
{
    use serde::de;

    struct MediaFilterVisitor;

    impl<'de> de::Visitor<'de> for MediaFilterVisitor {
        type Value = Option<Vec<String>>;

        fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            f.write_str("null, a string, or an array of strings")
        }

        fn visit_none<E: de::Error>(self) -> Result<Self::Value, E> {
            Ok(None)
        }

        fn visit_unit<E: de::Error>(self) -> Result<Self::Value, E> {
            Ok(None)
        }

        fn visit_str<E: de::Error>(self, v: &str) -> Result<Self::Value, E> {
            Ok(Some(vec![v.to_string()]))
        }

        fn visit_string<E: de::Error>(self, v: String) -> Result<Self::Value, E> {
            Ok(Some(vec![v]))
        }

        fn visit_seq<A: de::SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
            let mut vals = Vec::new();
            while let Some(s) = seq.next_element::<String>()? {
                vals.push(s);
            }
            if vals.is_empty() {
                Ok(None)
            } else {
                Ok(Some(vals))
            }
        }
    }

    deserializer.deserialize_any(MediaFilterVisitor)
}

// ---------------------------------------------------------------------------
// Memories
// ---------------------------------------------------------------------------

/// POST /api/v1/memories — request body.
#[derive(Debug, Deserialize, ToSchema)]
pub struct CreateMemoryRequest {
    /// The raw text to ingest.
    pub text: String,
    /// Optional metadata to attach.
    #[serde(default)]
    pub metadata: HashMap<String, serde_json::Value>,
}

/// POST /api/v1/memories — response.
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct CreateMemoryResponse {
    pub chunk_node_ids: Vec<u64>,
    pub entity_node_ids: Vec<u64>,
    pub edges_created: usize,
    pub metrics: IngestionMetricsResponse,
}

/// Subset of ingestion metrics exposed via API.
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct IngestionMetricsResponse {
    pub chunks_processed: usize,
    pub entities_extracted: usize,
    pub relations_extracted: usize,
    pub contradictions_detected: usize,
    pub total_us: u64,
}

/// GET /api/v1/memories — query parameters.
#[derive(Debug, Deserialize, utoipa::IntoParams)]
pub struct ListMemoriesParams {
    /// Filter by node type (entity, event, fact, skill).
    pub node_type: Option<String>,
    /// Maximum number of results.
    #[serde(default = "default_limit")]
    pub limit: usize,
    /// Offset for pagination.
    #[serde(default)]
    pub offset: usize,
}

fn default_limit() -> usize {
    50
}

/// A memory node as returned by the API.
#[derive(Debug, Serialize, Clone, ToSchema)]
pub struct MemoryResponse {
    pub id: u64,
    pub content: String,
    pub node_type: String,
    pub timestamp: u64,
    pub metadata: HashMap<String, serde_json::Value>,
}

/// PUT /api/v1/memories/:id — request body.
#[derive(Debug, Deserialize, ToSchema)]
pub struct UpdateMemoryRequest {
    /// Updated text content (optional).
    pub content: Option<String>,
    /// Updated metadata (merged with existing).
    #[serde(default)]
    pub metadata: HashMap<String, serde_json::Value>,
}

// ---------------------------------------------------------------------------
// Search
// ---------------------------------------------------------------------------

/// POST /api/v1/memories/search — request body.
#[derive(Debug, Deserialize, ToSchema)]
pub struct SearchRequest {
    /// Natural language query.
    pub query: String,
    /// Max results (default 10).
    #[serde(default = "default_search_limit")]
    pub limit: Option<usize>,
    /// Filter by node type.
    pub node_type: Option<String>,
    /// Time range filter [min_ts, max_ts].
    pub time_range: Option<(u64, u64)>,
    /// Optional cognitive mindset for scoring ("convergent", "divergent", "algorithmic").
    pub query_mindset: Option<String>,
}

fn default_search_limit() -> Option<usize> {
    Some(10)
}

/// Single search result.
#[derive(Debug, Serialize, ToSchema)]
pub struct SearchResultItem {
    pub id: u64,
    pub content: String,
    pub node_type: String,
    pub score: f32,
    pub vector_sim: f32,
    pub graph_centrality: f32,
    pub recency: f32,
    /// Mindset-aware score component (0.0 when no mindset is configured).
    pub mindset_score: f32,
    /// For video/audio segments: (start_ms, end_ms) timestamp range.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timestamp_range: Option<(u64, u64)>,
    /// For video segments: the parent video node ID.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent_video_id: Option<u64>,
}

/// POST /api/v1/memories/search — response.
#[derive(Debug, Serialize, ToSchema)]
pub struct SearchResponse {
    pub results: Vec<SearchResultItem>,
    pub total: usize,
    pub query: String,
}

// ---------------------------------------------------------------------------
// Entities
// ---------------------------------------------------------------------------

/// GET /api/v1/entities — query parameters.
#[derive(Debug, Deserialize, utoipa::IntoParams)]
pub struct ListEntitiesParams {
    #[serde(default = "default_limit")]
    pub limit: usize,
    #[serde(default)]
    pub offset: usize,
}

/// A graph entity as returned by the API.
#[derive(Debug, Serialize, ToSchema)]
pub struct EntityResponse {
    pub id: u64,
    pub content: String,
    pub node_type: String,
    pub timestamp: u64,
    pub metadata: HashMap<String, serde_json::Value>,
    /// Neighbors (1-hop).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub neighbors: Option<Vec<NeighborResponse>>,
}

/// A neighbor edge+node pair.
#[derive(Debug, Serialize, ToSchema)]
pub struct NeighborResponse {
    pub node_id: u64,
    pub content: String,
    pub edge_type: String,
    pub weight: f32,
}

// ---------------------------------------------------------------------------
// Augment / Learn
// ---------------------------------------------------------------------------

/// POST /api/v1/augment — request body.
#[derive(Debug, Deserialize, ToSchema)]
pub struct AugmentRequest {
    /// The context or user message to augment with memories.
    pub context: String,
    /// Max memories to return (default 10).
    #[serde(default = "default_search_limit")]
    pub limit: Option<usize>,
    /// When true, include full pipeline debug information in the response.
    #[serde(default)]
    pub debug: bool,
}

/// POST /api/v1/augment — query parameters.
#[derive(Debug, Deserialize, utoipa::IntoParams)]
pub struct AugmentQueryParams {
    /// When true, include explainability metadata showing how results were retrieved.
    #[serde(default)]
    pub explain: Option<bool>,
}

/// POST /api/v1/augment — response.
#[derive(Debug, Serialize, ToSchema)]
pub struct AugmentResponse {
    pub memories: Vec<SearchResultItem>,
    pub entities: Vec<EntityResponse>,
    pub context_text: String,
    /// Debug information about the retrieval pipeline (only present when debug=true).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub debug: Option<AugmentDebugInfo>,
    /// Query explainability metadata (only present when ?explain=true).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub explain: Option<ExplainabilityInfo>,
}

/// Debug information for the augment pipeline.
#[derive(Debug, Serialize, ToSchema)]
pub struct AugmentDebugInfo {
    /// Per-step pipeline timings in microseconds.
    pub pipeline_timings: PipelineTimings,
    /// Total pipeline duration in milliseconds.
    pub pipeline_duration_ms: f64,
    /// Number of vector search seed results.
    pub vector_results_count: usize,
    /// Number of entities extracted from the query.
    pub query_entities_count: usize,
    /// Score breakdown for each returned memory.
    pub score_breakdown: Vec<ScoreBreakdown>,
}

/// Per-step pipeline timings in microseconds.
#[derive(Debug, Serialize, ToSchema)]
pub struct PipelineTimings {
    pub query_embedding_us: u64,
    pub vector_search_us: u64,
    pub entity_extraction_us: u64,
    pub graph_expansion_us: u64,
    pub community_selection_us: u64,
    pub reranking_us: u64,
    pub context_assembly_us: u64,
    pub total_us: u64,
}

/// Score breakdown for a single memory in the augment pipeline.
#[derive(Debug, Serialize, ToSchema)]
pub struct ScoreBreakdown {
    pub id: u64,
    pub final_score: f32,
    pub vector_sim: f32,
    pub graph_centrality: f32,
    pub recency: f32,
    pub mindset_score: f32,
    pub path_reward: f32,
}

// ---------------------------------------------------------------------------
// Query Explainability
// ---------------------------------------------------------------------------

/// Full explainability metadata for an augment query.
/// Returned when `?explain=true` is passed.
#[derive(Debug, Serialize, ToSchema)]
pub struct ExplainabilityInfo {
    /// Source nodes that contributed to the response, with relevance scores.
    pub source_nodes: Vec<ExplainSourceNode>,
    /// The retrieval path describing how results were found.
    pub retrieval_path: RetrievalPath,
    /// Context window composition: how many tokens each node contributed.
    pub context_composition: Vec<ContextContribution>,
    /// Total approximate token count of the assembled context.
    pub total_context_tokens: usize,
}

/// A source node with its relevance scores for explainability.
#[derive(Debug, Serialize, ToSchema)]
pub struct ExplainSourceNode {
    /// Node ID.
    pub id: u64,
    /// Text content (truncated for readability).
    pub content: String,
    /// Node type (entity, event, fact, skill).
    pub node_type: String,
    /// Overall relevance score.
    pub relevance_score: f32,
    /// Embedding cosine similarity score.
    pub embedding_similarity: f32,
    /// Graph centrality component.
    pub graph_centrality: f32,
    /// Recency decay component.
    pub recency: f32,
}

/// Describes the retrieval pipeline path taken to find results.
#[derive(Debug, Serialize, ToSchema)]
pub struct RetrievalPath {
    /// The query text that was embedded.
    pub query: String,
    /// Dimension of the query embedding vector.
    pub embedding_dimensions: usize,
    /// Node IDs returned by the initial HNSW vector search.
    pub hnsw_seed_ids: Vec<u64>,
    /// Node IDs added via graph expansion.
    pub graph_expanded_ids: Vec<u64>,
    /// Node IDs from community selection.
    pub community_ids: Vec<u64>,
    /// Final ranked node IDs after re-ranking.
    pub final_ranked_ids: Vec<u64>,
}

/// How many approximate tokens a single node contributed to the context window.
#[derive(Debug, Serialize, ToSchema)]
pub struct ContextContribution {
    /// Node ID.
    pub id: u64,
    /// Approximate token count (chars / 4 heuristic).
    pub approx_tokens: usize,
    /// Fraction of total context tokens.
    pub fraction: f32,
}

/// POST /api/v1/learn — request body.
#[derive(Debug, Deserialize, ToSchema)]
pub struct LearnRequest {
    /// Agent output or text to extract and store memories from.
    pub output: String,
    /// Optional metadata.
    #[serde(default)]
    pub metadata: HashMap<String, serde_json::Value>,
    /// Optional conversation ID to group messages into a session.
    #[serde(default)]
    pub conversation_id: Option<String>,
}

/// POST /api/v1/learn — response (same as create memory).
#[derive(Debug, Serialize, ToSchema)]
pub struct LearnResponse {
    pub memories_created: usize,
    pub entities_found: usize,
    pub conflicts_found: usize,
}

// ---------------------------------------------------------------------------
// Health / Metrics
// ---------------------------------------------------------------------------

/// GET /api/v1/health — response.
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct HealthResponse {
    pub status: String,
    pub version: String,
    pub instance_id: String,
    pub instance_role: String,
    pub storage_mode: String,
    pub vector_backend: String,
    pub graph_backend: String,
    pub models: ModelStatus,
}

/// Model availability status reported by health check.
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct ModelStatus {
    /// Whether a real embedding pipeline is loaded (vs stub).
    pub embedder_loaded: bool,
    /// Name of the configured embedding model.
    pub embedding_model: String,
    /// Whether the NER pipeline is loaded.
    pub ner_loaded: bool,
    /// Whether a relation extractor is loaded.
    pub relation_extractor_loaded: bool,
    /// Whether an LLM model is loaded for relation extraction (e.g., Qwen3-4B-GGUF).
    pub llm_loaded: bool,
    /// Name of the configured LLM model (empty if none).
    pub llm_model: String,
    /// Active relation extraction strategy ("co_occurrence", "llm", or "fireworks").
    pub relation_strategy: String,
    /// Whether the audio transcription pipeline is loaded (Whisper ONNX).
    pub transcriber_loaded: bool,
    /// Whether the CLIP image embedding pipeline is loaded.
    pub image_embedder_loaded: bool,
    /// Whether the CLIP cross-modal text encoder is loaded.
    pub cross_modal_encoder_loaded: bool,
    /// Whether the document OCR pipeline is loaded (pdf_extract + Tesseract).
    pub ocr_pipeline_loaded: bool,
    /// Embedding provider in use: "onnx" or "sidecar".
    pub embedding_provider: String,
    /// Whether the cross-encoder reranker is loaded (via sidecar).
    pub reranker_loaded: bool,
}

/// GET /api/v1/metrics — response.
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct MetricsResponse {
    pub instance_id: String,
    pub total_requests: u64,
    pub total_ingestions: u64,
    pub total_searches: u64,
    pub uptime_secs: u64,
}

/// Structured API error response.
#[derive(Debug, Serialize, ToSchema)]
pub struct ApiErrorResponse {
    pub code: String,
    pub message: String,
}

// ---------------------------------------------------------------------------
// Graph Visualization
// ---------------------------------------------------------------------------

/// GET /api/v1/graph — query parameters.
#[derive(Debug, Deserialize, utoipa::IntoParams)]
pub struct GraphParams {
    /// Maximum number of nodes to return (default 500).
    #[serde(default = "default_graph_limit")]
    pub limit: usize,
    /// Filter by community ID.
    pub community_id: Option<u64>,
    /// Filter by node type (entity, event, fact, skill).
    pub node_type: Option<String>,
}

fn default_graph_limit() -> usize {
    500
}

/// A node in the graph visualization.
#[derive(Debug, Serialize, ToSchema)]
pub struct GraphNode {
    pub id: u64,
    pub content: String,
    pub node_type: String,
    pub timestamp: u64,
    pub community_id: Option<u64>,
}

/// An edge in the graph visualization.
#[derive(Debug, Serialize, ToSchema)]
pub struct GraphEdge {
    pub source: u64,
    pub target: u64,
    pub weight: f32,
}

/// GET /api/v1/graph — response.
#[derive(Debug, Serialize, ToSchema)]
pub struct GraphResponse {
    pub nodes: Vec<GraphNode>,
    pub edges: Vec<GraphEdge>,
    pub total_nodes: usize,
    pub total_edges: usize,
}

// ---------------------------------------------------------------------------
// Export / Import
// ---------------------------------------------------------------------------

/// GET /api/v1/export — query parameters.
#[derive(Debug, Deserialize, utoipa::IntoParams)]
pub struct ExportParams {
    /// Namespace to export (default: from X-Ucotron-Namespace header).
    pub namespace: Option<String>,
    /// Export format: "jsonld" (default).
    #[serde(default = "default_export_format")]
    pub format: String,
    /// Whether to include embedding vectors (default: true).
    #[serde(default = "default_true")]
    pub include_embeddings: bool,
    /// Only export nodes with timestamp >= this value (incremental export).
    pub from_timestamp: Option<u64>,
}

fn default_export_format() -> String {
    "jsonld".to_string()
}

fn default_true() -> bool {
    true
}

/// GET /api/v1/export — response (the full JSON-LD document).
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct ExportResponse {
    /// JSON-LD context.
    #[serde(rename = "@context")]
    pub context: serde_json::Value,
    /// Type identifier.
    #[serde(rename = "@type")]
    pub graph_type: String,
    /// Export format version.
    pub version: String,
    /// Unix timestamp of export.
    pub exported_at: u64,
    /// Source namespace.
    pub namespace: String,
    /// Exported nodes.
    pub nodes: Vec<ExportNodeResponse>,
    /// Exported edges.
    pub edges: Vec<ExportEdgeResponse>,
    /// Community assignments.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub communities: HashMap<String, u64>,
    /// Export statistics.
    pub stats: ExportStatsResponse,
}

/// A node in the export.
#[derive(Debug, Serialize, Deserialize, Clone, ToSchema)]
pub struct ExportNodeResponse {
    /// Node ID in JSON-LD format.
    #[serde(rename = "@id")]
    pub id: String,
    /// Node type.
    #[serde(rename = "@type")]
    pub node_type: String,
    /// Text content.
    pub content: String,
    /// Unix timestamp.
    pub timestamp: u64,
    /// Metadata.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub metadata: HashMap<String, serde_json::Value>,
    /// Embedding vector (optional).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embedding: Option<Vec<f32>>,
}

/// An edge in the export.
#[derive(Debug, Serialize, Deserialize, Clone, ToSchema)]
pub struct ExportEdgeResponse {
    /// Source node ID.
    pub source: String,
    /// Target node ID.
    pub target: String,
    /// Edge type.
    pub edge_type: String,
    /// Edge weight.
    pub weight: f32,
    /// Edge metadata.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Export statistics.
#[derive(Debug, Serialize, Deserialize, Clone, ToSchema)]
pub struct ExportStatsResponse {
    pub total_nodes: usize,
    pub total_edges: usize,
    pub has_embeddings: bool,
    pub is_incremental: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub from_timestamp: Option<u64>,
}

/// POST /api/v1/import — request body (the full JSON-LD document to import).
#[derive(Debug, Deserialize, ToSchema)]
pub struct ImportRequest {
    /// JSON-LD context (validated but not used for import).
    #[serde(rename = "@context")]
    pub context: serde_json::Value,
    /// Type identifier.
    #[serde(rename = "@type")]
    pub graph_type: String,
    /// Export format version.
    pub version: String,
    /// Unix timestamp of original export.
    pub exported_at: u64,
    /// Source namespace.
    pub namespace: String,
    /// Nodes to import.
    pub nodes: Vec<ExportNodeResponse>,
    /// Edges to import.
    pub edges: Vec<ExportEdgeResponse>,
    /// Community assignments (optional).
    #[serde(default)]
    pub communities: HashMap<String, u64>,
    /// Export stats (optional).
    pub stats: Option<ExportStatsResponse>,
}

/// POST /api/v1/import — response.
#[derive(Debug, Serialize, ToSchema)]
pub struct ImportResponse {
    /// Number of nodes imported.
    pub nodes_imported: usize,
    /// Number of edges imported.
    pub edges_imported: usize,
    /// Target namespace where data was imported.
    pub target_namespace: String,
}

// ---------------------------------------------------------------------------
// Agents
// ---------------------------------------------------------------------------

/// POST /api/v1/agents — request body.
#[derive(Debug, Deserialize, ToSchema)]
#[schema(example = json!({"name": "support-bot", "config": {"model": "gpt-4", "temperature": 0.7}}))]
pub struct CreateAgentRequest {
    /// Human-readable name for the agent.
    pub name: String,
    /// Optional agent-specific configuration (model, prompt template, etc.).
    #[serde(default)]
    pub config: HashMap<String, serde_json::Value>,
}

/// POST /api/v1/agents — response.
#[derive(Debug, Serialize, Deserialize, ToSchema)]
#[schema(example = json!({"id": "agent_abc123", "name": "support-bot", "namespace": "agent_abc123", "owner": "admin", "created_at": 1700000000}))]
pub struct CreateAgentResponse {
    /// Generated agent ID.
    pub id: String,
    /// Agent display name.
    pub name: String,
    /// Auto-created isolated namespace for this agent.
    pub namespace: String,
    /// Owner identifier (from auth context).
    pub owner: String,
    /// Unix timestamp when the agent was created.
    pub created_at: u64,
}

/// GET /api/v1/agents — query parameters.
#[derive(Debug, Deserialize, utoipa::IntoParams)]
pub struct ListAgentsParams {
    /// Filter by owner name.
    pub owner: Option<String>,
    /// Maximum number of results.
    #[serde(default = "default_agents_limit")]
    pub limit: usize,
    /// Offset for pagination.
    #[serde(default)]
    pub offset: usize,
}

fn default_agents_limit() -> usize {
    50
}

/// GET /api/v1/agents — response.
#[derive(Debug, Serialize, ToSchema)]
#[schema(example = json!({"agents": [{"id": "agent_abc123", "name": "support-bot", "namespace": "agent_abc123", "owner": "admin", "created_at": 1700000000, "config": {}}], "total": 1, "limit": 50, "offset": 0}))]
pub struct ListAgentsResponse {
    pub agents: Vec<AgentResponse>,
    pub total: usize,
    pub limit: usize,
    pub offset: usize,
}

/// Agent details returned in API responses.
#[derive(Debug, Serialize, ToSchema)]
#[schema(example = json!({"id": "agent_abc123", "name": "support-bot", "namespace": "agent_abc123", "owner": "admin", "created_at": 1700000000, "config": {"model": "gpt-4"}}))]
pub struct AgentResponse {
    pub id: String,
    pub name: String,
    pub namespace: String,
    pub owner: String,
    pub created_at: u64,
    pub config: HashMap<String, serde_json::Value>,
}

// ---------------------------------------------------------------------------
// Agent Clone
// ---------------------------------------------------------------------------

/// POST /api/v1/agents/{id}/clone — request body.
#[derive(Debug, Deserialize, ToSchema)]
#[schema(example = json!({"target_namespace": "cloned-bot-ns", "node_types": ["Entity", "Fact"], "time_range_start": 1690000000, "time_range_end": 1700000000}))]
pub struct CloneAgentRequest {
    /// Target namespace for the cloned graph. If omitted, auto-generates one.
    pub target_namespace: Option<String>,
    /// Optional node type filter (only clone nodes of these types).
    pub node_types: Option<Vec<String>>,
    /// Optional timestamp range start (only clone nodes >= this timestamp).
    pub time_range_start: Option<u64>,
    /// Optional timestamp range end (only clone nodes <= this timestamp).
    pub time_range_end: Option<u64>,
}

/// POST /api/v1/agents/{id}/clone — response.
#[derive(Debug, Serialize, ToSchema)]
#[schema(example = json!({"source_agent_id": "agent_abc123", "source_namespace": "agent_abc123", "target_namespace": "cloned-bot-ns", "nodes_copied": 42, "edges_copied": 85}))]
pub struct CloneAgentResponse {
    /// Source agent ID.
    pub source_agent_id: String,
    /// Source namespace.
    pub source_namespace: String,
    /// Destination namespace where the clone was placed.
    pub target_namespace: String,
    /// Number of nodes copied.
    pub nodes_copied: usize,
    /// Number of edges copied.
    pub edges_copied: usize,
}

// ---------------------------------------------------------------------------
// Agent Merge
// ---------------------------------------------------------------------------

/// POST /api/v1/agents/{id}/merge — request body.
#[derive(Debug, Deserialize, ToSchema)]
#[schema(example = json!({"source_agent_id": "agent_xyz789"}))]
pub struct MergeAgentRequest {
    /// Source agent ID whose graph will be merged into this agent.
    pub source_agent_id: String,
}

/// POST /api/v1/agents/{id}/merge — response.
#[derive(Debug, Serialize, ToSchema)]
#[schema(example = json!({"source_namespace": "agent_xyz789", "target_namespace": "agent_abc123", "nodes_copied": 30, "edges_copied": 55, "nodes_deduplicated": 5, "ids_remapped": 3}))]
pub struct MergeAgentResponse {
    /// Source namespace that was merged from.
    pub source_namespace: String,
    /// Destination namespace that received the merge.
    pub target_namespace: String,
    /// Number of new nodes copied (non-duplicates).
    pub nodes_copied: usize,
    /// Number of edges copied.
    pub edges_copied: usize,
    /// Number of duplicate nodes that were deduplicated.
    pub nodes_deduplicated: usize,
    /// Number of node IDs that were remapped to avoid conflicts.
    pub ids_remapped: usize,
}

// ---------------------------------------------------------------------------
// Agent Share
// ---------------------------------------------------------------------------

/// POST /api/v1/agents/{id}/share — request body.
#[derive(Debug, Deserialize, ToSchema)]
#[schema(example = json!({"target_agent_id": "agent_xyz789", "permission": "read"}))]
pub struct CreateShareRequest {
    /// Target agent ID to share with.
    pub target_agent_id: String,
    /// Permission level: "read" or "read_write".
    #[serde(default = "default_share_permission")]
    pub permission: String,
}

fn default_share_permission() -> String {
    "read".to_string()
}

/// POST /api/v1/agents/{id}/share — response.
#[derive(Debug, Serialize, ToSchema)]
#[schema(example = json!({"agent_id": "agent_abc123", "target_agent_id": "agent_xyz789", "permission": "read", "created_at": 1700000000}))]
pub struct CreateShareResponse {
    /// Source agent ID (granting access).
    pub agent_id: String,
    /// Target agent ID (receiving access).
    pub target_agent_id: String,
    /// Permission level granted.
    pub permission: String,
    /// Unix timestamp when the share was created.
    pub created_at: u64,
}

/// GET /api/v1/agents/{id}/share — response (list shares).
#[derive(Debug, Serialize, ToSchema)]
#[schema(example = json!({"shares": [{"agent_id": "agent_abc123", "target_agent_id": "agent_xyz789", "permission": "read", "created_at": 1700000000}], "total": 1}))]
pub struct ListSharesResponse {
    /// All share grants for this agent.
    pub shares: Vec<ShareResponse>,
    /// Total number of shares.
    pub total: usize,
}

/// Individual share grant in API responses.
#[derive(Debug, Serialize, ToSchema)]
#[schema(example = json!({"agent_id": "agent_abc123", "target_agent_id": "agent_xyz789", "permission": "read", "created_at": 1700000000}))]
pub struct ShareResponse {
    /// Source agent ID.
    pub agent_id: String,
    /// Target agent ID.
    pub target_agent_id: String,
    /// Permission level.
    pub permission: String,
    /// Unix timestamp when the share was created.
    pub created_at: u64,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Convert a core `Value` to serde_json `Value`.
pub fn core_value_to_json(val: &ucotron_core::Value) -> serde_json::Value {
    match val {
        ucotron_core::Value::String(s) => serde_json::Value::String(s.clone()),
        ucotron_core::Value::Integer(i) => serde_json::json!(*i),
        ucotron_core::Value::Float(f) => serde_json::json!(*f),
        ucotron_core::Value::Bool(b) => serde_json::Value::Bool(*b),
    }
}

/// Convert a core `Node` to a `MemoryResponse`.
pub fn node_to_memory_response(node: &ucotron_core::Node) -> MemoryResponse {
    MemoryResponse {
        id: node.id,
        content: node.content.clone(),
        node_type: format!("{:?}", node.node_type),
        timestamp: node.timestamp,
        metadata: node
            .metadata
            .iter()
            .map(|(k, v)| (k.clone(), core_value_to_json(v)))
            .collect(),
    }
}

/// Convert a core `Node` to an `EntityResponse` (without neighbors).
pub fn node_to_entity_response(node: &ucotron_core::Node) -> EntityResponse {
    EntityResponse {
        id: node.id,
        content: node.content.clone(),
        node_type: format!("{:?}", node.node_type),
        timestamp: node.timestamp,
        metadata: node
            .metadata
            .iter()
            .map(|(k, v)| (k.clone(), core_value_to_json(v)))
            .collect(),
        neighbors: None,
    }
}

// ---------------------------------------------------------------------------
// Audio Transcription
// ---------------------------------------------------------------------------

/// Response from audio transcription and ingestion.
#[derive(Debug, Serialize, Deserialize, utoipa::ToSchema)]
pub struct TranscribeResponse {
    /// Full transcribed text.
    pub text: String,
    /// Per-chunk transcriptions (for audio > 30s).
    pub chunks: Vec<TranscribeChunk>,
    /// Audio metadata.
    pub audio: AudioMetadataResponse,
    /// Ingestion results (if text was ingested into memory).
    pub ingestion: Option<CreateMemoryResponse>,
}

/// Transcription of a single audio chunk.
#[derive(Debug, Serialize, Deserialize, utoipa::ToSchema)]
pub struct TranscribeChunk {
    /// Transcribed text.
    pub text: String,
    /// Start time in seconds.
    pub start_secs: f32,
    /// End time in seconds.
    pub end_secs: f32,
}

/// Audio file metadata.
#[derive(Debug, Serialize, Deserialize, utoipa::ToSchema)]
pub struct AudioMetadataResponse {
    /// Duration in seconds.
    pub duration_secs: f32,
    /// Original sample rate.
    pub sample_rate: u32,
    /// Number of channels.
    pub channels: u16,
    /// Detected language.
    pub detected_language: Option<String>,
}

// ---------------------------------------------------------------------------
// Image Embedding
// ---------------------------------------------------------------------------

/// Response from indexing an image via CLIP embeddings.
#[derive(Debug, Serialize, Deserialize, utoipa::ToSchema)]
pub struct ImageIndexResponse {
    /// Node ID assigned to this image memory.
    pub node_id: u64,
    /// Image dimensions (width, height).
    pub width: u32,
    pub height: u32,
    /// Detected image format.
    pub format: String,
    /// Embedding dimensionality (512 for CLIP ViT-B/32).
    pub embedding_dim: usize,
}

/// POST /api/v1/images/search — request body for cross-modal text-to-image search.
///
/// Encodes the text query with the CLIP text encoder and searches the visual
/// (512-dim) index for similar image nodes, returning results sorted by
/// cosine similarity.
#[derive(Debug, Deserialize, utoipa::ToSchema)]
pub struct ImageSearchRequest {
    /// Text query to search for similar images.
    pub query: String,
    /// Maximum number of results (default 10).
    #[serde(default = "default_search_limit")]
    pub limit: Option<usize>,
    /// Minimum similarity threshold (0.0 to 1.0). Results below this score are excluded.
    pub min_similarity: Option<f32>,
}

/// Single image search result.
#[derive(Debug, Serialize, utoipa::ToSchema)]
pub struct ImageSearchResultItem {
    /// Node ID of the matched image.
    pub node_id: u64,
    /// Cosine similarity score (0.0 to 1.0).
    pub score: f32,
    /// Description/content stored for this image node.
    pub content: String,
    /// Timestamp of when the image was indexed.
    pub timestamp: u64,
}

/// POST /api/v1/images/search — response.
#[derive(Debug, Serialize, utoipa::ToSchema)]
pub struct ImageSearchResponse {
    /// Matched image results, sorted by similarity.
    pub results: Vec<ImageSearchResultItem>,
    /// Total results returned.
    pub total: usize,
    /// The original query.
    pub query: String,
}

// ---------------------------------------------------------------------------
// Document OCR
// ---------------------------------------------------------------------------

/// POST /api/v1/ocr — response: extracted text from document.
#[derive(Debug, Serialize, ToSchema)]
pub struct OcrResponse {
    /// Full extracted text.
    pub text: String,
    /// Per-page extractions.
    pub pages: Vec<OcrPageResponse>,
    /// Document metadata.
    pub document: OcrDocumentMetadata,
    /// Ingestion results (if text was ingested into memory).
    pub ingestion: Option<CreateMemoryResponse>,
}

/// Extracted text from a single page.
#[derive(Debug, Serialize, ToSchema)]
pub struct OcrPageResponse {
    /// Page number (1-based).
    pub page_number: usize,
    /// Raw text content.
    pub text: String,
}

/// Metadata about the processed document.
#[derive(Debug, Serialize, ToSchema)]
pub struct OcrDocumentMetadata {
    /// Total number of pages.
    pub total_pages: usize,
    /// Detected document format (pdf, jpeg, png, etc.).
    pub format: String,
    /// Whether the document appeared to be a scanned image.
    pub is_scanned: bool,
}

// ---------------------------------------------------------------------------
// Admin / Namespace Management
// ---------------------------------------------------------------------------

/// GET /api/v1/admin/namespaces — response: list of discovered namespaces.
#[derive(Debug, Serialize, ToSchema)]
pub struct NamespaceListResponse {
    pub namespaces: Vec<NamespaceInfo>,
    pub total: usize,
}

/// Information about a single namespace.
#[derive(Debug, Serialize, ToSchema)]
pub struct NamespaceInfo {
    /// Namespace name.
    pub name: String,
    /// Number of memory nodes in this namespace.
    pub memory_count: usize,
    /// Number of entity nodes in this namespace.
    pub entity_count: usize,
    /// Total nodes in this namespace.
    pub total_nodes: usize,
    /// Timestamp of the most recent node (0 if empty).
    pub last_activity: u64,
}

/// POST /api/v1/admin/namespaces — request body.
#[derive(Debug, Deserialize, ToSchema)]
pub struct CreateNamespaceRequest {
    /// Name for the new namespace.
    pub name: String,
}

/// POST /api/v1/admin/namespaces — response.
#[derive(Debug, Serialize, ToSchema)]
pub struct CreateNamespaceResponse {
    pub name: String,
    pub created: bool,
}

/// DELETE /api/v1/admin/namespaces/:name — response.
#[derive(Debug, Serialize, ToSchema)]
pub struct DeleteNamespaceResponse {
    pub name: String,
    pub nodes_deleted: usize,
}

/// GET /api/v1/admin/config — response: read-only config summary.
#[derive(Debug, Serialize, ToSchema)]
pub struct ConfigSummaryResponse {
    pub server: ConfigServerSection,
    pub storage: ConfigStorageSection,
    pub models: ConfigModelsSection,
    pub instance: ConfigInstanceSection,
    pub namespaces: ConfigNamespacesSection,
}

/// Server config section.
#[derive(Debug, Serialize, ToSchema)]
pub struct ConfigServerSection {
    pub host: String,
    pub port: u16,
}

/// Storage config section.
#[derive(Debug, Serialize, ToSchema)]
pub struct ConfigStorageSection {
    pub mode: String,
    pub vector_backend: String,
    pub graph_backend: String,
    pub vector_data_dir: String,
    pub graph_data_dir: String,
}

/// Models config section.
#[derive(Debug, Serialize, ToSchema)]
pub struct ConfigModelsSection {
    pub models_dir: String,
    pub embedding_model: String,
}

/// Instance config section.
#[derive(Debug, Serialize, ToSchema)]
pub struct ConfigInstanceSection {
    pub instance_id: String,
    pub role: String,
    pub id_range_start: u64,
    pub id_range_size: u64,
}

/// Namespaces config section.
#[derive(Debug, Serialize, ToSchema)]
pub struct ConfigNamespacesSection {
    pub default_namespace: String,
    pub allowed_namespaces: Vec<String>,
    pub max_namespaces: usize,
}

/// GET /api/v1/admin/system — response: system resource info.
#[derive(Debug, Serialize, ToSchema)]
pub struct SystemInfoResponse {
    /// Process memory usage in bytes (RSS).
    pub memory_rss_bytes: u64,
    /// Number of logical CPUs.
    pub cpu_count: usize,
    /// Next node ID that will be allocated.
    pub next_node_id: u64,
    /// Upper bound of ID range.
    pub id_range_end: u64,
    /// Total nodes in the graph.
    pub total_nodes: usize,
    /// Total edges in the graph.
    pub total_edges: usize,
    /// Server uptime in seconds.
    pub uptime_secs: u64,
}

// ---------------------------------------------------------------------------
// Mem0 Import
// ---------------------------------------------------------------------------

/// POST /api/v1/import/mem0 — request body.
///
/// Accepts Mem0 exported memory data in any supported format:
/// - v2 object with `results` key
/// - Object with `memories` key
/// - Bare array of memory objects (v1)
#[derive(Debug, Deserialize, ToSchema)]
pub struct Mem0ImportRequest {
    /// The raw Mem0 JSON data to import. Can be the full response from
    /// `GET /v1/memories/` or a file export.
    pub data: serde_json::Value,
    /// Whether to create edges between memories sharing the same user_id.
    #[serde(default = "default_true")]
    pub link_same_user: bool,
    /// Whether to create edges between memories sharing the same agent_id.
    #[serde(default)]
    pub link_same_agent: bool,
}

/// POST /api/v1/import/mem0 — response.
#[derive(Debug, Serialize, ToSchema)]
pub struct Mem0ImportResponse {
    /// Number of Mem0 memories parsed.
    pub memories_parsed: usize,
    /// Number of nodes imported into Ucotron.
    pub nodes_imported: usize,
    /// Number of edges inferred and imported.
    pub edges_imported: usize,
    /// Target namespace where data was imported.
    pub target_namespace: String,
}

// ---------------------------------------------------------------------------
// Zep/Graphiti Import
// ---------------------------------------------------------------------------

/// POST /api/v1/import/zep — request body.
///
/// Accepts Zep or Graphiti exported data in any supported format:
/// - Graphiti export with `entities`, `episodes`, and `edges`
/// - Zep sessions with `sessions` and optional `facts`
/// - Bare array of sessions
#[derive(Debug, Deserialize, ToSchema)]
pub struct ZepImportRequest {
    /// The raw Zep/Graphiti JSON data to import.
    pub data: serde_json::Value,
    /// Whether to create edges between items sharing the same user_id.
    #[serde(default = "default_true")]
    pub link_same_user: bool,
    /// Whether to create edges between items sharing the same group_id.
    #[serde(default)]
    pub link_same_group: bool,
    /// Whether to preserve expired/invalid edges from Graphiti (default: true).
    #[serde(default = "default_true")]
    pub preserve_expired: bool,
}

/// POST /api/v1/import/zep — response.
#[derive(Debug, Serialize, ToSchema)]
pub struct ZepImportResponse {
    /// Number of memories/items parsed.
    pub memories_parsed: usize,
    /// Number of nodes imported into Ucotron.
    pub nodes_imported: usize,
    /// Number of edges imported or inferred.
    pub edges_imported: usize,
    /// Target namespace where data was imported.
    pub target_namespace: String,
}

// ---------------------------------------------------------------------------
// GDPR Compliance
// ---------------------------------------------------------------------------

/// DELETE /api/v1/gdpr/forget — query parameters.
#[derive(Debug, Deserialize, utoipa::IntoParams)]
pub struct GdprForgetParams {
    /// The user identifier whose data should be erased (right to be forgotten).
    /// At least one of user_id or email must be provided.
    #[serde(default)]
    pub user_id: Option<String>,
    /// Email address whose data should be erased. Used as an alternative or
    /// supplement to user_id for matching nodes with `_email` metadata.
    #[serde(default)]
    pub email: Option<String>,
}

/// DELETE /api/v1/gdpr/forget — response (deletion receipt).
#[derive(Debug, Serialize, ToSchema)]
pub struct GdprForgetResponse {
    /// User ID that was erased (if provided).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user_id: Option<String>,
    /// Email that was erased (if provided).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub email: Option<String>,
    /// Number of memory nodes deleted.
    pub memories_deleted: usize,
    /// Number of entity nodes deleted.
    pub entities_deleted: usize,
    /// Number of edges removed.
    pub edges_removed: usize,
    /// Number of embeddings deleted.
    pub embeddings_deleted: usize,
    /// Total items removed (nodes + edges + embeddings).
    pub total_items_removed: usize,
    /// Unix timestamp of the operation.
    pub erased_at: u64,
    /// Namespaces from which data was removed.
    pub namespaces_affected: Vec<String>,
}

/// GET /api/v1/gdpr/export — query parameters.
#[derive(Debug, Deserialize, utoipa::IntoParams)]
pub struct GdprExportParams {
    /// The user identifier whose data should be exported.
    pub user_id: String,
}

/// GET /api/v1/gdpr/export — response.
#[derive(Debug, Serialize, ToSchema)]
pub struct GdprExportResponse {
    /// User ID whose data is being exported.
    pub user_id: String,
    /// All memory/entity nodes belonging to this user.
    pub nodes: Vec<MemoryResponse>,
    /// All edges connecting this user's nodes.
    pub edges: Vec<GdprExportEdge>,
    /// Export statistics.
    pub stats: GdprExportStats,
    /// Unix timestamp of the export.
    pub exported_at: u64,
}

/// An edge in the GDPR export.
#[derive(Debug, Serialize, ToSchema)]
pub struct GdprExportEdge {
    pub source: u64,
    pub target: u64,
    pub edge_type: String,
    pub weight: f32,
}

/// GDPR export statistics.
#[derive(Debug, Serialize, ToSchema)]
pub struct GdprExportStats {
    pub total_nodes: usize,
    pub total_edges: usize,
}

/// GET /api/v1/gdpr/retention — response.
#[derive(Debug, Serialize, ToSchema)]
pub struct RetentionStatusResponse {
    /// Namespace-level retention policies.
    pub policies: Vec<RetentionPolicy>,
    /// Number of nodes expired by the last retention sweep.
    pub last_sweep_expired: usize,
    /// Unix timestamp of the last retention sweep.
    pub last_sweep_at: u64,
}

/// A retention policy for a namespace.
#[derive(Debug, Serialize, Deserialize, Clone, ToSchema)]
pub struct RetentionPolicy {
    /// Namespace this policy applies to ("*" = all).
    pub namespace: String,
    /// Time-to-live in seconds (0 = no expiry).
    pub ttl_secs: u64,
}

/// POST /api/v1/gdpr/retention — request body to configure retention.
#[derive(Debug, Deserialize, ToSchema)]
pub struct SetRetentionRequest {
    /// Namespace to configure (use "*" for all namespaces).
    pub namespace: String,
    /// Time-to-live in seconds (0 = disable retention).
    pub ttl_secs: u64,
}

/// POST /api/v1/gdpr/retention — response.
#[derive(Debug, Serialize, ToSchema)]
pub struct SetRetentionResponse {
    /// The configured policy.
    pub policy: RetentionPolicy,
    /// Whether this was a new policy or an update.
    pub created: bool,
}

/// POST /api/v1/gdpr/retention/sweep — response.
#[derive(Debug, Serialize, ToSchema)]
pub struct RetentionSweepResponse {
    /// Number of nodes expired and deleted.
    pub nodes_expired: usize,
    /// Number of namespaces checked.
    pub namespaces_checked: usize,
    /// Unix timestamp of this sweep.
    pub swept_at: u64,
}

/// A single GDPR audit log entry.
#[derive(Debug, Serialize, Deserialize, Clone, ToSchema)]
pub struct GdprAuditEntry {
    /// Unix timestamp of the operation.
    pub timestamp: u64,
    /// Operation type (forget, export, retention_sweep, retention_set).
    pub operation: String,
    /// Target user_id or namespace.
    pub target: String,
    /// Outcome details.
    pub details: String,
}

// ---------------------------------------------------------------------------
// RBAC / API Keys
// ---------------------------------------------------------------------------

/// POST /api/v1/auth/keys — request body.
#[derive(Debug, Deserialize, ToSchema)]
pub struct CreateApiKeyRequest {
    /// Human-readable name for this key.
    pub name: String,
    /// Role: "admin", "writer", "reader", or "viewer".
    pub role: String,
    /// Optional namespace scope. Omit for unrestricted access.
    #[serde(default)]
    pub namespace: Option<String>,
}

/// POST /api/v1/auth/keys — response (includes the secret key, shown only once).
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct CreateApiKeyResponse {
    /// Human-readable name.
    pub name: String,
    /// The generated secret key. Store securely — not retrievable again.
    pub key: String,
    /// Assigned role.
    pub role: String,
    /// Namespace scope (if any).
    pub namespace: Option<String>,
    /// Whether the key is active.
    pub active: bool,
}

/// GET /api/v1/auth/keys — response (key values are masked).
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct ListApiKeysResponse {
    pub keys: Vec<ApiKeyInfo>,
}

/// Summary of an API key (key value is masked).
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct ApiKeyInfo {
    pub name: String,
    /// Masked key (e.g., "mk_****abcd").
    pub key_preview: String,
    pub role: String,
    pub namespace: Option<String>,
    pub active: bool,
}

/// DELETE /api/v1/auth/keys/:name — response.
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct RevokeApiKeyResponse {
    pub name: String,
    pub revoked: bool,
}

/// GET /api/v1/auth/whoami — response.
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct WhoamiResponse {
    pub role: String,
    pub namespace_scope: Option<String>,
    pub key_name: Option<String>,
    pub auth_enabled: bool,
}

// ---------------------------------------------------------------------------
// Audit
// ---------------------------------------------------------------------------

/// GET /api/v1/audit — response.
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct AuditQueryResponse {
    /// Matching audit entries.
    pub entries: Vec<crate::audit::AuditEntry>,
    /// Total number of entries matching the filter.
    pub total: usize,
}

/// GET /api/v1/audit/export — response.
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct AuditExportResponse {
    /// All audit entries (unfiltered).
    pub entries: Vec<crate::audit::AuditEntry>,
    /// Total entries in the audit log.
    pub total: usize,
    /// Unix timestamp of the export.
    pub exported_at: u64,
}

// ---------------------------------------------------------------------------
// Fine-Tuning
// ---------------------------------------------------------------------------

/// Request body for generating a fine-tuning training dataset.
#[derive(Debug, Deserialize, utoipa::ToSchema)]
pub struct GenerateDatasetRequest {
    /// Maximum number of training samples to generate.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_samples: Option<usize>,
    /// Train/validation split ratio (0.0–1.0).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub train_ratio: Option<f32>,
    /// Minimum relations per sample.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_relations: Option<usize>,
    /// Maximum text length per sample.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_text_length: Option<usize>,
    /// Seed for deterministic shuffling.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
}

/// Response body from generating a fine-tuning training dataset.
#[derive(Debug, Serialize, utoipa::ToSchema)]
pub struct GenerateDatasetResponse {
    /// Training samples in SFT messages format.
    pub train_samples: Vec<serde_json::Value>,
    /// Validation samples in SFT messages format.
    pub validation_samples: Vec<serde_json::Value>,
    /// Total entity nodes in the graph.
    pub total_entities: usize,
    /// Total edges in the graph.
    pub total_edges: usize,
    /// Samples skipped (below threshold).
    pub skipped: usize,
}

// ---------------------------------------------------------------------------
// Multimodal Memory Ingestion
// ---------------------------------------------------------------------------

/// POST /api/v1/memories/text — request body for explicit text memory ingestion.
#[derive(Debug, Deserialize, ToSchema)]
pub struct CreateTextMemoryRequest {
    /// The raw text to ingest.
    pub text: String,
    /// Optional metadata to attach.
    #[serde(default)]
    pub metadata: HashMap<String, serde_json::Value>,
}

/// POST /api/v1/memories/text — response.
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct CreateTextMemoryResponse {
    /// Node IDs for the text chunks created.
    pub chunk_node_ids: Vec<u64>,
    /// Node IDs for extracted entities.
    pub entity_node_ids: Vec<u64>,
    /// Number of edges created.
    pub edges_created: usize,
    /// The media type of the ingested content.
    pub media_type: String,
    /// Ingestion pipeline metrics.
    pub metrics: IngestionMetricsResponse,
}

// ---------------------------------------------------------------------------
// Audio Memory Ingestion
// ---------------------------------------------------------------------------

/// POST /api/v1/memories/audio — response.
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct CreateAudioMemoryResponse {
    /// Node IDs for the text chunks created from the transcription.
    pub chunk_node_ids: Vec<u64>,
    /// Node IDs for extracted entities.
    pub entity_node_ids: Vec<u64>,
    /// Number of edges created.
    pub edges_created: usize,
    /// The media type of the ingested content.
    pub media_type: String,
    /// The transcribed text.
    pub transcription: String,
    /// Audio metadata (duration, sample rate, etc.).
    pub audio: AudioMetadataResponse,
    /// Ingestion pipeline metrics.
    pub metrics: IngestionMetricsResponse,
}

// ---------------------------------------------------------------------------
// Image Memory Ingestion
// ---------------------------------------------------------------------------

/// POST /api/v1/memories/image — response.
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct CreateImageMemoryResponse {
    /// Node ID assigned to the image memory node.
    pub node_id: u64,
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// Detected image format (e.g., "png", "jpeg").
    pub format: String,
    /// CLIP embedding dimensionality (512 for ViT-B/32).
    pub embedding_dim: usize,
    /// The media type of the ingested content.
    pub media_type: String,
    /// Whether the description was also ingested into the text index.
    pub description_ingested: bool,
    /// Ingestion metrics (only present when description was ingested).
    pub metrics: Option<IngestionMetricsResponse>,
}

// ---------------------------------------------------------------------------
// Video Memory Ingestion
// ---------------------------------------------------------------------------

/// POST /api/v1/memories/video — response.
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct CreateVideoMemoryResponse {
    /// Node ID assigned to the parent video node.
    pub video_node_id: u64,
    /// Node IDs for each temporal segment.
    pub segment_node_ids: Vec<u64>,
    /// Number of edges created (parent→segment links).
    pub edges_created: usize,
    /// Total number of frames extracted from the video.
    pub total_frames: usize,
    /// Number of temporal segments created.
    pub total_segments: usize,
    /// Video duration in milliseconds.
    pub duration_ms: u64,
    /// Original video resolution.
    pub video_width: u32,
    pub video_height: u32,
    /// The media type of the ingested content.
    pub media_type: String,
    /// Per-segment details.
    pub segments: Vec<VideoSegmentInfo>,
    /// Ingestion metrics (present when audio transcription was run).
    pub transcription_metrics: Option<IngestionMetricsResponse>,
}

/// Details about a single video segment.
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct VideoSegmentInfo {
    /// Node ID for this segment.
    pub node_id: u64,
    /// Segment start time in milliseconds.
    pub start_ms: u64,
    /// Segment end time in milliseconds.
    pub end_ms: u64,
    /// Number of frames in this segment.
    pub frame_count: usize,
    /// Whether this segment starts with a scene change.
    pub is_scene_change: bool,
}

// ---------------------------------------------------------------------------
// Video Segment Navigation
// ---------------------------------------------------------------------------

/// A video segment with navigation links to adjacent segments.
#[derive(Debug, Serialize, ToSchema)]
pub struct VideoSegmentDetail {
    /// Node ID of this segment.
    pub node_id: u64,
    /// Text content / transcript of this segment.
    pub content: String,
    /// Start time in milliseconds.
    pub start_ms: u64,
    /// End time in milliseconds.
    pub end_ms: u64,
    /// Media URI for the video file.
    pub media_uri: Option<String>,
    /// Node ID of the previous segment (null if this is the first).
    pub prev_segment_id: Option<u64>,
    /// Node ID of the next segment (null if this is the last).
    pub next_segment_id: Option<u64>,
}

/// GET /api/v1/videos/{parent_id}/segments — response.
#[derive(Debug, Serialize, ToSchema)]
pub struct VideoSegmentsResponse {
    /// The parent video node ID.
    pub parent_video_id: u64,
    /// Total number of segments.
    pub total: usize,
    /// Segments sorted by start time, with prev/next navigation links.
    pub segments: Vec<VideoSegmentDetail>,
}

// ---------------------------------------------------------------------------
// Helpers (continued)
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Multimodal Search
// ---------------------------------------------------------------------------

/// POST /api/v1/search/multimodal — request body for unified cross-modal search.
///
/// Accepts a query type and the corresponding query payload (text or base64 image),
/// dispatching to the [`CrossModalSearch`] orchestrator for unified multi-modal
/// search across text, image, audio, and video memories.
#[derive(Debug, Deserialize, ToSchema)]
pub struct MultimodalSearchRequest {
    /// Query type: "text", "text_to_image", "image", "image_to_text", "audio".
    pub query_type: String,
    /// Text query (required for query_type "text", "text_to_image", "audio").
    pub query_text: Option<String>,
    /// Base64-encoded image bytes (required for query_type "image", "image_to_text").
    pub query_image: Option<String>,
    /// Filter results by media type(s). Accepts a single type or array of types.
    /// Valid values: "text", "audio", "image", "video".
    /// When multiple types are provided, results matching ANY of them are returned.
    #[serde(default, deserialize_with = "deserialize_media_filter")]
    pub media_filter: Option<Vec<String>>,
    /// Time range filter \[min_ts, max_ts\]. Only results within this range are returned.
    pub time_range: Option<(u64, u64)>,
    /// Maximum number of results (default 10).
    #[serde(default = "default_search_limit")]
    pub limit: Option<usize>,
}

/// A single multimodal search result.
#[derive(Debug, Serialize, ToSchema)]
pub struct MultimodalSearchResultItem {
    /// Node ID of the matched memory.
    pub node_id: u64,
    /// Content/description of the matched memory.
    pub content: String,
    /// Similarity score (0.0 to 1.0).
    pub score: f32,
    /// Media type of the matched memory (text, audio, image, video).
    pub media_type: String,
    /// URI for media files (images, audio, video). Null for text.
    pub media_uri: Option<String>,
    /// Which index produced this result: "text_index", "visual_index", or "fused".
    pub source: String,
    /// For video/audio segments: (start_ms, end_ms) timestamp range.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timestamp_range: Option<(u64, u64)>,
    /// For video segments: the parent video node ID.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent_video_id: Option<u64>,
}

/// POST /api/v1/search/multimodal — response.
#[derive(Debug, Serialize, ToSchema)]
pub struct MultimodalSearchResponse {
    /// Ranked search results sorted by descending score.
    pub results: Vec<MultimodalSearchResultItem>,
    /// Total results returned.
    pub total: usize,
    /// The original query type.
    pub query_type: String,
    /// Search timing metrics.
    pub metrics: MultimodalSearchMetrics,
}

/// Timing metrics for a multimodal search operation.
#[derive(Debug, Serialize, ToSchema)]
pub struct MultimodalSearchMetrics {
    /// Time spent encoding the query in microseconds.
    pub query_encoding_us: u64,
    /// Time spent searching the text index in microseconds.
    pub text_search_us: u64,
    /// Time spent searching the visual index in microseconds.
    pub visual_search_us: u64,
    /// Time spent fusing results in microseconds.
    pub fusion_us: u64,
    /// Total search time in microseconds.
    pub total_us: u64,
    /// Number of final results.
    pub final_result_count: usize,
}

/// Parse a node type string into `NodeType`.
pub fn parse_node_type(s: &str) -> Option<ucotron_core::NodeType> {
    match s.to_lowercase().as_str() {
        "entity" => Some(ucotron_core::NodeType::Entity),
        "event" => Some(ucotron_core::NodeType::Event),
        "fact" => Some(ucotron_core::NodeType::Fact),
        "skill" => Some(ucotron_core::NodeType::Skill),
        _ => None,
    }
}

// --- Connector sync types ---

/// Response for connector sync trigger.
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct ConnectorSyncTriggerResponse {
    /// Whether the sync was triggered successfully.
    pub triggered: bool,
    /// Connector instance ID.
    pub connector_id: String,
    /// Message describing the result.
    pub message: String,
}

/// Response for listing connector schedules.
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct ConnectorScheduleResponse {
    /// Connector instance ID.
    pub connector_id: String,
    /// Cron expression (if configured).
    pub cron_expression: Option<String>,
    /// Whether this schedule is enabled.
    pub enabled: bool,
    /// Timeout in seconds.
    pub timeout_secs: u64,
    /// Max retries on failure.
    pub max_retries: u32,
    /// Next fire time as Unix timestamp (if cron is set).
    pub next_fire_at: Option<u64>,
}

/// Response for listing connector sync history.
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct ConnectorSyncHistoryResponse {
    /// Connector instance ID.
    pub connector_id: String,
    /// Sync records (most recent first).
    pub records: Vec<ConnectorSyncRecordResponse>,
}

/// A single sync record.
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct ConnectorSyncRecordResponse {
    /// When the sync started (Unix seconds).
    pub started_at: u64,
    /// When the sync finished (Unix seconds, null if still running).
    pub finished_at: Option<u64>,
    /// Number of items fetched.
    pub items_fetched: usize,
    /// Number of items skipped.
    pub items_skipped: usize,
    /// Sync status: "success", "failed", "running", or "cancelled".
    pub status: String,
    /// Error message (if failed).
    pub error: Option<String>,
}

/// Response for webhook receipt.
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct WebhookResponse {
    /// Whether the webhook was accepted.
    pub accepted: bool,
    /// Connector instance ID.
    pub connector_id: String,
    /// Number of content items parsed from the webhook.
    pub items_processed: usize,
    /// Whether an incremental sync was triggered.
    pub sync_triggered: bool,
    /// Message describing the result.
    pub message: String,
}

// ---------------------------------------------------------------------------
// Conversations
// ---------------------------------------------------------------------------

/// Query parameters for listing conversations.
#[derive(Debug, Deserialize, ToSchema)]
pub struct ListConversationsParams {
    #[serde(default = "default_limit")]
    pub limit: usize,
    #[serde(default)]
    pub offset: usize,
}

/// Summary of a conversation session.
#[derive(Debug, Serialize, ToSchema)]
pub struct ConversationSummary {
    /// Unique conversation identifier.
    pub conversation_id: String,
    /// Namespace the conversation belongs to.
    pub namespace: String,
    /// Number of messages (memory nodes) in this conversation.
    pub message_count: usize,
    /// Timestamp of the first message.
    pub first_message_at: Option<String>,
    /// Timestamp of the last message.
    pub last_message_at: Option<String>,
    /// Preview of the first message content.
    pub preview: String,
}

/// Full conversation with messages.
#[derive(Debug, Serialize, ToSchema)]
pub struct ConversationDetail {
    /// Unique conversation identifier.
    pub conversation_id: String,
    /// Namespace the conversation belongs to.
    pub namespace: String,
    /// Messages ordered by timestamp.
    pub messages: Vec<ConversationMessage>,
}

/// A single message within a conversation.
#[derive(Debug, Serialize, ToSchema)]
pub struct ConversationMessage {
    /// Node ID of the memory.
    pub id: u64,
    /// Content of the message.
    pub content: String,
    /// Timestamp of creation.
    pub created_at: String,
    /// Entities extracted from this message.
    pub entities: Vec<String>,
}

// ---------------------------------------------------------------------------
// Frame Embed Widget
// ---------------------------------------------------------------------------

/// GET /api/v1/frames/{id}/embed — query parameters.
#[derive(Debug, Deserialize, utoipa::IntoParams)]
pub struct FrameEmbedParams {
    /// Widget theme: "light", "dark", or "auto".
    #[serde(default = "default_theme")]
    pub theme: String,
    /// Widget width in pixels or CSS units.
    #[serde(default = "default_width")]
    pub width: String,
    /// Widget height in pixels or CSS units.
    #[serde(default = "default_height")]
    pub height: String,
    /// Widget title displayed in header.
    #[serde(default)]
    pub title: Option<String>,
    /// API key for frame-scoped authentication.
    #[serde(default)]
    pub api_key: Option<String>,
    /// Base URL for API requests (defaults to current origin).
    #[serde(default)]
    pub base_url: Option<String>,
    /// Primary accent color (hex).
    #[serde(default = "default_accent_color")]
    pub accent_color: String,
    /// Show powered-by footer.
    #[serde(default = "default_true")]
    pub show_branding: bool,
}

fn default_theme() -> String {
    "auto".to_string()
}
fn default_width() -> String {
    "400px".to_string()
}
fn default_height() -> String {
    "600px".to_string()
}
fn default_accent_color() -> String {
    "#6366f1".to_string()
}

/// POST /api/v1/frames/{id}/embed-key — response.
#[derive(Debug, Serialize, ToSchema)]
pub struct FrameEmbedKeyResponse {
    /// The generated embed API key (scoped to this frame).
    pub embed_key: String,
    /// Frame ID the key is scoped to.
    pub frame_id: String,
    /// Timestamp when the key was generated.
    pub created_at: u64,
}
