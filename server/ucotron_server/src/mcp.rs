//! MCP (Model Context Protocol) server implementation for Ucotron.
//!
//! Exposes 6 tools for LLM agents:
//! - `ucotron_add_memory` — Ingest a text as memory
//! - `ucotron_search` — Search relevant memories
//! - `ucotron_get_entity` — Get info about a named entity
//! - `ucotron_list_entities` — List entities in the graph
//! - `ucotron_augment` — Context augmentation
//! - `ucotron_learn` — Learn from agent output

use std::sync::Arc;

use rmcp::{
    ErrorData as McpError, ServerHandler,
    handler::server::tool::ToolRouter,
    handler::server::wrapper::Parameters,
    model::*,
    tool, tool_router, tool_handler,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::state::AppState;

// ---------------------------------------------------------------------------
// Parameter structs
// ---------------------------------------------------------------------------

/// Parameters for `ucotron_add_memory` tool.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct AddMemoryParams {
    /// The text to ingest as a memory.
    pub text: String,
}

/// Parameters for `ucotron_search` tool.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct SearchParams {
    /// Natural language query to search for.
    pub query: String,
    /// Maximum number of results (default: 10).
    pub limit: Option<usize>,
}

/// Parameters for `ucotron_get_entity` tool.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct GetEntityParams {
    /// The name of the entity to look up.
    pub name: String,
}

/// Parameters for `ucotron_list_entities` tool.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct ListEntitiesParams {
    /// Filter by entity type (e.g., "entity", "event", "fact").
    #[serde(rename = "type")]
    pub entity_type: Option<String>,
    /// Maximum number of results (default: 20).
    pub limit: Option<usize>,
}

/// Parameters for `ucotron_augment` tool.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct AugmentParams {
    /// The context or user message to augment with relevant memories.
    pub context: String,
    /// Maximum number of memories to return (default: 10).
    pub limit: Option<usize>,
}

/// Parameters for `ucotron_learn` tool.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct LearnParams {
    /// Agent output or text to extract and store memories from.
    pub output: String,
}

// ---------------------------------------------------------------------------
// Response types (for structured JSON output)
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct MemoryResult {
    pub id: u64,
    pub content: String,
    pub node_type: String,
    pub score: Option<f32>,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct EntityResult {
    pub id: u64,
    pub name: String,
    pub node_type: String,
    pub neighbors: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct AddMemoryResult {
    pub memories_created: usize,
    pub entities_found: usize,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct AugmentResult {
    pub memories: Vec<MemoryResult>,
    pub entities: Vec<String>,
    pub context_text: String,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct LearnResult {
    pub memories_created: usize,
    pub entities_found: usize,
    pub conflicts_found: usize,
}

// ---------------------------------------------------------------------------
// MCP Server
// ---------------------------------------------------------------------------

/// The Ucotron MCP server that exposes cognitive memory tools.
#[derive(Clone)]
pub struct UcotronMcpServer {
    state: Arc<AppState>,
    #[allow(dead_code)]
    tool_router: ToolRouter<Self>,
}

#[tool_router]
impl UcotronMcpServer {
    /// Create a new MCP server wrapping the shared application state.
    pub fn new(state: Arc<AppState>) -> Self {
        Self {
            state,
            tool_router: Self::tool_router(),
        }
    }

    /// Ingest a text as memory into the cognitive graph.
    #[tool(
        name = "ucotron_add_memory",
        description = "Add a text as memory to the Ucotron cognitive graph. The text will be chunked, embedded, and entities/relations will be extracted automatically."
    )]
    async fn add_memory(
        &self,
        params: Parameters<AddMemoryParams>,
    ) -> Result<CallToolResult, McpError> {
        use ucotron_extraction::ingestion::{IngestionConfig, IngestionOrchestrator};
        use std::sync::atomic::Ordering;

        let params = params.0;

        if params.text.trim().is_empty() {
            return Ok(CallToolResult::error(vec![Content::text(
                "Error: text must not be empty",
            )]));
        }

        self.state.total_ingestions.fetch_add(1, Ordering::Relaxed);

        let next_id = self.state.alloc_next_node_id();
        let config = IngestionConfig {
            next_node_id: Some(next_id),
            ..IngestionConfig::default()
        };

        let ner_ref: Option<&dyn ucotron_extraction::NerPipeline> =
            self.state.ner.as_ref().map(|n| n.as_ref());
        let re_ref: Option<&dyn ucotron_extraction::RelationExtractor> =
            self.state.relation_extractor.as_ref().map(|r| r.as_ref());

        let mut orchestrator = IngestionOrchestrator::new(
            &self.state.registry,
            self.state.embedder.as_ref(),
            ner_ref,
            re_ref,
            config,
        );

        match orchestrator.ingest(&params.text) {
            Ok(result) => {
                let ids_used = result.chunk_node_ids.len() + result.entity_node_ids.len();
                {
                    let mut id_lock = self.state.next_node_id.lock().unwrap();
                    let used_max = next_id + ids_used as u64;
                    if used_max > *id_lock {
                        *id_lock = used_max;
                    }
                }

                let response = AddMemoryResult {
                    memories_created: result.chunk_node_ids.len(),
                    entities_found: result.entity_node_ids.len(),
                };
                let json = serde_json::to_string_pretty(&response).unwrap_or_default();
                Ok(CallToolResult::success(vec![Content::text(json)]))
            }
            Err(e) => Ok(CallToolResult::error(vec![Content::text(format!(
                "Ingestion failed: {}",
                e
            ))])),
        }
    }

    /// Search for memories matching a natural language query.
    #[tool(
        name = "ucotron_search",
        description = "Search for relevant memories in the Ucotron cognitive graph using semantic similarity and graph-based ranking."
    )]
    async fn search(
        &self,
        params: Parameters<SearchParams>,
    ) -> Result<CallToolResult, McpError> {
        use ucotron_extraction::retrieval::{RetrievalConfig, RetrievalOrchestrator};
        use std::sync::atomic::Ordering;

        let params = params.0;

        self.state.total_searches.fetch_add(1, Ordering::Relaxed);

        let limit = params.limit.unwrap_or(10);
        let retrieval_config = RetrievalConfig {
            final_top_k: limit,
            ..RetrievalConfig::default()
        };

        let ner_ref: Option<&dyn ucotron_extraction::NerPipeline> =
            self.state.ner.as_ref().map(|n| n.as_ref());

        let mut orchestrator = RetrievalOrchestrator::new(
            &self.state.registry,
            self.state.embedder.as_ref(),
            ner_ref,
            retrieval_config,
        );
        if self.state.config.mindset.enabled {
            let alg: Vec<&str> = self.state.config.mindset.algorithmic_keywords.iter().map(|s| s.as_str()).collect();
            let div: Vec<&str> = self.state.config.mindset.divergent_keywords.iter().map(|s| s.as_str()).collect();
            let con: Vec<&str> = self.state.config.mindset.convergent_keywords.iter().map(|s| s.as_str()).collect();
            let spa: Vec<&str> = self.state.config.mindset.spatial_keywords.iter().map(|s| s.as_str()).collect();
            orchestrator = orchestrator.with_mindset_detector(
                ucotron_core::MindsetDetector::from_keyword_lists(&alg, &div, &con, &spa),
            );
        }

        match orchestrator.retrieve(&params.query) {
            Ok(result) => {
                let memories: Vec<MemoryResult> = result
                    .memories
                    .iter()
                    .map(|m| MemoryResult {
                        id: m.node.id,
                        content: m.node.content.clone(),
                        node_type: format!("{:?}", m.node.node_type),
                        score: Some(m.score),
                    })
                    .collect();

                let json = serde_json::to_string_pretty(&memories).unwrap_or_default();
                Ok(CallToolResult::success(vec![Content::text(json)]))
            }
            Err(e) => Ok(CallToolResult::error(vec![Content::text(format!(
                "Search failed: {}",
                e
            ))])),
        }
    }

    /// Get information about a named entity from the knowledge graph.
    #[tool(
        name = "ucotron_get_entity",
        description = "Look up a named entity in the Ucotron knowledge graph and return its information and relationships."
    )]
    async fn get_entity(
        &self,
        params: Parameters<GetEntityParams>,
    ) -> Result<CallToolResult, McpError> {
        let params = params.0;

        let embedding = match self.state.embedder.embed_text(&params.name) {
            Ok(e) => e,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Embedding failed: {}",
                    e
                ))]));
            }
        };

        let results = match self.state.registry.vector().search(&embedding, 10) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Vector search failed: {}",
                    e
                ))]));
            }
        };

        let name_lower = params.name.to_lowercase();
        for (node_id, _score) in &results {
            if let Ok(Some(node)) = self.state.registry.graph().get_node(*node_id) {
                if node.node_type == ucotron_core::NodeType::Entity
                    && node.content.to_lowercase().contains(&name_lower)
                {
                    let neighbors = self
                        .state
                        .registry
                        .graph()
                        .get_neighbors(*node_id, 1)
                        .unwrap_or_default();

                    let entity = EntityResult {
                        id: node.id,
                        name: node.content.clone(),
                        node_type: format!("{:?}", node.node_type),
                        neighbors: neighbors.iter().map(|n| n.content.clone()).collect(),
                    };

                    let json = serde_json::to_string_pretty(&entity).unwrap_or_default();
                    return Ok(CallToolResult::success(vec![Content::text(json)]));
                }
            }
        }

        Ok(CallToolResult::success(vec![Content::text(format!(
            "Entity '{}' not found in the knowledge graph.",
            params.name
        ))]))
    }

    /// List entities from the knowledge graph.
    #[tool(
        name = "ucotron_list_entities",
        description = "List entities stored in the Ucotron knowledge graph, optionally filtered by type."
    )]
    async fn list_entities(
        &self,
        params: Parameters<ListEntitiesParams>,
    ) -> Result<CallToolResult, McpError> {
        let params = params.0;
        let limit = params.limit.unwrap_or(20);

        let query_vec = vec![0.0f32; 384];
        let results = match self.state.registry.vector().search(&query_vec, limit * 2) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Vector search failed: {}",
                    e
                ))]));
            }
        };

        let type_filter = params.entity_type.as_deref().and_then(|t| {
            match t.to_lowercase().as_str() {
                "entity" => Some(ucotron_core::NodeType::Entity),
                "event" => Some(ucotron_core::NodeType::Event),
                "fact" => Some(ucotron_core::NodeType::Fact),
                "skill" => Some(ucotron_core::NodeType::Skill),
                _ => None,
            }
        });

        let mut entities: Vec<EntityResult> = Vec::new();
        for (node_id, _score) in &results {
            if entities.len() >= limit {
                break;
            }
            if let Ok(Some(node)) = self.state.registry.graph().get_node(*node_id) {
                let matches = match &type_filter {
                    Some(nt) => node.node_type == *nt,
                    None => node.node_type == ucotron_core::NodeType::Entity,
                };
                if matches {
                    entities.push(EntityResult {
                        id: node.id,
                        name: node.content.clone(),
                        node_type: format!("{:?}", node.node_type),
                        neighbors: Vec::new(),
                    });
                }
            }
        }

        let json = serde_json::to_string_pretty(&entities).unwrap_or_default();
        Ok(CallToolResult::success(vec![Content::text(json)]))
    }

    /// Augment a context with relevant memories from the knowledge graph.
    #[tool(
        name = "ucotron_augment",
        description = "Retrieve relevant memories and entities to augment a given context. Returns structured context for LLM prompt injection."
    )]
    async fn augment(
        &self,
        params: Parameters<AugmentParams>,
    ) -> Result<CallToolResult, McpError> {
        use ucotron_extraction::retrieval::{RetrievalConfig, RetrievalOrchestrator};
        use std::sync::atomic::Ordering;

        let params = params.0;

        self.state.total_searches.fetch_add(1, Ordering::Relaxed);

        let limit = params.limit.unwrap_or(10);
        let retrieval_config = RetrievalConfig {
            final_top_k: limit,
            ..RetrievalConfig::default()
        };

        let ner_ref: Option<&dyn ucotron_extraction::NerPipeline> =
            self.state.ner.as_ref().map(|n| n.as_ref());

        let mut orchestrator = RetrievalOrchestrator::new(
            &self.state.registry,
            self.state.embedder.as_ref(),
            ner_ref,
            retrieval_config,
        );
        if self.state.config.mindset.enabled {
            let alg: Vec<&str> = self.state.config.mindset.algorithmic_keywords.iter().map(|s| s.as_str()).collect();
            let div: Vec<&str> = self.state.config.mindset.divergent_keywords.iter().map(|s| s.as_str()).collect();
            let con: Vec<&str> = self.state.config.mindset.convergent_keywords.iter().map(|s| s.as_str()).collect();
            let spa: Vec<&str> = self.state.config.mindset.spatial_keywords.iter().map(|s| s.as_str()).collect();
            orchestrator = orchestrator.with_mindset_detector(
                ucotron_core::MindsetDetector::from_keyword_lists(&alg, &div, &con, &spa),
            );
        }

        match orchestrator.retrieve(&params.context) {
            Ok(result) => {
                let augment_result = AugmentResult {
                    memories: result
                        .memories
                        .iter()
                        .map(|m| MemoryResult {
                            id: m.node.id,
                            content: m.node.content.clone(),
                            node_type: format!("{:?}", m.node.node_type),
                            score: Some(m.score),
                        })
                        .collect(),
                    entities: result.entities.iter().map(|n| n.content.clone()).collect(),
                    context_text: result.context_text,
                };

                let json = serde_json::to_string_pretty(&augment_result).unwrap_or_default();
                Ok(CallToolResult::success(vec![Content::text(json)]))
            }
            Err(e) => Ok(CallToolResult::error(vec![Content::text(format!(
                "Augmentation failed: {}",
                e
            ))])),
        }
    }

    /// Learn from agent output by extracting and storing memories.
    #[tool(
        name = "ucotron_learn",
        description = "Extract memories, entities, and relationships from agent output text and store them in the cognitive graph."
    )]
    async fn learn(
        &self,
        params: Parameters<LearnParams>,
    ) -> Result<CallToolResult, McpError> {
        use ucotron_extraction::ingestion::{IngestionConfig, IngestionOrchestrator};
        use std::sync::atomic::Ordering;

        let params = params.0;

        if params.output.trim().is_empty() {
            return Ok(CallToolResult::error(vec![Content::text(
                "Error: output must not be empty",
            )]));
        }

        self.state.total_ingestions.fetch_add(1, Ordering::Relaxed);

        let next_id = self.state.alloc_next_node_id();
        let config = IngestionConfig {
            next_node_id: Some(next_id),
            ..IngestionConfig::default()
        };

        let ner_ref: Option<&dyn ucotron_extraction::NerPipeline> =
            self.state.ner.as_ref().map(|n| n.as_ref());
        let re_ref: Option<&dyn ucotron_extraction::RelationExtractor> =
            self.state.relation_extractor.as_ref().map(|r| r.as_ref());

        let mut orchestrator = IngestionOrchestrator::new(
            &self.state.registry,
            self.state.embedder.as_ref(),
            ner_ref,
            re_ref,
            config,
        );

        match orchestrator.ingest(&params.output) {
            Ok(result) => {
                let ids_used = result.chunk_node_ids.len() + result.entity_node_ids.len();
                {
                    let mut id_lock = self.state.next_node_id.lock().unwrap();
                    let used_max = next_id + ids_used as u64;
                    if used_max > *id_lock {
                        *id_lock = used_max;
                    }
                }

                let learn_result = LearnResult {
                    memories_created: result.chunk_node_ids.len(),
                    entities_found: result.entity_node_ids.len(),
                    conflicts_found: result.metrics.contradictions_detected,
                };

                let json = serde_json::to_string_pretty(&learn_result).unwrap_or_default();
                Ok(CallToolResult::success(vec![Content::text(json)]))
            }
            Err(e) => Ok(CallToolResult::error(vec![Content::text(format!(
                "Learn/ingestion failed: {}",
                e
            ))])),
        }
    }
}

/// Implement the MCP ServerHandler trait for UcotronMcpServer.
#[tool_handler]
impl ServerHandler for UcotronMcpServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            protocol_version: ProtocolVersion::V_2024_11_05,
            capabilities: ServerCapabilities::builder()
                .enable_tools()
                .build(),
            server_info: Implementation {
                name: "ucotron-server".to_string(),
                title: Some("Ucotron MCP Server".to_string()),
                version: env!("CARGO_PKG_VERSION").to_string(),
                description: None,
                icons: None,
                website_url: None,
            },
            instructions: Some(
                "Ucotron is a cognitive memory server for LLMs. Use the tools to store, \
                 search, and retrieve memories from a knowledge graph with semantic similarity \
                 and entity-relationship awareness."
                    .to_string(),
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::sync::Mutex;

    use ucotron_core::{BackendRegistry, Node, VectorBackend, GraphBackend};

    // -- Mock backends for testing --

    struct MockVectorBackend {
        embeddings: Mutex<HashMap<u64, Vec<f32>>>,
    }

    impl MockVectorBackend {
        fn new() -> Self {
            Self {
                embeddings: Mutex::new(HashMap::new()),
            }
        }
    }

    impl VectorBackend for MockVectorBackend {
        fn upsert_embeddings(&self, items: &[(u64, Vec<f32>)]) -> anyhow::Result<()> {
            let mut store = self.embeddings.lock().unwrap();
            for (id, emb) in items {
                store.insert(*id, emb.clone());
            }
            Ok(())
        }

        fn search(&self, _query: &[f32], top_k: usize) -> anyhow::Result<Vec<(u64, f32)>> {
            let store = self.embeddings.lock().unwrap();
            let mut results: Vec<(u64, f32)> = store.keys().map(|id| (*id, 0.5)).collect();
            results.sort_by_key(|(id, _)| *id);
            results.truncate(top_k);
            Ok(results)
        }

        fn delete(&self, ids: &[u64]) -> anyhow::Result<()> {
            let mut store = self.embeddings.lock().unwrap();
            for id in ids {
                store.remove(id);
            }
            Ok(())
        }
    }

    struct MockGraphBackend {
        nodes: Mutex<HashMap<u64, Node>>,
    }

    impl MockGraphBackend {
        fn new() -> Self {
            Self {
                nodes: Mutex::new(HashMap::new()),
            }
        }
    }

    impl GraphBackend for MockGraphBackend {
        fn upsert_nodes(&self, nodes: &[Node]) -> anyhow::Result<()> {
            let mut store = self.nodes.lock().unwrap();
            for node in nodes {
                store.insert(node.id, node.clone());
            }
            Ok(())
        }

        fn upsert_edges(&self, _edges: &[ucotron_core::Edge]) -> anyhow::Result<()> {
            Ok(())
        }

        fn get_node(&self, id: u64) -> anyhow::Result<Option<Node>> {
            let store = self.nodes.lock().unwrap();
            Ok(store.get(&id).cloned())
        }

        fn get_neighbors(&self, _id: u64, _hops: u8) -> anyhow::Result<Vec<Node>> {
            Ok(Vec::new())
        }

        fn find_path(
            &self,
            _source: u64,
            _target: u64,
            _max_depth: u32,
        ) -> anyhow::Result<Option<Vec<u64>>> {
            Ok(None)
        }

        fn get_community(&self, _node_id: u64) -> anyhow::Result<Vec<u64>> {
            Ok(Vec::new())
        }

        fn get_all_nodes(&self) -> anyhow::Result<Vec<Node>> {
            let map = self.nodes.lock().unwrap();
            Ok(map.values().cloned().collect())
        }

        fn get_all_edges(&self) -> anyhow::Result<Vec<(u64, u64, f32)>> {
            Ok(Vec::new())
        }

        fn delete_nodes(&self, ids: &[u64]) -> anyhow::Result<()> {
            let mut nodes = self.nodes.lock().unwrap();
            for id in ids {
                nodes.remove(id);
            }
            Ok(())
        }

        fn store_community_assignments(
            &self,
            _assignments: &HashMap<u64, u64>,
        ) -> anyhow::Result<()> {
            Ok(())
        }
    }

    // -- Mock embedder --

    struct MockEmbedder;
    impl ucotron_extraction::EmbeddingPipeline for MockEmbedder {
        fn embed_text(&self, _text: &str) -> anyhow::Result<Vec<f32>> {
            Ok(vec![0.1f32; 384])
        }
        fn embed_batch(&self, texts: &[&str]) -> anyhow::Result<Vec<Vec<f32>>> {
            Ok(texts.iter().map(|_| vec![0.1f32; 384]).collect())
        }
    }

    /// Extract text from the first Content item in a CallToolResult.
    fn extract_text(result: &CallToolResult) -> String {
        result.content.first().map(|c| {
            match &c.raw {
                RawContent::Text(t) => t.text.clone(),
                _ => String::new(),
            }
        }).unwrap_or_default()
    }

    fn build_mcp_server() -> UcotronMcpServer {
        let vector = Box::new(MockVectorBackend::new()) as Box<dyn VectorBackend>;
        let graph = Box::new(MockGraphBackend::new()) as Box<dyn GraphBackend>;
        let registry = Arc::new(BackendRegistry::new(vector, graph));
        let embedder = Arc::new(MockEmbedder) as Arc<dyn ucotron_extraction::EmbeddingPipeline>;
        let config = ucotron_config::UcotronConfig::default();

        let state = Arc::new(AppState::new(registry, embedder, None, None, config, None, None, None, None));
        UcotronMcpServer::new(state)
    }

    #[tokio::test]
    async fn test_add_memory_empty_text() {
        let server = build_mcp_server();
        let params = AddMemoryParams {
            text: "  ".to_string(),
        };
        let result = server.add_memory(Parameters(params)).await.unwrap();
        assert!(result.is_error.unwrap_or(false));
    }

    #[tokio::test]
    async fn test_add_memory_success() {
        let server = build_mcp_server();
        let params = AddMemoryParams {
            text: "Juan moved from Madrid to Berlin.".to_string(),
        };
        let result = server.add_memory(Parameters(params)).await.unwrap();
        assert!(!result.is_error.unwrap_or(false));
        let text = extract_text(&result);
        assert!(text.contains("memories_created"));
    }

    #[tokio::test]
    async fn test_search_empty_db() {
        let server = build_mcp_server();
        let params = SearchParams {
            query: "where does Juan live?".to_string(),
            limit: Some(5),
        };
        let result = server.search(Parameters(params)).await.unwrap();
        assert!(!result.is_error.unwrap_or(false));
    }

    #[tokio::test]
    async fn test_get_entity_not_found() {
        let server = build_mcp_server();
        let params = GetEntityParams {
            name: "nonexistent".to_string(),
        };
        let result = server.get_entity(Parameters(params)).await.unwrap();
        let text = extract_text(&result);
        assert!(text.contains("not found"));
    }

    #[tokio::test]
    async fn test_list_entities_empty() {
        let server = build_mcp_server();
        let params = ListEntitiesParams {
            entity_type: None,
            limit: Some(10),
        };
        let result = server.list_entities(Parameters(params)).await.unwrap();
        assert!(!result.is_error.unwrap_or(false));
    }

    #[tokio::test]
    async fn test_augment_empty_context() {
        let server = build_mcp_server();
        let params = AugmentParams {
            context: "Tell me about Juan".to_string(),
            limit: Some(5),
        };
        let result = server.augment(Parameters(params)).await.unwrap();
        assert!(!result.is_error.unwrap_or(false));
    }

    #[tokio::test]
    async fn test_learn_empty_output() {
        let server = build_mcp_server();
        let params = LearnParams {
            output: "".to_string(),
        };
        let result = server.learn(Parameters(params)).await.unwrap();
        assert!(result.is_error.unwrap_or(false));
    }

    #[tokio::test]
    async fn test_learn_success() {
        let server = build_mcp_server();
        let params = LearnParams {
            output: "The capital of France is Paris. Marie Curie was born in Warsaw.".to_string(),
        };
        let result = server.learn(Parameters(params)).await.unwrap();
        assert!(!result.is_error.unwrap_or(false));
        let text = extract_text(&result);
        assert!(text.contains("memories_created"));
    }

    #[test]
    fn test_server_info() {
        let server = build_mcp_server();
        let info = server.get_info();
        assert_eq!(info.server_info.name, "ucotron-server");
        assert!(info.instructions.is_some());
        assert!(info.instructions.unwrap().contains("cognitive memory"));
    }

    #[test]
    fn test_mcp_server_is_clone() {
        let server = build_mcp_server();
        let _clone = server.clone();
    }
}
