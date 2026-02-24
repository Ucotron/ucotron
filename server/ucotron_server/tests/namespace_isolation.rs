//! Comprehensive namespace isolation test suite.
//!
//! Verifies that multi-tenant namespace boundaries are enforced across:
//! - Augment responses (memories array AND context_text)
//! - Search responses
//! - Audit log namespace capture
//!
//! These tests complement the existing api_tests.rs namespace tests by
//! providing focused, thorough coverage of isolation guarantees.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use axum::body::Body;
use axum::http::{Request, StatusCode};
use axum::middleware;
use axum::routing::{get, post};
use axum::Router;
use http_body_util::BodyExt;
use tower::ServiceExt;

use ucotron_config::{ApiKeyEntry, UcotronConfig};
use ucotron_core::{BackendRegistry, Edge, Node, NodeId, NodeType};
use ucotron_extraction::EmbeddingPipeline;
use ucotron_server::handlers;
use ucotron_server::state::AppState;

// ---------------------------------------------------------------------------
// Mock Backends (duplicated from api_tests.rs — Rust integration tests are
// separate compilation units and cannot share non-lib code)
// ---------------------------------------------------------------------------

struct MockEmbedder;

impl EmbeddingPipeline for MockEmbedder {
    fn embed_text(&self, text: &str) -> anyhow::Result<Vec<f32>> {
        let hash = text.len() as f32 / 100.0;
        let mut v = vec![hash; 384];
        v[0] = 1.0;
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in &mut v {
                *x /= norm;
            }
        }
        Ok(v)
    }

    fn embed_batch(&self, texts: &[&str]) -> anyhow::Result<Vec<Vec<f32>>> {
        texts.iter().map(|t| self.embed_text(t)).collect()
    }
}

struct MockVectorBackend {
    embeddings: Mutex<HashMap<NodeId, Vec<f32>>>,
}

impl MockVectorBackend {
    fn new() -> Self {
        Self {
            embeddings: Mutex::new(HashMap::new()),
        }
    }
}

impl ucotron_core::VectorBackend for MockVectorBackend {
    fn upsert_embeddings(&self, items: &[(NodeId, Vec<f32>)]) -> anyhow::Result<()> {
        let mut store = self.embeddings.lock().unwrap();
        for (id, emb) in items {
            store.insert(*id, emb.clone());
        }
        Ok(())
    }

    fn search(&self, query: &[f32], top_k: usize) -> anyhow::Result<Vec<(NodeId, f32)>> {
        let store = self.embeddings.lock().unwrap();
        let mut results: Vec<(NodeId, f32)> = store
            .iter()
            .map(|(id, emb)| {
                let sim: f32 = query.iter().zip(emb.iter()).map(|(a, b)| a * b).sum();
                (*id, sim)
            })
            .collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(top_k);
        Ok(results)
    }

    fn delete(&self, ids: &[NodeId]) -> anyhow::Result<()> {
        let mut store = self.embeddings.lock().unwrap();
        for id in ids {
            store.remove(id);
        }
        Ok(())
    }
}

struct MockGraphBackend {
    nodes: Mutex<HashMap<NodeId, Node>>,
    edges: Mutex<Vec<Edge>>,
    agents: Mutex<HashMap<String, ucotron_core::Agent>>,
    shares: Mutex<HashMap<(String, String), ucotron_core::AgentShare>>,
}

impl MockGraphBackend {
    fn new() -> Self {
        Self {
            nodes: Mutex::new(HashMap::new()),
            edges: Mutex::new(Vec::new()),
            agents: Mutex::new(HashMap::new()),
            shares: Mutex::new(HashMap::new()),
        }
    }
}

impl ucotron_core::GraphBackend for MockGraphBackend {
    fn upsert_nodes(&self, nodes: &[Node]) -> anyhow::Result<()> {
        let mut store = self.nodes.lock().unwrap();
        for node in nodes {
            store.insert(node.id, node.clone());
        }
        Ok(())
    }

    fn upsert_edges(&self, edges: &[Edge]) -> anyhow::Result<()> {
        let mut store = self.edges.lock().unwrap();
        for edge in edges {
            store.push(edge.clone());
        }
        Ok(())
    }

    fn get_node(&self, id: NodeId) -> anyhow::Result<Option<Node>> {
        let store = self.nodes.lock().unwrap();
        Ok(store.get(&id).cloned())
    }

    fn get_neighbors(&self, id: NodeId, _hops: u8) -> anyhow::Result<Vec<Node>> {
        let nodes = self.nodes.lock().unwrap();
        let edges = self.edges.lock().unwrap();
        let mut result = Vec::new();
        for edge in edges.iter() {
            if edge.source == id {
                if let Some(n) = nodes.get(&edge.target) {
                    result.push(n.clone());
                }
            } else if edge.target == id {
                if let Some(n) = nodes.get(&edge.source) {
                    result.push(n.clone());
                }
            }
        }
        Ok(result)
    }

    fn find_path(
        &self,
        _source: NodeId,
        _target: NodeId,
        _max_depth: u32,
    ) -> anyhow::Result<Option<Vec<NodeId>>> {
        Ok(None)
    }

    fn get_community(&self, _node_id: NodeId) -> anyhow::Result<Vec<NodeId>> {
        Ok(Vec::new())
    }

    fn get_all_nodes(&self) -> anyhow::Result<Vec<Node>> {
        let store = self.nodes.lock().unwrap();
        Ok(store.values().cloned().collect())
    }

    fn get_all_edges(&self) -> anyhow::Result<Vec<(NodeId, NodeId, f32)>> {
        let store = self.edges.lock().unwrap();
        Ok(store.iter().map(|e| (e.source, e.target, e.weight)).collect())
    }

    fn get_all_edges_full(&self) -> anyhow::Result<Vec<Edge>> {
        let store = self.edges.lock().unwrap();
        Ok(store.clone())
    }

    fn delete_nodes(&self, ids: &[NodeId]) -> anyhow::Result<()> {
        let mut nodes = self.nodes.lock().unwrap();
        let mut edges = self.edges.lock().unwrap();
        let id_set: std::collections::HashSet<NodeId> = ids.iter().copied().collect();
        nodes.retain(|id, _| !id_set.contains(id));
        edges.retain(|e| !id_set.contains(&e.source) && !id_set.contains(&e.target));
        Ok(())
    }

    fn store_community_assignments(
        &self,
        _assignments: &HashMap<NodeId, ucotron_core::community::CommunityId>,
    ) -> anyhow::Result<()> {
        Ok(())
    }

    fn create_agent(&self, agent: &ucotron_core::Agent) -> anyhow::Result<()> {
        let mut store = self.agents.lock().unwrap();
        store.insert(agent.id.clone(), agent.clone());
        Ok(())
    }

    fn get_agent(&self, id: &str) -> anyhow::Result<Option<ucotron_core::Agent>> {
        let store = self.agents.lock().unwrap();
        Ok(store.get(id).cloned())
    }

    fn list_agents(&self, owner: Option<&str>) -> anyhow::Result<Vec<ucotron_core::Agent>> {
        let store = self.agents.lock().unwrap();
        let agents: Vec<ucotron_core::Agent> = if let Some(o) = owner {
            store.values().filter(|a| a.owner == o).cloned().collect()
        } else {
            store.values().cloned().collect()
        };
        Ok(agents)
    }

    fn delete_agent(&self, id: &str) -> anyhow::Result<()> {
        let mut agents = self.agents.lock().unwrap();
        agents.remove(id);
        let mut shares = self.shares.lock().unwrap();
        shares.retain(|(a, t), _| a != id && t != id);
        Ok(())
    }

    fn create_share(&self, share: &ucotron_core::AgentShare) -> anyhow::Result<()> {
        let mut store = self.shares.lock().unwrap();
        let key = (share.agent_id.clone(), share.target_agent_id.clone());
        store.insert(key, share.clone());
        Ok(())
    }

    fn get_share(
        &self,
        agent_id: &str,
        target_id: &str,
    ) -> anyhow::Result<Option<ucotron_core::AgentShare>> {
        let store = self.shares.lock().unwrap();
        let key = (agent_id.to_string(), target_id.to_string());
        Ok(store.get(&key).cloned())
    }

    fn list_shares(&self, agent_id: &str) -> anyhow::Result<Vec<ucotron_core::AgentShare>> {
        let store = self.shares.lock().unwrap();
        Ok(store
            .values()
            .filter(|s| s.agent_id == agent_id)
            .cloned()
            .collect())
    }

    fn delete_share(&self, agent_id: &str, target_id: &str) -> anyhow::Result<()> {
        let mut store = self.shares.lock().unwrap();
        let key = (agent_id.to_string(), target_id.to_string());
        store.remove(&key);
        Ok(())
    }

    fn clone_graph(
        &self,
        _src_ns: &str,
        _dst_ns: &str,
        _filter: &ucotron_core::CloneFilter,
        _id_start: u64,
    ) -> anyhow::Result<ucotron_core::CloneResult> {
        Ok(ucotron_core::CloneResult::default())
    }

    fn merge_graph(
        &self,
        _src_ns: &str,
        _dst_ns: &str,
        _id_start: u64,
    ) -> anyhow::Result<ucotron_core::MergeResult> {
        Ok(ucotron_core::MergeResult::default())
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build a minimal app with shared state. Auth middleware is layered so
/// `AuthContext` is always available (default config = auth disabled = all
/// requests pass through).
fn build_app() -> (Router, Arc<AppState>) {
    let registry = Arc::new(BackendRegistry::new(
        Box::new(MockVectorBackend::new()),
        Box::new(MockGraphBackend::new()),
    ));
    let embedder: Arc<dyn EmbeddingPipeline> = Arc::new(MockEmbedder);
    let config = UcotronConfig::default();
    let state = Arc::new(AppState::new(registry, embedder, None, None, config, None, None, None, None));

    let app = Router::new()
        .route("/api/v1/memories", post(handlers::create_memory_handler))
        .route("/api/v1/memories", get(handlers::list_memories_handler))
        .route("/api/v1/memories/search", post(handlers::search_handler))
        .route("/api/v1/augment", post(handlers::augment_handler))
        .layer(middleware::from_fn_with_state(
            state.clone(),
            ucotron_server::auth::auth_middleware,
        ))
        .with_state(state.clone());

    (app, state)
}

/// Wrap a router with auth middleware against the given shared state.
fn with_auth(router: Router<Arc<AppState>>, state: &Arc<AppState>) -> Router {
    router
        .layer(middleware::from_fn_with_state(
            state.clone(),
            ucotron_server::auth::auth_middleware,
        ))
        .with_state(state.clone())
}

/// Build an app with auth + audit middleware enabled, and pre-configured API keys.
fn build_app_with_audit() -> (Router, Arc<AppState>) {
    let registry = Arc::new(BackendRegistry::new(
        Box::new(MockVectorBackend::new()),
        Box::new(MockGraphBackend::new()),
    ));
    let embedder: Arc<dyn EmbeddingPipeline> = Arc::new(MockEmbedder);

    let mut config = UcotronConfig::default();
    config.auth.enabled = true;
    config.audit.enabled = true;
    config.audit.max_entries = 1000;
    config.audit.retention_secs = 0;
    config.auth.api_keys = vec![
        ApiKeyEntry {
            name: "admin-key".into(),
            key: "sk-admin-test".into(),
            role: "admin".into(),
            namespace: None,
            active: true,
        },
        ApiKeyEntry {
            name: "reader-key".into(),
            key: "sk-reader-test".into(),
            role: "reader".into(),
            namespace: None,
            active: true,
        },
    ];

    let state = Arc::new(AppState::new(registry, embedder, None, None, config, None, None, None, None));

    (Router::new(), state)
}

/// Insert a node with explicit namespace metadata into the shared state.
fn insert_namespaced_node(
    state: &AppState,
    id: NodeId,
    content: &str,
    namespace: &str,
) {
    let embedding = vec![0.5f32; 384];
    let mut metadata = HashMap::new();
    metadata.insert(
        "_namespace".to_string(),
        ucotron_core::Value::String(namespace.to_string()),
    );
    let node = Node {
        id,
        content: content.to_string(),
        embedding: embedding.clone(),
        metadata,
        node_type: NodeType::Event,
        timestamp: 1700000000 + id,
        media_type: None,
        media_uri: None,
        embedding_visual: None,
        timestamp_range: None,
        parent_video_id: None,
    };
    state.registry.graph().upsert_nodes(&[node]).unwrap();
    state
        .registry
        .vector()
        .upsert_embeddings(&[(id, embedding)])
        .unwrap();
}

async fn body_to_json(body: Body) -> serde_json::Value {
    let bytes = body.collect().await.unwrap().to_bytes();
    serde_json::from_slice(&bytes).unwrap()
}

// ===========================================================================
// Test 1: Augment namespace isolation — memories array
// ===========================================================================

/// Verify that the augment endpoint's `memories` array only contains nodes
/// from the requested namespace. Nodes from other namespaces must be excluded
/// even though they exist in the same backing store.
#[tokio::test]
async fn test_augment_namespace_isolation() {
    let (_, state) = build_app();

    // Insert nodes for two different tenants with similar content so both
    // would match a query about "database technology".
    insert_namespaced_node(&state, 100, "Acme Corp uses PostgreSQL for analytics", "acme");
    insert_namespaced_node(&state, 101, "Globex uses MySQL for analytics", "globex");
    insert_namespaced_node(&state, 102, "Acme Corp runs Redis for caching", "acme");

    // Query augment as tenant "acme"
    let app = with_auth(
        Router::new().route("/api/v1/augment", post(handlers::augment_handler)),
        &state,
    );
    let req = Request::post("/api/v1/augment")
        .header("Content-Type", "application/json")
        .header("X-Ucotron-Namespace", "acme")
        .body(Body::from(r#"{"context":"database analytics","limit":10}"#))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let body = body_to_json(resp.into_body()).await;
    let memories = body["memories"].as_array().unwrap();

    // All returned memories must belong to "acme", none to "globex"
    for mem in memories {
        let content = mem["content"].as_str().unwrap_or("");
        assert!(
            !content.contains("Globex"),
            "augment memories for 'acme' must not contain Globex data, got: {content}"
        );
        assert!(
            !content.contains("MySQL"),
            "augment memories for 'acme' must not contain MySQL (globex data), got: {content}"
        );
    }
}

// ===========================================================================
// Test 2: Search namespace isolation
// ===========================================================================

/// Verify that the search endpoint only returns nodes from the requested
/// namespace. Even if another namespace has highly relevant content, it must
/// not leak into search results.
#[tokio::test]
async fn test_search_namespace_isolation() {
    let (_, state) = build_app();

    // Insert nodes for two tenants with overlapping themes
    insert_namespaced_node(&state, 200, "Alpha project uses Kubernetes for deployment", "alpha");
    insert_namespaced_node(&state, 201, "Beta project uses Docker Swarm for deployment", "beta");
    insert_namespaced_node(&state, 202, "Alpha project migrated from Heroku to AWS", "alpha");

    // Search as tenant "beta" for "deployment"
    let app = with_auth(
        Router::new().route("/api/v1/memories/search", post(handlers::search_handler)),
        &state,
    );
    let req = Request::post("/api/v1/memories/search")
        .header("Content-Type", "application/json")
        .header("X-Ucotron-Namespace", "beta")
        .body(Body::from(r#"{"query":"deployment","limit":10}"#))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let body = body_to_json(resp.into_body()).await;
    let results = body["results"].as_array().unwrap();

    // No result should contain alpha-namespace content
    for r in results {
        let content = r["content"].as_str().unwrap_or("");
        assert!(
            !content.contains("Alpha"),
            "search results for 'beta' must not contain Alpha data, got: {content}"
        );
        assert!(
            !content.contains("Kubernetes"),
            "search results for 'beta' must not contain Kubernetes (alpha data), got: {content}"
        );
        assert!(
            !content.contains("Heroku"),
            "search results for 'beta' must not contain Heroku (alpha data), got: {content}"
        );
    }
}

// ===========================================================================
// Test 3: Context text namespace isolation (regression for BUG-1)
// ===========================================================================

/// Verify that the augment endpoint's `context_text` field — the plain-text
/// summary assembled for LLM consumption — only contains content from the
/// requested namespace. This is a regression test for BUG-1 where context_text
/// was built from ALL retrieved nodes regardless of namespace.
#[tokio::test]
async fn test_context_text_namespace_isolation() {
    let (_, state) = build_app();

    // Two tenants with clearly distinguishable content
    insert_namespaced_node(&state, 300, "Tenant-X secret: launch code is ALPHA-7", "tenant-x");
    insert_namespaced_node(&state, 301, "Tenant-Y secret: launch code is BRAVO-9", "tenant-y");

    // Augment as tenant-x
    let app = with_auth(
        Router::new().route("/api/v1/augment", post(handlers::augment_handler)),
        &state,
    );
    let req = Request::post("/api/v1/augment")
        .header("Content-Type", "application/json")
        .header("X-Ucotron-Namespace", "tenant-x")
        .body(Body::from(r#"{"context":"launch code","limit":10}"#))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let body = body_to_json(resp.into_body()).await;
    let context_text = body["context_text"].as_str().unwrap_or("");

    // context_text must NOT contain tenant-y's secret
    assert!(
        !context_text.contains("BRAVO-9"),
        "context_text for tenant-x must not contain tenant-y secret BRAVO-9"
    );
    assert!(
        !context_text.contains("Tenant-Y"),
        "context_text for tenant-x must not mention Tenant-Y"
    );

    // Now augment as tenant-y and verify the reverse
    let app2 = with_auth(
        Router::new().route("/api/v1/augment", post(handlers::augment_handler)),
        &state,
    );
    let req2 = Request::post("/api/v1/augment")
        .header("Content-Type", "application/json")
        .header("X-Ucotron-Namespace", "tenant-y")
        .body(Body::from(r#"{"context":"launch code","limit":10}"#))
        .unwrap();
    let resp2 = app2.oneshot(req2).await.unwrap();
    assert_eq!(resp2.status(), StatusCode::OK);

    let body2 = body_to_json(resp2.into_body()).await;
    let context_text2 = body2["context_text"].as_str().unwrap_or("");

    assert!(
        !context_text2.contains("ALPHA-7"),
        "context_text for tenant-y must not contain tenant-x secret ALPHA-7"
    );
    assert!(
        !context_text2.contains("Tenant-X"),
        "context_text for tenant-y must not mention Tenant-X"
    );
}

// ===========================================================================
// Test 4: Audit log namespace capture (regression for BUG-5)
// ===========================================================================

/// Verify that audit log entries correctly capture the namespace from the
/// X-Ucotron-Namespace request header. This is a regression test for BUG-5
/// where the namespace field was always null.
#[tokio::test]
async fn test_audit_namespace_capture() {
    let (_, state) = build_app_with_audit();

    // Build an app with BOTH audit and auth middleware layered correctly.
    // Auth runs first (outermost), then audit can read AuthContext.
    let app = Router::new()
        .route("/api/v1/memories", get(handlers::list_memories_handler))
        .layer(middleware::from_fn_with_state(
            state.clone(),
            ucotron_server::audit::audit_middleware,
        ))
        .layer(middleware::from_fn_with_state(
            state.clone(),
            ucotron_server::auth::auth_middleware,
        ))
        .with_state(state.clone());

    // Make a request with explicit namespace header
    let req = Request::get("/api/v1/memories")
        .header("Authorization", "Bearer sk-reader-test")
        .header("X-Ucotron-Namespace", "production-tenant")
        .body(Body::empty())
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    // Check audit log captured the namespace
    let entries = state.audit_log.export_all();
    let mem_entry = entries
        .iter()
        .find(|e| e.action == "memories.list")
        .expect("should have a memories.list audit entry");
    assert_eq!(
        mem_entry.namespace,
        Some("production-tenant".to_string()),
        "audit entry namespace should match X-Ucotron-Namespace header"
    );

    // Make a second request with a different namespace
    let req2 = Request::get("/api/v1/memories")
        .header("Authorization", "Bearer sk-reader-test")
        .header("X-Ucotron-Namespace", "staging-tenant")
        .body(Body::empty())
        .unwrap();
    let _resp2 = app.oneshot(req2).await.unwrap();

    let entries2 = state.audit_log.export_all();
    let staging_entry = entries2
        .iter()
        .filter(|e| e.action == "memories.list")
        .last()
        .expect("should have a second memories.list audit entry");
    assert_eq!(
        staging_entry.namespace,
        Some("staging-tenant".to_string()),
        "second audit entry should capture the staging-tenant namespace"
    );
}
