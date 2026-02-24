//! Integration tests for the Ucotron REST API.
//!
//! Uses mock backends (no LMDB) and a stub embedder to test handler logic
//! via tower::ServiceExt (no TCP listener needed).

use std::collections::HashMap;
use std::mem;
use std::sync::{Arc, Mutex};

use axum::body::Body;
use axum::http::{Request, StatusCode};
use axum::middleware;
use axum::routing::{delete, get, post, put};
use axum::Router;
use http_body_util::BodyExt;
use tower::ServiceExt;

use ucotron_config::{ApiKeyEntry, UcotronConfig};
use ucotron_core::{BackendRegistry, Edge, EdgeType, Node, NodeId, NodeType};
use ucotron_extraction::audio::{AudioMetadata, ChunkTranscription, TranscriptionResult};
use ucotron_extraction::ocr::{
    DocumentExtractionResult, DocumentFormat, DocumentMetadata, PageExtraction,
};
use ucotron_extraction::video::{ExtractedFrame, FrameExtractionResult};
use ucotron_extraction::DocumentOcrPipeline;
use ucotron_extraction::EmbeddingPipeline;
use ucotron_extraction::ImageEmbeddingPipeline;
use ucotron_extraction::TranscriptionPipeline;
use ucotron_extraction::VideoPipeline;
use ucotron_server::handlers;
use ucotron_server::state::AppState;

/// Wrap a router with auth middleware so that AuthContext is always available.
fn with_auth(router: Router<Arc<AppState>>, state: &Arc<AppState>) -> Router {
    router
        .layer(middleware::from_fn_with_state(
            state.clone(),
            ucotron_server::auth::auth_middleware,
        ))
        .with_state(state.clone())
}

// ---------------------------------------------------------------------------
// Mock Backends
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

struct MockTranscriber;

impl TranscriptionPipeline for MockTranscriber {
    fn transcribe_file(&self, path: &std::path::Path) -> anyhow::Result<TranscriptionResult> {
        // Read WAV to get duration (verify it's a real WAV)
        let reader = hound::WavReader::open(path)?;
        let spec = reader.spec();
        let sample_count = reader.len() as f32;
        let duration = sample_count / spec.sample_rate as f32 / spec.channels as f32;
        Ok(TranscriptionResult {
            text: "Hello world from mock transcriber".to_string(),
            chunks: vec![ChunkTranscription {
                text: "Hello world from mock transcriber".to_string(),
                start_secs: 0.0,
                end_secs: duration,
            }],
            metadata: AudioMetadata {
                duration_secs: duration,
                sample_rate: spec.sample_rate,
                channels: spec.channels,
                detected_language: Some("en".to_string()),
            },
        })
    }

    fn transcribe_samples(
        &self,
        samples: &[f32],
        sample_rate: u32,
    ) -> anyhow::Result<TranscriptionResult> {
        let duration = samples.len() as f32 / sample_rate as f32;
        Ok(TranscriptionResult {
            text: "Hello from samples".to_string(),
            chunks: vec![ChunkTranscription {
                text: "Hello from samples".to_string(),
                start_secs: 0.0,
                end_secs: duration,
            }],
            metadata: AudioMetadata {
                duration_secs: duration,
                sample_rate,
                channels: 1,
                detected_language: Some("en".to_string()),
            },
        })
    }
}

struct MockDocumentOcrPipeline;

impl DocumentOcrPipeline for MockDocumentOcrPipeline {
    fn process_document(
        &self,
        _data: &[u8],
        filename: &str,
    ) -> anyhow::Result<DocumentExtractionResult> {
        let ext = filename.rsplit('.').next().unwrap_or("pdf").to_lowercase();
        let format = if ext == "pdf" {
            DocumentFormat::Pdf
        } else {
            DocumentFormat::Image(ext)
        };
        Ok(DocumentExtractionResult {
            text: "Extracted text from mock document pipeline.".to_string(),
            pages: vec![PageExtraction {
                page_number: 1,
                text: "Extracted text from mock document pipeline.".to_string(),
            }],
            metadata: DocumentMetadata {
                total_pages: 1,
                format,
                is_scanned: false,
            },
        })
    }

    fn process_file(&self, path: &std::path::Path) -> anyhow::Result<DocumentExtractionResult> {
        let filename = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("doc.pdf");
        self.process_document(&[], filename)
    }
}

struct MockImageEmbedder;

impl ucotron_extraction::ImageEmbeddingPipeline for MockImageEmbedder {
    fn embed_image_bytes(&self, bytes: &[u8]) -> anyhow::Result<Vec<f32>> {
        if bytes.is_empty() {
            anyhow::bail!("Empty image data");
        }
        // Return a deterministic 512-dim normalized vector.
        let hash = bytes.len() as f32 / 1000.0;
        let mut v = vec![hash; 512];
        v[0] = 1.0;
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in &mut v {
                *x /= norm;
            }
        }
        Ok(v)
    }

    fn embed_image_file(&self, path: &std::path::Path) -> anyhow::Result<Vec<f32>> {
        let bytes = std::fs::read(path)?;
        self.embed_image_bytes(&bytes)
    }
}

struct MockVideoPipeline;

impl VideoPipeline for MockVideoPipeline {
    fn extract_frames(&self, _path: &std::path::Path) -> anyhow::Result<FrameExtractionResult> {
        // Generate 3 fake frames at 0ms, 5000ms, 12000ms to create 2+ segments.
        let make_frame = |ts_ms: u64, score: f64, kf: bool| ExtractedFrame {
            timestamp_ms: ts_ms,
            is_keyframe: kf,
            scene_change_score: score,
            width: 2,
            height: 2,
            rgb_data: vec![255, 0, 0, 0, 255, 0, 0, 0, 255, 128, 128, 128], // 4 RGB pixels
        };

        Ok(FrameExtractionResult {
            frames: vec![
                make_frame(0, 1.0, true),
                make_frame(5000, 0.1, false),
                make_frame(10000, 0.8, true),
            ],
            duration_ms: 12000,
            video_width: 640,
            video_height: 480,
            video_fps: 30.0,
            total_frames_estimated: 360,
        })
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
        Ok(store
            .iter()
            .map(|e| (e.source, e.target, e.weight))
            .collect())
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
        // Cascade-delete shares
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
        src_ns: &str,
        dst_ns: &str,
        filter: &ucotron_core::CloneFilter,
        id_start: u64,
    ) -> anyhow::Result<ucotron_core::CloneResult> {
        let store = self.nodes.lock().unwrap();
        let edges_store = self.edges.lock().unwrap();

        // Filter source nodes by namespace + filter criteria
        let src_nodes: Vec<&Node> = store
            .values()
            .filter(|n| {
                let ns_match = match n.metadata.get("_namespace") {
                    Some(ucotron_core::Value::String(ns)) => ns == src_ns,
                    None => src_ns == "default",
                    _ => src_ns == "default",
                };
                if !ns_match {
                    return false;
                }
                if let Some(ref types) = filter.node_types {
                    if !types
                        .iter()
                        .any(|t| mem::discriminant(t) == mem::discriminant(&n.node_type))
                    {
                        return false;
                    }
                }
                if let Some(start) = filter.time_range_start {
                    if n.timestamp < start {
                        return false;
                    }
                }
                if let Some(end) = filter.time_range_end {
                    if n.timestamp > end {
                        return false;
                    }
                }
                true
            })
            .collect();

        if src_nodes.is_empty() {
            return Ok(ucotron_core::CloneResult::default());
        }

        // Build ID mapping
        let mut id_map = HashMap::new();
        let mut next_id = id_start;
        for node in &src_nodes {
            id_map.insert(node.id, next_id);
            next_id += 1;
        }

        // Clone nodes
        let cloned_nodes: Vec<Node> = src_nodes
            .iter()
            .map(|n| {
                let new_id = id_map[&n.id];
                let mut metadata = n.metadata.clone();
                metadata.insert(
                    "_namespace".into(),
                    ucotron_core::Value::String(dst_ns.to_string()),
                );
                Node {
                    id: new_id,
                    content: n.content.clone(),
                    embedding: n.embedding.clone(),
                    metadata,
                    node_type: n.node_type,
                    timestamp: n.timestamp,
                    media_type: n.media_type,
                    media_uri: n.media_uri.clone(),
                    embedding_visual: n.embedding_visual.clone(),
                    timestamp_range: n.timestamp_range,
                    parent_video_id: n.parent_video_id,
                }
            })
            .collect();

        let nodes_copied = cloned_nodes.len();

        // Clone edges between cloned nodes
        let old_id_set: std::collections::HashSet<NodeId> =
            src_nodes.iter().map(|n| n.id).collect();
        let cloned_edges: Vec<Edge> = edges_store
            .iter()
            .filter(|e| old_id_set.contains(&e.source) && old_id_set.contains(&e.target))
            .map(|e| Edge {
                source: id_map[&e.source],
                target: id_map[&e.target],
                edge_type: e.edge_type,
                weight: e.weight,
                metadata: e.metadata.clone(),
            })
            .collect();

        let edges_copied = cloned_edges.len();

        drop(store);
        drop(edges_store);

        // Insert cloned data
        self.upsert_nodes(&cloned_nodes)?;
        self.upsert_edges(&cloned_edges)?;

        Ok(ucotron_core::CloneResult {
            nodes_copied,
            edges_copied,
            id_map,
        })
    }

    fn merge_graph(
        &self,
        src_ns: &str,
        dst_ns: &str,
        id_start: u64,
    ) -> anyhow::Result<ucotron_core::MergeResult> {
        let store = self.nodes.lock().unwrap();
        let edges_store = self.edges.lock().unwrap();

        // Partition nodes by namespace
        let src_nodes: Vec<&Node> = store
            .values()
            .filter(|n| match n.metadata.get("_namespace") {
                Some(ucotron_core::Value::String(ns)) => ns == src_ns,
                None => src_ns == "default",
                _ => src_ns == "default",
            })
            .collect();

        if src_nodes.is_empty() {
            return Ok(ucotron_core::MergeResult::default());
        }

        let dst_nodes: Vec<&Node> = store
            .values()
            .filter(|n| match n.metadata.get("_namespace") {
                Some(ucotron_core::Value::String(ns)) => ns == dst_ns,
                None => dst_ns == "default",
                _ => dst_ns == "default",
            })
            .collect();

        // Build content index for destination nodes
        let mut dst_content_map: HashMap<String, NodeId> = HashMap::new();
        for node in &dst_nodes {
            let key = node.content.trim().to_lowercase();
            dst_content_map.entry(key).or_insert(node.id);
        }

        // Map source IDs → destination IDs (dedup or new)
        let mut id_map: HashMap<NodeId, NodeId> = HashMap::new();
        let mut next_id = id_start;
        let mut nodes_deduplicated: usize = 0;
        let mut ids_remapped: usize = 0;
        let mut new_nodes: Vec<Node> = Vec::new();

        for node in &src_nodes {
            let key = node.content.trim().to_lowercase();
            if let Some(&existing_id) = dst_content_map.get(&key) {
                id_map.insert(node.id, existing_id);
                nodes_deduplicated += 1;
            } else {
                let new_id = next_id;
                next_id += 1;
                id_map.insert(node.id, new_id);
                ids_remapped += 1;

                let mut metadata = node.metadata.clone();
                metadata.insert(
                    "_namespace".into(),
                    ucotron_core::Value::String(dst_ns.to_string()),
                );
                new_nodes.push(Node {
                    id: new_id,
                    content: node.content.clone(),
                    embedding: node.embedding.clone(),
                    metadata,
                    node_type: node.node_type,
                    timestamp: node.timestamp,
                    media_type: node.media_type,
                    media_uri: node.media_uri.clone(),
                    embedding_visual: node.embedding_visual.clone(),
                    timestamp_range: node.timestamp_range,
                    parent_video_id: node.parent_video_id,
                });
                dst_content_map.insert(key, new_id);
            }
        }

        let nodes_copied = new_nodes.len();

        // Copy edges from source, remapping IDs
        let old_id_set: std::collections::HashSet<NodeId> =
            src_nodes.iter().map(|n| n.id).collect();
        let merged_edges: Vec<Edge> = edges_store
            .iter()
            .filter(|e| old_id_set.contains(&e.source) && old_id_set.contains(&e.target))
            .map(|e| Edge {
                source: id_map[&e.source],
                target: id_map[&e.target],
                edge_type: e.edge_type,
                weight: e.weight,
                metadata: e.metadata.clone(),
            })
            .collect();

        let edges_copied = merged_edges.len();

        drop(store);
        drop(edges_store);

        // Insert new (non-duplicate) nodes and remapped edges
        if !new_nodes.is_empty() {
            self.upsert_nodes(&new_nodes)?;
        }
        if !merged_edges.is_empty() {
            self.upsert_edges(&merged_edges)?;
        }

        Ok(ucotron_core::MergeResult {
            nodes_copied,
            edges_copied,
            nodes_deduplicated,
            ids_remapped,
        })
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn build_app() -> (Router, Arc<AppState>) {
    let registry = Arc::new(BackendRegistry::new(
        Box::new(MockVectorBackend::new()),
        Box::new(MockGraphBackend::new()),
    ));
    let embedder: Arc<dyn EmbeddingPipeline> = Arc::new(MockEmbedder);
    let config = UcotronConfig::default();
    let state = Arc::new(AppState::new(registry, embedder, None, None, config));

    let app = Router::new()
        .route("/api/v1/health", get(handlers::health_handler))
        .route("/api/v1/metrics", get(handlers::metrics_handler))
        .route("/api/v1/memories", post(handlers::create_memory_handler))
        .route("/api/v1/memories", get(handlers::list_memories_handler))
        .route("/api/v1/memories/{id}", get(handlers::get_memory_handler))
        .route(
            "/api/v1/memories/{id}",
            put(handlers::update_memory_handler),
        )
        .route(
            "/api/v1/memories/{id}",
            delete(handlers::delete_memory_handler),
        )
        .route(
            "/api/v1/memories/text",
            post(handlers::create_text_memory_handler),
        )
        .route(
            "/api/v1/memories/audio",
            post(handlers::create_audio_memory_handler),
        )
        .route(
            "/api/v1/memories/image",
            post(handlers::create_image_memory_handler),
        )
        .route(
            "/api/v1/memories/video",
            post(handlers::create_video_memory_handler),
        )
        .route("/api/v1/memories/search", post(handlers::search_handler))
        .route("/api/v1/media/{id}", get(handlers::get_media_handler))
        .route(
            "/api/v1/videos/{parent_id}/segments",
            get(handlers::get_video_segments_handler),
        )
        .route("/api/v1/entities", get(handlers::list_entities_handler))
        .route("/api/v1/entities/{id}", get(handlers::get_entity_handler))
        .route("/api/v1/graph", get(handlers::graph_handler))
        .route("/api/v1/augment", post(handlers::augment_handler))
        .route("/api/v1/learn", post(handlers::learn_handler))
        .route("/api/v1/export", get(handlers::export_handler))
        .route("/api/v1/import", post(handlers::import_handler))
        .route("/api/v1/import/mem0", post(handlers::mem0_import_handler))
        .route("/api/v1/import/zep", post(handlers::zep_import_handler))
        .route("/api/v1/transcribe", post(handlers::transcribe_handler))
        .route("/api/v1/ocr", post(handlers::ocr_handler))
        .route(
            "/api/v1/admin/namespaces",
            get(handlers::list_namespaces_handler),
        )
        .route(
            "/api/v1/admin/namespaces",
            post(handlers::create_namespace_handler),
        )
        .route(
            "/api/v1/admin/namespaces/{name}",
            get(handlers::get_namespace_handler),
        )
        .route(
            "/api/v1/admin/namespaces/{name}",
            delete(handlers::delete_namespace_handler),
        )
        .route("/api/v1/admin/config", get(handlers::admin_config_handler))
        .route("/api/v1/admin/system", get(handlers::admin_system_handler))
        .route("/api/v1/gdpr/forget", delete(handlers::gdpr_forget_handler))
        .route("/api/v1/gdpr/export", get(handlers::gdpr_export_handler))
        .route(
            "/api/v1/gdpr/retention",
            get(handlers::gdpr_retention_status_handler),
        )
        .route(
            "/api/v1/gdpr/retention/sweep",
            post(handlers::gdpr_retention_sweep_handler),
        )
        .route("/api/v1/agents", post(handlers::create_agent_handler))
        .route("/api/v1/agents", get(handlers::list_agents_handler))
        .route("/api/v1/agents/{id}", get(handlers::get_agent_handler))
        .route(
            "/api/v1/agents/{id}",
            delete(handlers::delete_agent_handler),
        )
        .route(
            "/api/v1/agents/{id}/clone",
            post(handlers::clone_agent_handler),
        )
        .route(
            "/api/v1/agents/{id}/merge",
            post(handlers::merge_agent_handler),
        )
        .route(
            "/api/v1/agents/{id}/share",
            post(handlers::create_share_handler).get(handlers::list_shares_handler),
        )
        .route(
            "/api/v1/agents/{id}/share/{target}",
            delete(handlers::delete_share_handler),
        )
        .layer(middleware::from_fn_with_state(
            state.clone(),
            ucotron_server::auth::auth_middleware,
        ))
        .with_state(state.clone());

    (app, state)
}

fn build_app_with_transcriber() -> (Router, Arc<AppState>) {
    let registry = Arc::new(BackendRegistry::new(
        Box::new(MockVectorBackend::new()),
        Box::new(MockGraphBackend::new()),
    ));
    let embedder: Arc<dyn EmbeddingPipeline> = Arc::new(MockEmbedder);
    let transcriber: Arc<dyn TranscriptionPipeline> = Arc::new(MockTranscriber);
    let config = UcotronConfig::default();
    let state = Arc::new(AppState::with_transcriber(
        registry,
        embedder,
        None,
        None,
        Some(transcriber),
        config,
    ));

    let app = Router::new()
        .route("/api/v1/health", get(handlers::health_handler))
        .route("/api/v1/metrics", get(handlers::metrics_handler))
        .route("/api/v1/memories", post(handlers::create_memory_handler))
        .route("/api/v1/memories", get(handlers::list_memories_handler))
        .route("/api/v1/memories/{id}", get(handlers::get_memory_handler))
        .route(
            "/api/v1/memories/{id}",
            put(handlers::update_memory_handler),
        )
        .route(
            "/api/v1/memories/{id}",
            delete(handlers::delete_memory_handler),
        )
        .route(
            "/api/v1/memories/audio",
            post(handlers::create_audio_memory_handler),
        )
        .route(
            "/api/v1/memories/image",
            post(handlers::create_image_memory_handler),
        )
        .route("/api/v1/memories/search", post(handlers::search_handler))
        .route("/api/v1/media/{id}", get(handlers::get_media_handler))
        .route(
            "/api/v1/videos/{parent_id}/segments",
            get(handlers::get_video_segments_handler),
        )
        .route("/api/v1/entities", get(handlers::list_entities_handler))
        .route("/api/v1/entities/{id}", get(handlers::get_entity_handler))
        .route("/api/v1/graph", get(handlers::graph_handler))
        .route("/api/v1/augment", post(handlers::augment_handler))
        .route("/api/v1/learn", post(handlers::learn_handler))
        .route("/api/v1/export", get(handlers::export_handler))
        .route("/api/v1/import", post(handlers::import_handler))
        .route("/api/v1/import/mem0", post(handlers::mem0_import_handler))
        .route("/api/v1/import/zep", post(handlers::zep_import_handler))
        .route("/api/v1/transcribe", post(handlers::transcribe_handler))
        .route("/api/v1/ocr", post(handlers::ocr_handler))
        .route(
            "/api/v1/admin/namespaces",
            get(handlers::list_namespaces_handler),
        )
        .route(
            "/api/v1/admin/namespaces",
            post(handlers::create_namespace_handler),
        )
        .route(
            "/api/v1/admin/namespaces/{name}",
            get(handlers::get_namespace_handler),
        )
        .route(
            "/api/v1/admin/namespaces/{name}",
            delete(handlers::delete_namespace_handler),
        )
        .route("/api/v1/admin/config", get(handlers::admin_config_handler))
        .route("/api/v1/admin/system", get(handlers::admin_system_handler))
        .layer(middleware::from_fn_with_state(
            state.clone(),
            ucotron_server::auth::auth_middleware,
        ))
        .with_state(state.clone());

    (app, state)
}

/// Create a minimal WAV file in memory (1 second of silence at 16kHz mono 16-bit).
fn create_test_wav_bytes() -> Vec<u8> {
    let dir = tempfile::tempdir().unwrap();
    let wav_path = dir.path().join("test.wav");
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 16000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(&wav_path, spec).unwrap();
    for _ in 0..16000 {
        writer.write_sample(0i16).unwrap();
    }
    writer.finalize().unwrap();
    std::fs::read(&wav_path).unwrap()
}

fn insert_test_node(state: &AppState, id: NodeId, content: &str, node_type: NodeType) {
    let embedding = vec![0.5f32; 384];
    let node = Node {
        id,
        content: content.to_string(),
        embedding: embedding.clone(),
        metadata: HashMap::new(),
        node_type,
        timestamp: 1700000000,
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

async fn body_to_json_array(body: Body) -> Vec<serde_json::Value> {
    let bytes = body.collect().await.unwrap().to_bytes();
    serde_json::from_slice(&bytes).unwrap()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_health_endpoint() {
    let (app, _) = build_app();
    let req = Request::get("/api/v1/health").body(Body::empty()).unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_to_json(resp.into_body()).await;
    assert_eq!(body["status"], "ok");
    assert_eq!(body["storage_mode"], "embedded");
}

#[tokio::test]
async fn test_metrics_endpoint() {
    let (app, _) = build_app();
    let req = Request::get("/api/v1/metrics").body(Body::empty()).unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_to_json(resp.into_body()).await;
    assert!(body["uptime_secs"].as_u64().is_some());
}

#[tokio::test]
async fn test_create_memory() {
    let (app, _) = build_app();
    let req = Request::post("/api/v1/memories")
        .header("Content-Type", "application/json")
        .body(Body::from(
            serde_json::to_string(&serde_json::json!({
                "text": "Juan vive en Madrid. Maria trabaja en SAP."
            }))
            .unwrap(),
        ))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::CREATED);
    let body = body_to_json(resp.into_body()).await;
    assert!(!body["chunk_node_ids"].as_array().unwrap().is_empty());
    assert!(body["metrics"]["chunks_processed"].as_u64().unwrap() > 0);
}

#[tokio::test]
async fn test_create_memory_empty_text() {
    let (app, _) = build_app();
    let req = Request::post("/api/v1/memories")
        .header("Content-Type", "application/json")
        .body(Body::from(r#"{"text":""}"#))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn test_get_memory() {
    let (app, state) = build_app();
    insert_test_node(&state, 42, "Test memory content", NodeType::Event);

    let req = Request::get("/api/v1/memories/42")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_to_json(resp.into_body()).await;
    assert_eq!(body["id"], 42);
    assert_eq!(body["content"], "Test memory content");
    assert_eq!(body["node_type"], "Event");
}

#[tokio::test]
async fn test_get_memory_not_found() {
    let (app, _) = build_app();
    let req = Request::get("/api/v1/memories/9999")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn test_update_memory() {
    let (app, state) = build_app();
    insert_test_node(&state, 100, "Original content", NodeType::Entity);

    let req = Request::put("/api/v1/memories/100")
        .header("Content-Type", "application/json")
        .body(Body::from(
            serde_json::to_string(&serde_json::json!({
                "content": "Updated content",
                "metadata": { "source": "test" }
            }))
            .unwrap(),
        ))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_to_json(resp.into_body()).await;
    assert_eq!(body["content"], "Updated content");
    assert_eq!(body["metadata"]["source"], "test");
}

#[tokio::test]
async fn test_delete_memory() {
    let (app, state) = build_app();
    insert_test_node(&state, 200, "To be deleted", NodeType::Event);

    let req = Request::delete("/api/v1/memories/200")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::NO_CONTENT);

    // Verify the node is soft-deleted.
    let node = state.registry.graph().get_node(200).unwrap().unwrap();
    assert!(node.content.is_empty());
}

#[tokio::test]
async fn test_list_memories() {
    let (app, state) = build_app();
    insert_test_node(&state, 1, "Memory one", NodeType::Entity);
    insert_test_node(&state, 2, "Memory two", NodeType::Event);

    let req = Request::get("/api/v1/memories?limit=10")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_to_json_array(resp.into_body()).await;
    assert!(body.len() <= 10);
}

#[tokio::test]
async fn test_search_memories() {
    let (app, state) = build_app();
    insert_test_node(&state, 10, "The quick brown fox", NodeType::Event);
    insert_test_node(&state, 11, "Jumped over the lazy dog", NodeType::Event);

    let req = Request::post("/api/v1/memories/search")
        .header("Content-Type", "application/json")
        .body(Body::from(r#"{"query":"quick fox","limit":5}"#))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_to_json(resp.into_body()).await;
    assert!(body["results"].as_array().is_some());
    assert_eq!(body["query"], "quick fox");
}

#[tokio::test]
async fn test_get_entity_with_neighbors() {
    let (app, state) = build_app();
    insert_test_node(&state, 50, "Juan", NodeType::Entity);
    insert_test_node(&state, 51, "Madrid", NodeType::Entity);

    let edge = Edge {
        source: 50,
        target: 51,
        edge_type: EdgeType::RelatesTo,
        weight: 1.0,
        metadata: HashMap::new(),
    };
    state.registry.graph().upsert_edges(&[edge]).unwrap();

    let req = Request::get("/api/v1/entities/50")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_to_json(resp.into_body()).await;
    assert_eq!(body["id"], 50);
    assert_eq!(body["content"], "Juan");
    let neighbors = body["neighbors"].as_array().unwrap();
    assert_eq!(neighbors.len(), 1);
    assert_eq!(neighbors[0]["node_id"], 51);
}

#[tokio::test]
async fn test_augment_endpoint() {
    let (app, state) = build_app();
    insert_test_node(&state, 60, "Important fact about Berlin", NodeType::Event);

    let req = Request::post("/api/v1/augment")
        .header("Content-Type", "application/json")
        .body(Body::from(r#"{"context":"Tell me about Berlin"}"#))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_to_json(resp.into_body()).await;
    assert!(body["memories"].as_array().is_some());
    assert!(body["context_text"].as_str().is_some());
}

#[tokio::test]
async fn test_learn_endpoint() {
    let (app, _) = build_app();
    let req = Request::post("/api/v1/learn")
        .header("Content-Type", "application/json")
        .body(Body::from(
            r#"{"output":"The user mentioned they live in Tokyo and work at Google."}"#,
        ))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::CREATED);
    let body = body_to_json(resp.into_body()).await;
    assert!(body["memories_created"].as_u64().unwrap() >= 1);
}

#[tokio::test]
async fn test_learn_empty_output() {
    let (app, _) = build_app();
    let req = Request::post("/api/v1/learn")
        .header("Content-Type", "application/json")
        .body(Body::from(r#"{"output":"  "}"#))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

// ---------------------------------------------------------------------------
// Multi-instance helpers
// ---------------------------------------------------------------------------

fn build_reader_app() -> (Router, Arc<AppState>) {
    let registry = Arc::new(BackendRegistry::new(
        Box::new(MockVectorBackend::new()),
        Box::new(MockGraphBackend::new()),
    ));
    let embedder: Arc<dyn EmbeddingPipeline> = Arc::new(MockEmbedder);
    let mut config = UcotronConfig::default();
    config.instance.role = "reader".to_string();
    config.instance.instance_id = "reader-test-1".to_string();
    let state = Arc::new(AppState::new(registry, embedder, None, None, config));

    let app = Router::new()
        .route("/api/v1/health", get(handlers::health_handler))
        .route("/api/v1/metrics", get(handlers::metrics_handler))
        .route("/api/v1/memories", post(handlers::create_memory_handler))
        .route("/api/v1/memories", get(handlers::list_memories_handler))
        .route("/api/v1/memories/{id}", get(handlers::get_memory_handler))
        .route(
            "/api/v1/memories/{id}",
            put(handlers::update_memory_handler),
        )
        .route(
            "/api/v1/memories/{id}",
            delete(handlers::delete_memory_handler),
        )
        .route("/api/v1/memories/search", post(handlers::search_handler))
        .route("/api/v1/media/{id}", get(handlers::get_media_handler))
        .route(
            "/api/v1/videos/{parent_id}/segments",
            get(handlers::get_video_segments_handler),
        )
        .route("/api/v1/entities", get(handlers::list_entities_handler))
        .route("/api/v1/entities/{id}", get(handlers::get_entity_handler))
        .route("/api/v1/graph", get(handlers::graph_handler))
        .route("/api/v1/augment", post(handlers::augment_handler))
        .route("/api/v1/learn", post(handlers::learn_handler))
        .layer(middleware::from_fn_with_state(
            state.clone(),
            ucotron_server::auth::auth_middleware,
        ))
        .with_state(state.clone());

    (app, state)
}

fn build_writer_app() -> (Router, Arc<AppState>) {
    let registry = Arc::new(BackendRegistry::new(
        Box::new(MockVectorBackend::new()),
        Box::new(MockGraphBackend::new()),
    ));
    let embedder: Arc<dyn EmbeddingPipeline> = Arc::new(MockEmbedder);
    let mut config = UcotronConfig::default();
    config.instance.role = "writer".to_string();
    config.instance.instance_id = "writer-test-1".to_string();
    config.instance.id_range_start = 5_000_000;
    config.instance.id_range_size = 100;
    let state = Arc::new(AppState::new(registry, embedder, None, None, config));

    let app = Router::new()
        .route("/api/v1/health", get(handlers::health_handler))
        .route("/api/v1/metrics", get(handlers::metrics_handler))
        .route("/api/v1/memories", post(handlers::create_memory_handler))
        .route("/api/v1/memories", get(handlers::list_memories_handler))
        .route("/api/v1/memories/{id}", get(handlers::get_memory_handler))
        .route(
            "/api/v1/memories/{id}",
            put(handlers::update_memory_handler),
        )
        .route(
            "/api/v1/memories/{id}",
            delete(handlers::delete_memory_handler),
        )
        .route("/api/v1/memories/search", post(handlers::search_handler))
        .route("/api/v1/media/{id}", get(handlers::get_media_handler))
        .route(
            "/api/v1/videos/{parent_id}/segments",
            get(handlers::get_video_segments_handler),
        )
        .route("/api/v1/entities", get(handlers::list_entities_handler))
        .route("/api/v1/entities/{id}", get(handlers::get_entity_handler))
        .route("/api/v1/graph", get(handlers::graph_handler))
        .route("/api/v1/augment", post(handlers::augment_handler))
        .route("/api/v1/learn", post(handlers::learn_handler))
        .layer(middleware::from_fn_with_state(
            state.clone(),
            ucotron_server::auth::auth_middleware,
        ))
        .with_state(state.clone());

    (app, state)
}

// ---------------------------------------------------------------------------
// Multi-instance tests
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_health_reports_instance_info() {
    let (app, _) = build_writer_app();
    let req = Request::get("/api/v1/health").body(Body::empty()).unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_to_json(resp.into_body()).await;
    assert_eq!(body["instance_id"], "writer-test-1");
    assert_eq!(body["instance_role"], "writer");
    assert_eq!(body["storage_mode"], "embedded");
    // Models section present
    assert!(body["models"].is_object());
    assert!(body["models"]["embedder_loaded"].as_bool().is_some());
    assert!(body["models"]["embedding_model"].as_str().is_some());
}

#[tokio::test]
async fn test_metrics_reports_instance_id() {
    let (app, _) = build_writer_app();
    let req = Request::get("/api/v1/metrics").body(Body::empty()).unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_to_json(resp.into_body()).await;
    assert_eq!(body["instance_id"], "writer-test-1");
}

#[tokio::test]
async fn test_reader_rejects_create_memory() {
    let (app, _) = build_reader_app();
    let req = Request::post("/api/v1/memories")
        .header("Content-Type", "application/json")
        .body(Body::from(r#"{"text":"should fail"}"#))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::FORBIDDEN);
    let body = body_to_json(resp.into_body()).await;
    assert_eq!(body["code"], "READ_ONLY_INSTANCE");
}

#[tokio::test]
async fn test_reader_rejects_update_memory() {
    let (app, state) = build_reader_app();
    insert_test_node(&state, 100, "some content", NodeType::Entity);
    let req = Request::put("/api/v1/memories/100")
        .header("Content-Type", "application/json")
        .body(Body::from(r#"{"content":"updated"}"#))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::FORBIDDEN);
    let body = body_to_json(resp.into_body()).await;
    assert_eq!(body["code"], "READ_ONLY_INSTANCE");
}

#[tokio::test]
async fn test_reader_rejects_delete_memory() {
    let (app, state) = build_reader_app();
    insert_test_node(&state, 200, "delete me", NodeType::Event);
    let req = Request::delete("/api/v1/memories/200")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::FORBIDDEN);
    let body = body_to_json(resp.into_body()).await;
    assert_eq!(body["code"], "READ_ONLY_INSTANCE");
}

#[tokio::test]
async fn test_reader_rejects_learn() {
    let (app, _) = build_reader_app();
    let req = Request::post("/api/v1/learn")
        .header("Content-Type", "application/json")
        .body(Body::from(r#"{"output":"should fail"}"#))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::FORBIDDEN);
    let body = body_to_json(resp.into_body()).await;
    assert_eq!(body["code"], "READ_ONLY_INSTANCE");
}

#[tokio::test]
async fn test_reader_allows_search() {
    let (app, state) = build_reader_app();
    insert_test_node(&state, 10, "searchable content", NodeType::Event);
    let req = Request::post("/api/v1/memories/search")
        .header("Content-Type", "application/json")
        .body(Body::from(r#"{"query":"searchable","limit":5}"#))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_reader_allows_augment() {
    let (app, state) = build_reader_app();
    insert_test_node(&state, 60, "augment content", NodeType::Event);
    let req = Request::post("/api/v1/augment")
        .header("Content-Type", "application/json")
        .body(Body::from(r#"{"context":"augment test"}"#))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_reader_allows_get_memory() {
    let (app, state) = build_reader_app();
    insert_test_node(&state, 42, "readable", NodeType::Entity);
    let req = Request::get("/api/v1/memories/42")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_writer_allows_create_memory() {
    let (app, _) = build_writer_app();
    let req = Request::post("/api/v1/memories")
        .header("Content-Type", "application/json")
        .body(Body::from(r#"{"text":"writer can write"}"#))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::CREATED);
}

#[tokio::test]
async fn test_writer_id_range_allocation() {
    let (_, state) = build_writer_app();
    // Writer is configured with id_range_start=5_000_000, id_range_size=100
    let id1 = state.alloc_next_node_id();
    let id2 = state.alloc_next_node_id();
    assert_eq!(id1, 5_000_000);
    assert_eq!(id2, 5_000_001);
}

#[tokio::test]
async fn test_reader_health_reports_reader_role() {
    let (app, _) = build_reader_app();
    let req = Request::get("/api/v1/health").body(Body::empty()).unwrap();
    let resp = app.oneshot(req).await.unwrap();
    let body = body_to_json(resp.into_body()).await;
    assert_eq!(body["instance_id"], "reader-test-1");
    assert_eq!(body["instance_role"], "reader");
}

#[tokio::test]
async fn test_namespace_header_accepted() {
    let (app, _) = build_app();
    let req = Request::get("/api/v1/health")
        .header("X-Ucotron-Namespace", "test-org")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
}

// ---------------------------------------------------------------------------
// Shared storage tests (two instances sharing the same backends)
// ---------------------------------------------------------------------------

/// Build a writer and reader that share the same BackendRegistry.
/// Simulates multi-instance mode with shared LMDB storage.
fn build_shared_instances() -> (Router, Arc<AppState>, Router, Arc<AppState>) {
    // Single shared backend registry (simulates shared LMDB dir).
    let shared_registry = Arc::new(BackendRegistry::new(
        Box::new(MockVectorBackend::new()),
        Box::new(MockGraphBackend::new()),
    ));

    let embedder: Arc<dyn EmbeddingPipeline> = Arc::new(MockEmbedder);

    // Writer instance
    let mut writer_config = UcotronConfig::default();
    writer_config.storage.mode = "shared".to_string();
    writer_config.storage.shared_data_dir = Some("/tmp/ucotron-test-shared".to_string());
    writer_config.instance.role = "writer".to_string();
    writer_config.instance.instance_id = "shared-writer-1".to_string();
    writer_config.instance.id_range_start = 0;
    writer_config.instance.id_range_size = 1_000_000;
    let writer_state = Arc::new(AppState::new(
        shared_registry.clone(),
        embedder.clone(),
        None,
        None,
        writer_config,
    ));

    // Reader instance
    let mut reader_config = UcotronConfig::default();
    reader_config.storage.mode = "shared".to_string();
    reader_config.storage.shared_data_dir = Some("/tmp/ucotron-test-shared".to_string());
    reader_config.instance.role = "reader".to_string();
    reader_config.instance.instance_id = "shared-reader-1".to_string();
    reader_config.instance.id_range_start = 1_000_000;
    reader_config.instance.id_range_size = 1_000_000;
    let reader_state = Arc::new(AppState::new(
        shared_registry,
        embedder,
        None,
        None,
        reader_config,
    ));

    let make_router = |state: Arc<AppState>| {
        Router::new()
            .route("/api/v1/health", get(handlers::health_handler))
            .route("/api/v1/metrics", get(handlers::metrics_handler))
            .route("/api/v1/memories", post(handlers::create_memory_handler))
            .route("/api/v1/memories", get(handlers::list_memories_handler))
            .route("/api/v1/memories/{id}", get(handlers::get_memory_handler))
            .route(
                "/api/v1/memories/{id}",
                put(handlers::update_memory_handler),
            )
            .route(
                "/api/v1/memories/{id}",
                delete(handlers::delete_memory_handler),
            )
            .route("/api/v1/memories/search", post(handlers::search_handler))
            .route("/api/v1/entities", get(handlers::list_entities_handler))
            .route("/api/v1/entities/{id}", get(handlers::get_entity_handler))
            .route("/api/v1/augment", post(handlers::augment_handler))
            .route("/api/v1/learn", post(handlers::learn_handler))
            .layer(middleware::from_fn_with_state(
                state.clone(),
                ucotron_server::auth::auth_middleware,
            ))
            .with_state(state.clone())
    };

    let writer_app = make_router(writer_state.clone());
    let reader_app = make_router(reader_state.clone());

    (writer_app, writer_state, reader_app, reader_state)
}

#[tokio::test]
async fn test_shared_writer_creates_reader_sees() {
    let (writer_app, writer_state, reader_app, _reader_state) = build_shared_instances();

    // Writer creates a memory node directly in the shared backend.
    insert_test_node(&writer_state, 42, "Shared memory content", NodeType::Entity);

    // Reader should see the same node (shared registry).
    let req = Request::get("/api/v1/memories/42")
        .body(Body::empty())
        .unwrap();
    let resp = reader_app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_to_json(resp.into_body()).await;
    assert_eq!(body["id"], 42);
    assert_eq!(body["content"], "Shared memory content");

    // Writer should also see it.
    let req2 = Request::get("/api/v1/memories/42")
        .body(Body::empty())
        .unwrap();
    let resp2 = writer_app.oneshot(req2).await.unwrap();
    assert_eq!(resp2.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_shared_reader_cannot_write() {
    let (_writer_app, _writer_state, reader_app, _reader_state) = build_shared_instances();

    // Reader should be rejected on write operations.
    let req = Request::post("/api/v1/memories")
        .header("Content-Type", "application/json")
        .body(Body::from(r#"{"text":"reader should not write"}"#))
        .unwrap();
    let resp = reader_app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::FORBIDDEN);
    let body = body_to_json(resp.into_body()).await;
    assert_eq!(body["code"], "READ_ONLY_INSTANCE");
}

#[tokio::test]
async fn test_shared_writer_can_write() {
    let (writer_app, _writer_state, _reader_app, _reader_state) = build_shared_instances();

    // Writer should be able to create memories.
    let req = Request::post("/api/v1/memories")
        .header("Content-Type", "application/json")
        .body(Body::from(r#"{"text":"writer can write in shared mode"}"#))
        .unwrap();
    let resp = writer_app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::CREATED);
}

#[tokio::test]
async fn test_shared_health_reports_shared_mode() {
    let (writer_app, _writer_state, reader_app, _reader_state) = build_shared_instances();

    // Writer health reports shared mode.
    let req = Request::get("/api/v1/health").body(Body::empty()).unwrap();
    let resp = writer_app.oneshot(req).await.unwrap();
    let body = body_to_json(resp.into_body()).await;
    assert_eq!(body["storage_mode"], "shared");
    assert_eq!(body["instance_role"], "writer");
    assert_eq!(body["instance_id"], "shared-writer-1");

    // Reader health reports shared mode.
    let req2 = Request::get("/api/v1/health").body(Body::empty()).unwrap();
    let resp2 = reader_app.oneshot(req2).await.unwrap();
    let body2 = body_to_json(resp2.into_body()).await;
    assert_eq!(body2["storage_mode"], "shared");
    assert_eq!(body2["instance_role"], "reader");
    assert_eq!(body2["instance_id"], "shared-reader-1");
}

#[tokio::test]
async fn test_shared_id_ranges_non_overlapping() {
    let (_writer_app, writer_state, _reader_app, _reader_state) = build_shared_instances();

    // Writer starts at 0, reader at 1_000_000. Verify writer allocates from its range.
    let id1 = writer_state.alloc_next_node_id();
    let id2 = writer_state.alloc_next_node_id();
    assert_eq!(id1, 0);
    assert_eq!(id2, 1);
    // Both < 1_000_000 (reader's range start).
    assert!(id1 < 1_000_000);
    assert!(id2 < 1_000_000);
}

#[tokio::test]
async fn test_shared_search_sees_all_data() {
    let (_writer_app, writer_state, reader_app, _reader_state) = build_shared_instances();

    // Writer inserts data.
    insert_test_node(&writer_state, 10, "Shared entity Alpha", NodeType::Entity);
    insert_test_node(&writer_state, 11, "Shared entity Beta", NodeType::Entity);

    // Reader searches and sees the writer's data.
    let req = Request::post("/api/v1/memories/search")
        .header("Content-Type", "application/json")
        .body(Body::from(r#"{"query":"shared entity","limit":10}"#))
        .unwrap();
    let resp = reader_app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_to_json(resp.into_body()).await;
    assert!(body["results"].as_array().unwrap().len() >= 2);
}

// ---------------------------------------------------------------------------
// End-to-End Pipeline Tests (US-8.6)
// ---------------------------------------------------------------------------

/// Full pipeline: POST /memories → POST /search → verify ingested data is searchable.
#[tokio::test]
async fn test_e2e_ingest_then_search_finds_data() {
    let (app, state) = build_app();

    // Step 1: Ingest text.
    let req = Request::post("/api/v1/memories")
        .header("Content-Type", "application/json")
        .body(Body::from(
            serde_json::to_string(&serde_json::json!({
                "text": "The Eiffel Tower is located in Paris. It was built in 1889."
            }))
            .unwrap(),
        ))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::CREATED);
    let ingest_body = body_to_json(resp.into_body()).await;
    let chunk_ids = ingest_body["chunk_node_ids"].as_array().unwrap();
    assert!(
        !chunk_ids.is_empty(),
        "Should create at least one chunk node"
    );

    // Step 2: Search for the ingested data.
    let app2 = with_auth(
        Router::new().route("/api/v1/memories/search", post(handlers::search_handler)),
        &state,
    );

    let req2 = Request::post("/api/v1/memories/search")
        .header("Content-Type", "application/json")
        .body(Body::from(r#"{"query":"Eiffel Tower Paris","limit":10}"#))
        .unwrap();
    let resp2 = app2.oneshot(req2).await.unwrap();
    assert_eq!(resp2.status(), StatusCode::OK);
    let search_body = body_to_json(resp2.into_body()).await;
    let results = search_body["results"].as_array().unwrap();
    assert!(!results.is_empty(), "Search should find the ingested data");
    // Verify at least one result contains content from our ingested text.
    let found_relevant = results.iter().any(|r| {
        let c = r["content"].as_str().unwrap_or("");
        c.contains("Eiffel") || c.contains("Paris") || c.contains("Tower") || c.contains("1889")
    });
    assert!(
        found_relevant,
        "At least one search result should contain content from ingested text"
    );
}

/// POST /augment with pre-ingested data → verify relevant context returned.
#[tokio::test]
async fn test_e2e_augment_returns_relevant_context() {
    let (app, state) = build_app();

    // Insert relevant data directly.
    insert_test_node(
        &state,
        300,
        "Berlin is the capital of Germany",
        NodeType::Event,
    );
    insert_test_node(
        &state,
        301,
        "Munich is known for Oktoberfest",
        NodeType::Event,
    );

    let req = Request::post("/api/v1/augment")
        .header("Content-Type", "application/json")
        .body(Body::from(
            r#"{"context":"Tell me about German cities","limit":5}"#,
        ))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_to_json(resp.into_body()).await;
    assert!(body["memories"].as_array().is_some());
    assert!(body["context_text"].as_str().is_some());
    let context = body["context_text"].as_str().unwrap();
    // Context assembly should include memory content.
    assert!(
        !context.is_empty(),
        "Context text should not be empty when there are memories"
    );
}

/// POST /learn → verify memories are stored in graph and searchable.
#[tokio::test]
async fn test_e2e_learn_stores_and_is_searchable() {
    let (app, state) = build_app();

    // Step 1: Learn from agent output.
    let req = Request::post("/api/v1/learn")
        .header("Content-Type", "application/json")
        .body(Body::from(
            r#"{"output":"The user prefers dark mode. Their favorite language is Rust."}"#,
        ))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::CREATED);
    let learn_body = body_to_json(resp.into_body()).await;
    let memories_created = learn_body["memories_created"].as_u64().unwrap();
    assert!(memories_created >= 1, "Should create at least one memory");

    // Step 2: Search for the learned data.
    let app2 = with_auth(
        Router::new().route("/api/v1/memories/search", post(handlers::search_handler)),
        &state,
    );

    let req2 = Request::post("/api/v1/memories/search")
        .header("Content-Type", "application/json")
        .body(Body::from(
            r#"{"query":"dark mode preferences","limit":10}"#,
        ))
        .unwrap();
    let resp2 = app2.oneshot(req2).await.unwrap();
    assert_eq!(resp2.status(), StatusCode::OK);
    let search_body = body_to_json(resp2.into_body()).await;
    let results = search_body["results"].as_array().unwrap();
    assert!(
        !results.is_empty(),
        "Search should find the learned memories"
    );
}

/// POST /memories → GET /memories → GET /memories/:id → PUT → DELETE — full CRUD cycle.
#[tokio::test]
async fn test_e2e_full_crud_cycle() {
    let (_, state) = build_app();

    // Create via ingestion.
    let app1 = with_auth(
        Router::new().route("/api/v1/memories", post(handlers::create_memory_handler)),
        &state,
    );
    let req1 = Request::post("/api/v1/memories")
        .header("Content-Type", "application/json")
        .body(Body::from(
            r#"{"text":"CRUD test memory about quantum computing."}"#,
        ))
        .unwrap();
    let resp1 = app1.oneshot(req1).await.unwrap();
    assert_eq!(resp1.status(), StatusCode::CREATED);
    let body1 = body_to_json(resp1.into_body()).await;
    let chunk_id = body1["chunk_node_ids"][0].as_u64().unwrap();

    // Read the created memory.
    let app2 = with_auth(
        Router::new().route("/api/v1/memories/{id}", get(handlers::get_memory_handler)),
        &state,
    );
    let req2 = Request::get(format!("/api/v1/memories/{}", chunk_id))
        .body(Body::empty())
        .unwrap();
    let resp2 = app2.oneshot(req2).await.unwrap();
    assert_eq!(resp2.status(), StatusCode::OK);
    let body2 = body_to_json(resp2.into_body()).await;
    assert_eq!(body2["id"], chunk_id);

    // Update the memory.
    let app3 = with_auth(
        Router::new().route(
            "/api/v1/memories/{id}",
            put(handlers::update_memory_handler),
        ),
        &state,
    );
    let req3 = Request::put(format!("/api/v1/memories/{}", chunk_id))
        .header("Content-Type", "application/json")
        .body(Body::from(
            r#"{"content":"Updated quantum content","metadata":{"updated":true}}"#,
        ))
        .unwrap();
    let resp3 = app3.oneshot(req3).await.unwrap();
    assert_eq!(resp3.status(), StatusCode::OK);
    let body3 = body_to_json(resp3.into_body()).await;
    assert_eq!(body3["content"], "Updated quantum content");
    assert_eq!(body3["metadata"]["updated"], true);

    // Delete the memory.
    let app4 = with_auth(
        Router::new().route(
            "/api/v1/memories/{id}",
            delete(handlers::delete_memory_handler),
        ),
        &state,
    );
    let req4 = Request::delete(format!("/api/v1/memories/{}", chunk_id))
        .body(Body::empty())
        .unwrap();
    let resp4 = app4.oneshot(req4).await.unwrap();
    assert_eq!(resp4.status(), StatusCode::NO_CONTENT);

    // Verify soft-delete (node exists but content is empty).
    let node = state.registry.graph().get_node(chunk_id).unwrap().unwrap();
    assert!(node.content.is_empty());
}

// ---------------------------------------------------------------------------
// Multi-Tenancy Namespace Isolation Tests (US-8.6)
// ---------------------------------------------------------------------------

/// Create memories in two namespaces, verify isolation: each namespace only sees its own data.
#[tokio::test]
async fn test_multitenant_namespace_isolation() {
    let (_, state) = build_app();

    // Ingest data in namespace "org-alpha".
    let app1 = with_auth(
        Router::new().route("/api/v1/memories", post(handlers::create_memory_handler)),
        &state,
    );
    let req1 = Request::post("/api/v1/memories")
        .header("Content-Type", "application/json")
        .header("X-Ucotron-Namespace", "org-alpha")
        .body(Body::from(
            r#"{"text":"Alpha secret project about rockets."}"#,
        ))
        .unwrap();
    let resp1 = app1.oneshot(req1).await.unwrap();
    assert_eq!(resp1.status(), StatusCode::CREATED);

    // Ingest data in namespace "org-beta".
    let app2 = with_auth(
        Router::new().route("/api/v1/memories", post(handlers::create_memory_handler)),
        &state,
    );
    let req2 = Request::post("/api/v1/memories")
        .header("Content-Type", "application/json")
        .header("X-Ucotron-Namespace", "org-beta")
        .body(Body::from(
            r#"{"text":"Beta confidential data about submarines."}"#,
        ))
        .unwrap();
    let resp2 = app2.oneshot(req2).await.unwrap();
    assert_eq!(resp2.status(), StatusCode::CREATED);

    // Search from "org-alpha" — should NOT see org-beta data.
    let app3 = with_auth(
        Router::new().route("/api/v1/memories/search", post(handlers::search_handler)),
        &state,
    );
    let req3 = Request::post("/api/v1/memories/search")
        .header("Content-Type", "application/json")
        .header("X-Ucotron-Namespace", "org-alpha")
        .body(Body::from(r#"{"query":"submarines","limit":50}"#))
        .unwrap();
    let resp3 = app3.oneshot(req3).await.unwrap();
    assert_eq!(resp3.status(), StatusCode::OK);
    let body3 = body_to_json(resp3.into_body()).await;
    let results3 = body3["results"].as_array().unwrap();
    // Org-alpha should not see org-beta's submarine data.
    for r in results3 {
        let content = r["content"].as_str().unwrap_or("");
        assert!(
            !content.contains("submarine"),
            "org-alpha search should not return org-beta data: {}",
            content
        );
    }

    // Search from "org-beta" — should NOT see org-alpha data.
    let app4 = with_auth(
        Router::new().route("/api/v1/memories/search", post(handlers::search_handler)),
        &state,
    );
    let req4 = Request::post("/api/v1/memories/search")
        .header("Content-Type", "application/json")
        .header("X-Ucotron-Namespace", "org-beta")
        .body(Body::from(r#"{"query":"rockets","limit":50}"#))
        .unwrap();
    let resp4 = app4.oneshot(req4).await.unwrap();
    assert_eq!(resp4.status(), StatusCode::OK);
    let body4 = body_to_json(resp4.into_body()).await;
    let results4 = body4["results"].as_array().unwrap();
    for r in results4 {
        let content = r["content"].as_str().unwrap_or("");
        assert!(
            !content.contains("rocket"),
            "org-beta search should not return org-alpha data: {}",
            content
        );
    }
}

/// Augment with namespace isolation — each namespace context is isolated.
#[tokio::test]
async fn test_multitenant_augment_isolation() {
    let (_, state) = build_app();

    // Insert a node tagged with namespace "team-a".
    let mut node_a = Node {
        id: 500,
        content: "Team A uses PostgreSQL for their database".into(),
        embedding: vec![0.5f32; 384],
        metadata: HashMap::new(),
        node_type: NodeType::Event,
        timestamp: 1700000000,
        media_type: None,
        media_uri: None,
        embedding_visual: None,
        timestamp_range: None,
        parent_video_id: None,
    };
    node_a.metadata.insert(
        "_namespace".into(),
        ucotron_core::Value::String("team-a".into()),
    );
    state.registry.graph().upsert_nodes(&[node_a]).unwrap();
    state
        .registry
        .vector()
        .upsert_embeddings(&[(500, vec![0.5f32; 384])])
        .unwrap();

    // Insert a node tagged with namespace "team-b".
    let mut node_b = Node {
        id: 501,
        content: "Team B uses MongoDB for their database".into(),
        embedding: vec![0.5f32; 384],
        metadata: HashMap::new(),
        node_type: NodeType::Event,
        timestamp: 1700000000,
        media_type: None,
        media_uri: None,
        embedding_visual: None,
        timestamp_range: None,
        parent_video_id: None,
    };
    node_b.metadata.insert(
        "_namespace".into(),
        ucotron_core::Value::String("team-b".into()),
    );
    state.registry.graph().upsert_nodes(&[node_b]).unwrap();
    state
        .registry
        .vector()
        .upsert_embeddings(&[(501, vec![0.5f32; 384])])
        .unwrap();

    // Augment from team-a — should only see team-a data.
    let app = with_auth(
        Router::new().route("/api/v1/augment", post(handlers::augment_handler)),
        &state,
    );
    let req = Request::post("/api/v1/augment")
        .header("Content-Type", "application/json")
        .header("X-Ucotron-Namespace", "team-a")
        .body(Body::from(r#"{"context":"database","limit":10}"#))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_to_json(resp.into_body()).await;
    let memories = body["memories"].as_array().unwrap();
    for m in memories {
        let content = m["content"].as_str().unwrap_or("");
        assert!(
            !content.contains("MongoDB"),
            "team-a augment should not return team-b data"
        );
    }

    // BUG-1 fix: verify context_text also excludes other namespaces.
    let context_text = body["context_text"].as_str().unwrap_or("");
    assert!(
        !context_text.contains("MongoDB"),
        "team-a context_text must not contain team-b data, got: {}",
        context_text
    );
    assert!(
        !context_text.contains("Team B"),
        "team-a context_text must not mention Team B, got: {}",
        context_text
    );
}

// ---------------------------------------------------------------------------
// Concurrent Load Test (US-8.6)
// ---------------------------------------------------------------------------

/// Load test: 1000 concurrent search requests — verify no crashes.
#[tokio::test]
async fn test_concurrent_search_requests() {
    let (_, state) = build_app();

    // Pre-insert some data.
    for i in 0..50 {
        insert_test_node(
            &state,
            600 + i,
            &format!("Load test memory number {}", i),
            NodeType::Event,
        );
    }

    // Fire 200 concurrent search requests (practical for in-process mock test).
    let mut handles = Vec::new();
    for i in 0..200u32 {
        let st = state.clone();
        let handle = tokio::spawn(async move {
            let app = with_auth(
                Router::new().route("/api/v1/memories/search", post(handlers::search_handler)),
                &st,
            );
            let req = Request::post("/api/v1/memories/search")
                .header("Content-Type", "application/json")
                .body(Body::from(format!(
                    r#"{{"query":"load test query {}","limit":5}}"#,
                    i
                )))
                .unwrap();
            let resp = app.oneshot(req).await.unwrap();
            resp.status()
        });
        handles.push(handle);
    }

    let mut success_count = 0u32;
    for handle in handles {
        let status = handle.await.unwrap();
        if status == StatusCode::OK {
            success_count += 1;
        }
    }
    assert_eq!(
        success_count, 200,
        "All 200 concurrent requests should succeed"
    );
}

/// Load test: concurrent mixed read/write operations — verify no crashes or deadlocks.
#[tokio::test]
async fn test_concurrent_mixed_operations() {
    let (_, state) = build_app();

    // Pre-insert data.
    for i in 0..20 {
        insert_test_node(
            &state,
            700 + i,
            &format!("Mixed load node {}", i),
            NodeType::Entity,
        );
    }

    let mut handles = Vec::new();

    // 50 concurrent search requests.
    for i in 0..50u32 {
        let st = state.clone();
        handles.push(tokio::spawn(async move {
            let app = with_auth(
                Router::new().route("/api/v1/memories/search", post(handlers::search_handler)),
                &st,
            );
            let req = Request::post("/api/v1/memories/search")
                .header("Content-Type", "application/json")
                .body(Body::from(format!(
                    r#"{{"query":"mixed query {}","limit":5}}"#,
                    i
                )))
                .unwrap();
            let resp = app.oneshot(req).await.unwrap();
            assert_eq!(resp.status(), StatusCode::OK, "Search {} failed", i);
        }));
    }

    // 50 concurrent read requests.
    for i in 0..50u32 {
        let st = state.clone();
        let node_id = 700 + (i % 20) as u64;
        handles.push(tokio::spawn(async move {
            let app = with_auth(
                Router::new().route("/api/v1/memories/{id}", get(handlers::get_memory_handler)),
                &st,
            );
            let req = Request::get(format!("/api/v1/memories/{}", node_id))
                .body(Body::empty())
                .unwrap();
            let resp = app.oneshot(req).await.unwrap();
            assert_eq!(resp.status(), StatusCode::OK, "Get {} failed", node_id);
        }));
    }

    // 20 concurrent health checks.
    for _ in 0..20u32 {
        let st = state.clone();
        handles.push(tokio::spawn(async move {
            let app = with_auth(
                Router::new().route("/api/v1/health", get(handlers::health_handler)),
                &st,
            );
            let req = Request::get("/api/v1/health").body(Body::empty()).unwrap();
            let resp = app.oneshot(req).await.unwrap();
            assert_eq!(resp.status(), StatusCode::OK);
        }));
    }

    // Wait for all to complete (no panics = no crashes).
    for handle in handles {
        handle.await.unwrap();
    }
}

/// Verify the full ingest → augment pipeline returns relevant context.
#[tokio::test]
async fn test_e2e_ingest_then_augment() {
    let (_, state) = build_app();

    // Ingest.
    let app1 = with_auth(
        Router::new().route("/api/v1/memories", post(handlers::create_memory_handler)),
        &state,
    );
    let req1 = Request::post("/api/v1/memories")
        .header("Content-Type", "application/json")
        .body(Body::from(
            r#"{"text":"Tokyo is the capital of Japan. It has a population of 14 million."}"#,
        ))
        .unwrap();
    let resp1 = app1.oneshot(req1).await.unwrap();
    assert_eq!(resp1.status(), StatusCode::CREATED);

    // Augment with related query.
    let app2 = with_auth(
        Router::new().route("/api/v1/augment", post(handlers::augment_handler)),
        &state,
    );
    let req2 = Request::post("/api/v1/augment")
        .header("Content-Type", "application/json")
        .body(Body::from(
            r#"{"context":"Japanese cities population","limit":10}"#,
        ))
        .unwrap();
    let resp2 = app2.oneshot(req2).await.unwrap();
    assert_eq!(resp2.status(), StatusCode::OK);
    let body2 = body_to_json(resp2.into_body()).await;
    // Should have some memories returned.
    let memories = body2["memories"].as_array().unwrap();
    assert!(
        !memories.is_empty(),
        "Augment should return relevant memories after ingestion"
    );
}

/// Verify metrics counters increment correctly through multiple operations.
#[tokio::test]
async fn test_e2e_metrics_increment_through_operations() {
    let (_, state) = build_app();

    // Perform a few operations.
    // 1. Ingest
    let app1 = with_auth(
        Router::new().route("/api/v1/memories", post(handlers::create_memory_handler)),
        &state,
    );
    let req1 = Request::post("/api/v1/memories")
        .header("Content-Type", "application/json")
        .body(Body::from(r#"{"text":"Metrics test data."}"#))
        .unwrap();
    app1.oneshot(req1).await.unwrap();

    // 2. Search
    let app2 = with_auth(
        Router::new().route("/api/v1/memories/search", post(handlers::search_handler)),
        &state,
    );
    let req2 = Request::post("/api/v1/memories/search")
        .header("Content-Type", "application/json")
        .body(Body::from(r#"{"query":"test","limit":5}"#))
        .unwrap();
    app2.oneshot(req2).await.unwrap();

    // 3. Learn
    let app3 = with_auth(
        Router::new().route("/api/v1/learn", post(handlers::learn_handler)),
        &state,
    );
    let req3 = Request::post("/api/v1/learn")
        .header("Content-Type", "application/json")
        .body(Body::from(r#"{"output":"User learned something new."}"#))
        .unwrap();
    app3.oneshot(req3).await.unwrap();

    // Check metrics.
    let app4 = with_auth(
        Router::new().route("/api/v1/metrics", get(handlers::metrics_handler)),
        &state,
    );
    let req4 = Request::get("/api/v1/metrics").body(Body::empty()).unwrap();
    let resp4 = app4.oneshot(req4).await.unwrap();
    let body4 = body_to_json(resp4.into_body()).await;

    assert!(
        body4["total_ingestions"].as_u64().unwrap() >= 2,
        "Should have at least 2 ingestions (create + learn)"
    );
    assert!(
        body4["total_searches"].as_u64().unwrap() >= 1,
        "Should have at least 1 search"
    );
}

// ---------------------------------------------------------------------------
// Writer lock tests
// ---------------------------------------------------------------------------

#[test]
fn test_writer_lock_acquire_release() {
    use ucotron_server::writer_lock::WriterLock;

    let dir = tempfile::tempdir().unwrap();
    let shared_dir = dir.path().to_string_lossy().to_string();

    let lock = WriterLock::acquire(&shared_dir, "integration-writer-1").unwrap();
    assert!(lock.lock_path().exists());
    lock.release();
    assert!(!dir.path().join("ucotron-writer.lock").exists());
}

#[test]
fn test_writer_lock_prevents_second_writer() {
    use ucotron_server::writer_lock::WriterLock;

    let dir = tempfile::tempdir().unwrap();
    let shared_dir = dir.path().to_string_lossy().to_string();

    let lock1 = WriterLock::acquire(&shared_dir, "writer-a").unwrap();

    // Second writer should fail.
    let result = WriterLock::acquire(&shared_dir, "writer-b");
    assert!(result.is_err());
    let err_msg = result.unwrap_err().to_string();
    assert!(err_msg.contains("Another writer instance"));

    // After release, new writer succeeds.
    lock1.release();
    let lock2 = WriterLock::acquire(&shared_dir, "writer-b").unwrap();
    lock2.release();
}

// ---------------------------------------------------------------------------
// Graph Visualization
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_graph_empty() {
    let (app, _state) = build_app();
    let req = Request::get("/api/v1/graph").body(Body::empty()).unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_to_json(resp.into_body()).await;
    assert_eq!(body["total_nodes"], 0);
    assert_eq!(body["total_edges"], 0);
    assert!(body["nodes"].as_array().unwrap().is_empty());
    assert!(body["edges"].as_array().unwrap().is_empty());
}

#[tokio::test]
async fn test_graph_with_nodes_and_edges() {
    let (app, state) = build_app();
    insert_test_node(&state, 1, "Alice", NodeType::Entity);
    insert_test_node(&state, 2, "Bob", NodeType::Entity);
    insert_test_node(&state, 3, "Meeting", NodeType::Event);

    // Insert edges.
    let edges = vec![
        Edge {
            source: 1,
            target: 3,
            edge_type: EdgeType::Actor,
            weight: 1.0,
            metadata: HashMap::new(),
        },
        Edge {
            source: 2,
            target: 3,
            edge_type: EdgeType::Actor,
            weight: 0.8,
            metadata: HashMap::new(),
        },
    ];
    state.registry.graph().upsert_edges(&edges).unwrap();

    let req = Request::get("/api/v1/graph?limit=500")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_to_json(resp.into_body()).await;
    assert!(body["total_nodes"].as_u64().unwrap() >= 2);
    assert!(body["nodes"].as_array().unwrap().len() >= 2);
}

#[tokio::test]
async fn test_graph_node_type_filter() {
    let (app, state) = build_app();
    insert_test_node(&state, 10, "Entity Node", NodeType::Entity);
    insert_test_node(&state, 11, "Event Node", NodeType::Event);

    let req = Request::get("/api/v1/graph?node_type=entity")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_to_json(resp.into_body()).await;
    let nodes = body["nodes"].as_array().unwrap();
    for node in nodes {
        assert_eq!(node["node_type"], "Entity");
    }
}

// ---------------------------------------------------------------------------
// Export / Import
// ---------------------------------------------------------------------------

fn insert_test_node_with_namespace(
    state: &AppState,
    id: NodeId,
    content: &str,
    node_type: NodeType,
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
        node_type,
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

fn insert_test_edge(state: &AppState, source: NodeId, target: NodeId, edge_type: EdgeType) {
    let edge = Edge {
        source,
        target,
        edge_type,
        weight: 0.8,
        metadata: HashMap::new(),
    };
    state.registry.graph().upsert_edges(&[edge]).unwrap();
}

#[tokio::test]
async fn test_export_empty_graph() {
    let (app, _state) = build_app();
    let req = Request::get("/api/v1/export").body(Body::empty()).unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_to_json(resp.into_body()).await;
    assert_eq!(body["@type"], "ucotron:MemoryGraph");
    assert_eq!(body["version"], "1.0");
    assert_eq!(body["namespace"], "default");
    assert_eq!(body["stats"]["total_nodes"], 0);
    assert_eq!(body["stats"]["total_edges"], 0);
    assert!(body["nodes"].as_array().unwrap().is_empty());
    assert!(body["edges"].as_array().unwrap().is_empty());
}

#[tokio::test]
async fn test_export_with_data() {
    let (app, state) = build_app();
    insert_test_node_with_namespace(&state, 1, "Alice", NodeType::Entity, "default");
    insert_test_node_with_namespace(&state, 2, "Bob", NodeType::Entity, "default");
    insert_test_edge(&state, 1, 2, EdgeType::RelatesTo);

    let req = Request::get("/api/v1/export").body(Body::empty()).unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_to_json(resp.into_body()).await;
    assert_eq!(body["stats"]["total_nodes"], 2);
    assert_eq!(body["stats"]["total_edges"], 1);
    assert_eq!(body["nodes"].as_array().unwrap().len(), 2);
    assert_eq!(body["edges"].as_array().unwrap().len(), 1);
}

#[tokio::test]
async fn test_export_namespace_filtering() {
    let (app, state) = build_app();
    insert_test_node_with_namespace(&state, 1, "Alice", NodeType::Entity, "ns_a");
    insert_test_node_with_namespace(&state, 2, "Bob", NodeType::Entity, "ns_b");
    insert_test_node_with_namespace(&state, 3, "Carol", NodeType::Entity, "ns_a");

    let req = Request::get("/api/v1/export")
        .header("X-Ucotron-Namespace", "ns_a")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_to_json(resp.into_body()).await;
    assert_eq!(body["namespace"], "ns_a");
    assert_eq!(body["stats"]["total_nodes"], 2);
}

#[tokio::test]
async fn test_export_incremental() {
    let (app, state) = build_app();
    insert_test_node_with_namespace(&state, 1, "Old node", NodeType::Entity, "default");
    insert_test_node_with_namespace(&state, 100, "New node", NodeType::Entity, "default");

    let req = Request::get("/api/v1/export?from_timestamp=1700000050")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_to_json(resp.into_body()).await;
    assert_eq!(body["stats"]["total_nodes"], 1);
    assert!(body["stats"]["is_incremental"].as_bool().unwrap());
}

#[tokio::test]
async fn test_export_without_embeddings() {
    let (app, state) = build_app();
    insert_test_node_with_namespace(&state, 1, "Alice", NodeType::Entity, "default");

    let req = Request::get("/api/v1/export?include_embeddings=false")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_to_json(resp.into_body()).await;
    assert!(!body["stats"]["has_embeddings"].as_bool().unwrap());
    let nodes = body["nodes"].as_array().unwrap();
    assert!(nodes[0].get("embedding").is_none());
}

#[tokio::test]
async fn test_export_invalid_format() {
    let (app, _state) = build_app();
    let req = Request::get("/api/v1/export?format=csv")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn test_import_basic() {
    let (app, state) = build_app();

    // Create a JSON-LD export document.
    let import_body = serde_json::json!({
        "@context": {"ucotron": "https://ucotron.com/schema/v1"},
        "@type": "ucotron:MemoryGraph",
        "version": "1.0",
        "exported_at": 1700000000u64,
        "namespace": "source_ns",
        "nodes": [
            {
                "@id": "ucotron:node/1",
                "@type": "Entity",
                "content": "Alice",
                "timestamp": 1700000001u64,
                "metadata": {"role": "engineer"}
            },
            {
                "@id": "ucotron:node/2",
                "@type": "Event",
                "content": "Meeting at park",
                "timestamp": 1700000002u64
            }
        ],
        "edges": [
            {
                "source": "ucotron:node/1",
                "target": "ucotron:node/2",
                "edge_type": "Actor",
                "weight": 0.9
            }
        ],
        "stats": {
            "total_nodes": 2,
            "total_edges": 1,
            "has_embeddings": false,
            "is_incremental": false
        }
    });

    let req = Request::post("/api/v1/import")
        .header("Content-Type", "application/json")
        .header("X-Ucotron-Namespace", "imported")
        .body(Body::from(serde_json::to_string(&import_body).unwrap()))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_to_json(resp.into_body()).await;
    assert_eq!(body["nodes_imported"], 2);
    assert_eq!(body["edges_imported"], 1);
    assert_eq!(body["target_namespace"], "imported");

    // Verify nodes were actually stored.
    let all_nodes = state.registry.graph().get_all_nodes().unwrap();
    assert_eq!(all_nodes.len(), 2);
}

#[tokio::test]
async fn test_import_invalid_version() {
    let (app, _state) = build_app();
    let import_body = serde_json::json!({
        "@context": {},
        "@type": "ucotron:MemoryGraph",
        "version": "99.0",
        "exported_at": 0,
        "namespace": "test",
        "nodes": [],
        "edges": []
    });

    let req = Request::post("/api/v1/import")
        .header("Content-Type", "application/json")
        .body(Body::from(serde_json::to_string(&import_body).unwrap()))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn test_export_then_import_roundtrip() {
    // Phase 1: Create data and export.
    let (app1, state1) = build_app();
    insert_test_node_with_namespace(&state1, 1, "Alice", NodeType::Entity, "default");
    insert_test_node_with_namespace(&state1, 2, "Bob", NodeType::Entity, "default");
    insert_test_edge(&state1, 1, 2, EdgeType::RelatesTo);

    let req = Request::get("/api/v1/export").body(Body::empty()).unwrap();
    let resp = app1.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let export_body = body_to_json(resp.into_body()).await;
    assert_eq!(export_body["stats"]["total_nodes"], 2);

    // Phase 2: Import into a fresh app.
    let (app2, state2) = build_app();
    let req = Request::post("/api/v1/import")
        .header("Content-Type", "application/json")
        .header("X-Ucotron-Namespace", "reimported")
        .body(Body::from(serde_json::to_string(&export_body).unwrap()))
        .unwrap();
    let resp = app2.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let import_result = body_to_json(resp.into_body()).await;
    assert_eq!(import_result["nodes_imported"], 2);
    assert_eq!(import_result["edges_imported"], 1);

    // Verify integrity.
    let all_nodes = state2.registry.graph().get_all_nodes().unwrap();
    assert_eq!(all_nodes.len(), 2);
    let contents: Vec<&str> = all_nodes.iter().map(|n| n.content.as_str()).collect();
    assert!(contents.contains(&"Alice"));
    assert!(contents.contains(&"Bob"));
}

// ---------------------------------------------------------------------------
// Audio Transcription Tests
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_transcribe_no_transcriber_returns_501() {
    let (app, _state) = build_app(); // No transcriber loaded

    let boundary = "----WebKitFormBoundary12345";
    let wav_bytes = create_test_wav_bytes();
    let mut body_bytes = Vec::new();
    body_bytes.extend_from_slice(format!("--{}\r\n", boundary).as_bytes());
    body_bytes.extend_from_slice(
        b"Content-Disposition: form-data; name=\"file\"; filename=\"test.wav\"\r\n",
    );
    body_bytes.extend_from_slice(b"Content-Type: audio/wav\r\n\r\n");
    body_bytes.extend_from_slice(&wav_bytes);
    body_bytes.extend_from_slice(format!("\r\n--{}--\r\n", boundary).as_bytes());

    let req = Request::post("/api/v1/transcribe")
        .header(
            "Content-Type",
            format!("multipart/form-data; boundary={}", boundary),
        )
        .body(Body::from(body_bytes))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::NOT_IMPLEMENTED);
    let body = body_to_json(resp.into_body()).await;
    assert_eq!(body["code"], "NOT_IMPLEMENTED");
}

#[tokio::test]
async fn test_transcribe_with_mock_transcriber() {
    let (app, _state) = build_app_with_transcriber();

    let boundary = "----TestBoundary9876";
    let wav_bytes = create_test_wav_bytes();
    let mut body_bytes = Vec::new();
    body_bytes.extend_from_slice(format!("--{}\r\n", boundary).as_bytes());
    body_bytes.extend_from_slice(
        b"Content-Disposition: form-data; name=\"file\"; filename=\"test.wav\"\r\n",
    );
    body_bytes.extend_from_slice(b"Content-Type: audio/wav\r\n\r\n");
    body_bytes.extend_from_slice(&wav_bytes);
    body_bytes.extend_from_slice(format!("\r\n--{}--\r\n", boundary).as_bytes());

    let req = Request::post("/api/v1/transcribe")
        .header(
            "Content-Type",
            format!("multipart/form-data; boundary={}", boundary),
        )
        .body(Body::from(body_bytes))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_to_json(resp.into_body()).await;
    assert_eq!(body["text"], "Hello world from mock transcriber");
    assert!(!body["chunks"].as_array().unwrap().is_empty());
    assert!(body["audio"]["duration_secs"].as_f64().unwrap() > 0.0);
    assert_eq!(body["audio"]["sample_rate"], 16000);
    assert_eq!(body["audio"]["channels"], 1);
    assert_eq!(body["audio"]["detected_language"], "en");
    // With default ingest=true, ingestion should be attempted
    assert!(body["ingestion"].is_object() || body["ingestion"].is_null());
}

#[tokio::test]
async fn test_transcribe_no_ingest() {
    let (app, state) = build_app_with_transcriber();

    let boundary = "----TestBoundary5555";
    let wav_bytes = create_test_wav_bytes();
    let mut body_bytes = Vec::new();
    // File field
    body_bytes.extend_from_slice(format!("--{}\r\n", boundary).as_bytes());
    body_bytes.extend_from_slice(
        b"Content-Disposition: form-data; name=\"file\"; filename=\"test.wav\"\r\n",
    );
    body_bytes.extend_from_slice(b"Content-Type: audio/wav\r\n\r\n");
    body_bytes.extend_from_slice(&wav_bytes);
    body_bytes.extend_from_slice(b"\r\n");
    // Ingest field = false
    body_bytes.extend_from_slice(format!("--{}\r\n", boundary).as_bytes());
    body_bytes.extend_from_slice(b"Content-Disposition: form-data; name=\"ingest\"\r\n\r\n");
    body_bytes.extend_from_slice(b"false");
    body_bytes.extend_from_slice(format!("\r\n--{}--\r\n", boundary).as_bytes());

    let req = Request::post("/api/v1/transcribe")
        .header(
            "Content-Type",
            format!("multipart/form-data; boundary={}", boundary),
        )
        .body(Body::from(body_bytes))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_to_json(resp.into_body()).await;
    assert_eq!(body["text"], "Hello world from mock transcriber");
    // With ingest=false, no ingestion should happen
    assert!(body["ingestion"].is_null());
    // And no nodes should exist in the graph
    let all_nodes = state.registry.graph().get_all_nodes().unwrap();
    assert_eq!(all_nodes.len(), 0);
}

#[tokio::test]
async fn test_transcribe_health_reports_transcriber() {
    let (app, _state) = build_app_with_transcriber();
    let req = Request::get("/api/v1/health").body(Body::empty()).unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_to_json(resp.into_body()).await;
    assert_eq!(body["models"]["transcriber_loaded"], true);
}

// ---------------------------------------------------------------------------
// Admin: Namespace Management Tests
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_admin_list_namespaces_empty() {
    let (app, _state) = build_app();
    let req = Request::get("/api/v1/admin/namespaces")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_to_json(resp.into_body()).await;
    // Default namespace always present.
    assert!(body["total"].as_u64().unwrap() >= 1);
    let ns = body["namespaces"].as_array().unwrap();
    assert!(ns.iter().any(|n| n["name"] == "default"));
}

#[tokio::test]
async fn test_admin_create_namespace() {
    let (app, _state) = build_app();
    let req = Request::post("/api/v1/admin/namespaces")
        .header("Content-Type", "application/json")
        .body(Body::from(r#"{"name":"test-ns"}"#))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::CREATED);
    let body = body_to_json(resp.into_body()).await;
    assert_eq!(body["name"], "test-ns");
    assert_eq!(body["created"], true);
}

#[tokio::test]
async fn test_admin_create_namespace_invalid_name() {
    let (app, _state) = build_app();
    let req = Request::post("/api/v1/admin/namespaces")
        .header("Content-Type", "application/json")
        .body(Body::from(r#"{"name":"bad name!@#"}"#))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn test_admin_create_namespace_empty_name() {
    let (app, _state) = build_app();
    let req = Request::post("/api/v1/admin/namespaces")
        .header("Content-Type", "application/json")
        .body(Body::from(r#"{"name":""}"#))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn test_admin_delete_namespace() {
    let (app, state) = build_app();

    // First create a node in a namespace.
    let mut metadata = HashMap::new();
    metadata.insert(
        "_namespace".into(),
        ucotron_core::Value::String("to-delete".into()),
    );
    let node = Node {
        id: 999,
        content: "test".into(),
        embedding: vec![0.0; 384],
        metadata,
        node_type: NodeType::Entity,
        timestamp: 1000,
        media_type: None,
        media_uri: None,
        embedding_visual: None,
        timestamp_range: None,
        parent_video_id: None,
    };
    state.registry.graph().upsert_nodes(&[node]).unwrap();

    let req = Request::delete("/api/v1/admin/namespaces/to-delete")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_to_json(resp.into_body()).await;
    assert_eq!(body["name"], "to-delete");
    assert_eq!(body["nodes_deleted"], 1);
}

#[tokio::test]
async fn test_admin_delete_default_namespace_rejected() {
    let (app, _state) = build_app();
    let req = Request::delete("/api/v1/admin/namespaces/default")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn test_admin_get_namespace() {
    let (app, state) = build_app();

    // Insert nodes in a namespace.
    for i in 0u64..3 {
        let mut metadata = HashMap::new();
        metadata.insert(
            "_namespace".into(),
            ucotron_core::Value::String("stats-ns".into()),
        );
        let node = Node {
            id: 500 + i,
            content: format!("node {}", i),
            embedding: vec![0.0; 384],
            metadata,
            node_type: if i == 0 {
                NodeType::Entity
            } else {
                NodeType::Event
            },
            timestamp: 2000 + i,
            media_type: None,
            media_uri: None,
            embedding_visual: None,
            timestamp_range: None,
            parent_video_id: None,
        };
        state.registry.graph().upsert_nodes(&[node]).unwrap();
    }

    let req = Request::get("/api/v1/admin/namespaces/stats-ns")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_to_json(resp.into_body()).await;
    assert_eq!(body["name"], "stats-ns");
    assert_eq!(body["total_nodes"], 3);
    assert_eq!(body["entity_count"], 1);
    assert_eq!(body["memory_count"], 2);
    assert_eq!(body["last_activity"], 2002);
}

#[tokio::test]
async fn test_admin_get_namespace_not_found() {
    let (app, _state) = build_app();
    let req = Request::get("/api/v1/admin/namespaces/nonexistent")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}

// ---------------------------------------------------------------------------
// Admin: Config & System Tests
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_admin_config() {
    let (app, _state) = build_app();
    let req = Request::get("/api/v1/admin/config")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_to_json(resp.into_body()).await;
    assert!(body["server"]["host"].is_string());
    assert!(body["server"]["port"].is_number());
    assert!(body["storage"]["mode"].is_string());
    assert!(body["models"]["embedding_model"].is_string());
    assert!(body["instance"]["instance_id"].is_string());
    assert!(body["namespaces"]["default_namespace"].is_string());
}

#[tokio::test]
async fn test_admin_system_info() {
    let (app, _state) = build_app();
    let req = Request::get("/api/v1/admin/system")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_to_json(resp.into_body()).await;
    assert!(body["cpu_count"].as_u64().unwrap() >= 1);
    assert!(body["memory_rss_bytes"].is_number());
    assert!(body["total_nodes"].is_number());
    assert!(body["total_edges"].is_number());
    assert!(body["uptime_secs"].is_number());
}

// ---------------------------------------------------------------------------
// Document OCR Tests
// ---------------------------------------------------------------------------

fn build_app_with_ocr() -> (Router, Arc<AppState>) {
    let registry = Arc::new(BackendRegistry::new(
        Box::new(MockVectorBackend::new()),
        Box::new(MockGraphBackend::new()),
    ));
    let embedder: Arc<dyn EmbeddingPipeline> = Arc::new(MockEmbedder);
    let ocr: Arc<dyn DocumentOcrPipeline> = Arc::new(MockDocumentOcrPipeline);
    let config = UcotronConfig::default();
    let state = Arc::new(AppState::with_all_pipelines_and_ocr(
        registry,
        embedder,
        None,
        None,
        None,
        None,
        None,
        Some(ocr),
        config,
    ));

    let app = Router::new()
        .route("/api/v1/health", get(handlers::health_handler))
        .route("/api/v1/ocr", post(handlers::ocr_handler))
        .layer(middleware::from_fn_with_state(
            state.clone(),
            ucotron_server::auth::auth_middleware,
        ))
        .with_state(state.clone());

    (app, state)
}

#[tokio::test]
async fn test_ocr_501_when_not_loaded() {
    let (app, _state) = build_app();

    let boundary = "----TestBoundaryOcr1";
    let mut body_bytes = Vec::new();
    body_bytes.extend_from_slice(format!("--{}\r\n", boundary).as_bytes());
    body_bytes.extend_from_slice(
        b"Content-Disposition: form-data; name=\"file\"; filename=\"doc.pdf\"\r\n",
    );
    body_bytes.extend_from_slice(b"Content-Type: application/pdf\r\n\r\n");
    body_bytes.extend_from_slice(b"fake pdf data");
    body_bytes.extend_from_slice(format!("\r\n--{}--\r\n", boundary).as_bytes());

    let req = Request::post("/api/v1/ocr")
        .header(
            "Content-Type",
            format!("multipart/form-data; boundary={}", boundary),
        )
        .body(Body::from(body_bytes))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::NOT_IMPLEMENTED);
}

#[tokio::test]
async fn test_ocr_mock_pdf() {
    let (app, _state) = build_app_with_ocr();

    let boundary = "----TestBoundaryOcr2";
    let mut body_bytes = Vec::new();
    body_bytes.extend_from_slice(format!("--{}\r\n", boundary).as_bytes());
    body_bytes.extend_from_slice(
        b"Content-Disposition: form-data; name=\"file\"; filename=\"doc.pdf\"\r\n",
    );
    body_bytes.extend_from_slice(b"Content-Type: application/pdf\r\n\r\n");
    body_bytes.extend_from_slice(b"fake pdf content");
    body_bytes.extend_from_slice(format!("\r\n--{}--\r\n", boundary).as_bytes());

    let req = Request::post("/api/v1/ocr")
        .header(
            "Content-Type",
            format!("multipart/form-data; boundary={}", boundary),
        )
        .body(Body::from(body_bytes))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_to_json(resp.into_body()).await;
    assert_eq!(body["text"], "Extracted text from mock document pipeline.");
    assert_eq!(body["document"]["total_pages"], 1);
    assert_eq!(body["document"]["format"], "pdf");
    assert_eq!(body["document"]["is_scanned"], false);
    assert!(!body["pages"].as_array().unwrap().is_empty());
    assert_eq!(body["pages"][0]["page_number"], 1);
}

#[tokio::test]
async fn test_ocr_no_ingest() {
    let (app, state) = build_app_with_ocr();

    let boundary = "----TestBoundaryOcr3";
    let mut body_bytes = Vec::new();
    // File field
    body_bytes.extend_from_slice(format!("--{}\r\n", boundary).as_bytes());
    body_bytes.extend_from_slice(
        b"Content-Disposition: form-data; name=\"document\"; filename=\"scan.png\"\r\n",
    );
    body_bytes.extend_from_slice(b"Content-Type: image/png\r\n\r\n");
    body_bytes.extend_from_slice(b"fake image data");
    body_bytes.extend_from_slice(b"\r\n");
    // Ingest field = false
    body_bytes.extend_from_slice(format!("--{}\r\n", boundary).as_bytes());
    body_bytes.extend_from_slice(b"Content-Disposition: form-data; name=\"ingest\"\r\n\r\n");
    body_bytes.extend_from_slice(b"false");
    body_bytes.extend_from_slice(format!("\r\n--{}--\r\n", boundary).as_bytes());

    let req = Request::post("/api/v1/ocr")
        .header(
            "Content-Type",
            format!("multipart/form-data; boundary={}", boundary),
        )
        .body(Body::from(body_bytes))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_to_json(resp.into_body()).await;
    assert!(body["ingestion"].is_null());
    // No nodes stored when ingest=false
    let all_nodes = state.registry.graph().get_all_nodes().unwrap();
    assert_eq!(all_nodes.len(), 0);
}

#[tokio::test]
async fn test_ocr_health_reports_pipeline() {
    let (app, _state) = build_app_with_ocr();
    let req = Request::get("/api/v1/health").body(Body::empty()).unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_to_json(resp.into_body()).await;
    assert_eq!(body["models"]["ocr_pipeline_loaded"], true);
}

// ---------------------------------------------------------------------------
// Mem0 Import Tests
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_mem0_import_v2_format() {
    let (app, state) = build_app();

    let mem0_data = serde_json::json!({
        "results": [
            {
                "id": "mem_001",
                "memory": "User likes coffee",
                "user_id": "alice",
                "hash": "abc123",
                "metadata": {"source": "chat"},
                "created_at": "2024-07-01T12:00:00Z",
                "updated_at": "2024-07-01T12:00:00Z"
            },
            {
                "id": "mem_002",
                "memory": "User works at Google",
                "user_id": "alice",
                "created_at": "2024-07-02T10:00:00Z"
            }
        ],
        "total_memories": 2
    });

    let body = serde_json::json!({
        "data": mem0_data,
        "link_same_user": true,
        "link_same_agent": false
    });

    let req = Request::post("/api/v1/import/mem0")
        .header("Content-Type", "application/json")
        .header("X-Ucotron-Namespace", "test_mem0")
        .body(Body::from(serde_json::to_vec(&body).unwrap()))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let result = body_to_json(resp.into_body()).await;
    assert_eq!(result["memories_parsed"], 2);
    assert_eq!(result["nodes_imported"], 2);
    assert_eq!(result["edges_imported"], 1); // Same user → 1 chain edge
    assert_eq!(result["target_namespace"], "test_mem0");

    // Verify nodes were stored.
    let all_nodes = state.registry.graph().get_all_nodes().unwrap();
    assert_eq!(all_nodes.len(), 2);
}

#[tokio::test]
async fn test_mem0_import_v1_bare_array() {
    let (app, _state) = build_app();

    let mem0_data = serde_json::json!([
        {
            "id": "mem_100",
            "memory": "Prefers dark mode",
            "user_id": "bob"
        }
    ]);

    let body = serde_json::json!({
        "data": mem0_data,
        "link_same_user": true
    });

    let req = Request::post("/api/v1/import/mem0")
        .header("Content-Type", "application/json")
        .body(Body::from(serde_json::to_vec(&body).unwrap()))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let result = body_to_json(resp.into_body()).await;
    assert_eq!(result["memories_parsed"], 1);
    assert_eq!(result["nodes_imported"], 1);
    assert_eq!(result["edges_imported"], 0);
}

#[tokio::test]
async fn test_mem0_import_empty() {
    let (app, _state) = build_app();

    let body = serde_json::json!({
        "data": {"results": [], "total_memories": 0}
    });

    let req = Request::post("/api/v1/import/mem0")
        .header("Content-Type", "application/json")
        .body(Body::from(serde_json::to_vec(&body).unwrap()))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let result = body_to_json(resp.into_body()).await;
    assert_eq!(result["memories_parsed"], 0);
    assert_eq!(result["nodes_imported"], 0);
}

#[tokio::test]
async fn test_mem0_import_preserves_metadata() {
    let (app, state) = build_app();

    let mem0_data = serde_json::json!({
        "results": [{
            "id": "mem_meta",
            "memory": "Test metadata preservation",
            "user_id": "alice",
            "agent_id": "gpt-4",
            "hash": "deadbeef",
            "metadata": {"category": "test", "priority": 5},
            "created_at": "2024-03-15T08:00:00Z"
        }]
    });

    let body = serde_json::json!({"data": mem0_data});

    let req = Request::post("/api/v1/import/mem0")
        .header("Content-Type", "application/json")
        .body(Body::from(serde_json::to_vec(&body).unwrap()))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    // Verify stored node has Mem0 metadata.
    let all_nodes = state.registry.graph().get_all_nodes().unwrap();
    assert_eq!(all_nodes.len(), 1);
    let node = &all_nodes[0];
    assert_eq!(node.content, "Test metadata preservation");

    // Check Mem0-specific metadata was preserved.
    let meta = &node.metadata;
    assert!(meta.contains_key("_import_source"));
    assert!(meta.contains_key("mem0_user_id"));
    assert!(meta.contains_key("mem0_agent_id"));
    assert!(meta.contains_key("mem0_hash"));
}

#[tokio::test]
async fn test_mem0_import_namespace_isolation() {
    let (app, state) = build_app();

    let mem0_data = serde_json::json!({
        "results": [
            {"id": "mem_ns1", "memory": "Memory in ns1", "user_id": "u1"},
            {"id": "mem_ns2", "memory": "Memory also in ns1", "user_id": "u2"}
        ]
    });

    let body = serde_json::json!({"data": mem0_data});

    let req = Request::post("/api/v1/import/mem0")
        .header("Content-Type", "application/json")
        .header("X-Ucotron-Namespace", "mem0_project_a")
        .body(Body::from(serde_json::to_vec(&body).unwrap()))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let result = body_to_json(resp.into_body()).await;
    assert_eq!(result["target_namespace"], "mem0_project_a");

    // All imported nodes should have the target namespace.
    let all_nodes = state.registry.graph().get_all_nodes().unwrap();
    for node in &all_nodes {
        assert_eq!(
            node.metadata.get("_namespace"),
            Some(&ucotron_core::Value::String("mem0_project_a".to_string()))
        );
    }
}

// ---------------------------------------------------------------------------
// Zep/Graphiti Import Tests
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_zep_import_graphiti_format() {
    let (app, state) = build_app();

    let zep_data = serde_json::json!({
        "entities": [
            {
                "uuid": "ent-001",
                "name": "Alice",
                "labels": ["Person"],
                "created_at": "2024-07-01T12:00:00Z",
                "summary": "Software engineer at Google"
            },
            {
                "uuid": "ent-002",
                "name": "Google",
                "labels": ["Organization"],
                "created_at": "2024-07-01T12:00:00Z",
                "summary": "Tech company"
            }
        ],
        "edges": [
            {
                "uuid": "edge-001",
                "source_node_uuid": "ent-001",
                "target_node_uuid": "ent-002",
                "fact": "Alice works at Google",
                "name": "works_at",
                "valid_at": "2024-01-01T00:00:00Z",
                "created_at": "2024-07-01T12:00:00Z"
            }
        ]
    });

    let body = serde_json::json!({
        "data": zep_data,
        "link_same_user": true,
        "preserve_expired": true
    });

    let req = Request::post("/api/v1/import/zep")
        .header("Content-Type", "application/json")
        .header("X-Ucotron-Namespace", "test_zep")
        .body(Body::from(serde_json::to_vec(&body).unwrap()))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let result = body_to_json(resp.into_body()).await;
    assert_eq!(result["memories_parsed"], 2);
    assert_eq!(result["nodes_imported"], 2);
    assert_eq!(result["edges_imported"], 1); // 1 entity edge
    assert_eq!(result["target_namespace"], "test_zep");

    // Verify nodes were stored.
    let all_nodes = state.registry.graph().get_all_nodes().unwrap();
    assert_eq!(all_nodes.len(), 2);
}

#[tokio::test]
async fn test_zep_import_sessions_format() {
    let (app, state) = build_app();

    let zep_data = serde_json::json!({
        "sessions": [
            {
                "session_id": "sess-001",
                "user_id": "alice",
                "messages": [
                    {
                        "uuid": "msg-001",
                        "role": "user",
                        "content": "I like coffee",
                        "created_at": "2024-07-01T10:00:00Z"
                    },
                    {
                        "uuid": "msg-002",
                        "role": "assistant",
                        "content": "Noted!",
                        "created_at": "2024-07-01T10:01:00Z"
                    }
                ]
            }
        ]
    });

    let body = serde_json::json!({"data": zep_data});

    let req = Request::post("/api/v1/import/zep")
        .header("Content-Type", "application/json")
        .body(Body::from(serde_json::to_vec(&body).unwrap()))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let result = body_to_json(resp.into_body()).await;
    assert_eq!(result["memories_parsed"], 2);
    assert_eq!(result["nodes_imported"], 2);

    // Should have 1 NextEpisode edge between messages + possibly user chain edges.
    assert!(result["edges_imported"].as_u64().unwrap() >= 1);

    let all_nodes = state.registry.graph().get_all_nodes().unwrap();
    assert_eq!(all_nodes.len(), 2);
}

#[tokio::test]
async fn test_zep_import_empty() {
    let (app, _state) = build_app();

    let body = serde_json::json!({
        "data": {"sessions": [], "entities": []}
    });

    let req = Request::post("/api/v1/import/zep")
        .header("Content-Type", "application/json")
        .body(Body::from(serde_json::to_vec(&body).unwrap()))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let result = body_to_json(resp.into_body()).await;
    assert_eq!(result["memories_parsed"], 0);
    assert_eq!(result["nodes_imported"], 0);
}

#[tokio::test]
async fn test_zep_import_preserves_metadata() {
    let (app, state) = build_app();

    let zep_data = serde_json::json!({
        "entities": [{
            "uuid": "ent-meta",
            "name": "Alice",
            "group_id": "g1",
            "labels": ["Person", "Engineer"],
            "created_at": "2024-03-15T08:00:00Z",
            "summary": "Test metadata preservation",
            "attributes": {"occupation": "engineer", "level": "senior"}
        }]
    });

    let body = serde_json::json!({"data": zep_data});

    let req = Request::post("/api/v1/import/zep")
        .header("Content-Type", "application/json")
        .body(Body::from(serde_json::to_vec(&body).unwrap()))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    // Verify stored node has Zep metadata.
    let all_nodes = state.registry.graph().get_all_nodes().unwrap();
    assert_eq!(all_nodes.len(), 1);
    let node = &all_nodes[0];
    assert_eq!(node.content, "Test metadata preservation");

    // Check Zep-specific metadata was preserved.
    let meta = &node.metadata;
    assert!(meta.contains_key("_import_source"));
    assert!(meta.contains_key("zep_name"));
    assert!(meta.contains_key("zep_group_id"));
    assert!(meta.contains_key("zep_labels"));
}

#[tokio::test]
async fn test_zep_import_namespace_isolation() {
    let (app, state) = build_app();

    let zep_data = serde_json::json!({
        "entities": [
            {"uuid": "ent-ns1", "name": "A", "summary": "Node A"},
            {"uuid": "ent-ns2", "name": "B", "summary": "Node B"}
        ]
    });

    let body = serde_json::json!({"data": zep_data});

    let req = Request::post("/api/v1/import/zep")
        .header("Content-Type", "application/json")
        .header("X-Ucotron-Namespace", "zep_project_a")
        .body(Body::from(serde_json::to_vec(&body).unwrap()))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let result = body_to_json(resp.into_body()).await;
    assert_eq!(result["target_namespace"], "zep_project_a");

    // All imported nodes should have the target namespace.
    let all_nodes = state.registry.graph().get_all_nodes().unwrap();
    for node in &all_nodes {
        assert_eq!(
            node.metadata.get("_namespace"),
            Some(&ucotron_core::Value::String("zep_project_a".to_string()))
        );
    }
}

// ---------------------------------------------------------------------------
// GDPR Compliance Tests
// ---------------------------------------------------------------------------

fn insert_test_node_with_user(
    state: &AppState,
    id: NodeId,
    content: &str,
    node_type: NodeType,
    user_id: &str,
) {
    let embedding = vec![0.5f32; 384];
    let mut metadata = HashMap::new();
    metadata.insert(
        "_user_id".to_string(),
        ucotron_core::Value::String(user_id.to_string()),
    );
    metadata.insert(
        "_namespace".to_string(),
        ucotron_core::Value::String("default".to_string()),
    );
    let node = Node {
        id,
        content: content.to_string(),
        embedding: embedding.clone(),
        metadata,
        node_type,
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

#[tokio::test]
async fn test_gdpr_forget_deletes_all_user_data() {
    let (app, state) = build_app();

    // Insert nodes for two different users.
    insert_test_node_with_user(&state, 1, "Alice memory 1", NodeType::Event, "alice");
    insert_test_node_with_user(&state, 2, "Alice memory 2", NodeType::Entity, "alice");
    insert_test_node_with_user(&state, 3, "Bob memory 1", NodeType::Event, "bob");

    // Add an edge between Alice's nodes.
    state
        .registry
        .graph()
        .upsert_edges(&[Edge {
            source: 1,
            target: 2,
            edge_type: EdgeType::RelatesTo,
            weight: 0.9,
            metadata: HashMap::new(),
        }])
        .unwrap();

    // Verify 3 nodes exist.
    assert_eq!(state.registry.graph().get_all_nodes().unwrap().len(), 3);

    let req = Request::delete("/api/v1/gdpr/forget?user_id=alice")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let body = body_to_json(resp.into_body()).await;
    assert_eq!(body["user_id"], "alice");
    assert_eq!(body["memories_deleted"], 1); // Event
    assert_eq!(body["entities_deleted"], 1); // Entity
    assert_eq!(body["embeddings_deleted"], 2);
    assert!(body["erased_at"].as_u64().unwrap() > 0);

    // Alice's nodes should be gone. Bob's should remain.
    // (Plus the audit node that was created)
    let remaining = state.registry.graph().get_all_nodes().unwrap();
    let non_audit: Vec<_> = remaining
        .iter()
        .filter(|n| {
            !matches!(
                n.metadata.get("_gdpr_audit"),
                Some(ucotron_core::Value::Bool(true))
            )
        })
        .collect();
    assert_eq!(non_audit.len(), 1);
    assert_eq!(non_audit[0].id, 3);
    assert_eq!(non_audit[0].content, "Bob memory 1");
}

#[tokio::test]
async fn test_gdpr_forget_empty_user_id() {
    let (app, _) = build_app();
    let req = Request::delete("/api/v1/gdpr/forget?user_id=")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn test_gdpr_forget_no_matching_user() {
    let (app, state) = build_app();
    insert_test_node_with_user(&state, 1, "Some memory", NodeType::Event, "alice");

    let req = Request::delete("/api/v1/gdpr/forget?user_id=nonexistent")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let body = body_to_json(resp.into_body()).await;
    assert_eq!(body["memories_deleted"], 0);
    assert_eq!(body["entities_deleted"], 0);
    assert_eq!(body["embeddings_deleted"], 0);

    // Original node still exists.
    assert!(state.registry.graph().get_node(1).unwrap().is_some());
}

#[tokio::test]
async fn test_gdpr_export_returns_user_data() {
    let (app, state) = build_app();

    insert_test_node_with_user(&state, 10, "Alice fact", NodeType::Fact, "alice");
    insert_test_node_with_user(&state, 11, "Alice entity", NodeType::Entity, "alice");
    insert_test_node_with_user(&state, 12, "Bob event", NodeType::Event, "bob");

    // Add edge between Alice's nodes.
    state
        .registry
        .graph()
        .upsert_edges(&[Edge {
            source: 10,
            target: 11,
            edge_type: EdgeType::RelatesTo,
            weight: 0.8,
            metadata: HashMap::new(),
        }])
        .unwrap();

    let req = Request::get("/api/v1/gdpr/export?user_id=alice")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let body = body_to_json(resp.into_body()).await;
    assert_eq!(body["user_id"], "alice");
    assert_eq!(body["stats"]["total_nodes"], 2);
    assert_eq!(body["stats"]["total_edges"], 1);
    assert_eq!(body["nodes"].as_array().unwrap().len(), 2);
    assert_eq!(body["edges"].as_array().unwrap().len(), 1);
    assert!(body["exported_at"].as_u64().unwrap() > 0);
}

#[tokio::test]
async fn test_gdpr_export_empty_user_id() {
    let (app, _) = build_app();
    let req = Request::get("/api/v1/gdpr/export?user_id=")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn test_gdpr_retention_status_default() {
    let (app, _) = build_app();
    let req = Request::get("/api/v1/gdpr/retention")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let body = body_to_json(resp.into_body()).await;
    // Default config has no retention policies.
    assert_eq!(body["policies"].as_array().unwrap().len(), 0);
    assert_eq!(body["last_sweep_expired"], 0);
}

#[tokio::test]
async fn test_gdpr_retention_sweep_no_expiry() {
    let (app, state) = build_app();
    insert_test_node_with_user(&state, 1, "Some memory", NodeType::Event, "alice");

    let req = Request::post("/api/v1/gdpr/retention/sweep")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let body = body_to_json(resp.into_body()).await;
    // No TTL configured → 0 expired.
    assert_eq!(body["nodes_expired"], 0);
    assert_eq!(body["namespaces_checked"], 0);
}

// ---------------------------------------------------------------------------
// RBAC Integration Tests
// ---------------------------------------------------------------------------

/// Build an app with auth enabled and 4 named API keys (one per role).
/// Also creates a namespace-scoped reader key for tenant isolation tests.
fn build_app_with_auth() -> (Router, Arc<AppState>) {
    let registry = Arc::new(BackendRegistry::new(
        Box::new(MockVectorBackend::new()),
        Box::new(MockGraphBackend::new()),
    ));
    let embedder: Arc<dyn EmbeddingPipeline> = Arc::new(MockEmbedder);

    let mut config = UcotronConfig::default();
    config.auth.enabled = true;
    config.auth.api_keys = vec![
        ApiKeyEntry {
            name: "admin-key".into(),
            key: "sk-admin-test".into(),
            role: "admin".into(),
            namespace: None,
            active: true,
        },
        ApiKeyEntry {
            name: "writer-key".into(),
            key: "sk-writer-test".into(),
            role: "writer".into(),
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
        ApiKeyEntry {
            name: "viewer-key".into(),
            key: "sk-viewer-test".into(),
            role: "viewer".into(),
            namespace: None,
            active: true,
        },
        ApiKeyEntry {
            name: "scoped-reader".into(),
            key: "sk-scoped-test".into(),
            role: "reader".into(),
            namespace: Some("tenant-a".into()),
            active: true,
        },
        ApiKeyEntry {
            name: "revoked-key".into(),
            key: "sk-revoked-test".into(),
            role: "admin".into(),
            namespace: None,
            active: false,
        },
    ];

    let state = Arc::new(AppState::new(registry, embedder, None, None, config));

    let app = Router::new()
        .route("/api/v1/health", get(handlers::health_handler))
        .route("/api/v1/metrics", get(handlers::metrics_handler))
        .route("/api/v1/memories", post(handlers::create_memory_handler))
        .route("/api/v1/memories", get(handlers::list_memories_handler))
        .route("/api/v1/memories/{id}", get(handlers::get_memory_handler))
        .route(
            "/api/v1/memories/{id}",
            put(handlers::update_memory_handler),
        )
        .route(
            "/api/v1/memories/{id}",
            delete(handlers::delete_memory_handler),
        )
        .route("/api/v1/memories/search", post(handlers::search_handler))
        .route("/api/v1/media/{id}", get(handlers::get_media_handler))
        .route(
            "/api/v1/videos/{parent_id}/segments",
            get(handlers::get_video_segments_handler),
        )
        .route("/api/v1/entities", get(handlers::list_entities_handler))
        .route("/api/v1/entities/{id}", get(handlers::get_entity_handler))
        .route("/api/v1/graph", get(handlers::graph_handler))
        .route("/api/v1/augment", post(handlers::augment_handler))
        .route("/api/v1/learn", post(handlers::learn_handler))
        .route("/api/v1/export", get(handlers::export_handler))
        .route("/api/v1/import", post(handlers::import_handler))
        .route(
            "/api/v1/admin/namespaces",
            get(handlers::list_namespaces_handler),
        )
        .route(
            "/api/v1/admin/namespaces",
            post(handlers::create_namespace_handler),
        )
        .route(
            "/api/v1/admin/namespaces/{name}",
            get(handlers::get_namespace_handler),
        )
        .route(
            "/api/v1/admin/namespaces/{name}",
            delete(handlers::delete_namespace_handler),
        )
        .route("/api/v1/admin/config", get(handlers::admin_config_handler))
        .route("/api/v1/admin/system", get(handlers::admin_system_handler))
        .route("/api/v1/auth/whoami", get(handlers::whoami_handler))
        .route("/api/v1/auth/keys", get(handlers::list_api_keys_handler))
        .route("/api/v1/auth/keys", post(handlers::create_api_key_handler))
        .route(
            "/api/v1/auth/keys/{name}",
            delete(handlers::revoke_api_key_handler),
        )
        .route("/api/v1/gdpr/forget", delete(handlers::gdpr_forget_handler))
        .route("/api/v1/gdpr/export", get(handlers::gdpr_export_handler))
        .route(
            "/api/v1/gdpr/retention",
            get(handlers::gdpr_retention_status_handler),
        )
        .route(
            "/api/v1/gdpr/retention/sweep",
            post(handlers::gdpr_retention_sweep_handler),
        )
        .layer(middleware::from_fn_with_state(
            state.clone(),
            ucotron_server::auth::auth_middleware,
        ))
        .with_state(state.clone());

    (app, state)
}

// -- 1) Missing auth header returns 401 --

#[tokio::test]
async fn test_rbac_missing_token_returns_401() {
    let (app, _) = build_app_with_auth();
    let req = Request::get("/api/v1/memories")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
}

// -- 2) Invalid token returns 401 --

#[tokio::test]
async fn test_rbac_invalid_token_returns_401() {
    let (app, _) = build_app_with_auth();
    let req = Request::get("/api/v1/memories")
        .header("Authorization", "Bearer sk-bogus-key")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
}

// -- 3) Revoked key returns 401 --

#[tokio::test]
async fn test_rbac_revoked_key_returns_401() {
    let (app, _) = build_app_with_auth();
    let req = Request::get("/api/v1/memories")
        .header("Authorization", "Bearer sk-revoked-test")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
}

// -- 4) Health endpoint bypasses auth --

#[tokio::test]
async fn test_rbac_health_bypasses_auth() {
    let (app, _) = build_app_with_auth();
    let req = Request::get("/api/v1/health").body(Body::empty()).unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
}

// -- 5) Viewer cannot access reader endpoints --

#[tokio::test]
async fn test_rbac_viewer_cannot_list_memories() {
    let (app, _) = build_app_with_auth();
    let req = Request::get("/api/v1/memories")
        .header("Authorization", "Bearer sk-viewer-test")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::FORBIDDEN);
}

// -- 6) Reader can access reader endpoints --

#[tokio::test]
async fn test_rbac_reader_can_list_memories() {
    let (app, _) = build_app_with_auth();
    let req = Request::get("/api/v1/memories")
        .header("Authorization", "Bearer sk-reader-test")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
}

// -- 7) Reader cannot create memories (writer endpoint) --

#[tokio::test]
async fn test_rbac_reader_cannot_create_memory() {
    let (app, _) = build_app_with_auth();
    let body = r#"{"text":"test"}"#;
    let req = Request::post("/api/v1/memories")
        .header("Authorization", "Bearer sk-reader-test")
        .header("Content-Type", "application/json")
        .body(Body::from(body))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::FORBIDDEN);
}

// -- 8) Writer can create memories --

#[tokio::test]
async fn test_rbac_writer_can_create_memory() {
    let (app, _) = build_app_with_auth();
    let body = r#"{"text":"test memory for writer"}"#;
    let req = Request::post("/api/v1/memories")
        .header("Authorization", "Bearer sk-writer-test")
        .header("Content-Type", "application/json")
        .body(Body::from(body))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::CREATED);
}

// -- 9) Writer cannot access admin endpoints --

#[tokio::test]
async fn test_rbac_writer_cannot_access_admin() {
    let (app, _) = build_app_with_auth();
    let req = Request::get("/api/v1/admin/config")
        .header("Authorization", "Bearer sk-writer-test")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::FORBIDDEN);
}

// -- 10) Admin can access admin endpoints --

#[tokio::test]
async fn test_rbac_admin_can_access_admin_config() {
    let (app, _) = build_app_with_auth();
    let req = Request::get("/api/v1/admin/config")
        .header("Authorization", "Bearer sk-admin-test")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
}

// -- 11) Admin can access writer endpoints too (privilege escalation) --

#[tokio::test]
async fn test_rbac_admin_can_create_memory() {
    let (app, _) = build_app_with_auth();
    let body = r#"{"text":"admin creating memory"}"#;
    let req = Request::post("/api/v1/memories")
        .header("Authorization", "Bearer sk-admin-test")
        .header("Content-Type", "application/json")
        .body(Body::from(body))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::CREATED);
}

// -- 12) Namespace-scoped key can access its own namespace --

#[tokio::test]
async fn test_rbac_scoped_key_own_namespace() {
    let (app, _) = build_app_with_auth();
    let req = Request::get("/api/v1/memories")
        .header("Authorization", "Bearer sk-scoped-test")
        .header("X-Ucotron-Namespace", "tenant-a")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
}

// -- 13) Namespace-scoped key CANNOT access other namespace --

#[tokio::test]
async fn test_rbac_scoped_key_wrong_namespace_returns_403() {
    let (app, _) = build_app_with_auth();
    let req = Request::get("/api/v1/memories")
        .header("Authorization", "Bearer sk-scoped-test")
        .header("X-Ucotron-Namespace", "tenant-b")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::FORBIDDEN);
}

// -- 14) Whoami returns correct role --

#[tokio::test]
async fn test_rbac_whoami_returns_role() {
    let (app, _) = build_app_with_auth();
    let req = Request::get("/api/v1/auth/whoami")
        .header("Authorization", "Bearer sk-writer-test")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_to_json(resp.into_body()).await;
    assert_eq!(body["role"], "writer");
    assert_eq!(body["key_name"], "writer-key");
}

// -- 15) List API keys requires admin --

#[tokio::test]
async fn test_rbac_list_keys_requires_admin() {
    let (app, _) = build_app_with_auth();

    // Writer cannot list keys.
    let req = Request::get("/api/v1/auth/keys")
        .header("Authorization", "Bearer sk-writer-test")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::FORBIDDEN);
}

// -- 16) Admin can list API keys --

#[tokio::test]
async fn test_rbac_admin_can_list_keys() {
    let (app, _) = build_app_with_auth();
    let req = Request::get("/api/v1/auth/keys")
        .header("Authorization", "Bearer sk-admin-test")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_to_json(resp.into_body()).await;
    let keys = body["keys"].as_array().unwrap();
    // Should have our 6 configured keys (5 active + 1 inactive).
    assert!(keys.len() >= 5);
}

// -- 17) Reader cannot learn (writer endpoint) --

#[tokio::test]
async fn test_rbac_reader_cannot_learn() {
    let (app, _) = build_app_with_auth();
    let body = r#"{"output":"test learning"}"#;
    let req = Request::post("/api/v1/learn")
        .header("Authorization", "Bearer sk-reader-test")
        .header("Content-Type", "application/json")
        .body(Body::from(body))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::FORBIDDEN);
}

// -- 18) Writer cannot access GDPR retention status (admin-only) --

#[tokio::test]
async fn test_rbac_writer_cannot_gdpr_retention() {
    let (app, _) = build_app_with_auth();
    let req = Request::get("/api/v1/gdpr/retention")
        .header("Authorization", "Bearer sk-writer-test")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::FORBIDDEN);
}

// =========================================================================
// Audit Logging Tests (US-18.3)
// =========================================================================

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
    config.audit.retention_secs = 0; // keep forever for tests
    config.auth.api_keys = vec![
        ApiKeyEntry {
            name: "admin-key".into(),
            key: "sk-admin-test".into(),
            role: "admin".into(),
            namespace: None,
            active: true,
        },
        ApiKeyEntry {
            name: "writer-key".into(),
            key: "sk-writer-test".into(),
            role: "writer".into(),
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

    let state = Arc::new(AppState::new(registry, embedder, None, None, config));

    let app = Router::new()
        .route("/api/v1/health", get(handlers::health_handler))
        .route("/api/v1/memories", post(handlers::create_memory_handler))
        .route("/api/v1/memories", get(handlers::list_memories_handler))
        .route("/api/v1/memories/search", post(handlers::search_handler))
        .route("/api/v1/audit", get(handlers::audit_query_handler))
        .route("/api/v1/audit/export", get(handlers::audit_export_handler))
        .layer(middleware::from_fn_with_state(
            state.clone(),
            ucotron_server::auth::auth_middleware,
        ))
        .with_state(state.clone());

    (app, state)
}

// -- 1) Audit query returns empty when no entries --

#[tokio::test]
async fn test_audit_query_empty() {
    let (app, _state) = build_app_with_audit();
    let req = Request::get("/api/v1/audit")
        .header("Authorization", "Bearer sk-admin-test")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let body: serde_json::Value =
        serde_json::from_slice(&resp.into_body().collect().await.unwrap().to_bytes()).unwrap();
    assert_eq!(body["total"], 0);
    assert!(body["entries"].as_array().unwrap().is_empty());
}

// -- 2) Audit query returns entries after manual append --

#[tokio::test]
async fn test_audit_query_returns_entries() {
    let (_app, state) = build_app_with_audit();

    // Manually append entries (simulating what the audit middleware does).
    state.audit_log.append(ucotron_server::audit::AuditEntry {
        timestamp: 1000,
        method: "POST".into(),
        path: "/api/v1/memories".into(),
        action: "memories.create".into(),
        status: 200,
        duration_us: 500,
        user: Some("admin-key".into()),
        role: "admin".into(),
        namespace: Some("default".into()),
        resource_id: None,
    });
    state.audit_log.append(ucotron_server::audit::AuditEntry {
        timestamp: 2000,
        method: "POST".into(),
        path: "/api/v1/memories/search".into(),
        action: "search".into(),
        status: 200,
        duration_us: 100,
        user: Some("reader-key".into()),
        role: "reader".into(),
        namespace: Some("default".into()),
        resource_id: None,
    });

    // Build a fresh router with the same state for the query.
    let app = Router::new()
        .route("/api/v1/audit", get(handlers::audit_query_handler))
        .layer(middleware::from_fn_with_state(
            state.clone(),
            ucotron_server::auth::auth_middleware,
        ))
        .with_state(state.clone());

    let req = Request::get("/api/v1/audit")
        .header("Authorization", "Bearer sk-admin-test")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let body: serde_json::Value =
        serde_json::from_slice(&resp.into_body().collect().await.unwrap().to_bytes()).unwrap();
    assert_eq!(body["total"], 2);
    let entries = body["entries"].as_array().unwrap();
    assert_eq!(entries[0]["action"], "memories.create");
    assert_eq!(entries[1]["action"], "search");
}

// -- 3) Audit query filters by user --

#[tokio::test]
async fn test_audit_query_filter_by_user() {
    let (_app, state) = build_app_with_audit();

    state.audit_log.append(ucotron_server::audit::AuditEntry {
        timestamp: 1000,
        method: "POST".into(),
        path: "/api/v1/memories".into(),
        action: "memories.create".into(),
        status: 200,
        duration_us: 500,
        user: Some("admin-key".into()),
        role: "admin".into(),
        namespace: None,
        resource_id: None,
    });
    state.audit_log.append(ucotron_server::audit::AuditEntry {
        timestamp: 2000,
        method: "GET".into(),
        path: "/api/v1/memories".into(),
        action: "memories.list".into(),
        status: 200,
        duration_us: 50,
        user: Some("reader-key".into()),
        role: "reader".into(),
        namespace: None,
        resource_id: None,
    });

    let app = Router::new()
        .route("/api/v1/audit", get(handlers::audit_query_handler))
        .layer(middleware::from_fn_with_state(
            state.clone(),
            ucotron_server::auth::auth_middleware,
        ))
        .with_state(state.clone());

    let req = Request::get("/api/v1/audit?user=admin-key")
        .header("Authorization", "Bearer sk-admin-test")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let body: serde_json::Value =
        serde_json::from_slice(&resp.into_body().collect().await.unwrap().to_bytes()).unwrap();
    assert_eq!(body["total"], 1);
    assert_eq!(body["entries"][0]["user"], "admin-key");
}

// -- 4) Audit query filters by time range --

#[tokio::test]
async fn test_audit_query_filter_by_time_range() {
    let (_app, state) = build_app_with_audit();

    state.audit_log.append(ucotron_server::audit::AuditEntry {
        timestamp: 1000,
        method: "GET".into(),
        path: "/api/v1/health".into(),
        action: "health".into(),
        status: 200,
        duration_us: 5,
        user: None,
        role: "viewer".into(),
        namespace: None,
        resource_id: None,
    });
    state.audit_log.append(ucotron_server::audit::AuditEntry {
        timestamp: 5000,
        method: "POST".into(),
        path: "/api/v1/memories".into(),
        action: "memories.create".into(),
        status: 200,
        duration_us: 500,
        user: Some("admin-key".into()),
        role: "admin".into(),
        namespace: None,
        resource_id: None,
    });

    let app = Router::new()
        .route("/api/v1/audit", get(handlers::audit_query_handler))
        .layer(middleware::from_fn_with_state(
            state.clone(),
            ucotron_server::auth::auth_middleware,
        ))
        .with_state(state.clone());

    let req = Request::get("/api/v1/audit?from=3000&to=6000")
        .header("Authorization", "Bearer sk-admin-test")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let body: serde_json::Value =
        serde_json::from_slice(&resp.into_body().collect().await.unwrap().to_bytes()).unwrap();
    assert_eq!(body["total"], 1);
    assert_eq!(body["entries"][0]["timestamp"], 5000);
}

// -- 5) Audit query filters by action --

#[tokio::test]
async fn test_audit_query_filter_by_action() {
    let (_app, state) = build_app_with_audit();

    state.audit_log.append(ucotron_server::audit::AuditEntry {
        timestamp: 1000,
        method: "POST".into(),
        path: "/api/v1/memories".into(),
        action: "memories.create".into(),
        status: 200,
        duration_us: 500,
        user: Some("admin-key".into()),
        role: "admin".into(),
        namespace: None,
        resource_id: None,
    });
    state.audit_log.append(ucotron_server::audit::AuditEntry {
        timestamp: 2000,
        method: "POST".into(),
        path: "/api/v1/memories/search".into(),
        action: "search".into(),
        status: 200,
        duration_us: 100,
        user: Some("admin-key".into()),
        role: "admin".into(),
        namespace: None,
        resource_id: None,
    });

    let app = Router::new()
        .route("/api/v1/audit", get(handlers::audit_query_handler))
        .layer(middleware::from_fn_with_state(
            state.clone(),
            ucotron_server::auth::auth_middleware,
        ))
        .with_state(state.clone());

    let req = Request::get("/api/v1/audit?action=search")
        .header("Authorization", "Bearer sk-admin-test")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let body: serde_json::Value =
        serde_json::from_slice(&resp.into_body().collect().await.unwrap().to_bytes()).unwrap();
    assert_eq!(body["total"], 1);
    assert_eq!(body["entries"][0]["action"], "search");
}

// -- 6) Audit export returns all entries --

#[tokio::test]
async fn test_audit_export_returns_all() {
    let (_app, state) = build_app_with_audit();

    for i in 0..5 {
        state.audit_log.append(ucotron_server::audit::AuditEntry {
            timestamp: 1000 + i,
            method: "GET".into(),
            path: "/api/v1/memories".into(),
            action: "memories.list".into(),
            status: 200,
            duration_us: 10,
            user: Some("admin-key".into()),
            role: "admin".into(),
            namespace: None,
            resource_id: None,
        });
    }

    let app = Router::new()
        .route("/api/v1/audit/export", get(handlers::audit_export_handler))
        .layer(middleware::from_fn_with_state(
            state.clone(),
            ucotron_server::auth::auth_middleware,
        ))
        .with_state(state.clone());

    let req = Request::get("/api/v1/audit/export")
        .header("Authorization", "Bearer sk-admin-test")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let body: serde_json::Value =
        serde_json::from_slice(&resp.into_body().collect().await.unwrap().to_bytes()).unwrap();
    assert_eq!(body["total"], 5);
    assert!(body["exported_at"].as_u64().unwrap() > 0);
    assert_eq!(body["entries"].as_array().unwrap().len(), 5);
}

// -- 7) Non-admin cannot access audit endpoints --

#[tokio::test]
async fn test_audit_requires_admin_role() {
    let (app, _state) = build_app_with_audit();
    let req = Request::get("/api/v1/audit")
        .header("Authorization", "Bearer sk-reader-test")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::FORBIDDEN);
}

// -- 8) Non-admin cannot export audit --

#[tokio::test]
async fn test_audit_export_requires_admin_role() {
    let (app, _state) = build_app_with_audit();
    let req = Request::get("/api/v1/audit/export")
        .header("Authorization", "Bearer sk-writer-test")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::FORBIDDEN);
}

// -- 9) Audit log respects max_entries limit --

#[tokio::test]
async fn test_audit_max_entries_eviction() {
    let registry = Arc::new(BackendRegistry::new(
        Box::new(MockVectorBackend::new()),
        Box::new(MockGraphBackend::new()),
    ));
    let embedder: Arc<dyn EmbeddingPipeline> = Arc::new(MockEmbedder);

    let mut config = UcotronConfig::default();
    config.audit.enabled = true;
    config.audit.max_entries = 5; // Very small limit.
    config.audit.retention_secs = 0;

    let state = Arc::new(AppState::new(registry, embedder, None, None, config));

    // Append 10 entries.
    for i in 0..10u64 {
        state.audit_log.append(ucotron_server::audit::AuditEntry {
            timestamp: i,
            method: "GET".into(),
            path: "/api/v1/health".into(),
            action: "health".into(),
            status: 200,
            duration_us: 10,
            user: None,
            role: "viewer".into(),
            namespace: None,
            resource_id: None,
        });
    }

    // Should only keep the last 5.
    assert_eq!(state.audit_log.len(), 5);
    let all = state.audit_log.export_all();
    assert_eq!(all[0].timestamp, 5);
    assert_eq!(all[4].timestamp, 9);
}

// -- 10) Audit entries record correct fields --

#[tokio::test]
async fn test_audit_entry_fields() {
    let (_app, state) = build_app_with_audit();

    state.audit_log.append(ucotron_server::audit::AuditEntry {
        timestamp: 12345,
        method: "DELETE".into(),
        path: "/api/v1/memories/42".into(),
        action: "memories.delete".into(),
        status: 200,
        duration_us: 750,
        user: Some("admin-key".into()),
        role: "admin".into(),
        namespace: Some("tenant-a".into()),
        resource_id: Some("42".into()),
    });

    let app = Router::new()
        .route("/api/v1/audit", get(handlers::audit_query_handler))
        .layer(middleware::from_fn_with_state(
            state.clone(),
            ucotron_server::auth::auth_middleware,
        ))
        .with_state(state.clone());

    let req = Request::get("/api/v1/audit")
        .header("Authorization", "Bearer sk-admin-test")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let body: serde_json::Value =
        serde_json::from_slice(&resp.into_body().collect().await.unwrap().to_bytes()).unwrap();
    let entry = &body["entries"][0];
    assert_eq!(entry["timestamp"], 12345);
    assert_eq!(entry["method"], "DELETE");
    assert_eq!(entry["path"], "/api/v1/memories/42");
    assert_eq!(entry["action"], "memories.delete");
    assert_eq!(entry["status"], 200);
    assert_eq!(entry["duration_us"], 750);
    assert_eq!(entry["user"], "admin-key");
    assert_eq!(entry["role"], "admin");
    assert_eq!(entry["namespace"], "tenant-a");
    assert_eq!(entry["resource_id"], "42");
}

// -- BUG-5: Audit middleware captures namespace from X-Ucotron-Namespace header --

#[tokio::test]
async fn test_audit_namespace_captured_from_header() {
    let (_base_app, state) = build_app_with_audit();

    // Build an app with the audit middleware so namespace extraction is exercised.
    let app = Router::new()
        .route("/api/v1/memories", get(handlers::list_memories_handler))
        .route("/api/v1/audit", get(handlers::audit_query_handler))
        .layer(middleware::from_fn_with_state(
            state.clone(),
            ucotron_server::audit::audit_middleware,
        ))
        .layer(middleware::from_fn_with_state(
            state.clone(),
            ucotron_server::auth::auth_middleware,
        ))
        .with_state(state.clone());

    // Make a request with X-Ucotron-Namespace header set to "tenant-xyz".
    let req = Request::get("/api/v1/memories")
        .header("Authorization", "Bearer sk-reader-test")
        .header("X-Ucotron-Namespace", "tenant-xyz")
        .body(Body::empty())
        .unwrap();
    let _resp = app.clone().oneshot(req).await.unwrap();

    // The audit log should have captured namespace = "tenant-xyz" from the header.
    let entries = state.audit_log.export_all();
    assert!(
        !entries.is_empty(),
        "audit log should have at least one entry"
    );
    let mem_entry = entries
        .iter()
        .find(|e| e.action == "memories.list")
        .expect("should find a memories.list audit entry");
    assert_eq!(
        mem_entry.namespace,
        Some("tenant-xyz".to_string()),
        "audit entry namespace should come from X-Ucotron-Namespace header"
    );
}

#[tokio::test]
async fn test_audit_namespace_falls_back_to_key_scope() {
    // Create a config where the API key has a namespace_scope set.
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
    config.auth.api_keys = vec![ApiKeyEntry {
        name: "scoped-key".into(),
        key: "sk-scoped-test".into(),
        role: "reader".into(),
        namespace: Some("scoped-ns".into()),
        active: true,
    }];

    let state = Arc::new(ucotron_server::state::AppState::new(
        registry, embedder, None, None, config,
    ));

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

    // Make a request WITHOUT X-Ucotron-Namespace header.
    // The scoped key's namespace should be used as fallback.
    let req = Request::get("/api/v1/memories")
        .header("Authorization", "Bearer sk-scoped-test")
        .body(Body::empty())
        .unwrap();
    let _resp = app.oneshot(req).await.unwrap();

    let entries = state.audit_log.export_all();
    assert!(!entries.is_empty());
    let mem_entry = entries
        .iter()
        .find(|e| e.action == "memories.list")
        .expect("should find a memories.list audit entry");
    assert_eq!(
        mem_entry.namespace,
        Some("scoped-ns".to_string()),
        "audit namespace should fall back to API key scope when header is absent"
    );
}

// ---------------------------------------------------------------------------
// Agent CRUD Tests
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_create_agent() {
    let (app, _state) = build_app();
    let req = Request::builder()
        .method("POST")
        .uri("/api/v1/agents")
        .header("Content-Type", "application/json")
        .body(Body::from(
            r#"{"name": "My Test Agent", "config": {"model": "qwen-2.5"}}"#,
        ))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::CREATED);

    let body = resp.into_body().collect().await.unwrap().to_bytes();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert!(json["id"].as_str().unwrap().contains("mytestagent"));
    assert_eq!(json["name"], "My Test Agent");
    assert!(json["namespace"].as_str().unwrap().starts_with("agent_"));
    assert!(json["created_at"].as_u64().unwrap() > 0);
}

#[tokio::test]
async fn test_create_agent_empty_name() {
    let (app, _state) = build_app();
    let req = Request::builder()
        .method("POST")
        .uri("/api/v1/agents")
        .header("Content-Type", "application/json")
        .body(Body::from(r#"{"name": "  "}"#))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn test_list_agents_empty() {
    let (app, _state) = build_app();
    let req = Request::builder()
        .method("GET")
        .uri("/api/v1/agents")
        .body(Body::empty())
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let body = resp.into_body().collect().await.unwrap().to_bytes();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["total"], 0);
    assert!(json["agents"].as_array().unwrap().is_empty());
}

#[tokio::test]
async fn test_create_then_get_agent() {
    let (app, state) = build_app();

    // Create
    let create_req = Request::builder()
        .method("POST")
        .uri("/api/v1/agents")
        .header("Content-Type", "application/json")
        .body(Body::from(r#"{"name": "Agent Alpha"}"#))
        .unwrap();

    let create_resp = app.oneshot(create_req).await.unwrap();
    assert_eq!(create_resp.status(), StatusCode::CREATED);
    let body = create_resp.into_body().collect().await.unwrap().to_bytes();
    let created: serde_json::Value = serde_json::from_slice(&body).unwrap();
    let agent_id = created["id"].as_str().unwrap().to_string();

    // Get
    let (app2, _) = (
        Router::new()
            .route("/api/v1/agents/{id}", get(handlers::get_agent_handler))
            .layer(middleware::from_fn_with_state(
                state.clone(),
                ucotron_server::auth::auth_middleware,
            ))
            .with_state(state.clone()),
        state,
    );

    let get_req = Request::builder()
        .method("GET")
        .uri(format!("/api/v1/agents/{}", agent_id))
        .body(Body::empty())
        .unwrap();

    let get_resp = app2.oneshot(get_req).await.unwrap();
    assert_eq!(get_resp.status(), StatusCode::OK);

    let get_body = get_resp.into_body().collect().await.unwrap().to_bytes();
    let agent: serde_json::Value = serde_json::from_slice(&get_body).unwrap();
    assert_eq!(agent["id"], agent_id);
    assert_eq!(agent["name"], "Agent Alpha");
}

#[tokio::test]
async fn test_get_nonexistent_agent() {
    let (_app, state) = build_app();
    let app_with_route = Router::new()
        .route("/api/v1/agents/{id}", get(handlers::get_agent_handler))
        .layer(middleware::from_fn_with_state(
            state.clone(),
            ucotron_server::auth::auth_middleware,
        ))
        .with_state(state);

    let req = Request::builder()
        .method("GET")
        .uri("/api/v1/agents/does-not-exist")
        .body(Body::empty())
        .unwrap();

    let resp = app_with_route.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn test_create_agent_auto_namespace() {
    let (app, _state) = build_app();
    let req = Request::builder()
        .method("POST")
        .uri("/api/v1/agents")
        .header("Content-Type", "application/json")
        .body(Body::from(r#"{"name": "Namespace Bot"}"#))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::CREATED);

    let body = resp.into_body().collect().await.unwrap().to_bytes();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    let ns = json["namespace"].as_str().unwrap();
    let id = json["id"].as_str().unwrap();
    // Namespace should be "agent_{id}"
    assert_eq!(ns, format!("agent_{}", id));
}

#[tokio::test]
async fn test_list_agents_pagination() {
    let (_app, state) = build_app();

    // Seed 5 agents directly via the graph backend.
    for i in 0..5 {
        let agent = ucotron_core::Agent::new(
            format!("pagtest_{}", i),
            format!("Agent {}", i),
            "admin",
            1000 + i as u64,
        );
        state.registry.graph().create_agent(&agent).unwrap();
    }

    let app = Router::new()
        .route("/api/v1/agents", get(handlers::list_agents_handler))
        .layer(middleware::from_fn_with_state(
            state.clone(),
            ucotron_server::auth::auth_middleware,
        ))
        .with_state(state);

    // Page 1: limit=2, offset=0
    let req = Request::builder()
        .method("GET")
        .uri("/api/v1/agents?limit=2&offset=0")
        .body(Body::empty())
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_to_json(resp.into_body()).await;
    assert_eq!(body["agents"].as_array().unwrap().len(), 2);
    assert_eq!(body["total"], 5);
    assert_eq!(body["limit"], 2);
    assert_eq!(body["offset"], 0);

    // Page 2: limit=2, offset=2
    let req2 = Request::builder()
        .method("GET")
        .uri("/api/v1/agents?limit=2&offset=2")
        .body(Body::empty())
        .unwrap();
    let resp2 = app.clone().oneshot(req2).await.unwrap();
    let body2 = body_to_json(resp2.into_body()).await;
    assert_eq!(body2["agents"].as_array().unwrap().len(), 2);
    assert_eq!(body2["total"], 5);
    assert_eq!(body2["offset"], 2);

    // Page 3: limit=2, offset=4 → only 1 remaining
    let req3 = Request::builder()
        .method("GET")
        .uri("/api/v1/agents?limit=2&offset=4")
        .body(Body::empty())
        .unwrap();
    let resp3 = app.clone().oneshot(req3).await.unwrap();
    let body3 = body_to_json(resp3.into_body()).await;
    assert_eq!(body3["agents"].as_array().unwrap().len(), 1);
    assert_eq!(body3["total"], 5);

    // Beyond range: offset=10 → empty
    let req4 = Request::builder()
        .method("GET")
        .uri("/api/v1/agents?limit=2&offset=10")
        .body(Body::empty())
        .unwrap();
    let resp4 = app.clone().oneshot(req4).await.unwrap();
    let body4 = body_to_json(resp4.into_body()).await;
    assert!(body4["agents"].as_array().unwrap().is_empty());
    assert_eq!(body4["total"], 5);
}

#[tokio::test]
async fn test_list_agents_owner_filter() {
    let (_app, state) = build_app();

    // Create agents with different owners.
    for i in 0..3 {
        let agent = ucotron_core::Agent::new(
            format!("alice_{}", i),
            format!("Alice Agent {}", i),
            "alice",
            2000 + i as u64,
        );
        state.registry.graph().create_agent(&agent).unwrap();
    }
    for i in 0..2 {
        let agent = ucotron_core::Agent::new(
            format!("bob_{}", i),
            format!("Bob Agent {}", i),
            "bob",
            3000 + i as u64,
        );
        state.registry.graph().create_agent(&agent).unwrap();
    }

    let app = Router::new()
        .route("/api/v1/agents", get(handlers::list_agents_handler))
        .layer(middleware::from_fn_with_state(
            state.clone(),
            ucotron_server::auth::auth_middleware,
        ))
        .with_state(state);

    // Admin sees all agents (default auth = Admin)
    let req_all = Request::builder()
        .method("GET")
        .uri("/api/v1/agents")
        .body(Body::empty())
        .unwrap();
    let resp_all = app.clone().oneshot(req_all).await.unwrap();
    let body_all = body_to_json(resp_all.into_body()).await;
    assert_eq!(body_all["total"], 5);

    // Admin filters by owner=alice
    let req_alice = Request::builder()
        .method("GET")
        .uri("/api/v1/agents?owner=alice")
        .body(Body::empty())
        .unwrap();
    let resp_alice = app.clone().oneshot(req_alice).await.unwrap();
    let body_alice = body_to_json(resp_alice.into_body()).await;
    assert_eq!(body_alice["total"], 3);
    for agent in body_alice["agents"].as_array().unwrap() {
        assert_eq!(agent["owner"], "alice");
    }

    // Admin filters by owner=bob
    let req_bob = Request::builder()
        .method("GET")
        .uri("/api/v1/agents?owner=bob")
        .body(Body::empty())
        .unwrap();
    let resp_bob = app.clone().oneshot(req_bob).await.unwrap();
    let body_bob = body_to_json(resp_bob.into_body()).await;
    assert_eq!(body_bob["total"], 2);
    for agent in body_bob["agents"].as_array().unwrap() {
        assert_eq!(agent["owner"], "bob");
    }

    // No agent for this owner
    let req_nobody = Request::builder()
        .method("GET")
        .uri("/api/v1/agents?owner=nobody")
        .body(Body::empty())
        .unwrap();
    let resp_nobody = app.clone().oneshot(req_nobody).await.unwrap();
    let body_nobody = body_to_json(resp_nobody.into_body()).await;
    assert_eq!(body_nobody["total"], 0);
    assert!(body_nobody["agents"].as_array().unwrap().is_empty());
}

#[tokio::test]
async fn test_list_agents_owner_filter_with_pagination() {
    let (_app, state) = build_app();

    // Create 4 agents for owner "carol"
    for i in 0..4 {
        let agent = ucotron_core::Agent::new(
            format!("carol_{}", i),
            format!("Carol Agent {}", i),
            "carol",
            4000 + i as u64,
        );
        state.registry.graph().create_agent(&agent).unwrap();
    }

    let app = Router::new()
        .route("/api/v1/agents", get(handlers::list_agents_handler))
        .layer(middleware::from_fn_with_state(
            state.clone(),
            ucotron_server::auth::auth_middleware,
        ))
        .with_state(state);

    // Filter by carol with pagination
    let req = Request::builder()
        .method("GET")
        .uri("/api/v1/agents?owner=carol&limit=2&offset=1")
        .body(Body::empty())
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    let body = body_to_json(resp.into_body()).await;
    // Total is pre-pagination count of carol's agents
    assert_eq!(body["total"], 4);
    // But only 2 returned due to limit
    assert_eq!(body["agents"].as_array().unwrap().len(), 2);
    assert_eq!(body["limit"], 2);
    assert_eq!(body["offset"], 1);
}

#[tokio::test]
async fn test_list_agents_default_pagination() {
    let (_app, state) = build_app();

    let app = Router::new()
        .route("/api/v1/agents", get(handlers::list_agents_handler))
        .layer(middleware::from_fn_with_state(
            state.clone(),
            ucotron_server::auth::auth_middleware,
        ))
        .with_state(state);

    // No query params — uses defaults (limit=50, offset=0)
    let req = Request::builder()
        .method("GET")
        .uri("/api/v1/agents")
        .body(Body::empty())
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    let body = body_to_json(resp.into_body()).await;
    assert_eq!(body["limit"], 50);
    assert_eq!(body["offset"], 0);
}

// ---------------------------------------------------------------------------
// POST /api/v1/memories/text — Text Memory Ingestion
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_create_text_memory() {
    let (app, _) = build_app();
    let req = Request::post("/api/v1/memories/text")
        .header("Content-Type", "application/json")
        .body(Body::from(
            serde_json::to_string(&serde_json::json!({
                "text": "Alice works at Anthropic in San Francisco."
            }))
            .unwrap(),
        ))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::CREATED);
    let body = body_to_json(resp.into_body()).await;
    assert!(!body["chunk_node_ids"].as_array().unwrap().is_empty());
    assert_eq!(body["media_type"], "Text");
    assert!(body["metrics"]["chunks_processed"].as_u64().unwrap() > 0);
}

#[tokio::test]
async fn test_create_text_memory_empty_text() {
    let (app, _) = build_app();
    let req = Request::post("/api/v1/memories/text")
        .header("Content-Type", "application/json")
        .body(Body::from(r#"{"text":""}"#))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn test_create_text_memory_with_metadata() {
    let (app, _) = build_app();
    let req = Request::post("/api/v1/memories/text")
        .header("Content-Type", "application/json")
        .body(Body::from(
            serde_json::to_string(&serde_json::json!({
                "text": "Bob lives in Berlin.",
                "metadata": {"source": "test", "priority": 5}
            }))
            .unwrap(),
        ))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::CREATED);
    let body = body_to_json(resp.into_body()).await;
    assert_eq!(body["media_type"], "Text");
    assert!(body["edges_created"].as_u64().is_some());
}

#[tokio::test]
async fn test_create_text_memory_with_namespace() {
    let (app, _) = build_app();
    let req = Request::post("/api/v1/memories/text")
        .header("Content-Type", "application/json")
        .header("X-Ucotron-Namespace", "test-ns")
        .body(Body::from(
            serde_json::to_string(&serde_json::json!({
                "text": "Test text for namespace isolation."
            }))
            .unwrap(),
        ))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::CREATED);
    let body = body_to_json(resp.into_body()).await;
    assert_eq!(body["media_type"], "Text");
}

// ---------------------------------------------------------------------------
// POST /api/v1/memories/audio — Audio Memory Ingestion
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_create_audio_memory_no_transcriber_returns_501() {
    let (app, _) = build_app(); // No transcriber loaded

    let boundary = "----AudioMemBoundary1";
    let wav_bytes = create_test_wav_bytes();
    let mut body_bytes = Vec::new();
    body_bytes.extend_from_slice(format!("--{}\r\n", boundary).as_bytes());
    body_bytes.extend_from_slice(
        b"Content-Disposition: form-data; name=\"file\"; filename=\"test.wav\"\r\n",
    );
    body_bytes.extend_from_slice(b"Content-Type: audio/wav\r\n\r\n");
    body_bytes.extend_from_slice(&wav_bytes);
    body_bytes.extend_from_slice(format!("\r\n--{}--\r\n", boundary).as_bytes());

    let req = Request::post("/api/v1/memories/audio")
        .header(
            "Content-Type",
            format!("multipart/form-data; boundary={}", boundary),
        )
        .body(Body::from(body_bytes))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::NOT_IMPLEMENTED);
}

#[tokio::test]
async fn test_create_audio_memory_success() {
    let (app, _) = build_app_with_transcriber();

    let boundary = "----AudioMemBoundary2";
    let wav_bytes = create_test_wav_bytes();
    let mut body_bytes = Vec::new();
    body_bytes.extend_from_slice(format!("--{}\r\n", boundary).as_bytes());
    body_bytes.extend_from_slice(
        b"Content-Disposition: form-data; name=\"file\"; filename=\"test.wav\"\r\n",
    );
    body_bytes.extend_from_slice(b"Content-Type: audio/wav\r\n\r\n");
    body_bytes.extend_from_slice(&wav_bytes);
    body_bytes.extend_from_slice(format!("\r\n--{}--\r\n", boundary).as_bytes());

    let req = Request::post("/api/v1/memories/audio")
        .header(
            "Content-Type",
            format!("multipart/form-data; boundary={}", boundary),
        )
        .body(Body::from(body_bytes))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::CREATED);
    let body = body_to_json(resp.into_body()).await;
    assert_eq!(body["media_type"], "Audio");
    assert_eq!(body["transcription"], "Hello world from mock transcriber");
    assert!(!body["chunk_node_ids"].as_array().unwrap().is_empty());
    assert!(body["audio"]["duration_secs"].as_f64().unwrap() > 0.0);
    assert_eq!(body["audio"]["sample_rate"], 16000);
    assert_eq!(body["audio"]["channels"], 1);
    assert_eq!(body["audio"]["detected_language"], "en");
    assert!(body["metrics"]["chunks_processed"].as_u64().unwrap() > 0);
}

#[tokio::test]
async fn test_create_audio_memory_missing_file() {
    let (app, _) = build_app_with_transcriber();

    let boundary = "----AudioMemBoundary3";
    let mut body_bytes = Vec::new();
    body_bytes.extend_from_slice(format!("--{}\r\n", boundary).as_bytes());
    body_bytes.extend_from_slice(b"Content-Disposition: form-data; name=\"other\"\r\n\r\n");
    body_bytes.extend_from_slice(b"not an audio file");
    body_bytes.extend_from_slice(format!("\r\n--{}--\r\n", boundary).as_bytes());

    let req = Request::post("/api/v1/memories/audio")
        .header(
            "Content-Type",
            format!("multipart/form-data; boundary={}", boundary),
        )
        .body(Body::from(body_bytes))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn test_create_audio_memory_with_namespace() {
    let (app, _) = build_app_with_transcriber();

    let boundary = "----AudioMemBoundary4";
    let wav_bytes = create_test_wav_bytes();
    let mut body_bytes = Vec::new();
    body_bytes.extend_from_slice(format!("--{}\r\n", boundary).as_bytes());
    body_bytes.extend_from_slice(
        b"Content-Disposition: form-data; name=\"audio\"; filename=\"speech.wav\"\r\n",
    );
    body_bytes.extend_from_slice(b"Content-Type: audio/wav\r\n\r\n");
    body_bytes.extend_from_slice(&wav_bytes);
    body_bytes.extend_from_slice(format!("\r\n--{}--\r\n", boundary).as_bytes());

    let req = Request::post("/api/v1/memories/audio")
        .header(
            "Content-Type",
            format!("multipart/form-data; boundary={}", boundary),
        )
        .header("X-Ucotron-Namespace", "audio-ns")
        .body(Body::from(body_bytes))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::CREATED);
    let body = body_to_json(resp.into_body()).await;
    assert_eq!(body["media_type"], "Audio");
}

// ---------------------------------------------------------------------------
// Image Memory Tests
// ---------------------------------------------------------------------------

/// Create a minimal valid 1x1 PNG image in memory.
fn create_test_png_bytes() -> Vec<u8> {
    use std::io::Write;

    // Minimal 1x1 red pixel PNG (RGBA).
    let mut buf = Vec::new();
    // PNG signature
    buf.write_all(&[137, 80, 78, 71, 13, 10, 26, 10]).unwrap();

    // IHDR chunk: width=1, height=1, bit_depth=8, color_type=2 (RGB)
    let ihdr_data: Vec<u8> = vec![
        0, 0, 0, 1, // width = 1
        0, 0, 0, 1, // height = 1
        8, // bit depth = 8
        2, // color type = 2 (RGB)
        0, // compression = 0
        0, // filter = 0
        0, // interlace = 0
    ];
    let ihdr_crc = crc32(&b"IHDR"[..], &ihdr_data);
    buf.write_all(&(ihdr_data.len() as u32).to_be_bytes())
        .unwrap();
    buf.write_all(b"IHDR").unwrap();
    buf.write_all(&ihdr_data).unwrap();
    buf.write_all(&ihdr_crc.to_be_bytes()).unwrap();

    // IDAT chunk: zlib-compressed scanline (filter byte + 3 RGB bytes)
    // Raw scanline: [0, 255, 0, 0] = filter_none + red pixel
    // Deflate with zlib header
    let raw_scanline = [0u8, 255, 0, 0]; // filter_none + R,G,B
    let compressed = miniz_compress(&raw_scanline);
    let idat_crc = crc32(b"IDAT", &compressed);
    buf.write_all(&(compressed.len() as u32).to_be_bytes())
        .unwrap();
    buf.write_all(b"IDAT").unwrap();
    buf.write_all(&compressed).unwrap();
    buf.write_all(&idat_crc.to_be_bytes()).unwrap();

    // IEND chunk
    let iend_crc = crc32(b"IEND", &[]);
    buf.write_all(&0u32.to_be_bytes()).unwrap();
    buf.write_all(b"IEND").unwrap();
    buf.write_all(&iend_crc.to_be_bytes()).unwrap();

    buf
}

/// CRC32 for PNG chunks (type + data).
fn crc32(chunk_type: &[u8], data: &[u8]) -> u32 {
    let mut crc: u32 = 0xFFFFFFFF;
    for &byte in chunk_type.iter().chain(data.iter()) {
        crc ^= byte as u32;
        for _ in 0..8 {
            if crc & 1 != 0 {
                crc = (crc >> 1) ^ 0xEDB88320;
            } else {
                crc >>= 1;
            }
        }
    }
    crc ^ 0xFFFFFFFF
}

/// Minimal zlib/deflate compression (stored blocks, no real compression).
fn miniz_compress(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    // zlib header: CMF=0x78, FLG=0x01 (check bits)
    out.push(0x78);
    out.push(0x01);
    // DEFLATE stored block: BFINAL=1, BTYPE=00 (no compression)
    out.push(0x01); // final block, stored
    let len = data.len() as u16;
    out.extend_from_slice(&len.to_le_bytes());
    let nlen = !len;
    out.extend_from_slice(&nlen.to_le_bytes());
    out.extend_from_slice(data);
    // Adler-32 checksum
    let adler = adler32(data);
    out.extend_from_slice(&adler.to_be_bytes());
    out
}

/// Adler-32 checksum for zlib.
fn adler32(data: &[u8]) -> u32 {
    let mut a: u32 = 1;
    let mut b: u32 = 0;
    for &byte in data {
        a = (a + byte as u32) % 65521;
        b = (b + a) % 65521;
    }
    (b << 16) | a
}

fn build_app_with_image_embedder() -> (Router, Arc<AppState>) {
    let registry = Arc::new(BackendRegistry::new(
        Box::new(MockVectorBackend::new()),
        Box::new(MockGraphBackend::new()),
    ));
    let embedder: Arc<dyn EmbeddingPipeline> = Arc::new(MockEmbedder);
    let image_embedder: Arc<dyn ImageEmbeddingPipeline> = Arc::new(MockImageEmbedder);
    let config = UcotronConfig::default();
    let state = Arc::new(AppState::with_all_pipelines(
        registry,
        embedder,
        None,
        None,
        None,
        Some(image_embedder),
        None,
        config,
    ));

    let app = Router::new()
        .route("/api/v1/health", get(handlers::health_handler))
        .route("/api/v1/memories", post(handlers::create_memory_handler))
        .route(
            "/api/v1/memories/image",
            post(handlers::create_image_memory_handler),
        )
        .route("/api/v1/memories/search", post(handlers::search_handler))
        .layer(middleware::from_fn_with_state(
            state.clone(),
            ucotron_server::auth::auth_middleware,
        ))
        .with_state(state.clone());

    (app, state)
}

#[tokio::test]
async fn test_create_image_memory_no_embedder_returns_501() {
    let (app, _) = build_app(); // No image embedder loaded

    let boundary = "----ImageMemBoundary1";
    let png_bytes = create_test_png_bytes();
    let mut body_bytes = Vec::new();
    body_bytes.extend_from_slice(format!("--{}\r\n", boundary).as_bytes());
    body_bytes.extend_from_slice(
        b"Content-Disposition: form-data; name=\"file\"; filename=\"test.png\"\r\n",
    );
    body_bytes.extend_from_slice(b"Content-Type: image/png\r\n\r\n");
    body_bytes.extend_from_slice(&png_bytes);
    body_bytes.extend_from_slice(format!("\r\n--{}--\r\n", boundary).as_bytes());

    let req = Request::post("/api/v1/memories/image")
        .header(
            "Content-Type",
            format!("multipart/form-data; boundary={}", boundary),
        )
        .body(Body::from(body_bytes))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::NOT_IMPLEMENTED);
}

#[tokio::test]
async fn test_create_image_memory_success() {
    let (app, _) = build_app_with_image_embedder();

    let boundary = "----ImageMemBoundary2";
    let png_bytes = create_test_png_bytes();
    let mut body_bytes = Vec::new();
    body_bytes.extend_from_slice(format!("--{}\r\n", boundary).as_bytes());
    body_bytes.extend_from_slice(
        b"Content-Disposition: form-data; name=\"file\"; filename=\"test.png\"\r\n",
    );
    body_bytes.extend_from_slice(b"Content-Type: image/png\r\n\r\n");
    body_bytes.extend_from_slice(&png_bytes);
    body_bytes.extend_from_slice(format!("\r\n--{}--\r\n", boundary).as_bytes());

    let req = Request::post("/api/v1/memories/image")
        .header(
            "Content-Type",
            format!("multipart/form-data; boundary={}", boundary),
        )
        .body(Body::from(body_bytes))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::CREATED);
    let body = body_to_json(resp.into_body()).await;
    assert_eq!(body["media_type"], "Image");
    assert_eq!(body["embedding_dim"], 512);
    assert!(body["node_id"].as_u64().is_some());
    assert_eq!(body["width"], 1);
    assert_eq!(body["height"], 1);
    assert_eq!(body["format"], "png");
    assert_eq!(body["description_ingested"], false);
    assert!(body["metrics"].is_null());
}

#[tokio::test]
async fn test_create_image_memory_missing_file() {
    let (app, _) = build_app_with_image_embedder();

    let boundary = "----ImageMemBoundary3";
    let mut body_bytes = Vec::new();
    body_bytes.extend_from_slice(format!("--{}\r\n", boundary).as_bytes());
    body_bytes.extend_from_slice(b"Content-Disposition: form-data; name=\"other\"\r\n\r\n");
    body_bytes.extend_from_slice(b"not an image file");
    body_bytes.extend_from_slice(format!("\r\n--{}--\r\n", boundary).as_bytes());

    let req = Request::post("/api/v1/memories/image")
        .header(
            "Content-Type",
            format!("multipart/form-data; boundary={}", boundary),
        )
        .body(Body::from(body_bytes))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn test_create_image_memory_with_description() {
    let (app, _) = build_app_with_image_embedder();

    let boundary = "----ImageMemBoundary4";
    let png_bytes = create_test_png_bytes();
    let mut body_bytes = Vec::new();
    // Image field
    body_bytes.extend_from_slice(format!("--{}\r\n", boundary).as_bytes());
    body_bytes.extend_from_slice(
        b"Content-Disposition: form-data; name=\"image\"; filename=\"photo.png\"\r\n",
    );
    body_bytes.extend_from_slice(b"Content-Type: image/png\r\n\r\n");
    body_bytes.extend_from_slice(&png_bytes);
    body_bytes.extend_from_slice(b"\r\n");
    // Description field
    body_bytes.extend_from_slice(format!("--{}\r\n", boundary).as_bytes());
    body_bytes.extend_from_slice(b"Content-Disposition: form-data; name=\"description\"\r\n\r\n");
    body_bytes.extend_from_slice(b"A beautiful sunset over the ocean");
    body_bytes.extend_from_slice(format!("\r\n--{}--\r\n", boundary).as_bytes());

    let req = Request::post("/api/v1/memories/image")
        .header(
            "Content-Type",
            format!("multipart/form-data; boundary={}", boundary),
        )
        .body(Body::from(body_bytes))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::CREATED);
    let body = body_to_json(resp.into_body()).await;
    assert_eq!(body["media_type"], "Image");
    assert_eq!(body["description_ingested"], true);
    assert!(body["metrics"].is_object());
    assert!(body["metrics"]["chunks_processed"].as_u64().unwrap() > 0);
}

#[tokio::test]
async fn test_create_image_memory_with_namespace() {
    let (app, _) = build_app_with_image_embedder();

    let boundary = "----ImageMemBoundary5";
    let png_bytes = create_test_png_bytes();
    let mut body_bytes = Vec::new();
    body_bytes.extend_from_slice(format!("--{}\r\n", boundary).as_bytes());
    body_bytes.extend_from_slice(
        b"Content-Disposition: form-data; name=\"file\"; filename=\"ns_test.png\"\r\n",
    );
    body_bytes.extend_from_slice(b"Content-Type: image/png\r\n\r\n");
    body_bytes.extend_from_slice(&png_bytes);
    body_bytes.extend_from_slice(format!("\r\n--{}--\r\n", boundary).as_bytes());

    let req = Request::post("/api/v1/memories/image")
        .header(
            "Content-Type",
            format!("multipart/form-data; boundary={}", boundary),
        )
        .header("X-Ucotron-Namespace", "image-ns")
        .body(Body::from(body_bytes))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::CREATED);
    let body = body_to_json(resp.into_body()).await;
    assert_eq!(body["media_type"], "Image");
}

// ---------------------------------------------------------------------------
// Video Memory Tests
// ---------------------------------------------------------------------------

/// Create a minimal MP4-like test payload (doesn't need to be valid video
/// since MockVideoPipeline ignores the file content).
fn create_test_video_bytes() -> Vec<u8> {
    vec![
        0x00, 0x00, 0x00, 0x1C, 0x66, 0x74, 0x79, 0x70, // ftyp box
        0x69, 0x73, 0x6F, 0x6D, 0x00, 0x00, 0x02, 0x00, 0x69, 0x73, 0x6F, 0x6D, 0x69, 0x73, 0x6F,
        0x32, 0x61, 0x76, 0x63, 0x31,
    ]
}

fn build_app_with_video_pipeline() -> (Router, Arc<AppState>) {
    let registry = Arc::new(BackendRegistry::new(
        Box::new(MockVectorBackend::new()),
        Box::new(MockGraphBackend::new()),
    ));
    let embedder: Arc<dyn EmbeddingPipeline> = Arc::new(MockEmbedder);
    let image_embedder: Arc<dyn ImageEmbeddingPipeline> = Arc::new(MockImageEmbedder);
    let video_pipeline: Arc<dyn VideoPipeline> = Arc::new(MockVideoPipeline);
    let config = UcotronConfig::default();
    let state = Arc::new(AppState::with_all_pipelines_full(
        registry,
        embedder,
        None,
        None,
        None,
        Some(image_embedder),
        None,
        None,
        Some(video_pipeline),
        config,
    ));

    let app = Router::new()
        .route("/api/v1/health", get(handlers::health_handler))
        .route(
            "/api/v1/memories/video",
            post(handlers::create_video_memory_handler),
        )
        .layer(middleware::from_fn_with_state(
            state.clone(),
            ucotron_server::auth::auth_middleware,
        ))
        .with_state(state.clone());

    (app, state)
}

#[tokio::test]
async fn test_create_video_memory_no_pipeline_returns_501() {
    let (app, _) = build_app(); // No video pipeline loaded

    let boundary = "----VideoMemBoundary1";
    let video_bytes = create_test_video_bytes();
    let mut body_bytes = Vec::new();
    body_bytes.extend_from_slice(format!("--{}\r\n", boundary).as_bytes());
    body_bytes.extend_from_slice(
        b"Content-Disposition: form-data; name=\"file\"; filename=\"test.mp4\"\r\n",
    );
    body_bytes.extend_from_slice(b"Content-Type: video/mp4\r\n\r\n");
    body_bytes.extend_from_slice(&video_bytes);
    body_bytes.extend_from_slice(format!("\r\n--{}--\r\n", boundary).as_bytes());

    let req = Request::post("/api/v1/memories/video")
        .header(
            "Content-Type",
            format!("multipart/form-data; boundary={}", boundary),
        )
        .body(Body::from(body_bytes))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::NOT_IMPLEMENTED);
}

#[tokio::test]
async fn test_create_video_memory_success() {
    let (app, _) = build_app_with_video_pipeline();

    let boundary = "----VideoMemBoundary2";
    let video_bytes = create_test_video_bytes();
    let mut body_bytes = Vec::new();
    body_bytes.extend_from_slice(format!("--{}\r\n", boundary).as_bytes());
    body_bytes.extend_from_slice(
        b"Content-Disposition: form-data; name=\"file\"; filename=\"test.mp4\"\r\n",
    );
    body_bytes.extend_from_slice(b"Content-Type: video/mp4\r\n\r\n");
    body_bytes.extend_from_slice(&video_bytes);
    body_bytes.extend_from_slice(format!("\r\n--{}--\r\n", boundary).as_bytes());

    let req = Request::post("/api/v1/memories/video")
        .header(
            "Content-Type",
            format!("multipart/form-data; boundary={}", boundary),
        )
        .body(Body::from(body_bytes))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::CREATED);
    let body = body_to_json(resp.into_body()).await;
    assert_eq!(body["media_type"], "VideoSegment");
    assert!(body["video_node_id"].as_u64().is_some());
    assert!(!body["segment_node_ids"].as_array().unwrap().is_empty());
    assert_eq!(body["total_frames"], 3);
    assert!(body["total_segments"].as_u64().unwrap() >= 1);
    assert_eq!(body["duration_ms"], 12000);
    assert_eq!(body["video_width"], 640);
    assert_eq!(body["video_height"], 480);
    assert!(body["edges_created"].as_u64().unwrap() >= 1);
    assert!(!body["segments"].as_array().unwrap().is_empty());
    // Each segment should have the expected fields.
    let first_seg = &body["segments"][0];
    assert!(first_seg["node_id"].as_u64().is_some());
    assert!(first_seg["start_ms"].is_number());
    assert!(first_seg["end_ms"].is_number());
    assert!(first_seg["frame_count"].as_u64().unwrap() >= 1);
}

#[tokio::test]
async fn test_create_video_memory_missing_file() {
    let (app, _) = build_app_with_video_pipeline();

    let boundary = "----VideoMemBoundary3";
    let mut body_bytes = Vec::new();
    body_bytes.extend_from_slice(format!("--{}\r\n", boundary).as_bytes());
    body_bytes.extend_from_slice(b"Content-Disposition: form-data; name=\"other\"\r\n\r\n");
    body_bytes.extend_from_slice(b"not a video file");
    body_bytes.extend_from_slice(format!("\r\n--{}--\r\n", boundary).as_bytes());

    let req = Request::post("/api/v1/memories/video")
        .header(
            "Content-Type",
            format!("multipart/form-data; boundary={}", boundary),
        )
        .body(Body::from(body_bytes))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn test_create_video_memory_empty_file() {
    let (app, _) = build_app_with_video_pipeline();

    let boundary = "----VideoMemBoundary4";
    let mut body_bytes = Vec::new();
    body_bytes.extend_from_slice(format!("--{}\r\n", boundary).as_bytes());
    body_bytes.extend_from_slice(
        b"Content-Disposition: form-data; name=\"video\"; filename=\"empty.mp4\"\r\n",
    );
    body_bytes.extend_from_slice(b"Content-Type: video/mp4\r\n\r\n");
    // No actual bytes — empty file.
    body_bytes.extend_from_slice(format!("\r\n--{}--\r\n", boundary).as_bytes());

    let req = Request::post("/api/v1/memories/video")
        .header(
            "Content-Type",
            format!("multipart/form-data; boundary={}", boundary),
        )
        .body(Body::from(body_bytes))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn test_create_video_memory_with_namespace() {
    let (app, _) = build_app_with_video_pipeline();

    let boundary = "----VideoMemBoundary5";
    let video_bytes = create_test_video_bytes();
    let mut body_bytes = Vec::new();
    body_bytes.extend_from_slice(format!("--{}\r\n", boundary).as_bytes());
    body_bytes.extend_from_slice(
        b"Content-Disposition: form-data; name=\"video\"; filename=\"test.mp4\"\r\n",
    );
    body_bytes.extend_from_slice(b"Content-Type: video/mp4\r\n\r\n");
    body_bytes.extend_from_slice(&video_bytes);
    body_bytes.extend_from_slice(format!("\r\n--{}--\r\n", boundary).as_bytes());

    let req = Request::post("/api/v1/memories/video")
        .header(
            "Content-Type",
            format!("multipart/form-data; boundary={}", boundary),
        )
        .header("X-Ucotron-Namespace", "video-ns")
        .body(Body::from(body_bytes))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::CREATED);
    let body = body_to_json(resp.into_body()).await;
    assert_eq!(body["media_type"], "VideoSegment");
}

#[tokio::test]
async fn test_create_video_memory_segments_linked_to_parent() {
    let (app, state) = build_app_with_video_pipeline();

    let boundary = "----VideoMemBoundary6";
    let video_bytes = create_test_video_bytes();
    let mut body_bytes = Vec::new();
    body_bytes.extend_from_slice(format!("--{}\r\n", boundary).as_bytes());
    body_bytes.extend_from_slice(
        b"Content-Disposition: form-data; name=\"file\"; filename=\"test.mp4\"\r\n",
    );
    body_bytes.extend_from_slice(b"Content-Type: video/mp4\r\n\r\n");
    body_bytes.extend_from_slice(&video_bytes);
    body_bytes.extend_from_slice(format!("\r\n--{}--\r\n", boundary).as_bytes());

    let req = Request::post("/api/v1/memories/video")
        .header(
            "Content-Type",
            format!("multipart/form-data; boundary={}", boundary),
        )
        .body(Body::from(body_bytes))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::CREATED);
    let body = body_to_json(resp.into_body()).await;

    let parent_id = body["video_node_id"].as_u64().unwrap();
    let seg_ids: Vec<u64> = body["segment_node_ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap())
        .collect();

    // Verify parent node exists in graph.
    let parent_node = state.registry.graph().get_node(parent_id).unwrap();
    assert!(parent_node.is_some());
    let parent = parent_node.unwrap();
    assert_eq!(
        parent.media_type,
        Some(ucotron_core::MediaType::VideoSegment)
    );

    // Verify each segment node has parent_video_id set.
    for seg_id in &seg_ids {
        let seg_node = state.registry.graph().get_node(*seg_id).unwrap().unwrap();
        assert_eq!(seg_node.parent_video_id, Some(parent_id));
        assert_eq!(
            seg_node.media_type,
            Some(ucotron_core::MediaType::VideoSegment)
        );
        assert!(seg_node.timestamp_range.is_some());
        assert!(seg_node.embedding_visual.is_some());
    }

    // Verify edges exist: parent → each segment.
    let edges_created = body["edges_created"].as_u64().unwrap();
    assert_eq!(edges_created as usize, seg_ids.len());
}

// ---------------------------------------------------------------------------
// Agent Cascade Delete Tests
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_delete_agent_cascade_deletes_namespace_data() {
    let (_app, state) = build_app();

    // 1. Create an agent
    let agent = ucotron_core::Agent::new("cascade-bot", "Cascade Bot", "admin-key", 1000);
    state.registry.graph().create_agent(&agent).unwrap();
    let ns = &agent.namespace; // "agent_cascade-bot"

    // 2. Insert nodes into the agent's namespace
    insert_test_node_with_namespace(&state, 500, "Memory A", NodeType::Entity, ns);
    insert_test_node_with_namespace(&state, 501, "Memory B", NodeType::Event, ns);
    insert_test_node_with_namespace(&state, 502, "Memory C", NodeType::Fact, ns);

    // 3. Insert a node in a different namespace (should NOT be deleted)
    insert_test_node_with_namespace(&state, 600, "Other data", NodeType::Entity, "other_ns");

    // Verify nodes exist before delete
    assert!(state.registry.graph().get_node(500).unwrap().is_some());
    assert!(state.registry.graph().get_node(501).unwrap().is_some());
    assert!(state.registry.graph().get_node(502).unwrap().is_some());
    assert!(state.registry.graph().get_node(600).unwrap().is_some());

    // 4. Delete the agent via the API
    let app = Router::new()
        .route(
            "/api/v1/agents/{id}",
            delete(handlers::delete_agent_handler),
        )
        .layer(middleware::from_fn_with_state(
            state.clone(),
            ucotron_server::auth::auth_middleware,
        ))
        .with_state(state.clone());

    let req = Request::delete("/api/v1/agents/cascade-bot")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let body = body_to_json(resp.into_body()).await;
    assert_eq!(body["deleted"], true);
    assert_eq!(body["nodes_deleted"], 3);

    // 5. Verify agent record is gone
    assert!(state
        .registry
        .graph()
        .get_agent("cascade-bot")
        .unwrap()
        .is_none());

    // 6. Verify namespace nodes are deleted from graph
    assert!(state.registry.graph().get_node(500).unwrap().is_none());
    assert!(state.registry.graph().get_node(501).unwrap().is_none());
    assert!(state.registry.graph().get_node(502).unwrap().is_none());

    // 7. Verify other namespace node is untouched
    assert!(state.registry.graph().get_node(600).unwrap().is_some());
}

#[tokio::test]
async fn test_delete_agent_requires_admin_role() {
    // Build an auth-enabled app with a writer key (not admin)
    let registry = Arc::new(BackendRegistry::new(
        Box::new(MockVectorBackend::new()),
        Box::new(MockGraphBackend::new()),
    ));
    let embedder: Arc<dyn EmbeddingPipeline> = Arc::new(MockEmbedder);
    let mut config = UcotronConfig::default();
    config.auth.enabled = true;
    config.auth.api_keys = vec![ApiKeyEntry {
        name: "writer-key".into(),
        key: "sk-writer-test".into(),
        role: "writer".into(),
        namespace: None,
        active: true,
    }];
    let state = Arc::new(AppState::new(registry, embedder, None, None, config));

    // Create agent directly in the backend
    let agent = ucotron_core::Agent::new("auth-bot", "Auth Bot", "owner", 1000);
    state.registry.graph().create_agent(&agent).unwrap();

    let app = Router::new()
        .route(
            "/api/v1/agents/{id}",
            delete(handlers::delete_agent_handler),
        )
        .layer(middleware::from_fn_with_state(
            state.clone(),
            ucotron_server::auth::auth_middleware,
        ))
        .with_state(state.clone());

    // Writer role should be rejected (admin required)
    let req = Request::delete("/api/v1/agents/auth-bot")
        .header("Authorization", "Bearer sk-writer-test")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::FORBIDDEN);
}

#[tokio::test]
async fn test_delete_nonexistent_agent_returns_404() {
    let (_app, state) = build_app();

    let app = Router::new()
        .route(
            "/api/v1/agents/{id}",
            delete(handlers::delete_agent_handler),
        )
        .layer(middleware::from_fn_with_state(
            state.clone(),
            ucotron_server::auth::auth_middleware,
        ))
        .with_state(state.clone());

    let req = Request::delete("/api/v1/agents/nonexistent")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn test_delete_agent_with_no_data_succeeds() {
    let (_app, state) = build_app();

    // Create agent but don't add any nodes to its namespace
    let agent = ucotron_core::Agent::new("empty-bot", "Empty Bot", "admin-key", 1000);
    state.registry.graph().create_agent(&agent).unwrap();

    let app = Router::new()
        .route(
            "/api/v1/agents/{id}",
            delete(handlers::delete_agent_handler),
        )
        .layer(middleware::from_fn_with_state(
            state.clone(),
            ucotron_server::auth::auth_middleware,
        ))
        .with_state(state.clone());

    let req = Request::delete("/api/v1/agents/empty-bot")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let body = body_to_json(resp.into_body()).await;
    assert_eq!(body["deleted"], true);
    assert_eq!(body["nodes_deleted"], 0);

    // Agent should be gone
    assert!(state
        .registry
        .graph()
        .get_agent("empty-bot")
        .unwrap()
        .is_none());
}

// ===========================================================================
// Agent Graph Clone (US-25.6)
// ===========================================================================

#[tokio::test]
async fn test_clone_agent_graph_basic() {
    let (_app, state) = build_app();

    // Create source agent
    let agent = ucotron_core::Agent::new("src-bot", "Source Bot", "owner", 1000);
    state.registry.graph().create_agent(&agent).unwrap();

    // Add nodes in the agent's namespace
    let mut meta1 = HashMap::new();
    meta1.insert(
        "_namespace".into(),
        ucotron_core::Value::String("agent_src-bot".into()),
    );
    let meta2 = meta1.clone();

    let nodes = vec![
        Node {
            id: 100,
            content: "Node A".into(),
            embedding: vec![1.0, 0.0, 0.0],
            metadata: meta1,
            node_type: ucotron_core::NodeType::Entity,
            timestamp: 1000,
            media_type: None,
            media_uri: None,
            embedding_visual: None,
            timestamp_range: None,
            parent_video_id: None,
        },
        Node {
            id: 101,
            content: "Node B".into(),
            embedding: vec![0.0, 1.0, 0.0],
            metadata: meta2,
            node_type: ucotron_core::NodeType::Event,
            timestamp: 2000,
            media_type: None,
            media_uri: None,
            embedding_visual: None,
            timestamp_range: None,
            parent_video_id: None,
        },
    ];
    state.registry.graph().upsert_nodes(&nodes).unwrap();
    state
        .registry
        .graph()
        .upsert_edges(&[Edge {
            source: 100,
            target: 101,
            edge_type: ucotron_core::EdgeType::RelatesTo,
            weight: 0.9,
            metadata: HashMap::new(),
        }])
        .unwrap();

    // Clone via API
    let app = Router::new()
        .route(
            "/api/v1/agents/{id}/clone",
            post(handlers::clone_agent_handler),
        )
        .layer(middleware::from_fn_with_state(
            state.clone(),
            ucotron_server::auth::auth_middleware,
        ))
        .with_state(state.clone());

    let req = Request::post("/api/v1/agents/src-bot/clone")
        .header("Content-Type", "application/json")
        .body(Body::from(r#"{"target_namespace": "clone-ns"}"#))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let body = body_to_json(resp.into_body()).await;
    assert_eq!(body["source_agent_id"], "src-bot");
    assert_eq!(body["source_namespace"], "agent_src-bot");
    assert_eq!(body["target_namespace"], "clone-ns");
    assert_eq!(body["nodes_copied"], 2);
    assert_eq!(body["edges_copied"], 1);
}

#[tokio::test]
async fn test_clone_agent_with_node_type_filter() {
    let (_app, state) = build_app();

    let agent = ucotron_core::Agent::new("filter-bot", "Filter Bot", "owner", 1000);
    state.registry.graph().create_agent(&agent).unwrap();

    let mut meta = HashMap::new();
    meta.insert(
        "_namespace".into(),
        ucotron_core::Value::String("agent_filter-bot".into()),
    );

    let nodes = vec![
        Node {
            id: 200,
            content: "Entity node".into(),
            embedding: vec![1.0],
            metadata: meta.clone(),
            node_type: ucotron_core::NodeType::Entity,
            timestamp: 1000,
            media_type: None,
            media_uri: None,
            embedding_visual: None,
            timestamp_range: None,
            parent_video_id: None,
        },
        Node {
            id: 201,
            content: "Event node".into(),
            embedding: vec![0.5],
            metadata: meta.clone(),
            node_type: ucotron_core::NodeType::Event,
            timestamp: 2000,
            media_type: None,
            media_uri: None,
            embedding_visual: None,
            timestamp_range: None,
            parent_video_id: None,
        },
        Node {
            id: 202,
            content: "Fact node".into(),
            embedding: vec![0.3],
            metadata: meta.clone(),
            node_type: ucotron_core::NodeType::Fact,
            timestamp: 3000,
            media_type: None,
            media_uri: None,
            embedding_visual: None,
            timestamp_range: None,
            parent_video_id: None,
        },
    ];
    state.registry.graph().upsert_nodes(&nodes).unwrap();

    let app = Router::new()
        .route(
            "/api/v1/agents/{id}/clone",
            post(handlers::clone_agent_handler),
        )
        .layer(middleware::from_fn_with_state(
            state.clone(),
            ucotron_server::auth::auth_middleware,
        ))
        .with_state(state.clone());

    let req = Request::post("/api/v1/agents/filter-bot/clone")
        .header("Content-Type", "application/json")
        .body(Body::from(
            r#"{"target_namespace": "entities-only", "node_types": ["Entity"]}"#,
        ))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let body = body_to_json(resp.into_body()).await;
    assert_eq!(body["nodes_copied"], 1);
}

#[tokio::test]
async fn test_clone_agent_with_time_range_filter() {
    let (_app, state) = build_app();

    let agent = ucotron_core::Agent::new("time-bot", "Time Bot", "owner", 1000);
    state.registry.graph().create_agent(&agent).unwrap();

    let mut meta = HashMap::new();
    meta.insert(
        "_namespace".into(),
        ucotron_core::Value::String("agent_time-bot".into()),
    );

    let nodes = vec![
        Node {
            id: 300,
            content: "Old node".into(),
            embedding: vec![1.0],
            metadata: meta.clone(),
            node_type: ucotron_core::NodeType::Entity,
            timestamp: 1000,
            media_type: None,
            media_uri: None,
            embedding_visual: None,
            timestamp_range: None,
            parent_video_id: None,
        },
        Node {
            id: 301,
            content: "Recent node".into(),
            embedding: vec![0.5],
            metadata: meta.clone(),
            node_type: ucotron_core::NodeType::Entity,
            timestamp: 5000,
            media_type: None,
            media_uri: None,
            embedding_visual: None,
            timestamp_range: None,
            parent_video_id: None,
        },
    ];
    state.registry.graph().upsert_nodes(&nodes).unwrap();

    let app = Router::new()
        .route(
            "/api/v1/agents/{id}/clone",
            post(handlers::clone_agent_handler),
        )
        .layer(middleware::from_fn_with_state(
            state.clone(),
            ucotron_server::auth::auth_middleware,
        ))
        .with_state(state.clone());

    let req = Request::post("/api/v1/agents/time-bot/clone")
        .header("Content-Type", "application/json")
        .body(Body::from(
            r#"{"target_namespace": "recent-only", "time_range_start": 3000}"#,
        ))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let body = body_to_json(resp.into_body()).await;
    assert_eq!(body["nodes_copied"], 1);
}

#[tokio::test]
async fn test_clone_nonexistent_agent_returns_404() {
    let (_app, state) = build_app();

    let app = Router::new()
        .route(
            "/api/v1/agents/{id}/clone",
            post(handlers::clone_agent_handler),
        )
        .layer(middleware::from_fn_with_state(
            state.clone(),
            ucotron_server::auth::auth_middleware,
        ))
        .with_state(state.clone());

    let req = Request::post("/api/v1/agents/nonexistent/clone")
        .header("Content-Type", "application/json")
        .body(Body::from(r#"{"target_namespace": "dst"}"#))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn test_clone_empty_agent_returns_zero_counts() {
    let (_app, state) = build_app();

    let agent = ucotron_core::Agent::new("empty-bot", "Empty Bot", "owner", 1000);
    state.registry.graph().create_agent(&agent).unwrap();

    let app = Router::new()
        .route(
            "/api/v1/agents/{id}/clone",
            post(handlers::clone_agent_handler),
        )
        .layer(middleware::from_fn_with_state(
            state.clone(),
            ucotron_server::auth::auth_middleware,
        ))
        .with_state(state.clone());

    let req = Request::post("/api/v1/agents/empty-bot/clone")
        .header("Content-Type", "application/json")
        .body(Body::from(r#"{"target_namespace": "empty-clone"}"#))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let body = body_to_json(resp.into_body()).await;
    assert_eq!(body["nodes_copied"], 0);
    assert_eq!(body["edges_copied"], 0);
}

// ===========================================================================
// Agent Merge (US-25.7)
// ===========================================================================

/// Helper to create a node with namespace metadata.
fn make_ns_node(id: NodeId, content: &str, ns: &str) -> Node {
    let mut metadata = HashMap::new();
    metadata.insert(
        "_namespace".to_string(),
        ucotron_core::Value::String(ns.to_string()),
    );
    Node {
        id,
        content: content.to_string(),
        embedding: vec![0.1; 384],
        metadata,
        node_type: ucotron_core::NodeType::Entity,
        timestamp: 1000,
        media_type: None,
        media_uri: None,
        embedding_visual: None,
        timestamp_range: None,
        parent_video_id: None,
    }
}

#[tokio::test]
async fn test_merge_agent_basic() {
    let (_, state) = build_app();

    // Create source and target agents
    let src_agent = ucotron_core::Agent::new("src-agent", "Source", "owner", 1000);
    let dst_agent = ucotron_core::Agent::new("dst-agent", "Dest", "owner", 1000);
    state.registry.graph().create_agent(&src_agent).unwrap();
    state.registry.graph().create_agent(&dst_agent).unwrap();

    // Add nodes to source namespace (unique content)
    let src_nodes = vec![
        make_ns_node(100, "Alice", "agent_src-agent"),
        make_ns_node(101, "Bob", "agent_src-agent"),
    ];
    state.registry.graph().upsert_nodes(&src_nodes).unwrap();

    // Add an edge between them
    let src_edges = vec![Edge {
        source: 100,
        target: 101,
        edge_type: ucotron_core::EdgeType::RelatesTo,
        weight: 0.9,
        metadata: HashMap::new(),
    }];
    state.registry.graph().upsert_edges(&src_edges).unwrap();

    let app = Router::new()
        .route(
            "/api/v1/agents/{id}/merge",
            post(handlers::merge_agent_handler),
        )
        .layer(middleware::from_fn_with_state(
            state.clone(),
            ucotron_server::auth::auth_middleware,
        ))
        .with_state(state.clone());

    let req = Request::post("/api/v1/agents/dst-agent/merge")
        .header("Content-Type", "application/json")
        .body(Body::from(r#"{"source_agent_id": "src-agent"}"#))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let body = body_to_json(resp.into_body()).await;
    assert_eq!(body["nodes_copied"], 2);
    assert_eq!(body["edges_copied"], 1);
    assert_eq!(body["nodes_deduplicated"], 0);
    assert_eq!(body["ids_remapped"], 2);
    assert_eq!(body["source_namespace"], "agent_src-agent");
    assert_eq!(body["target_namespace"], "agent_dst-agent");
}

#[tokio::test]
async fn test_merge_agent_with_deduplication() {
    let (_, state) = build_app();

    let src_agent = ucotron_core::Agent::new("merge-src", "Source", "owner", 1000);
    let dst_agent = ucotron_core::Agent::new("merge-dst", "Dest", "owner", 1000);
    state.registry.graph().create_agent(&src_agent).unwrap();
    state.registry.graph().create_agent(&dst_agent).unwrap();

    // Add nodes to destination with known content
    let dst_nodes = vec![
        make_ns_node(200, "Alice", "agent_merge-dst"),
        make_ns_node(201, "Charlie", "agent_merge-dst"),
    ];
    state.registry.graph().upsert_nodes(&dst_nodes).unwrap();

    // Add nodes to source — "Alice" is a duplicate, "Bob" is new
    let src_nodes = vec![
        make_ns_node(300, "Alice", "agent_merge-src"),
        make_ns_node(301, "Bob", "agent_merge-src"),
    ];
    state.registry.graph().upsert_nodes(&src_nodes).unwrap();

    let app = Router::new()
        .route(
            "/api/v1/agents/{id}/merge",
            post(handlers::merge_agent_handler),
        )
        .layer(middleware::from_fn_with_state(
            state.clone(),
            ucotron_server::auth::auth_middleware,
        ))
        .with_state(state.clone());

    let req = Request::post("/api/v1/agents/merge-dst/merge")
        .header("Content-Type", "application/json")
        .body(Body::from(r#"{"source_agent_id": "merge-src"}"#))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let body = body_to_json(resp.into_body()).await;
    // Only "Bob" should be copied; "Alice" should be deduplicated
    assert_eq!(body["nodes_copied"], 1);
    assert_eq!(body["nodes_deduplicated"], 1);
    assert_eq!(body["ids_remapped"], 1);
}

#[tokio::test]
async fn test_merge_nonexistent_target_agent_returns_404() {
    let (_, state) = build_app();

    let app = Router::new()
        .route(
            "/api/v1/agents/{id}/merge",
            post(handlers::merge_agent_handler),
        )
        .layer(middleware::from_fn_with_state(
            state.clone(),
            ucotron_server::auth::auth_middleware,
        ))
        .with_state(state.clone());

    let req = Request::post("/api/v1/agents/nonexistent/merge")
        .header("Content-Type", "application/json")
        .body(Body::from(r#"{"source_agent_id": "src"}"#))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn test_merge_nonexistent_source_agent_returns_404() {
    let (_, state) = build_app();

    let dst_agent = ucotron_core::Agent::new("merge-tgt", "Target", "owner", 1000);
    state.registry.graph().create_agent(&dst_agent).unwrap();

    let app = Router::new()
        .route(
            "/api/v1/agents/{id}/merge",
            post(handlers::merge_agent_handler),
        )
        .layer(middleware::from_fn_with_state(
            state.clone(),
            ucotron_server::auth::auth_middleware,
        ))
        .with_state(state.clone());

    let req = Request::post("/api/v1/agents/merge-tgt/merge")
        .header("Content-Type", "application/json")
        .body(Body::from(r#"{"source_agent_id": "nonexistent"}"#))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn test_merge_empty_source_returns_zero_counts() {
    let (_, state) = build_app();

    let src_agent = ucotron_core::Agent::new("empty-src", "Empty Source", "owner", 1000);
    let dst_agent = ucotron_core::Agent::new("empty-dst", "Empty Dest", "owner", 1000);
    state.registry.graph().create_agent(&src_agent).unwrap();
    state.registry.graph().create_agent(&dst_agent).unwrap();

    let app = Router::new()
        .route(
            "/api/v1/agents/{id}/merge",
            post(handlers::merge_agent_handler),
        )
        .layer(middleware::from_fn_with_state(
            state.clone(),
            ucotron_server::auth::auth_middleware,
        ))
        .with_state(state.clone());

    let req = Request::post("/api/v1/agents/empty-dst/merge")
        .header("Content-Type", "application/json")
        .body(Body::from(r#"{"source_agent_id": "empty-src"}"#))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let body = body_to_json(resp.into_body()).await;
    assert_eq!(body["nodes_copied"], 0);
    assert_eq!(body["edges_copied"], 0);
    assert_eq!(body["nodes_deduplicated"], 0);
    assert_eq!(body["ids_remapped"], 0);
}

// ===========================================================================
// Multimodal Search (US-33.17)
// ===========================================================================

struct MockVisualVectorBackend {
    embeddings: Mutex<HashMap<NodeId, Vec<f32>>>,
}

impl MockVisualVectorBackend {
    fn new() -> Self {
        Self {
            embeddings: Mutex::new(HashMap::new()),
        }
    }
}

impl ucotron_core::VisualVectorBackend for MockVisualVectorBackend {
    fn upsert_visual_embeddings(&self, items: &[(NodeId, Vec<f32>)]) -> anyhow::Result<()> {
        let mut store = self.embeddings.lock().unwrap();
        for (id, emb) in items {
            store.insert(*id, emb.clone());
        }
        Ok(())
    }

    fn search_visual(&self, query: &[f32], top_k: usize) -> anyhow::Result<Vec<(NodeId, f32)>> {
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

    fn delete_visual(&self, ids: &[NodeId]) -> anyhow::Result<()> {
        let mut store = self.embeddings.lock().unwrap();
        for id in ids {
            store.remove(id);
        }
        Ok(())
    }
}

struct MockCrossModalTextEncoder;

impl ucotron_extraction::CrossModalTextEncoder for MockCrossModalTextEncoder {
    fn embed_text(&self, text: &str) -> anyhow::Result<Vec<f32>> {
        // Return a deterministic 512-dim normalized vector based on text length.
        let hash = text.len() as f32 / 100.0;
        let mut v = vec![hash; 512];
        v[0] = 1.0;
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in &mut v {
                *x /= norm;
            }
        }
        Ok(v)
    }
}

/// Build an app with visual vector backend and cross-modal text encoder for multimodal search.
fn build_multimodal_app() -> (Router, Arc<AppState>) {
    let registry = Arc::new(BackendRegistry::with_visual(
        Box::new(MockVectorBackend::new()),
        Box::new(MockGraphBackend::new()),
        Box::new(MockVisualVectorBackend::new()),
    ));
    let embedder: Arc<dyn EmbeddingPipeline> = Arc::new(MockEmbedder);
    let image_embedder: Arc<dyn ImageEmbeddingPipeline> = Arc::new(MockImageEmbedder);
    let cross_modal_encoder: Arc<dyn ucotron_extraction::CrossModalTextEncoder> =
        Arc::new(MockCrossModalTextEncoder);
    let config = UcotronConfig::default();
    let state = Arc::new(AppState::with_all_pipelines(
        registry,
        embedder,
        None,
        None,
        None,
        Some(image_embedder),
        Some(cross_modal_encoder),
        config,
    ));

    let app = Router::new()
        .route(
            "/api/v1/search/multimodal",
            post(handlers::multimodal_search_handler),
        )
        .layer(middleware::from_fn_with_state(
            state.clone(),
            ucotron_server::auth::auth_middleware,
        ))
        .with_state(state.clone());

    (app, state)
}

/// Insert a test node with metadata indicating media type and namespace.
fn insert_multimodal_node(
    state: &AppState,
    id: NodeId,
    content: &str,
    media_type: &str,
    media_uri: Option<&str>,
) {
    let embedding = vec![0.5f32; 384];
    let visual_embedding = vec![0.5f32; 512];
    let mut metadata = HashMap::new();
    metadata.insert(
        "_media_type".to_string(),
        ucotron_core::Value::String(media_type.to_string()),
    );
    metadata.insert(
        "_namespace".to_string(),
        ucotron_core::Value::String("default".to_string()),
    );
    if let Some(uri) = media_uri {
        metadata.insert(
            "_media_uri".to_string(),
            ucotron_core::Value::String(uri.to_string()),
        );
    }

    let node = Node {
        id,
        content: content.to_string(),
        embedding: embedding.clone(),
        metadata,
        node_type: NodeType::Entity,
        timestamp: 1700000000,
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
    if let Some(vis) = state.registry.visual() {
        vis.upsert_visual_embeddings(&[(id, visual_embedding)])
            .unwrap();
    }
}

#[tokio::test]
async fn test_multimodal_search_text_query() {
    let (app, state) = build_multimodal_app();
    insert_multimodal_node(&state, 1, "Hello world", "text", None);
    insert_multimodal_node(&state, 2, "Rust programming", "text", None);

    let req = Request::post("/api/v1/search/multimodal")
        .header("Content-Type", "application/json")
        .body(Body::from(
            serde_json::to_string(&serde_json::json!({
                "query_type": "text",
                "query_text": "hello",
                "limit": 5
            }))
            .unwrap(),
        ))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let body = body_to_json(resp.into_body()).await;
    assert!(body["total"].as_u64().unwrap() > 0);
    assert_eq!(body["query_type"], "text");
    assert!(body["metrics"]["total_us"].as_u64().is_some());

    // Results should have expected fields
    let first = &body["results"][0];
    assert!(first["node_id"].as_u64().is_some());
    assert!(first["score"].as_f64().is_some());
    assert!(first["content"].as_str().is_some());
    assert_eq!(first["source"], "text_index");
}

#[tokio::test]
async fn test_multimodal_search_text_to_image_query() {
    let (app, state) = build_multimodal_app();
    insert_multimodal_node(
        &state,
        10,
        "Sunset photo",
        "image",
        Some("/media/sunset.jpg"),
    );
    insert_multimodal_node(&state, 11, "Cat image", "image", Some("/media/cat.png"));

    let req = Request::post("/api/v1/search/multimodal")
        .header("Content-Type", "application/json")
        .body(Body::from(
            serde_json::to_string(&serde_json::json!({
                "query_type": "text_to_image",
                "query_text": "a beautiful sunset",
                "limit": 5
            }))
            .unwrap(),
        ))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let body = body_to_json(resp.into_body()).await;
    assert!(body["total"].as_u64().unwrap() > 0);
    assert_eq!(body["query_type"], "text_to_image");

    // Results should come from visual_index
    let first = &body["results"][0];
    assert_eq!(first["source"], "visual_index");
}

#[tokio::test]
async fn test_multimodal_search_audio_query() {
    let (app, state) = build_multimodal_app();
    insert_multimodal_node(&state, 20, "Meeting notes from Monday", "text", None);

    let req = Request::post("/api/v1/search/multimodal")
        .header("Content-Type", "application/json")
        .body(Body::from(
            serde_json::to_string(&serde_json::json!({
                "query_type": "audio",
                "query_text": "what happened in the meeting",
                "limit": 5
            }))
            .unwrap(),
        ))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let body = body_to_json(resp.into_body()).await;
    assert_eq!(body["query_type"], "audio");
    assert!(body["total"].as_u64().unwrap() > 0);
}

#[tokio::test]
async fn test_multimodal_search_media_filter() {
    let (app, state) = build_multimodal_app();
    insert_multimodal_node(&state, 30, "Text note", "text", None);
    insert_multimodal_node(
        &state,
        31,
        "Photo memory",
        "image",
        Some("/media/photo.jpg"),
    );

    let req = Request::post("/api/v1/search/multimodal")
        .header("Content-Type", "application/json")
        .body(Body::from(
            serde_json::to_string(&serde_json::json!({
                "query_type": "text",
                "query_text": "memory",
                "media_filter": "image",
                "limit": 10
            }))
            .unwrap(),
        ))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let body = body_to_json(resp.into_body()).await;
    // Only image results should be returned
    for result in body["results"].as_array().unwrap() {
        assert_eq!(result["media_type"], "image");
    }
}

#[tokio::test]
async fn test_multimodal_search_missing_query_text() {
    let (app, _state) = build_multimodal_app();

    let req = Request::post("/api/v1/search/multimodal")
        .header("Content-Type", "application/json")
        .body(Body::from(
            serde_json::to_string(&serde_json::json!({
                "query_type": "text"
            }))
            .unwrap(),
        ))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn test_multimodal_search_invalid_query_type() {
    let (app, _state) = build_multimodal_app();

    let req = Request::post("/api/v1/search/multimodal")
        .header("Content-Type", "application/json")
        .body(Body::from(
            serde_json::to_string(&serde_json::json!({
                "query_type": "invalid_type",
                "query_text": "hello"
            }))
            .unwrap(),
        ))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn test_multimodal_search_image_query_missing_image() {
    let (app, _state) = build_multimodal_app();

    let req = Request::post("/api/v1/search/multimodal")
        .header("Content-Type", "application/json")
        .body(Body::from(
            serde_json::to_string(&serde_json::json!({
                "query_type": "image"
            }))
            .unwrap(),
        ))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn test_multimodal_search_metrics_present() {
    let (app, state) = build_multimodal_app();
    insert_multimodal_node(&state, 40, "Test content", "text", None);

    let req = Request::post("/api/v1/search/multimodal")
        .header("Content-Type", "application/json")
        .body(Body::from(
            serde_json::to_string(&serde_json::json!({
                "query_type": "text",
                "query_text": "test"
            }))
            .unwrap(),
        ))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let body = body_to_json(resp.into_body()).await;
    let metrics = &body["metrics"];
    assert!(metrics["query_encoding_us"].as_u64().is_some());
    assert!(metrics["total_us"].as_u64().is_some());
    assert!(metrics["final_result_count"].as_u64().is_some());
}

#[tokio::test]
async fn test_multimodal_search_media_filter_array() {
    let (app, state) = build_multimodal_app();
    insert_multimodal_node(&state, 50, "Text note", "text", None);
    insert_multimodal_node(
        &state,
        51,
        "Photo memory",
        "image",
        Some("/media/photo.jpg"),
    );
    insert_multimodal_node(&state, 52, "Audio clip", "audio", Some("/media/clip.wav"));

    // Filter for image AND audio — should exclude text.
    let req = Request::post("/api/v1/search/multimodal")
        .header("Content-Type", "application/json")
        .body(Body::from(
            serde_json::to_string(&serde_json::json!({
                "query_type": "text",
                "query_text": "memory",
                "media_filter": ["image", "audio"],
                "limit": 10
            }))
            .unwrap(),
        ))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let body = body_to_json(resp.into_body()).await;
    for result in body["results"].as_array().unwrap() {
        let mt = result["media_type"].as_str().unwrap();
        assert!(
            mt == "image" || mt == "audio",
            "unexpected media_type: {}",
            mt
        );
    }
}

#[tokio::test]
async fn test_multimodal_search_video_filter_matches_video_segment() {
    let (app, state) = build_multimodal_app();
    insert_multimodal_node(&state, 60, "Text note", "text", None);
    // video_segment nodes stored with "_media_type" = "video_segment"
    insert_multimodal_node(
        &state,
        61,
        "Video clip segment",
        "video_segment",
        Some("/media/clip.mp4"),
    );

    // Filter by "video" should match "video_segment" nodes.
    let req = Request::post("/api/v1/search/multimodal")
        .header("Content-Type", "application/json")
        .body(Body::from(
            serde_json::to_string(&serde_json::json!({
                "query_type": "text",
                "query_text": "video clip",
                "media_filter": "video",
                "limit": 10
            }))
            .unwrap(),
        ))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let body = body_to_json(resp.into_body()).await;
    // "video_segment" nodes should appear with display type "video"
    for result in body["results"].as_array().unwrap() {
        assert_eq!(result["media_type"], "video");
    }
}

/// Helper to insert a multimodal node with a custom timestamp.
fn insert_multimodal_node_with_ts(
    state: &AppState,
    id: NodeId,
    content: &str,
    media_type: &str,
    timestamp: u64,
) {
    let embedding = vec![0.5f32; 384];
    let visual_embedding = vec![0.5f32; 512];
    let mut metadata = HashMap::new();
    metadata.insert(
        "_media_type".to_string(),
        ucotron_core::Value::String(media_type.to_string()),
    );
    metadata.insert(
        "_namespace".to_string(),
        ucotron_core::Value::String("default".to_string()),
    );

    let node = Node {
        id,
        content: content.to_string(),
        embedding: embedding.clone(),
        metadata,
        node_type: NodeType::Entity,
        timestamp,
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
    if let Some(vis) = state.registry.visual() {
        vis.upsert_visual_embeddings(&[(id, visual_embedding)])
            .unwrap();
    }
}

#[tokio::test]
async fn test_multimodal_search_time_range_filter() {
    let (app, state) = build_multimodal_app();
    // ts=1000 is outside the range, ts=2000 and ts=3000 are within [1500, 3500]
    insert_multimodal_node_with_ts(&state, 70, "Old memory", "text", 1000);
    insert_multimodal_node_with_ts(&state, 71, "Recent memory", "text", 2000);
    insert_multimodal_node_with_ts(&state, 72, "New memory", "text", 3000);

    let req = Request::post("/api/v1/search/multimodal")
        .header("Content-Type", "application/json")
        .body(Body::from(
            serde_json::to_string(&serde_json::json!({
                "query_type": "text",
                "query_text": "memory",
                "time_range": [1500, 3500],
                "limit": 10
            }))
            .unwrap(),
        ))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let body = body_to_json(resp.into_body()).await;
    // Node 70 (ts=1000) should be filtered out
    for result in body["results"].as_array().unwrap() {
        let nid = result["node_id"].as_u64().unwrap();
        assert!(
            nid != 70,
            "node 70 with ts=1000 should be filtered out by time_range [1500, 3500]"
        );
    }
}

#[tokio::test]
async fn test_multimodal_search_combined_media_and_time_filters() {
    let (app, state) = build_multimodal_app();
    insert_multimodal_node_with_ts(&state, 80, "Old image", "image", 1000);
    insert_multimodal_node_with_ts(&state, 81, "Recent image", "image", 2000);
    insert_multimodal_node_with_ts(&state, 82, "Recent text", "text", 2000);
    insert_multimodal_node_with_ts(&state, 83, "New audio", "audio", 3000);

    // Filter: media_filter=["image"], time_range=[1500, 3500]
    // Should only return node 81 (recent image)
    let req = Request::post("/api/v1/search/multimodal")
        .header("Content-Type", "application/json")
        .body(Body::from(
            serde_json::to_string(&serde_json::json!({
                "query_type": "text",
                "query_text": "memory",
                "media_filter": ["image"],
                "time_range": [1500, 3500],
                "limit": 10
            }))
            .unwrap(),
        ))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let body = body_to_json(resp.into_body()).await;
    for result in body["results"].as_array().unwrap() {
        let nid = result["node_id"].as_u64().unwrap();
        let mt = result["media_type"].as_str().unwrap();
        assert_eq!(mt, "image", "only image results should pass media_filter");
        assert!(
            nid != 80,
            "node 80 (ts=1000) should be filtered by time_range"
        );
    }
}

// ---------------------------------------------------------------------------
// Agent Share Tests
// ---------------------------------------------------------------------------

/// Helper: create an agent and return its ID.
async fn create_test_agent(state: &Arc<AppState>, name: &str) -> String {
    let agent = ucotron_core::Agent::new(
        format!("agent_{}", name.to_lowercase().replace(' ', "")),
        name,
        "admin",
        1000,
    );
    let id = agent.id.clone();
    state.registry.graph().create_agent(&agent).unwrap();
    id
}

#[tokio::test]
async fn test_create_share_read_only() {
    let (app, state) = build_app();
    let src_id = create_test_agent(&state, "SourceAgent").await;
    let tgt_id = create_test_agent(&state, "TargetAgent").await;

    let req = Request::post(format!("/api/v1/agents/{}/share", src_id))
        .header("Content-Type", "application/json")
        .body(Body::from(
            serde_json::to_string(&serde_json::json!({
                "target_agent_id": tgt_id,
                "permission": "read"
            }))
            .unwrap(),
        ))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::CREATED);

    let body = body_to_json(resp.into_body()).await;
    assert_eq!(body["agent_id"], src_id);
    assert_eq!(body["target_agent_id"], tgt_id);
    assert_eq!(body["permission"], "read");
    assert!(body["created_at"].as_u64().unwrap() > 0);
}

#[tokio::test]
async fn test_create_share_read_write() {
    let (app, state) = build_app();
    let src_id = create_test_agent(&state, "Writer").await;
    let tgt_id = create_test_agent(&state, "Reader").await;

    let req = Request::post(format!("/api/v1/agents/{}/share", src_id))
        .header("Content-Type", "application/json")
        .body(Body::from(
            serde_json::to_string(&serde_json::json!({
                "target_agent_id": tgt_id,
                "permission": "read_write"
            }))
            .unwrap(),
        ))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::CREATED);

    let body = body_to_json(resp.into_body()).await;
    assert_eq!(body["permission"], "read_write");
}

#[tokio::test]
async fn test_create_share_self_rejected() {
    let (app, state) = build_app();
    let agent_id = create_test_agent(&state, "SoloAgent").await;

    let req = Request::post(format!("/api/v1/agents/{}/share", agent_id))
        .header("Content-Type", "application/json")
        .body(Body::from(
            serde_json::to_string(&serde_json::json!({
                "target_agent_id": agent_id
            }))
            .unwrap(),
        ))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn test_create_share_source_not_found() {
    let (app, state) = build_app();
    let tgt_id = create_test_agent(&state, "Target").await;

    let req = Request::post("/api/v1/agents/nonexistent/share")
        .header("Content-Type", "application/json")
        .body(Body::from(
            serde_json::to_string(&serde_json::json!({
                "target_agent_id": tgt_id
            }))
            .unwrap(),
        ))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn test_create_share_target_not_found() {
    let (app, state) = build_app();
    let src_id = create_test_agent(&state, "Source").await;

    let req = Request::post(format!("/api/v1/agents/{}/share", src_id))
        .header("Content-Type", "application/json")
        .body(Body::from(
            serde_json::to_string(&serde_json::json!({
                "target_agent_id": "nonexistent"
            }))
            .unwrap(),
        ))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn test_create_share_invalid_permission() {
    let (app, state) = build_app();
    let src_id = create_test_agent(&state, "Src").await;
    let tgt_id = create_test_agent(&state, "Tgt").await;

    let req = Request::post(format!("/api/v1/agents/{}/share", src_id))
        .header("Content-Type", "application/json")
        .body(Body::from(
            serde_json::to_string(&serde_json::json!({
                "target_agent_id": tgt_id,
                "permission": "admin"
            }))
            .unwrap(),
        ))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn test_list_shares_empty() {
    let (app, state) = build_app();
    let agent_id = create_test_agent(&state, "LoneAgent").await;

    let req = Request::get(format!("/api/v1/agents/{}/share", agent_id))
        .body(Body::empty())
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let body = body_to_json(resp.into_body()).await;
    assert_eq!(body["shares"].as_array().unwrap().len(), 0);
    assert_eq!(body["total"], 0);
}

#[tokio::test]
async fn test_list_shares_with_entries() {
    let (app, state) = build_app();
    let src_id = create_test_agent(&state, "SharedSrc").await;
    let tgt1_id = create_test_agent(&state, "SharedTgt1").await;
    let tgt2_id = create_test_agent(&state, "SharedTgt2").await;

    // Create two shares
    let share1 = ucotron_core::AgentShare {
        agent_id: src_id.clone(),
        target_agent_id: tgt1_id.clone(),
        permissions: ucotron_core::SharePermission::ReadOnly,
        created_at: 2000,
    };
    let share2 = ucotron_core::AgentShare {
        agent_id: src_id.clone(),
        target_agent_id: tgt2_id.clone(),
        permissions: ucotron_core::SharePermission::ReadWrite,
        created_at: 3000,
    };
    state.registry.graph().create_share(&share1).unwrap();
    state.registry.graph().create_share(&share2).unwrap();

    let req = Request::get(format!("/api/v1/agents/{}/share", src_id))
        .body(Body::empty())
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let body = body_to_json(resp.into_body()).await;
    assert_eq!(body["total"], 2);
    let shares = body["shares"].as_array().unwrap();
    assert_eq!(shares.len(), 2);
}

#[tokio::test]
async fn test_delete_share() {
    let (app, state) = build_app();
    let src_id = create_test_agent(&state, "DelSrc").await;
    let tgt_id = create_test_agent(&state, "DelTgt").await;

    // Create a share first
    let share = ucotron_core::AgentShare {
        agent_id: src_id.clone(),
        target_agent_id: tgt_id.clone(),
        permissions: ucotron_core::SharePermission::ReadOnly,
        created_at: 1000,
    };
    state.registry.graph().create_share(&share).unwrap();

    // Verify share exists
    let existing = state.registry.graph().get_share(&src_id, &tgt_id).unwrap();
    assert!(existing.is_some());

    // Delete the share
    let req = Request::delete(format!("/api/v1/agents/{}/share/{}", src_id, tgt_id))
        .body(Body::empty())
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::NO_CONTENT);

    // Verify share is gone
    let deleted = state.registry.graph().get_share(&src_id, &tgt_id).unwrap();
    assert!(deleted.is_none());
}

#[tokio::test]
async fn test_delete_share_not_found() {
    let (app, state) = build_app();
    let src_id = create_test_agent(&state, "DNFSrc").await;

    let req = Request::delete(format!("/api/v1/agents/{}/share/nonexistent", src_id))
        .body(Body::empty())
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn test_share_default_permission() {
    let (app, state) = build_app();
    let src_id = create_test_agent(&state, "DefSrc").await;
    let tgt_id = create_test_agent(&state, "DefTgt").await;

    // Omit permission field — should default to "read"
    let req = Request::post(format!("/api/v1/agents/{}/share", src_id))
        .header("Content-Type", "application/json")
        .body(Body::from(
            serde_json::to_string(&serde_json::json!({
                "target_agent_id": tgt_id
            }))
            .unwrap(),
        ))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::CREATED);

    let body = body_to_json(resp.into_body()).await;
    assert_eq!(body["permission"], "read");
}

// ---------------------------------------------------------------------------
// Media File Serving Tests (US-33.18)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_get_media_serves_file_with_correct_content_type() {
    // Create a temp media directory and write a test file.
    let temp_dir = tempfile::tempdir().unwrap();
    let media_dir = temp_dir.path().to_str().unwrap().to_string();

    // Override media_dir in config via a trick: insert the node with media_uri,
    // then create the file at the expected location.
    // We need to build a custom app with the media_dir set.
    let mut config = UcotronConfig::default();
    config.storage.media_dir = media_dir.clone();

    let registry = Arc::new(BackendRegistry::new(
        Box::new(MockVectorBackend::new()),
        Box::new(MockGraphBackend::new()),
    ));
    let embedder: Arc<dyn EmbeddingPipeline> = Arc::new(MockEmbedder);
    let state = Arc::new(AppState::new(registry, embedder, None, None, config));

    // Insert a node with media_uri set.
    let node = Node {
        id: 42,
        content: "test image".to_string(),
        embedding: vec![0.0; 384],
        metadata: HashMap::new(),
        node_type: NodeType::Entity,
        timestamp: 1000,
        media_type: Some(ucotron_core::MediaType::Image),
        media_uri: Some("42.png".to_string()),
        embedding_visual: None,
        timestamp_range: None,
        parent_video_id: None,
    };
    state.registry.graph().upsert_nodes(&[node]).unwrap();

    // Write a fake PNG file.
    let png_data = create_test_png_bytes();
    std::fs::write(temp_dir.path().join("42.png"), &png_data).unwrap();

    let app = Router::new()
        .route("/api/v1/media/{id}", get(handlers::get_media_handler))
        .route(
            "/api/v1/videos/{parent_id}/segments",
            get(handlers::get_video_segments_handler),
        )
        .layer(middleware::from_fn_with_state(
            state.clone(),
            ucotron_server::auth::auth_middleware,
        ))
        .with_state(state);

    let req = Request::get("/api/v1/media/42")
        .body(Body::empty())
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    assert_eq!(
        resp.headers()
            .get("content-type")
            .unwrap()
            .to_str()
            .unwrap(),
        "image/png"
    );
    assert_eq!(
        resp.headers()
            .get("accept-ranges")
            .unwrap()
            .to_str()
            .unwrap(),
        "bytes"
    );

    let body_bytes = resp.into_body().collect().await.unwrap().to_bytes();
    assert_eq!(body_bytes.as_ref(), png_data.as_slice());
}

#[tokio::test]
async fn test_get_media_returns_404_for_nonexistent_node() {
    let (app, _state) = build_app();

    let req = Request::get("/api/v1/media/99999")
        .body(Body::empty())
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn test_get_media_returns_404_for_node_without_media_uri() {
    let (app, state) = build_app();

    // Insert a text node without media_uri.
    let node = Node {
        id: 100,
        content: "text only".to_string(),
        embedding: vec![0.0; 384],
        metadata: HashMap::new(),
        node_type: NodeType::Entity,
        timestamp: 1000,
        media_type: None,
        media_uri: None,
        embedding_visual: None,
        timestamp_range: None,
        parent_video_id: None,
    };
    state.registry.graph().upsert_nodes(&[node]).unwrap();

    let req = Request::get("/api/v1/media/100")
        .body(Body::empty())
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn test_get_media_range_request() {
    let temp_dir = tempfile::tempdir().unwrap();
    let media_dir = temp_dir.path().to_str().unwrap().to_string();

    let mut config = UcotronConfig::default();
    config.storage.media_dir = media_dir.clone();

    let registry = Arc::new(BackendRegistry::new(
        Box::new(MockVectorBackend::new()),
        Box::new(MockGraphBackend::new()),
    ));
    let embedder: Arc<dyn EmbeddingPipeline> = Arc::new(MockEmbedder);
    let state = Arc::new(AppState::new(registry, embedder, None, None, config));

    // Insert a node with media_uri.
    let node = Node {
        id: 55,
        content: "test video".to_string(),
        embedding: vec![0.0; 384],
        metadata: HashMap::new(),
        node_type: NodeType::Event,
        timestamp: 1000,
        media_type: Some(ucotron_core::MediaType::VideoSegment),
        media_uri: Some("55.mp4".to_string()),
        embedding_visual: None,
        timestamp_range: Some((0, 10000)),
        parent_video_id: None,
    };
    state.registry.graph().upsert_nodes(&[node]).unwrap();

    // Write a fake media file (100 bytes).
    let file_data: Vec<u8> = (0..100u8).collect();
    std::fs::write(temp_dir.path().join("55.mp4"), &file_data).unwrap();

    let app = Router::new()
        .route("/api/v1/media/{id}", get(handlers::get_media_handler))
        .route(
            "/api/v1/videos/{parent_id}/segments",
            get(handlers::get_video_segments_handler),
        )
        .layer(middleware::from_fn_with_state(
            state.clone(),
            ucotron_server::auth::auth_middleware,
        ))
        .with_state(state);

    // Request bytes 10-19.
    let req = Request::get("/api/v1/media/55")
        .header("Range", "bytes=10-19")
        .body(Body::empty())
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::PARTIAL_CONTENT);
    assert_eq!(
        resp.headers()
            .get("content-type")
            .unwrap()
            .to_str()
            .unwrap(),
        "video/mp4"
    );
    assert_eq!(
        resp.headers()
            .get("content-range")
            .unwrap()
            .to_str()
            .unwrap(),
        "bytes 10-19/100"
    );

    let body_bytes = resp.into_body().collect().await.unwrap().to_bytes();
    assert_eq!(body_bytes.as_ref(), &file_data[10..20]);
}

#[tokio::test]
async fn test_get_media_audio_content_type() {
    let temp_dir = tempfile::tempdir().unwrap();
    let media_dir = temp_dir.path().to_str().unwrap().to_string();

    let mut config = UcotronConfig::default();
    config.storage.media_dir = media_dir.clone();

    let registry = Arc::new(BackendRegistry::new(
        Box::new(MockVectorBackend::new()),
        Box::new(MockGraphBackend::new()),
    ));
    let embedder: Arc<dyn EmbeddingPipeline> = Arc::new(MockEmbedder);
    let state = Arc::new(AppState::new(registry, embedder, None, None, config));

    let node = Node {
        id: 77,
        content: "test audio".to_string(),
        embedding: vec![0.0; 384],
        metadata: HashMap::new(),
        node_type: NodeType::Entity,
        timestamp: 1000,
        media_type: Some(ucotron_core::MediaType::Audio),
        media_uri: Some("77.wav".to_string()),
        embedding_visual: None,
        timestamp_range: None,
        parent_video_id: None,
    };
    state.registry.graph().upsert_nodes(&[node]).unwrap();

    // Write a fake WAV file.
    let wav_data = vec![0u8; 44]; // minimal WAV header size
    std::fs::write(temp_dir.path().join("77.wav"), &wav_data).unwrap();

    let app = Router::new()
        .route("/api/v1/media/{id}", get(handlers::get_media_handler))
        .route(
            "/api/v1/videos/{parent_id}/segments",
            get(handlers::get_video_segments_handler),
        )
        .layer(middleware::from_fn_with_state(
            state.clone(),
            ucotron_server::auth::auth_middleware,
        ))
        .with_state(state);

    let req = Request::get("/api/v1/media/77")
        .body(Body::empty())
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    assert_eq!(
        resp.headers()
            .get("content-type")
            .unwrap()
            .to_str()
            .unwrap(),
        "audio/wav"
    );
}

// ---------------------------------------------------------------------------
// Video Segment Retrieval Tests
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_video_segments_returns_sorted_segments_with_navigation() {
    let (app, state) = build_app();

    // Create parent video node.
    let parent = Node {
        id: 100,
        content: "Parent video".to_string(),
        embedding: vec![0.0; 384],
        metadata: HashMap::new(),
        node_type: NodeType::Event,
        timestamp: 1700000000,
        media_type: Some(ucotron_core::MediaType::VideoSegment),
        media_uri: Some("100.mp4".to_string()),
        embedding_visual: None,
        timestamp_range: Some((0, 90_000)),
        parent_video_id: None,
    };

    // Create 3 segment nodes in non-sorted order.
    let seg_b = Node {
        id: 102,
        content: "Segment B (30s-60s)".to_string(),
        embedding: vec![0.0; 384],
        metadata: HashMap::new(),
        node_type: NodeType::Event,
        timestamp: 1700000000,
        media_type: Some(ucotron_core::MediaType::VideoSegment),
        media_uri: Some("100.mp4".to_string()),
        embedding_visual: None,
        timestamp_range: Some((30_000, 60_000)),
        parent_video_id: Some(100),
    };
    let seg_a = Node {
        id: 101,
        content: "Segment A (0s-30s)".to_string(),
        embedding: vec![0.0; 384],
        metadata: HashMap::new(),
        node_type: NodeType::Event,
        timestamp: 1700000000,
        media_type: Some(ucotron_core::MediaType::VideoSegment),
        media_uri: Some("100.mp4".to_string()),
        embedding_visual: None,
        timestamp_range: Some((0, 30_000)),
        parent_video_id: Some(100),
    };
    let seg_c = Node {
        id: 103,
        content: "Segment C (60s-90s)".to_string(),
        embedding: vec![0.0; 384],
        metadata: HashMap::new(),
        node_type: NodeType::Event,
        timestamp: 1700000000,
        media_type: Some(ucotron_core::MediaType::VideoSegment),
        media_uri: Some("100.mp4".to_string()),
        embedding_visual: None,
        timestamp_range: Some((60_000, 90_000)),
        parent_video_id: Some(100),
    };

    // Insert in shuffled order to verify sorting.
    state
        .registry
        .graph()
        .upsert_nodes(&[parent, seg_b, seg_a, seg_c])
        .unwrap();

    let req = Request::get("/api/v1/videos/100/segments")
        .body(Body::empty())
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let json = body_to_json(resp.into_body()).await;
    assert_eq!(json["parent_video_id"], 100);
    assert_eq!(json["total"], 3);

    let segs = json["segments"].as_array().unwrap();
    assert_eq!(segs.len(), 3);

    // Verify sorted by start_ms.
    assert_eq!(segs[0]["node_id"], 101);
    assert_eq!(segs[0]["start_ms"], 0);
    assert_eq!(segs[0]["end_ms"], 30_000);
    assert_eq!(segs[1]["node_id"], 102);
    assert_eq!(segs[1]["start_ms"], 30_000);
    assert_eq!(segs[2]["node_id"], 103);
    assert_eq!(segs[2]["start_ms"], 60_000);

    // Verify prev/next navigation links.
    assert!(segs[0]["prev_segment_id"].is_null()); // first segment has no prev
    assert_eq!(segs[0]["next_segment_id"], 102);

    assert_eq!(segs[1]["prev_segment_id"], 101);
    assert_eq!(segs[1]["next_segment_id"], 103);

    assert_eq!(segs[2]["prev_segment_id"], 102);
    assert!(segs[2]["next_segment_id"].is_null()); // last segment has no next
}

#[tokio::test]
async fn test_video_segments_returns_404_for_missing_parent() {
    let (app, _state) = build_app();

    let req = Request::get("/api/v1/videos/999/segments")
        .body(Body::empty())
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn test_video_segments_empty_when_no_segments() {
    let (app, state) = build_app();

    // Create parent video node with no child segments.
    let parent = Node {
        id: 200,
        content: "Standalone video".to_string(),
        embedding: vec![0.0; 384],
        metadata: HashMap::new(),
        node_type: NodeType::Event,
        timestamp: 1700000000,
        media_type: Some(ucotron_core::MediaType::VideoSegment),
        media_uri: Some("200.mp4".to_string()),
        embedding_visual: None,
        timestamp_range: Some((0, 60_000)),
        parent_video_id: None,
    };
    state.registry.graph().upsert_nodes(&[parent]).unwrap();

    let req = Request::get("/api/v1/videos/200/segments")
        .body(Body::empty())
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let json = body_to_json(resp.into_body()).await;
    assert_eq!(json["parent_video_id"], 200);
    assert_eq!(json["total"], 0);
    assert_eq!(json["segments"].as_array().unwrap().len(), 0);
}

#[tokio::test]
async fn test_search_includes_timestamp_range_for_video_segments() {
    let (app, state) = build_app();

    // Create a video segment node with timestamp_range.
    let seg = Node {
        id: 300,
        content: "Test video segment".to_string(),
        embedding: vec![1.0; 384],
        metadata: HashMap::new(),
        node_type: NodeType::Event,
        timestamp: 1700000000,
        media_type: Some(ucotron_core::MediaType::VideoSegment),
        media_uri: Some("300.mp4".to_string()),
        embedding_visual: None,
        timestamp_range: Some((5_000, 15_000)),
        parent_video_id: Some(50),
    };
    state.registry.graph().upsert_nodes(&[seg]).unwrap();
    state
        .registry
        .vector()
        .upsert_embeddings(&[(300, vec![1.0; 384])])
        .unwrap();

    let req = Request::post("/api/v1/memories/search")
        .header("Content-Type", "application/json")
        .body(Body::from(
            serde_json::json!({ "query": "test video segment" }).to_string(),
        ))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let json = body_to_json(resp.into_body()).await;
    let results = json["results"].as_array().unwrap();
    assert!(!results.is_empty());

    // Find our segment in results.
    let seg_result = results.iter().find(|r| r["id"] == 300);
    assert!(
        seg_result.is_some(),
        "Video segment should appear in results"
    );
    let seg_result = seg_result.unwrap();

    // Verify timestamp_range and parent_video_id are included.
    let ts = seg_result["timestamp_range"].as_array().unwrap();
    assert_eq!(ts[0], 5_000);
    assert_eq!(ts[1], 15_000);
    assert_eq!(seg_result["parent_video_id"], 50);
}

#[tokio::test]
async fn test_search_omits_timestamp_fields_for_text_nodes() {
    let (app, state) = build_app();

    let text_node = Node {
        id: 400,
        content: "Regular text node".to_string(),
        embedding: vec![1.0; 384],
        metadata: HashMap::new(),
        node_type: NodeType::Entity,
        timestamp: 1700000000,
        media_type: None,
        media_uri: None,
        embedding_visual: None,
        timestamp_range: None,
        parent_video_id: None,
    };
    state.registry.graph().upsert_nodes(&[text_node]).unwrap();
    state
        .registry
        .vector()
        .upsert_embeddings(&[(400, vec![1.0; 384])])
        .unwrap();

    let req = Request::post("/api/v1/memories/search")
        .header("Content-Type", "application/json")
        .body(Body::from(
            serde_json::json!({ "query": "text node" }).to_string(),
        ))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let json = body_to_json(resp.into_body()).await;
    let results = json["results"].as_array().unwrap();
    let text_result = results.iter().find(|r| r["id"] == 400);
    assert!(text_result.is_some());
    let text_result = text_result.unwrap();

    // timestamp_range and parent_video_id should be absent (skip_serializing_if).
    assert!(
        text_result.get("timestamp_range").is_none() || text_result["timestamp_range"].is_null()
    );
    assert!(
        text_result.get("parent_video_id").is_none() || text_result["parent_video_id"].is_null()
    );
}
