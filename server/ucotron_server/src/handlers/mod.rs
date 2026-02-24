//! Axum route handlers for the Ucotron REST API.

use std::sync::atomic::Ordering;
use std::sync::Arc;

use axum::extract::{Multipart, Path, Query, State};
use axum::http::HeaderMap;
use axum::{Extension, Json};
use tracing::instrument;

use ucotron_connectors::next_fire_time;
use ucotron_core::Agent;
use ucotron_extraction::ingestion::{IngestionConfig, IngestionOrchestrator};
use ucotron_extraction::retrieval::{RetrievalConfig, RetrievalOrchestrator};

use crate::auth::{require_namespace_access, require_role, AuthContext};
use crate::error::AppError;
use crate::state::AppState;
use crate::types::*;

/// Extract the namespace from the `X-Ucotron-Namespace` header, defaulting to "default".
fn extract_namespace(headers: &HeaderMap) -> String {
    headers
        .get("X-Ucotron-Namespace")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("default")
        .to_string()
}

/// Record an ingestion event in Prometheus metrics.
fn record_ingestion(state: &AppState) {
    if let Some(prom) = &state.prometheus {
        prom.ingestions_total.inc();
    }
}

/// Record a search event in Prometheus metrics.
fn record_search(state: &AppState) {
    if let Some(prom) = &state.prometheus {
        prom.searches_total.inc();
    }
}

/// Persist uploaded media bytes to disk and return the file path relative to media_dir.
///
/// Creates the media directory if it doesn't exist. Files are stored as `{node_id}.{ext}`.
/// Returns the relative path `"{node_id}.{ext}"` on success.
fn persist_media_file(state: &AppState, node_id: u64, ext: &str, data: &[u8]) -> Result<String, AppError> {
    let media_dir = std::path::Path::new(state.config.storage.effective_media_dir());
    std::fs::create_dir_all(media_dir)
        .map_err(|e| AppError::internal(format!("Failed to create media directory: {}", e)))?;
    let filename = format!("{}.{}", node_id, ext);
    let file_path = media_dir.join(&filename);
    std::fs::write(&file_path, data)
        .map_err(|e| AppError::internal(format!("Failed to write media file: {}", e)))?;
    Ok(filename)
}

/// Detect Content-Type from file extension.
fn content_type_for_ext(ext: &str) -> &'static str {
    match ext {
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "gif" => "image/gif",
        "webp" => "image/webp",
        "bmp" => "image/bmp",
        "tiff" | "tif" => "image/tiff",
        "svg" => "image/svg+xml",
        "wav" => "audio/wav",
        "mp3" => "audio/mpeg",
        "ogg" => "audio/ogg",
        "flac" => "audio/flac",
        "mp4" => "video/mp4",
        "avi" => "video/x-msvideo",
        "mov" => "video/quicktime",
        "webm" => "video/webm",
        "mkv" => "video/x-matroska",
        "pdf" => "application/pdf",
        _ => "application/octet-stream",
    }
}

// ---------------------------------------------------------------------------
// Health & Metrics
// ---------------------------------------------------------------------------

/// Health check endpoint returning server status and component availability.
#[utoipa::path(
    get,
    path = "/api/v1/health",
    tag = "Health",
    responses(
        (status = 200, description = "Server is healthy", body = HealthResponse)
    )
)]
pub async fn health_handler(
    State(state): State<Arc<AppState>>,
) -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok".into(),
        version: env!("CARGO_PKG_VERSION").into(),
        instance_id: state.instance_id.clone(),
        instance_role: state.config.instance.role.clone(),
        storage_mode: state.config.storage.mode.clone(),
        vector_backend: state.config.storage.vector.backend.clone(),
        graph_backend: state.config.storage.graph.backend.clone(),
        models: ModelStatus {
            embedder_loaded: true, // Embedder always present (stub or real)
            embedding_model: state.config.models.embedding_model.clone(),
            ner_loaded: state.ner.is_some(),
            relation_extractor_loaded: state.relation_extractor.is_some(),
            transcriber_loaded: state.transcriber.is_some(),
            image_embedder_loaded: state.image_embedder.is_some(),
            cross_modal_encoder_loaded: state.cross_modal_encoder.is_some(),
            ocr_pipeline_loaded: state.ocr_pipeline.is_some(),
        },
    })
}

/// Server metrics including request counts and uptime.
#[utoipa::path(
    get,
    path = "/api/v1/metrics",
    tag = "Health",
    responses(
        (status = 200, description = "Server metrics", body = MetricsResponse)
    )
)]
pub async fn metrics_handler(
    State(state): State<Arc<AppState>>,
) -> Json<MetricsResponse> {
    Json(MetricsResponse {
        instance_id: state.instance_id.clone(),
        total_requests: state.total_requests.load(Ordering::Relaxed),
        total_ingestions: state.total_ingestions.load(Ordering::Relaxed),
        total_searches: state.total_searches.load(Ordering::Relaxed),
        uptime_secs: state.start_time.elapsed().as_secs(),
    })
}

// ---------------------------------------------------------------------------
// Memories CRUD
// ---------------------------------------------------------------------------

/// Ingest text through the extraction pipeline to create memories.
///
/// The text is chunked, embedded, entities are extracted via NER, relations via
/// the relation extractor, and the resulting knowledge is stored in the graph.
#[utoipa::path(
    post,
    path = "/api/v1/memories",
    tag = "Memories",
    request_body = CreateMemoryRequest,
    params(
        ("X-Ucotron-Namespace" = Option<String>, Header, description = "Namespace for multi-tenancy isolation")
    ),
    responses(
        (status = 201, description = "Memory created successfully", body = CreateMemoryResponse),
        (status = 400, description = "Invalid request", body = ApiErrorResponse),
        (status = 403, description = "Read-only instance", body = ApiErrorResponse),
        (status = 500, description = "Internal server error", body = ApiErrorResponse)
    )
)]
pub async fn create_memory_handler(
    State(state): State<Arc<AppState>>,
    Extension(auth): Extension<AuthContext>,
    headers: HeaderMap,
    Json(body): Json<CreateMemoryRequest>,
) -> Result<(axum::http::StatusCode, Json<CreateMemoryResponse>), AppError> {
    require_role(&auth, ucotron_config::AuthRole::Writer)?;
    let ns = extract_namespace(&headers);
    require_namespace_access(&auth, &ns)?;
    if state.is_reader_only() {
        return Err(AppError::read_only());
    }
    let _namespace = ns;
    state.total_ingestions.fetch_add(1, Ordering::Relaxed);
    record_ingestion(&state);

    if body.text.trim().is_empty() {
        return Err(AppError::bad_request("text must not be empty"));
    }

    let next_id = state.alloc_next_node_id();
    let config = IngestionConfig {
        next_node_id: Some(next_id),
        ..IngestionConfig::default()
    };

    let ner_ref: Option<&dyn ucotron_extraction::NerPipeline> =
        state.ner.as_ref().map(|n| n.as_ref());
    let re_ref: Option<&dyn ucotron_extraction::RelationExtractor> =
        state.relation_extractor.as_ref().map(|r| r.as_ref());

    let mut orchestrator = IngestionOrchestrator::new(
        &state.registry,
        state.embedder.as_ref(),
        ner_ref,
        re_ref,
        config,
    );

    let result = orchestrator
        .ingest(&body.text)
        .map_err(|e| AppError::internal(format!("Ingestion failed: {}", e)))?;

    // Advance the shared ID counter past what was used.
    let ids_used = result.chunk_node_ids.len() + result.entity_node_ids.len();
    {
        let mut id_lock = state.next_node_id.lock().unwrap();
        let used_max = next_id + ids_used as u64;
        if used_max > *id_lock {
            *id_lock = used_max;
        }
    }

    // Tag all created nodes with the namespace for multi-tenancy isolation.
    let namespace = extract_namespace(&headers);
    tag_nodes_with_namespace(&state, &result.chunk_node_ids, &namespace);
    tag_nodes_with_namespace(&state, &result.entity_node_ids, &namespace);

    let response = CreateMemoryResponse {
        chunk_node_ids: result.chunk_node_ids,
        entity_node_ids: result.entity_node_ids,
        edges_created: result.edges_created.len(),
        metrics: IngestionMetricsResponse {
            chunks_processed: result.metrics.chunks_processed,
            entities_extracted: result.metrics.entities_extracted,
            relations_extracted: result.metrics.relations_extracted,
            contradictions_detected: result.metrics.contradictions_detected,
            total_us: result.metrics.total_us,
        },
    };

    Ok((axum::http::StatusCode::CREATED, Json(response)))
}

/// List memories with optional filtering by node type and pagination.
#[utoipa::path(
    get,
    path = "/api/v1/memories",
    tag = "Memories",
    params(ListMemoriesParams, ("X-Ucotron-Namespace" = Option<String>, Header, description = "Namespace for multi-tenancy isolation")),
    responses(
        (status = 200, description = "List of memories", body = Vec<MemoryResponse>),
        (status = 500, description = "Internal server error", body = ApiErrorResponse)
    )
)]
pub async fn list_memories_handler(
    State(state): State<Arc<AppState>>,
    Extension(auth): Extension<AuthContext>,
    headers: HeaderMap,
    Query(params): Query<ListMemoriesParams>,
) -> Result<Json<Vec<MemoryResponse>>, AppError> {
    require_role(&auth, ucotron_config::AuthRole::Reader)?;
    let namespace = extract_namespace(&headers);
    require_namespace_access(&auth, &namespace)?;

    // Use a zero-vector search to get "all" nodes ranked by recency.
    // This is a pragmatic approach — a true "list all" would need a scan API.
    let top_k = params.limit.min(1000);
    let query_vec = vec![0.0f32; 384];
    let results = state
        .registry
        .vector()
        .search(&query_vec, top_k + params.offset)
        .map_err(|e| AppError::internal(format!("Vector search failed: {}", e)))?;

    let mut memories = Vec::new();
    for (i, (node_id, _score)) in results.iter().enumerate() {
        if i < params.offset {
            continue;
        }
        if memories.len() >= params.limit {
            break;
        }
        if let Ok(Some(node)) = state.registry.graph().get_node(*node_id) {
            // Filter by namespace for multi-tenancy isolation.
            if !node_matches_namespace(&node, &namespace) {
                continue;
            }
            if let Some(ref nt_filter) = params.node_type {
                if let Some(expected) = parse_node_type(nt_filter) {
                    if node.node_type != expected {
                        continue;
                    }
                }
            }
            memories.push(node_to_memory_response(&node));
        }
    }

    Ok(Json(memories))
}

/// Get a single memory node by ID.
#[utoipa::path(
    get,
    path = "/api/v1/memories/{id}",
    tag = "Memories",
    params(
        ("id" = u64, Path, description = "Memory node ID")
    ),
    responses(
        (status = 200, description = "Memory found", body = MemoryResponse),
        (status = 404, description = "Memory not found", body = ApiErrorResponse)
    )
)]
pub async fn get_memory_handler(
    State(state): State<Arc<AppState>>,
    Extension(auth): Extension<AuthContext>,
    Path(id): Path<u64>,
) -> Result<Json<MemoryResponse>, AppError> {
    require_role(&auth, ucotron_config::AuthRole::Reader)?;
    let node = state
        .registry
        .graph()
        .get_node(id)
        .map_err(|e| AppError::internal(format!("Graph lookup failed: {}", e)))?
        .ok_or_else(|| AppError::not_found(format!("Memory {} not found", id)))?;

    Ok(Json(node_to_memory_response(&node)))
}

/// Update a memory node's content and/or metadata.
#[utoipa::path(
    put,
    path = "/api/v1/memories/{id}",
    tag = "Memories",
    request_body = UpdateMemoryRequest,
    params(
        ("id" = u64, Path, description = "Memory node ID")
    ),
    responses(
        (status = 200, description = "Memory updated", body = MemoryResponse),
        (status = 403, description = "Read-only instance", body = ApiErrorResponse),
        (status = 404, description = "Memory not found", body = ApiErrorResponse),
        (status = 500, description = "Internal server error", body = ApiErrorResponse)
    )
)]
pub async fn update_memory_handler(
    State(state): State<Arc<AppState>>,
    Extension(auth): Extension<AuthContext>,
    Path(id): Path<u64>,
    Json(body): Json<UpdateMemoryRequest>,
) -> Result<Json<MemoryResponse>, AppError> {
    require_role(&auth, ucotron_config::AuthRole::Writer)?;
    if state.is_reader_only() {
        return Err(AppError::read_only());
    }
    let mut node = state
        .registry
        .graph()
        .get_node(id)
        .map_err(|e| AppError::internal(format!("Graph lookup failed: {}", e)))?
        .ok_or_else(|| AppError::not_found(format!("Memory {} not found", id)))?;

    if let Some(content) = body.content {
        node.content = content;

        // Re-embed the updated content.
        let embedding = state
            .embedder
            .embed_text(&node.content)
            .map_err(|e| AppError::internal(format!("Embedding failed: {}", e)))?;
        node.embedding = embedding.clone();

        // Update vector index.
        state
            .registry
            .vector()
            .upsert_embeddings(&[(id, embedding)])
            .map_err(|e| AppError::internal(format!("Vector upsert failed: {}", e)))?;
    }

    // Merge metadata.
    for (k, v) in body.metadata {
        let core_val = json_to_core_value(&v);
        node.metadata.insert(k, core_val);
    }

    // Persist node update.
    state
        .registry
        .graph()
        .upsert_nodes(&[node.clone()])
        .map_err(|e| AppError::internal(format!("Node upsert failed: {}", e)))?;

    Ok(Json(node_to_memory_response(&node)))
}

/// Delete a memory node (soft delete — removes from vector index and clears content).
#[utoipa::path(
    delete,
    path = "/api/v1/memories/{id}",
    tag = "Memories",
    params(
        ("id" = u64, Path, description = "Memory node ID")
    ),
    responses(
        (status = 204, description = "Memory deleted"),
        (status = 403, description = "Read-only instance", body = ApiErrorResponse),
        (status = 404, description = "Memory not found", body = ApiErrorResponse)
    )
)]
pub async fn delete_memory_handler(
    State(state): State<Arc<AppState>>,
    Extension(auth): Extension<AuthContext>,
    Path(id): Path<u64>,
) -> Result<axum::http::StatusCode, AppError> {
    require_role(&auth, ucotron_config::AuthRole::Writer)?;
    if state.is_reader_only() {
        return Err(AppError::read_only());
    }
    // Verify existence first.
    let _node = state
        .registry
        .graph()
        .get_node(id)
        .map_err(|e| AppError::internal(format!("Graph lookup failed: {}", e)))?
        .ok_or_else(|| AppError::not_found(format!("Memory {} not found", id)))?;

    // Delete from vector index.
    state
        .registry
        .vector()
        .delete(&[id])
        .map_err(|e| AppError::internal(format!("Vector delete failed: {}", e)))?;

    // Note: Full node deletion from graph backend would require a delete_node
    // method on GraphBackend. For now we remove the vector embedding making it
    // unsearchable, and upsert with empty content to mark as deleted.
    let deleted_node = ucotron_core::Node {
        id,
        content: String::new(),
        embedding: Vec::new(),
        metadata: {
            let mut m = std::collections::HashMap::new();
            m.insert("deleted".into(), ucotron_core::Value::Bool(true));
            m
        },
        node_type: _node.node_type,
        timestamp: _node.timestamp,
        media_type: None,
        media_uri: None,
        embedding_visual: None,
        timestamp_range: None,
        parent_video_id: None,
    };
    state
        .registry
        .graph()
        .upsert_nodes(&[deleted_node])
        .map_err(|e| AppError::internal(format!("Node upsert failed: {}", e)))?;

    Ok(axum::http::StatusCode::NO_CONTENT)
}

// ---------------------------------------------------------------------------
// Search
// ---------------------------------------------------------------------------

/// Semantic search via the retrieval pipeline (vector + graph + community re-ranking).
#[utoipa::path(
    post,
    path = "/api/v1/memories/search",
    tag = "Search",
    request_body = SearchRequest,
    params(
        ("X-Ucotron-Namespace" = Option<String>, Header, description = "Namespace for multi-tenancy isolation")
    ),
    responses(
        (status = 200, description = "Search results", body = SearchResponse),
        (status = 500, description = "Internal server error", body = ApiErrorResponse)
    )
)]
#[instrument(name = "search", skip(state, auth, headers, body), fields(namespace, query_len))]
pub async fn search_handler(
    State(state): State<Arc<AppState>>,
    Extension(auth): Extension<AuthContext>,
    headers: HeaderMap,
    Json(body): Json<SearchRequest>,
) -> Result<Json<SearchResponse>, AppError> {
    require_role(&auth, ucotron_config::AuthRole::Reader)?;
    let namespace = extract_namespace(&headers);
    tracing::Span::current().record("namespace", &namespace.as_str());
    tracing::Span::current().record("query_len", body.query.len());
    require_namespace_access(&auth, &namespace)?;
    state.total_searches.fetch_add(1, Ordering::Relaxed);
    record_search(&state);

    let limit = body.limit.unwrap_or(10);

    let mut retrieval_config = RetrievalConfig {
        final_top_k: limit,
        ..RetrievalConfig::default()
    };

    if let Some(ref nt) = body.node_type {
        retrieval_config.entity_type_filter = parse_node_type(nt);
    }
    if let Some(tr) = body.time_range {
        retrieval_config.time_range = Some(tr);
    }
    if let Some(ref mindset) = body.query_mindset {
        retrieval_config.query_mindset = parse_mindset_tag(mindset);
    }

    let ner_ref: Option<&dyn ucotron_extraction::NerPipeline> =
        state.ner.as_ref().map(|n| n.as_ref());

    let mut orchestrator = RetrievalOrchestrator::new(
        &state.registry,
        state.embedder.as_ref(),
        ner_ref,
        retrieval_config,
    );
    if let Some(detector) = build_mindset_detector(&state.config.mindset) {
        orchestrator = orchestrator.with_mindset_detector(detector);
    }

    let result = orchestrator
        .retrieve(&body.query)
        .map_err(|e| AppError::internal(format!("Retrieval failed: {}", e)))?;

    // Filter results by namespace for multi-tenancy isolation.
    let results: Vec<SearchResultItem> = result
        .memories
        .iter()
        .filter(|m| node_matches_namespace(&m.node, &namespace))
        .map(|m| SearchResultItem {
            id: m.node.id,
            content: m.node.content.clone(),
            node_type: format!("{:?}", m.node.node_type),
            score: m.score,
            vector_sim: m.vector_sim,
            graph_centrality: m.graph_centrality,
            recency: m.recency,
            mindset_score: m.mindset_score,
            timestamp_range: m.node.timestamp_range,
            parent_video_id: m.node.parent_video_id,
        })
        .collect();

    let total = results.len();
    Ok(Json(SearchResponse {
        results,
        total,
        query: body.query,
    }))
}

// ---------------------------------------------------------------------------
// Entities
// ---------------------------------------------------------------------------

/// List entities from the knowledge graph.
#[utoipa::path(
    get,
    path = "/api/v1/entities",
    tag = "Entities",
    params(ListEntitiesParams, ("X-Ucotron-Namespace" = Option<String>, Header, description = "Namespace for multi-tenancy isolation")),
    responses(
        (status = 200, description = "List of entities", body = Vec<EntityResponse>),
        (status = 500, description = "Internal server error", body = ApiErrorResponse)
    )
)]
pub async fn list_entities_handler(
    State(state): State<Arc<AppState>>,
    Extension(auth): Extension<AuthContext>,
    headers: HeaderMap,
    Query(params): Query<ListEntitiesParams>,
) -> Result<Json<Vec<EntityResponse>>, AppError> {
    require_role(&auth, ucotron_config::AuthRole::Reader)?;
    let namespace = extract_namespace(&headers);
    require_namespace_access(&auth, &namespace)?;

    // Use vector search to discover entities.
    let query_vec = vec![0.0f32; 384];
    let top_k = params.limit + params.offset;
    let results = state
        .registry
        .vector()
        .search(&query_vec, top_k.min(1000))
        .map_err(|e| AppError::internal(format!("Vector search failed: {}", e)))?;

    let mut entities = Vec::new();
    for (i, (node_id, _score)) in results.iter().enumerate() {
        if i < params.offset {
            continue;
        }
        if entities.len() >= params.limit {
            break;
        }
        if let Ok(Some(node)) = state.registry.graph().get_node(*node_id) {
            if !node_matches_namespace(&node, &namespace) {
                continue;
            }
            if node.node_type == ucotron_core::NodeType::Entity {
                entities.push(node_to_entity_response(&node));
            }
        }
    }

    Ok(Json(entities))
}

/// Get entity with its 1-hop neighbor relations.
#[utoipa::path(
    get,
    path = "/api/v1/entities/{id}",
    tag = "Entities",
    params(
        ("id" = u64, Path, description = "Entity node ID")
    ),
    responses(
        (status = 200, description = "Entity with neighbors", body = EntityResponse),
        (status = 404, description = "Entity not found", body = ApiErrorResponse)
    )
)]
pub async fn get_entity_handler(
    State(state): State<Arc<AppState>>,
    Extension(auth): Extension<AuthContext>,
    Path(id): Path<u64>,
) -> Result<Json<EntityResponse>, AppError> {
    require_role(&auth, ucotron_config::AuthRole::Reader)?;
    let node = state
        .registry
        .graph()
        .get_node(id)
        .map_err(|e| AppError::internal(format!("Graph lookup failed: {}", e)))?
        .ok_or_else(|| AppError::not_found(format!("Entity {} not found", id)))?;

    let neighbors = state
        .registry
        .graph()
        .get_neighbors(id, 1)
        .map_err(|e| AppError::internal(format!("Neighbor lookup failed: {}", e)))?;

    let neighbor_responses: Vec<NeighborResponse> = neighbors
        .iter()
        .map(|n| NeighborResponse {
            node_id: n.id,
            content: n.content.clone(),
            edge_type: "RelatesTo".into(), // Simplified; full edge data requires edge lookup
            weight: 1.0,
        })
        .collect();

    let mut resp = node_to_entity_response(&node);
    resp.neighbors = Some(neighbor_responses);

    Ok(Json(resp))
}

// ---------------------------------------------------------------------------
// Graph Visualization
// ---------------------------------------------------------------------------

/// Get graph data for visualization (nodes + edges).
///
/// Returns up to `limit` nodes and the edges between them, suitable for
/// rendering with a force-directed graph library like D3.js.
#[utoipa::path(
    get,
    path = "/api/v1/graph",
    tag = "Graph",
    params(GraphParams, ("X-Ucotron-Namespace" = Option<String>, Header, description = "Namespace for multi-tenancy isolation")),
    responses(
        (status = 200, description = "Graph nodes and edges", body = GraphResponse),
        (status = 500, description = "Internal server error", body = ApiErrorResponse)
    )
)]
pub async fn graph_handler(
    State(state): State<Arc<AppState>>,
    Extension(auth): Extension<AuthContext>,
    headers: HeaderMap,
    Query(params): Query<GraphParams>,
) -> Result<Json<GraphResponse>, AppError> {
    require_role(&auth, ucotron_config::AuthRole::Reader)?;
    let namespace = extract_namespace(&headers);
    require_namespace_access(&auth, &namespace)?;
    let limit = params.limit.min(500);

    // Discover nodes via vector backend (zero-vector returns arbitrary nodes).
    let query_vec = vec![0.0f32; 384];
    let candidates = state
        .registry
        .vector()
        .search(&query_vec, limit * 2) // Over-fetch to compensate for namespace/type filtering
        .map_err(|e| AppError::internal(format!("Vector search failed: {}", e)))?;

    let node_type_filter = params.node_type.as_deref().and_then(parse_node_type);

    // Collect nodes that pass filters.
    let mut graph_nodes = Vec::new();
    let mut node_id_set = std::collections::HashSet::new();
    for (node_id, _score) in &candidates {
        if graph_nodes.len() >= limit {
            break;
        }
        if let Ok(Some(node)) = state.registry.graph().get_node(*node_id) {
            if !node_matches_namespace(&node, &namespace) {
                continue;
            }
            if let Some(ref nt) = node_type_filter {
                if std::mem::discriminant(&node.node_type) != std::mem::discriminant(nt) {
                    continue;
                }
            }
            // Get community assignment if filtering by community or for display.
            let community_id = state
                .registry
                .graph()
                .get_community(node.id)
                .ok()
                .and_then(|members| if members.is_empty() { None } else { Some(node.id) });

            if let Some(cid) = params.community_id {
                // Filter: only include nodes in the requested community.
                let node_community = state
                    .registry
                    .graph()
                    .get_community(node.id)
                    .unwrap_or_default();
                if !node_community.contains(&cid) && node.id != cid {
                    continue;
                }
            }

            node_id_set.insert(node.id);
            graph_nodes.push(GraphNode {
                id: node.id,
                content: node.content.clone(),
                node_type: format!("{:?}", node.node_type),
                timestamp: node.timestamp,
                community_id,
            });
        }
    }

    // Get all edges and filter to only those between our node set.
    let all_edges = state
        .registry
        .graph()
        .get_all_edges()
        .map_err(|e| AppError::internal(format!("Edge retrieval failed: {}", e)))?;

    let graph_edges: Vec<GraphEdge> = all_edges
        .into_iter()
        .filter(|(src, tgt, _)| node_id_set.contains(src) && node_id_set.contains(tgt))
        .map(|(source, target, weight)| GraphEdge {
            source,
            target,
            weight,
        })
        .collect();

    let total_nodes = graph_nodes.len();
    let total_edges = graph_edges.len();

    Ok(Json(GraphResponse {
        nodes: graph_nodes,
        edges: graph_edges,
        total_nodes,
        total_edges,
    }))
}

// ---------------------------------------------------------------------------
// Augment & Learn
// ---------------------------------------------------------------------------

/// Context augmentation — retrieve relevant memories for a given context.
///
/// Uses the LazyGraphRAG retrieval pipeline: vector search + entity extraction +
/// graph expansion + community selection + re-ranking with temporal decay.
#[utoipa::path(
    post,
    path = "/api/v1/augment",
    tag = "Augment & Learn",
    request_body = AugmentRequest,
    params(
        ("X-Ucotron-Namespace" = Option<String>, Header, description = "Namespace for multi-tenancy isolation"),
        AugmentQueryParams,
    ),
    responses(
        (status = 200, description = "Augmented context with relevant memories", body = AugmentResponse),
        (status = 500, description = "Internal server error", body = ApiErrorResponse)
    )
)]
#[instrument(name = "augment", skip(state, auth, headers, query_params, body), fields(namespace, query_len))]
pub async fn augment_handler(
    State(state): State<Arc<AppState>>,
    Extension(auth): Extension<AuthContext>,
    headers: HeaderMap,
    Query(query_params): Query<AugmentQueryParams>,
    Json(body): Json<AugmentRequest>,
) -> Result<Json<AugmentResponse>, AppError> {
    require_role(&auth, ucotron_config::AuthRole::Reader)?;
    let namespace = extract_namespace(&headers);
    tracing::Span::current().record("namespace", &namespace.as_str());
    tracing::Span::current().record("query_len", body.context.len());
    require_namespace_access(&auth, &namespace)?;
    state.total_searches.fetch_add(1, Ordering::Relaxed);
    record_search(&state);

    let limit = body.limit.unwrap_or(10);

    let retrieval_config = RetrievalConfig {
        final_top_k: limit,
        ..RetrievalConfig::default()
    };

    let ner_ref: Option<&dyn ucotron_extraction::NerPipeline> =
        state.ner.as_ref().map(|n| n.as_ref());

    let mut orchestrator = RetrievalOrchestrator::new(
        &state.registry,
        state.embedder.as_ref(),
        ner_ref,
        retrieval_config,
    );
    if let Some(detector) = build_mindset_detector(&state.config.mindset) {
        orchestrator = orchestrator.with_mindset_detector(detector);
    }

    let result = orchestrator
        .retrieve(&body.context)
        .map_err(|e| AppError::internal(format!("Retrieval failed: {}", e)))?;

    // Filter results by namespace for multi-tenancy isolation.
    let memories: Vec<SearchResultItem> = result
        .memories
        .iter()
        .filter(|m| node_matches_namespace(&m.node, &namespace))
        .map(|m| SearchResultItem {
            id: m.node.id,
            content: m.node.content.clone(),
            node_type: format!("{:?}", m.node.node_type),
            score: m.score,
            vector_sim: m.vector_sim,
            graph_centrality: m.graph_centrality,
            recency: m.recency,
            mindset_score: m.mindset_score,
            timestamp_range: m.node.timestamp_range,
            parent_video_id: m.node.parent_video_id,
        })
        .collect();

    let entities: Vec<EntityResponse> = result
        .entities
        .iter()
        .map(node_to_entity_response)
        .collect();

    let debug_info = if body.debug {
        let m = &result.metrics;
        let score_breakdown: Vec<crate::types::ScoreBreakdown> = result
            .memories
            .iter()
            .filter(|sm| node_matches_namespace(&sm.node, &namespace))
            .map(|sm| crate::types::ScoreBreakdown {
                id: sm.node.id,
                final_score: sm.score,
                vector_sim: sm.vector_sim,
                graph_centrality: sm.graph_centrality,
                recency: sm.recency,
                mindset_score: sm.mindset_score,
                path_reward: sm.path_reward_score,
            })
            .collect();
        Some(crate::types::AugmentDebugInfo {
            pipeline_timings: crate::types::PipelineTimings {
                query_embedding_us: m.query_embedding_us,
                vector_search_us: m.vector_search_us,
                entity_extraction_us: m.entity_extraction_us,
                graph_expansion_us: m.graph_expansion_us,
                community_selection_us: m.community_selection_us,
                reranking_us: m.reranking_us,
                context_assembly_us: m.context_assembly_us,
                total_us: m.total_us,
            },
            pipeline_duration_ms: m.total_us as f64 / 1000.0,
            vector_results_count: m.vector_results_count,
            query_entities_count: m.query_entities_count,
            score_breakdown,
        })
    } else {
        None
    };

    // Rebuild context_text from namespace-filtered memories only.
    // The original result.context_text includes nodes from ALL namespaces,
    // which is a data leak across tenants. We reconstruct it here using
    // only the memories that passed the namespace filter.
    let mut context_text = String::new();
    if !memories.is_empty() {
        context_text.push_str("## Relevant Memories\n");
        for (i, mem) in memories.iter().enumerate() {
            let content = if mem.content.len() > 200 {
                let mut end = 200;
                while !mem.content.is_char_boundary(end) && end > 0 {
                    end -= 1;
                }
                format!("{}...", &mem.content[..end])
            } else {
                mem.content.clone()
            };
            context_text.push_str(&format!(
                "[{}] (score: {:.2}) \"{}\"\n",
                i + 1,
                mem.score,
                content,
            ));
        }
    }
    if !entities.is_empty() {
        context_text.push_str("\n## Known Entities\n");
        for entity in &entities {
            context_text.push_str(&format!("- {} ({})\n", entity.content, entity.node_type));
        }
    }

    tracing::debug!(
        namespace = %namespace,
        total_memories = result.memories.len(),
        filtered_memories = memories.len(),
        "Namespace-filtered augment context_text"
    );

    // Build explainability info when ?explain=true
    let explain_info = if query_params.explain.unwrap_or(false) {
        // Source nodes with relevance scores
        let source_nodes: Vec<crate::types::ExplainSourceNode> = result
            .memories
            .iter()
            .filter(|sm| node_matches_namespace(&sm.node, &namespace))
            .map(|sm| {
                let content = if sm.node.content.len() > 200 {
                    let mut end = 200;
                    while !sm.node.content.is_char_boundary(end) && end > 0 {
                        end -= 1;
                    }
                    format!("{}...", &sm.node.content[..end])
                } else {
                    sm.node.content.clone()
                };
                crate::types::ExplainSourceNode {
                    id: sm.node.id,
                    content,
                    node_type: format!("{:?}", sm.node.node_type),
                    relevance_score: sm.score,
                    embedding_similarity: sm.vector_sim,
                    graph_centrality: sm.graph_centrality,
                    recency: sm.recency,
                }
            })
            .collect();

        // Final ranked IDs (namespace-filtered)
        let final_ranked_ids: Vec<u64> = result
            .memories
            .iter()
            .filter(|sm| node_matches_namespace(&sm.node, &namespace))
            .map(|sm| sm.node.id)
            .collect();

        let retrieval_path = crate::types::RetrievalPath {
            query: body.context.clone(),
            embedding_dimensions: result.embedding_dimensions,
            hnsw_seed_ids: result.hnsw_seed_ids,
            graph_expanded_ids: result.graph_expanded_ids,
            community_ids: result.community_member_ids,
            final_ranked_ids,
        };

        // Context composition: approximate tokens per memory
        let token_counts: Vec<(u64, usize)> = memories
            .iter()
            .map(|m| {
                // Approximate tokens: chars / 4 heuristic
                let approx_tokens = (m.content.len() + 3) / 4;
                (m.id, approx_tokens)
            })
            .collect();
        let total_tokens: usize = token_counts.iter().map(|(_, t)| t).sum();
        let total_tokens = total_tokens.max(1); // avoid division by zero

        let context_composition: Vec<crate::types::ContextContribution> = token_counts
            .into_iter()
            .map(|(id, approx_tokens)| crate::types::ContextContribution {
                id,
                approx_tokens,
                fraction: approx_tokens as f32 / total_tokens as f32,
            })
            .collect();

        Some(crate::types::ExplainabilityInfo {
            source_nodes,
            retrieval_path,
            context_composition,
            total_context_tokens: total_tokens,
        })
    } else {
        None
    };

    Ok(Json(AugmentResponse {
        memories,
        entities,
        context_text,
        debug: debug_info,
        explain: explain_info,
    }))
}

/// Learn from agent output — extract and store memories.
///
/// The output text is processed through the full ingestion pipeline:
/// chunking, embedding, NER, relation extraction, entity resolution,
/// contradiction detection, and graph update.
#[utoipa::path(
    post,
    path = "/api/v1/learn",
    tag = "Augment & Learn",
    request_body = LearnRequest,
    params(
        ("X-Ucotron-Namespace" = Option<String>, Header, description = "Namespace for multi-tenancy isolation")
    ),
    responses(
        (status = 201, description = "Memories extracted and stored", body = LearnResponse),
        (status = 400, description = "Invalid request", body = ApiErrorResponse),
        (status = 403, description = "Read-only instance", body = ApiErrorResponse),
        (status = 500, description = "Internal server error", body = ApiErrorResponse)
    )
)]
#[instrument(name = "learn", skip(state, auth, headers, body), fields(namespace, text_len))]
pub async fn learn_handler(
    State(state): State<Arc<AppState>>,
    Extension(auth): Extension<AuthContext>,
    headers: HeaderMap,
    Json(body): Json<LearnRequest>,
) -> Result<(axum::http::StatusCode, Json<LearnResponse>), AppError> {
    require_role(&auth, ucotron_config::AuthRole::Writer)?;
    let ns = extract_namespace(&headers);
    tracing::Span::current().record("namespace", &ns.as_str());
    tracing::Span::current().record("text_len", body.output.len());
    require_namespace_access(&auth, &ns)?;
    if state.is_reader_only() {
        return Err(AppError::read_only());
    }
    let _namespace = ns;
    state.total_ingestions.fetch_add(1, Ordering::Relaxed);
    record_ingestion(&state);

    if body.output.trim().is_empty() {
        return Err(AppError::bad_request("output must not be empty"));
    }

    let next_id = state.alloc_next_node_id();
    let config = IngestionConfig {
        next_node_id: Some(next_id),
        ..IngestionConfig::default()
    };

    let ner_ref: Option<&dyn ucotron_extraction::NerPipeline> =
        state.ner.as_ref().map(|n| n.as_ref());
    let re_ref: Option<&dyn ucotron_extraction::RelationExtractor> =
        state.relation_extractor.as_ref().map(|r| r.as_ref());

    let mut orchestrator = IngestionOrchestrator::new(
        &state.registry,
        state.embedder.as_ref(),
        ner_ref,
        re_ref,
        config,
    );

    let result = orchestrator
        .ingest(&body.output)
        .map_err(|e| AppError::internal(format!("Learn/ingestion failed: {}", e)))?;

    // Advance shared counter.
    let ids_used = result.chunk_node_ids.len() + result.entity_node_ids.len();
    {
        let mut id_lock = state.next_node_id.lock().unwrap();
        let used_max = next_id + ids_used as u64;
        if used_max > *id_lock {
            *id_lock = used_max;
        }
    }

    // Tag all created nodes with the namespace for multi-tenancy isolation.
    let namespace = extract_namespace(&headers);
    tag_nodes_with_namespace(&state, &result.chunk_node_ids, &namespace);
    tag_nodes_with_namespace(&state, &result.entity_node_ids, &namespace);

    // Tag chunk nodes with conversation_id if provided.
    if let Some(ref conv_id) = body.conversation_id {
        tag_nodes_with_conversation(&state, &result.chunk_node_ids, conv_id);
    }

    Ok((
        axum::http::StatusCode::CREATED,
        Json(LearnResponse {
            memories_created: result.chunk_node_ids.len(),
            entities_found: result.entity_node_ids.len(),
            conflicts_found: result.metrics.contradictions_detected,
        }),
    ))
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Tag nodes with a namespace in their metadata for multi-tenancy isolation.
///
/// After ingestion, each created node is updated to include a `_namespace` metadata
/// field. This allows namespace-based filtering in search/list/augment queries.
fn tag_nodes_with_namespace(state: &AppState, node_ids: &[u64], namespace: &str) {
    for &id in node_ids {
        if let Ok(Some(mut node)) = state.registry.graph().get_node(id) {
            node.metadata
                .insert("_namespace".into(), ucotron_core::Value::String(namespace.to_string()));
            let _ = state.registry.graph().upsert_nodes(&[node]);
        }
    }
}

/// Tag chunk nodes with a conversation ID in their metadata.
fn tag_nodes_with_conversation(state: &AppState, node_ids: &[u64], conversation_id: &str) {
    for &id in node_ids {
        if let Ok(Some(mut node)) = state.registry.graph().get_node(id) {
            node.metadata
                .insert("_conversation_id".into(), ucotron_core::Value::String(conversation_id.to_string()));
            let _ = state.registry.graph().upsert_nodes(&[node]);
        }
    }
}

/// Check if a node belongs to the given namespace.
///
/// Returns `true` if the node has a `_namespace` metadata matching the namespace,
/// or if the node has no `_namespace` metadata (pre-existing data without namespace).
/// The "default" namespace matches all nodes without explicit namespace metadata.
fn node_matches_namespace(node: &ucotron_core::Node, namespace: &str) -> bool {
    match node.metadata.get("_namespace") {
        Some(ucotron_core::Value::String(ns)) => ns == namespace,
        // Nodes without namespace metadata belong to "default" namespace.
        None => namespace == "default",
        _ => namespace == "default",
    }
}

/// Convert a serde_json::Value to a core `Value`.
fn json_to_core_value(v: &serde_json::Value) -> ucotron_core::Value {
    match v {
        serde_json::Value::String(s) => ucotron_core::Value::String(s.clone()),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                ucotron_core::Value::Integer(i)
            } else {
                ucotron_core::Value::Float(n.as_f64().unwrap_or(0.0))
            }
        }
        serde_json::Value::Bool(b) => ucotron_core::Value::Bool(*b),
        _ => ucotron_core::Value::String(v.to_string()),
    }
}

// ---------------------------------------------------------------------------
// Export / Import
// ---------------------------------------------------------------------------

/// Export the memory graph as JSON-LD.
///
/// Returns a complete JSON-LD document containing all nodes and edges
/// in the specified namespace. Supports incremental export via `from_timestamp`.
#[utoipa::path(
    get,
    path = "/api/v1/export",
    tag = "Export & Import",
    params(
        ExportParams,
        ("X-Ucotron-Namespace" = Option<String>, Header, description = "Namespace to export from (overridden by query param if set)")
    ),
    responses(
        (status = 200, description = "JSON-LD export of the memory graph", body = ExportResponse),
        (status = 400, description = "Invalid format parameter", body = ApiErrorResponse),
        (status = 500, description = "Internal error", body = ApiErrorResponse)
    )
)]
pub async fn export_handler(
    State(state): State<Arc<AppState>>,
    Extension(auth): Extension<AuthContext>,
    headers: HeaderMap,
    Query(params): Query<ExportParams>,
) -> Result<Json<ExportResponse>, AppError> {
    require_role(&auth, ucotron_config::AuthRole::Reader)?;
    let namespace = params
        .namespace
        .unwrap_or_else(|| extract_namespace(&headers));
    require_namespace_access(&auth, &namespace)?;

    if params.format != "jsonld" {
        return Err(AppError::bad_request(format!(
            "Unsupported export format: '{}'. Supported: jsonld",
            params.format
        )));
    }

    // Retrieve all nodes from graph backend.
    let all_nodes = state
        .registry
        .graph()
        .get_all_nodes()
        .map_err(|e| AppError::internal(format!("Failed to get nodes: {}", e)))?;

    // Retrieve all edges with full metadata.
    let all_edges = state
        .registry
        .graph()
        .get_all_edges_full()
        .map_err(|e| AppError::internal(format!("Failed to get edges: {}", e)))?;

    // Use core export function.
    let options = ucotron_core::jsonld_export::ExportOptions {
        include_embeddings: params.include_embeddings,
        from_timestamp: params.from_timestamp,
        namespace: namespace.clone(),
    };

    let export = ucotron_core::jsonld_export::export_graph(&all_nodes, &all_edges, &options);

    // Convert to response types.
    let response = ExportResponse {
        context: export.context,
        graph_type: export.graph_type,
        version: export.version,
        exported_at: export.exported_at,
        namespace: export.namespace,
        nodes: export
            .nodes
            .into_iter()
            .map(|n| ExportNodeResponse {
                id: n.id,
                node_type: n.node_type,
                content: n.content,
                timestamp: n.timestamp,
                metadata: n.metadata,
                embedding: n.embedding,
            })
            .collect(),
        edges: export
            .edges
            .into_iter()
            .map(|e| ExportEdgeResponse {
                source: e.source,
                target: e.target,
                edge_type: e.edge_type,
                weight: e.weight,
                metadata: e.metadata,
            })
            .collect(),
        communities: export.communities,
        stats: ExportStatsResponse {
            total_nodes: export.stats.total_nodes,
            total_edges: export.stats.total_edges,
            has_embeddings: export.stats.has_embeddings,
            is_incremental: export.stats.is_incremental,
            from_timestamp: export.stats.from_timestamp,
        },
    };

    Ok(Json(response))
}

/// Import a JSON-LD memory graph export.
///
/// Imports all nodes and edges from the JSON-LD document into the target namespace.
/// Node IDs are remapped to avoid collisions with existing data.
#[utoipa::path(
    post,
    path = "/api/v1/import",
    tag = "Export & Import",
    params(
        ("X-Ucotron-Namespace" = Option<String>, Header, description = "Target namespace for imported data")
    ),
    request_body = ImportRequest,
    responses(
        (status = 200, description = "Import completed", body = ImportResponse),
        (status = 400, description = "Invalid import data", body = ApiErrorResponse),
        (status = 403, description = "Instance is read-only", body = ApiErrorResponse),
        (status = 500, description = "Internal error", body = ApiErrorResponse)
    )
)]
pub async fn import_handler(
    State(state): State<Arc<AppState>>,
    Extension(auth): Extension<AuthContext>,
    headers: HeaderMap,
    Json(body): Json<ImportRequest>,
) -> Result<Json<ImportResponse>, AppError> {
    require_role(&auth, ucotron_config::AuthRole::Writer)?;
    if state.is_reader_only() {
        return Err(AppError::read_only());
    }

    let target_namespace = extract_namespace(&headers);
    require_namespace_access(&auth, &target_namespace)?;

    // Validate version.
    if body.version != "1.0" {
        return Err(AppError::bad_request(format!(
            "Unsupported export version: '{}'. Supported: 1.0",
            body.version
        )));
    }

    // Convert ImportRequest to MemoryGraphExport for the core import function.
    let export = ucotron_core::jsonld_export::MemoryGraphExport {
        context: body.context,
        graph_type: body.graph_type,
        version: body.version,
        exported_at: body.exported_at,
        namespace: body.namespace,
        nodes: body
            .nodes
            .into_iter()
            .map(|n| ucotron_core::jsonld_export::ExportNode {
                id: n.id,
                node_type: n.node_type,
                content: n.content,
                timestamp: n.timestamp,
                metadata: n.metadata,
                embedding: n.embedding,
            })
            .collect(),
        edges: body
            .edges
            .into_iter()
            .map(|e| ucotron_core::jsonld_export::ExportEdge {
                source: e.source,
                target: e.target,
                edge_type: e.edge_type,
                weight: e.weight,
                metadata: e.metadata,
            })
            .collect(),
        communities: body.communities,
        stats: ucotron_core::jsonld_export::ExportStats {
            total_nodes: 0,
            total_edges: 0,
            has_embeddings: false,
            is_incremental: false,
            from_timestamp: None,
        },
    };

    // Allocate starting ID for imported nodes.
    let next_id = state.alloc_next_node_id();

    let result = ucotron_core::jsonld_export::import_graph(&export, next_id, &target_namespace)
        .map_err(|e| AppError::bad_request(format!("Import failed: {}", e)))?;

    // Advance the ID counter past all imported nodes.
    // We already allocated one ID; now skip the rest.
    for _ in 1..result.nodes_imported {
        state.alloc_next_node_id();
    }

    // Insert nodes into graph backend.
    if !result.nodes.is_empty() {
        state
            .registry
            .graph()
            .upsert_nodes(&result.nodes)
            .map_err(|e| AppError::internal(format!("Failed to insert nodes: {}", e)))?;

        // Insert embeddings into vector backend.
        let embeddings: Vec<(u64, Vec<f32>)> = result
            .nodes
            .iter()
            .map(|n| (n.id, n.embedding.clone()))
            .collect();
        state
            .registry
            .vector()
            .upsert_embeddings(&embeddings)
            .map_err(|e| AppError::internal(format!("Failed to insert embeddings: {}", e)))?;
    }

    // Insert edges into graph backend.
    if !result.edges.is_empty() {
        state
            .registry
            .graph()
            .upsert_edges(&result.edges)
            .map_err(|e| AppError::internal(format!("Failed to insert edges: {}", e)))?;
    }

    Ok(Json(ImportResponse {
        nodes_imported: result.nodes_imported,
        edges_imported: result.edges_imported,
        target_namespace,
    }))
}

// ---------------------------------------------------------------------------
// Mem0 Import
// ---------------------------------------------------------------------------

/// Import memories from a Mem0 export.
///
/// Accepts Mem0 JSON data in v1 (bare array), v2 (`results` key), or file
/// export (`memories` key) format. Converts to Ucotron format, assigns fresh
/// node IDs, and stores in the graph and vector backends.
#[utoipa::path(
    post,
    path = "/api/v1/import/mem0",
    tag = "Import",
    request_body = Mem0ImportRequest,
    responses(
        (status = 200, description = "Mem0 memories imported successfully", body = Mem0ImportResponse),
        (status = 400, description = "Invalid Mem0 data", body = ApiErrorResponse),
        (status = 409, description = "Read-only instance", body = ApiErrorResponse),
    ),
    params(
        ("X-Ucotron-Namespace" = Option<String>, Header, description = "Target namespace")
    )
)]
pub async fn mem0_import_handler(
    State(state): State<Arc<AppState>>,
    Extension(auth): Extension<AuthContext>,
    headers: HeaderMap,
    Json(body): Json<Mem0ImportRequest>,
) -> Result<Json<Mem0ImportResponse>, AppError> {
    require_role(&auth, ucotron_config::AuthRole::Writer)?;
    if state.is_reader_only() {
        return Err(AppError::read_only());
    }

    let target_namespace = extract_namespace(&headers);
    require_namespace_access(&auth, &target_namespace)?;

    // Parse Mem0 JSON data.
    let json_str = serde_json::to_string(&body.data)
        .map_err(|e| AppError::bad_request(format!("Invalid JSON data: {}", e)))?;

    let memories = ucotron_core::mem0_adapter::parse_mem0_json(&json_str)
        .map_err(|e| AppError::bad_request(format!("Failed to parse Mem0 data: {}", e)))?;

    if memories.is_empty() {
        return Ok(Json(Mem0ImportResponse {
            memories_parsed: 0,
            nodes_imported: 0,
            edges_imported: 0,
            target_namespace,
        }));
    }

    // Convert to Ucotron format.
    let options = ucotron_core::mem0_adapter::Mem0ImportOptions {
        namespace: target_namespace.clone(),
        link_same_user: body.link_same_user,
        link_same_agent: body.link_same_agent,
    };
    let parse_result = ucotron_core::mem0_adapter::mem0_to_ucotron(&memories, &options);
    let memories_parsed = parse_result.memories_parsed;

    // Use the standard import pipeline.
    let next_id = state.alloc_next_node_id();
    let import_result =
        ucotron_core::jsonld_export::import_graph(&parse_result.export, next_id, &target_namespace)
            .map_err(|e| AppError::bad_request(format!("Import conversion failed: {}", e)))?;

    // Advance the ID counter past all imported nodes.
    for _ in 1..import_result.nodes_imported {
        state.alloc_next_node_id();
    }

    // Insert nodes into graph backend.
    if !import_result.nodes.is_empty() {
        state
            .registry
            .graph()
            .upsert_nodes(&import_result.nodes)
            .map_err(|e| AppError::internal(format!("Failed to insert nodes: {}", e)))?;

        // Insert embeddings into vector backend.
        let embeddings: Vec<(u64, Vec<f32>)> = import_result
            .nodes
            .iter()
            .map(|n| (n.id, n.embedding.clone()))
            .collect();
        state
            .registry
            .vector()
            .upsert_embeddings(&embeddings)
            .map_err(|e| AppError::internal(format!("Failed to insert embeddings: {}", e)))?;
    }

    // Insert edges into graph backend.
    if !import_result.edges.is_empty() {
        state
            .registry
            .graph()
            .upsert_edges(&import_result.edges)
            .map_err(|e| AppError::internal(format!("Failed to insert edges: {}", e)))?;
    }

    Ok(Json(Mem0ImportResponse {
        memories_parsed,
        nodes_imported: import_result.nodes_imported,
        edges_imported: import_result.edges_imported,
        target_namespace,
    }))
}

// ---------------------------------------------------------------------------
// Zep/Graphiti Import
// ---------------------------------------------------------------------------

/// Import memories from a Zep or Graphiti export.
///
/// Accepts Zep/Graphiti JSON data in multiple formats:
/// - Graphiti temporal KG: `{ "entities": [...], "episodes": [...], "edges": [...] }`
/// - Zep sessions: `{ "sessions": [...] }`
/// - Zep facts: `{ "facts": [...] }`
///
/// Temporal metadata (`valid_at`, `invalid_at`) is preserved as node/edge metadata.
#[utoipa::path(
    post,
    path = "/api/v1/import/zep",
    tag = "Import",
    request_body = ZepImportRequest,
    responses(
        (status = 200, description = "Zep/Graphiti data imported successfully", body = ZepImportResponse),
        (status = 400, description = "Invalid Zep/Graphiti data", body = ApiErrorResponse),
        (status = 409, description = "Read-only instance", body = ApiErrorResponse),
    ),
    params(
        ("X-Ucotron-Namespace" = Option<String>, Header, description = "Target namespace")
    )
)]
pub async fn zep_import_handler(
    State(state): State<Arc<AppState>>,
    Extension(auth): Extension<AuthContext>,
    headers: HeaderMap,
    Json(body): Json<ZepImportRequest>,
) -> Result<Json<ZepImportResponse>, AppError> {
    require_role(&auth, ucotron_config::AuthRole::Writer)?;
    if state.is_reader_only() {
        return Err(AppError::read_only());
    }

    let target_namespace = extract_namespace(&headers);
    require_namespace_access(&auth, &target_namespace)?;

    // Parse Zep/Graphiti JSON data.
    let json_str = serde_json::to_string(&body.data)
        .map_err(|e| AppError::bad_request(format!("Invalid JSON data: {}", e)))?;

    let zep_data = ucotron_core::zep_adapter::parse_zep_json(&json_str)
        .map_err(|e| AppError::bad_request(format!("Failed to parse Zep/Graphiti data: {}", e)))?;

    // Convert to Ucotron format.
    let options = ucotron_core::zep_adapter::ZepImportOptions {
        namespace: target_namespace.clone(),
        link_same_user: body.link_same_user,
        link_same_group: body.link_same_group,
        preserve_expired: body.preserve_expired,
    };
    let parse_result = ucotron_core::zep_adapter::zep_to_ucotron(&zep_data, &options);
    let memories_parsed = parse_result.memories_parsed;

    if memories_parsed == 0 {
        return Ok(Json(ZepImportResponse {
            memories_parsed: 0,
            nodes_imported: 0,
            edges_imported: 0,
            target_namespace,
        }));
    }

    // Use the standard import pipeline.
    let next_id = state.alloc_next_node_id();
    let import_result =
        ucotron_core::jsonld_export::import_graph(&parse_result.export, next_id, &target_namespace)
            .map_err(|e| AppError::bad_request(format!("Import conversion failed: {}", e)))?;

    // Advance the ID counter past all imported nodes.
    for _ in 1..import_result.nodes_imported {
        state.alloc_next_node_id();
    }

    // Insert nodes into graph backend.
    if !import_result.nodes.is_empty() {
        state
            .registry
            .graph()
            .upsert_nodes(&import_result.nodes)
            .map_err(|e| AppError::internal(format!("Failed to insert nodes: {}", e)))?;

        // Insert embeddings into vector backend.
        let embeddings: Vec<(u64, Vec<f32>)> = import_result
            .nodes
            .iter()
            .map(|n| (n.id, n.embedding.clone()))
            .collect();
        state
            .registry
            .vector()
            .upsert_embeddings(&embeddings)
            .map_err(|e| AppError::internal(format!("Failed to insert embeddings: {}", e)))?;
    }

    // Insert edges into graph backend.
    if !import_result.edges.is_empty() {
        state
            .registry
            .graph()
            .upsert_edges(&import_result.edges)
            .map_err(|e| AppError::internal(format!("Failed to insert edges: {}", e)))?;
    }

    Ok(Json(ZepImportResponse {
        memories_parsed,
        nodes_imported: import_result.nodes_imported,
        edges_imported: import_result.edges_imported,
        target_namespace,
    }))
}

// ---------------------------------------------------------------------------
// Audio Transcription
// ---------------------------------------------------------------------------

/// Transcribe audio from an uploaded WAV file and optionally ingest as memory.
///
/// Accepts multipart/form-data with a `file` field containing a WAV file.
/// An optional `ingest` field (default: true) controls whether the transcribed
/// text is automatically ingested into the memory graph.
#[utoipa::path(
    post,
    path = "/api/v1/transcribe",
    tag = "Audio",
    request_body(content_type = "multipart/form-data", description = "WAV audio file"),
    params(
        ("X-Ucotron-Namespace" = Option<String>, Header, description = "Namespace for multi-tenancy isolation")
    ),
    responses(
        (status = 200, description = "Audio transcribed successfully", body = TranscribeResponse),
        (status = 400, description = "Invalid request (no file or unsupported format)", body = ApiErrorResponse),
        (status = 403, description = "Read-only instance", body = ApiErrorResponse),
        (status = 500, description = "Transcription failed", body = ApiErrorResponse),
        (status = 501, description = "Transcription not available", body = ApiErrorResponse)
    )
)]
pub async fn transcribe_handler(
    State(state): State<Arc<AppState>>,
    Extension(auth): Extension<AuthContext>,
    headers: HeaderMap,
    mut multipart: Multipart,
) -> Result<Json<TranscribeResponse>, AppError> {
    require_role(&auth, ucotron_config::AuthRole::Writer)?;
    if state.is_reader_only() {
        return Err(AppError::read_only());
    }

    let transcriber = state
        .transcriber
        .as_ref()
        .ok_or_else(|| AppError::not_implemented("Audio transcription not available — Whisper model not loaded"))?;

    // Extract the audio file from multipart form data
    let mut audio_data: Option<Vec<u8>> = None;
    let mut should_ingest = true;

    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|e| AppError::bad_request(format!("Failed to read multipart field: {}", e)))?
    {
        let name = field.name().unwrap_or("").to_string();
        match name.as_str() {
            "file" | "audio" => {
                let data = field
                    .bytes()
                    .await
                    .map_err(|e| AppError::bad_request(format!("Failed to read file data: {}", e)))?;
                audio_data = Some(data.to_vec());
            }
            "ingest" => {
                let text = field
                    .text()
                    .await
                    .map_err(|e| AppError::bad_request(format!("Failed to read ingest field: {}", e)))?;
                should_ingest = text != "false" && text != "0";
            }
            _ => {}
        }
    }

    let audio_bytes = audio_data.ok_or_else(|| {
        AppError::bad_request("Missing 'file' or 'audio' field in multipart form data")
    })?;

    if audio_bytes.is_empty() {
        return Err(AppError::bad_request("Audio file is empty"));
    }

    // Write to temp file for WAV parsing (hound requires seekable reader)
    let temp_dir = tempfile::tempdir()
        .map_err(|e| AppError::internal(format!("Failed to create temp directory: {}", e)))?;
    let temp_path = temp_dir.path().join("upload.wav");
    std::fs::write(&temp_path, &audio_bytes)
        .map_err(|e| AppError::internal(format!("Failed to write temp file: {}", e)))?;

    // Transcribe
    let result = transcriber
        .transcribe_file(&temp_path)
        .map_err(|e| AppError::internal(format!("Transcription failed: {}", e)))?;

    // Build response
    let chunks: Vec<TranscribeChunk> = result
        .chunks
        .iter()
        .map(|c| TranscribeChunk {
            text: c.text.clone(),
            start_secs: c.start_secs,
            end_secs: c.end_secs,
        })
        .collect();

    let audio_meta = AudioMetadataResponse {
        duration_secs: result.metadata.duration_secs,
        sample_rate: result.metadata.sample_rate,
        channels: result.metadata.channels,
        detected_language: result.metadata.detected_language.clone(),
    };

    // Optionally ingest the transcribed text
    let ingestion = if should_ingest && !result.text.is_empty() {
        state.total_ingestions.fetch_add(1, Ordering::Relaxed);
        record_ingestion(&state);
        let next_id = state.alloc_next_node_id();
        let config = IngestionConfig {
            next_node_id: Some(next_id),
            ..IngestionConfig::default()
        };

        let ner_ref: Option<&dyn ucotron_extraction::NerPipeline> =
            state.ner.as_ref().map(|n| n.as_ref());
        let re_ref: Option<&dyn ucotron_extraction::RelationExtractor> =
            state.relation_extractor.as_ref().map(|r| r.as_ref());

        let mut orchestrator = IngestionOrchestrator::new(
            &state.registry,
            state.embedder.as_ref(),
            ner_ref,
            re_ref,
            config,
        );

        match orchestrator.ingest(&result.text) {
            Ok(ingest_result) => {
                // Advance shared ID counter
                let ids_used =
                    ingest_result.chunk_node_ids.len() + ingest_result.entity_node_ids.len();
                {
                    let mut id_lock = state.next_node_id.lock().unwrap();
                    let used_max = next_id + ids_used as u64;
                    if used_max > *id_lock {
                        *id_lock = used_max;
                    }
                }

                // Tag nodes with namespace
                let namespace = extract_namespace(&headers);
                tag_nodes_with_namespace(&state, &ingest_result.chunk_node_ids, &namespace);
                tag_nodes_with_namespace(&state, &ingest_result.entity_node_ids, &namespace);

                Some(CreateMemoryResponse {
                    chunk_node_ids: ingest_result.chunk_node_ids,
                    entity_node_ids: ingest_result.entity_node_ids,
                    edges_created: ingest_result.edges_created.len(),
                    metrics: IngestionMetricsResponse {
                        chunks_processed: ingest_result.metrics.chunks_processed,
                        entities_extracted: ingest_result.metrics.entities_extracted,
                        relations_extracted: ingest_result.metrics.relations_extracted,
                        contradictions_detected: ingest_result.metrics.contradictions_detected,
                        total_us: ingest_result.metrics.total_us,
                    },
                })
            }
            Err(e) => {
                tracing::warn!("Ingestion of transcribed text failed: {}", e);
                None
            }
        }
    } else {
        None
    };

    Ok(Json(TranscribeResponse {
        text: result.text,
        chunks,
        audio: audio_meta,
        ingestion,
    }))
}

// ---------------------------------------------------------------------------
// Image Embedding (CLIP)
// ---------------------------------------------------------------------------

/// Index an image by generating a CLIP embedding and storing it in the vector backend.
///
/// Accepts multipart/form-data with:
/// - `file` or `image`: the image file (JPEG, PNG, etc.)
/// - `description` (optional): text description to store as the node's content
///
/// The image embedding is stored in the vector backend for cross-modal search.
#[utoipa::path(
    post,
    path = "/api/v1/images",
    tag = "Images",
    request_body(content_type = "multipart/form-data", description = "Image file (JPEG, PNG, etc.)"),
    params(
        ("X-Ucotron-Namespace" = Option<String>, Header, description = "Namespace for multi-tenancy isolation")
    ),
    responses(
        (status = 200, description = "Image indexed successfully", body = ImageIndexResponse),
        (status = 400, description = "Invalid request (no file or unsupported format)", body = ApiErrorResponse),
        (status = 403, description = "Read-only instance", body = ApiErrorResponse),
        (status = 500, description = "Image processing failed", body = ApiErrorResponse),
        (status = 501, description = "Image embedding not available", body = ApiErrorResponse)
    )
)]
pub async fn index_image_handler(
    State(state): State<Arc<AppState>>,
    Extension(auth): Extension<AuthContext>,
    headers: HeaderMap,
    mut multipart: Multipart,
) -> Result<Json<ImageIndexResponse>, AppError> {
    require_role(&auth, ucotron_config::AuthRole::Writer)?;
    if state.is_reader_only() {
        return Err(AppError::read_only());
    }

    let image_embedder = state
        .image_embedder
        .as_ref()
        .ok_or_else(|| {
            AppError::not_implemented("Image embedding not available — CLIP model not loaded")
        })?;

    // Extract multipart fields
    let mut image_data: Option<Vec<u8>> = None;
    let mut description = String::new();

    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|e| AppError::bad_request(format!("Failed to read multipart field: {}", e)))?
    {
        let name = field.name().unwrap_or("").to_string();
        match name.as_str() {
            "file" | "image" => {
                let data = field
                    .bytes()
                    .await
                    .map_err(|e| {
                        AppError::bad_request(format!("Failed to read image data: {}", e))
                    })?;
                image_data = Some(data.to_vec());
            }
            "description" => {
                description = field
                    .text()
                    .await
                    .map_err(|e| {
                        AppError::bad_request(format!("Failed to read description: {}", e))
                    })?;
            }
            _ => {}
        }
    }

    let image_bytes = image_data.ok_or_else(|| {
        AppError::bad_request("Missing 'file' or 'image' field in multipart form data")
    })?;

    if image_bytes.is_empty() {
        return Err(AppError::bad_request("Image file is empty"));
    }

    // Detect image format and dimensions
    let img = image::load_from_memory(&image_bytes)
        .map_err(|e| AppError::bad_request(format!("Failed to decode image: {}", e)))?;
    let (width, height) = (img.width(), img.height());
    let format = image::guess_format(&image_bytes)
        .map(|f| format!("{:?}", f).to_lowercase())
        .unwrap_or_else(|_| "unknown".to_string());

    // Generate CLIP embedding
    let embedding = image_embedder
        .embed_image_bytes(&image_bytes)
        .map_err(|e| AppError::internal(format!("Image embedding failed: {}", e)))?;

    let embed_dim = embedding.len();

    // Create a node for this image
    let node_id = state.alloc_next_node_id();
    let namespace = extract_namespace(&headers);
    let content = if description.is_empty() {
        format!("[image:{}x{} {}]", width, height, format)
    } else {
        description
    };

    let mut metadata = std::collections::HashMap::new();
    metadata.insert(
        "_namespace".into(),
        ucotron_core::Value::String(namespace),
    );
    metadata.insert(
        "_media_type".into(),
        ucotron_core::Value::String("image".to_string()),
    );
    metadata.insert(
        "_image_format".into(),
        ucotron_core::Value::String(format.clone()),
    );
    metadata.insert(
        "_image_width".into(),
        ucotron_core::Value::Integer(width as i64),
    );
    metadata.insert(
        "_image_height".into(),
        ucotron_core::Value::Integer(height as i64),
    );

    let node = ucotron_core::Node {
        id: node_id,
        content,
        embedding: embedding.clone(),
        metadata,
        node_type: ucotron_core::NodeType::Entity,
        timestamp: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs(),
        media_type: None,
        media_uri: None,
        embedding_visual: None,
        timestamp_range: None,
        parent_video_id: None,
    };

    // Store node in graph backend
    state
        .registry
        .graph()
        .upsert_nodes(&[node])
        .map_err(|e| AppError::internal(format!("Failed to store image node: {}", e)))?;

    // Store CLIP embedding in the visual index (512-dim CLIP space).
    // This is the correct index for image embeddings — the text index uses
    // 384-dim MiniLM vectors for text nodes, while the visual index uses
    // 512-dim CLIP vectors for image nodes, enabling cross-modal text-to-image search.
    if let Some(visual_backend) = state.registry.visual() {
        visual_backend
            .upsert_visual_embeddings(&[(node_id, embedding.clone())])
            .map_err(|e| {
                AppError::internal(format!("Failed to store image embedding in visual index: {}", e))
            })?;
    } else {
        // Fallback: store in text vector backend if no visual backend available.
        // This is a degraded mode — text-to-image search will not work correctly.
        state
            .registry
            .vector()
            .upsert_embeddings(&[(node_id, embedding)])
            .map_err(|e| {
                AppError::internal(format!("Failed to store image embedding: {}", e))
            })?;
    }

    state.total_ingestions.fetch_add(1, Ordering::Relaxed);
    record_ingestion(&state);

    Ok(Json(ImageIndexResponse {
        node_id,
        width,
        height,
        format,
        embedding_dim: embed_dim,
    }))
}

/// Cross-modal search: find images similar to a text query using CLIP.
///
/// The text query is encoded using CLIP's text encoder and compared against
/// stored image embeddings using cosine similarity.
#[utoipa::path(
    post,
    path = "/api/v1/images/search",
    tag = "Images",
    request_body = ImageSearchRequest,
    params(
        ("X-Ucotron-Namespace" = Option<String>, Header, description = "Namespace for multi-tenancy isolation")
    ),
    responses(
        (status = 200, description = "Cross-modal search results", body = ImageSearchResponse),
        (status = 400, description = "Invalid request", body = ApiErrorResponse),
        (status = 501, description = "Cross-modal search not available", body = ApiErrorResponse)
    )
)]
pub async fn image_search_handler(
    State(state): State<Arc<AppState>>,
    Extension(auth): Extension<AuthContext>,
    headers: HeaderMap,
    Json(req): Json<ImageSearchRequest>,
) -> Result<Json<ImageSearchResponse>, AppError> {
    require_role(&auth, ucotron_config::AuthRole::Reader)?;
    let cross_modal = state
        .cross_modal_encoder
        .as_ref()
        .ok_or_else(|| {
            AppError::not_implemented(
                "Cross-modal search not available — CLIP text model not loaded",
            )
        })?;

    let namespace = extract_namespace(&headers);
    let limit = req.limit.unwrap_or(10);

    let min_sim = req.min_similarity.unwrap_or(0.0);

    // Encode text query into CLIP's 512-dim image embedding space
    let query_embedding = cross_modal
        .embed_text(&req.query)
        .map_err(|e| AppError::internal(format!("Text encoding failed: {}", e)))?;

    // Search the visual (CLIP 512-dim) index — NOT the text (MiniLM 384-dim) index.
    // The visual backend stores image embeddings in the same space as the CLIP text encoder.
    let visual_backend = state.registry.visual().ok_or_else(|| {
        AppError::not_implemented(
            "Visual vector backend not available — required for text-to-image search",
        )
    })?;

    // Use a larger top_k to account for namespace filtering and min_similarity
    let search_limit = limit * 3;
    let raw_results = visual_backend
        .search_visual(&query_embedding, search_limit)
        .map_err(|e| AppError::internal(format!("Visual index search failed: {}", e)))?;

    // Filter by min_similarity threshold, media type (image), and namespace
    let mut results = Vec::new();
    for (node_id, score) in raw_results {
        if results.len() >= limit {
            break;
        }
        // Skip results below the minimum similarity threshold
        if score < min_sim {
            continue;
        }
        if let Ok(Some(node)) = state.registry.graph().get_node(node_id) {
            // Only include image nodes
            let is_image = node
                .metadata
                .get("_media_type")
                .map(|v| matches!(v, ucotron_core::Value::String(s) if s == "image"))
                .unwrap_or(false);
            if is_image && node_matches_namespace(&node, &namespace) {
                results.push(ImageSearchResultItem {
                    node_id,
                    score,
                    content: node.content.clone(),
                    timestamp: node.timestamp,
                });
            }
        }
    }

    let total = results.len();
    state.total_searches.fetch_add(1, Ordering::Relaxed);
    record_search(&state);

    Ok(Json(ImageSearchResponse {
        results,
        total,
        query: req.query,
    }))
}

// ---------------------------------------------------------------------------
// Document OCR
// ---------------------------------------------------------------------------

/// Extract text from a PDF or document image and optionally ingest as memory.
///
/// Accepts multipart/form-data with:
/// - `file` or `document`: the document file (PDF, JPEG, PNG, TIFF, BMP)
/// - `ingest` (optional): boolean string (default "true"), whether to ingest extracted text
///
/// For PDFs, text is extracted directly using pure Rust PDF parsing.
/// For images, Tesseract OCR is used if available on the system.
#[utoipa::path(
    post,
    path = "/api/v1/ocr",
    tag = "Documents",
    request_body(content_type = "multipart/form-data", description = "Document file (PDF, JPEG, PNG, TIFF, BMP)"),
    params(
        ("X-Ucotron-Namespace" = Option<String>, Header, description = "Namespace for multi-tenancy isolation")
    ),
    responses(
        (status = 200, description = "Document processed successfully", body = OcrResponse),
        (status = 400, description = "Invalid request (no file or unsupported format)", body = ApiErrorResponse),
        (status = 403, description = "Read-only instance", body = ApiErrorResponse),
        (status = 500, description = "Document processing failed", body = ApiErrorResponse),
        (status = 501, description = "Document OCR not available", body = ApiErrorResponse)
    )
)]
pub async fn ocr_handler(
    State(state): State<Arc<AppState>>,
    Extension(auth): Extension<AuthContext>,
    headers: HeaderMap,
    mut multipart: Multipart,
) -> Result<Json<OcrResponse>, AppError> {
    require_role(&auth, ucotron_config::AuthRole::Writer)?;
    if state.is_reader_only() {
        return Err(AppError::read_only());
    }

    let ocr_pipeline = state
        .ocr_pipeline
        .as_ref()
        .ok_or_else(|| {
            AppError::not_implemented("Document OCR not available — pipeline not loaded")
        })?;

    // Extract the document file from multipart form data
    let mut doc_data: Option<Vec<u8>> = None;
    let mut filename: Option<String> = None;
    let mut should_ingest = true;

    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|e| AppError::bad_request(format!("Failed to read multipart field: {}", e)))?
    {
        let name = field.name().unwrap_or("").to_string();
        match name.as_str() {
            "file" | "document" => {
                // Capture filename from the field
                let field_filename = field
                    .file_name()
                    .unwrap_or("document.pdf")
                    .to_string();
                let data = field
                    .bytes()
                    .await
                    .map_err(|e| {
                        AppError::bad_request(format!("Failed to read file data: {}", e))
                    })?;
                filename = Some(field_filename);
                doc_data = Some(data.to_vec());
            }
            "ingest" => {
                let text = field
                    .text()
                    .await
                    .map_err(|e| {
                        AppError::bad_request(format!("Failed to read ingest field: {}", e))
                    })?;
                should_ingest = text != "false" && text != "0";
            }
            _ => {}
        }
    }

    let doc_bytes = doc_data.ok_or_else(|| {
        AppError::bad_request("Missing 'file' or 'document' field in multipart form data")
    })?;

    if doc_bytes.is_empty() {
        return Err(AppError::bad_request("Document file is empty"));
    }

    let fname = filename.unwrap_or_else(|| "document.pdf".to_string());

    // Process the document
    let result = ocr_pipeline
        .process_document(&doc_bytes, &fname)
        .map_err(|e| AppError::internal(format!("Document processing failed: {}", e)))?;

    // Build page responses
    let pages: Vec<OcrPageResponse> = result
        .pages
        .iter()
        .map(|p| OcrPageResponse {
            page_number: p.page_number,
            text: p.text.clone(),
        })
        .collect();

    let doc_meta = OcrDocumentMetadata {
        total_pages: result.metadata.total_pages,
        format: result.metadata.format.to_string(),
        is_scanned: result.metadata.is_scanned,
    };

    // Optionally ingest the extracted text
    let ingestion = if should_ingest && !result.text.is_empty() {
        state.total_ingestions.fetch_add(1, Ordering::Relaxed);
        record_ingestion(&state);
        let next_id = state.alloc_next_node_id();
        let config = IngestionConfig {
            next_node_id: Some(next_id),
            ..IngestionConfig::default()
        };

        let ner_ref: Option<&dyn ucotron_extraction::NerPipeline> =
            state.ner.as_ref().map(|n| n.as_ref());
        let re_ref: Option<&dyn ucotron_extraction::RelationExtractor> =
            state.relation_extractor.as_ref().map(|r| r.as_ref());

        let mut orchestrator = IngestionOrchestrator::new(
            &state.registry,
            state.embedder.as_ref(),
            ner_ref,
            re_ref,
            config,
        );

        match orchestrator.ingest(&result.text) {
            Ok(ingest_result) => {
                // Advance shared ID counter
                let ids_used =
                    ingest_result.chunk_node_ids.len() + ingest_result.entity_node_ids.len();
                {
                    let mut id_lock = state.next_node_id.lock().unwrap();
                    let used_max = next_id + ids_used as u64;
                    if used_max > *id_lock {
                        *id_lock = used_max;
                    }
                }

                // Tag nodes with namespace and media type
                let namespace = extract_namespace(&headers);
                tag_nodes_with_namespace(&state, &ingest_result.chunk_node_ids, &namespace);
                tag_nodes_with_namespace(&state, &ingest_result.entity_node_ids, &namespace);

                // Tag chunk nodes with document metadata
                for &nid in &ingest_result.chunk_node_ids {
                    if let Ok(Some(mut node)) = state.registry.graph().get_node(nid) {
                        node.metadata.insert(
                            "_media_type".to_string(),
                            ucotron_core::Value::String("document".to_string()),
                        );
                        node.metadata.insert(
                            "_document_format".to_string(),
                            ucotron_core::Value::String(
                                result.metadata.format.to_string(),
                            ),
                        );
                        let _ = state.registry.graph().upsert_nodes(&[node]);
                    }
                }

                Some(CreateMemoryResponse {
                    chunk_node_ids: ingest_result.chunk_node_ids,
                    entity_node_ids: ingest_result.entity_node_ids,
                    edges_created: ingest_result.edges_created.len(),
                    metrics: IngestionMetricsResponse {
                        chunks_processed: ingest_result.metrics.chunks_processed,
                        entities_extracted: ingest_result.metrics.entities_extracted,
                        relations_extracted: ingest_result.metrics.relations_extracted,
                        contradictions_detected: ingest_result.metrics.contradictions_detected,
                        total_us: ingest_result.metrics.total_us,
                    },
                })
            }
            Err(e) => {
                tracing::warn!("Ingestion of extracted document text failed: {}", e);
                None
            }
        }
    } else {
        None
    };

    Ok(Json(OcrResponse {
        text: result.text,
        pages,
        document: doc_meta,
        ingestion,
    }))
}

// ---------------------------------------------------------------------------
// Admin: Namespace Management
// ---------------------------------------------------------------------------

/// List all discovered namespaces with per-namespace statistics.
///
/// Scans all nodes in the graph to discover which namespaces exist and
/// computes per-namespace counts and last activity timestamp.
#[utoipa::path(
    get,
    path = "/api/v1/admin/namespaces",
    tag = "Admin",
    responses(
        (status = 200, description = "List of namespaces", body = NamespaceListResponse),
        (status = 500, description = "Internal error", body = ApiErrorResponse)
    )
)]
pub async fn list_namespaces_handler(
    State(state): State<Arc<AppState>>,
    Extension(auth): Extension<AuthContext>,
) -> Result<Json<NamespaceListResponse>, AppError> {
    require_role(&auth, ucotron_config::AuthRole::Admin)?;
    let all_nodes = state
        .registry
        .graph()
        .get_all_nodes()
        .map_err(|e| AppError::internal(format!("Failed to get nodes: {}", e)))?;

    // Aggregate per-namespace stats.
    let mut ns_map: std::collections::HashMap<String, NamespaceInfo> = std::collections::HashMap::new();

    for node in &all_nodes {
        let ns_name = match node.metadata.get("_namespace") {
            Some(ucotron_core::Value::String(s)) => s.clone(),
            _ => "default".to_string(),
        };

        let entry = ns_map.entry(ns_name.clone()).or_insert_with(|| NamespaceInfo {
            name: ns_name,
            memory_count: 0,
            entity_count: 0,
            total_nodes: 0,
            last_activity: 0,
        });

        entry.total_nodes += 1;
        if node.timestamp > entry.last_activity {
            entry.last_activity = node.timestamp;
        }

        match node.node_type {
            ucotron_core::NodeType::Entity => entry.entity_count += 1,
            _ => entry.memory_count += 1,
        }
    }

    // Ensure "default" namespace always appears.
    ns_map.entry("default".to_string()).or_insert_with(|| NamespaceInfo {
        name: "default".to_string(),
        memory_count: 0,
        entity_count: 0,
        total_nodes: 0,
        last_activity: 0,
    });

    let mut namespaces: Vec<NamespaceInfo> = ns_map.into_values().collect();
    namespaces.sort_by(|a, b| a.name.cmp(&b.name));
    let total = namespaces.len();

    Ok(Json(NamespaceListResponse { namespaces, total }))
}

/// Create a new namespace.
///
/// Namespaces are implicit in Ucotron (defined by node metadata). This endpoint
/// validates the name against the allowed list and max limit, and creates
/// a sentinel node to register the namespace.
#[utoipa::path(
    post,
    path = "/api/v1/admin/namespaces",
    tag = "Admin",
    request_body = CreateNamespaceRequest,
    responses(
        (status = 201, description = "Namespace created", body = CreateNamespaceResponse),
        (status = 400, description = "Invalid namespace name", body = ApiErrorResponse),
        (status = 403, description = "Read-only instance", body = ApiErrorResponse),
        (status = 409, description = "Namespace already exists", body = ApiErrorResponse)
    )
)]
pub async fn create_namespace_handler(
    State(state): State<Arc<AppState>>,
    Extension(auth): Extension<AuthContext>,
    Json(body): Json<CreateNamespaceRequest>,
) -> Result<(axum::http::StatusCode, Json<CreateNamespaceResponse>), AppError> {
    require_role(&auth, ucotron_config::AuthRole::Admin)?;
    if state.is_reader_only() {
        return Err(AppError::read_only());
    }

    let name = body.name.trim().to_string();
    if name.is_empty() {
        return Err(AppError::bad_request("Namespace name cannot be empty"));
    }
    if name.len() > 64 {
        return Err(AppError::bad_request("Namespace name must be 64 characters or fewer"));
    }
    // Only allow alphanumeric, hyphens, and underscores.
    if !name.chars().all(|c| c.is_alphanumeric() || c == '-' || c == '_') {
        return Err(AppError::bad_request(
            "Namespace name must contain only alphanumeric characters, hyphens, and underscores",
        ));
    }

    // Check allowed namespaces config.
    let ns_config = &state.config.namespaces;
    if !ns_config.allowed_namespaces.is_empty()
        && !ns_config.allowed_namespaces.contains(&name)
    {
        return Err(AppError::bad_request(format!(
            "Namespace '{}' is not in the allowed list",
            name
        )));
    }

    // Check if namespace already exists by scanning nodes.
    let all_nodes = state
        .registry
        .graph()
        .get_all_nodes()
        .map_err(|e| AppError::internal(format!("Failed to get nodes: {}", e)))?;

    let existing_namespaces: std::collections::HashSet<String> = all_nodes
        .iter()
        .filter_map(|n| match n.metadata.get("_namespace") {
            Some(ucotron_core::Value::String(s)) => Some(s.clone()),
            _ => None,
        })
        .collect();

    if existing_namespaces.contains(&name) {
        return Err(AppError {
            status: axum::http::StatusCode::CONFLICT,
            code: "CONFLICT".into(),
            message: format!("Namespace '{}' already exists", name),
        });
    }

    // Check max namespaces.
    if ns_config.max_namespaces > 0 && existing_namespaces.len() >= ns_config.max_namespaces {
        return Err(AppError::bad_request(format!(
            "Maximum number of namespaces ({}) reached",
            ns_config.max_namespaces
        )));
    }

    // Create a sentinel node to register the namespace.
    let node_id = state.alloc_next_node_id();
    let mut metadata = std::collections::HashMap::new();
    metadata.insert(
        "_namespace".into(),
        ucotron_core::Value::String(name.clone()),
    );
    metadata.insert(
        "_sentinel".into(),
        ucotron_core::Value::Bool(true),
    );

    let sentinel = ucotron_core::Node {
        id: node_id,
        content: format!("[namespace:{}]", name),
        embedding: vec![0.0f32; 384],
        metadata,
        node_type: ucotron_core::NodeType::Entity,
        timestamp: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs(),
        media_type: None,
        media_uri: None,
        embedding_visual: None,
        timestamp_range: None,
        parent_video_id: None,
    };

    state
        .registry
        .graph()
        .upsert_nodes(&[sentinel])
        .map_err(|e| AppError::internal(format!("Failed to create sentinel: {}", e)))?;

    Ok((
        axum::http::StatusCode::CREATED,
        Json(CreateNamespaceResponse {
            name,
            created: true,
        }),
    ))
}

/// Delete a namespace and all its nodes.
///
/// Removes all nodes tagged with the given namespace from the graph and vector backends.
/// The "default" namespace cannot be deleted.
#[utoipa::path(
    delete,
    path = "/api/v1/admin/namespaces/{name}",
    tag = "Admin",
    params(
        ("name" = String, Path, description = "Namespace name to delete")
    ),
    responses(
        (status = 200, description = "Namespace deleted", body = DeleteNamespaceResponse),
        (status = 400, description = "Cannot delete default namespace", body = ApiErrorResponse),
        (status = 403, description = "Read-only instance", body = ApiErrorResponse),
        (status = 500, description = "Internal error", body = ApiErrorResponse)
    )
)]
pub async fn delete_namespace_handler(
    State(state): State<Arc<AppState>>,
    Extension(auth): Extension<AuthContext>,
    Path(name): Path<String>,
) -> Result<Json<DeleteNamespaceResponse>, AppError> {
    require_role(&auth, ucotron_config::AuthRole::Admin)?;
    if state.is_reader_only() {
        return Err(AppError::read_only());
    }

    if name == "default" {
        return Err(AppError::bad_request("Cannot delete the default namespace"));
    }

    let all_nodes = state
        .registry
        .graph()
        .get_all_nodes()
        .map_err(|e| AppError::internal(format!("Failed to get nodes: {}", e)))?;

    let node_ids_to_delete: Vec<u64> = all_nodes
        .iter()
        .filter(|n| {
            matches!(
                n.metadata.get("_namespace"),
                Some(ucotron_core::Value::String(s)) if s == &name
            )
        })
        .map(|n| n.id)
        .collect();

    let count = node_ids_to_delete.len();

    // Cascade delete: vector embeddings, then graph nodes (which also removes edges).
    if !node_ids_to_delete.is_empty() {
        state
            .registry
            .vector()
            .delete(&node_ids_to_delete)
            .map_err(|e| AppError::internal(format!("Failed to delete vectors: {}", e)))?;

        state
            .registry
            .graph()
            .delete_nodes(&node_ids_to_delete)
            .map_err(|e| AppError::internal(format!("Failed to delete graph nodes: {}", e)))?;
    }

    Ok(Json(DeleteNamespaceResponse {
        name,
        nodes_deleted: count,
    }))
}

/// Get namespace statistics for a specific namespace.
#[utoipa::path(
    get,
    path = "/api/v1/admin/namespaces/{name}",
    tag = "Admin",
    params(
        ("name" = String, Path, description = "Namespace name")
    ),
    responses(
        (status = 200, description = "Namespace info", body = NamespaceInfo),
        (status = 404, description = "Namespace not found", body = ApiErrorResponse),
        (status = 500, description = "Internal error", body = ApiErrorResponse)
    )
)]
pub async fn get_namespace_handler(
    State(state): State<Arc<AppState>>,
    Extension(auth): Extension<AuthContext>,
    Path(name): Path<String>,
) -> Result<Json<NamespaceInfo>, AppError> {
    require_role(&auth, ucotron_config::AuthRole::Admin)?;
    let all_nodes = state
        .registry
        .graph()
        .get_all_nodes()
        .map_err(|e| AppError::internal(format!("Failed to get nodes: {}", e)))?;

    let mut info = NamespaceInfo {
        name: name.clone(),
        memory_count: 0,
        entity_count: 0,
        total_nodes: 0,
        last_activity: 0,
    };

    for node in &all_nodes {
        let ns = match node.metadata.get("_namespace") {
            Some(ucotron_core::Value::String(s)) => s.as_str(),
            _ => "default",
        };
        if ns != name {
            continue;
        }
        info.total_nodes += 1;
        if node.timestamp > info.last_activity {
            info.last_activity = node.timestamp;
        }
        match node.node_type {
            ucotron_core::NodeType::Entity => info.entity_count += 1,
            _ => info.memory_count += 1,
        }
    }

    if info.total_nodes == 0 && name != "default" {
        return Err(AppError::not_found(format!("Namespace '{}' not found", name)));
    }

    Ok(Json(info))
}

// ---------------------------------------------------------------------------
// Admin: Configuration & System Info
// ---------------------------------------------------------------------------

/// Get a read-only summary of the server configuration.
///
/// Returns the effective configuration (TOML + env overrides) without secrets.
#[utoipa::path(
    get,
    path = "/api/v1/admin/config",
    tag = "Admin",
    responses(
        (status = 200, description = "Configuration summary", body = ConfigSummaryResponse)
    )
)]
pub async fn admin_config_handler(
    State(state): State<Arc<AppState>>,
    Extension(auth): Extension<AuthContext>,
) -> Result<Json<ConfigSummaryResponse>, AppError> {
    require_role(&auth, ucotron_config::AuthRole::Admin)?;
    let cfg = &state.config;
    Ok(Json(ConfigSummaryResponse {
        server: ConfigServerSection {
            host: cfg.server.host.clone(),
            port: cfg.server.port,
        },
        storage: ConfigStorageSection {
            mode: cfg.storage.mode.clone(),
            vector_backend: cfg.storage.vector.backend.clone(),
            graph_backend: cfg.storage.graph.backend.clone(),
            vector_data_dir: cfg.storage.vector.data_dir.clone(),
            graph_data_dir: cfg.storage.graph.data_dir.clone(),
        },
        models: ConfigModelsSection {
            models_dir: cfg.models.models_dir.clone(),
            embedding_model: cfg.models.embedding_model.clone(),
        },
        instance: ConfigInstanceSection {
            instance_id: state.instance_id.clone(),
            role: cfg.instance.role.clone(),
            id_range_start: cfg.instance.id_range_start,
            id_range_size: cfg.instance.id_range_size,
        },
        namespaces: ConfigNamespacesSection {
            default_namespace: cfg.namespaces.default_namespace.clone(),
            allowed_namespaces: cfg.namespaces.allowed_namespaces.clone(),
            max_namespaces: cfg.namespaces.max_namespaces,
        },
    }))
}

/// Get system information including memory, CPU, and storage metrics.
#[utoipa::path(
    get,
    path = "/api/v1/admin/system",
    tag = "Admin",
    responses(
        (status = 200, description = "System information", body = SystemInfoResponse),
        (status = 500, description = "Internal error", body = ApiErrorResponse)
    )
)]
pub async fn admin_system_handler(
    State(state): State<Arc<AppState>>,
    Extension(auth): Extension<AuthContext>,
) -> Result<Json<SystemInfoResponse>, AppError> {
    require_role(&auth, ucotron_config::AuthRole::Admin)?;
    let all_nodes = state
        .registry
        .graph()
        .get_all_nodes()
        .map_err(|e| AppError::internal(format!("Failed to get nodes: {}", e)))?;
    let all_edges = state
        .registry
        .graph()
        .get_all_edges_full()
        .map_err(|e| AppError::internal(format!("Failed to get edges: {}", e)))?;

    let next_id = {
        let id = state.next_node_id.lock().unwrap();
        *id
    };

    // Get process RSS (platform-specific).
    let memory_rss_bytes = get_process_rss();

    Ok(Json(SystemInfoResponse {
        memory_rss_bytes,
        cpu_count: std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1),
        next_node_id: next_id,
        id_range_end: state.id_range_end,
        total_nodes: all_nodes.len(),
        total_edges: all_edges.len(),
        uptime_secs: state.start_time.elapsed().as_secs(),
    }))
}

// ---------------------------------------------------------------------------
// GDPR Compliance
// ---------------------------------------------------------------------------

/// Right to be forgotten: delete ALL data associated with a user_id or email.
///
/// Performs a cascade delete across ALL namespaces: removes all nodes
/// (memories, entities) tagged with `_user_id` or `_email` matching the given
/// identifiers, their embeddings from the HNSW vector index, and any edges
/// connecting those nodes. At least one of `user_id` or `email` must be provided.
/// Returns a deletion receipt with counts of all items removed.
#[utoipa::path(
    delete,
    path = "/api/v1/gdpr/forget",
    tag = "GDPR",
    params(GdprForgetParams),
    responses(
        (status = 200, description = "User data erased — deletion receipt", body = GdprForgetResponse),
        (status = 400, description = "Missing user_id and email", body = ApiErrorResponse),
        (status = 403, description = "Read-only instance or GDPR disabled", body = ApiErrorResponse),
        (status = 500, description = "Internal error", body = ApiErrorResponse)
    )
)]
pub async fn gdpr_forget_handler(
    State(state): State<Arc<AppState>>,
    Extension(auth): Extension<AuthContext>,
    Query(params): Query<GdprForgetParams>,
) -> Result<Json<GdprForgetResponse>, AppError> {
    require_role(&auth, ucotron_config::AuthRole::Writer)?;
    if state.is_reader_only() {
        return Err(AppError::read_only());
    }
    if !state.config.gdpr.enabled {
        return Err(AppError {
            status: axum::http::StatusCode::FORBIDDEN,
            code: "GDPR_DISABLED".into(),
            message: "GDPR endpoints are disabled in configuration".into(),
        });
    }

    let user_id = params.user_id.as_deref().map(|s| s.trim()).filter(|s| !s.is_empty()).map(String::from);
    let email = params.email.as_deref().map(|s| s.trim()).filter(|s| !s.is_empty()).map(String::from);

    if user_id.is_none() && email.is_none() {
        return Err(AppError::bad_request(
            "At least one of user_id or email must be provided",
        ));
    }

    // Collect all nodes across ALL namespaces belonging to this user.
    let all_nodes = state
        .registry
        .graph()
        .get_all_nodes()
        .map_err(|e| AppError::internal(format!("Failed to get nodes: {}", e)))?;

    let user_node_ids: Vec<u64> = all_nodes
        .iter()
        .filter(|n| {
            let matches_user_id = user_id.as_ref().map_or(false, |uid| {
                matches!(
                    n.metadata.get("_user_id"),
                    Some(ucotron_core::Value::String(s)) if s == uid
                )
            });
            let matches_email = email.as_ref().map_or(false, |em| {
                matches!(
                    n.metadata.get("_email"),
                    Some(ucotron_core::Value::String(s)) if s == em
                )
            });
            matches_user_id || matches_email
        })
        .map(|n| n.id)
        .collect();

    let mut memories_deleted = 0usize;
    let mut entities_deleted = 0usize;
    let mut namespaces_affected: std::collections::HashSet<String> =
        std::collections::HashSet::new();

    for n in &all_nodes {
        if user_node_ids.contains(&n.id) {
            // Track which namespaces are affected.
            let ns = n
                .metadata
                .get("_namespace")
                .and_then(|v| match v {
                    ucotron_core::Value::String(s) => Some(s.clone()),
                    _ => None,
                })
                .unwrap_or_else(|| "default".to_string());
            namespaces_affected.insert(ns);

            match n.node_type {
                ucotron_core::NodeType::Entity => entities_deleted += 1,
                _ => memories_deleted += 1,
            }
        }
    }

    // Count edges that will be removed (edges where either endpoint belongs to user).
    let all_edges = state
        .registry
        .graph()
        .get_all_edges_full()
        .map_err(|e| AppError::internal(format!("Failed to get edges: {}", e)))?;

    let user_id_set: std::collections::HashSet<u64> = user_node_ids.iter().copied().collect();
    let edges_removed = all_edges
        .iter()
        .filter(|e| user_id_set.contains(&e.source) || user_id_set.contains(&e.target))
        .count();

    let embeddings_deleted = user_node_ids.len();

    // Cascade delete: remove from HNSW vector index, then graph nodes (which also removes edges).
    if !user_node_ids.is_empty() {
        state
            .registry
            .vector()
            .delete(&user_node_ids)
            .map_err(|e| AppError::internal(format!("Vector delete failed: {}", e)))?;

        state
            .registry
            .graph()
            .delete_nodes(&user_node_ids)
            .map_err(|e| AppError::internal(format!("Graph delete failed: {}", e)))?;
    }

    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let total_items_removed = memories_deleted + entities_deleted + edges_removed + embeddings_deleted;

    // Build target identifier string for audit trail.
    let target_desc = match (&user_id, &email) {
        (Some(uid), Some(em)) => format!("user_id={}, email={}", uid, em),
        (Some(uid), None) => format!("user_id={}", uid),
        (None, Some(em)) => format!("email={}", em),
        (None, None) => "unknown".to_string(),
    };

    // Record audit entry in node metadata (append-only audit trail — keep audit, delete data).
    let audit_node_id = state.alloc_next_node_id();
    let mut audit_meta = std::collections::HashMap::new();
    audit_meta.insert(
        "_gdpr_audit".into(),
        ucotron_core::Value::Bool(true),
    );
    audit_meta.insert(
        "_gdpr_operation".into(),
        ucotron_core::Value::String("forget".into()),
    );
    audit_meta.insert(
        "_gdpr_target".into(),
        ucotron_core::Value::String(target_desc.clone()),
    );
    audit_meta.insert(
        "_gdpr_details".into(),
        ucotron_core::Value::String(format!(
            "Erased {} memories, {} entities, {} edges, {} embeddings across {} namespaces (total: {})",
            memories_deleted, entities_deleted, edges_removed, embeddings_deleted,
            namespaces_affected.len(), total_items_removed
        )),
    );
    audit_meta.insert(
        "_namespace".into(),
        ucotron_core::Value::String("_gdpr_audit".into()),
    );

    let audit_node = ucotron_core::Node {
        id: audit_node_id,
        content: format!("[gdpr:forget:{}]", target_desc),
        embedding: vec![0.0f32; 384],
        metadata: audit_meta,
        node_type: ucotron_core::NodeType::Event,
        timestamp: now,
        media_type: None,
        media_uri: None,
        embedding_visual: None,
        timestamp_range: None,
        parent_video_id: None,
    };
    let _ = state.registry.graph().upsert_nodes(&[audit_node]);

    let mut sorted_namespaces: Vec<String> = namespaces_affected.into_iter().collect();
    sorted_namespaces.sort();

    Ok(Json(GdprForgetResponse {
        user_id,
        email,
        memories_deleted,
        entities_deleted,
        edges_removed,
        embeddings_deleted,
        total_items_removed,
        erased_at: now,
        namespaces_affected: sorted_namespaces,
    }))
}

/// Export all data associated with a user_id for GDPR data portability.
///
/// Returns all nodes and edges belonging to the user in a machine-readable format.
#[utoipa::path(
    get,
    path = "/api/v1/gdpr/export",
    tag = "GDPR",
    params(GdprExportParams),
    responses(
        (status = 200, description = "User data export", body = GdprExportResponse),
        (status = 400, description = "Missing user_id", body = ApiErrorResponse),
        (status = 403, description = "GDPR disabled", body = ApiErrorResponse),
        (status = 500, description = "Internal error", body = ApiErrorResponse)
    )
)]
pub async fn gdpr_export_handler(
    State(state): State<Arc<AppState>>,
    Extension(auth): Extension<AuthContext>,
    Query(params): Query<GdprExportParams>,
) -> Result<Json<GdprExportResponse>, AppError> {
    require_role(&auth, ucotron_config::AuthRole::Writer)?;
    if !state.config.gdpr.enabled {
        return Err(AppError {
            status: axum::http::StatusCode::FORBIDDEN,
            code: "GDPR_DISABLED".into(),
            message: "GDPR endpoints are disabled in configuration".into(),
        });
    }

    let user_id = params.user_id.trim().to_string();
    if user_id.is_empty() {
        return Err(AppError::bad_request("user_id must not be empty"));
    }

    let all_nodes = state
        .registry
        .graph()
        .get_all_nodes()
        .map_err(|e| AppError::internal(format!("Failed to get nodes: {}", e)))?;

    let user_nodes: Vec<&ucotron_core::Node> = all_nodes
        .iter()
        .filter(|n| {
            matches!(
                n.metadata.get("_user_id"),
                Some(ucotron_core::Value::String(s)) if s == &user_id
            )
        })
        .collect();

    let user_id_set: std::collections::HashSet<u64> =
        user_nodes.iter().map(|n| n.id).collect();

    let all_edges = state
        .registry
        .graph()
        .get_all_edges_full()
        .map_err(|e| AppError::internal(format!("Failed to get edges: {}", e)))?;

    let user_edges: Vec<GdprExportEdge> = all_edges
        .iter()
        .filter(|e| user_id_set.contains(&e.source) && user_id_set.contains(&e.target))
        .map(|e| GdprExportEdge {
            source: e.source,
            target: e.target,
            edge_type: format!("{:?}", e.edge_type),
            weight: e.weight,
        })
        .collect();

    let node_responses: Vec<MemoryResponse> =
        user_nodes.iter().map(|n| node_to_memory_response(n)).collect();

    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    // Record audit entry.
    let audit_node_id = state.alloc_next_node_id();
    let mut audit_meta = std::collections::HashMap::new();
    audit_meta.insert("_gdpr_audit".into(), ucotron_core::Value::Bool(true));
    audit_meta.insert("_gdpr_operation".into(), ucotron_core::Value::String("export".into()));
    audit_meta.insert("_gdpr_target".into(), ucotron_core::Value::String(user_id.clone()));
    audit_meta.insert(
        "_gdpr_details".into(),
        ucotron_core::Value::String(format!(
            "Exported {} nodes, {} edges",
            node_responses.len(),
            user_edges.len()
        )),
    );
    audit_meta.insert("_namespace".into(), ucotron_core::Value::String("_gdpr_audit".into()));

    let audit_node = ucotron_core::Node {
        id: audit_node_id,
        content: format!("[gdpr:export:{}]", user_id),
        embedding: vec![0.0f32; 384],
        metadata: audit_meta,
        node_type: ucotron_core::NodeType::Event,
        timestamp: now,
        media_type: None,
        media_uri: None,
        embedding_visual: None,
        timestamp_range: None,
        parent_video_id: None,
    };
    let _ = state.registry.graph().upsert_nodes(&[audit_node]);

    let stats = GdprExportStats {
        total_nodes: node_responses.len(),
        total_edges: user_edges.len(),
    };

    Ok(Json(GdprExportResponse {
        user_id,
        nodes: node_responses,
        edges: user_edges,
        stats,
        exported_at: now,
    }))
}

/// Get current retention policies.
#[utoipa::path(
    get,
    path = "/api/v1/gdpr/retention",
    tag = "GDPR",
    responses(
        (status = 200, description = "Current retention policies", body = RetentionStatusResponse),
        (status = 403, description = "GDPR disabled", body = ApiErrorResponse)
    )
)]
pub async fn gdpr_retention_status_handler(
    State(state): State<Arc<AppState>>,
    Extension(auth): Extension<AuthContext>,
) -> Result<Json<RetentionStatusResponse>, AppError> {
    require_role(&auth, ucotron_config::AuthRole::Admin)?;
    if !state.config.gdpr.enabled {
        return Err(AppError {
            status: axum::http::StatusCode::FORBIDDEN,
            code: "GDPR_DISABLED".into(),
            message: "GDPR endpoints are disabled in configuration".into(),
        });
    }

    let mut policies: Vec<RetentionPolicy> = state
        .config
        .gdpr
        .retention_policies
        .iter()
        .map(|p| RetentionPolicy {
            namespace: p.namespace.clone(),
            ttl_secs: p.ttl_secs,
        })
        .collect();

    // Add default policy if configured.
    if state.config.gdpr.default_retention_ttl_secs > 0 {
        policies.push(RetentionPolicy {
            namespace: "*".into(),
            ttl_secs: state.config.gdpr.default_retention_ttl_secs,
        });
    }

    Ok(Json(RetentionStatusResponse {
        policies,
        last_sweep_expired: 0,
        last_sweep_at: 0,
    }))
}

/// Manually trigger a retention sweep: delete nodes older than their namespace TTL.
#[utoipa::path(
    post,
    path = "/api/v1/gdpr/retention/sweep",
    tag = "GDPR",
    responses(
        (status = 200, description = "Sweep completed", body = RetentionSweepResponse),
        (status = 403, description = "Read-only or GDPR disabled", body = ApiErrorResponse),
        (status = 500, description = "Internal error", body = ApiErrorResponse)
    )
)]
pub async fn gdpr_retention_sweep_handler(
    State(state): State<Arc<AppState>>,
    Extension(auth): Extension<AuthContext>,
) -> Result<Json<RetentionSweepResponse>, AppError> {
    require_role(&auth, ucotron_config::AuthRole::Admin)?;
    if state.is_reader_only() {
        return Err(AppError::read_only());
    }
    if !state.config.gdpr.enabled {
        return Err(AppError {
            status: axum::http::StatusCode::FORBIDDEN,
            code: "GDPR_DISABLED".into(),
            message: "GDPR endpoints are disabled in configuration".into(),
        });
    }

    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    // Build a map of namespace → TTL.
    let mut ns_ttl: std::collections::HashMap<String, u64> = std::collections::HashMap::new();
    let mut default_ttl = state.config.gdpr.default_retention_ttl_secs;

    for policy in &state.config.gdpr.retention_policies {
        if policy.namespace == "*" {
            default_ttl = policy.ttl_secs;
        } else {
            ns_ttl.insert(policy.namespace.clone(), policy.ttl_secs);
        }
    }

    // No retention configured at all → nothing to do.
    if default_ttl == 0 && ns_ttl.is_empty() {
        return Ok(Json(RetentionSweepResponse {
            nodes_expired: 0,
            namespaces_checked: 0,
            swept_at: now,
        }));
    }

    let all_nodes = state
        .registry
        .graph()
        .get_all_nodes()
        .map_err(|e| AppError::internal(format!("Failed to get nodes: {}", e)))?;

    let mut expired_ids: Vec<u64> = Vec::new();
    let mut namespaces_seen: std::collections::HashSet<String> = std::collections::HashSet::new();

    for node in &all_nodes {
        // Skip audit nodes.
        if matches!(
            node.metadata.get("_gdpr_audit"),
            Some(ucotron_core::Value::Bool(true))
        ) {
            continue;
        }
        // Skip sentinel nodes.
        if matches!(
            node.metadata.get("_sentinel"),
            Some(ucotron_core::Value::Bool(true))
        ) {
            continue;
        }

        let ns = match node.metadata.get("_namespace") {
            Some(ucotron_core::Value::String(s)) => s.clone(),
            _ => "default".to_string(),
        };
        namespaces_seen.insert(ns.clone());

        // Find applicable TTL.
        let ttl = ns_ttl.get(&ns).copied().unwrap_or(default_ttl);
        if ttl == 0 {
            continue; // No retention for this namespace.
        }

        // Check if node has expired.
        if node.timestamp > 0 && now.saturating_sub(node.timestamp) > ttl {
            expired_ids.push(node.id);
        }
    }

    // Delete expired nodes.
    if !expired_ids.is_empty() {
        state
            .registry
            .vector()
            .delete(&expired_ids)
            .map_err(|e| AppError::internal(format!("Vector delete failed: {}", e)))?;
        state
            .registry
            .graph()
            .delete_nodes(&expired_ids)
            .map_err(|e| AppError::internal(format!("Graph delete failed: {}", e)))?;
    }

    // Record audit entry for the sweep.
    let audit_node_id = state.alloc_next_node_id();
    let mut audit_meta = std::collections::HashMap::new();
    audit_meta.insert("_gdpr_audit".into(), ucotron_core::Value::Bool(true));
    audit_meta.insert("_gdpr_operation".into(), ucotron_core::Value::String("retention_sweep".into()));
    audit_meta.insert("_gdpr_target".into(), ucotron_core::Value::String("*".into()));
    audit_meta.insert(
        "_gdpr_details".into(),
        ucotron_core::Value::String(format!(
            "Swept {} expired nodes across {} namespaces",
            expired_ids.len(),
            namespaces_seen.len()
        )),
    );
    audit_meta.insert("_namespace".into(), ucotron_core::Value::String("_gdpr_audit".into()));

    let audit_node = ucotron_core::Node {
        id: audit_node_id,
        content: "[gdpr:retention_sweep]".into(),
        embedding: vec![0.0f32; 384],
        metadata: audit_meta,
        node_type: ucotron_core::NodeType::Event,
        timestamp: now,
        media_type: None,
        media_uri: None,
        embedding_visual: None,
        timestamp_range: None,
        parent_video_id: None,
    };
    let _ = state.registry.graph().upsert_nodes(&[audit_node]);

    Ok(Json(RetentionSweepResponse {
        nodes_expired: expired_ids.len(),
        namespaces_checked: namespaces_seen.len(),
        swept_at: now,
    }))
}

// ---------------------------------------------------------------------------
// Internal Helpers
// ---------------------------------------------------------------------------

/// Get process RSS memory in bytes (platform-specific).
#[allow(deprecated)]
fn get_process_rss() -> u64 {
    #[cfg(target_os = "macos")]
    {
        use std::mem;
        let mut info: libc::mach_task_basic_info = unsafe { mem::zeroed() };
        let mut count = (mem::size_of::<libc::mach_task_basic_info>()
            / mem::size_of::<libc::natural_t>()) as libc::mach_msg_type_number_t;
        let kr = unsafe {
            libc::task_info(
                libc::mach_task_self(),
                libc::MACH_TASK_BASIC_INFO,
                &mut info as *mut _ as *mut _,
                &mut count,
            )
        };
        if kr == libc::KERN_SUCCESS {
            info.resident_size
        } else {
            0
        }
    }
    #[cfg(target_os = "linux")]
    {
        if let Ok(contents) = std::fs::read_to_string("/proc/self/status") {
            for line in contents.lines() {
                if line.starts_with("VmRSS:") {
                    let parts: Vec<&str> = line.split_whitespace().collect();
                    if parts.len() >= 2 {
                        if let Ok(kb) = parts[1].parse::<u64>() {
                            return kb * 1024;
                        }
                    }
                }
            }
        }
        0
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        0
    }
}

// ---------------------------------------------------------------------------
// RBAC / API Key Management
// ---------------------------------------------------------------------------

/// GET /api/v1/auth/whoami — return the role and scope of the current caller.
#[utoipa::path(
    get,
    path = "/api/v1/auth/whoami",
    tag = "Auth",
    responses(
        (status = 200, description = "Current authentication context", body = WhoamiResponse),
        (status = 401, description = "Unauthorized"),
    )
)]
pub async fn whoami_handler(
    State(state): State<Arc<AppState>>,
    request: axum::extract::Request,
) -> Result<Json<WhoamiResponse>, AppError> {
    let ctx = request
        .extensions()
        .get::<AuthContext>()
        .cloned()
        .unwrap_or(AuthContext {
            role: ucotron_config::AuthRole::Admin,
            namespace_scope: None,
            key_name: None,
        });

    Ok(Json(WhoamiResponse {
        role: ctx.role.as_str().to_string(),
        namespace_scope: ctx.namespace_scope,
        key_name: ctx.key_name,
        auth_enabled: state.config.auth.enabled,
    }))
}

/// GET /api/v1/auth/keys — list all API keys (values masked). Admin only.
#[utoipa::path(
    get,
    path = "/api/v1/auth/keys",
    tag = "Auth",
    responses(
        (status = 200, description = "List of API keys (masked)", body = ListApiKeysResponse),
        (status = 403, description = "Forbidden — requires admin role"),
    )
)]
pub async fn list_api_keys_handler(
    State(state): State<Arc<AppState>>,
    request: axum::extract::Request,
) -> Result<Json<ListApiKeysResponse>, AppError> {
    let ctx = request.extensions().get::<AuthContext>().cloned().unwrap_or(AuthContext {
        role: ucotron_config::AuthRole::Admin,
        namespace_scope: None,
        key_name: None,
    });
    require_role(&ctx, ucotron_config::AuthRole::Admin)?;

    let keys: Vec<ApiKeyInfo> = {
        let api_keys = state.api_keys.read().unwrap();
        api_keys
            .iter()
            .map(|entry| ApiKeyInfo {
                name: entry.name.clone(),
                key_preview: mask_key(&entry.key),
                role: entry.role.clone(),
                namespace: entry.namespace.clone(),
                active: entry.active,
            })
            .collect()
    };

    Ok(Json(ListApiKeysResponse { keys }))
}

/// POST /api/v1/auth/keys — create a new API key. Admin only.
#[utoipa::path(
    post,
    path = "/api/v1/auth/keys",
    tag = "Auth",
    request_body = CreateApiKeyRequest,
    responses(
        (status = 200, description = "API key created (secret shown once)", body = CreateApiKeyResponse),
        (status = 400, description = "Invalid role or duplicate name"),
        (status = 403, description = "Forbidden — requires admin role"),
    )
)]
pub async fn create_api_key_handler(
    State(state): State<Arc<AppState>>,
    request: axum::extract::Request,
) -> Result<Json<CreateApiKeyResponse>, AppError> {
    // Parse body manually since we also need extensions.
    let (parts, body) = request.into_parts();
    let ctx = parts.extensions.get::<AuthContext>().cloned().unwrap_or(AuthContext {
        role: ucotron_config::AuthRole::Admin,
        namespace_scope: None,
        key_name: None,
    });
    require_role(&ctx, ucotron_config::AuthRole::Admin)?;

    let bytes = axum::body::to_bytes(body, 1024 * 64)
        .await
        .map_err(|e| AppError::bad_request(format!("Failed to read body: {}", e)))?;
    let req: CreateApiKeyRequest = serde_json::from_slice(&bytes)
        .map_err(|e| AppError::bad_request(format!("Invalid JSON: {}", e)))?;

    // Validate role.
    let valid_roles = ["admin", "writer", "reader", "viewer"];
    if !valid_roles.contains(&req.role.as_str()) {
        return Err(AppError::bad_request(format!(
            "Invalid role '{}'. Must be one of: {}",
            req.role,
            valid_roles.join(", ")
        )));
    }

    if req.name.is_empty() {
        return Err(AppError::bad_request("Key name must not be empty."));
    }

    // Check for duplicate name.
    {
        let api_keys = state.api_keys.read().unwrap();
        if api_keys.iter().any(|e| e.name == req.name) {
            return Err(AppError::bad_request(format!(
                "API key with name '{}' already exists.",
                req.name
            )));
        }
    }

    // Generate a random key.
    let key = generate_api_key();

    // Store the new key in the runtime-mutable api_keys list.
    // Note: Runtime-created keys are ephemeral (lost on restart).
    // The canonical source for persistent keys is ucotron.toml.
    {
        let mut api_keys = state.api_keys.write().unwrap();
        api_keys.push(ucotron_config::ApiKeyEntry {
            name: req.name.clone(),
            key: key.clone(),
            role: req.role.clone(),
            namespace: req.namespace.clone(),
            active: true,
        });
    }

    // Return the key (shown only once).
    Ok(Json(CreateApiKeyResponse {
        name: req.name,
        key,
        role: req.role,
        namespace: req.namespace,
        active: true,
    }))
}

/// DELETE /api/v1/auth/keys/:name — revoke an API key. Admin only.
#[utoipa::path(
    delete,
    path = "/api/v1/auth/keys/{name}",
    tag = "Auth",
    params(
        ("name" = String, Path, description = "Name of the API key to revoke"),
    ),
    responses(
        (status = 200, description = "API key revoked", body = RevokeApiKeyResponse),
        (status = 403, description = "Forbidden — requires admin role"),
        (status = 404, description = "Key not found"),
    )
)]
pub async fn revoke_api_key_handler(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
    request: axum::extract::Request,
) -> Result<Json<RevokeApiKeyResponse>, AppError> {
    let ctx = request.extensions().get::<AuthContext>().cloned().unwrap_or(AuthContext {
        role: ucotron_config::AuthRole::Admin,
        namespace_scope: None,
        key_name: None,
    });
    require_role(&ctx, ucotron_config::AuthRole::Admin)?;

    // Deactivate the key in the runtime api_keys list.
    let revoked = {
        let mut api_keys = state.api_keys.write().unwrap();
        if let Some(entry) = api_keys.iter_mut().find(|e| e.name == name) {
            entry.active = false;
            true
        } else {
            false
        }
    };

    if !revoked {
        return Err(AppError::not_found(format!("API key '{}' not found.", name)));
    }

    Ok(Json(RevokeApiKeyResponse {
        name,
        revoked: true,
    }))
}

/// Mask an API key for display (show first 3 and last 4 chars).
fn mask_key(key: &str) -> String {
    if key.len() <= 8 {
        return "****".to_string();
    }
    format!("{}****{}", &key[..3], &key[key.len() - 4..])
}

/// Generate a random API key (32 hex chars prefixed with "mk_").
fn generate_api_key() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    // Simple but unique key generation using timestamp + pid.
    let pid = std::process::id();
    format!("mk_{:016x}{:08x}", ts, pid)
}

// ---------------------------------------------------------------------------
// Audit Endpoints
// ---------------------------------------------------------------------------

/// Query audit log entries with optional filters.
///
/// Returns matching audit entries sorted by timestamp (oldest first).
/// Requires admin role when auth is enabled.
#[utoipa::path(
    get,
    path = "/api/v1/audit",
    params(crate::audit::AuditFilter),
    responses(
        (status = 200, description = "Audit entries matching filter", body = AuditQueryResponse),
        (status = 403, description = "Forbidden — requires admin role"),
    )
)]
pub async fn audit_query_handler(
    State(state): State<Arc<AppState>>,
    Query(filter): Query<crate::audit::AuditFilter>,
    request: axum::extract::Request,
) -> Result<Json<AuditQueryResponse>, AppError> {
    let ctx = request.extensions().get::<AuthContext>().cloned().unwrap_or(AuthContext {
        role: ucotron_config::AuthRole::Admin,
        namespace_scope: None,
        key_name: None,
    });
    require_role(&ctx, ucotron_config::AuthRole::Admin)?;

    let entries = state.audit_log.query(&filter);
    let total = entries.len();

    Ok(Json(AuditQueryResponse { entries, total }))
}

/// Export the full audit log for compliance purposes.
///
/// Returns all audit entries (unfiltered) as a JSON array.
/// Requires admin role when auth is enabled.
#[utoipa::path(
    get,
    path = "/api/v1/audit/export",
    responses(
        (status = 200, description = "Full audit log export", body = AuditExportResponse),
        (status = 403, description = "Forbidden — requires admin role"),
    )
)]
pub async fn audit_export_handler(
    State(state): State<Arc<AppState>>,
    request: axum::extract::Request,
) -> Result<Json<AuditExportResponse>, AppError> {
    let ctx = request.extensions().get::<AuthContext>().cloned().unwrap_or(AuthContext {
        role: ucotron_config::AuthRole::Admin,
        namespace_scope: None,
        key_name: None,
    });
    require_role(&ctx, ucotron_config::AuthRole::Admin)?;

    let entries = state.audit_log.export_all();
    let total = entries.len();
    let exported_at = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    Ok(Json(AuditExportResponse {
        entries,
        total,
        exported_at,
    }))
}

// ---------------------------------------------------------------------------
// Fine-Tuning Dataset Generation
// ---------------------------------------------------------------------------

/// Generate a training dataset for relation extraction fine-tuning.
///
/// Reads the knowledge graph and produces JSONL training data in SFT messages format,
/// suitable for fine-tuning with TRL (Transformer Reinforcement Learning).
#[utoipa::path(
    post,
    path = "/api/v1/finetune/generate-dataset",
    tag = "Fine-Tuning",
    request_body = GenerateDatasetRequest,
    params(
        ("X-Ucotron-Namespace" = Option<String>, Header, description = "Namespace")
    ),
    responses(
        (status = 200, description = "Dataset generated", body = GenerateDatasetResponse),
        (status = 403, description = "Forbidden"),
    ),
)]
pub async fn generate_dataset_handler(
    State(state): State<Arc<AppState>>,
    Extension(auth): Extension<AuthContext>,
    headers: HeaderMap,
    Json(body): Json<GenerateDatasetRequest>,
) -> Result<Json<GenerateDatasetResponse>, AppError> {
    require_role(&auth, ucotron_config::AuthRole::Admin)?;
    let _ns = extract_namespace(&headers);

    let config = ucotron_extraction::fine_tuning::DatasetConfig {
        max_samples: body.max_samples.unwrap_or(10_000),
        train_ratio: body.train_ratio.unwrap_or(0.8),
        min_relations: body.min_relations.unwrap_or(1),
        max_text_length: body.max_text_length.unwrap_or(2048),
        seed: body.seed.unwrap_or(42),
    };

    let result = ucotron_extraction::fine_tuning::generate_dataset(
        &state.registry,
        &config,
    )
    .map_err(|e| AppError::internal(format!("Dataset generation failed: {}", e)))?;

    // Convert to SFT messages format
    let train_samples: Vec<serde_json::Value> = result
        .train
        .iter()
        .map(ucotron_extraction::fine_tuning::sample_to_messages)
        .collect();
    let validation_samples: Vec<serde_json::Value> = result
        .validation
        .iter()
        .map(ucotron_extraction::fine_tuning::sample_to_messages)
        .collect();

    Ok(Json(GenerateDatasetResponse {
        train_samples,
        validation_samples,
        total_entities: result.total_entities,
        total_edges: result.total_edges,
        skipped: result.skipped,
    }))
}

// ---------------------------------------------------------------------------
// Agents
// ---------------------------------------------------------------------------

/// Generate a short unique agent ID from a sanitized name and timestamp.
fn generate_agent_id(name: &str) -> String {
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    let slug: String = name
        .chars()
        .filter(|c| c.is_alphanumeric() || *c == '-' || *c == '_')
        .take(32)
        .collect::<String>()
        .to_lowercase();
    if slug.is_empty() {
        format!("agent-{}", ts)
    } else {
        format!("{}-{}", slug, ts % 1_000_000)
    }
}

/// Create a new agent with an auto-generated isolated namespace.
#[utoipa::path(
    post,
    path = "/api/v1/agents",
    tag = "Agents",
    request_body = CreateAgentRequest,
    responses(
        (status = 201, description = "Agent created with isolated namespace", body = CreateAgentResponse),
        (status = 400, description = "Invalid request", body = ApiErrorResponse),
        (status = 403, description = "Forbidden", body = ApiErrorResponse),
        (status = 500, description = "Internal server error", body = ApiErrorResponse)
    )
)]
pub async fn create_agent_handler(
    State(state): State<Arc<AppState>>,
    Extension(auth): Extension<AuthContext>,
    Json(body): Json<CreateAgentRequest>,
) -> Result<(axum::http::StatusCode, Json<CreateAgentResponse>), AppError> {
    require_role(&auth, ucotron_config::AuthRole::Writer)?;
    if state.is_reader_only() {
        return Err(AppError::read_only());
    }

    if body.name.trim().is_empty() {
        return Err(AppError::bad_request("agent name must not be empty"));
    }

    let agent_id = generate_agent_id(&body.name);
    let created_at = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let owner = auth.key_name.clone().unwrap_or_else(|| format!("{:?}", auth.role));

    // Convert JSON config values to core Value type.
    let config: std::collections::HashMap<String, ucotron_core::Value> = body
        .config
        .iter()
        .filter_map(|(k, v)| {
            let core_val = match v {
                serde_json::Value::String(s) => Some(ucotron_core::Value::String(s.clone())),
                serde_json::Value::Number(n) => {
                    if let Some(i) = n.as_i64() {
                        Some(ucotron_core::Value::Integer(i))
                    } else {
                        n.as_f64().map(ucotron_core::Value::Float)
                    }
                }
                serde_json::Value::Bool(b) => Some(ucotron_core::Value::Bool(*b)),
                _ => None,
            };
            core_val.map(|cv| (k.clone(), cv))
        })
        .collect();

    let agent = Agent::new(&agent_id, body.name.trim(), &owner, created_at)
        .with_config(config);

    state
        .registry
        .graph()
        .create_agent(&agent)
        .map_err(|e| AppError::internal(format!("Failed to create agent: {}", e)))?;

    Ok((
        axum::http::StatusCode::CREATED,
        Json(CreateAgentResponse {
            id: agent.id,
            name: agent.name,
            namespace: agent.namespace,
            owner: agent.owner,
            created_at: agent.created_at,
        }),
    ))
}

/// List all agents, optionally filtered by owner via query param, with pagination.
#[utoipa::path(
    get,
    path = "/api/v1/agents",
    tag = "Agents",
    params(ListAgentsParams),
    responses(
        (status = 200, description = "List of agents", body = ListAgentsResponse),
        (status = 403, description = "Forbidden", body = ApiErrorResponse),
    )
)]
pub async fn list_agents_handler(
    State(state): State<Arc<AppState>>,
    Extension(auth): Extension<AuthContext>,
    Query(params): Query<ListAgentsParams>,
) -> Result<Json<ListAgentsResponse>, AppError> {
    require_role(&auth, ucotron_config::AuthRole::Reader)?;

    // Determine owner filter: explicit query param takes precedence,
    // otherwise non-admin users are scoped to their own agents.
    let owner_filter: Option<&str> = if auth.role == ucotron_config::AuthRole::Admin {
        // Admins can filter by owner via query param, or see all agents if omitted.
        params.owner.as_deref()
    } else {
        // Non-admins always scoped to their own agents (query param ignored).
        auth.key_name.as_deref()
    };

    let agents = state
        .registry
        .graph()
        .list_agents(owner_filter)
        .map_err(|e| AppError::internal(format!("Failed to list agents: {}", e)))?;

    let total = agents.len();
    let agent_responses: Vec<AgentResponse> = agents
        .into_iter()
        .skip(params.offset)
        .take(params.limit)
        .map(|a| AgentResponse {
            id: a.id,
            name: a.name,
            namespace: a.namespace,
            owner: a.owner,
            created_at: a.created_at,
            config: a
                .config
                .into_iter()
                .map(|(k, v)| (k, core_value_to_json(&v)))
                .collect(),
        })
        .collect();

    Ok(Json(ListAgentsResponse {
        agents: agent_responses,
        total,
        limit: params.limit,
        offset: params.offset,
    }))
}

/// Get a specific agent by ID.
#[utoipa::path(
    get,
    path = "/api/v1/agents/{id}",
    tag = "Agents",
    params(("id" = String, Path, description = "Agent ID")),
    responses(
        (status = 200, description = "Agent details", body = AgentResponse),
        (status = 404, description = "Agent not found", body = ApiErrorResponse),
    )
)]
pub async fn get_agent_handler(
    State(state): State<Arc<AppState>>,
    Extension(auth): Extension<AuthContext>,
    Path(id): Path<String>,
) -> Result<Json<AgentResponse>, AppError> {
    require_role(&auth, ucotron_config::AuthRole::Reader)?;

    let agent = state
        .registry
        .graph()
        .get_agent(&id)
        .map_err(|e| AppError::internal(format!("Failed to get agent: {}", e)))?
        .ok_or_else(|| AppError::not_found(format!("Agent '{}' not found", id)))?;

    Ok(Json(AgentResponse {
        id: agent.id,
        name: agent.name,
        namespace: agent.namespace,
        owner: agent.owner,
        created_at: agent.created_at,
        config: agent
            .config
            .into_iter()
            .map(|(k, v)| (k, core_value_to_json(&v)))
            .collect(),
    }))
}

/// Delete an agent and cascade-delete all associated data.
///
/// Performs a full cascade delete:
/// 1. Finds all nodes in the agent's namespace
/// 2. Deletes those nodes and their edges from the graph backend
/// 3. Deletes the corresponding embeddings from the vector backend
/// 4. Deletes the agent record and share grants
#[utoipa::path(
    delete,
    path = "/api/v1/agents/{id}",
    tag = "Agents",
    params(("id" = String, Path, description = "Agent ID")),
    responses(
        (status = 200, description = "Agent deleted with all associated data"),
        (status = 404, description = "Agent not found", body = ApiErrorResponse),
        (status = 403, description = "Forbidden", body = ApiErrorResponse),
    )
)]
pub async fn delete_agent_handler(
    State(state): State<Arc<AppState>>,
    Extension(auth): Extension<AuthContext>,
    Path(id): Path<String>,
) -> Result<Json<serde_json::Value>, AppError> {
    require_role(&auth, ucotron_config::AuthRole::Admin)?;
    if state.is_reader_only() {
        return Err(AppError::read_only());
    }

    // Verify agent exists before deleting
    let agent = state
        .registry
        .graph()
        .get_agent(&id)
        .map_err(|e| AppError::internal(format!("Failed to get agent: {}", e)))?
        .ok_or_else(|| AppError::not_found(format!("Agent '{}' not found", id)))?;

    // Cascade delete: find all nodes in the agent's namespace and delete them
    let namespace = &agent.namespace;
    let all_nodes = state
        .registry
        .graph()
        .get_all_nodes()
        .map_err(|e| AppError::internal(format!("Failed to list nodes: {}", e)))?;

    let namespace_node_ids: Vec<u64> = all_nodes
        .iter()
        .filter(|n| node_matches_namespace(n, namespace))
        .map(|n| n.id)
        .collect();

    let nodes_deleted = namespace_node_ids.len();

    if !namespace_node_ids.is_empty() {
        // Delete embeddings from vector backend
        state
            .registry
            .vector()
            .delete(&namespace_node_ids)
            .map_err(|e| AppError::internal(format!("Failed to delete embeddings: {}", e)))?;

        // Delete nodes and their edges from graph backend
        state
            .registry
            .graph()
            .delete_nodes(&namespace_node_ids)
            .map_err(|e| AppError::internal(format!("Failed to delete nodes: {}", e)))?;
    }

    // Delete agent record and share grants
    state
        .registry
        .graph()
        .delete_agent(&id)
        .map_err(|e| AppError::internal(format!("Failed to delete agent: {}", e)))?;

    Ok(Json(serde_json::json!({
        "id": id,
        "deleted": true,
        "nodes_deleted": nodes_deleted
    })))
}

// ---------------------------------------------------------------------------
// Agent Graph Clone
// ---------------------------------------------------------------------------

/// Parse a string to a NodeType. Case-insensitive.
fn parse_node_type(s: &str) -> Option<ucotron_core::NodeType> {
    match s.to_lowercase().as_str() {
        "entity" => Some(ucotron_core::NodeType::Entity),
        "event" => Some(ucotron_core::NodeType::Event),
        "fact" => Some(ucotron_core::NodeType::Fact),
        "skill" => Some(ucotron_core::NodeType::Skill),
        _ => None,
    }
}

fn parse_mindset_tag(s: &str) -> Option<ucotron_core::MindsetTag> {
    match s.to_lowercase().as_str() {
        "convergent" => Some(ucotron_core::MindsetTag::Convergent),
        "divergent" => Some(ucotron_core::MindsetTag::Divergent),
        "algorithmic" => Some(ucotron_core::MindsetTag::Algorithmic),
        _ => None,
    }
}

/// Build a [`MindsetDetector`] from the server's mindset config.
///
/// Returns `None` if mindset auto-detection is disabled in the config.
fn build_mindset_detector(config: &ucotron_config::MindsetDetectorConfig) -> Option<ucotron_core::MindsetDetector> {
    if !config.enabled {
        return None;
    }
    let alg: Vec<&str> = config.algorithmic_keywords.iter().map(|s| s.as_str()).collect();
    let div: Vec<&str> = config.divergent_keywords.iter().map(|s| s.as_str()).collect();
    let con: Vec<&str> = config.convergent_keywords.iter().map(|s| s.as_str()).collect();
    Some(ucotron_core::MindsetDetector::from_keyword_lists(&alg, &div, &con))
}

/// Clone an agent's graph into a new namespace with optional filters.
///
/// Copies all matching nodes and their interconnecting edges from the
/// source agent's namespace to a target namespace. Node IDs are remapped
/// to avoid collisions. Supports filtering by node_type and time_range.
#[utoipa::path(
    post,
    path = "/api/v1/agents/{id}/clone",
    tag = "Agents",
    params(("id" = String, Path, description = "Source agent ID")),
    request_body = CloneAgentRequest,
    responses(
        (status = 200, description = "Graph cloned successfully", body = CloneAgentResponse),
        (status = 400, description = "Invalid request", body = ApiErrorResponse),
        (status = 404, description = "Agent not found", body = ApiErrorResponse),
        (status = 403, description = "Forbidden", body = ApiErrorResponse),
        (status = 500, description = "Internal server error", body = ApiErrorResponse)
    )
)]
pub async fn clone_agent_handler(
    State(state): State<Arc<AppState>>,
    Extension(auth): Extension<AuthContext>,
    Path(id): Path<String>,
    Json(body): Json<CloneAgentRequest>,
) -> Result<Json<CloneAgentResponse>, AppError> {
    require_role(&auth, ucotron_config::AuthRole::Writer)?;
    if state.is_reader_only() {
        return Err(AppError::read_only());
    }

    // Verify source agent exists
    let agent = state
        .registry
        .graph()
        .get_agent(&id)
        .map_err(|e| AppError::internal(format!("Failed to get agent: {}", e)))?
        .ok_or_else(|| AppError::not_found(format!("Agent '{}' not found", id)))?;

    let src_ns = &agent.namespace;

    // Determine target namespace
    let dst_ns = body
        .target_namespace
        .as_deref()
        .filter(|s| !s.trim().is_empty())
        .map(|s| s.to_string())
        .unwrap_or_else(|| {
            let ts = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis();
            format!("{}_clone_{}", src_ns, ts % 1_000_000)
        });

    // Parse node_type filter
    let node_types: Option<Vec<ucotron_core::NodeType>> = body
        .node_types
        .as_ref()
        .map(|types| {
            types
                .iter()
                .filter_map(|s| parse_node_type(s))
                .collect::<Vec<_>>()
        })
        .filter(|v| !v.is_empty());

    let filter = ucotron_core::CloneFilter {
        node_types,
        time_range_start: body.time_range_start,
        time_range_end: body.time_range_end,
    };

    // Allocate IDs for the cloned nodes
    let id_start = state.alloc_next_node_id();

    let result = state
        .registry
        .graph()
        .clone_graph(src_ns, &dst_ns, &filter, id_start)
        .map_err(|e| AppError::internal(format!("Failed to clone graph: {}", e)))?;

    // Advance the shared ID counter past what was used
    if result.nodes_copied > 1 {
        let mut id_lock = state.next_node_id.lock().unwrap();
        let needed = id_start + result.nodes_copied as u64;
        if needed > *id_lock {
            *id_lock = needed;
        }
    }

    // Also upsert the cloned embeddings into the vector backend
    if result.nodes_copied > 0 {
        let emb_items: Vec<(u64, Vec<f32>)> = result
            .id_map
            .values()
            .filter_map(|&new_id| {
                state
                    .registry
                    .graph()
                    .get_node(new_id)
                    .ok()
                    .flatten()
                    .filter(|n| !n.embedding.is_empty())
                    .map(|n| (new_id, n.embedding.clone()))
            })
            .collect();
        if !emb_items.is_empty() {
            state
                .registry
                .vector()
                .upsert_embeddings(&emb_items)
                .map_err(|e| {
                    AppError::internal(format!("Failed to clone embeddings: {}", e))
                })?;
        }
    }

    Ok(Json(CloneAgentResponse {
        source_agent_id: id,
        source_namespace: src_ns.clone(),
        target_namespace: dst_ns,
        nodes_copied: result.nodes_copied,
        edges_copied: result.edges_copied,
    }))
}

// ---------------------------------------------------------------------------
// Agent Merge
// ---------------------------------------------------------------------------

/// Merge another agent's graph into this agent's namespace with entity deduplication.
///
/// Nodes from the source agent that have identical content to nodes already
/// in the target namespace are deduplicated — their edges are redirected to
/// the existing node. Non-duplicate nodes are copied with new IDs.
#[utoipa::path(
    post,
    path = "/api/v1/agents/{id}/merge",
    tag = "Agents",
    request_body = MergeAgentRequest,
    params(
        ("id" = String, Path, description = "Target agent ID (destination)")
    ),
    responses(
        (status = 200, description = "Merge completed", body = MergeAgentResponse),
        (status = 404, description = "Agent not found"),
    ),
    security(("api_key" = []))
)]
pub async fn merge_agent_handler(
    State(state): State<Arc<AppState>>,
    Extension(auth): Extension<AuthContext>,
    Path(id): Path<String>,
    Json(body): Json<MergeAgentRequest>,
) -> Result<Json<MergeAgentResponse>, AppError> {
    require_role(&auth, ucotron_config::AuthRole::Writer)?;
    if state.is_reader_only() {
        return Err(AppError::read_only());
    }

    // Verify target agent exists
    let target_agent = state
        .registry
        .graph()
        .get_agent(&id)
        .map_err(|e| AppError::internal(format!("Failed to get agent: {}", e)))?
        .ok_or_else(|| AppError::not_found(format!("Agent '{}' not found", id)))?;

    // Verify source agent exists
    let source_agent = state
        .registry
        .graph()
        .get_agent(&body.source_agent_id)
        .map_err(|e| AppError::internal(format!("Failed to get agent: {}", e)))?
        .ok_or_else(|| {
            AppError::not_found(format!("Source agent '{}' not found", body.source_agent_id))
        })?;

    let src_ns = &source_agent.namespace;
    let dst_ns = &target_agent.namespace;

    // Allocate starting ID for new nodes
    let id_start = state.alloc_next_node_id();

    let result = state
        .registry
        .graph()
        .merge_graph(src_ns, dst_ns, id_start)
        .map_err(|e| AppError::internal(format!("Failed to merge graph: {}", e)))?;

    // Advance the shared ID counter past what was used
    if result.nodes_copied > 0 {
        let mut id_lock = state.next_node_id.lock().unwrap();
        let needed = id_start + result.ids_remapped as u64;
        if needed > *id_lock {
            *id_lock = needed;
        }
    }

    // Sync embeddings for newly copied nodes to the vector backend
    if result.nodes_copied > 0 {
        // Fetch newly created nodes to get their embeddings
        let all_nodes = state
            .registry
            .graph()
            .get_all_nodes()
            .unwrap_or_default();
        let emb_items: Vec<(u64, Vec<f32>)> = all_nodes
            .iter()
            .filter(|n| {
                // Only nodes in the target namespace that were just created
                match n.metadata.get("_namespace") {
                    Some(ucotron_core::Value::String(ns)) => ns == dst_ns,
                    _ => false,
                }
            })
            .filter(|n| n.id >= id_start && n.id < id_start + result.ids_remapped as u64)
            .filter(|n| !n.embedding.is_empty())
            .map(|n| (n.id, n.embedding.clone()))
            .collect();
        if !emb_items.is_empty() {
            state
                .registry
                .vector()
                .upsert_embeddings(&emb_items)
                .map_err(|e| {
                    AppError::internal(format!("Failed to sync merged embeddings: {}", e))
                })?;
        }
    }

    Ok(Json(MergeAgentResponse {
        source_namespace: src_ns.clone(),
        target_namespace: dst_ns.clone(),
        nodes_copied: result.nodes_copied,
        edges_copied: result.edges_copied,
        nodes_deduplicated: result.nodes_deduplicated,
        ids_remapped: result.ids_remapped,
    }))
}

// ---------------------------------------------------------------------------
// Agent Share
// ---------------------------------------------------------------------------

/// Share an agent's memory namespace with another agent.
///
/// Creates a read-only or read-write share grant from the source agent
/// to the target agent. The target agent can then access memories in
/// the source agent's namespace according to the granted permission.
#[utoipa::path(
    post,
    path = "/api/v1/agents/{id}/share",
    tag = "Agents",
    request_body = CreateShareRequest,
    params(
        ("id" = String, Path, description = "Source agent ID (granting access)")
    ),
    responses(
        (status = 201, description = "Share created", body = CreateShareResponse),
        (status = 400, description = "Invalid request", body = ApiErrorResponse),
        (status = 404, description = "Agent not found", body = ApiErrorResponse),
        (status = 403, description = "Forbidden", body = ApiErrorResponse),
    ),
    security(("api_key" = []))
)]
pub async fn create_share_handler(
    State(state): State<Arc<AppState>>,
    Extension(auth): Extension<AuthContext>,
    Path(id): Path<String>,
    Json(body): Json<CreateShareRequest>,
) -> Result<(axum::http::StatusCode, Json<CreateShareResponse>), AppError> {
    require_role(&auth, ucotron_config::AuthRole::Writer)?;
    if state.is_reader_only() {
        return Err(AppError::read_only());
    }

    // Validate source agent exists
    let _agent = state
        .registry
        .graph()
        .get_agent(&id)
        .map_err(|e| AppError::internal(format!("Failed to get agent: {}", e)))?
        .ok_or_else(|| AppError::not_found(format!("Agent '{}' not found", id)))?;

    // Validate target agent exists
    let _target = state
        .registry
        .graph()
        .get_agent(&body.target_agent_id)
        .map_err(|e| AppError::internal(format!("Failed to get agent: {}", e)))?
        .ok_or_else(|| {
            AppError::not_found(format!(
                "Target agent '{}' not found",
                body.target_agent_id
            ))
        })?;

    // Cannot share with self
    if id == body.target_agent_id {
        return Err(AppError::bad_request("cannot share an agent with itself"));
    }

    // Parse permission
    let permission = match body.permission.as_str() {
        "read" | "read_only" => ucotron_core::SharePermission::ReadOnly,
        "read_write" | "readwrite" => ucotron_core::SharePermission::ReadWrite,
        _ => {
            return Err(AppError::bad_request(
                "permission must be 'read' or 'read_write'",
            ))
        }
    };

    let created_at = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let share = ucotron_core::AgentShare {
        agent_id: id.clone(),
        target_agent_id: body.target_agent_id.clone(),
        permissions: permission,
        created_at,
    };

    state
        .registry
        .graph()
        .create_share(&share)
        .map_err(|e| AppError::internal(format!("Failed to create share: {}", e)))?;

    let perm_str = match permission {
        ucotron_core::SharePermission::ReadOnly => "read",
        ucotron_core::SharePermission::ReadWrite => "read_write",
    };

    Ok((
        axum::http::StatusCode::CREATED,
        Json(CreateShareResponse {
            agent_id: id,
            target_agent_id: body.target_agent_id,
            permission: perm_str.to_string(),
            created_at,
        }),
    ))
}

/// List all share grants for a specific agent.
#[utoipa::path(
    get,
    path = "/api/v1/agents/{id}/share",
    tag = "Agents",
    params(
        ("id" = String, Path, description = "Agent ID to list shares for")
    ),
    responses(
        (status = 200, description = "List of shares", body = ListSharesResponse),
        (status = 404, description = "Agent not found", body = ApiErrorResponse),
        (status = 403, description = "Forbidden", body = ApiErrorResponse),
    ),
    security(("api_key" = []))
)]
pub async fn list_shares_handler(
    State(state): State<Arc<AppState>>,
    Extension(auth): Extension<AuthContext>,
    Path(id): Path<String>,
) -> Result<Json<ListSharesResponse>, AppError> {
    require_role(&auth, ucotron_config::AuthRole::Reader)?;

    // Verify agent exists
    let _agent = state
        .registry
        .graph()
        .get_agent(&id)
        .map_err(|e| AppError::internal(format!("Failed to get agent: {}", e)))?
        .ok_or_else(|| AppError::not_found(format!("Agent '{}' not found", id)))?;

    let shares = state
        .registry
        .graph()
        .list_shares(&id)
        .map_err(|e| AppError::internal(format!("Failed to list shares: {}", e)))?;

    let total = shares.len();
    let share_responses: Vec<ShareResponse> = shares
        .into_iter()
        .map(|s| {
            let perm_str = match s.permissions {
                ucotron_core::SharePermission::ReadOnly => "read",
                ucotron_core::SharePermission::ReadWrite => "read_write",
            };
            ShareResponse {
                agent_id: s.agent_id,
                target_agent_id: s.target_agent_id,
                permission: perm_str.to_string(),
                created_at: s.created_at,
            }
        })
        .collect();

    Ok(Json(ListSharesResponse {
        shares: share_responses,
        total,
    }))
}

/// Revoke a share grant from one agent to another.
///
/// Removes the share grant so the target agent can no longer access
/// the source agent's memory namespace.
#[utoipa::path(
    delete,
    path = "/api/v1/agents/{id}/share/{target}",
    tag = "Agents",
    params(
        ("id" = String, Path, description = "Source agent ID"),
        ("target" = String, Path, description = "Target agent ID to revoke access from")
    ),
    responses(
        (status = 204, description = "Share revoked"),
        (status = 404, description = "Agent or share not found", body = ApiErrorResponse),
        (status = 403, description = "Forbidden", body = ApiErrorResponse),
    ),
    security(("api_key" = []))
)]
pub async fn delete_share_handler(
    State(state): State<Arc<AppState>>,
    Extension(auth): Extension<AuthContext>,
    Path((id, target)): Path<(String, String)>,
) -> Result<axum::http::StatusCode, AppError> {
    require_role(&auth, ucotron_config::AuthRole::Writer)?;
    if state.is_reader_only() {
        return Err(AppError::read_only());
    }

    // Verify source agent exists
    let _agent = state
        .registry
        .graph()
        .get_agent(&id)
        .map_err(|e| AppError::internal(format!("Failed to get agent: {}", e)))?
        .ok_or_else(|| AppError::not_found(format!("Agent '{}' not found", id)))?;

    // Check that the share exists
    let _share = state
        .registry
        .graph()
        .get_share(&id, &target)
        .map_err(|e| AppError::internal(format!("Failed to get share: {}", e)))?
        .ok_or_else(|| {
            AppError::not_found(format!(
                "Share from '{}' to '{}' not found",
                id, target
            ))
        })?;

    state
        .registry
        .graph()
        .delete_share(&id, &target)
        .map_err(|e| AppError::internal(format!("Failed to delete share: {}", e)))?;

    Ok(axum::http::StatusCode::NO_CONTENT)
}

// ---------------------------------------------------------------------------
// Multimodal Memory Ingestion — Text
// ---------------------------------------------------------------------------

/// Ingest text through the extraction pipeline and tag the resulting nodes
/// with `media_type = Text`. This is the dedicated multimodal text endpoint.
#[utoipa::path(
    post,
    path = "/api/v1/memories/text",
    tag = "Memories",
    request_body = CreateTextMemoryRequest,
    params(
        ("X-Ucotron-Namespace" = Option<String>, Header, description = "Namespace for multi-tenancy isolation")
    ),
    responses(
        (status = 201, description = "Text memory created successfully", body = CreateTextMemoryResponse),
        (status = 400, description = "Invalid request", body = ApiErrorResponse),
        (status = 403, description = "Read-only instance", body = ApiErrorResponse),
        (status = 500, description = "Internal server error", body = ApiErrorResponse)
    )
)]
pub async fn create_text_memory_handler(
    State(state): State<Arc<AppState>>,
    Extension(auth): Extension<AuthContext>,
    headers: HeaderMap,
    Json(body): Json<CreateTextMemoryRequest>,
) -> Result<(axum::http::StatusCode, Json<CreateTextMemoryResponse>), AppError> {
    require_role(&auth, ucotron_config::AuthRole::Writer)?;
    let ns = extract_namespace(&headers);
    require_namespace_access(&auth, &ns)?;
    if state.is_reader_only() {
        return Err(AppError::read_only());
    }

    state.total_ingestions.fetch_add(1, Ordering::Relaxed);
    record_ingestion(&state);

    if body.text.trim().is_empty() {
        return Err(AppError::bad_request("text must not be empty"));
    }

    let next_id = state.alloc_next_node_id();
    let config = IngestionConfig {
        next_node_id: Some(next_id),
        ..IngestionConfig::default()
    };

    let ner_ref: Option<&dyn ucotron_extraction::NerPipeline> =
        state.ner.as_ref().map(|n| n.as_ref());
    let re_ref: Option<&dyn ucotron_extraction::RelationExtractor> =
        state.relation_extractor.as_ref().map(|r| r.as_ref());

    let mut orchestrator = IngestionOrchestrator::new(
        &state.registry,
        state.embedder.as_ref(),
        ner_ref,
        re_ref,
        config,
    );

    let result = orchestrator
        .ingest(&body.text)
        .map_err(|e| AppError::internal(format!("Ingestion failed: {}", e)))?;

    // Advance the shared ID counter past what was used.
    let ids_used = result.chunk_node_ids.len() + result.entity_node_ids.len();
    {
        let mut id_lock = state.next_node_id.lock().unwrap();
        let used_max = next_id + ids_used as u64;
        if used_max > *id_lock {
            *id_lock = used_max;
        }
    }

    // Tag all created nodes with the namespace and media_type=Text.
    let namespace = extract_namespace(&headers);
    tag_nodes_with_namespace(&state, &result.chunk_node_ids, &namespace);
    tag_nodes_with_namespace(&state, &result.entity_node_ids, &namespace);

    // Set media_type on chunk nodes to Text.
    for &node_id in &result.chunk_node_ids {
        if let Ok(Some(mut node)) = state.registry.graph().get_node(node_id) {
            node.media_type = Some(ucotron_core::MediaType::Text);
            let _ = state.registry.graph().upsert_nodes(&[node]);
        }
    }

    let response = CreateTextMemoryResponse {
        chunk_node_ids: result.chunk_node_ids,
        entity_node_ids: result.entity_node_ids,
        edges_created: result.edges_created.len(),
        media_type: "Text".to_string(),
        metrics: IngestionMetricsResponse {
            chunks_processed: result.metrics.chunks_processed,
            entities_extracted: result.metrics.entities_extracted,
            relations_extracted: result.metrics.relations_extracted,
            contradictions_detected: result.metrics.contradictions_detected,
            total_us: result.metrics.total_us,
        },
    };

    Ok((axum::http::StatusCode::CREATED, Json(response)))
}

// ---------------------------------------------------------------------------
// Multimodal Memory Ingestion — Audio
// ---------------------------------------------------------------------------

/// Ingest audio through Whisper transcription and the extraction pipeline.
///
/// Accepts `multipart/form-data` with:
/// - `file` or `audio`: the audio file (WAV format, 16 kHz mono recommended)
///
/// The audio is transcribed via the Whisper ONNX pipeline, the resulting text
/// is run through the full ingestion pipeline (embedding, NER, relations),
/// and all created nodes are tagged with `media_type = Audio`.
#[utoipa::path(
    post,
    path = "/api/v1/memories/audio",
    tag = "Memories",
    request_body(content_type = "multipart/form-data", description = "Audio file (WAV, 16 kHz mono)"),
    params(
        ("X-Ucotron-Namespace" = Option<String>, Header, description = "Namespace for multi-tenancy isolation")
    ),
    responses(
        (status = 201, description = "Audio memory created successfully", body = CreateAudioMemoryResponse),
        (status = 400, description = "Invalid request (no file or empty)", body = ApiErrorResponse),
        (status = 403, description = "Read-only instance", body = ApiErrorResponse),
        (status = 500, description = "Transcription or ingestion failed", body = ApiErrorResponse),
        (status = 501, description = "Transcription not available", body = ApiErrorResponse)
    )
)]
pub async fn create_audio_memory_handler(
    State(state): State<Arc<AppState>>,
    Extension(auth): Extension<AuthContext>,
    headers: HeaderMap,
    mut multipart: Multipart,
) -> Result<(axum::http::StatusCode, Json<CreateAudioMemoryResponse>), AppError> {
    require_role(&auth, ucotron_config::AuthRole::Writer)?;
    let ns = extract_namespace(&headers);
    require_namespace_access(&auth, &ns)?;
    if state.is_reader_only() {
        return Err(AppError::read_only());
    }

    let transcriber = state
        .transcriber
        .as_ref()
        .ok_or_else(|| {
            AppError::not_implemented(
                "Audio transcription not available — Whisper model not loaded",
            )
        })?;

    // Extract audio file from multipart form data.
    let mut audio_data: Option<Vec<u8>> = None;

    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|e| AppError::bad_request(format!("Failed to read multipart field: {}", e)))?
    {
        let name = field.name().unwrap_or("").to_string();
        match name.as_str() {
            "file" | "audio" => {
                let data = field
                    .bytes()
                    .await
                    .map_err(|e| {
                        AppError::bad_request(format!("Failed to read file data: {}", e))
                    })?;
                audio_data = Some(data.to_vec());
            }
            _ => {}
        }
    }

    let audio_bytes = audio_data.ok_or_else(|| {
        AppError::bad_request("Missing 'file' or 'audio' field in multipart form data")
    })?;

    if audio_bytes.is_empty() {
        return Err(AppError::bad_request("Audio file is empty"));
    }

    // Write to temp file for WAV parsing (hound requires seekable reader).
    let temp_dir = tempfile::tempdir()
        .map_err(|e| AppError::internal(format!("Failed to create temp directory: {}", e)))?;
    let temp_path = temp_dir.path().join("upload.wav");
    std::fs::write(&temp_path, &audio_bytes)
        .map_err(|e| AppError::internal(format!("Failed to write temp file: {}", e)))?;

    // Transcribe audio.
    let transcription = transcriber
        .transcribe_file(&temp_path)
        .map_err(|e| AppError::internal(format!("Transcription failed: {}", e)))?;

    if transcription.text.trim().is_empty() {
        return Err(AppError::bad_request(
            "Transcription produced no text — audio may be silent or unrecognizable",
        ));
    }

    // Ingest the transcribed text through the full pipeline.
    state.total_ingestions.fetch_add(1, Ordering::Relaxed);
    record_ingestion(&state);

    let next_id = state.alloc_next_node_id();
    let config = IngestionConfig {
        next_node_id: Some(next_id),
        ..IngestionConfig::default()
    };

    let ner_ref: Option<&dyn ucotron_extraction::NerPipeline> =
        state.ner.as_ref().map(|n| n.as_ref());
    let re_ref: Option<&dyn ucotron_extraction::RelationExtractor> =
        state.relation_extractor.as_ref().map(|r| r.as_ref());

    let mut orchestrator = IngestionOrchestrator::new(
        &state.registry,
        state.embedder.as_ref(),
        ner_ref,
        re_ref,
        config,
    );

    let result = orchestrator
        .ingest(&transcription.text)
        .map_err(|e| AppError::internal(format!("Ingestion failed: {}", e)))?;

    // Advance the shared ID counter past what was used.
    let ids_used = result.chunk_node_ids.len() + result.entity_node_ids.len();
    {
        let mut id_lock = state.next_node_id.lock().unwrap();
        let used_max = next_id + ids_used as u64;
        if used_max > *id_lock {
            *id_lock = used_max;
        }
    }

    // Tag all created nodes with the namespace and media_type=Audio.
    let namespace = extract_namespace(&headers);
    tag_nodes_with_namespace(&state, &result.chunk_node_ids, &namespace);
    tag_nodes_with_namespace(&state, &result.entity_node_ids, &namespace);

    // Persist audio file and set media_type on chunk nodes.
    let audio_file_uri = persist_media_file(&state, next_id, "wav", &audio_bytes).ok();
    for &node_id in &result.chunk_node_ids {
        if let Ok(Some(mut node)) = state.registry.graph().get_node(node_id) {
            node.media_type = Some(ucotron_core::MediaType::Audio);
            node.media_uri = audio_file_uri.clone();
            node.metadata.insert(
                "audio_duration_secs".into(),
                ucotron_core::Value::Float(transcription.metadata.duration_secs as f64),
            );
            if let Some(ref lang) = transcription.metadata.detected_language {
                node.metadata.insert(
                    "audio_language".into(),
                    ucotron_core::Value::String(lang.clone()),
                );
            }
            let _ = state.registry.graph().upsert_nodes(&[node]);
        }
    }

    let audio_meta = AudioMetadataResponse {
        duration_secs: transcription.metadata.duration_secs,
        sample_rate: transcription.metadata.sample_rate,
        channels: transcription.metadata.channels,
        detected_language: transcription.metadata.detected_language.clone(),
    };

    let response = CreateAudioMemoryResponse {
        chunk_node_ids: result.chunk_node_ids,
        entity_node_ids: result.entity_node_ids,
        edges_created: result.edges_created.len(),
        media_type: "Audio".to_string(),
        transcription: transcription.text,
        audio: audio_meta,
        metrics: IngestionMetricsResponse {
            chunks_processed: result.metrics.chunks_processed,
            entities_extracted: result.metrics.entities_extracted,
            relations_extracted: result.metrics.relations_extracted,
            contradictions_detected: result.metrics.contradictions_detected,
            total_us: result.metrics.total_us,
        },
    };

    Ok((axum::http::StatusCode::CREATED, Json(response)))
}

/// Ingest an image through CLIP encoding and store in the visual index.
///
/// Accepts `multipart/form-data` with:
/// - `file` or `image`: the image file (JPEG, PNG, etc.)
/// - `description` (optional): text description to also ingest into the text index
///
/// The image is encoded via the CLIP visual pipeline (512-dim embedding) and
/// stored in both the graph backend and the visual vector index. If a description
/// is provided, it is run through the full ingestion pipeline (embedding, NER,
/// relations) so the image can also be found via text search.
#[utoipa::path(
    post,
    path = "/api/v1/memories/image",
    tag = "Memories",
    request_body(content_type = "multipart/form-data", description = "Image file (JPEG, PNG) and optional description"),
    params(
        ("X-Ucotron-Namespace" = Option<String>, Header, description = "Namespace for multi-tenancy isolation")
    ),
    responses(
        (status = 201, description = "Image memory created successfully", body = CreateImageMemoryResponse),
        (status = 400, description = "Invalid request (no file or empty)", body = ApiErrorResponse),
        (status = 403, description = "Read-only instance", body = ApiErrorResponse),
        (status = 500, description = "Image processing or ingestion failed", body = ApiErrorResponse),
        (status = 501, description = "Image embedding not available", body = ApiErrorResponse)
    )
)]
pub async fn create_image_memory_handler(
    State(state): State<Arc<AppState>>,
    Extension(auth): Extension<AuthContext>,
    headers: HeaderMap,
    mut multipart: Multipart,
) -> Result<(axum::http::StatusCode, Json<CreateImageMemoryResponse>), AppError> {
    require_role(&auth, ucotron_config::AuthRole::Writer)?;
    let ns = extract_namespace(&headers);
    require_namespace_access(&auth, &ns)?;
    if state.is_reader_only() {
        return Err(AppError::read_only());
    }

    let image_embedder = state
        .image_embedder
        .as_ref()
        .ok_or_else(|| {
            AppError::not_implemented("Image embedding not available — CLIP model not loaded")
        })?;

    // Extract multipart fields: image file + optional description.
    let mut image_data: Option<Vec<u8>> = None;
    let mut description = String::new();

    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|e| AppError::bad_request(format!("Failed to read multipart field: {}", e)))?
    {
        let name = field.name().unwrap_or("").to_string();
        match name.as_str() {
            "file" | "image" => {
                let data = field
                    .bytes()
                    .await
                    .map_err(|e| {
                        AppError::bad_request(format!("Failed to read image data: {}", e))
                    })?;
                image_data = Some(data.to_vec());
            }
            "description" => {
                description = field
                    .text()
                    .await
                    .map_err(|e| {
                        AppError::bad_request(format!("Failed to read description: {}", e))
                    })?;
            }
            _ => {}
        }
    }

    let image_bytes = image_data.ok_or_else(|| {
        AppError::bad_request("Missing 'file' or 'image' field in multipart form data")
    })?;

    if image_bytes.is_empty() {
        return Err(AppError::bad_request("Image file is empty"));
    }

    // Detect image format and dimensions.
    let img = image::load_from_memory(&image_bytes)
        .map_err(|e| AppError::bad_request(format!("Failed to decode image: {}", e)))?;
    let (width, height) = (img.width(), img.height());
    let format = image::guess_format(&image_bytes)
        .map(|f| format!("{:?}", f).to_lowercase())
        .unwrap_or_else(|_| "unknown".to_string());

    // Generate CLIP embedding (512-dim).
    let embedding = image_embedder
        .embed_image_bytes(&image_bytes)
        .map_err(|e| AppError::internal(format!("Image embedding failed: {}", e)))?;
    let embed_dim = embedding.len();

    // Create node for this image.
    let node_id = state.alloc_next_node_id();
    let namespace = extract_namespace(&headers);
    let content = if description.is_empty() {
        format!("[image:{}x{} {}]", width, height, format)
    } else {
        description.clone()
    };

    let mut metadata = std::collections::HashMap::new();
    metadata.insert(
        "_namespace".into(),
        ucotron_core::Value::String(namespace.clone()),
    );
    metadata.insert(
        "_media_type".into(),
        ucotron_core::Value::String("image".to_string()),
    );
    metadata.insert(
        "_image_format".into(),
        ucotron_core::Value::String(format.clone()),
    );
    metadata.insert(
        "_image_width".into(),
        ucotron_core::Value::Integer(width as i64),
    );
    metadata.insert(
        "_image_height".into(),
        ucotron_core::Value::Integer(height as i64),
    );

    let node = ucotron_core::Node {
        id: node_id,
        content,
        embedding: embedding.clone(),
        metadata,
        node_type: ucotron_core::NodeType::Entity,
        timestamp: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs(),
        media_type: Some(ucotron_core::MediaType::Image),
        media_uri: persist_media_file(&state, node_id, &format, &image_bytes).ok(),
        embedding_visual: None,
        timestamp_range: None,
        parent_video_id: None,
    };

    // Store in graph backend.
    state
        .registry
        .graph()
        .upsert_nodes(&[node])
        .map_err(|e| AppError::internal(format!("Failed to store image node: {}", e)))?;

    // Store CLIP embedding in visual index (if available), otherwise in text vector index.
    if let Some(visual) = state.registry.visual() {
        visual
            .upsert_visual_embeddings(&[(node_id, embedding)])
            .map_err(|e| {
                AppError::internal(format!("Failed to store visual embedding: {}", e))
            })?;
    } else {
        state
            .registry
            .vector()
            .upsert_embeddings(&[(node_id, embedding)])
            .map_err(|e| {
                AppError::internal(format!("Failed to store image embedding: {}", e))
            })?;
    }

    state.total_ingestions.fetch_add(1, Ordering::Relaxed);
    record_ingestion(&state);

    // If a description was provided, also ingest it through the text pipeline.
    let mut description_ingested = false;
    let mut metrics_response = None;

    if !description.is_empty() {
        let next_id = state.alloc_next_node_id();
        let config = IngestionConfig {
            next_node_id: Some(next_id),
            ..IngestionConfig::default()
        };

        let ner_ref: Option<&dyn ucotron_extraction::NerPipeline> =
            state.ner.as_ref().map(|n| n.as_ref());
        let re_ref: Option<&dyn ucotron_extraction::RelationExtractor> =
            state.relation_extractor.as_ref().map(|r| r.as_ref());

        let mut orchestrator = IngestionOrchestrator::new(
            &state.registry,
            state.embedder.as_ref(),
            ner_ref,
            re_ref,
            config,
        );

        if let Ok(result) = orchestrator.ingest(&description) {
            // Advance ID counter.
            let ids_used = result.chunk_node_ids.len() + result.entity_node_ids.len();
            {
                let mut id_lock = state.next_node_id.lock().unwrap();
                let used_max = next_id + ids_used as u64;
                if used_max > *id_lock {
                    *id_lock = used_max;
                }
            }

            // Tag description nodes with namespace.
            tag_nodes_with_namespace(&state, &result.chunk_node_ids, &namespace);
            tag_nodes_with_namespace(&state, &result.entity_node_ids, &namespace);

            description_ingested = true;
            metrics_response = Some(IngestionMetricsResponse {
                chunks_processed: result.metrics.chunks_processed,
                entities_extracted: result.metrics.entities_extracted,
                relations_extracted: result.metrics.relations_extracted,
                contradictions_detected: result.metrics.contradictions_detected,
                total_us: result.metrics.total_us,
            });
        }
    }

    let response = CreateImageMemoryResponse {
        node_id,
        width,
        height,
        format,
        embedding_dim: embed_dim,
        media_type: "Image".to_string(),
        description_ingested,
        metrics: metrics_response,
    };

    Ok((axum::http::StatusCode::CREATED, Json(response)))
}

/// Ingest a video through frame extraction, temporal segmentation, and dual embedding.
///
/// Accepts `multipart/form-data` with:
/// - `file` or `video`: the video file (MP4, AVI, MOV, etc.)
///
/// The video is processed through the full pipeline:
/// 1. Frame extraction via FFmpeg at configured FPS
/// 2. Temporal segmentation into scene-based groups
/// 3. CLIP embedding of a representative frame per segment
/// 4. Optional Whisper transcription of audio per segment
/// 5. Each segment stored as a separate VideoSegment node linked to a parent video node
#[utoipa::path(
    post,
    path = "/api/v1/memories/video",
    tag = "Memories",
    request_body(content_type = "multipart/form-data", description = "Video file (MP4, AVI, MOV, etc.)"),
    params(
        ("X-Ucotron-Namespace" = Option<String>, Header, description = "Namespace for multi-tenancy isolation")
    ),
    responses(
        (status = 201, description = "Video memory created successfully", body = CreateVideoMemoryResponse),
        (status = 400, description = "Invalid request (no file or empty)", body = ApiErrorResponse),
        (status = 403, description = "Read-only instance", body = ApiErrorResponse),
        (status = 500, description = "Video processing or ingestion failed", body = ApiErrorResponse),
        (status = 501, description = "Video pipeline or image embedder not available", body = ApiErrorResponse)
    )
)]
pub async fn create_video_memory_handler(
    State(state): State<Arc<AppState>>,
    Extension(auth): Extension<AuthContext>,
    headers: HeaderMap,
    mut multipart: Multipart,
) -> Result<(axum::http::StatusCode, Json<CreateVideoMemoryResponse>), AppError> {
    require_role(&auth, ucotron_config::AuthRole::Writer)?;
    let ns = extract_namespace(&headers);
    require_namespace_access(&auth, &ns)?;
    if state.is_reader_only() {
        return Err(AppError::read_only());
    }

    // Require both video pipeline and image embedder.
    let video_pipeline = state
        .video_pipeline
        .as_ref()
        .ok_or_else(|| {
            AppError::not_implemented("Video pipeline not available — FFmpeg pipeline not loaded")
        })?;

    let image_embedder = state
        .image_embedder
        .as_ref()
        .ok_or_else(|| {
            AppError::not_implemented(
                "Image embedding not available — CLIP model not loaded (required for video)",
            )
        })?;

    // Extract video file from multipart form data.
    let mut video_data: Option<Vec<u8>> = None;

    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|e| AppError::bad_request(format!("Failed to read multipart field: {}", e)))?
    {
        let name = field.name().unwrap_or("").to_string();
        match name.as_str() {
            "file" | "video" => {
                let data = field
                    .bytes()
                    .await
                    .map_err(|e| {
                        AppError::bad_request(format!("Failed to read video data: {}", e))
                    })?;
                video_data = Some(data.to_vec());
            }
            _ => {}
        }
    }

    let video_bytes = video_data.ok_or_else(|| {
        AppError::bad_request("Missing 'file' or 'video' field in multipart form data")
    })?;

    if video_bytes.is_empty() {
        return Err(AppError::bad_request("Video file is empty"));
    }

    // Write to temp file for FFmpeg processing.
    let temp_dir = tempfile::tempdir()
        .map_err(|e| AppError::internal(format!("Failed to create temp directory: {}", e)))?;
    let temp_path = temp_dir.path().join("upload.mp4");
    std::fs::write(&temp_path, &video_bytes)
        .map_err(|e| AppError::internal(format!("Failed to write temp file: {}", e)))?;

    // Step 1: Extract frames from video.
    let extraction_result = video_pipeline
        .extract_frames(&temp_path)
        .map_err(|e| AppError::internal(format!("Frame extraction failed: {}", e)))?;

    if extraction_result.frames.is_empty() {
        return Err(AppError::bad_request(
            "No frames could be extracted from the video",
        ));
    }

    // Step 2: Temporal segmentation.
    let segment_config = ucotron_extraction::video::SegmentConfig::default();
    let segments =
        ucotron_extraction::video::segment_frames(&extraction_result.frames, &segment_config);

    if segments.is_empty() {
        return Err(AppError::internal("Segmentation produced no segments"));
    }

    // Step 3: Create parent video node.
    let namespace = extract_namespace(&headers);
    let parent_id = state.alloc_next_node_id();
    let now_ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let mut parent_metadata = std::collections::HashMap::new();
    parent_metadata.insert(
        "_namespace".into(),
        ucotron_core::Value::String(namespace.clone()),
    );
    parent_metadata.insert(
        "_media_type".into(),
        ucotron_core::Value::String("video".to_string()),
    );
    parent_metadata.insert(
        "_video_width".into(),
        ucotron_core::Value::Integer(extraction_result.video_width as i64),
    );
    parent_metadata.insert(
        "_video_height".into(),
        ucotron_core::Value::Integer(extraction_result.video_height as i64),
    );
    parent_metadata.insert(
        "_video_duration_ms".into(),
        ucotron_core::Value::Integer(extraction_result.duration_ms as i64),
    );
    parent_metadata.insert(
        "_video_fps".into(),
        ucotron_core::Value::Float(extraction_result.video_fps),
    );
    parent_metadata.insert(
        "_total_frames".into(),
        ucotron_core::Value::Integer(extraction_result.frames.len() as i64),
    );
    parent_metadata.insert(
        "_total_segments".into(),
        ucotron_core::Value::Integer(segments.len() as i64),
    );

    // Use embedding from the first frame for the parent node.
    let first_frame = &extraction_result.frames[0];
    let parent_embedding = embed_frame_rgb(image_embedder.as_ref(), first_frame)?;

    let parent_content = format!(
        "[video:{}x{} {:.1}s {} segments]",
        extraction_result.video_width,
        extraction_result.video_height,
        extraction_result.duration_ms as f64 / 1000.0,
        segments.len()
    );

    // Persist video file for later retrieval.
    let video_file_uri = persist_media_file(&state, parent_id, "mp4", &video_bytes).ok();

    let parent_node = ucotron_core::Node {
        id: parent_id,
        content: parent_content,
        embedding: parent_embedding.clone(),
        metadata: parent_metadata,
        node_type: ucotron_core::NodeType::Event,
        timestamp: now_ts,
        media_type: Some(ucotron_core::MediaType::VideoSegment),
        media_uri: video_file_uri.clone(),
        embedding_visual: None,
        timestamp_range: Some((0, extraction_result.duration_ms)),
        parent_video_id: None,
    };

    state
        .registry
        .graph()
        .upsert_nodes(&[parent_node])
        .map_err(|e| AppError::internal(format!("Failed to store parent video node: {}", e)))?;

    // Store parent embedding in visual index (or vector index).
    if let Some(visual) = state.registry.visual() {
        visual
            .upsert_visual_embeddings(&[(parent_id, parent_embedding)])
            .map_err(|e| {
                AppError::internal(format!("Failed to store parent visual embedding: {}", e))
            })?;
    } else {
        state
            .registry
            .vector()
            .upsert_embeddings(&[(parent_id, parent_embedding)])
            .map_err(|e| {
                AppError::internal(format!("Failed to store parent embedding: {}", e))
            })?;
    }

    // Step 4: Create segment nodes with CLIP embeddings.
    let mut segment_node_ids = Vec::with_capacity(segments.len());
    let mut all_edges = Vec::new();
    let mut segment_infos = Vec::with_capacity(segments.len());

    for segment in &segments {
        let seg_id = state.alloc_next_node_id();
        segment_node_ids.push(seg_id);

        // Pick representative frame: first keyframe in segment, or first frame.
        let rep_frame_idx = segment
            .frame_indices
            .iter()
            .find(|&&idx| extraction_result.frames[idx].is_keyframe)
            .copied()
            .unwrap_or(segment.frame_indices[0]);
        let rep_frame = &extraction_result.frames[rep_frame_idx];

        // CLIP embedding of the representative frame.
        let visual_embedding = embed_frame_rgb(image_embedder.as_ref(), rep_frame)?;

        let seg_content = format!(
            "[video_segment:{} {:.1}s-{:.1}s {} frames]",
            segment.index,
            segment.start_ms as f64 / 1000.0,
            segment.end_ms as f64 / 1000.0,
            segment.frame_count()
        );

        let mut seg_metadata = std::collections::HashMap::new();
        seg_metadata.insert(
            "_namespace".into(),
            ucotron_core::Value::String(namespace.clone()),
        );
        seg_metadata.insert(
            "_media_type".into(),
            ucotron_core::Value::String("video_segment".to_string()),
        );
        seg_metadata.insert(
            "_segment_index".into(),
            ucotron_core::Value::Integer(segment.index as i64),
        );
        seg_metadata.insert(
            "_frame_count".into(),
            ucotron_core::Value::Integer(segment.frame_count() as i64),
        );
        seg_metadata.insert(
            "_is_scene_change".into(),
            ucotron_core::Value::Bool(segment.is_scene_change),
        );

        let seg_node = ucotron_core::Node {
            id: seg_id,
            content: seg_content,
            embedding: visual_embedding.clone(),
            metadata: seg_metadata,
            node_type: ucotron_core::NodeType::Event,
            timestamp: now_ts,
            media_type: Some(ucotron_core::MediaType::VideoSegment),
            media_uri: video_file_uri.clone(),
            embedding_visual: Some(visual_embedding.clone()),
            timestamp_range: Some((segment.start_ms, segment.end_ms)),
            parent_video_id: Some(parent_id),
        };

        state
            .registry
            .graph()
            .upsert_nodes(&[seg_node])
            .map_err(|e| {
                AppError::internal(format!("Failed to store segment node: {}", e))
            })?;

        // Store visual embedding.
        if let Some(visual) = state.registry.visual() {
            visual
                .upsert_visual_embeddings(&[(seg_id, visual_embedding)])
                .map_err(|e| {
                    AppError::internal(format!("Failed to store segment visual embedding: {}", e))
                })?;
        } else {
            state
                .registry
                .vector()
                .upsert_embeddings(&[(seg_id, visual_embedding)])
                .map_err(|e| {
                    AppError::internal(format!("Failed to store segment embedding: {}", e))
                })?;
        }

        // Create edge from parent → segment.
        let edge = ucotron_core::Edge {
            source: parent_id,
            target: seg_id,
            edge_type: ucotron_core::EdgeType::RelatesTo,
            weight: 1.0,
            metadata: std::collections::HashMap::new(),
        };
        all_edges.push(edge);

        segment_infos.push(VideoSegmentInfo {
            node_id: seg_id,
            start_ms: segment.start_ms,
            end_ms: segment.end_ms,
            frame_count: segment.frame_count(),
            is_scene_change: segment.is_scene_change,
        });
    }

    // Store all parent→segment edges.
    state
        .registry
        .graph()
        .upsert_edges(&all_edges)
        .map_err(|e| AppError::internal(format!("Failed to store edges: {}", e)))?;

    // Step 5: Optional audio transcription per segment via Whisper.
    let mut transcription_metrics = None;
    if let Some(transcriber) = state.transcriber.as_ref() {
        // Try to transcribe the full video audio.
        if let Ok(transcription) = transcriber.transcribe_file(&temp_path) {
            if !transcription.text.trim().is_empty() {
                let next_id = state.alloc_next_node_id();
                let config = IngestionConfig {
                    next_node_id: Some(next_id),
                    ..IngestionConfig::default()
                };

                let ner_ref: Option<&dyn ucotron_extraction::NerPipeline> =
                    state.ner.as_ref().map(|n| n.as_ref());
                let re_ref: Option<&dyn ucotron_extraction::RelationExtractor> =
                    state.relation_extractor.as_ref().map(|r| r.as_ref());

                let mut orchestrator = IngestionOrchestrator::new(
                    &state.registry,
                    state.embedder.as_ref(),
                    ner_ref,
                    re_ref,
                    config,
                );

                if let Ok(result) = orchestrator.ingest(&transcription.text) {
                    let ids_used = result.chunk_node_ids.len() + result.entity_node_ids.len();
                    {
                        let mut id_lock = state.next_node_id.lock().unwrap();
                        let used_max = next_id + ids_used as u64;
                        if used_max > *id_lock {
                            *id_lock = used_max;
                        }
                    }

                    tag_nodes_with_namespace(&state, &result.chunk_node_ids, &namespace);
                    tag_nodes_with_namespace(&state, &result.entity_node_ids, &namespace);

                    transcription_metrics = Some(IngestionMetricsResponse {
                        chunks_processed: result.metrics.chunks_processed,
                        entities_extracted: result.metrics.entities_extracted,
                        relations_extracted: result.metrics.relations_extracted,
                        contradictions_detected: result.metrics.contradictions_detected,
                        total_us: result.metrics.total_us,
                    });
                }
            }
        }
    }

    // Advance ID counter for all allocated segment IDs.
    {
        let mut id_lock = state.next_node_id.lock().unwrap();
        // parent_id + segment_node_ids were already allocated via alloc_next_node_id
        // but ensure the counter is past the max used.
        let max_used = segment_node_ids
            .last()
            .copied()
            .unwrap_or(parent_id)
            + 1;
        if max_used > *id_lock {
            *id_lock = max_used;
        }
    }

    // Tag parent and segment nodes with namespace.
    tag_nodes_with_namespace(&state, &[parent_id], &namespace);
    tag_nodes_with_namespace(&state, &segment_node_ids, &namespace);

    state.total_ingestions.fetch_add(1, Ordering::Relaxed);
    record_ingestion(&state);

    let response = CreateVideoMemoryResponse {
        video_node_id: parent_id,
        segment_node_ids,
        edges_created: all_edges.len(),
        total_frames: extraction_result.frames.len(),
        total_segments: segments.len(),
        duration_ms: extraction_result.duration_ms,
        video_width: extraction_result.video_width,
        video_height: extraction_result.video_height,
        media_type: "VideoSegment".to_string(),
        segments: segment_infos,
        transcription_metrics,
    };

    Ok((axum::http::StatusCode::CREATED, Json(response)))
}

// ---------------------------------------------------------------------------
// Multimodal Search
// ---------------------------------------------------------------------------

/// Unified cross-modal search endpoint dispatching to the [`CrossModalSearch`]
/// orchestrator based on the requested query type.
///
/// Supported query types:
/// - `"text"` — Standard text search in the 384-dim MiniLM index
/// - `"text_to_image"` — Text query → CLIP text encoder → visual 512-dim index
/// - `"image"` — Image bytes → CLIP image encoder → visual 512-dim index
/// - `"image_to_text"` — Image bytes → CLIP image encoder → projection → text 384-dim index
/// - `"audio"` — Audio transcript text → text 384-dim index
#[utoipa::path(
    post,
    path = "/api/v1/search/multimodal",
    tag = "Multimodal",
    request_body = MultimodalSearchRequest,
    params(
        ("X-Ucotron-Namespace" = Option<String>, Header, description = "Namespace for multi-tenancy isolation")
    ),
    responses(
        (status = 200, description = "Multimodal search results", body = MultimodalSearchResponse),
        (status = 400, description = "Invalid request", body = ApiErrorResponse),
        (status = 501, description = "Required pipeline not loaded", body = ApiErrorResponse)
    )
)]
pub async fn multimodal_search_handler(
    State(state): State<Arc<AppState>>,
    Extension(auth): Extension<AuthContext>,
    headers: HeaderMap,
    Json(req): Json<MultimodalSearchRequest>,
) -> Result<Json<MultimodalSearchResponse>, AppError> {
    use ucotron_extraction::cross_modal_search::{
        CrossModalQuery, CrossModalSearch, CrossModalSearchConfig, ResultSource,
    };

    require_role(&auth, ucotron_config::AuthRole::Reader)?;
    let namespace = extract_namespace(&headers);
    require_namespace_access(&auth, &namespace)?;

    let limit = req.limit.unwrap_or(10);
    let config = CrossModalSearchConfig {
        top_k: limit,
        ..CrossModalSearchConfig::default()
    };

    // Build the orchestrator from available pipelines in state.
    let mut searcher = CrossModalSearch::new(
        &state.registry,
        state.embedder.as_ref(),
        config,
    );

    if let Some(ref img_emb) = state.image_embedder {
        searcher = searcher.with_image_embedder(img_emb.as_ref());
    }
    if let Some(ref clip_enc) = state.cross_modal_encoder {
        searcher = searcher.with_clip_text_encoder(clip_enc.as_ref());
    }

    // Decode image bytes if provided.
    let image_bytes = if let Some(ref b64) = req.query_image {
        use base64::Engine;
        let decoded = base64::engine::general_purpose::STANDARD
            .decode(b64)
            .map_err(|e| AppError::bad_request(format!("Invalid base64 image: {}", e)))?;
        Some(decoded)
    } else {
        None
    };

    // Build the cross-modal query based on query_type.
    let query_type_lower = req.query_type.to_lowercase();
    let response = match query_type_lower.as_str() {
        "text" => {
            let text = req.query_text.as_deref().ok_or_else(|| {
                AppError::bad_request("query_text is required for query_type 'text'")
            })?;
            searcher.search(&CrossModalQuery::Text { text })
        }
        "text_to_image" => {
            let text = req.query_text.as_deref().ok_or_else(|| {
                AppError::bad_request("query_text is required for query_type 'text_to_image'")
            })?;
            searcher.search(&CrossModalQuery::TextToImage { text })
        }
        "image" => {
            let bytes = image_bytes.as_deref().ok_or_else(|| {
                AppError::bad_request("query_image is required for query_type 'image'")
            })?;
            searcher.search(&CrossModalQuery::Image { image_bytes: bytes })
        }
        "image_to_text" => {
            let bytes = image_bytes.as_deref().ok_or_else(|| {
                AppError::bad_request("query_image is required for query_type 'image_to_text'")
            })?;
            searcher.search(&CrossModalQuery::ImageToText { image_bytes: bytes })
        }
        "audio" => {
            let text = req.query_text.as_deref().ok_or_else(|| {
                AppError::bad_request("query_text is required for query_type 'audio'")
            })?;
            searcher.search(&CrossModalQuery::Audio { transcript: text })
        }
        other => {
            return Err(AppError::bad_request(format!(
                "Unsupported query_type '{}'. Supported: text, text_to_image, image, image_to_text, audio",
                other
            )));
        }
    };

    let search_response = response
        .map_err(|e| AppError::internal(format!("Multimodal search failed: {}", e)))?;

    // Normalize media_filter values to lowercase for comparison.
    // "video" also matches "video_segment" for user convenience.
    let media_filters: Option<Vec<String>> = req.media_filter.map(|filters| {
        filters.iter().map(|s| s.to_lowercase()).collect()
    });

    // Enrich results with node content and apply namespace + media_filter + time_range.
    let mut results = Vec::new();
    for r in &search_response.results {
        if results.len() >= limit {
            break;
        }
        if let Ok(Some(node)) = state.registry.graph().get_node(r.node_id) {
            if !node_matches_namespace(&node, &namespace) {
                continue;
            }

            // Determine the media type: check _media_type metadata first, fall back to Node.media_type field.
            let node_media_type = node
                .metadata
                .get("_media_type")
                .and_then(|v| match v {
                    ucotron_core::Value::String(s) => Some(s.clone()),
                    _ => None,
                })
                .unwrap_or_else(|| {
                    match node.media_type {
                        Some(ucotron_core::MediaType::Audio) => "audio".to_string(),
                        Some(ucotron_core::MediaType::Image) => "image".to_string(),
                        Some(ucotron_core::MediaType::VideoSegment) => "video_segment".to_string(),
                        _ => "text".to_string(),
                    }
                });

            // Apply media_filter if specified. "video" matches both "video" and "video_segment".
            if let Some(ref filters) = media_filters {
                let mt_lower = node_media_type.to_lowercase();
                let matches = filters.iter().any(|f| {
                    if f == "video" {
                        mt_lower == "video" || mt_lower == "video_segment"
                    } else {
                        mt_lower == *f
                    }
                });
                if !matches {
                    continue;
                }
            }

            // Apply time_range filter if specified.
            if let Some((min_ts, max_ts)) = req.time_range {
                if node.timestamp < min_ts || node.timestamp > max_ts {
                    continue;
                }
            }

            let media_uri = node
                .metadata
                .get("_media_uri")
                .and_then(|v| match v {
                    ucotron_core::Value::String(s) => Some(s.clone()),
                    _ => None,
                });

            let source_str = match r.source {
                ResultSource::TextIndex => "text_index",
                ResultSource::VisualIndex => "visual_index",
                ResultSource::Fused => "fused",
            };

            // Normalize display media_type: "video_segment" → "video" for consistency.
            let display_media_type = if node_media_type == "video_segment" {
                "video".to_string()
            } else {
                node_media_type
            };

            results.push(MultimodalSearchResultItem {
                node_id: r.node_id,
                content: node.content.clone(),
                score: r.score,
                media_type: display_media_type,
                media_uri,
                source: source_str.to_string(),
                timestamp_range: node.timestamp_range,
                parent_video_id: node.parent_video_id,
            });
        }
    }

    let total = results.len();
    state.total_searches.fetch_add(1, Ordering::Relaxed);
    record_search(&state);

    let metrics = &search_response.metrics;
    Ok(Json(MultimodalSearchResponse {
        results,
        total,
        query_type: req.query_type,
        metrics: MultimodalSearchMetrics {
            query_encoding_us: metrics.query_encoding_us,
            text_search_us: metrics.text_search_us,
            visual_search_us: metrics.visual_search_us,
            fusion_us: metrics.fusion_us,
            total_us: metrics.total_us,
            final_result_count: metrics.final_result_count,
        },
    }))
}

// ---------------------------------------------------------------------------
// Video Segment Retrieval
// ---------------------------------------------------------------------------

/// List all segments for a parent video, sorted by start time, with prev/next navigation.
#[utoipa::path(
    get,
    path = "/api/v1/videos/{parent_id}/segments",
    tag = "Multimodal",
    params(
        ("parent_id" = u64, Path, description = "Parent video node ID"),
    ),
    responses(
        (status = 200, description = "Video segments with navigation", body = VideoSegmentsResponse),
        (status = 404, description = "Parent video not found"),
    )
)]
pub async fn get_video_segments_handler(
    State(state): State<Arc<AppState>>,
    Extension(auth): Extension<AuthContext>,
    headers: HeaderMap,
    Path(parent_id): Path<u64>,
) -> Result<Json<VideoSegmentsResponse>, AppError> {
    require_role(&auth, ucotron_config::AuthRole::Reader)?;
    let namespace = extract_namespace(&headers);
    require_namespace_access(&auth, &namespace)?;

    // Verify parent video node exists.
    let parent_node = state
        .registry
        .graph()
        .get_node(parent_id)
        .map_err(|e| AppError::internal(format!("Failed to fetch parent node: {}", e)))?
        .ok_or_else(|| AppError::not_found("Parent video node not found"))?;

    if !node_matches_namespace(&parent_node, &namespace) {
        return Err(AppError::not_found("Parent video node not found"));
    }

    // Collect all segment nodes that reference this parent.
    let all_nodes = state
        .registry
        .graph()
        .get_all_nodes()
        .map_err(|e| AppError::internal(format!("Failed to fetch nodes: {}", e)))?;

    let mut segments: Vec<_> = all_nodes
        .into_iter()
        .filter(|n| n.parent_video_id == Some(parent_id))
        .filter(|n| node_matches_namespace(n, &namespace))
        .collect();

    // Sort segments by start time.
    segments.sort_by_key(|n| n.timestamp_range.map(|(start, _)| start).unwrap_or(0));

    let total = segments.len();

    // Build response with prev/next navigation links.
    let details: Vec<VideoSegmentDetail> = segments
        .iter()
        .enumerate()
        .map(|(i, seg)| {
            let (start_ms, end_ms) = seg.timestamp_range.unwrap_or((0, 0));
            VideoSegmentDetail {
                node_id: seg.id,
                content: seg.content.clone(),
                start_ms,
                end_ms,
                media_uri: seg.media_uri.clone(),
                prev_segment_id: if i > 0 { Some(segments[i - 1].id) } else { None },
                next_segment_id: segments.get(i + 1).map(|s| s.id),
            }
        })
        .collect();

    Ok(Json(VideoSegmentsResponse {
        parent_video_id: parent_id,
        total,
        segments: details,
    }))
}

// ---------------------------------------------------------------------------
// Video Helpers
// ---------------------------------------------------------------------------

/// Encode an extracted video frame's RGB data into a CLIP embedding.
///
/// Constructs an `image::DynamicImage` from the raw RGB pixel data and
/// passes it through the CLIP image embedding pipeline.
fn embed_frame_rgb(
    embedder: &dyn ucotron_extraction::ImageEmbeddingPipeline,
    frame: &ucotron_extraction::video::ExtractedFrame,
) -> Result<Vec<f32>, AppError> {
    // Encode the raw RGB bytes to PNG in-memory, then pass to embed_image_bytes.
    let img = image::RgbImage::from_raw(frame.width, frame.height, frame.rgb_data.clone())
        .ok_or_else(|| {
            AppError::internal("Failed to construct image from frame RGB data")
        })?;
    let mut buf = std::io::Cursor::new(Vec::new());
    image::DynamicImage::ImageRgb8(img)
        .write_to(&mut buf, image::ImageFormat::Png)
        .map_err(|e| AppError::internal(format!("Failed to encode frame to PNG: {}", e)))?;
    embedder
        .embed_image_bytes(buf.get_ref())
        .map_err(|e| AppError::internal(format!("Frame embedding failed: {}", e)))
}

// ---------------------------------------------------------------------------
// Media File Serving
// ---------------------------------------------------------------------------

/// Serve a media file (image, audio, video) by node ID.
///
/// Looks up the node's `media_uri`, resolves it against the configured media directory,
/// and streams the file with the correct `Content-Type` header.
/// Supports HTTP Range requests for video streaming.
#[utoipa::path(
    get,
    path = "/api/v1/media/{id}",
    tag = "Media",
    params(
        ("id" = u64, Path, description = "Node ID of the media memory")
    ),
    responses(
        (status = 200, description = "Media file served"),
        (status = 206, description = "Partial content (range request)"),
        (status = 404, description = "Media not found")
    )
)]
pub async fn get_media_handler(
    State(state): State<Arc<AppState>>,
    Extension(auth): Extension<AuthContext>,
    Path(id): Path<u64>,
    headers: HeaderMap,
) -> Result<axum::response::Response, AppError> {
    use axum::body::Body;
    use axum::http::{header, StatusCode};
    use axum::response::IntoResponse;

    require_role(&auth, ucotron_config::AuthRole::Reader)?;

    // Look up the node to find its media_uri.
    let node = state
        .registry
        .graph()
        .get_node(id)
        .map_err(|e| AppError::internal(format!("Graph lookup failed: {}", e)))?
        .ok_or_else(|| AppError::not_found(format!("Memory {} not found", id)))?;

    let media_filename = node
        .media_uri
        .as_deref()
        .ok_or_else(|| AppError::not_found(format!("Memory {} has no media file", id)))?;

    // Resolve path against media directory. Prevent path traversal.
    let safe_name = std::path::Path::new(media_filename)
        .file_name()
        .ok_or_else(|| AppError::bad_request("Invalid media filename"))?;
    let media_dir = std::path::Path::new(state.config.storage.effective_media_dir());
    let file_path = media_dir.join(safe_name);

    if !file_path.exists() {
        return Err(AppError::not_found(format!(
            "Media file not found on disk for memory {}",
            id
        )));
    }

    let file_metadata = std::fs::metadata(&file_path)
        .map_err(|e| AppError::internal(format!("Failed to read file metadata: {}", e)))?;
    let file_size = file_metadata.len();

    // Detect content type from extension.
    let ext = std::path::Path::new(media_filename)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("");
    let content_type = content_type_for_ext(ext);

    // Check for Range header.
    if let Some(range_header) = headers.get(header::RANGE) {
        let range_str = range_header
            .to_str()
            .map_err(|_| AppError::bad_request("Invalid Range header"))?;

        // Parse "bytes=START-END" (END is optional).
        if let Some(range) = range_str.strip_prefix("bytes=") {
            let parts: Vec<&str> = range.splitn(2, '-').collect();
            let start: u64 = parts[0]
                .parse()
                .map_err(|_| AppError::bad_request("Invalid range start"))?;
            let end: u64 = if parts.len() > 1 && !parts[1].is_empty() {
                parts[1]
                    .parse()
                    .map_err(|_| AppError::bad_request("Invalid range end"))?
            } else {
                file_size - 1
            };

            if start >= file_size || end >= file_size || start > end {
                // 416 Range Not Satisfiable
                return Ok((
                    StatusCode::RANGE_NOT_SATISFIABLE,
                    [(
                        header::CONTENT_RANGE,
                        format!("bytes */{}", file_size),
                    )],
                )
                    .into_response());
            }

            let length = end - start + 1;
            let mut file = std::fs::File::open(&file_path)
                .map_err(|e| AppError::internal(format!("Failed to open file: {}", e)))?;
            std::io::Seek::seek(&mut file, std::io::SeekFrom::Start(start))
                .map_err(|e| AppError::internal(format!("Failed to seek: {}", e)))?;
            let mut buf = vec![0u8; length as usize];
            std::io::Read::read_exact(&mut file, &mut buf)
                .map_err(|e| AppError::internal(format!("Failed to read range: {}", e)))?;

            return Ok((
                StatusCode::PARTIAL_CONTENT,
                [
                    (header::CONTENT_TYPE, content_type.to_string()),
                    (header::CONTENT_LENGTH, length.to_string()),
                    (
                        header::CONTENT_RANGE,
                        format!("bytes {}-{}/{}", start, end, file_size),
                    ),
                    (header::ACCEPT_RANGES, "bytes".to_string()),
                ],
                Body::from(buf),
            )
                .into_response());
        }
    }

    // Full file response.
    let data = std::fs::read(&file_path)
        .map_err(|e| AppError::internal(format!("Failed to read media file: {}", e)))?;

    Ok((
        StatusCode::OK,
        [
            (header::CONTENT_TYPE, content_type.to_string()),
            (header::CONTENT_LENGTH, file_size.to_string()),
            (header::ACCEPT_RANGES, "bytes".to_string()),
            (
                header::CONTENT_DISPOSITION,
                format!("inline; filename=\"{}\"", safe_name.to_string_lossy()),
            ),
        ],
        Body::from(data),
    )
        .into_response())
}

// --- Connector sync handlers ---

/// POST /api/v1/connectors/:id/sync — Trigger a manual sync for a connector.
///
/// Requires admin or writer role.
pub async fn trigger_connector_sync_handler(
    State(state): State<Arc<AppState>>,
    Extension(auth): Extension<AuthContext>,
    Path(connector_id): Path<String>,
) -> Result<Json<ConnectorSyncTriggerResponse>, AppError> {
    require_role(&auth, ucotron_config::AuthRole::Writer)?;

    let cron_scheduler = state.cron_scheduler.as_ref().ok_or_else(|| {
        AppError::bad_request("Connector scheduling is not enabled")
    })?;

    match cron_scheduler.trigger_sync(&connector_id).await {
        Ok(()) => Ok(Json(ConnectorSyncTriggerResponse {
            triggered: true,
            connector_id,
            message: "Sync triggered successfully".to_string(),
        })),
        Err(e) => Err(AppError::internal(format!(
            "Failed to trigger sync: {}",
            e
        ))),
    }
}

/// GET /api/v1/connectors/schedules — List all connector schedules.
///
/// Requires admin or reader role.
pub async fn list_connector_schedules_handler(
    State(state): State<Arc<AppState>>,
    Extension(auth): Extension<AuthContext>,
) -> Result<Json<Vec<ConnectorScheduleResponse>>, AppError> {
    require_role(&auth, ucotron_config::AuthRole::Reader)?;

    let cron_scheduler = state.cron_scheduler.as_ref().ok_or_else(|| {
        AppError::bad_request("Connector scheduling is not enabled")
    })?;

    let sched = cron_scheduler.state().read().await;
    let schedules: Vec<ConnectorScheduleResponse> = sched
        .list_schedules()
        .iter()
        .map(|s| {
            let next = s
                .cron_expression
                .as_deref()
                .and_then(next_fire_time);
            ConnectorScheduleResponse {
                connector_id: s.connector_id.clone(),
                cron_expression: s.cron_expression.clone(),
                enabled: s.enabled,
                timeout_secs: s.timeout_secs,
                max_retries: s.max_retries,
                next_fire_at: next,
            }
        })
        .collect();

    Ok(Json(schedules))
}

/// POST /api/v1/webhooks/:connector_id — Receive webhook from external service.
///
/// Parses the webhook payload per connector type and triggers incremental sync.
/// Accepts any content type (JSON, form-encoded, etc.) and forwards raw body + headers.
pub async fn webhook_handler(
    State(state): State<Arc<AppState>>,
    Path(connector_id): Path<String>,
    headers: HeaderMap,
    body: axum::body::Body,
) -> Result<Json<WebhookResponse>, AppError> {
    // Read raw body bytes (limit to 1MB for safety).
    let bytes = axum::body::to_bytes(body, 1024 * 1024)
        .await
        .map_err(|e| AppError::bad_request(format!("Failed to read webhook body: {}", e)))?;

    let cron_scheduler = state.cron_scheduler.as_ref().ok_or_else(|| {
        AppError::bad_request("Connector scheduling is not enabled")
    })?;

    // Build WebhookPayload from request.
    let content_type = headers
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string());

    let mut header_map = std::collections::HashMap::new();
    for (name, value) in headers.iter() {
        if let Ok(v) = value.to_str() {
            header_map.insert(name.as_str().to_string(), v.to_string());
        }
    }

    let payload = ucotron_connectors::WebhookPayload {
        body: bytes.to_vec(),
        headers: header_map,
        content_type,
    };

    // Invoke the registered webhook handler.
    let items = cron_scheduler
        .handle_webhook(&connector_id, payload)
        .await
        .map_err(|e| {
            let msg = e.to_string();
            if msg.contains("not found") || msg.contains("not registered") {
                AppError::not_found(msg)
            } else if msg.contains("disabled") {
                AppError::bad_request(msg)
            } else {
                AppError::internal(format!("Webhook processing failed: {}", msg))
            }
        })?;

    let items_count = items.len();

    // Trigger incremental sync after processing webhook items.
    let sync_triggered = if items_count > 0 {
        cron_scheduler
            .trigger_sync(&connector_id)
            .await
            .is_ok()
    } else {
        false
    };

    Ok(Json(WebhookResponse {
        accepted: true,
        connector_id,
        items_processed: items_count,
        sync_triggered,
        message: format!("Webhook processed: {} items extracted", items_count),
    }))
}

/// GET /api/v1/connectors/:id/history — Get sync history for a connector.
///
/// Requires admin or reader role.
pub async fn connector_sync_history_handler(
    State(state): State<Arc<AppState>>,
    Extension(auth): Extension<AuthContext>,
    Path(connector_id): Path<String>,
) -> Result<Json<ConnectorSyncHistoryResponse>, AppError> {
    require_role(&auth, ucotron_config::AuthRole::Reader)?;

    let cron_scheduler = state.cron_scheduler.as_ref().ok_or_else(|| {
        AppError::bad_request("Connector scheduling is not enabled")
    })?;

    let sched = cron_scheduler.state().read().await;
    let history = sched.get_history(&connector_id);

    let records: Vec<ConnectorSyncRecordResponse> = history
        .iter()
        .map(|r| {
            let (status_str, error) = match &r.status {
                ucotron_connectors::SyncStatus::Success => ("success".to_string(), None),
                ucotron_connectors::SyncStatus::Failed { error } => {
                    ("failed".to_string(), Some(error.clone()))
                }
                ucotron_connectors::SyncStatus::Running => ("running".to_string(), None),
                ucotron_connectors::SyncStatus::Cancelled => ("cancelled".to_string(), None),
            };
            ConnectorSyncRecordResponse {
                started_at: r.started_at,
                finished_at: r.finished_at,
                items_fetched: r.items_fetched,
                items_skipped: r.items_skipped,
                status: status_str,
                error,
            }
        })
        .collect();

    Ok(Json(ConnectorSyncHistoryResponse {
        connector_id,
        records,
    }))
}

// ---------------------------------------------------------------------------
// Conversations
// ---------------------------------------------------------------------------

/// List all conversations in the current namespace.
#[utoipa::path(
    get,
    path = "/api/v1/conversations",
    params(
        ("limit" = Option<usize>, Query, description = "Max conversations to return (default 50)"),
        ("offset" = Option<usize>, Query, description = "Offset for pagination"),
        ("X-Ucotron-Namespace" = Option<String>, Header, description = "Target namespace")
    ),
    responses(
        (status = 200, description = "List of conversations", body = Vec<ConversationSummary>),
        (status = 500, description = "Internal error", body = ApiErrorResponse)
    )
)]
pub async fn list_conversations_handler(
    State(state): State<Arc<AppState>>,
    Extension(auth): Extension<AuthContext>,
    headers: HeaderMap,
    Query(params): Query<ListConversationsParams>,
) -> Result<Json<Vec<ConversationSummary>>, AppError> {
    require_role(&auth, ucotron_config::AuthRole::Reader)?;
    let namespace = extract_namespace(&headers);
    require_namespace_access(&auth, &namespace)?;

    // Scan nodes to find all conversation IDs in this namespace.
    // Use a zero-vector search to get all nodes (pragmatic approach).
    let query_vec = vec![0.0f32; 384];
    let results = state
        .registry
        .vector()
        .search(&query_vec, 5000)
        .map_err(|e| AppError::internal(format!("Vector search failed: {}", e)))?;

    // Group nodes by conversation_id.
    let mut conversations: std::collections::HashMap<String, Vec<ucotron_core::Node>> =
        std::collections::HashMap::new();

    for (node_id, _score) in &results {
        if let Ok(Some(node)) = state.registry.graph().get_node(*node_id) {
            if !node_matches_namespace(&node, &namespace) {
                continue;
            }
            if let Some(ucotron_core::Value::String(conv_id)) =
                node.metadata.get("_conversation_id")
            {
                conversations
                    .entry(conv_id.clone())
                    .or_default()
                    .push(node);
            }
        }
    }

    // Build summaries sorted by last message time (newest first).
    let mut summaries: Vec<ConversationSummary> = conversations
        .into_iter()
        .map(|(conv_id, mut nodes)| {
            nodes.sort_by_key(|n| n.timestamp);
            let first = nodes.first();
            let last = nodes.last();
            let preview = first
                .map(|n| {
                    let content = &n.content;
                    if content.len() > 120 {
                        format!("{}...", &content[..120])
                    } else {
                        content.clone()
                    }
                })
                .unwrap_or_default();
            ConversationSummary {
                conversation_id: conv_id,
                namespace: namespace.clone(),
                message_count: nodes.len(),
                first_message_at: first.map(|n| format!("{}", n.timestamp)),
                last_message_at: last.map(|n| format!("{}", n.timestamp)),
                preview,
            }
        })
        .collect();

    summaries.sort_by(|a, b| b.last_message_at.cmp(&a.last_message_at));

    // Apply pagination.
    let paginated: Vec<ConversationSummary> = summaries
        .into_iter()
        .skip(params.offset)
        .take(params.limit)
        .collect();

    Ok(Json(paginated))
}

/// Get all messages in a specific conversation.
#[utoipa::path(
    get,
    path = "/api/v1/conversations/{id}/messages",
    params(
        ("id" = String, Path, description = "Conversation ID"),
        ("X-Ucotron-Namespace" = Option<String>, Header, description = "Target namespace")
    ),
    responses(
        (status = 200, description = "Conversation detail with messages", body = ConversationDetail),
        (status = 404, description = "Conversation not found"),
        (status = 500, description = "Internal error", body = ApiErrorResponse)
    )
)]
pub async fn get_conversation_messages_handler(
    State(state): State<Arc<AppState>>,
    Extension(auth): Extension<AuthContext>,
    headers: HeaderMap,
    Path(conversation_id): Path<String>,
) -> Result<Json<ConversationDetail>, AppError> {
    require_role(&auth, ucotron_config::AuthRole::Reader)?;
    let namespace = extract_namespace(&headers);
    require_namespace_access(&auth, &namespace)?;

    // Scan for nodes with matching conversation_id.
    let query_vec = vec![0.0f32; 384];
    let results = state
        .registry
        .vector()
        .search(&query_vec, 5000)
        .map_err(|e| AppError::internal(format!("Vector search failed: {}", e)))?;

    let mut messages: Vec<ConversationMessage> = Vec::new();

    for (node_id, _score) in &results {
        if let Ok(Some(node)) = state.registry.graph().get_node(*node_id) {
            if !node_matches_namespace(&node, &namespace) {
                continue;
            }
            if let Some(ucotron_core::Value::String(conv_id)) =
                node.metadata.get("_conversation_id")
            {
                if conv_id == &conversation_id {
                    // Collect entity names from edges.
                    let entity_names: Vec<String> = state
                        .registry
                        .graph()
                        .get_neighbors(node.id, 1)
                        .unwrap_or_default()
                        .iter()
                        .filter_map(|n| {
                            if n.node_type == ucotron_core::NodeType::Entity {
                                Some(n.content.clone())
                            } else {
                                None
                            }
                        })
                        .collect();

                    messages.push(ConversationMessage {
                        id: node.id,
                        content: node.content.clone(),
                        created_at: format!("{}", node.timestamp),
                        entities: entity_names,
                    });
                }
            }
        }
    }

    if messages.is_empty() {
        return Err(AppError::not_found(format!(
            "Conversation '{}' not found",
            conversation_id
        )));
    }

    // Sort by creation time.
    messages.sort_by(|a, b| a.created_at.cmp(&b.created_at));

    Ok(Json(ConversationDetail {
        conversation_id,
        namespace,
        messages,
    }))
}
