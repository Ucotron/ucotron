//! Ingestion flow orchestrator for the Ucotron cognitive memory framework.
//!
//! Implements the complete text-to-graph pipeline:
//! 1. **Chunking** — Split input text into sentences
//! 2. **Embedding** — Generate 384-dim vectors for each chunk
//! 3. **NER** — Extract named entities via GLiNER (zero-shot)
//! 4. **Relation Extraction** — Infer relations between entities (co-occurrence or LLM)
//! 5. **Entity Resolution** — Merge or create entity nodes in the graph
//! 6. **Contradiction Detection** — Detect and resolve conflicting facts
//! 7. **Graph Update** — Upsert nodes, edges, and embeddings
//! 8. **Store Raw Chunks** — Persist original text for LazyGraphRAG retrieval
//!
//! Each step emits timing metrics. Failed chunks do not cancel the entire pipeline.

use std::collections::HashMap;
use std::time::Instant;

use anyhow::{Context, Result};
use tracing::{debug, info, info_span, warn};

use ucotron_core::contradictions::{build_conflict_edges, detect_conflict, resolve_conflict};
use ucotron_core::entity_resolution::structural_similarity;
use ucotron_core::types::{ConflictConfig, Fact};
use ucotron_core::{BackendRegistry, Edge, EdgeType, Node, NodeId, NodeType, Value};

use crate::{
    EmbeddingPipeline, ExtractedEntity, ExtractedRelation, NerPipeline, RelationExtractor,
};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the ingestion pipeline.
#[derive(Debug, Clone)]
pub struct IngestionConfig {
    /// NER labels to extract (zero-shot). Empty disables NER step.
    pub ner_labels: Vec<String>,
    /// Whether to run relation extraction.
    pub enable_relations: bool,
    /// Whether to run entity resolution against existing graph nodes.
    pub enable_entity_resolution: bool,
    /// Whether to run contradiction detection.
    pub enable_contradiction_detection: bool,
    /// Similarity threshold for entity resolution merge.
    pub entity_resolution_threshold: f32,
    /// Conflict config for contradiction detection.
    pub conflict_config: ConflictConfig,
    /// Starting node ID for newly created nodes (auto-incremented).
    /// If None, uses a hash-based ID scheme.
    pub next_node_id: Option<u64>,
    /// Batch size for NER inference. Multiple chunks are batched into a single
    /// model call for improved throughput. Default: 8.
    pub ner_batch_size: usize,
    /// Sub-batch size for embedding inference. When processing many chunks,
    /// they are split into sub-batches of this size. Default: 32.
    pub embedding_batch_size: usize,
}

impl Default for IngestionConfig {
    fn default() -> Self {
        Self {
            ner_labels: vec![
                "person".into(),
                "location".into(),
                "organization".into(),
                "date".into(),
                "concept".into(),
            ],
            enable_relations: true,
            enable_entity_resolution: true,
            enable_contradiction_detection: true,
            entity_resolution_threshold: 0.5,
            conflict_config: ConflictConfig::default(),
            next_node_id: None,
            ner_batch_size: 8,
            embedding_batch_size: 32,
        }
    }
}

// ---------------------------------------------------------------------------
// Metrics
// ---------------------------------------------------------------------------

/// Timing and count metrics for a single ingestion run.
#[derive(Debug, Clone, Default)]
pub struct IngestionMetrics {
    /// Number of chunks processed successfully.
    pub chunks_processed: usize,
    /// Number of chunks that failed (partial pipeline).
    pub chunks_failed: usize,
    /// Total entities extracted across all chunks.
    pub entities_extracted: usize,
    /// Total relations extracted across all chunks.
    pub relations_extracted: usize,
    /// Number of entity nodes created in the graph.
    pub entity_nodes_created: usize,
    /// Number of edges created in the graph.
    pub edges_created: usize,
    /// Number of contradictions detected.
    pub contradictions_detected: usize,
    /// Number of entity merges (existing node reused).
    pub entity_merges: usize,
    /// Microseconds spent on chunking.
    pub chunking_us: u64,
    /// Microseconds spent on embedding.
    pub embedding_us: u64,
    /// Microseconds spent on NER.
    pub ner_us: u64,
    /// Microseconds spent on relation extraction.
    pub relation_extraction_us: u64,
    /// Microseconds spent on entity resolution.
    pub entity_resolution_us: u64,
    /// Microseconds spent on contradiction detection.
    pub contradiction_detection_us: u64,
    /// Microseconds spent on graph update.
    pub graph_update_us: u64,
    /// Total pipeline duration in microseconds.
    pub total_us: u64,
}

// ---------------------------------------------------------------------------
// Result type
// ---------------------------------------------------------------------------

/// Result of ingesting a single text through the pipeline.
#[derive(Debug)]
pub struct IngestionResult {
    /// Metrics for this ingestion run.
    pub metrics: IngestionMetrics,
    /// Node IDs of chunk nodes created (raw text storage).
    pub chunk_node_ids: Vec<NodeId>,
    /// Node IDs of entity nodes created or reused.
    pub entity_node_ids: Vec<NodeId>,
    /// Edges created (entity-entity relations, chunk-entity links).
    pub edges_created: Vec<Edge>,
}

// ---------------------------------------------------------------------------
// Orchestrator
// ---------------------------------------------------------------------------

/// The ingestion orchestrator chains all extraction pipeline steps.
///
/// It is parameterized over trait objects so that:
/// - In production, real ONNX pipelines are used
/// - In tests, lightweight mocks can be substituted
pub struct IngestionOrchestrator<'a> {
    registry: &'a BackendRegistry,
    embedder: &'a dyn EmbeddingPipeline,
    ner: Option<&'a dyn NerPipeline>,
    relation_extractor: Option<&'a dyn RelationExtractor>,
    config: IngestionConfig,
    /// Monotonically increasing ID counter for new nodes.
    next_id: u64,
}

impl<'a> IngestionOrchestrator<'a> {
    /// Create a new orchestrator.
    ///
    /// NER and relation extraction are optional — pass `None` to skip those steps.
    pub fn new(
        registry: &'a BackendRegistry,
        embedder: &'a dyn EmbeddingPipeline,
        ner: Option<&'a dyn NerPipeline>,
        relation_extractor: Option<&'a dyn RelationExtractor>,
        config: IngestionConfig,
    ) -> Self {
        let next_id = config.next_node_id.unwrap_or(1_000_000);
        Self {
            registry,
            embedder,
            ner,
            relation_extractor,
            config,
            next_id,
        }
    }

    /// Allocate a fresh node ID.
    fn alloc_id(&mut self) -> NodeId {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    /// Ingest a single text through the full pipeline.
    ///
    /// Each chunk is processed independently. If embedding or NER fails for a
    /// chunk, the orchestrator logs the error and continues with the next chunk.
    pub fn ingest(&mut self, text: &str) -> Result<IngestionResult> {
        let ingest_span = info_span!(
            "ucotron.ingest",
            otel.kind = "internal",
            ucotron.pipeline = "ingestion",
            text_length = text.len(),
            chunks = tracing::field::Empty,
            entities_found = tracing::field::Empty,
            relations_count = tracing::field::Empty,
            edges_created = tracing::field::Empty,
            duration_us = tracing::field::Empty,
        );
        let _ingest_guard = ingest_span.enter();

        let pipeline_start = Instant::now();
        let mut metrics = IngestionMetrics::default();
        let mut chunk_node_ids = Vec::new();
        let mut entity_node_ids = Vec::new();
        let mut all_edges: Vec<Edge> = Vec::new();

        // ── Step 1: Chunking ───────────────────────────────────────────
        let chunks = {
            let chunk_span = info_span!(
                "ucotron.chunk",
                otel.kind = "internal",
                text_length = text.len(),
                chunks = tracing::field::Empty,
            );
            let _chunk_guard = chunk_span.enter();

            let chunk_start = Instant::now();
            let chunks = chunk_text(text);
            metrics.chunking_us = chunk_start.elapsed().as_micros() as u64;

            chunk_span.record("chunks", chunks.len());
            debug!(
                "Chunking: {} chunks from {} chars",
                chunks.len(),
                text.len()
            );
            chunks
        };

        if chunks.is_empty() {
            metrics.total_us = pipeline_start.elapsed().as_micros() as u64;
            return Ok(IngestionResult {
                metrics,
                chunk_node_ids,
                entity_node_ids,
                edges_created: all_edges,
            });
        }

        // ── Step 2: Embedding (batch) ──────────────────────────────────
        let embeddings = {
            let embed_span = info_span!(
                "ucotron.embed",
                otel.kind = "internal",
                chunks = chunks.len(),
                vectors = tracing::field::Empty,
                duration_us = tracing::field::Empty,
            );
            let _embed_guard = embed_span.enter();

            let embed_start = Instant::now();
            let chunk_refs: Vec<&str> = chunks.iter().map(|s| s.as_str()).collect();
            let batch_size = self.config.embedding_batch_size.max(1);

            // Process in sub-batches for controlled memory usage
            let mut embeddings = Vec::with_capacity(chunk_refs.len());
            for sub_batch in chunk_refs.chunks(batch_size) {
                let batch_result = self
                    .embedder
                    .embed_batch(sub_batch)
                    .context("Embedding batch failed")?;
                embeddings.extend(batch_result);
            }
            metrics.embedding_us = embed_start.elapsed().as_micros() as u64;

            embed_span.record("vectors", embeddings.len());
            embed_span.record("duration_us", metrics.embedding_us);
            debug!(
                "Embedding: {} vectors in {}us",
                embeddings.len(),
                metrics.embedding_us
            );
            embeddings
        };

        // ── Steps 3-7: Per-chunk processing ────────────────────────────
        // We collect entities and relations per-chunk, then do resolution
        // and graph update at the end for cross-chunk entity merging.

        let mut all_chunk_entities: Vec<Vec<ExtractedEntity>> = Vec::new();
        let mut all_chunk_relations: Vec<Vec<ExtractedRelation>> = Vec::new();

        // ── Step 3: Batch NER ────────────────────────────────────────
        let ner_span = info_span!(
            "ucotron.ner",
            otel.kind = "internal",
            chunks = chunks.len(),
            batch_size = self.config.ner_batch_size,
            entities_found = tracing::field::Empty,
            duration_us = tracing::field::Empty,
        );
        {
            let _ner_guard = ner_span.enter();
            let ner_start = Instant::now();

            if let Some(ner) = self.ner {
                if !self.config.ner_labels.is_empty() {
                    let labels: Vec<&str> =
                        self.config.ner_labels.iter().map(|s| s.as_str()).collect();
                    let batch_size = self.config.ner_batch_size.max(1);

                    // Process chunks in batches
                    for batch_start in (0..chunks.len()).step_by(batch_size) {
                        let batch_end = (batch_start + batch_size).min(chunks.len());
                        let batch_refs: Vec<&str> = chunks[batch_start..batch_end]
                            .iter()
                            .map(|s| s.as_str())
                            .collect();

                        match ner.extract_entities_batch(&batch_refs, &labels) {
                            Ok(batch_results) => {
                                for (j, ents) in batch_results.into_iter().enumerate() {
                                    metrics.entities_extracted += ents.len();
                                    all_chunk_entities.push(ents);
                                    debug!(
                                        "NER batch [{}-{}]: chunk {} processed",
                                        batch_start,
                                        batch_end,
                                        batch_start + j
                                    );
                                }
                            }
                            Err(e) => {
                                // Batch failed — fall back to per-chunk for this batch
                                warn!(
                                    "NER batch [{}-{}] failed ({}), falling back to per-chunk",
                                    batch_start, batch_end, e
                                );
                                for (j, chunk) in chunks[batch_start..batch_end].iter().enumerate()
                                {
                                    match ner.extract_entities(chunk, &labels) {
                                        Ok(ents) => {
                                            metrics.entities_extracted += ents.len();
                                            all_chunk_entities.push(ents);
                                        }
                                        Err(e2) => {
                                            warn!(
                                                "NER failed for chunk {}: {}",
                                                batch_start + j,
                                                e2
                                            );
                                            metrics.chunks_failed += 1;
                                            all_chunk_entities.push(Vec::new());
                                        }
                                    }
                                }
                            }
                        }
                    }
                } else {
                    all_chunk_entities.resize(chunks.len(), Vec::new());
                }
            } else {
                all_chunk_entities.resize(chunks.len(), Vec::new());
            }

            metrics.ner_us = ner_start.elapsed().as_micros() as u64;
        }
        ner_span.record("entities_found", metrics.entities_extracted);
        ner_span.record("duration_us", metrics.ner_us);

        // ── Step 4: Relation extraction (per-chunk, uses NER results) ──
        let relations_span = info_span!(
            "ucotron.relations",
            otel.kind = "internal",
            chunks = chunks.len(),
            relations_count = tracing::field::Empty,
            duration_us = tracing::field::Empty,
        );
        {
            let _rel_guard = relations_span.enter();
            let re_start = Instant::now();

            for (i, chunk) in chunks.iter().enumerate() {
                let entities = &all_chunk_entities[i];
                let relations = if self.config.enable_relations {
                    if let Some(re) = self.relation_extractor {
                        if entities.len() >= 2 {
                            match re.extract_relations(chunk, entities) {
                                Ok(rels) => rels,
                                Err(e) => {
                                    warn!("Relation extraction failed for chunk {}: {}", i, e);
                                    Vec::new()
                                }
                            }
                        } else {
                            Vec::new()
                        }
                    } else {
                        Vec::new()
                    }
                } else {
                    Vec::new()
                };
                metrics.relations_extracted += relations.len();
                all_chunk_relations.push(relations);
                metrics.chunks_processed += 1;
            }

            metrics.relation_extraction_us = re_start.elapsed().as_micros() as u64;
        }
        relations_span.record("relations_count", metrics.relations_extracted);
        relations_span.record("duration_us", metrics.relation_extraction_us);

        // ── Step 5: Entity Resolution ──────────────────────────────────
        // Deduplicate entities across all chunks and resolve against existing graph.
        let entity_map = {
            let er_span = info_span!(
                "ucotron.entity_resolution",
                otel.kind = "internal",
                entities_input = metrics.entities_extracted,
                entities_created = tracing::field::Empty,
                entity_merges = tracing::field::Empty,
                duration_us = tracing::field::Empty,
            );
            let _er_guard = er_span.enter();

            let er_start = Instant::now();
            let entity_map =
                self.resolve_and_create_entities(&chunks, &embeddings, &all_chunk_entities)?;
            metrics.entity_resolution_us = er_start.elapsed().as_micros() as u64;
            metrics.entity_nodes_created = entity_map.len();
            metrics.entity_merges = entity_map.values().filter(|info| info.merged).count();

            er_span.record("entities_created", metrics.entity_nodes_created);
            er_span.record("entity_merges", metrics.entity_merges);
            er_span.record("duration_us", metrics.entity_resolution_us);
            entity_map
        };

        for info in entity_map.values() {
            entity_node_ids.push(info.node_id);
        }

        // ── Step 6: Contradiction Detection ────────────────────────────
        let mut conflict_edges = Vec::new();
        {
            let cd_span = info_span!(
                "ucotron.contradiction_detection",
                otel.kind = "internal",
                contradictions_detected = tracing::field::Empty,
                duration_us = tracing::field::Empty,
            );
            let _cd_guard = cd_span.enter();

            let cd_start = Instant::now();
            if self.config.enable_contradiction_detection {
                conflict_edges = self.detect_contradictions(
                    &all_chunk_entities,
                    &all_chunk_relations,
                    &entity_map,
                );
                metrics.contradictions_detected = conflict_edges.len() / 2; // CONFLICTS_WITH + SUPERSEDES = 2 edges per conflict
            }
            metrics.contradiction_detection_us = cd_start.elapsed().as_micros() as u64;

            cd_span.record("contradictions_detected", metrics.contradictions_detected);
            cd_span.record("duration_us", metrics.contradiction_detection_us);
        }

        // ── Step 7: Graph Update ───────────────────────────────────────
        let graph_update_span = info_span!(
            "ucotron.graph_update",
            otel.kind = "internal",
            nodes_upserted = tracing::field::Empty,
            edges_upserted = tracing::field::Empty,
            embeddings_upserted = tracing::field::Empty,
            duration_us = tracing::field::Empty,
        );
        let _gu_guard = graph_update_span.enter();
        let gu_start = Instant::now();

        // 7a: Create chunk nodes (raw text storage for LazyGraphRAG)
        let mut chunk_nodes = Vec::new();
        let mut chunk_embeddings = Vec::new();
        for (i, chunk) in chunks.iter().enumerate() {
            let chunk_id = self.alloc_id();
            chunk_node_ids.push(chunk_id);
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();
            chunk_nodes.push(Node {
                id: chunk_id,
                content: chunk.clone(),
                embedding: embeddings[i].clone(),
                metadata: {
                    let mut m = HashMap::new();
                    m.insert("chunk_index".into(), Value::Integer(i as i64));
                    m.insert("source_type".into(), Value::String("ingestion".into()));
                    m
                },
                node_type: NodeType::Event, // Episodic memory: raw text chunks
                timestamp: now,
                media_type: None,
                media_uri: None,
                embedding_visual: None,
                timestamp_range: None,
                parent_video_id: None,
            });
            chunk_embeddings.push((chunk_id, embeddings[i].clone()));
        }

        // 7b: Collect entity nodes (already resolved)
        let entity_nodes: Vec<Node> = entity_map
            .values()
            .filter(|info| !info.merged) // Only upsert newly-created entities
            .map(|info| info.node.clone())
            .collect();

        let entity_embeddings: Vec<(NodeId, Vec<f32>)> = entity_map
            .values()
            .filter(|info| !info.merged)
            .map(|info| (info.node_id, info.node.embedding.clone()))
            .collect();

        // 7c: Build edges from relations
        let mut relation_edges = Vec::new();
        for (i, relations) in all_chunk_relations.iter().enumerate() {
            let chunk_id = chunk_node_ids[i];
            for rel in relations {
                // Find entity node IDs for subject and object
                let subj_key = normalize_entity_key(&rel.subject);
                let obj_key = normalize_entity_key(&rel.object);
                if let (Some(subj_info), Some(obj_info)) =
                    (entity_map.get(&subj_key), entity_map.get(&obj_key))
                {
                    // Entity → Entity relation edge
                    relation_edges.push(Edge {
                        source: subj_info.node_id,
                        target: obj_info.node_id,
                        edge_type: predicate_to_edge_type(&rel.predicate),
                        weight: rel.confidence,
                        metadata: {
                            let mut m = HashMap::new();
                            m.insert("predicate".into(), Value::String(rel.predicate.clone()));
                            m.insert("source_chunk".into(), Value::Integer(chunk_id as i64));
                            m
                        },
                    });
                }
            }

            // Link chunk to its entities
            for entity in &all_chunk_entities[i] {
                let key = normalize_entity_key(&entity.text);
                if let Some(info) = entity_map.get(&key) {
                    relation_edges.push(Edge {
                        source: chunk_id,
                        target: info.node_id,
                        edge_type: EdgeType::RelatesTo,
                        weight: entity.confidence,
                        metadata: {
                            let mut m = HashMap::new();
                            m.insert("entity_label".into(), Value::String(entity.label.clone()));
                            m
                        },
                    });
                }
            }
        }

        // 7d: Upsert everything to the graph
        let mut all_nodes = chunk_nodes;
        all_nodes.extend(entity_nodes);
        if !all_nodes.is_empty() {
            self.registry
                .graph()
                .upsert_nodes(&all_nodes)
                .context("Failed to upsert nodes")?;
        }

        let mut all_upsert_edges = relation_edges.clone();
        all_upsert_edges.extend(conflict_edges.clone());
        if !all_upsert_edges.is_empty() {
            self.registry
                .graph()
                .upsert_edges(&all_upsert_edges)
                .context("Failed to upsert edges")?;
        }

        // 7e: Upsert embeddings to vector backend
        let mut all_embeddings = chunk_embeddings;
        all_embeddings.extend(entity_embeddings);
        if !all_embeddings.is_empty() {
            self.registry
                .vector()
                .upsert_embeddings(&all_embeddings)
                .context("Failed to upsert embeddings")?;
        }

        all_edges.extend(relation_edges);
        all_edges.extend(conflict_edges);
        metrics.edges_created = all_edges.len();
        metrics.graph_update_us = gu_start.elapsed().as_micros() as u64;

        graph_update_span.record("nodes_upserted", all_nodes.len());
        graph_update_span.record("edges_upserted", all_upsert_edges.len());
        graph_update_span.record("embeddings_upserted", all_embeddings.len());
        graph_update_span.record("duration_us", metrics.graph_update_us);
        drop(_gu_guard);

        metrics.total_us = pipeline_start.elapsed().as_micros() as u64;

        // Record final attributes on the root ingest span
        ingest_span.record("chunks", metrics.chunks_processed);
        ingest_span.record("entities_found", metrics.entities_extracted);
        ingest_span.record("relations_count", metrics.relations_extracted);
        ingest_span.record("edges_created", metrics.edges_created);
        ingest_span.record("duration_us", metrics.total_us);

        info!(
            "Ingestion complete: {} chunks, {} entities, {} relations, {} edges in {}us",
            metrics.chunks_processed,
            metrics.entities_extracted,
            metrics.relations_extracted,
            metrics.edges_created,
            metrics.total_us,
        );

        Ok(IngestionResult {
            metrics,
            chunk_node_ids,
            entity_node_ids,
            edges_created: all_edges,
        })
    }

    /// Resolve extracted entities: deduplicate across chunks and optionally
    /// merge with existing graph nodes.
    fn resolve_and_create_entities(
        &mut self,
        _chunks: &[String],
        _embeddings: &[Vec<f32>],
        all_chunk_entities: &[Vec<ExtractedEntity>],
    ) -> Result<HashMap<String, EntityInfo>> {
        let mut entity_map: HashMap<String, EntityInfo> = HashMap::new();

        // Collect unique entities by normalized name
        for chunk_entities in all_chunk_entities {
            for entity in chunk_entities {
                let key = normalize_entity_key(&entity.text);
                if entity_map.contains_key(&key) {
                    // Already seen this entity — keep highest confidence
                    let existing = entity_map.get_mut(&key).unwrap();
                    if entity.confidence > existing.best_confidence {
                        existing.best_confidence = entity.confidence;
                        existing.label = entity.label.clone();
                    }
                    continue;
                }

                // Try to match against existing graph nodes if entity resolution is enabled
                let mut merged = false;
                let mut node_id = 0u64;

                if self.config.enable_entity_resolution {
                    // Search vector backend for similar entities
                    if let Ok(embedding) = self.embedder.embed_text(&entity.text) {
                        if let Ok(candidates) = self.registry.vector().search(&embedding, 5) {
                            for (cand_id, sim) in &candidates {
                                if *sim >= self.config.entity_resolution_threshold {
                                    // Check if candidate is an Entity node with similar content
                                    if let Ok(Some(cand_node)) =
                                        self.registry.graph().get_node(*cand_id)
                                    {
                                        if matches!(cand_node.node_type, NodeType::Entity) {
                                            // Use structural similarity for confirmation
                                            let cand_neighbors: std::collections::HashSet<NodeId> =
                                                self.registry
                                                    .graph()
                                                    .get_neighbors(*cand_id, 1)
                                                    .unwrap_or_default()
                                                    .iter()
                                                    .map(|n| n.id)
                                                    .collect();

                                            // New entity has no neighbors yet — use pure cosine
                                            let combined = structural_similarity(
                                                &embedding,
                                                &std::collections::HashSet::new(),
                                                &cand_node.embedding,
                                                &cand_neighbors,
                                            );

                                            if combined >= self.config.entity_resolution_threshold {
                                                debug!(
                                                    "Merging entity '{}' with existing node {} (sim={:.3})",
                                                    entity.text, cand_id, combined
                                                );
                                                node_id = *cand_id;
                                                merged = true;
                                                break;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                if !merged {
                    node_id = self.alloc_id();
                }

                let now = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs();

                let embedding = self
                    .embedder
                    .embed_text(&entity.text)
                    .unwrap_or_else(|_| vec![0.0; 384]);

                let node = Node {
                    id: node_id,
                    content: entity.text.clone(),
                    embedding,
                    metadata: {
                        let mut m = HashMap::new();
                        m.insert("entity_label".into(), Value::String(entity.label.clone()));
                        m.insert("confidence".into(), Value::Float(entity.confidence as f64));
                        m
                    },
                    node_type: NodeType::Entity,
                    timestamp: now,
                    media_type: None,
                    media_uri: None,
                    embedding_visual: None,
                    timestamp_range: None,
                    parent_video_id: None,
                };

                entity_map.insert(
                    key,
                    EntityInfo {
                        node_id,
                        node,
                        merged,
                        best_confidence: entity.confidence,
                        label: entity.label.clone(),
                    },
                );
            }
        }

        Ok(entity_map)
    }

    /// Detect contradictions among the extracted relations.
    ///
    /// For each relation (subject, predicate, object), check if an existing
    /// relation with the same subject+predicate but different object exists.
    fn detect_contradictions(
        &self,
        _all_chunk_entities: &[Vec<ExtractedEntity>],
        all_chunk_relations: &[Vec<ExtractedRelation>],
        entity_map: &HashMap<String, EntityInfo>,
    ) -> Vec<Edge> {
        let mut conflict_edges = Vec::new();
        let mut known_facts: Vec<Fact> = Vec::new();
        let mut fact_id_counter: u64 = 1;

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        for relations in all_chunk_relations {
            for rel in relations {
                let subj_key = normalize_entity_key(&rel.subject);
                if let Some(subj_info) = entity_map.get(&subj_key) {
                    let new_fact = Fact::new(
                        fact_id_counter,
                        subj_info.node_id,
                        rel.predicate.clone(),
                        rel.object.clone(),
                        rel.confidence,
                        now,
                    );
                    fact_id_counter += 1;

                    if let Some(conflict) =
                        detect_conflict(&new_fact, &known_facts, &self.config.conflict_config)
                    {
                        debug!(
                            "Contradiction: '{}' {} '{}' vs '{}'",
                            rel.subject, rel.predicate, rel.object, conflict.incoming.object,
                        );
                        let resolution = resolve_conflict(
                            &conflict.existing,
                            &new_fact,
                            &self.config.conflict_config,
                        );
                        let edges = build_conflict_edges(&resolution, now);
                        conflict_edges.extend(edges);
                    }

                    known_facts.push(new_fact);
                }
            }
        }

        conflict_edges
    }
}

// ---------------------------------------------------------------------------
// Internal types
// ---------------------------------------------------------------------------

/// Information about a resolved entity.
#[derive(Debug, Clone)]
struct EntityInfo {
    node_id: NodeId,
    node: Node,
    merged: bool,
    best_confidence: f32,
    #[allow(dead_code)]
    label: String,
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

/// Split text into sentence-level chunks.
///
/// Uses a simple rule-based approach: split on `.`, `!`, `?` that are actual
/// sentence boundaries. Decimal points within numbers (e.g. 99.99, 2.0.1) are
/// preserved and NOT treated as sentence boundaries.
pub fn chunk_text(text: &str) -> Vec<String> {
    if text.trim().is_empty() {
        return Vec::new();
    }

    let chars: Vec<char> = text.chars().collect();
    let len = chars.len();
    let mut chunks = Vec::new();
    let mut current = String::new();

    for i in 0..len {
        let ch = chars[i];
        current.push(ch);

        if ch == '!' || ch == '?' {
            // Always split on ! and ?
            if !current.trim().is_empty() {
                let trimmed = current.trim().to_string();
                if !trimmed.is_empty() && trimmed.len() > 1 {
                    chunks.push(trimmed);
                }
                current.clear();
            }
        } else if ch == '.' {
            // Check if this period is a decimal point: digit before AND digit after
            let prev_is_digit = i > 0 && chars[i - 1].is_ascii_digit();
            let next_is_digit = i + 1 < len && chars[i + 1].is_ascii_digit();

            if prev_is_digit && next_is_digit {
                // Decimal point inside a number — do NOT split
                continue;
            }

            // Regular sentence-ending period
            if !current.trim().is_empty() {
                let trimmed = current.trim().to_string();
                if !trimmed.is_empty() && trimmed.len() > 1 {
                    chunks.push(trimmed);
                }
                current.clear();
            }
        }
    }

    // Remaining text (no terminal punctuation)
    let trimmed = current.trim().to_string();
    if !trimmed.is_empty() && trimmed.len() > 1 {
        chunks.push(trimmed);
    }

    chunks
}

/// Normalize entity text for deduplication (lowercase, trimmed).
fn normalize_entity_key(text: &str) -> String {
    text.trim().to_lowercase()
}

/// Map a predicate string to an EdgeType.
fn predicate_to_edge_type(predicate: &str) -> EdgeType {
    let p = predicate.to_lowercase();
    match p.as_str() {
        "lives_in" | "moved_to" | "born_in" | "traveled_to" | "located_in" => EdgeType::RelatesTo,
        "works_at" | "works_for" | "employed_by" | "joined" => EdgeType::RelatesTo,
        "caused" | "caused_by" | "leads_to" => EdgeType::CausedBy,
        "conflicts_with" | "contradicts" => EdgeType::ConflictsWith,
        "follows" | "precedes" | "next" => EdgeType::NextEpisode,
        "has" | "has_property" | "owns" => EdgeType::HasProperty,
        "supersedes" | "replaces" => EdgeType::Supersedes,
        _ => EdgeType::RelatesTo,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex};
    use ucotron_core::{BackendRegistry, Edge, GraphBackend, Node, NodeId, VectorBackend};

    // ── Mock Embedding Pipeline ────────────────────────────────────────

    struct MockEmbedder;

    impl EmbeddingPipeline for MockEmbedder {
        fn embed_text(&self, text: &str) -> anyhow::Result<Vec<f32>> {
            // Deterministic: hash text to produce a normalized 384-dim vector
            let hash = text.bytes().fold(0u32, |acc, b| acc.wrapping_add(b as u32));
            let mut vec = vec![0.0f32; 384];
            for (i, v) in vec.iter_mut().enumerate() {
                *v = ((hash.wrapping_add(i as u32)) as f32).sin();
            }
            let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                for v in vec.iter_mut() {
                    *v /= norm;
                }
            }
            Ok(vec)
        }

        fn embed_batch(&self, texts: &[&str]) -> anyhow::Result<Vec<Vec<f32>>> {
            texts.iter().map(|t| self.embed_text(t)).collect()
        }
    }

    // ── Mock NER Pipeline ──────────────────────────────────────────────

    struct MockNer;

    impl NerPipeline for MockNer {
        fn extract_entities(
            &self,
            text: &str,
            _labels: &[&str],
        ) -> anyhow::Result<Vec<ExtractedEntity>> {
            // Simple: extract capitalized words as person/location entities
            let mut entities = Vec::new();
            for word in text.split_whitespace() {
                let clean = word.trim_matches(|c: char| !c.is_alphanumeric());
                if !clean.is_empty() && clean.chars().next().unwrap().is_uppercase() {
                    let label = if entities.len() % 2 == 0 {
                        "person"
                    } else {
                        "location"
                    };
                    let start = text.find(clean).unwrap_or(0);
                    entities.push(ExtractedEntity {
                        text: clean.to_string(),
                        label: label.to_string(),
                        start,
                        end: start + clean.len(),
                        confidence: 0.9,
                    });
                }
            }
            Ok(entities)
        }
    }

    // ── Mock Relation Extractor ────────────────────────────────────────

    struct MockRelationExtractor;

    impl RelationExtractor for MockRelationExtractor {
        fn extract_relations(
            &self,
            _text: &str,
            entities: &[ExtractedEntity],
        ) -> anyhow::Result<Vec<ExtractedRelation>> {
            let mut rels = Vec::new();
            if entities.len() >= 2 {
                rels.push(ExtractedRelation {
                    subject: entities[0].text.clone(),
                    predicate: "relates_to".into(),
                    object: entities[1].text.clone(),
                    confidence: 0.8,
                });
            }
            Ok(rels)
        }
    }

    // ── Mock Vector Backend ────────────────────────────────────────────

    struct MockVec {
        data: Mutex<HashMap<NodeId, Vec<f32>>>,
    }

    impl MockVec {
        fn new() -> Self {
            Self {
                data: Mutex::new(HashMap::new()),
            }
        }
    }

    impl VectorBackend for MockVec {
        fn upsert_embeddings(&self, items: &[(NodeId, Vec<f32>)]) -> anyhow::Result<()> {
            let mut data = self.data.lock().unwrap();
            for (id, emb) in items {
                data.insert(*id, emb.clone());
            }
            Ok(())
        }

        fn search(&self, query: &[f32], top_k: usize) -> anyhow::Result<Vec<(NodeId, f32)>> {
            let data = self.data.lock().unwrap();
            let mut results: Vec<(NodeId, f32)> = data
                .iter()
                .map(|(id, emb)| {
                    let sim: f32 = query.iter().zip(emb.iter()).map(|(a, b)| a * b).sum();
                    (*id, sim)
                })
                .collect();
            results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            results.truncate(top_k);
            Ok(results)
        }

        fn delete(&self, ids: &[NodeId]) -> anyhow::Result<()> {
            let mut data = self.data.lock().unwrap();
            for id in ids {
                data.remove(id);
            }
            Ok(())
        }
    }

    // ── Mock Graph Backend ─────────────────────────────────────────────

    struct MockGraph {
        nodes: Mutex<HashMap<NodeId, Node>>,
        edges: Mutex<Vec<Edge>>,
    }

    impl MockGraph {
        fn new() -> Self {
            Self {
                nodes: Mutex::new(HashMap::new()),
                edges: Mutex::new(Vec::new()),
            }
        }
    }

    impl GraphBackend for MockGraph {
        fn upsert_nodes(&self, nodes: &[Node]) -> anyhow::Result<()> {
            let mut data = self.nodes.lock().unwrap();
            for node in nodes {
                data.insert(node.id, node.clone());
            }
            Ok(())
        }

        fn upsert_edges(&self, edges: &[Edge]) -> anyhow::Result<()> {
            let mut data = self.edges.lock().unwrap();
            data.extend(edges.iter().cloned());
            Ok(())
        }

        fn get_node(&self, id: NodeId) -> anyhow::Result<Option<Node>> {
            let data = self.nodes.lock().unwrap();
            Ok(data.get(&id).cloned())
        }

        fn get_neighbors(&self, id: NodeId, _hops: u8) -> anyhow::Result<Vec<Node>> {
            let edges = self.edges.lock().unwrap();
            let nodes = self.nodes.lock().unwrap();
            let mut result = Vec::new();
            for edge in edges.iter() {
                if edge.source == id {
                    if let Some(node) = nodes.get(&edge.target) {
                        result.push(node.clone());
                    }
                } else if edge.target == id {
                    if let Some(node) = nodes.get(&edge.source) {
                        result.push(node.clone());
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
            let map = self.nodes.lock().unwrap();
            Ok(map.values().cloned().collect())
        }

        fn get_all_edges(&self) -> anyhow::Result<Vec<(NodeId, NodeId, f32)>> {
            let edges = self.edges.lock().unwrap();
            Ok(edges
                .iter()
                .map(|e| (e.source, e.target, e.weight))
                .collect())
        }

        fn delete_nodes(&self, ids: &[NodeId]) -> anyhow::Result<()> {
            let mut data = self.nodes.lock().unwrap();
            for id in ids {
                data.remove(id);
            }
            Ok(())
        }

        fn store_community_assignments(
            &self,
            _assignments: &HashMap<NodeId, ucotron_core::community::CommunityId>,
        ) -> anyhow::Result<()> {
            Ok(())
        }
    }

    // ── Helper ─────────────────────────────────────────────────────────

    fn boxed_registry() -> BackendRegistry {
        BackendRegistry::new(Box::new(MockVec::new()), Box::new(MockGraph::new()))
    }

    // ── chunk_text tests ───────────────────────────────────────────────

    #[test]
    fn test_chunk_text_simple_sentences() {
        let chunks = chunk_text("Hello world. How are you? I am fine!");
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0], "Hello world.");
        assert_eq!(chunks[1], "How are you?");
        assert_eq!(chunks[2], "I am fine!");
    }

    #[test]
    fn test_chunk_text_no_punctuation() {
        let chunks = chunk_text("This is a sentence without punctuation");
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], "This is a sentence without punctuation");
    }

    #[test]
    fn test_chunk_text_empty() {
        let chunks = chunk_text("");
        assert!(chunks.is_empty());
        let chunks2 = chunk_text("   ");
        assert!(chunks2.is_empty());
    }

    #[test]
    fn test_chunk_text_single_sentence() {
        let chunks = chunk_text("Juan se mudó de Madrid a Berlín en enero 2026.");
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].contains("Juan"));
    }

    // ── normalize_entity_key tests ─────────────────────────────────────

    #[test]
    fn test_normalize_entity_key() {
        assert_eq!(normalize_entity_key("Juan"), "juan");
        assert_eq!(normalize_entity_key("  Madrid  "), "madrid");
        assert_eq!(normalize_entity_key("SAP"), "sap");
    }

    // ── predicate_to_edge_type tests ───────────────────────────────────

    #[test]
    fn test_predicate_to_edge_type() {
        assert!(matches!(
            predicate_to_edge_type("lives_in"),
            EdgeType::RelatesTo
        ));
        assert!(matches!(
            predicate_to_edge_type("caused_by"),
            EdgeType::CausedBy
        ));
        assert!(matches!(
            predicate_to_edge_type("conflicts_with"),
            EdgeType::ConflictsWith
        ));
        assert!(matches!(
            predicate_to_edge_type("follows"),
            EdgeType::NextEpisode
        ));
        assert!(matches!(
            predicate_to_edge_type("has_property"),
            EdgeType::HasProperty
        ));
        assert!(matches!(
            predicate_to_edge_type("supersedes"),
            EdgeType::Supersedes
        ));
        assert!(matches!(
            predicate_to_edge_type("unknown_pred"),
            EdgeType::RelatesTo
        ));
    }

    // ── Full pipeline tests ────────────────────────────────────────────

    #[test]
    fn test_ingest_simple_text() {
        let registry = boxed_registry();
        let embedder = MockEmbedder;
        let ner = MockNer;
        let re = MockRelationExtractor;

        let config = IngestionConfig {
            enable_entity_resolution: false, // Disable for simple test
            ..Default::default()
        };

        let mut orchestrator =
            IngestionOrchestrator::new(&registry, &embedder, Some(&ner), Some(&re), config);

        let result = orchestrator
            .ingest("Juan lives in Madrid. He works at SAP.")
            .unwrap();

        assert!(result.metrics.chunks_processed >= 1);
        assert!(result.metrics.entities_extracted > 0);
        assert!(!result.chunk_node_ids.is_empty());
        assert!(result.metrics.total_us > 0);
    }

    #[test]
    fn test_ingest_empty_text() {
        let registry = boxed_registry();
        let embedder = MockEmbedder;

        let config = IngestionConfig {
            enable_entity_resolution: false,
            ..Default::default()
        };

        let mut orchestrator = IngestionOrchestrator::new(&registry, &embedder, None, None, config);

        let result = orchestrator.ingest("").unwrap();
        assert_eq!(result.metrics.chunks_processed, 0);
        assert!(result.chunk_node_ids.is_empty());
    }

    #[test]
    fn test_ingest_no_ner_pipeline() {
        let registry = boxed_registry();
        let embedder = MockEmbedder;

        let config = IngestionConfig {
            enable_entity_resolution: false,
            ..Default::default()
        };

        let mut orchestrator = IngestionOrchestrator::new(&registry, &embedder, None, None, config);

        let result = orchestrator
            .ingest("Some text with no NER extraction.")
            .unwrap();

        assert_eq!(result.metrics.chunks_processed, 1);
        assert_eq!(result.metrics.entities_extracted, 0);
        assert_eq!(result.metrics.relations_extracted, 0);
    }

    #[test]
    fn test_ingest_with_relations() {
        let registry = boxed_registry();
        let embedder = MockEmbedder;
        let ner = MockNer;
        let re = MockRelationExtractor;

        let config = IngestionConfig {
            enable_entity_resolution: false,
            enable_contradiction_detection: false,
            ..Default::default()
        };

        let mut orchestrator =
            IngestionOrchestrator::new(&registry, &embedder, Some(&ner), Some(&re), config);

        let result = orchestrator.ingest("Alice met Bob in London.").unwrap();

        assert!(result.metrics.relations_extracted > 0);
        assert!(!result.edges_created.is_empty());
    }

    #[test]
    fn test_ingest_metrics_timing() {
        let registry = boxed_registry();
        let embedder = MockEmbedder;
        let ner = MockNer;

        let config = IngestionConfig {
            enable_entity_resolution: false,
            ..Default::default()
        };

        let mut orchestrator =
            IngestionOrchestrator::new(&registry, &embedder, Some(&ner), None, config);

        let result = orchestrator
            .ingest("First sentence. Second sentence.")
            .unwrap();

        assert!(result.metrics.chunking_us > 0);
        assert!(result.metrics.embedding_us > 0);
        assert!(result.metrics.total_us > 0);
        assert!(result.metrics.total_us >= result.metrics.chunking_us);
    }

    #[test]
    fn test_ingest_multiple_texts() {
        let registry = boxed_registry();
        let embedder = MockEmbedder;
        let ner = MockNer;
        let re = MockRelationExtractor;

        let config = IngestionConfig {
            enable_entity_resolution: false,
            enable_contradiction_detection: false,
            ..Default::default()
        };

        let mut orchestrator =
            IngestionOrchestrator::new(&registry, &embedder, Some(&ner), Some(&re), config);

        // Ingest 3 texts as specified by PRD
        let texts = vec![
            "Juan se mudó de Madrid a Berlín en enero 2026.",
            "Ahora trabaja en SAP.",
            "María vive en Barcelona y estudia en la universidad.",
        ];

        let mut total_chunks = 0;
        let mut total_entities = 0;
        for text in texts {
            let result = orchestrator.ingest(text).unwrap();
            total_chunks += result.metrics.chunks_processed;
            total_entities += result.metrics.entities_extracted;
        }

        assert!(total_chunks >= 3);
        assert!(total_entities > 0);
    }

    #[test]
    fn test_ingest_partial_failure_continues() {
        // A NER pipeline that fails on specific chunks
        struct FailingNer;
        impl NerPipeline for FailingNer {
            fn extract_entities(
                &self,
                text: &str,
                _labels: &[&str],
            ) -> anyhow::Result<Vec<ExtractedEntity>> {
                if text.contains("FAIL") {
                    anyhow::bail!("Intentional NER failure");
                }
                Ok(vec![ExtractedEntity {
                    text: "Test".into(),
                    label: "person".into(),
                    start: 0,
                    end: 4,
                    confidence: 0.9,
                }])
            }
        }

        let registry = boxed_registry();
        let embedder = MockEmbedder;
        let ner = FailingNer;

        let config = IngestionConfig {
            enable_entity_resolution: false,
            ..Default::default()
        };

        let mut orchestrator =
            IngestionOrchestrator::new(&registry, &embedder, Some(&ner), None, config);

        // Second sentence will fail NER, but pipeline should continue
        let result = orchestrator
            .ingest("Good sentence. FAIL sentence. Another good one.")
            .unwrap();

        // Should have processed 3 chunks (one failed NER but chunk still created)
        assert_eq!(result.metrics.chunks_processed, 3);
        assert_eq!(result.metrics.chunks_failed, 1);
        // 2 successful NER extractions (chunk 0 and chunk 2)
        assert_eq!(result.metrics.entities_extracted, 2);
    }

    #[test]
    fn test_ingest_disabled_steps() {
        let registry = boxed_registry();
        let embedder = MockEmbedder;
        let ner = MockNer;
        let re = MockRelationExtractor;

        let config = IngestionConfig {
            enable_relations: false,
            enable_entity_resolution: false,
            enable_contradiction_detection: false,
            ..Default::default()
        };

        let mut orchestrator =
            IngestionOrchestrator::new(&registry, &embedder, Some(&ner), Some(&re), config);

        let result = orchestrator.ingest("Alice met Bob in London.").unwrap();

        // Relations disabled, so no relations should be extracted
        assert_eq!(result.metrics.relations_extracted, 0);
        // Contradiction detection disabled
        assert_eq!(result.metrics.contradictions_detected, 0);
    }

    #[test]
    fn test_entity_deduplication_across_chunks() {
        let registry = boxed_registry();
        let embedder = MockEmbedder;
        let ner = MockNer;

        let config = IngestionConfig {
            enable_entity_resolution: false,
            enable_contradiction_detection: false,
            ..Default::default()
        };

        let mut orchestrator =
            IngestionOrchestrator::new(&registry, &embedder, Some(&ner), None, config);

        // Both chunks mention "Juan" — should be deduplicated
        let result = orchestrator
            .ingest("Juan lives in Madrid. Juan works at SAP.")
            .unwrap();

        // Entities extracted from NER will include "Juan" twice,
        // but entity_nodes_created should deduplicate by name
        let unique_entities = result.entity_node_ids.len();
        // The MockNer extracts capitalized words, so both chunks have "Juan" and another entity
        // After dedup, "juan" should appear only once
        assert!(unique_entities > 0);
    }

    #[test]
    fn test_chunk_node_ids_match_chunk_count() {
        let registry = boxed_registry();
        let embedder = MockEmbedder;

        let config = IngestionConfig {
            enable_entity_resolution: false,
            ..Default::default()
        };

        let mut orchestrator = IngestionOrchestrator::new(&registry, &embedder, None, None, config);

        let result = orchestrator.ingest("First. Second. Third.").unwrap();

        assert_eq!(result.chunk_node_ids.len(), 3);
        // All IDs should be unique
        let mut ids = result.chunk_node_ids.clone();
        ids.sort();
        ids.dedup();
        assert_eq!(ids.len(), 3);
    }

    // ---- Edge-case tests ----

    #[test]
    fn test_chunk_text_only_whitespace() {
        let chunks = chunk_text("   \t\n  ");
        assert!(
            chunks.is_empty(),
            "Whitespace-only text should produce no chunks"
        );
    }

    #[test]
    fn test_chunk_text_decimal_numbers_preserved() {
        let chunks = chunk_text("The price is $99.99 for this item.");
        assert_eq!(chunks.len(), 1);
        assert!(
            chunks[0].contains("$99.99"),
            "Decimal price should not be split: {:?}",
            chunks
        );
    }

    #[test]
    fn test_chunk_text_version_numbers_preserved() {
        let chunks = chunk_text("Using version 2.0.1 of the library.");
        assert_eq!(chunks.len(), 1);
        assert!(
            chunks[0].contains("2.0.1"),
            "Version number should not be split: {:?}",
            chunks
        );
    }

    #[test]
    fn test_chunk_text_range_numbers_preserved() {
        let chunks = chunk_text("Growth from 8.2 to 6.5 percent.");
        assert_eq!(chunks.len(), 1);
        assert!(
            chunks[0].contains("8.2"),
            "First number should be intact: {:?}",
            chunks
        );
        assert!(
            chunks[0].contains("6.5"),
            "Second number should be intact: {:?}",
            chunks
        );
    }

    #[test]
    fn test_chunk_text_percentage_preserved() {
        let chunks = chunk_text("Inflation was 15.5 percent last year. It may decrease.");
        assert_eq!(chunks.len(), 2);
        assert!(
            chunks[0].contains("15.5"),
            "Percentage should not be split: {:?}",
            chunks
        );
    }

    #[test]
    fn test_chunk_text_consecutive_punctuation() {
        let chunks = chunk_text("Hello!! World?? Yes...");
        // Should not produce empty chunks between consecutive punctuation
        for chunk in &chunks {
            assert!(
                !chunk.trim().is_empty(),
                "Chunks should not be empty: {:?}",
                chunks
            );
        }
    }

    #[test]
    fn test_ingest_whitespace_only_text() {
        let registry = boxed_registry();
        let embedder = MockEmbedder;
        let config = IngestionConfig {
            enable_entity_resolution: false,
            ..Default::default()
        };
        let mut orchestrator = IngestionOrchestrator::new(&registry, &embedder, None, None, config);

        let result = orchestrator.ingest("   \n\t  ").unwrap();
        assert_eq!(
            result.chunk_node_ids.len(),
            0,
            "Whitespace text should produce no chunks"
        );
        assert_eq!(result.metrics.chunks_processed, 0);
    }

    #[test]
    fn test_ingest_unicode_text() {
        let registry = boxed_registry();
        let embedder = MockEmbedder;
        let ner = MockNer;
        let config = IngestionConfig {
            enable_entity_resolution: false,
            ..Default::default()
        };
        let mut orchestrator =
            IngestionOrchestrator::new(&registry, &embedder, Some(&ner), None, config);

        // Spanish text with accented characters
        let result = orchestrator
            .ingest("María vive en Berlín. Juan trabaja en París.")
            .unwrap();
        assert_eq!(result.chunk_node_ids.len(), 2);
        // MockNer extracts capitalized words, so should find entities
        assert!(
            !result.entity_node_ids.is_empty(),
            "Should extract entities from Unicode text"
        );
    }

    // ── Span instrumentation tests ────────────────────────────────────

    #[test]
    fn test_ingest_with_tracing_spans_no_panic() {
        // Verify that the instrumented ingestion pipeline runs correctly
        // with a tracing subscriber active — all spans are created and
        // recorded without panicking, and metrics are still accurate.
        // Spans are verified by running in a dedicated thread with an
        // isolated subscriber that captures span names and attributes.
        use std::sync::Arc;

        let span_names: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        let fields: Arc<Mutex<Vec<(String, String, String)>>> = Arc::new(Mutex::new(Vec::new()));

        let names_clone = span_names.clone();
        let fields_clone = fields.clone();

        let handle = std::thread::spawn(move || {
            let layer_names = SpanRecorderLayer { names: names_clone };
            let layer_fields = FieldRecorderLayer {
                fields: fields_clone,
            };
            let subscriber = tracing_subscriber::layer::SubscriberExt::with(
                tracing_subscriber::layer::SubscriberExt::with(
                    tracing_subscriber::registry::Registry::default(),
                    layer_names,
                ),
                layer_fields,
            );

            let _guard = tracing::subscriber::set_default(subscriber);

            let registry = boxed_registry();
            let embedder = MockEmbedder;
            let ner = MockNer;
            let re = MockRelationExtractor;
            let config = IngestionConfig {
                enable_entity_resolution: false,
                ..Default::default()
            };
            let mut orchestrator =
                IngestionOrchestrator::new(&registry, &embedder, Some(&ner), Some(&re), config);

            let result = orchestrator
                .ingest("Alice met Bob in Berlin. They visited the museum.")
                .unwrap();

            // Pipeline metrics should still be correct with spans active
            assert!(result.metrics.chunks_processed >= 1);
            assert!(result.metrics.entities_extracted > 0);
            assert!(result.metrics.total_us > 0);
            assert!(result.metrics.chunking_us > 0 || result.metrics.total_us > 0);
            assert!(!result.chunk_node_ids.is_empty());
        });

        handle.join().expect("Tracing span test thread panicked");

        // Verify all expected spans were created
        let names = span_names.lock().unwrap();
        assert!(
            names.contains(&"ucotron.ingest".to_string()),
            "Missing ucotron.ingest span. Got: {:?}",
            *names
        );
        assert!(
            names.contains(&"ucotron.chunk".to_string()),
            "Missing ucotron.chunk span. Got: {:?}",
            *names
        );
        assert!(
            names.contains(&"ucotron.embed".to_string()),
            "Missing ucotron.embed span. Got: {:?}",
            *names
        );
        assert!(
            names.contains(&"ucotron.ner".to_string()),
            "Missing ucotron.ner span. Got: {:?}",
            *names
        );
        assert!(
            names.contains(&"ucotron.relations".to_string()),
            "Missing ucotron.relations span. Got: {:?}",
            *names
        );
        assert!(
            names.contains(&"ucotron.graph_update".to_string()),
            "Missing ucotron.graph_update span. Got: {:?}",
            *names
        );

        // Verify key span attributes were populated
        let recorded = fields.lock().unwrap();

        let ingest_text_len = recorded
            .iter()
            .any(|(span, field, _)| span == "ucotron.ingest" && field == "text_length");
        assert!(
            ingest_text_len,
            "ucotron.ingest should have text_length attribute"
        );

        let chunk_count = recorded
            .iter()
            .any(|(span, field, _)| span == "ucotron.chunk" && field == "chunks");
        assert!(chunk_count, "ucotron.chunk should have chunks attribute");

        let embed_vectors = recorded
            .iter()
            .any(|(span, field, _)| span == "ucotron.embed" && field == "vectors");
        assert!(embed_vectors, "ucotron.embed should have vectors attribute");

        let ner_entities = recorded
            .iter()
            .any(|(span, field, _)| span == "ucotron.ner" && field == "entities_found");
        assert!(
            ner_entities,
            "ucotron.ner should have entities_found attribute"
        );

        let rel_count = recorded
            .iter()
            .any(|(span, field, _)| span == "ucotron.relations" && field == "relations_count");
        assert!(
            rel_count,
            "ucotron.relations should have relations_count attribute"
        );
    }

    // ── Tracing test helpers ──────────────────────────────────────────

    /// A tracing layer that records span names when they are created.
    struct SpanRecorderLayer {
        names: Arc<Mutex<Vec<String>>>,
    }

    impl<
            S: tracing::Subscriber + for<'lookup> tracing_subscriber::registry::LookupSpan<'lookup>,
        > tracing_subscriber::Layer<S> for SpanRecorderLayer
    {
        fn on_new_span(
            &self,
            attrs: &tracing::span::Attributes<'_>,
            _id: &tracing::span::Id,
            _ctx: tracing_subscriber::layer::Context<'_, S>,
        ) {
            let mut names = self.names.lock().unwrap();
            names.push(attrs.metadata().name().to_string());
        }
    }

    /// A tracing layer that records field values when they are set.
    struct FieldRecorderLayer {
        fields: Arc<Mutex<Vec<(String, String, String)>>>,
    }

    impl<
            S: tracing::Subscriber + for<'lookup> tracing_subscriber::registry::LookupSpan<'lookup>,
        > tracing_subscriber::Layer<S> for FieldRecorderLayer
    {
        fn on_new_span(
            &self,
            attrs: &tracing::span::Attributes<'_>,
            _id: &tracing::span::Id,
            _ctx: tracing_subscriber::layer::Context<'_, S>,
        ) {
            let span_name = attrs.metadata().name().to_string();
            let fields = self.fields.clone();
            attrs.record(&mut FieldVisitor { span_name, fields });
        }

        fn on_record(
            &self,
            id: &tracing::span::Id,
            values: &tracing::span::Record<'_>,
            ctx: tracing_subscriber::layer::Context<'_, S>,
        ) {
            if let Some(span) = ctx.span(id) {
                let span_name = span.name().to_string();
                let fields = self.fields.clone();
                values.record(&mut FieldVisitor { span_name, fields });
            }
        }
    }

    struct FieldVisitor {
        span_name: String,
        fields: Arc<Mutex<Vec<(String, String, String)>>>,
    }

    impl tracing::field::Visit for FieldVisitor {
        fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
            let mut fields = self.fields.lock().unwrap();
            fields.push((
                self.span_name.clone(),
                field.name().to_string(),
                format!("{:?}", value),
            ));
        }

        fn record_u64(&mut self, field: &tracing::field::Field, value: u64) {
            let mut fields = self.fields.lock().unwrap();
            fields.push((
                self.span_name.clone(),
                field.name().to_string(),
                value.to_string(),
            ));
        }

        fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
            let mut fields = self.fields.lock().unwrap();
            fields.push((
                self.span_name.clone(),
                field.name().to_string(),
                value.to_string(),
            ));
        }
    }

    // ── Batch NER tests ─────────────────────────────────────────────

    /// NER pipeline that tracks how many times batch vs individual was called.
    struct BatchTrackingNer {
        batch_calls: Mutex<usize>,
        individual_calls: Mutex<usize>,
    }

    impl BatchTrackingNer {
        fn new() -> Self {
            Self {
                batch_calls: Mutex::new(0),
                individual_calls: Mutex::new(0),
            }
        }
    }

    impl NerPipeline for BatchTrackingNer {
        fn extract_entities(
            &self,
            text: &str,
            _labels: &[&str],
        ) -> anyhow::Result<Vec<ExtractedEntity>> {
            *self.individual_calls.lock().unwrap() += 1;
            let mut entities = Vec::new();
            for word in text.split_whitespace() {
                let clean = word.trim_matches(|c: char| !c.is_alphanumeric());
                if !clean.is_empty() && clean.chars().next().unwrap().is_uppercase() {
                    let start = text.find(clean).unwrap_or(0);
                    entities.push(ExtractedEntity {
                        text: clean.to_string(),
                        label: "person".to_string(),
                        start,
                        end: start + clean.len(),
                        confidence: 0.9,
                    });
                }
            }
            Ok(entities)
        }

        fn extract_entities_batch(
            &self,
            texts: &[&str],
            labels: &[&str],
        ) -> anyhow::Result<Vec<Vec<ExtractedEntity>>> {
            *self.batch_calls.lock().unwrap() += 1;
            // Delegate to individual for correctness, but track that batch was called
            texts
                .iter()
                .map(|text| self.extract_entities(text, labels))
                .collect()
        }
    }

    #[test]
    fn test_ingest_uses_batch_ner() {
        let registry = boxed_registry();
        let embedder = MockEmbedder;
        let ner = BatchTrackingNer::new();

        let config = IngestionConfig {
            enable_entity_resolution: false,
            enable_contradiction_detection: false,
            ner_batch_size: 8,
            ..Default::default()
        };

        let mut orchestrator =
            IngestionOrchestrator::new(&registry, &embedder, Some(&ner), None, config);

        // 3 chunks, batch_size=8, so one batch call
        let result = orchestrator
            .ingest("Alice lives in Paris. Bob works in Berlin. Charlie visits London.")
            .unwrap();

        assert_eq!(result.metrics.chunks_processed, 3);
        assert!(*ner.batch_calls.lock().unwrap() > 0, "Should use batch NER");
    }

    #[test]
    fn test_ingest_batch_ner_multiple_batches() {
        let registry = boxed_registry();
        let embedder = MockEmbedder;
        let ner = BatchTrackingNer::new();

        let config = IngestionConfig {
            enable_entity_resolution: false,
            enable_contradiction_detection: false,
            ner_batch_size: 2, // Small batch size to force multiple batches
            ..Default::default()
        };

        let mut orchestrator =
            IngestionOrchestrator::new(&registry, &embedder, Some(&ner), None, config);

        // 3 chunks with batch_size=2 → 2 batch calls (batch of 2 + batch of 1)
        let result = orchestrator
            .ingest("Alice met Bob. Charlie met Diana. Eve met Frank.")
            .unwrap();

        assert_eq!(result.metrics.chunks_processed, 3);
        assert_eq!(
            *ner.batch_calls.lock().unwrap(),
            2,
            "Should make 2 batch calls for 3 chunks with batch_size=2"
        );
        assert!(result.metrics.entities_extracted > 0);
    }

    #[test]
    fn test_ingest_batch_ner_fallback_on_error() {
        /// NER pipeline where batch fails but individual works.
        struct FailingBatchNer;

        impl NerPipeline for FailingBatchNer {
            fn extract_entities(
                &self,
                text: &str,
                _labels: &[&str],
            ) -> anyhow::Result<Vec<ExtractedEntity>> {
                let mut entities = Vec::new();
                for word in text.split_whitespace() {
                    let clean = word.trim_matches(|c: char| !c.is_alphanumeric());
                    if !clean.is_empty() && clean.chars().next().unwrap().is_uppercase() {
                        let start = text.find(clean).unwrap_or(0);
                        entities.push(ExtractedEntity {
                            text: clean.to_string(),
                            label: "person".to_string(),
                            start,
                            end: start + clean.len(),
                            confidence: 0.9,
                        });
                    }
                }
                Ok(entities)
            }

            fn extract_entities_batch(
                &self,
                _texts: &[&str],
                _labels: &[&str],
            ) -> anyhow::Result<Vec<Vec<ExtractedEntity>>> {
                anyhow::bail!("Batch NER not supported")
            }
        }

        let registry = boxed_registry();
        let embedder = MockEmbedder;
        let ner = FailingBatchNer;

        let config = IngestionConfig {
            enable_entity_resolution: false,
            enable_contradiction_detection: false,
            ner_batch_size: 4,
            ..Default::default()
        };

        let mut orchestrator =
            IngestionOrchestrator::new(&registry, &embedder, Some(&ner), None, config);

        // Should fall back to per-chunk when batch fails
        let result = orchestrator
            .ingest("Alice met Bob. Charlie met Diana.")
            .unwrap();

        assert_eq!(result.metrics.chunks_processed, 2);
        assert!(
            result.metrics.entities_extracted > 0,
            "Should still extract entities via fallback"
        );
    }

    #[test]
    fn test_ingest_batch_size_config_default() {
        let config = IngestionConfig::default();
        assert_eq!(
            config.ner_batch_size, 8,
            "Default NER batch size should be 8"
        );
        assert_eq!(
            config.embedding_batch_size, 32,
            "Default embedding batch size should be 32"
        );
    }

    #[test]
    fn test_ingest_embedding_sub_batching() {
        // Tracks how many times embed_batch is called to verify sub-batching
        struct BatchCountingEmbedder {
            call_count: Mutex<usize>,
        }

        impl EmbeddingPipeline for BatchCountingEmbedder {
            fn embed_text(&self, text: &str) -> anyhow::Result<Vec<f32>> {
                let hash = text.bytes().fold(0u32, |acc, b| acc.wrapping_add(b as u32));
                let mut vec = vec![0.0f32; 384];
                for (i, v) in vec.iter_mut().enumerate() {
                    *v = ((hash.wrapping_add(i as u32)) as f32).sin();
                }
                let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm > 0.0 {
                    for v in vec.iter_mut() {
                        *v /= norm;
                    }
                }
                Ok(vec)
            }

            fn embed_batch(&self, texts: &[&str]) -> anyhow::Result<Vec<Vec<f32>>> {
                *self.call_count.lock().unwrap() += 1;
                texts.iter().map(|t| self.embed_text(t)).collect()
            }
        }

        let registry = boxed_registry();
        let embedder = BatchCountingEmbedder {
            call_count: Mutex::new(0),
        };

        let config = IngestionConfig {
            enable_entity_resolution: false,
            enable_contradiction_detection: false,
            embedding_batch_size: 2, // Small batch size to force sub-batching
            ..Default::default()
        };

        let mut orchestrator = IngestionOrchestrator::new(&registry, &embedder, None, None, config);

        // 3 chunks with embedding_batch_size=2 → 2 embed_batch calls (batch of 2 + batch of 1)
        let result = orchestrator
            .ingest("First sentence. Second sentence. Third sentence.")
            .unwrap();

        assert_eq!(result.metrics.chunks_processed, 3);
        assert_eq!(
            *embedder.call_count.lock().unwrap(),
            2,
            "Should make 2 embed_batch calls for 3 chunks with embedding_batch_size=2"
        );
    }

    #[test]
    fn test_ingest_batch_ner_entity_counts_match() {
        let registry = boxed_registry();
        let embedder = MockEmbedder;
        let ner = MockNer;

        // Batch size = 1 (effectively per-chunk) should produce same results as default
        let config_batch1 = IngestionConfig {
            enable_entity_resolution: false,
            enable_contradiction_detection: false,
            ner_batch_size: 1,
            ..Default::default()
        };

        let config_batch8 = IngestionConfig {
            enable_entity_resolution: false,
            enable_contradiction_detection: false,
            ner_batch_size: 8,
            ..Default::default()
        };

        let text = "Alice lives in Paris. Bob works in Berlin.";

        let mut orch1 =
            IngestionOrchestrator::new(&registry, &embedder, Some(&ner), None, config_batch1);
        let result1 = orch1.ingest(text).unwrap();

        let registry2 = boxed_registry();
        let mut orch2 =
            IngestionOrchestrator::new(&registry2, &embedder, Some(&ner), None, config_batch8);
        let result2 = orch2.ingest(text).unwrap();

        assert_eq!(
            result1.metrics.entities_extracted, result2.metrics.entities_extracted,
            "Entity count should be same regardless of batch size"
        );
        assert_eq!(
            result1.metrics.chunks_processed,
            result2.metrics.chunks_processed,
        );
    }
}
