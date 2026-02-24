//! Retrieval flow orchestrator for the Ucotron cognitive memory framework.
//!
//! Implements the LazyGraphRAG-variant retrieval pipeline:
//! 1. **Query Embedding** — Generate 384-dim vector for the query
//! 2. **Vector Search** — HNSW top-N for semantically similar chunks/entities
//! 3. **Entity Extraction** — GLiNER NER on the query to find mentioned entities
//! 4. **Graph Expansion** — 1-hop neighbors of matched entities + vector results
//! 5. **Community Selection** — Leiden clusters containing matched nodes
//! 6. **Re-ranking** — final = base×0.7 + mindset×0.15 + path_reward×0.15
//! 7. **Temporal Decay** — Penalize old unaccessed memories
//! 8. **Context Assembly** — Structured `RetrievalResult` with memories + entities
//!
//! Each step emits timing metrics. Optional steps (NER) can be disabled.

use std::collections::{HashMap, HashSet};
use std::time::Instant;

use anyhow::{Context, Result};
use tracing::{debug, info, info_span, warn};

use ucotron_core::{
    find_paths, BackendRegistry, MindsetDetector, MindsetScorer, MindsetTag, Node, NodeId,
    NodeType, PathFinderConfig, PathRewardCalculator, Value,
};

use crate::{EmbeddingPipeline, ExtractedEntity, NerPipeline};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the retrieval pipeline.
#[derive(Debug, Clone)]
pub struct RetrievalConfig {
    /// Number of top vector search results (seed nodes).
    pub vector_top_k: usize,
    /// Number of hops for graph expansion from seed nodes.
    pub graph_expansion_hops: u8,
    /// Whether to run entity extraction on the query.
    pub enable_entity_extraction: bool,
    /// Whether to include community members in results.
    pub enable_community_expansion: bool,
    /// Maximum number of community members to include per matched node.
    pub max_community_members: usize,
    /// NER labels for entity extraction from query.
    pub ner_labels: Vec<String>,
    /// Final top-k results to return after re-ranking.
    pub final_top_k: usize,
    /// Weights for re-ranking score components.
    pub vector_sim_weight: f32,
    /// Weight for graph centrality (degree-based) in re-ranking.
    pub graph_centrality_weight: f32,
    /// Weight for recency in re-ranking.
    pub recency_weight: f32,
    /// Temporal decay half-life in seconds (memories older than this lose half their recency score).
    pub temporal_decay_half_life_secs: u64,
    /// Minimum similarity threshold for vector results.
    pub min_similarity: f32,
    /// Optional namespace filter.
    pub namespace: Option<String>,
    /// Optional time range filter: (min_timestamp, max_timestamp).
    pub time_range: Option<(u64, u64)>,
    /// Optional entity type filter.
    pub entity_type_filter: Option<NodeType>,
    /// Optional cognitive mindset for mindset-aware scoring.
    /// When set, the MindsetScorer adjusts re-ranking weights based on the
    /// retrieval context (Convergent, Divergent, or Algorithmic).
    /// Default: None (standard scoring without mindset adjustment).
    pub query_mindset: Option<MindsetTag>,
}

impl Default for RetrievalConfig {
    fn default() -> Self {
        Self {
            vector_top_k: 50,
            graph_expansion_hops: 1,
            enable_entity_extraction: true,
            enable_community_expansion: true,
            max_community_members: 20,
            ner_labels: vec![
                "person".into(),
                "location".into(),
                "organization".into(),
                "date".into(),
                "concept".into(),
            ],
            final_top_k: 10,
            vector_sim_weight: 0.5,
            graph_centrality_weight: 0.3,
            recency_weight: 0.2,
            temporal_decay_half_life_secs: 30 * 24 * 3600, // 30 days
            min_similarity: 0.0,
            namespace: None,
            time_range: None,
            entity_type_filter: None,
            query_mindset: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Metrics
// ---------------------------------------------------------------------------

/// Timing and count metrics for a single retrieval run.
#[derive(Debug, Clone, Default)]
pub struct RetrievalMetrics {
    /// Microseconds spent embedding the query.
    pub query_embedding_us: u64,
    /// Microseconds spent on vector search.
    pub vector_search_us: u64,
    /// Microseconds spent on entity extraction from query.
    pub entity_extraction_us: u64,
    /// Microseconds spent on graph expansion.
    pub graph_expansion_us: u64,
    /// Microseconds spent on community selection.
    pub community_selection_us: u64,
    /// Microseconds spent on re-ranking.
    pub reranking_us: u64,
    /// Microseconds spent on context assembly.
    pub context_assembly_us: u64,
    /// Total pipeline duration in microseconds.
    pub total_us: u64,
    /// Number of vector search results (seeds).
    pub vector_results_count: usize,
    /// Number of entities extracted from query.
    pub query_entities_count: usize,
    /// Number of nodes after graph expansion.
    pub expanded_nodes_count: usize,
    /// Number of community members included.
    pub community_nodes_count: usize,
    /// Number of final results returned.
    pub final_results_count: usize,
}

// ---------------------------------------------------------------------------
// Scored memory
// ---------------------------------------------------------------------------

/// A memory node with its computed relevance score and component breakdown.
#[derive(Debug, Clone)]
pub struct ScoredMemory {
    /// The memory node.
    pub node: Node,
    /// Overall relevance score after re-ranking.
    pub score: f32,
    /// Vector similarity component (before weighting).
    pub vector_sim: f32,
    /// Graph centrality component (before weighting).
    pub graph_centrality: f32,
    /// Recency component (before weighting).
    pub recency: f32,
    /// Mindset score component (before weighting). Only present when query_mindset is set.
    pub mindset_score: f32,
    /// Path reward component (before weighting). Derived from graph path coherence,
    /// hop decay, and centrality between seed nodes and this node.
    pub path_reward_score: f32,
}

// ---------------------------------------------------------------------------
// Result type
// ---------------------------------------------------------------------------

/// Result of a retrieval query through the full pipeline.
#[derive(Debug)]
pub struct RetrievalResult {
    /// Metrics for this retrieval run.
    pub metrics: RetrievalMetrics,
    /// Ranked list of memory nodes with scores.
    pub memories: Vec<ScoredMemory>,
    /// Entity nodes mentioned in the query and found in the graph.
    pub entities: Vec<Node>,
    /// Assembled context text for LLM injection.
    pub context_text: String,
}

// ---------------------------------------------------------------------------
// Orchestrator
// ---------------------------------------------------------------------------

/// The retrieval orchestrator chains all query pipeline steps.
///
/// It is parameterized over trait objects so that:
/// - In production, real ONNX pipelines are used
/// - In tests, lightweight mocks can be substituted
pub struct RetrievalOrchestrator<'a> {
    registry: &'a BackendRegistry,
    embedder: &'a dyn EmbeddingPipeline,
    ner: Option<&'a dyn NerPipeline>,
    config: RetrievalConfig,
    mindset_detector: Option<MindsetDetector>,
}

impl<'a> RetrievalOrchestrator<'a> {
    /// Create a new retrieval orchestrator.
    ///
    /// NER is optional — pass `None` to skip entity extraction from query.
    pub fn new(
        registry: &'a BackendRegistry,
        embedder: &'a dyn EmbeddingPipeline,
        ner: Option<&'a dyn NerPipeline>,
        config: RetrievalConfig,
    ) -> Self {
        Self {
            registry,
            embedder,
            ner,
            config,
            mindset_detector: None,
        }
    }

    /// Set the mindset auto-detector for keyword-based mindset classification.
    ///
    /// When set and `query_mindset` is `None`, the detector scans the query
    /// for keyword patterns to automatically determine the cognitive mindset.
    pub fn with_mindset_detector(mut self, detector: MindsetDetector) -> Self {
        self.mindset_detector = Some(detector);
        self
    }

    /// Execute the full retrieval pipeline for a query string.
    pub fn retrieve(&self, query: &str) -> Result<RetrievalResult> {
        let retrieve_span = info_span!(
            "ucotron.retrieve",
            otel.kind = "internal",
            ucotron.pipeline = "retrieval",
            query_length = query.len(),
            top_k = self.config.final_top_k,
            nodes_visited = tracing::field::Empty,
            final_results = tracing::field::Empty,
            duration_us = tracing::field::Empty,
        );
        let _retrieve_guard = retrieve_span.enter();

        let pipeline_start = Instant::now();
        let mut metrics = RetrievalMetrics::default();
        let now_secs = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        // ── Step 0: Mindset Auto-Detection ──────────────────────────────
        // If no explicit mindset is set but a detector is configured,
        // scan the query for keyword patterns and analyze query structure
        // to automatically determine the cognitive mindset.
        let effective_mindset = self.config.query_mindset.or_else(|| {
            self.mindset_detector.as_ref().and_then(|detector| {
                let detected = detector.analyze(query);
                if let Some(tag) = detected {
                    debug!("Auto-detected mindset: {:?} from query", tag);
                }
                detected
            })
        });

        // ── Step 1: Query Embedding ─────────────────────────────────────
        let query_embedding = {
            let embed_span = info_span!(
                "ucotron.query_embed",
                otel.kind = "internal",
                query_length = query.len(),
                duration_us = tracing::field::Empty,
            );
            let _embed_guard = embed_span.enter();

            let embed_start = Instant::now();
            let embedding = self
                .embedder
                .embed_text(query)
                .context("Failed to embed query")?;
            metrics.query_embedding_us = embed_start.elapsed().as_micros() as u64;
            embed_span.record("duration_us", metrics.query_embedding_us);
            debug!("Query embedding: {}us", metrics.query_embedding_us);
            embedding
        };

        // ── Step 2: Vector Search ───────────────────────────────────────
        let vector_results = {
            let vs_span = info_span!(
                "ucotron.vector_search",
                otel.kind = "internal",
                top_k = self.config.vector_top_k,
                results = tracing::field::Empty,
                latency_ms = tracing::field::Empty,
                duration_us = tracing::field::Empty,
            );
            let _vs_guard = vs_span.enter();

            let vs_start = Instant::now();
            let raw_results = self
                .registry
                .vector()
                .search(&query_embedding, self.config.vector_top_k)
                .context("Vector search failed")?;
            metrics.vector_search_us = vs_start.elapsed().as_micros() as u64;

            // Filter by min_similarity
            let filtered: Vec<(NodeId, f32)> = raw_results
                .into_iter()
                .filter(|(_, sim)| *sim >= self.config.min_similarity)
                .collect();
            metrics.vector_results_count = filtered.len();

            vs_span.record("results", metrics.vector_results_count as u64);
            vs_span.record("latency_ms", metrics.vector_search_us / 1000);
            vs_span.record("duration_us", metrics.vector_search_us);
            debug!(
                "Vector search: {} results in {}us",
                metrics.vector_results_count, metrics.vector_search_us
            );
            filtered
        };

        // Build similarity map for re-ranking
        let mut similarity_map: HashMap<NodeId, f32> = HashMap::new();
        for (id, sim) in &vector_results {
            similarity_map.insert(*id, *sim);
        }

        // ── Step 3: Entity Extraction from Query ────────────────────────
        let (_query_entities, entity_graph_nodes, entity_node_ids) = {
            let ner_span = info_span!(
                "ucotron.query_ner",
                otel.kind = "internal",
                enabled = self.config.enable_entity_extraction,
                entities_found = tracing::field::Empty,
                graph_matches = tracing::field::Empty,
                duration_us = tracing::field::Empty,
            );
            let _ner_guard = ner_span.enter();

            let ee_start = Instant::now();
            let mut qe: Vec<ExtractedEntity> = Vec::new();
            let mut egn: Vec<Node> = Vec::new();
            let mut eni: HashSet<NodeId> = HashSet::new();

            if self.config.enable_entity_extraction {
                if let Some(ner) = self.ner {
                    if !self.config.ner_labels.is_empty() {
                        let labels: Vec<&str> =
                            self.config.ner_labels.iter().map(|s| s.as_str()).collect();
                        match ner.extract_entities(query, &labels) {
                            Ok(ents) => {
                                qe = ents;
                            }
                            Err(e) => {
                                warn!("Entity extraction from query failed: {}", e);
                            }
                        }
                    }
                }

                // For each extracted entity, search the graph for matching nodes
                for entity in &qe {
                    if let Ok(entity_emb) = self.embedder.embed_text(&entity.text) {
                        if let Ok(matches) = self.registry.vector().search(&entity_emb, 5) {
                            for (match_id, match_sim) in matches {
                                if let Ok(Some(node)) = self.registry.graph().get_node(match_id) {
                                    // Namespace isolation: skip entities from other namespaces (H9).
                                    if let Some(ref ns) = self.config.namespace {
                                        if !node_in_namespace(&node, ns) {
                                            continue;
                                        }
                                    }
                                    if matches!(node.node_type, NodeType::Entity) {
                                        let node_name = node.content.to_lowercase();
                                        let entity_name = entity.text.to_lowercase();
                                        if node_name.contains(&entity_name)
                                            || entity_name.contains(&node_name)
                                            || match_sim > 0.7
                                        {
                                            eni.insert(match_id);
                                            egn.push(node);
                                            similarity_map
                                                .entry(match_id)
                                                .and_modify(|s| *s = s.max(match_sim))
                                                .or_insert(match_sim);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            metrics.entity_extraction_us = ee_start.elapsed().as_micros() as u64;
            metrics.query_entities_count = qe.len();
            ner_span.record("entities_found", qe.len() as u64);
            ner_span.record("graph_matches", eni.len() as u64);
            ner_span.record("duration_us", metrics.entity_extraction_us);
            debug!(
                "Entity extraction: {} entities, {} graph matches in {}us",
                qe.len(),
                eni.len(),
                metrics.entity_extraction_us
            );
            (qe, egn, eni)
        };

        // ── Step 4: Graph Expansion ─────────────────────────────────────
        let (expanded_nodes, seed_ids) = {
            let ge_span = info_span!(
                "ucotron.graph_traverse",
                otel.kind = "internal",
                hops = self.config.graph_expansion_hops,
                seed_nodes = tracing::field::Empty,
                nodes_visited = tracing::field::Empty,
                duration_us = tracing::field::Empty,
            );
            let _ge_guard = ge_span.enter();

            let ge_start = Instant::now();
            let mut exp_nodes: HashMap<NodeId, Node> = HashMap::new();

            // Seed nodes: vector results + entity matches
            let mut seeds: HashSet<NodeId> = HashSet::new();
            for (id, _) in &vector_results {
                seeds.insert(*id);
            }
            for id in &entity_node_ids {
                seeds.insert(*id);
            }

            ge_span.record("seed_nodes", seeds.len() as u64);

            // Fetch full nodes for all seeds
            for &id in &seeds {
                if let Ok(Some(node)) = self.registry.graph().get_node(id) {
                    exp_nodes.insert(id, node);
                }
            }

            // Expand: get N-hop neighbors of all seeds
            if self.config.graph_expansion_hops > 0 {
                for &id in &seeds {
                    if let Ok(neighbors) = self
                        .registry
                        .graph()
                        .get_neighbors(id, self.config.graph_expansion_hops)
                    {
                        for neighbor in neighbors {
                            let nid = neighbor.id;
                            #[allow(clippy::map_entry)]
                            if !exp_nodes.contains_key(&nid) {
                                let seed_sim = similarity_map.get(&id).copied().unwrap_or(0.5);
                                let decay = 0.5_f32.powi(1); // 1-hop decay
                                similarity_map
                                    .entry(nid)
                                    .and_modify(|s| *s = s.max(seed_sim * decay))
                                    .or_insert(seed_sim * decay);
                                exp_nodes.insert(nid, neighbor);
                            }
                        }
                    }
                }
            }

            // ── Namespace isolation: filter expanded nodes ──────────────
            // BFS graph traversal follows edges without checking namespace,
            // which can pull in nodes from other tenants. Remove them here
            // to prevent cross-namespace data leaks (BUG-1 / H9).
            if let Some(ref ns) = self.config.namespace {
                let before = exp_nodes.len();
                exp_nodes.retain(|_id, node| node_in_namespace(node, ns));
                let removed = before - exp_nodes.len();
                if removed > 0 {
                    debug!(
                        "Namespace filter removed {} cross-namespace nodes from graph expansion",
                        removed
                    );
                }
            }

            metrics.graph_expansion_us = ge_start.elapsed().as_micros() as u64;
            metrics.expanded_nodes_count = exp_nodes.len();
            ge_span.record("nodes_visited", exp_nodes.len() as u64);
            ge_span.record("duration_us", metrics.graph_expansion_us);
            debug!(
                "Graph expansion: {} nodes in {}us",
                exp_nodes.len(),
                metrics.graph_expansion_us
            );
            (exp_nodes, seeds)
        };

        // ── Step 5: Community Selection ─────────────────────────────────
        let mut expanded_nodes = expanded_nodes;
        {
            let cs_span = info_span!(
                "ucotron.community",
                otel.kind = "internal",
                enabled = self.config.enable_community_expansion,
                community_nodes_added = tracing::field::Empty,
                duration_us = tracing::field::Empty,
            );
            let _cs_guard = cs_span.enter();

            let cs_start = Instant::now();
            let mut community_node_count = 0usize;

            if self.config.enable_community_expansion {
                let existing_ids: HashSet<NodeId> = expanded_nodes.keys().copied().collect();
                let mut community_candidates: Vec<NodeId> = Vec::new();

                for &id in &seed_ids {
                    if let Ok(members) = self.registry.graph().get_community(id) {
                        for member_id in members {
                            if !existing_ids.contains(&member_id)
                                && !community_candidates.contains(&member_id)
                            {
                                community_candidates.push(member_id);
                            }
                        }
                    }
                }

                // Limit community expansion
                community_candidates.truncate(self.config.max_community_members);

                for cid in &community_candidates {
                    if let Ok(Some(node)) = self.registry.graph().get_node(*cid) {
                        // Namespace isolation: skip community members from other namespaces (H9).
                        if let Some(ref ns) = self.config.namespace {
                            if !node_in_namespace(&node, ns) {
                                continue;
                            }
                        }
                        if !expanded_nodes.contains_key(cid) {
                            similarity_map.entry(*cid).or_insert(0.1);
                            expanded_nodes.insert(*cid, node);
                            community_node_count += 1;
                        }
                    }
                }
            }

            metrics.community_selection_us = cs_start.elapsed().as_micros() as u64;
            metrics.community_nodes_count = community_node_count;
            cs_span.record("community_nodes_added", community_node_count as u64);
            cs_span.record("duration_us", metrics.community_selection_us);
            debug!(
                "Community selection: {} nodes added in {}us",
                community_node_count, metrics.community_selection_us
            );
        }

        // ── Step 6: Re-ranking ──────────────────────────────────────────
        let scored = {
            let rr_span = info_span!(
                "ucotron.rerank",
                otel.kind = "internal",
                candidates = expanded_nodes.len(),
                final_results = tracing::field::Empty,
                duration_us = tracing::field::Empty,
            );
            let _rr_guard = rr_span.enter();

            let rr_start = Instant::now();

            // Compute graph centrality (simple degree-based: number of neighbors / max)
            let mut degree_map: HashMap<NodeId, usize> = HashMap::new();
            for &id in expanded_nodes.keys() {
                if let Ok(neighbors) = self.registry.graph().get_neighbors(id, 1) {
                    degree_map.insert(id, neighbors.len());
                }
            }
            let max_degree = degree_map.values().copied().max().unwrap_or(1).max(1);

            // Build mindset scorer if a mindset is configured (explicit or auto-detected)
            let mindset_scorer = effective_mindset.map(|_| MindsetScorer::default());

            // Build path reward calculator and pre-compute path rewards
            // for each candidate node relative to seed nodes.
            let path_reward_calc = PathRewardCalculator::default();
            let path_finder_config = PathFinderConfig {
                max_hops: 3,
                max_paths: 10,
            };
            let mut path_reward_map: HashMap<NodeId, f32> = HashMap::new();
            for &candidate_id in expanded_nodes.keys() {
                let mut best_reward: f32 = 0.0;
                for &seed_id in &seed_ids {
                    if seed_id == candidate_id {
                        // Seed node itself — give maximum path reward (no hops needed)
                        best_reward = 1.0;
                        break;
                    }
                    if let Ok(paths) = find_paths(
                        self.registry.graph(),
                        seed_id,
                        candidate_id,
                        &path_finder_config,
                    ) {
                        for path in &paths {
                            let degree_fn = |nid: NodeId| -> u32 {
                                degree_map.get(&nid).copied().unwrap_or(1) as u32
                            };
                            let reward = path_reward_calc.calculate_reward(path, &degree_fn);
                            best_reward = best_reward.max(reward.total);
                        }
                    }
                }
                path_reward_map.insert(candidate_id, best_reward);
            }

            // Score each node
            let mut s: Vec<ScoredMemory> = Vec::new();
            for (id, node) in &expanded_nodes {
                // Apply filters
                if let Some(ref time_range) = self.config.time_range {
                    if node.timestamp < time_range.0 || node.timestamp > time_range.1 {
                        continue;
                    }
                }
                if let Some(ref type_filter) = self.config.entity_type_filter {
                    if std::mem::discriminant(&node.node_type)
                        != std::mem::discriminant(type_filter)
                    {
                        continue;
                    }
                }

                let vector_sim = similarity_map.get(id).copied().unwrap_or(0.0);
                let degree = degree_map.get(id).copied().unwrap_or(0);
                let graph_centrality = degree as f32 / max_degree as f32;

                // Recency: exponential decay based on age
                let age_secs = now_secs.saturating_sub(node.timestamp);
                let recency = if self.config.temporal_decay_half_life_secs > 0 {
                    let half_lives =
                        age_secs as f64 / self.config.temporal_decay_half_life_secs as f64;
                    (0.5_f64.powf(half_lives)) as f32
                } else {
                    1.0
                };

                // Compute mindset score if configured (explicit or auto-detected)
                let mindset_score = match (&mindset_scorer, effective_mindset) {
                    (Some(scorer), Some(tag)) => {
                        // Use the node's confidence from metadata, defaulting to 0.5
                        let confidence = node
                            .metadata
                            .get("confidence")
                            .and_then(|v| match v {
                                ucotron_core::Value::Float(f) => Some(*f as f32),
                                _ => None,
                            })
                            .unwrap_or(0.5);
                        // diversity: inverse of vector_sim (less similar = more diverse)
                        let diversity = 1.0 - vector_sim;
                        scorer.score(tag, confidence, recency, diversity, graph_centrality)
                    }
                    _ => 0.0,
                };

                let base_score = vector_sim * self.config.vector_sim_weight
                    + graph_centrality * self.config.graph_centrality_weight
                    + recency * self.config.recency_weight;

                let path_reward = path_reward_map.get(id).copied().unwrap_or(0.0);

                // Combined scoring: base * 0.7 + mindset * 0.15 + path_reward * 0.15
                let score = base_score * 0.7 + mindset_score * 0.15 + path_reward * 0.15;

                s.push(ScoredMemory {
                    node: node.clone(),
                    score,
                    vector_sim,
                    graph_centrality,
                    recency,
                    mindset_score,
                    path_reward_score: path_reward,
                });
            }

            // Sort by score descending
            s.sort_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            s.truncate(self.config.final_top_k);

            metrics.reranking_us = rr_start.elapsed().as_micros() as u64;
            metrics.final_results_count = s.len();
            rr_span.record("final_results", s.len() as u64);
            rr_span.record("duration_us", metrics.reranking_us);
            debug!(
                "Re-ranking: {} results in {}us",
                s.len(),
                metrics.reranking_us
            );
            s
        };

        // ── Step 7+8: Context Assembly ──────────────────────────────────
        let ca_start = Instant::now();
        let context_text = assemble_context(&scored, &entity_graph_nodes);
        metrics.context_assembly_us = ca_start.elapsed().as_micros() as u64;

        metrics.total_us = pipeline_start.elapsed().as_micros() as u64;

        retrieve_span.record("nodes_visited", metrics.expanded_nodes_count as u64);
        retrieve_span.record("final_results", metrics.final_results_count as u64);
        retrieve_span.record("duration_us", metrics.total_us);

        info!(
            "Retrieval complete: {} results from {} expanded nodes in {}us (vector={}us, graph={}us, rank={}us)",
            metrics.final_results_count,
            metrics.expanded_nodes_count,
            metrics.total_us,
            metrics.vector_search_us,
            metrics.graph_expansion_us,
            metrics.reranking_us,
        );

        Ok(RetrievalResult {
            metrics,
            memories: scored,
            entities: entity_graph_nodes,
            context_text,
        })
    }
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

/// Check if a node belongs to the given namespace.
///
/// Returns `true` if the node's `_namespace` metadata matches, or if the node
/// has no `_namespace` metadata and the target namespace is "default".
fn node_in_namespace(node: &Node, namespace: &str) -> bool {
    match node.metadata.get("_namespace") {
        Some(Value::String(ns)) => ns == namespace,
        None => namespace == "default",
        _ => namespace == "default",
    }
}

/// Assemble a context string from scored memories and entity info.
///
/// Format:
/// ```text
/// ## Relevant Memories
/// [1] (score: 0.85) "Memory content here..."
/// [2] (score: 0.72) "Another memory..."
///
/// ## Known Entities
/// - EntityName (type: Entity)
/// ```
fn assemble_context(memories: &[ScoredMemory], entities: &[Node]) -> String {
    let mut ctx = String::new();

    if !memories.is_empty() {
        ctx.push_str("## Relevant Memories\n");
        for (i, mem) in memories.iter().enumerate() {
            ctx.push_str(&format!(
                "[{}] (score: {:.2}) \"{}\"\n",
                i + 1,
                mem.score,
                truncate_content(&mem.node.content, 200),
            ));
        }
    }

    if !entities.is_empty() {
        ctx.push_str("\n## Known Entities\n");
        for entity in entities {
            let label = entity
                .metadata
                .get("entity_label")
                .map(|v| format!("{:?}", v))
                .unwrap_or_else(|| format!("{:?}", entity.node_type));
            ctx.push_str(&format!("- {} ({})\n", entity.content, label));
        }
    }

    ctx
}

/// Truncate content to a maximum length, adding "..." if truncated.
fn truncate_content(content: &str, max_len: usize) -> String {
    if content.len() <= max_len {
        content.to_string()
    } else {
        // Find a safe char boundary
        let mut end = max_len;
        while !content.is_char_boundary(end) && end > 0 {
            end -= 1;
        }
        format!("{}...", &content[..end])
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
    use ucotron_core::{
        BackendRegistry, Edge, EdgeType, GraphBackend, Node, NodeId, NodeType, Value, VectorBackend,
    };

    // ── Mock Embedding Pipeline ────────────────────────────────────────

    struct MockEmbedder;

    impl EmbeddingPipeline for MockEmbedder {
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
        communities: Mutex<HashMap<NodeId, Vec<NodeId>>>,
    }

    impl MockGraph {
        fn new() -> Self {
            Self {
                nodes: Mutex::new(HashMap::new()),
                edges: Mutex::new(Vec::new()),
                communities: Mutex::new(HashMap::new()),
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
            let mut seen = HashSet::new();
            for edge in edges.iter() {
                if edge.source == id && !seen.contains(&edge.target) {
                    if let Some(node) = nodes.get(&edge.target) {
                        result.push(node.clone());
                        seen.insert(edge.target);
                    }
                } else if edge.target == id && !seen.contains(&edge.source) {
                    if let Some(node) = nodes.get(&edge.source) {
                        result.push(node.clone());
                        seen.insert(edge.source);
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

        fn get_community(&self, node_id: NodeId) -> anyhow::Result<Vec<NodeId>> {
            let communities = self.communities.lock().unwrap();
            Ok(communities.get(&node_id).cloned().unwrap_or_default())
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

        fn get_all_edges_full(&self) -> anyhow::Result<Vec<Edge>> {
            Ok(self.edges.lock().unwrap().clone())
        }

        fn delete_nodes(&self, ids: &[NodeId]) -> anyhow::Result<()> {
            let mut nodes = self.nodes.lock().unwrap();
            for id in ids {
                nodes.remove(id);
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

    // ── Test Helpers ───────────────────────────────────────────────────

    fn make_node(id: NodeId, content: &str, node_type: NodeType, timestamp: u64) -> Node {
        let embedder = MockEmbedder;
        let embedding = embedder.embed_text(content).unwrap();
        Node {
            id,
            content: content.to_string(),
            embedding,
            metadata: HashMap::new(),
            node_type,
            timestamp,
            media_type: None,
            media_uri: None,
            embedding_visual: None,
            timestamp_range: None,
            parent_video_id: None,
        }
    }

    fn make_entity_node(id: NodeId, name: &str, label: &str, timestamp: u64) -> Node {
        let embedder = MockEmbedder;
        let embedding = embedder.embed_text(name).unwrap();
        let mut metadata = HashMap::new();
        metadata.insert("entity_label".into(), Value::String(label.into()));
        Node {
            id,
            content: name.to_string(),
            embedding,
            metadata,
            node_type: NodeType::Entity,
            timestamp,
            media_type: None,
            media_uri: None,
            embedding_visual: None,
            timestamp_range: None,
            parent_video_id: None,
        }
    }

    fn now_secs() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs()
    }

    fn setup_populated_registry() -> (BackendRegistry, MockEmbedder) {
        let vec_backend = MockVec::new();
        let graph_backend = MockGraph::new();

        let now = now_secs();

        // Create some nodes
        let nodes = vec![
            make_node(
                1,
                "Juan moved from Madrid to Berlin in January 2026",
                NodeType::Event,
                now - 100,
            ),
            make_node(
                2,
                "Maria lives in Barcelona and studies at the university",
                NodeType::Event,
                now - 200,
            ),
            make_entity_node(3, "Juan", "person", now - 50),
            make_entity_node(4, "Madrid", "location", now - 50),
            make_entity_node(5, "Berlin", "location", now - 50),
            make_entity_node(6, "Maria", "person", now - 50),
            make_entity_node(7, "Barcelona", "location", now - 50),
            make_node(
                8,
                "SAP is a German software company",
                NodeType::Event,
                now - 300,
            ),
            make_entity_node(9, "SAP", "organization", now - 50),
        ];

        // Edges
        let edges = vec![
            Edge {
                source: 1,
                target: 3,
                edge_type: EdgeType::Actor,
                weight: 0.9,
                metadata: HashMap::new(),
            },
            Edge {
                source: 1,
                target: 4,
                edge_type: EdgeType::Location,
                weight: 0.8,
                metadata: HashMap::new(),
            },
            Edge {
                source: 1,
                target: 5,
                edge_type: EdgeType::Location,
                weight: 0.8,
                metadata: HashMap::new(),
            },
            Edge {
                source: 3,
                target: 9,
                edge_type: EdgeType::RelatesTo,
                weight: 0.7,
                metadata: HashMap::new(),
            },
            Edge {
                source: 2,
                target: 6,
                edge_type: EdgeType::Actor,
                weight: 0.9,
                metadata: HashMap::new(),
            },
            Edge {
                source: 2,
                target: 7,
                edge_type: EdgeType::Location,
                weight: 0.8,
                metadata: HashMap::new(),
            },
        ];

        // Insert into backends
        graph_backend.upsert_nodes(&nodes).unwrap();
        graph_backend.upsert_edges(&edges).unwrap();

        // Insert embeddings
        let embeddings: Vec<(NodeId, Vec<f32>)> =
            nodes.iter().map(|n| (n.id, n.embedding.clone())).collect();
        vec_backend.upsert_embeddings(&embeddings).unwrap();

        let registry = BackendRegistry::new(Box::new(vec_backend), Box::new(graph_backend));
        let embedder = MockEmbedder;

        (registry, embedder)
    }

    // ── Tests ──────────────────────────────────────────────────────────

    #[test]
    fn test_retrieve_basic() {
        let (registry, embedder) = setup_populated_registry();
        let config = RetrievalConfig {
            enable_entity_extraction: false,
            enable_community_expansion: false,
            final_top_k: 5,
            ..Default::default()
        };
        let orchestrator = RetrievalOrchestrator::new(&registry, &embedder, None, config);

        let result = orchestrator.retrieve("Where does Juan live?").unwrap();

        assert!(result.metrics.total_us > 0);
        assert!(result.metrics.vector_search_us > 0);
        assert!(result.metrics.query_embedding_us > 0);
        assert!(!result.memories.is_empty());
        assert!(result.metrics.final_results_count > 0);
        assert!(result.metrics.final_results_count <= 5);
    }

    #[test]
    fn test_retrieve_empty_query() {
        let (registry, embedder) = setup_populated_registry();
        let config = RetrievalConfig {
            enable_entity_extraction: false,
            enable_community_expansion: false,
            ..Default::default()
        };
        let orchestrator = RetrievalOrchestrator::new(&registry, &embedder, None, config);

        let result = orchestrator.retrieve("").unwrap();
        assert!(result.metrics.total_us > 0);
    }

    #[test]
    fn test_retrieve_with_ner() {
        let (registry, embedder) = setup_populated_registry();
        let ner = MockNer;
        let config = RetrievalConfig {
            enable_community_expansion: false,
            final_top_k: 10,
            ..Default::default()
        };
        let orchestrator = RetrievalOrchestrator::new(&registry, &embedder, Some(&ner), config);

        let result = orchestrator.retrieve("Where does Juan work?").unwrap();

        assert!(result.metrics.query_entities_count > 0);
        assert!(result.metrics.total_us > 0);
    }

    #[test]
    fn test_retrieve_scores_ordered() {
        let (registry, embedder) = setup_populated_registry();
        let config = RetrievalConfig {
            enable_entity_extraction: false,
            enable_community_expansion: false,
            final_top_k: 10,
            ..Default::default()
        };
        let orchestrator = RetrievalOrchestrator::new(&registry, &embedder, None, config);

        let result = orchestrator.retrieve("Juan Berlin move").unwrap();

        // Scores should be in descending order
        for window in result.memories.windows(2) {
            assert!(
                window[0].score >= window[1].score,
                "Scores not in descending order: {} < {}",
                window[0].score,
                window[1].score
            );
        }
    }

    #[test]
    fn test_retrieve_with_time_range_filter() {
        let (registry, embedder) = setup_populated_registry();
        let now = now_secs();
        let config = RetrievalConfig {
            enable_entity_extraction: false,
            enable_community_expansion: false,
            time_range: Some((now - 150, now)),
            final_top_k: 10,
            ..Default::default()
        };
        let orchestrator = RetrievalOrchestrator::new(&registry, &embedder, None, config);

        let result = orchestrator.retrieve("Juan Berlin").unwrap();

        // All returned nodes should have timestamps in range
        for mem in &result.memories {
            assert!(mem.node.timestamp >= now - 150);
            assert!(mem.node.timestamp <= now);
        }
    }

    #[test]
    fn test_retrieve_with_entity_type_filter() {
        let (registry, embedder) = setup_populated_registry();
        let config = RetrievalConfig {
            enable_entity_extraction: false,
            enable_community_expansion: false,
            entity_type_filter: Some(NodeType::Entity),
            final_top_k: 10,
            ..Default::default()
        };
        let orchestrator = RetrievalOrchestrator::new(&registry, &embedder, None, config);

        let result = orchestrator.retrieve("Juan Madrid").unwrap();

        for mem in &result.memories {
            assert!(
                matches!(mem.node.node_type, NodeType::Entity),
                "Expected Entity type, got {:?}",
                mem.node.node_type
            );
        }
    }

    #[test]
    fn test_retrieve_graph_expansion() {
        let (registry, embedder) = setup_populated_registry();
        let config = RetrievalConfig {
            enable_entity_extraction: false,
            enable_community_expansion: false,
            graph_expansion_hops: 1,
            final_top_k: 20,
            ..Default::default()
        };
        let orchestrator = RetrievalOrchestrator::new(&registry, &embedder, None, config);

        let result = orchestrator.retrieve("Juan Berlin").unwrap();

        // With graph expansion, should get more results than just vector seeds
        assert!(result.metrics.expanded_nodes_count >= result.metrics.vector_results_count);
    }

    #[test]
    fn test_retrieve_no_graph_expansion() {
        let (registry, embedder) = setup_populated_registry();
        let config = RetrievalConfig {
            enable_entity_extraction: false,
            enable_community_expansion: false,
            graph_expansion_hops: 0,
            final_top_k: 20,
            ..Default::default()
        };
        let orchestrator = RetrievalOrchestrator::new(&registry, &embedder, None, config);

        let result = orchestrator.retrieve("Juan Berlin").unwrap();

        // Without expansion, expanded == vector results
        assert_eq!(
            result.metrics.expanded_nodes_count,
            result.metrics.vector_results_count
        );
    }

    #[test]
    fn test_retrieve_metrics_timing() {
        let (registry, embedder) = setup_populated_registry();
        let config = RetrievalConfig {
            enable_entity_extraction: false,
            enable_community_expansion: false,
            ..Default::default()
        };
        let orchestrator = RetrievalOrchestrator::new(&registry, &embedder, None, config);

        let result = orchestrator.retrieve("test query").unwrap();
        let m = &result.metrics;

        assert!(m.query_embedding_us > 0);
        assert!(m.vector_search_us > 0);
        assert!(m.total_us > 0);
        assert!(m.total_us >= m.query_embedding_us + m.vector_search_us);
    }

    #[test]
    fn test_retrieve_context_assembly() {
        let (registry, embedder) = setup_populated_registry();
        let config = RetrievalConfig {
            enable_entity_extraction: false,
            enable_community_expansion: false,
            final_top_k: 3,
            ..Default::default()
        };
        let orchestrator = RetrievalOrchestrator::new(&registry, &embedder, None, config);

        let result = orchestrator.retrieve("Juan Berlin").unwrap();

        assert!(!result.context_text.is_empty());
        assert!(result.context_text.contains("Relevant Memories"));
    }

    #[test]
    fn test_retrieve_score_components() {
        let (registry, embedder) = setup_populated_registry();
        let config = RetrievalConfig {
            enable_entity_extraction: false,
            enable_community_expansion: false,
            final_top_k: 5,
            ..Default::default()
        };
        let orchestrator = RetrievalOrchestrator::new(&registry, &embedder, None, config);

        let result = orchestrator.retrieve("Juan Berlin").unwrap();

        for mem in &result.memories {
            // Score should be: base * 0.7 + mindset * 0.15 + path_reward * 0.15
            let base = mem.vector_sim * 0.5 + mem.graph_centrality * 0.3 + mem.recency * 0.2;
            let expected = base * 0.7 + mem.mindset_score * 0.15 + mem.path_reward_score * 0.15;
            assert!(
                (mem.score - expected).abs() < 0.01,
                "Score mismatch: {} vs expected {} (base={}, mindset={}, path_reward={})",
                mem.score,
                expected,
                base,
                mem.mindset_score,
                mem.path_reward_score,
            );
        }
    }

    #[test]
    fn test_retrieve_temporal_decay() {
        let vec_backend = MockVec::new();
        let graph_backend = MockGraph::new();
        let now = now_secs();

        // Create a recent node and an old node with same content similarity
        let recent = make_node(1, "recent memory about cats", NodeType::Event, now - 60);
        let old = make_node(
            2,
            "old memory about cats",
            NodeType::Event,
            now - 365 * 24 * 3600,
        );

        graph_backend
            .upsert_nodes(&[recent.clone(), old.clone()])
            .unwrap();
        vec_backend
            .upsert_embeddings(&[(1, recent.embedding.clone()), (2, old.embedding.clone())])
            .unwrap();

        let registry = BackendRegistry::new(Box::new(vec_backend), Box::new(graph_backend));
        let embedder = MockEmbedder;

        let config = RetrievalConfig {
            enable_entity_extraction: false,
            enable_community_expansion: false,
            temporal_decay_half_life_secs: 30 * 24 * 3600, // 30 days
            final_top_k: 10,
            ..Default::default()
        };
        let orchestrator = RetrievalOrchestrator::new(&registry, &embedder, None, config);

        let result = orchestrator.retrieve("cats").unwrap();

        // Both should appear, but recent should have higher recency
        if result.memories.len() >= 2 {
            let recent_mem = result.memories.iter().find(|m| m.node.id == 1);
            let old_mem = result.memories.iter().find(|m| m.node.id == 2);
            if let (Some(r), Some(o)) = (recent_mem, old_mem) {
                assert!(
                    r.recency > o.recency,
                    "Recent memory recency ({}) should be > old memory recency ({})",
                    r.recency,
                    o.recency
                );
            }
        }
    }

    #[test]
    fn test_retrieve_with_community_expansion() {
        let vec_backend = MockVec::new();
        let graph_backend = MockGraph::new();
        let now = now_secs();

        let node1 = make_node(1, "Juan lives in Berlin", NodeType::Event, now);
        let node2 = make_node(2, "Berlin is the capital of Germany", NodeType::Event, now);
        let node3 = make_node(3, "Germany has many tech companies", NodeType::Event, now);

        graph_backend
            .upsert_nodes(&[node1.clone(), node2.clone(), node3.clone()])
            .unwrap();
        vec_backend
            .upsert_embeddings(&[
                (1, node1.embedding.clone()),
                (2, node2.embedding.clone()),
                (3, node3.embedding.clone()),
            ])
            .unwrap();

        // Set up communities: node 1 and 2 are in same community, node 3 is separate
        {
            let mut communities = graph_backend.communities.lock().unwrap();
            communities.insert(1, vec![1, 2]);
            communities.insert(2, vec![1, 2]);
            communities.insert(3, vec![3]);
        }

        let registry = BackendRegistry::new(Box::new(vec_backend), Box::new(graph_backend));
        let embedder = MockEmbedder;

        let config = RetrievalConfig {
            enable_entity_extraction: false,
            enable_community_expansion: true,
            max_community_members: 10,
            final_top_k: 10,
            ..Default::default()
        };
        let orchestrator = RetrievalOrchestrator::new(&registry, &embedder, None, config);

        let result = orchestrator.retrieve("Juan Berlin").unwrap();

        // Community expansion should potentially add more nodes
        assert!(result.metrics.total_us > 0);
    }

    #[test]
    fn test_truncate_content() {
        assert_eq!(truncate_content("short", 10), "short");
        assert_eq!(truncate_content("this is long text", 7), "this is...");
        assert_eq!(truncate_content("", 10), "");
    }

    #[test]
    fn test_assemble_context_empty() {
        let ctx = assemble_context(&[], &[]);
        assert_eq!(ctx, "");
    }

    #[test]
    fn test_assemble_context_with_memories() {
        let node = make_node(1, "test memory", NodeType::Event, now_secs());
        let memories = vec![ScoredMemory {
            node,
            score: 0.85,
            vector_sim: 0.9,
            graph_centrality: 0.5,
            recency: 1.0,
            mindset_score: 0.0,
            path_reward_score: 0.0,
        }];
        let ctx = assemble_context(&memories, &[]);
        assert!(ctx.contains("Relevant Memories"));
        assert!(ctx.contains("0.85"));
        assert!(ctx.contains("test memory"));
    }

    // ---- Edge-case tests ----

    #[test]
    fn test_truncate_content_unicode() {
        // Truncation by byte length should not break multi-byte UTF-8 characters
        let text = "café niño año"; // contains 2-byte chars (é, ñ)
        let truncated = truncate_content(text, 6); // truncate at byte 6
                                                   // Should produce valid UTF-8 (may back up to char boundary) + "..."
        assert!(truncated.ends_with("..."), "Should have ellipsis suffix");
        // Verify the truncated prefix is valid UTF-8 (it will be since we sliced at char boundary)
    }

    #[test]
    fn test_truncate_content_exact_length() {
        let text = "twelve chars";
        let truncated = truncate_content(text, 12);
        assert_eq!(truncated, "twelve chars"); // no truncation when len == max_len
    }

    #[test]
    fn test_retrieve_config_defaults() {
        let config = RetrievalConfig::default();
        assert_eq!(config.vector_top_k, 50);
        assert_eq!(config.graph_expansion_hops, 1);
        assert_eq!(config.final_top_k, 10);
        assert!(config.min_similarity >= 0.0);
    }

    #[test]
    fn test_scored_memory_ordering() {
        // Verify that ScoredMemory can be compared by score
        let node1 = make_node(1, "high score", NodeType::Event, now_secs());
        let node2 = make_node(2, "low score", NodeType::Event, now_secs());
        let m1 = ScoredMemory {
            node: node1,
            score: 0.9,
            vector_sim: 0.9,
            graph_centrality: 0.5,
            recency: 1.0,
            mindset_score: 0.0,
            path_reward_score: 0.0,
        };
        let m2 = ScoredMemory {
            node: node2,
            score: 0.3,
            vector_sim: 0.3,
            graph_centrality: 0.1,
            recency: 0.5,
            mindset_score: 0.0,
            path_reward_score: 0.0,
        };
        assert!(m1.score > m2.score);
    }

    // ---- Mindset-aware retrieval tests ----

    #[test]
    fn test_retrieve_with_mindset_convergent() {
        let (registry, embedder) = setup_populated_registry();
        let config = RetrievalConfig {
            enable_entity_extraction: false,
            enable_community_expansion: false,
            query_mindset: Some(ucotron_core::MindsetTag::Convergent),
            final_top_k: 5,
            ..Default::default()
        };
        let orchestrator = RetrievalOrchestrator::new(&registry, &embedder, None, config);
        let result = orchestrator.retrieve("Where does someone live?").unwrap();
        assert!(!result.memories.is_empty());
        // With mindset scoring, mindset_score should be non-zero for at least some results
        let has_mindset = result.memories.iter().any(|m| m.mindset_score > 0.0);
        assert!(
            has_mindset,
            "Expected non-zero mindset_score with Convergent mindset"
        );
    }

    #[test]
    fn test_retrieve_without_mindset_has_zero_mindset_score() {
        let (registry, embedder) = setup_populated_registry();
        let config = RetrievalConfig {
            enable_entity_extraction: false,
            enable_community_expansion: false,
            query_mindset: None,
            final_top_k: 5,
            ..Default::default()
        };
        let orchestrator = RetrievalOrchestrator::new(&registry, &embedder, None, config);
        let result = orchestrator.retrieve("Where does someone live?").unwrap();
        assert!(!result.memories.is_empty());
        // Without mindset, all mindset_scores should be 0.0
        for m in &result.memories {
            assert_eq!(
                m.mindset_score, 0.0,
                "Expected 0.0 mindset_score without mindset"
            );
        }
    }

    // ---- Tracing span tests ----

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
    }

    /// Verify that the retrieval pipeline runs correctly with tracing spans
    /// active. This test ensures the span instrumentation doesn't cause
    /// panics or change pipeline behavior.
    ///
    /// NOTE: To verify span names and attributes are correct, run with:
    ///   RUST_TEST_THREADS=1 cargo test -p ucotron-extraction --lib retrieval::tests::test_retrieve_tracing_span_names
    /// The tracing callsite cache is global and can be poisoned by concurrent
    /// tests that call retrieve() without a subscriber, causing info_span!
    /// to return disabled spans. See: https://github.com/tokio-rs/tracing/issues/2874
    #[test]
    fn test_retrieve_with_tracing_spans_no_panic() {
        let (registry, embedder) = setup_populated_registry();
        let ner = MockNer;
        let config = RetrievalConfig {
            enable_entity_extraction: true,
            enable_community_expansion: true,
            final_top_k: 5,
            ..Default::default()
        };
        let orchestrator = RetrievalOrchestrator::new(&registry, &embedder, Some(&ner), config);

        // The pipeline should succeed with or without an active tracing subscriber
        let result = orchestrator
            .retrieve("Where does Juan live in Berlin?")
            .unwrap();

        assert!(result.metrics.total_us > 0);
        assert!(result.metrics.vector_search_us > 0);
        assert!(result.metrics.query_embedding_us > 0);
        assert!(!result.memories.is_empty());
        assert!(result.metrics.final_results_count > 0);
        assert!(result.metrics.expanded_nodes_count > 0);
    }

    /// Verify that all expected tracing spans and attributes are created
    /// during retrieval. This test must run in isolation (RUST_TEST_THREADS=1)
    /// due to the tracing callsite cache being global and shared across threads.
    #[test]
    #[ignore] // Run with: RUST_TEST_THREADS=1 cargo test -- --ignored test_retrieve_tracing_span_names
    fn test_retrieve_tracing_span_names() {
        use std::sync::Arc;

        let span_names: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        let fields: Arc<Mutex<Vec<(String, String, String)>>> = Arc::new(Mutex::new(Vec::new()));

        let names_clone = span_names.clone();
        let fields_clone = fields.clone();

        let handle = std::thread::spawn(move || {
            let layer_names = SpanRecorderLayer {
                names: names_clone.clone(),
            };
            let layer_fields = FieldRecorderLayer {
                fields: fields_clone.clone(),
            };
            let subscriber = tracing_subscriber::layer::SubscriberExt::with(
                tracing_subscriber::layer::SubscriberExt::with(
                    tracing_subscriber::registry::Registry::default(),
                    layer_names,
                ),
                layer_fields,
            );

            tracing::subscriber::with_default(subscriber, || {
                tracing::callsite::rebuild_interest_cache();

                let (registry, embedder) = setup_populated_registry();
                let ner = MockNer;
                let config = RetrievalConfig {
                    enable_entity_extraction: true,
                    enable_community_expansion: true,
                    final_top_k: 5,
                    ..Default::default()
                };
                let orchestrator =
                    RetrievalOrchestrator::new(&registry, &embedder, Some(&ner), config);

                let result = orchestrator
                    .retrieve("Where does Juan live in Berlin?")
                    .unwrap();
                assert!(!result.memories.is_empty());

                // Verify all expected spans
                let names = names_clone.lock().unwrap();
                assert!(
                    names.contains(&"ucotron.retrieve".to_string()),
                    "Missing ucotron.retrieve. Got: {:?}",
                    *names
                );
                assert!(
                    names.contains(&"ucotron.vector_search".to_string()),
                    "Missing ucotron.vector_search. Got: {:?}",
                    *names
                );
                assert!(
                    names.contains(&"ucotron.graph_traverse".to_string()),
                    "Missing ucotron.graph_traverse. Got: {:?}",
                    *names
                );
                assert!(
                    names.contains(&"ucotron.community".to_string()),
                    "Missing ucotron.community. Got: {:?}",
                    *names
                );
                assert!(
                    names.contains(&"ucotron.query_embed".to_string()),
                    "Missing ucotron.query_embed. Got: {:?}",
                    *names
                );
                assert!(
                    names.contains(&"ucotron.rerank".to_string()),
                    "Missing ucotron.rerank. Got: {:?}",
                    *names
                );

                // Verify key attributes
                let recorded = fields_clone.lock().unwrap();
                assert!(recorded
                    .iter()
                    .any(|(s, f, _)| s == "ucotron.retrieve" && f == "query_length"));
                assert!(recorded
                    .iter()
                    .any(|(s, f, _)| s == "ucotron.vector_search" && f == "top_k"));
                assert!(recorded
                    .iter()
                    .any(|(s, f, _)| s == "ucotron.vector_search" && f == "results"));
                assert!(recorded
                    .iter()
                    .any(|(s, f, _)| s == "ucotron.graph_traverse" && f == "nodes_visited"));
                assert!(recorded
                    .iter()
                    .any(|(s, f, _)| s == "ucotron.retrieve" && f == "final_results"));
            });
        });

        handle.join().expect("Tracing span test thread panicked");
    }

    // ---- Mindset-aware retrieval benchmark (US-27.13) ----

    /// Builds a rich graph designed for benchmarking mindset-aware retrieval.
    ///
    /// The graph contains 30 nodes across 3 thematic clusters:
    /// - **Convergent cluster** (nodes 1-10): well-connected, high-confidence facts about
    ///   a person's career (multiple corroborating paths). Best retrieved under Convergent mindset.
    /// - **Divergent cluster** (nodes 11-20): loosely connected, novel/contradictory facts about
    ///   travel (rare predicates, contradictions). Best retrieved under Divergent mindset.
    /// - **Algorithmic cluster** (nodes 21-30): recent, verified, high-precision facts about
    ///   technical topics. Best retrieved under Algorithmic mindset.
    ///
    /// Returns (registry, embedder, ground_truth) where ground_truth maps
    /// (query_text, expected_mindset) → relevant_node_ids.
    #[allow(clippy::type_complexity)]
    fn setup_benchmark_registry() -> (
        BackendRegistry,
        MockEmbedder,
        Vec<(&'static str, Option<MindsetTag>, Vec<u64>)>,
    ) {
        let vec_backend = MockVec::new();
        let graph_backend = MockGraph::new();
        let now = now_secs();
        let embedder = MockEmbedder;

        // ── Convergent cluster: career facts (well-connected, high confidence) ──
        let mut nodes = vec![
            make_node(
                1,
                "Alice works at Google as a software engineer since 2020",
                NodeType::Event,
                now - 100,
            ),
            make_entity_node(2, "Alice", "person", now - 50),
            make_entity_node(3, "Google", "organization", now - 50),
            make_node(
                4,
                "Alice graduated from MIT with a computer science degree",
                NodeType::Event,
                now - 200,
            ),
            make_entity_node(5, "MIT", "organization", now - 50),
            make_node(
                6,
                "Alice published a paper on distributed systems at Google",
                NodeType::Event,
                now - 150,
            ),
            make_node(
                7,
                "Alice received a promotion at Google in 2023",
                NodeType::Event,
                now - 80,
            ),
            make_node(
                8,
                "Alice leads the infrastructure team at Google",
                NodeType::Event,
                now - 60,
            ),
            make_entity_node(9, "distributed systems", "topic", now - 50),
            make_entity_node(10, "infrastructure", "topic", now - 50),
        ];

        // ── Divergent cluster: travel (loosely connected, contradictory, novel) ──
        nodes.extend(vec![
            make_node(
                11,
                "Bob traveled to Japan and loved the sushi in Tokyo",
                NodeType::Event,
                now - 3000,
            ),
            make_entity_node(12, "Bob", "person", now - 50),
            make_entity_node(13, "Japan", "location", now - 50),
            make_entity_node(14, "Tokyo", "location", now - 50),
            make_node(
                15,
                "Bob said he disliked Japanese food surprisingly",
                NodeType::Event,
                now - 2500,
            ),
            make_node(
                16,
                "Bob explored hidden temples in rural Kyoto alone",
                NodeType::Event,
                now - 2800,
            ),
            make_entity_node(17, "Kyoto", "location", now - 50),
            make_node(
                18,
                "Bob found an unusual cafe that serves insects in Bangkok",
                NodeType::Event,
                now - 2000,
            ),
            make_entity_node(19, "Bangkok", "location", now - 50),
            make_node(
                20,
                "Bob contradicted himself about liking travel",
                NodeType::Event,
                now - 1500,
            ),
        ]);

        // ── Algorithmic cluster: technical (recent, verified, precise) ──
        nodes.extend(vec![
            make_node(
                21,
                "The Rust compiler version 1.93 was released on February 2026",
                NodeType::Fact,
                now - 10,
            ),
            make_node(
                22,
                "LMDB supports ACID transactions with memory-mapped files",
                NodeType::Fact,
                now - 15,
            ),
            make_node(
                23,
                "HNSW achieves O(log n) approximate nearest neighbor search",
                NodeType::Fact,
                now - 20,
            ),
            make_entity_node(24, "Rust", "technology", now - 5),
            make_entity_node(25, "LMDB", "technology", now - 5),
            make_entity_node(26, "HNSW", "technology", now - 5),
            make_node(
                27,
                "The cosine similarity formula is dot(a,b) divided by norms",
                NodeType::Fact,
                now - 25,
            ),
            make_node(
                28,
                "Binary search has O(log n) time complexity verified",
                NodeType::Fact,
                now - 30,
            ),
            make_node(
                29,
                "Graph traversal BFS uses O(V+E) time and space complexity",
                NodeType::Fact,
                now - 35,
            ),
            make_node(
                30,
                "Ucotron uses HelixDB as the storage backend confirmed",
                NodeType::Fact,
                now - 8,
            ),
        ]);

        // Add confidence metadata to convergent nodes
        for node in nodes.iter_mut() {
            if node.id <= 10 {
                node.metadata
                    .insert("confidence".into(), Value::Float(0.95));
            } else if node.id <= 20 {
                node.metadata.insert("confidence".into(), Value::Float(0.3));
            } else {
                node.metadata.insert("confidence".into(), Value::Float(0.9));
            }
        }

        // ── Edges: convergent cluster is densely connected ──
        let edges = vec![
            // Convergent: dense connections for Alice's career
            Edge {
                source: 1,
                target: 2,
                edge_type: EdgeType::Actor,
                weight: 0.95,
                metadata: HashMap::new(),
            },
            Edge {
                source: 1,
                target: 3,
                edge_type: EdgeType::Location,
                weight: 0.9,
                metadata: HashMap::new(),
            },
            Edge {
                source: 4,
                target: 2,
                edge_type: EdgeType::Actor,
                weight: 0.9,
                metadata: HashMap::new(),
            },
            Edge {
                source: 4,
                target: 5,
                edge_type: EdgeType::Location,
                weight: 0.85,
                metadata: HashMap::new(),
            },
            Edge {
                source: 6,
                target: 2,
                edge_type: EdgeType::Actor,
                weight: 0.9,
                metadata: HashMap::new(),
            },
            Edge {
                source: 6,
                target: 3,
                edge_type: EdgeType::Location,
                weight: 0.85,
                metadata: HashMap::new(),
            },
            Edge {
                source: 6,
                target: 9,
                edge_type: EdgeType::Object,
                weight: 0.8,
                metadata: HashMap::new(),
            },
            Edge {
                source: 7,
                target: 2,
                edge_type: EdgeType::Actor,
                weight: 0.9,
                metadata: HashMap::new(),
            },
            Edge {
                source: 7,
                target: 3,
                edge_type: EdgeType::Location,
                weight: 0.85,
                metadata: HashMap::new(),
            },
            Edge {
                source: 8,
                target: 2,
                edge_type: EdgeType::Actor,
                weight: 0.9,
                metadata: HashMap::new(),
            },
            Edge {
                source: 8,
                target: 3,
                edge_type: EdgeType::Location,
                weight: 0.85,
                metadata: HashMap::new(),
            },
            Edge {
                source: 8,
                target: 10,
                edge_type: EdgeType::Object,
                weight: 0.8,
                metadata: HashMap::new(),
            },
            Edge {
                source: 2,
                target: 5,
                edge_type: EdgeType::RelatesTo,
                weight: 0.7,
                metadata: HashMap::new(),
            },
            Edge {
                source: 3,
                target: 9,
                edge_type: EdgeType::RelatesTo,
                weight: 0.7,
                metadata: HashMap::new(),
            },
            Edge {
                source: 9,
                target: 10,
                edge_type: EdgeType::RelatesTo,
                weight: 0.6,
                metadata: HashMap::new(),
            },
            // Divergent: sparse, some contradictions
            Edge {
                source: 11,
                target: 12,
                edge_type: EdgeType::Actor,
                weight: 0.7,
                metadata: HashMap::new(),
            },
            Edge {
                source: 11,
                target: 14,
                edge_type: EdgeType::Location,
                weight: 0.6,
                metadata: HashMap::new(),
            },
            Edge {
                source: 15,
                target: 12,
                edge_type: EdgeType::Actor,
                weight: 0.5,
                metadata: HashMap::new(),
            },
            Edge {
                source: 11,
                target: 15,
                edge_type: EdgeType::ConflictsWith,
                weight: 0.9,
                metadata: HashMap::new(),
            },
            Edge {
                source: 16,
                target: 12,
                edge_type: EdgeType::Actor,
                weight: 0.5,
                metadata: HashMap::new(),
            },
            Edge {
                source: 16,
                target: 17,
                edge_type: EdgeType::Location,
                weight: 0.5,
                metadata: HashMap::new(),
            },
            Edge {
                source: 18,
                target: 12,
                edge_type: EdgeType::Actor,
                weight: 0.4,
                metadata: HashMap::new(),
            },
            Edge {
                source: 18,
                target: 19,
                edge_type: EdgeType::Location,
                weight: 0.4,
                metadata: HashMap::new(),
            },
            Edge {
                source: 20,
                target: 12,
                edge_type: EdgeType::Actor,
                weight: 0.3,
                metadata: HashMap::new(),
            },
            // Algorithmic: moderate connections, focused on topics
            Edge {
                source: 21,
                target: 24,
                edge_type: EdgeType::Object,
                weight: 0.9,
                metadata: HashMap::new(),
            },
            Edge {
                source: 22,
                target: 25,
                edge_type: EdgeType::Object,
                weight: 0.9,
                metadata: HashMap::new(),
            },
            Edge {
                source: 23,
                target: 26,
                edge_type: EdgeType::Object,
                weight: 0.9,
                metadata: HashMap::new(),
            },
            Edge {
                source: 24,
                target: 25,
                edge_type: EdgeType::RelatesTo,
                weight: 0.6,
                metadata: HashMap::new(),
            },
            Edge {
                source: 25,
                target: 26,
                edge_type: EdgeType::RelatesTo,
                weight: 0.6,
                metadata: HashMap::new(),
            },
            Edge {
                source: 27,
                target: 26,
                edge_type: EdgeType::RelatesTo,
                weight: 0.5,
                metadata: HashMap::new(),
            },
            Edge {
                source: 28,
                target: 24,
                edge_type: EdgeType::RelatesTo,
                weight: 0.5,
                metadata: HashMap::new(),
            },
            Edge {
                source: 29,
                target: 24,
                edge_type: EdgeType::RelatesTo,
                weight: 0.5,
                metadata: HashMap::new(),
            },
            Edge {
                source: 30,
                target: 25,
                edge_type: EdgeType::RelatesTo,
                weight: 0.7,
                metadata: HashMap::new(),
            },
        ];

        graph_backend.upsert_nodes(&nodes).unwrap();
        graph_backend.upsert_edges(&edges).unwrap();

        let embeddings: Vec<(NodeId, Vec<f32>)> =
            nodes.iter().map(|n| (n.id, n.embedding.clone())).collect();
        vec_backend.upsert_embeddings(&embeddings).unwrap();

        let registry = BackendRegistry::new(Box::new(vec_backend), Box::new(graph_backend));

        // Ground truth: queries with expected relevant results per mindset mode.
        // Each entry: (query_text, mindset, expected_relevant_node_ids)
        let ground_truth = vec![
            // Convergent queries: should find well-connected career facts
            (
                "Where does Alice work and what is her role?",
                Some(MindsetTag::Convergent),
                vec![1, 7, 8, 3],
            ),
            (
                "summarize Alice career history at Google",
                Some(MindsetTag::Convergent),
                vec![1, 4, 6, 7, 8],
            ),
            (
                "What is the consensus about Alice professional background?",
                Some(MindsetTag::Convergent),
                vec![1, 4, 6, 7, 8, 2],
            ),
            // Divergent queries: should find contradictory/novel travel memories
            (
                "What are alternative experiences Bob had while traveling?",
                Some(MindsetTag::Divergent),
                vec![11, 15, 16, 18, 20],
            ),
            (
                "explore contradictions in Bob travel stories",
                Some(MindsetTag::Divergent),
                vec![11, 15, 20],
            ),
            (
                "What unusual things did Bob discover while traveling?",
                Some(MindsetTag::Divergent),
                vec![16, 18],
            ),
            // Algorithmic queries: should find precise, recent tech facts
            (
                "verify what version of Rust was released recently",
                Some(MindsetTag::Algorithmic),
                vec![21, 24],
            ),
            (
                "confirm the complexity of HNSW nearest neighbor search",
                Some(MindsetTag::Algorithmic),
                vec![23, 26],
            ),
            (
                "check if Ucotron uses HelixDB as backend",
                Some(MindsetTag::Algorithmic),
                vec![30, 25],
            ),
            // Baseline queries (no mindset — tests standard retrieval)
            (
                "Where does Alice work and what is her role?",
                None,
                vec![1, 7, 8, 3],
            ),
            (
                "What are alternative experiences Bob had while traveling?",
                None,
                vec![11, 15, 16, 18, 20],
            ),
            (
                "verify what version of Rust was released recently",
                None,
                vec![21, 24],
            ),
        ];

        (registry, embedder, ground_truth)
    }

    /// US-27.13: Benchmark mindset-aware retrieval vs baseline.
    ///
    /// Runs all queries through the RetrievalOrchestrator with and without
    /// mindset scoring, then computes and compares Recall@10, MRR, NDCG
    /// using the bench_eval framework from ucotron_core.
    #[test]
    fn test_benchmark_mindset_vs_baseline() {
        use ucotron_core::bench_eval::{ndcg_at_k, recall_at_k, reciprocal_rank};

        let (registry, embedder, ground_truth) = setup_benchmark_registry();

        // Separate queries by mode
        let mindset_queries: Vec<_> = ground_truth
            .iter()
            .filter(|(_, m, _)| m.is_some())
            .collect();
        let baseline_queries: Vec<_> = ground_truth
            .iter()
            .filter(|(_, m, _)| m.is_none())
            .collect();

        // Run mindset-aware retrieval
        let mut mindset_recall_10 = Vec::new();
        let mut mindset_mrr = Vec::new();
        let mut mindset_ndcg_10 = Vec::new();
        let mut mindset_latencies = Vec::new();

        for (query, mindset, relevant_ids) in &mindset_queries {
            let config = RetrievalConfig {
                enable_entity_extraction: false,
                enable_community_expansion: false,
                query_mindset: *mindset,
                final_top_k: 10,
                vector_top_k: 20,
                ..Default::default()
            };
            let orchestrator = RetrievalOrchestrator::new(&registry, &embedder, None, config);
            let result = orchestrator.retrieve(query).unwrap();

            let retrieved: Vec<String> = result
                .memories
                .iter()
                .map(|m| m.node.id.to_string())
                .collect();
            let relevant: Vec<String> = relevant_ids.iter().map(|id| id.to_string()).collect();

            mindset_recall_10.push(recall_at_k(&retrieved, &relevant, 10));
            mindset_mrr.push(reciprocal_rank(&retrieved, &relevant));
            mindset_ndcg_10.push(ndcg_at_k(&retrieved, &relevant, &HashMap::new(), 10));
            mindset_latencies.push(result.metrics.total_us);
        }

        // Run baseline retrieval (no mindset)
        let mut baseline_recall_10 = Vec::new();
        let mut baseline_mrr = Vec::new();
        let mut baseline_ndcg_10 = Vec::new();
        let mut baseline_latencies = Vec::new();

        for (query, _, relevant_ids) in &baseline_queries {
            let config = RetrievalConfig {
                enable_entity_extraction: false,
                enable_community_expansion: false,
                query_mindset: None,
                final_top_k: 10,
                vector_top_k: 20,
                ..Default::default()
            };
            let orchestrator = RetrievalOrchestrator::new(&registry, &embedder, None, config);
            let result = orchestrator.retrieve(query).unwrap();

            let retrieved: Vec<String> = result
                .memories
                .iter()
                .map(|m| m.node.id.to_string())
                .collect();
            let relevant: Vec<String> = relevant_ids.iter().map(|id| id.to_string()).collect();

            baseline_recall_10.push(recall_at_k(&retrieved, &relevant, 10));
            baseline_mrr.push(reciprocal_rank(&retrieved, &relevant));
            baseline_ndcg_10.push(ndcg_at_k(&retrieved, &relevant, &HashMap::new(), 10));
            baseline_latencies.push(result.metrics.total_us);
        }

        // Also run the SAME mindset queries WITHOUT mindset for fair comparison
        let mut same_query_baseline_recall_10 = Vec::new();
        let mut same_query_baseline_mrr = Vec::new();
        let mut same_query_baseline_ndcg_10 = Vec::new();

        for (query, _, relevant_ids) in &mindset_queries {
            let config = RetrievalConfig {
                enable_entity_extraction: false,
                enable_community_expansion: false,
                query_mindset: None,
                final_top_k: 10,
                vector_top_k: 20,
                ..Default::default()
            };
            let orchestrator = RetrievalOrchestrator::new(&registry, &embedder, None, config);
            let result = orchestrator.retrieve(query).unwrap();

            let retrieved: Vec<String> = result
                .memories
                .iter()
                .map(|m| m.node.id.to_string())
                .collect();
            let relevant: Vec<String> = relevant_ids.iter().map(|id| id.to_string()).collect();

            same_query_baseline_recall_10.push(recall_at_k(&retrieved, &relevant, 10));
            same_query_baseline_mrr.push(reciprocal_rank(&retrieved, &relevant));
            same_query_baseline_ndcg_10.push(ndcg_at_k(&retrieved, &relevant, &HashMap::new(), 10));
        }

        // Compute averages
        let avg = |v: &[f64]| -> f64 {
            if v.is_empty() {
                0.0
            } else {
                v.iter().sum::<f64>() / v.len() as f64
            }
        };

        let mindset_avg_recall = avg(&mindset_recall_10);
        let mindset_avg_mrr = avg(&mindset_mrr);
        let mindset_avg_ndcg = avg(&mindset_ndcg_10);
        let mindset_avg_latency = if mindset_latencies.is_empty() {
            0
        } else {
            mindset_latencies.iter().sum::<u64>() / mindset_latencies.len() as u64
        };

        let baseline_same_avg_recall = avg(&same_query_baseline_recall_10);
        let baseline_same_avg_mrr = avg(&same_query_baseline_mrr);
        let baseline_same_avg_ndcg = avg(&same_query_baseline_ndcg_10);

        let _baseline_avg_recall = avg(&baseline_recall_10);
        let _baseline_avg_mrr = avg(&baseline_mrr);
        let _baseline_avg_ndcg = avg(&baseline_ndcg_10);
        let baseline_avg_latency = if baseline_latencies.is_empty() {
            0
        } else {
            baseline_latencies.iter().sum::<u64>() / baseline_latencies.len() as u64
        };

        // Print comparison table
        println!("\n=== Mindset-Aware Retrieval Benchmark (US-27.13) ===\n");
        println!("Configuration: 30 nodes, 3 clusters, mock backends, final_top_k=10\n");

        println!(
            "### Same-Query Comparison (mindset queries run with and without mindset scoring)"
        );
        println!("| Metric         | Baseline (no mindset) | Mindset-Aware | Delta  |");
        println!("|----------------|----------------------|---------------|--------|");
        println!(
            "| Recall@10      | {:.4}                | {:.4}         | {:+.4} |",
            baseline_same_avg_recall,
            mindset_avg_recall,
            mindset_avg_recall - baseline_same_avg_recall
        );
        println!(
            "| MRR            | {:.4}                | {:.4}         | {:+.4} |",
            baseline_same_avg_mrr,
            mindset_avg_mrr,
            mindset_avg_mrr - baseline_same_avg_mrr
        );
        println!(
            "| NDCG@10        | {:.4}                | {:.4}         | {:+.4} |",
            baseline_same_avg_ndcg,
            mindset_avg_ndcg,
            mindset_avg_ndcg - baseline_same_avg_ndcg
        );
        println!(
            "| Avg Latency    | {:.2}ms              | {:.2}ms       |        |",
            baseline_avg_latency as f64 / 1000.0,
            mindset_avg_latency as f64 / 1000.0
        );

        println!("\n### Per-Mindset Breakdown (mindset-aware only)");
        println!("| Mindset      | Queries | Recall@10 | MRR    | NDCG@10 |");
        println!("|--------------|---------|-----------|--------|---------|");
        for tag in &[
            MindsetTag::Convergent,
            MindsetTag::Divergent,
            MindsetTag::Algorithmic,
        ] {
            let indices: Vec<usize> = mindset_queries
                .iter()
                .enumerate()
                .filter(|(_, (_, m, _))| m.as_ref() == Some(tag))
                .map(|(i, _)| i)
                .collect();
            let tag_recall: Vec<f64> = indices.iter().map(|&i| mindset_recall_10[i]).collect();
            let tag_mrr: Vec<f64> = indices.iter().map(|&i| mindset_mrr[i]).collect();
            let tag_ndcg: Vec<f64> = indices.iter().map(|&i| mindset_ndcg_10[i]).collect();
            println!(
                "| {:12} | {:7} | {:.4}     | {:.4} | {:.4}   |",
                format!("{:?}", tag),
                indices.len(),
                avg(&tag_recall),
                avg(&tag_mrr),
                avg(&tag_ndcg),
            );
        }

        // Print per-query detail
        println!("\n### Per-Query Detail (mindset-aware)");
        for (i, (query, mindset, _relevant)) in mindset_queries.iter().enumerate() {
            println!(
                "  {:?} | R@10={:.3} MRR={:.3} NDCG={:.3} | \"{}\"",
                mindset.unwrap(),
                mindset_recall_10[i],
                mindset_mrr[i],
                mindset_ndcg_10[i],
                &query[..query.len().min(60)],
            );
        }

        println!("\n### Per-Query Detail (same queries, no mindset)");
        for (i, (query, _, _relevant)) in mindset_queries.iter().enumerate() {
            println!(
                "  None         | R@10={:.3} MRR={:.3} NDCG={:.3} | \"{}\"",
                same_query_baseline_recall_10[i],
                same_query_baseline_mrr[i],
                same_query_baseline_ndcg_10[i],
                &query[..query.len().min(60)],
            );
        }

        // Basic assertions: mindset-aware retrieval should produce results
        assert!(
            !mindset_recall_10.is_empty(),
            "Should have mindset query results"
        );
        assert!(
            mindset_avg_recall >= 0.0,
            "Recall@10 should be non-negative"
        );
        assert!(mindset_avg_mrr >= 0.0, "MRR should be non-negative");
        assert!(mindset_avg_ndcg >= 0.0, "NDCG should be non-negative");

        // Verify mindset scoring is actually applied
        for (query, mindset, _) in &mindset_queries {
            let config = RetrievalConfig {
                enable_entity_extraction: false,
                enable_community_expansion: false,
                query_mindset: *mindset,
                final_top_k: 10,
                vector_top_k: 20,
                ..Default::default()
            };
            let orchestrator = RetrievalOrchestrator::new(&registry, &embedder, None, config);
            let result = orchestrator.retrieve(query).unwrap();
            let has_mindset_score = result.memories.iter().any(|m| m.mindset_score > 0.0);
            assert!(
                has_mindset_score,
                "Query '{}' with {:?} mindset should have non-zero mindset scores",
                query, mindset
            );
        }
    }

    // ---- Namespace isolation tests (H9 / BUG-1) ----

    /// Create a node with a specific namespace in metadata.
    fn make_node_ns(
        id: NodeId,
        content: &str,
        node_type: NodeType,
        timestamp: u64,
        namespace: &str,
    ) -> Node {
        let mut node = make_node(id, content, node_type, timestamp);
        node.metadata
            .insert("_namespace".into(), Value::String(namespace.into()));
        node
    }

    #[test]
    fn test_node_in_namespace_helper() {
        let now = now_secs();
        let node_a = make_node_ns(1, "content", NodeType::Event, now, "tenant-a");
        let node_default = make_node(2, "content", NodeType::Event, now);

        assert!(node_in_namespace(&node_a, "tenant-a"));
        assert!(!node_in_namespace(&node_a, "tenant-b"));
        assert!(!node_in_namespace(&node_a, "default"));

        // Nodes without _namespace match "default"
        assert!(node_in_namespace(&node_default, "default"));
        assert!(!node_in_namespace(&node_default, "tenant-a"));
    }

    /// Verify that graph expansion does NOT leak data across namespaces.
    ///
    /// Setup: Two namespaces (tenant-a, tenant-b) with cross-namespace edges.
    /// The graph BFS would normally follow these edges and include nodes from
    /// the other namespace. With the namespace filter, only same-namespace
    /// nodes should appear in the results.
    #[test]
    fn test_namespace_isolation_graph_expansion() {
        let vec_backend = MockVec::new();
        let graph_backend = MockGraph::new();
        let now = now_secs();

        // Tenant A nodes
        let node_a1 = make_node_ns(
            1,
            "Project Alpha is a secret internal initiative",
            NodeType::Event,
            now,
            "tenant-a",
        );
        let node_a2 = make_node_ns(
            2,
            "Alpha team meets every Tuesday",
            NodeType::Event,
            now,
            "tenant-a",
        );

        // Tenant B nodes (should NEVER appear in tenant-a queries)
        let node_b1 = make_node_ns(
            3,
            "Confidential Beta project revenue is 10M",
            NodeType::Event,
            now,
            "tenant-b",
        );
        let node_b2 = make_node_ns(
            4,
            "Beta team salaries are above market rate",
            NodeType::Event,
            now,
            "tenant-b",
        );

        graph_backend
            .upsert_nodes(&[
                node_a1.clone(),
                node_a2.clone(),
                node_b1.clone(),
                node_b2.clone(),
            ])
            .unwrap();
        vec_backend
            .upsert_embeddings(&[
                (1, node_a1.embedding.clone()),
                (2, node_a2.embedding.clone()),
                (3, node_b1.embedding.clone()),
                (4, node_b2.embedding.clone()),
            ])
            .unwrap();

        // Cross-namespace edges (simulates shared entity connections)
        let edges = vec![
            Edge {
                source: 1,
                target: 2,
                edge_type: EdgeType::RelatesTo,
                weight: 0.9,
                metadata: HashMap::new(),
            },
            // Cross-namespace edge: A1 → B1 (this is the leak vector)
            Edge {
                source: 1,
                target: 3,
                edge_type: EdgeType::RelatesTo,
                weight: 0.8,
                metadata: HashMap::new(),
            },
            Edge {
                source: 3,
                target: 4,
                edge_type: EdgeType::RelatesTo,
                weight: 0.9,
                metadata: HashMap::new(),
            },
        ];
        graph_backend.upsert_edges(&edges).unwrap();

        let registry = BackendRegistry::new(Box::new(vec_backend), Box::new(graph_backend));
        let embedder = MockEmbedder;

        // Query as tenant-a with namespace filter
        let config = RetrievalConfig {
            enable_entity_extraction: false,
            enable_community_expansion: false,
            graph_expansion_hops: 2, // 2 hops would reach B2 via A1→B1→B2
            namespace: Some("tenant-a".to_string()),
            final_top_k: 20,
            ..Default::default()
        };
        let orchestrator = RetrievalOrchestrator::new(&registry, &embedder, None, config);
        let result = orchestrator.retrieve("Project Alpha initiative").unwrap();

        // CRITICAL: No tenant-b nodes should appear in results
        for mem in &result.memories {
            let ns = mem.node.metadata.get("_namespace");
            match ns {
                Some(Value::String(s)) => {
                    assert_eq!(
                        s, "tenant-a",
                        "NAMESPACE LEAK: Found tenant-b node {} in tenant-a results: '{}'",
                        mem.node.id, mem.node.content
                    );
                }
                None => {
                    // Nodes without namespace are OK for "default" but not for tenant-a
                    panic!(
                        "NAMESPACE LEAK: Found node without namespace {} in tenant-a results: '{}'",
                        mem.node.id, mem.node.content
                    );
                }
                _ => panic!("Unexpected namespace value type"),
            }
        }

        // Verify tenant-a nodes are returned
        assert!(
            !result.memories.is_empty(),
            "Should still return tenant-a nodes"
        );
    }

    /// Verify that community expansion does NOT leak data across namespaces.
    #[test]
    fn test_namespace_isolation_community_expansion() {
        let vec_backend = MockVec::new();
        let graph_backend = MockGraph::new();
        let now = now_secs();

        let node_a = make_node_ns(
            1,
            "Tenant A private data about revenue",
            NodeType::Event,
            now,
            "tenant-a",
        );
        let node_b = make_node_ns(
            2,
            "Tenant B confidential salaries",
            NodeType::Event,
            now,
            "tenant-b",
        );

        graph_backend
            .upsert_nodes(&[node_a.clone(), node_b.clone()])
            .unwrap();
        vec_backend
            .upsert_embeddings(&[(1, node_a.embedding.clone()), (2, node_b.embedding.clone())])
            .unwrap();

        // Both nodes in same community (simulates Leiden clustering without namespace awareness)
        {
            let mut communities = graph_backend.communities.lock().unwrap();
            communities.insert(1, vec![1, 2]);
            communities.insert(2, vec![1, 2]);
        }

        let registry = BackendRegistry::new(Box::new(vec_backend), Box::new(graph_backend));
        let embedder = MockEmbedder;

        let config = RetrievalConfig {
            enable_entity_extraction: false,
            enable_community_expansion: true,
            max_community_members: 10,
            namespace: Some("tenant-a".to_string()),
            final_top_k: 20,
            ..Default::default()
        };
        let orchestrator = RetrievalOrchestrator::new(&registry, &embedder, None, config);
        let result = orchestrator.retrieve("revenue data").unwrap();

        // No tenant-b nodes should appear
        for mem in &result.memories {
            if let Some(Value::String(ns)) = mem.node.metadata.get("_namespace") {
                assert_eq!(
                    ns, "tenant-a",
                    "NAMESPACE LEAK via community: Found tenant-b node {} in tenant-a results",
                    mem.node.id
                );
            }
        }
    }
}
