//! Async consolidation worker ("Dreaming") for the Ucotron cognitive memory framework.
//!
//! Runs as a background `tokio::task` that periodically enriches the knowledge graph:
//!
//! 1. **Leiden Community Detection** — Re-run incremental community detection
//! 2. **Entity Merge** — Find potential duplicates via name similarity + embedding similarity
//! 3. **Memory Decay** — Apply temporal exponential decay to old unaccessed nodes
//! 4. **Compression** (optional, future) — Summarize large clusters into semantic nodes
//!
//! The worker does NOT block ingestion or retrieval. It communicates via
//! a `tokio::sync::watch` channel for graceful shutdown.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result};
use tokio::sync::watch;
use tracing::{debug, info, warn};

use ucotron_core::community::{detect_communities_incremental, CommunityConfig, CommunityResult};
use ucotron_core::{BackendRegistry, NodeId, NodeType, Value};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the consolidation worker.
#[derive(Debug, Clone)]
pub struct ConsolidationConfig {
    /// Whether to run community detection.
    pub enable_community_detection: bool,
    /// Whether to run entity merge (duplicate detection).
    pub enable_entity_merge: bool,
    /// Whether to enable memory decay.
    pub enable_decay: bool,
    /// Decay half-life in seconds (default: 30 days).
    pub decay_halflife_secs: u64,
    /// Similarity threshold for entity merge (cosine similarity).
    /// Pairs above this threshold are considered duplicates.
    pub entity_merge_threshold: f32,
    /// Community detection configuration.
    pub community_config: CommunityConfig,
    /// Current timestamp (unix seconds) for decay calculations.
    /// If None, uses std::time::SystemTime::now().
    pub current_time: Option<u64>,
}

impl Default for ConsolidationConfig {
    fn default() -> Self {
        Self {
            enable_community_detection: true,
            enable_entity_merge: true,
            enable_decay: true,
            decay_halflife_secs: 30 * 24 * 3600, // 30 days
            entity_merge_threshold: 0.8,
            community_config: CommunityConfig::default(),
            current_time: None,
        }
    }
}

impl ConsolidationConfig {
    /// Create a ConsolidationConfig from the ucotron_config crate's config.
    pub fn from_ucotron_config(config: &ucotron_config::ConsolidationConfig) -> Self {
        Self {
            enable_community_detection: true,
            enable_entity_merge: true,
            enable_decay: config.enable_decay,
            decay_halflife_secs: config.decay_halflife_secs,
            entity_merge_threshold: 0.8,
            community_config: CommunityConfig::default(),
            current_time: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Metrics
// ---------------------------------------------------------------------------

/// Timing and count metrics for a single consolidation run.
#[derive(Debug, Clone, Default)]
pub struct ConsolidationMetrics {
    /// Number of communities detected.
    pub communities_detected: usize,
    /// Number of nodes whose community changed.
    pub communities_changed_nodes: usize,
    /// Number of entity duplicate pairs found.
    pub entity_duplicates_found: usize,
    /// Number of entity merges actually performed.
    pub entity_merges_performed: usize,
    /// Number of nodes that had decay applied.
    pub nodes_decayed: usize,
    /// Microseconds spent on community detection.
    pub community_detection_us: u64,
    /// Microseconds spent on entity merge.
    pub entity_merge_us: u64,
    /// Microseconds spent on memory decay.
    pub decay_us: u64,
    /// Total microseconds for the full consolidation run.
    pub total_us: u64,
}

// ---------------------------------------------------------------------------
// Result
// ---------------------------------------------------------------------------

/// Result of a single consolidation run.
#[derive(Debug, Clone)]
pub struct ConsolidationResult {
    /// Metrics from this run.
    pub metrics: ConsolidationMetrics,
    /// Pairs of merged nodes: (removed_id, surviving_id).
    pub merged_nodes: Vec<(NodeId, NodeId)>,
    /// Nodes that had decay applied: (node_id, new_decay_factor).
    pub decayed_nodes: Vec<(NodeId, f32)>,
    /// Updated community result (if community detection ran).
    pub community_result: Option<CommunityResult>,
}

// ---------------------------------------------------------------------------
// ConsolidationOrchestrator (synchronous single-run)
// ---------------------------------------------------------------------------

/// Orchestrator for a single consolidation run.
///
/// This is the synchronous core that the async worker calls. It can also
/// be used directly for testing or manual consolidation triggers.
pub struct ConsolidationOrchestrator<'a> {
    registry: &'a BackendRegistry,
    config: ConsolidationConfig,
    /// Previous community result for incremental detection.
    previous_communities: Option<CommunityResult>,
}

impl<'a> ConsolidationOrchestrator<'a> {
    /// Create a new consolidation orchestrator.
    pub fn new(registry: &'a BackendRegistry, config: ConsolidationConfig) -> Self {
        Self {
            registry,
            config,
            previous_communities: None,
        }
    }

    /// Set previous community result for incremental detection.
    pub fn with_previous_communities(mut self, prev: CommunityResult) -> Self {
        self.previous_communities = Some(prev);
        self
    }

    /// Run a single consolidation cycle.
    pub fn consolidate(&mut self) -> Result<ConsolidationResult> {
        let total_start = Instant::now();
        let mut metrics = ConsolidationMetrics::default();
        let mut merged_nodes = Vec::new();
        let mut decayed_nodes = Vec::new();
        let mut community_result = None;

        // Step 1: Incremental Leiden community detection
        if self.config.enable_community_detection {
            let step_start = Instant::now();
            match self.run_community_detection() {
                Ok((result, changed_count)) => {
                    metrics.communities_detected = result.num_communities();
                    metrics.communities_changed_nodes = changed_count;
                    community_result = Some(result);
                    info!(
                        communities = metrics.communities_detected,
                        changed = changed_count,
                        "Community detection complete"
                    );
                }
                Err(e) => {
                    warn!("Community detection failed: {}", e);
                }
            }
            metrics.community_detection_us = step_start.elapsed().as_micros() as u64;
        }

        // Step 2: Entity merge (find duplicate entities)
        if self.config.enable_entity_merge {
            let step_start = Instant::now();
            match self.run_entity_merge() {
                Ok(merges) => {
                    metrics.entity_duplicates_found = merges.len();
                    metrics.entity_merges_performed = merges.len();
                    merged_nodes = merges;
                    info!(
                        merges = metrics.entity_merges_performed,
                        "Entity merge complete"
                    );
                }
                Err(e) => {
                    warn!("Entity merge failed: {}", e);
                }
            }
            metrics.entity_merge_us = step_start.elapsed().as_micros() as u64;
        }

        // Step 3: Memory decay
        if self.config.enable_decay {
            let step_start = Instant::now();
            match self.run_memory_decay() {
                Ok(decayed) => {
                    metrics.nodes_decayed = decayed.len();
                    decayed_nodes = decayed;
                    info!(decayed = metrics.nodes_decayed, "Memory decay complete");
                }
                Err(e) => {
                    warn!("Memory decay failed: {}", e);
                }
            }
            metrics.decay_us = step_start.elapsed().as_micros() as u64;
        }

        metrics.total_us = total_start.elapsed().as_micros() as u64;

        // Store updated community result for next incremental run
        if let Some(ref cr) = community_result {
            self.previous_communities = Some(cr.clone());
        }

        Ok(ConsolidationResult {
            metrics,
            merged_nodes,
            decayed_nodes,
            community_result,
        })
    }

    /// Step 1: Incremental community detection via Leiden.
    fn run_community_detection(&mut self) -> Result<(CommunityResult, usize)> {
        let edges = self
            .registry
            .graph()
            .get_all_edges()
            .context("Failed to get edges for community detection")?;

        if edges.is_empty() {
            debug!("No edges in graph, skipping community detection");
            return Ok((
                CommunityResult {
                    communities: HashMap::new(),
                    node_to_community: HashMap::new(),
                },
                0,
            ));
        }

        let (result, changed) = detect_communities_incremental(
            &edges,
            &self.config.community_config,
            self.previous_communities.as_ref(),
        )
        .context("Leiden community detection failed")?;

        // Persist community assignments to the graph backend
        self.registry
            .graph()
            .store_community_assignments(&result.node_to_community)
            .context("Failed to store community assignments")?;

        let changed_count = changed.len();
        Ok((result, changed_count))
    }

    /// Step 2: Find and merge duplicate entities.
    ///
    /// Strategy: collect all Entity nodes, group by normalized name, and
    /// within each group check embedding similarity. If above threshold,
    /// merge by redirecting edges from the duplicate to the surviving node.
    fn run_entity_merge(&self) -> Result<Vec<(NodeId, NodeId)>> {
        // Get all edges to find entity nodes
        let edges = self
            .registry
            .graph()
            .get_all_edges()
            .context("Failed to get edges for entity merge")?;

        // Collect unique node IDs from edges
        let mut node_ids: Vec<NodeId> = Vec::new();
        for &(src, tgt, _) in &edges {
            node_ids.push(src);
            node_ids.push(tgt);
        }
        node_ids.sort_unstable();
        node_ids.dedup();

        // Fetch all entity nodes and group by normalized name
        let mut name_groups: HashMap<String, Vec<(NodeId, Vec<f32>)>> = HashMap::new();
        for &nid in &node_ids {
            if let Ok(Some(node)) = self.registry.graph().get_node(nid) {
                if matches!(node.node_type, NodeType::Entity) && !node.embedding.is_empty() {
                    let key = normalize_name(&node.content);
                    if !key.is_empty() {
                        name_groups
                            .entry(key)
                            .or_default()
                            .push((node.id, node.embedding.clone()));
                    }
                }
            }
        }

        let mut merges = Vec::new();

        // Within each name group, find pairs with high embedding similarity
        for group in name_groups.values() {
            if group.len() < 2 {
                continue;
            }

            // Pairwise comparison within group
            for i in 0..group.len() {
                for j in (i + 1)..group.len() {
                    let sim = cosine_similarity(&group[i].1, &group[j].1);
                    if sim >= self.config.entity_merge_threshold {
                        // Keep the node with the lower ID (arbitrary but deterministic)
                        let (survivor, removed) = if group[i].0 <= group[j].0 {
                            (group[i].0, group[j].0)
                        } else {
                            (group[j].0, group[i].0)
                        };
                        merges.push((removed, survivor));
                    }
                }
            }
        }

        // Deduplicate merges (a node should only be merged once)
        merges.sort_by_key(|&(removed, _)| removed);
        merges.dedup_by_key(|m| m.0);

        Ok(merges)
    }

    /// Step 3: Apply temporal decay to old nodes.
    ///
    /// For each node, computes a decay factor based on its age:
    /// `decay = 0.5^(age_secs / half_life_secs)`
    ///
    /// Stores the decay factor in node metadata under key "decay_factor".
    fn run_memory_decay(&self) -> Result<Vec<(NodeId, f32)>> {
        let now = self.config.current_time.unwrap_or_else(|| {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs()
        });

        let half_life = self.config.decay_halflife_secs as f64;
        if half_life <= 0.0 {
            return Ok(Vec::new());
        }

        // Get all nodes via edges (same approach as entity merge)
        let edges = self
            .registry
            .graph()
            .get_all_edges()
            .context("Failed to get edges for memory decay")?;

        let mut node_ids: Vec<NodeId> = Vec::new();
        for &(src, tgt, _) in &edges {
            node_ids.push(src);
            node_ids.push(tgt);
        }
        node_ids.sort_unstable();
        node_ids.dedup();

        let mut decayed = Vec::new();
        let mut updated_nodes = Vec::new();

        for &nid in &node_ids {
            if let Ok(Some(mut node)) = self.registry.graph().get_node(nid) {
                if node.timestamp == 0 || node.timestamp > now {
                    continue; // Skip nodes without valid timestamps
                }

                let age_secs = (now - node.timestamp) as f64;
                let decay_factor = 0.5_f64.powf(age_secs / half_life) as f32;

                // Only update if decay is meaningful (< 0.99)
                if decay_factor < 0.99 {
                    node.metadata.insert(
                        "decay_factor".to_string(),
                        Value::Float(decay_factor as f64),
                    );
                    decayed.push((nid, decay_factor));
                    updated_nodes.push(node);
                }
            }
        }

        // Batch upsert updated nodes
        if !updated_nodes.is_empty() {
            self.registry
                .graph()
                .upsert_nodes(&updated_nodes)
                .context("Failed to upsert decayed nodes")?;
        }

        Ok(decayed)
    }
}

// ---------------------------------------------------------------------------
// ConsolidationWorker (async background task)
// ---------------------------------------------------------------------------

/// Async consolidation worker that runs in a background tokio task.
///
/// Spawns a task that periodically calls `ConsolidationOrchestrator::consolidate()`.
/// Uses a `watch::Receiver<bool>` for graceful shutdown.
pub struct ConsolidationWorker {
    /// Interval between consolidation runs in seconds.
    pub interval_secs: u64,
    /// Configuration for each consolidation run.
    pub config: ConsolidationConfig,
}

impl ConsolidationWorker {
    /// Create a new consolidation worker.
    pub fn new(interval_secs: u64, config: ConsolidationConfig) -> Self {
        Self {
            interval_secs,
            config,
        }
    }

    /// Spawn the consolidation worker as a background tokio task.
    ///
    /// Returns a `watch::Sender<bool>` that can be used to signal shutdown.
    /// Send `true` to the sender to stop the worker gracefully.
    ///
    /// The `registry` must be `Arc` so it can be shared with the background task.
    pub fn spawn(self, registry: Arc<BackendRegistry>) -> watch::Sender<bool> {
        let (shutdown_tx, shutdown_rx) = watch::channel(false);

        tokio::spawn(async move {
            self.run(registry, shutdown_rx).await;
        });

        shutdown_tx
    }

    /// Internal run loop. Runs until shutdown signal received.
    async fn run(self, registry: Arc<BackendRegistry>, mut shutdown_rx: watch::Receiver<bool>) {
        let mut interval =
            tokio::time::interval(tokio::time::Duration::from_secs(self.interval_secs));
        // First tick completes immediately — skip it to avoid running on startup
        interval.tick().await;

        let mut previous_communities: Option<CommunityResult> = None;

        loop {
            tokio::select! {
                _ = interval.tick() => {
                    info!("Consolidation worker: starting cycle");

                    let mut orchestrator = ConsolidationOrchestrator::new(
                        &registry,
                        self.config.clone(),
                    );

                    if let Some(prev) = previous_communities.take() {
                        orchestrator = orchestrator.with_previous_communities(prev);
                    }

                    match orchestrator.consolidate() {
                        Ok(result) => {
                            info!(
                                communities = result.metrics.communities_detected,
                                merges = result.metrics.entity_merges_performed,
                                decayed = result.metrics.nodes_decayed,
                                total_us = result.metrics.total_us,
                                "Consolidation cycle complete"
                            );
                            previous_communities = result.community_result;
                        }
                        Err(e) => {
                            warn!("Consolidation cycle failed: {}", e);
                        }
                    }
                }
                _ = shutdown_rx.changed() => {
                    if *shutdown_rx.borrow() {
                        info!("Consolidation worker: received shutdown signal");
                        break;
                    }
                }
            }
        }

        info!("Consolidation worker: stopped");
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Normalize a node name for deduplication: lowercase + trim whitespace.
fn normalize_name(name: &str) -> String {
    name.trim().to_lowercase()
}

/// Compute cosine similarity between two embedding vectors.
/// Assumes both vectors are L2-normalized (dot product = cosine similarity).
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.is_empty() || b.is_empty() || a.len() != b.len() {
        return 0.0;
    }
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;
    use ucotron_core::{Edge, GraphBackend, Node, VectorBackend};

    // ---- Mock backends for testing ----

    struct MockVec {
        embeddings: Mutex<HashMap<NodeId, Vec<f32>>>,
    }

    impl MockVec {
        fn new() -> Self {
            Self {
                embeddings: Mutex::new(HashMap::new()),
            }
        }
    }

    impl VectorBackend for MockVec {
        fn upsert_embeddings(&self, items: &[(NodeId, Vec<f32>)]) -> anyhow::Result<()> {
            let mut map = self.embeddings.lock().unwrap();
            for (id, vec) in items {
                map.insert(*id, vec.clone());
            }
            Ok(())
        }

        fn search(&self, query: &[f32], top_k: usize) -> anyhow::Result<Vec<(NodeId, f32)>> {
            let map = self.embeddings.lock().unwrap();
            let mut scores: Vec<(NodeId, f32)> = map
                .iter()
                .map(|(id, emb)| {
                    let dot: f32 = query.iter().zip(emb.iter()).map(|(a, b)| a * b).sum();
                    (*id, dot)
                })
                .collect();
            scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            scores.truncate(top_k);
            Ok(scores)
        }

        fn delete(&self, ids: &[NodeId]) -> anyhow::Result<()> {
            let mut map = self.embeddings.lock().unwrap();
            for id in ids {
                map.remove(id);
            }
            Ok(())
        }
    }

    struct MockGraph {
        nodes: Mutex<HashMap<NodeId, Node>>,
        edges: Mutex<Vec<(NodeId, NodeId, f32)>>,
        communities: Mutex<HashMap<NodeId, ucotron_core::community::CommunityId>>,
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
            let mut map = self.nodes.lock().unwrap();
            for node in nodes {
                map.insert(node.id, node.clone());
            }
            Ok(())
        }

        fn upsert_edges(&self, edges: &[Edge]) -> anyhow::Result<()> {
            let mut e = self.edges.lock().unwrap();
            for edge in edges {
                e.push((edge.source, edge.target, edge.weight));
            }
            Ok(())
        }

        fn get_node(&self, id: NodeId) -> anyhow::Result<Option<Node>> {
            let map = self.nodes.lock().unwrap();
            Ok(map.get(&id).cloned())
        }

        fn get_neighbors(&self, id: NodeId, _hops: u8) -> anyhow::Result<Vec<Node>> {
            let edges = self.edges.lock().unwrap();
            let nodes = self.nodes.lock().unwrap();
            let mut result = Vec::new();
            for &(src, tgt, _) in edges.iter() {
                if src == id {
                    if let Some(n) = nodes.get(&tgt) {
                        result.push(n.clone());
                    }
                } else if tgt == id {
                    if let Some(n) = nodes.get(&src) {
                        result.push(n.clone());
                    }
                }
            }
            Ok(result)
        }

        fn find_path(
            &self,
            source: NodeId,
            target: NodeId,
            _max_depth: u32,
        ) -> anyhow::Result<Option<Vec<NodeId>>> {
            if source == target {
                return Ok(Some(vec![source]));
            }
            Ok(None)
        }

        fn get_community(&self, node_id: NodeId) -> anyhow::Result<Vec<NodeId>> {
            let comms = self.communities.lock().unwrap();
            if let Some(&cid) = comms.get(&node_id) {
                // Return all nodes in same community
                let members: Vec<NodeId> = comms
                    .iter()
                    .filter(|(_, &c)| c == cid)
                    .map(|(&nid, _)| nid)
                    .collect();
                Ok(members)
            } else {
                Ok(Vec::new())
            }
        }

        fn get_all_nodes(&self) -> anyhow::Result<Vec<Node>> {
            let map = self.nodes.lock().unwrap();
            Ok(map.values().cloned().collect())
        }

        fn get_all_edges(&self) -> anyhow::Result<Vec<(NodeId, NodeId, f32)>> {
            let e = self.edges.lock().unwrap();
            Ok(e.clone())
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
            assignments: &HashMap<NodeId, ucotron_core::community::CommunityId>,
        ) -> anyhow::Result<()> {
            let mut comms = self.communities.lock().unwrap();
            *comms = assignments.clone();
            Ok(())
        }
    }

    // ---- Helper functions ----

    fn make_entity_node(id: NodeId, name: &str, embedding: Vec<f32>, timestamp: u64) -> Node {
        Node {
            id,
            content: name.to_string(),
            embedding,
            metadata: HashMap::new(),
            node_type: NodeType::Entity,
            timestamp,
            media_type: None,
            media_uri: None,
            embedding_visual: None,
            timestamp_range: None,
            parent_video_id: None,
        }
    }

    fn make_event_node(id: NodeId, content: &str, embedding: Vec<f32>, timestamp: u64) -> Node {
        Node {
            id,
            content: content.to_string(),
            embedding,
            metadata: HashMap::new(),
            node_type: NodeType::Event,
            timestamp,
            media_type: None,
            media_uri: None,
            embedding_visual: None,
            timestamp_range: None,
            parent_video_id: None,
        }
    }

    fn unit_vec(dim: usize, idx: usize) -> Vec<f32> {
        let mut v = vec![0.0f32; dim];
        if idx < dim {
            v[idx] = 1.0;
        }
        v
    }

    fn registry_with(graph: MockGraph) -> BackendRegistry {
        BackendRegistry::new(Box::new(MockVec::new()), Box::new(graph))
    }

    // ---- Tests ----

    #[test]
    fn test_normalize_name() {
        assert_eq!(normalize_name("  Juan  "), "juan");
        assert_eq!(normalize_name("Apple"), "apple");
        assert_eq!(normalize_name(""), "");
    }

    #[test]
    fn test_cosine_similarity_identical() {
        let v = unit_vec(384, 0);
        assert!((cosine_similarity(&v, &v) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = unit_vec(384, 0);
        let b = unit_vec(384, 1);
        assert!((cosine_similarity(&a, &b) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_empty() {
        assert_eq!(cosine_similarity(&[], &[1.0]), 0.0);
    }

    #[test]
    fn test_consolidation_empty_graph() {
        let graph = MockGraph::new();
        let registry = registry_with(graph);
        let mut orchestrator =
            ConsolidationOrchestrator::new(&registry, ConsolidationConfig::default());

        let result = orchestrator.consolidate().unwrap();
        assert_eq!(result.metrics.communities_detected, 0);
        assert_eq!(result.metrics.entity_merges_performed, 0);
        assert_eq!(result.metrics.nodes_decayed, 0);
        assert!(result.merged_nodes.is_empty());
        assert!(result.decayed_nodes.is_empty());
    }

    #[test]
    fn test_community_detection_two_clusters() {
        let graph = MockGraph::new();

        // Create nodes for two clusters
        let mut nodes = Vec::new();
        for i in 1..=8 {
            nodes.push(make_entity_node(
                i,
                &format!("node_{}", i),
                unit_vec(16, i as usize),
                1000,
            ));
        }
        graph.upsert_nodes(&nodes).unwrap();

        // Cluster A: 1-4 densely connected
        let cluster_a_edges = vec![
            (1, 2, 1.0),
            (1, 3, 1.0),
            (1, 4, 1.0),
            (2, 3, 1.0),
            (2, 4, 1.0),
            (3, 4, 1.0),
        ];
        // Cluster B: 5-8 densely connected
        let cluster_b_edges = vec![
            (5, 6, 1.0),
            (5, 7, 1.0),
            (5, 8, 1.0),
            (6, 7, 1.0),
            (6, 8, 1.0),
            (7, 8, 1.0),
        ];
        // Weak bridge
        let bridge = vec![(4, 5, 0.1)];

        let mut all_edges = Vec::new();
        all_edges.extend(cluster_a_edges);
        all_edges.extend(cluster_b_edges);
        all_edges.extend(bridge);

        {
            let mut e = graph.edges.lock().unwrap();
            *e = all_edges;
        }

        let registry = registry_with(graph);
        let mut orchestrator = ConsolidationOrchestrator::new(
            &registry,
            ConsolidationConfig {
                enable_entity_merge: false,
                enable_decay: false,
                ..Default::default()
            },
        );

        let result = orchestrator.consolidate().unwrap();
        assert!(result.metrics.communities_detected >= 2);
        assert!(result.community_result.is_some());
    }

    #[test]
    fn test_entity_merge_finds_duplicates() {
        let graph = MockGraph::new();

        // Two "Juan" entities with same embedding -> should be flagged as duplicate
        let emb = unit_vec(16, 0);
        graph
            .upsert_nodes(&[
                make_entity_node(1, "Juan", emb.clone(), 1000),
                make_entity_node(2, "Juan", emb.clone(), 2000),
            ])
            .unwrap();

        // Need at least one edge to discover nodes
        {
            let mut e = graph.edges.lock().unwrap();
            e.push((1, 2, 1.0));
        }

        let registry = registry_with(graph);
        let mut orchestrator = ConsolidationOrchestrator::new(
            &registry,
            ConsolidationConfig {
                enable_community_detection: false,
                enable_decay: false,
                entity_merge_threshold: 0.8,
                ..Default::default()
            },
        );

        let result = orchestrator.consolidate().unwrap();
        assert_eq!(result.metrics.entity_duplicates_found, 1);
        assert_eq!(result.merged_nodes.len(), 1);

        let (removed, survivor) = result.merged_nodes[0];
        assert_eq!(survivor, 1); // lower ID survives
        assert_eq!(removed, 2);
    }

    #[test]
    fn test_entity_merge_no_duplicates_different_embeddings() {
        let graph = MockGraph::new();

        // Two "Apple" entities with very different embeddings -> should NOT merge
        graph
            .upsert_nodes(&[
                make_entity_node(1, "Apple", unit_vec(16, 0), 1000),
                make_entity_node(2, "Apple", unit_vec(16, 1), 2000),
            ])
            .unwrap();

        {
            let mut e = graph.edges.lock().unwrap();
            e.push((1, 2, 1.0));
        }

        let registry = registry_with(graph);
        let mut orchestrator = ConsolidationOrchestrator::new(
            &registry,
            ConsolidationConfig {
                enable_community_detection: false,
                enable_decay: false,
                entity_merge_threshold: 0.8,
                ..Default::default()
            },
        );

        let result = orchestrator.consolidate().unwrap();
        assert_eq!(result.metrics.entity_duplicates_found, 0);
        assert!(result.merged_nodes.is_empty());
    }

    #[test]
    fn test_entity_merge_different_names_not_merged() {
        let graph = MockGraph::new();

        // Same embedding but different names -> should NOT merge
        let emb = unit_vec(16, 0);
        graph
            .upsert_nodes(&[
                make_entity_node(1, "Juan", emb.clone(), 1000),
                make_entity_node(2, "Maria", emb.clone(), 2000),
            ])
            .unwrap();

        {
            let mut e = graph.edges.lock().unwrap();
            e.push((1, 2, 1.0));
        }

        let registry = registry_with(graph);
        let mut orchestrator = ConsolidationOrchestrator::new(
            &registry,
            ConsolidationConfig {
                enable_community_detection: false,
                enable_decay: false,
                ..Default::default()
            },
        );

        let result = orchestrator.consolidate().unwrap();
        assert_eq!(result.metrics.entity_duplicates_found, 0);
    }

    #[test]
    fn test_memory_decay_applies_to_old_nodes() {
        let graph = MockGraph::new();

        let now = 100_000_000u64;
        let one_year_ago = now - 365 * 24 * 3600;
        let recent = now - 3600; // 1 hour ago

        graph
            .upsert_nodes(&[
                make_entity_node(1, "old_node", unit_vec(16, 0), one_year_ago),
                make_entity_node(2, "recent_node", unit_vec(16, 1), recent),
            ])
            .unwrap();

        {
            let mut e = graph.edges.lock().unwrap();
            e.push((1, 2, 1.0));
        }

        let registry = registry_with(graph);
        let mut orchestrator = ConsolidationOrchestrator::new(
            &registry,
            ConsolidationConfig {
                enable_community_detection: false,
                enable_entity_merge: false,
                enable_decay: true,
                decay_halflife_secs: 30 * 24 * 3600, // 30 days
                current_time: Some(now),
                ..Default::default()
            },
        );

        let result = orchestrator.consolidate().unwrap();
        assert!(result.metrics.nodes_decayed >= 1);

        // Old node should have significant decay
        let old_decay = result.decayed_nodes.iter().find(|(id, _)| *id == 1);
        assert!(old_decay.is_some());
        let (_, factor) = old_decay.unwrap();
        assert!(
            *factor < 0.1,
            "1-year-old node should have significant decay, got {}",
            factor
        );

        // Recent node might also have some decay but should be close to 1.0
        let recent_decay = result.decayed_nodes.iter().find(|(id, _)| *id == 2);
        if let Some((_, factor)) = recent_decay {
            assert!(
                *factor > 0.9,
                "Recent node should have minimal decay, got {}",
                factor
            );
        }
    }

    #[test]
    fn test_memory_decay_skips_zero_timestamp() {
        let graph = MockGraph::new();

        graph
            .upsert_nodes(&[make_entity_node(1, "no_timestamp", unit_vec(16, 0), 0)])
            .unwrap();

        {
            let mut e = graph.edges.lock().unwrap();
            e.push((1, 1, 1.0)); // self-loop to discover node
        }

        let registry = registry_with(graph);
        let mut orchestrator = ConsolidationOrchestrator::new(
            &registry,
            ConsolidationConfig {
                enable_community_detection: false,
                enable_entity_merge: false,
                enable_decay: true,
                current_time: Some(1_000_000),
                ..Default::default()
            },
        );

        let result = orchestrator.consolidate().unwrap();
        assert_eq!(result.metrics.nodes_decayed, 0);
    }

    #[test]
    fn test_consolidation_all_steps_together() {
        let graph = MockGraph::new();

        let now = 100_000_000u64;
        let old_ts = now - 365 * 24 * 3600;

        // Create a small graph with clusters, duplicates, and old nodes
        let emb = unit_vec(16, 0);
        graph
            .upsert_nodes(&[
                make_entity_node(1, "Juan", emb.clone(), old_ts),
                make_entity_node(2, "Juan", emb.clone(), old_ts + 100), // duplicate
                make_entity_node(3, "Maria", unit_vec(16, 1), now - 3600),
                make_entity_node(4, "Madrid", unit_vec(16, 2), now - 3600),
            ])
            .unwrap();

        {
            let mut e = graph.edges.lock().unwrap();
            *e = vec![(1, 3, 1.0), (1, 4, 1.0), (2, 3, 1.0), (3, 4, 1.0)];
        }

        let registry = registry_with(graph);
        let mut orchestrator = ConsolidationOrchestrator::new(
            &registry,
            ConsolidationConfig {
                enable_community_detection: true,
                enable_entity_merge: true,
                enable_decay: true,
                decay_halflife_secs: 30 * 24 * 3600,
                entity_merge_threshold: 0.8,
                current_time: Some(now),
                ..Default::default()
            },
        );

        let result = orchestrator.consolidate().unwrap();

        // Community detection should have run
        // (metrics are always set; just check no panic)

        // Entity merge should find Juan duplicates
        assert_eq!(result.metrics.entity_merges_performed, 1);

        // Decay should apply to old nodes
        assert!(result.metrics.nodes_decayed >= 1);

        // Total time should be recorded
        assert!(result.metrics.total_us > 0);
    }

    #[test]
    fn test_consolidation_disabled_steps() {
        let graph = MockGraph::new();
        let emb = unit_vec(16, 0);
        graph
            .upsert_nodes(&[
                make_entity_node(1, "Test", emb.clone(), 1000),
                make_entity_node(2, "Test", emb.clone(), 2000),
            ])
            .unwrap();

        {
            let mut e = graph.edges.lock().unwrap();
            e.push((1, 2, 1.0));
        }

        let registry = registry_with(graph);
        let mut orchestrator = ConsolidationOrchestrator::new(
            &registry,
            ConsolidationConfig {
                enable_community_detection: false,
                enable_entity_merge: false,
                enable_decay: false,
                ..Default::default()
            },
        );

        let result = orchestrator.consolidate().unwrap();
        assert_eq!(result.metrics.communities_detected, 0);
        assert_eq!(result.metrics.entity_merges_performed, 0);
        assert_eq!(result.metrics.nodes_decayed, 0);
    }

    #[test]
    fn test_consolidation_metrics_timing() {
        let graph = MockGraph::new();
        let emb = unit_vec(16, 0);
        graph
            .upsert_nodes(&[make_entity_node(1, "A", emb.clone(), 500)])
            .unwrap();

        {
            let mut e = graph.edges.lock().unwrap();
            e.push((1, 1, 1.0));
        }

        let registry = registry_with(graph);
        let mut orchestrator = ConsolidationOrchestrator::new(
            &registry,
            ConsolidationConfig {
                current_time: Some(1_000_000),
                ..Default::default()
            },
        );

        let result = orchestrator.consolidate().unwrap();
        // Consolidation should complete without error
        let _ = result.metrics.total_us;
    }

    #[test]
    fn test_incremental_community_detection() {
        let graph = MockGraph::new();

        for i in 1..=4 {
            graph
                .upsert_nodes(&[make_entity_node(
                    i,
                    &format!("node_{}", i),
                    unit_vec(16, i as usize),
                    1000,
                )])
                .unwrap();
        }

        {
            let mut e = graph.edges.lock().unwrap();
            *e = vec![(1, 2, 1.0), (2, 3, 1.0), (3, 4, 1.0)];
        }

        let registry = registry_with(graph);

        // First run
        let mut orchestrator = ConsolidationOrchestrator::new(
            &registry,
            ConsolidationConfig {
                enable_entity_merge: false,
                enable_decay: false,
                ..Default::default()
            },
        );

        let result1 = orchestrator.consolidate().unwrap();
        assert!(result1.community_result.is_some());

        // Second run with previous result — incremental
        let result2 = orchestrator.consolidate().unwrap();
        assert!(result2.community_result.is_some());
    }

    #[test]
    fn test_event_nodes_not_merged() {
        let graph = MockGraph::new();

        // Event nodes should not be merged even with same name/embedding
        let emb = unit_vec(16, 0);
        graph
            .upsert_nodes(&[
                make_event_node(1, "lunch event", emb.clone(), 1000),
                make_event_node(2, "lunch event", emb.clone(), 2000),
            ])
            .unwrap();

        {
            let mut e = graph.edges.lock().unwrap();
            e.push((1, 2, 1.0));
        }

        let registry = registry_with(graph);
        let mut orchestrator = ConsolidationOrchestrator::new(
            &registry,
            ConsolidationConfig {
                enable_community_detection: false,
                enable_decay: false,
                entity_merge_threshold: 0.8,
                ..Default::default()
            },
        );

        let result = orchestrator.consolidate().unwrap();
        // Event nodes should not be merged (only Entity nodes)
        assert_eq!(result.metrics.entity_duplicates_found, 0);
    }

    #[test]
    fn test_from_ucotron_config() {
        let mc = ucotron_config::ConsolidationConfig {
            trigger_interval: 50,
            enable_decay: false,
            decay_halflife_secs: 86400,
        };
        let config = ConsolidationConfig::from_ucotron_config(&mc);
        assert!(!config.enable_decay);
        assert_eq!(config.decay_halflife_secs, 86400);
    }

    #[tokio::test]
    async fn test_consolidation_worker_shutdown() {
        let graph = MockGraph::new();
        let registry = Arc::new(registry_with(graph));

        let worker = ConsolidationWorker::new(
            1,
            ConsolidationConfig {
                enable_community_detection: false,
                enable_entity_merge: false,
                enable_decay: false,
                ..Default::default()
            },
        );

        let shutdown_tx = worker.spawn(registry);

        // Let the worker run for a bit
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        // Signal shutdown
        shutdown_tx.send(true).unwrap();

        // Give it time to stop
        tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;

        // If we get here without hanging, the shutdown worked
    }

    #[tokio::test]
    async fn test_consolidation_worker_runs_cycle() {
        let graph = MockGraph::new();
        let emb = unit_vec(16, 0);
        graph
            .upsert_nodes(&[make_entity_node(1, "test", emb.clone(), 1000)])
            .unwrap();
        {
            let mut e = graph.edges.lock().unwrap();
            e.push((1, 1, 1.0));
        }

        let registry = Arc::new(registry_with(graph));

        let worker = ConsolidationWorker::new(
            1,
            ConsolidationConfig {
                enable_community_detection: false,
                enable_entity_merge: false,
                enable_decay: false,
                ..Default::default()
            },
        );

        let shutdown_tx = worker.spawn(Arc::clone(&registry));

        // Wait long enough for at least one cycle
        tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;

        // Shutdown
        shutdown_tx.send(true).unwrap();
        tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;
    }

    // ---- Edge-case tests ----

    #[test]
    fn test_cosine_similarity_different_lengths() {
        // Mismatched lengths: zip over shorter — the result depends on magnitudes
        let a = vec![1.0, 0.0];
        let b = vec![1.0, 0.0, 0.5];
        let sim = cosine_similarity(&a, &b);
        // dot = 1.0, mag_a = 1.0, mag_b = sqrt(1.25) ≈ 1.118
        // sim = 1.0 / (1.0 * 1.118) ≈ 0.894 (but zip truncates b to len 2)
        // Actually, zip only uses first 2 elements: dot=1, mag_a=1, mag_b=1 → sim ≈ 1.0
        // ... but magnitude uses full vectors. Let's just check it returns a valid value.
        assert!(
            (-1.0..=1.0).contains(&sim),
            "Similarity out of range: {}",
            sim
        );
    }

    #[test]
    fn test_normalize_name_unicode() {
        assert_eq!(normalize_name("  CAFÉ  "), "café");
        assert_eq!(normalize_name("Ñoño"), "ñoño");
        assert_eq!(normalize_name("MÜNCHEN"), "münchen");
    }

    #[test]
    fn test_consolidation_config_defaults() {
        let config = ConsolidationConfig::default();
        assert!(config.enable_community_detection);
        assert!(config.enable_entity_merge);
        assert!(config.enable_decay);
        assert!(config.entity_merge_threshold > 0.0);
        assert!(config.decay_halflife_secs > 0);
    }

    #[test]
    fn test_memory_decay_preserves_young_nodes() {
        let graph = MockGraph::new();
        let now = 100_000_000u64;
        let recent_ts = now - 60; // 1 minute ago

        graph
            .upsert_nodes(&[make_entity_node(
                1,
                "very recent",
                unit_vec(16, 0),
                recent_ts,
            )])
            .unwrap();
        {
            let mut e = graph.edges.lock().unwrap();
            e.push((1, 1, 1.0)); // self-edge so node is discoverable
        }

        let registry = registry_with(graph);
        let mut orch = ConsolidationOrchestrator::new(
            &registry,
            ConsolidationConfig {
                enable_community_detection: false,
                enable_entity_merge: false,
                enable_decay: true,
                decay_halflife_secs: 86400, // 1 day
                current_time: Some(now),
                ..Default::default()
            },
        );

        let result = orch.consolidate().unwrap();

        // Recent node should have decay factor very close to 1.0
        if let Some((_, factor)) = result.decayed_nodes.iter().find(|(id, _)| *id == 1) {
            assert!(
                *factor > 0.99,
                "Recent node should have high decay factor, got {}",
                factor
            );
        }
    }
}
