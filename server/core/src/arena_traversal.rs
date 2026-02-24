//! Arena-allocated graph traversal for reduced allocation overhead.
//!
//! This module provides arena-backed BFS traversal that minimizes heap
//! allocations during graph exploration. Instead of creating new `HashSet`
//! and `VecDeque` per BFS call, it uses a [`bumpalo::Bump`] arena that
//! allocates from a contiguous memory region and resets between traversals.
//!
//! # Performance
//!
//! Arena allocation reduces allocation overhead in two ways:
//! 1. **Bulk deallocation**: The arena is reset in O(1) between traversals
//!    instead of individually freeing each hash map bucket
//! 2. **Cache locality**: Arena-allocated data lives in contiguous memory,
//!    improving CPU cache utilization during BFS iteration
//!
//! # Usage
//!
//! ```ignore
//! use ucotron_core::arena_traversal::BfsArena;
//!
//! let mut arena = BfsArena::new();
//! let results = arena.find_related(&engine, &query, 10, 2, 0.5)?;
//! ```

use crate::{Node, NodeId, StorageEngine};
use anyhow::Result;
use bumpalo::Bump;
use std::collections::VecDeque;

/// Default hop decay factor (matches hybrid.rs).
pub const DEFAULT_HOP_DECAY: f32 = 0.5;

/// Reusable BFS arena that minimizes heap allocations across multiple traversals.
///
/// The arena pre-allocates memory for visited sets and score maps, resetting
/// between BFS calls instead of creating new collections each time.
pub struct BfsArena {
    /// Bump allocator for per-traversal scratch data.
    bump: Bump,
    /// Reusable BFS queue (cleared between traversals, not reallocated).
    queue: VecDeque<(NodeId, u8)>,
}

impl BfsArena {
    /// Create a new BFS arena with default capacity.
    pub fn new() -> Self {
        Self {
            bump: Bump::with_capacity(4096),
            queue: VecDeque::with_capacity(256),
        }
    }

    /// Create a new BFS arena with custom initial capacity (bytes).
    pub fn with_capacity(bytes: usize) -> Self {
        Self {
            bump: Bump::with_capacity(bytes),
            queue: VecDeque::with_capacity(256),
        }
    }

    /// Reset the arena for reuse, freeing all bump-allocated memory in O(1).
    pub fn reset(&mut self) {
        self.bump.reset();
        self.queue.clear();
    }

    /// Returns the total bytes allocated by the arena (for benchmarking).
    pub fn allocated_bytes(&self) -> usize {
        self.bump.allocated_bytes()
    }

    /// Perform a hybrid search combining vector similarity with graph traversal,
    /// using arena allocation to reduce per-traversal overhead.
    ///
    /// This is functionally equivalent to [`crate::find_related`] but uses a
    /// bump allocator for the visited set and score tracking, reducing heap
    /// allocations when performing multiple BFS expansions from vector seeds.
    pub fn find_related<E: StorageEngine>(
        &mut self,
        engine: &E,
        query: &[f32],
        top_k: usize,
        hops: u8,
        hop_decay: f32,
    ) -> Result<Vec<Node>> {
        self.bump.reset();

        // Step 1: Vector search for seed nodes
        let vector_results = engine.vector_search(query, top_k)?;

        // Arena-allocated score map: Vec of (NodeId, f32) pairs.
        // We use a bump-allocated Vec for collecting scores, then consolidate.
        // For the final score map we need to look up by NodeId — use a simple
        // Vec of (id, score) pairs and linear scan (efficient for typical
        // graph traversal result sizes of < 10k nodes).
        let scores = bumpalo::collections::Vec::new_in(&self.bump);
        let mut score_map = ArenaScoreMap::new(scores);

        for (seed_id, similarity) in &vector_results {
            score_map.update(*seed_id, *similarity);

            if hops > 0 {
                // Arena-allocated visited set for this seed's BFS
                let visited_vec = bumpalo::collections::Vec::with_capacity_in(64, &self.bump);
                let mut visited = ArenaVisited::new(visited_vec);
                visited.insert(*seed_id);

                self.queue.clear();
                self.queue.push_back((*seed_id, 0u8));

                while let Some((current_id, depth)) = self.queue.pop_front() {
                    if depth >= hops {
                        continue;
                    }

                    let next_depth = depth + 1;
                    let decay = hop_decay.powi(next_depth as i32);
                    let decayed_score = similarity * decay;

                    let neighbors = engine.get_neighbors(current_id, 1)?;
                    for neighbor in &neighbors {
                        if visited.insert(neighbor.id) {
                            score_map.update(neighbor.id, decayed_score);
                            self.queue.push_back((neighbor.id, next_depth));
                        }
                    }
                }
            }
        }

        // Collect and sort scores (move to heap for sorting)
        let mut scored: Vec<(NodeId, f32)> = score_map.drain();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Fetch full node data
        let mut results = Vec::with_capacity(scored.len());
        for (node_id, _score) in &scored {
            if let Some(node) = engine.get_node(*node_id)? {
                results.push(node);
            }
        }

        Ok(results)
    }
}

impl Default for BfsArena {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Arena-backed visited set
// ---------------------------------------------------------------------------

/// A visited set backed by an arena-allocated Vec.
///
/// Uses linear probing on a sorted-insert Vec for small sets, which is
/// cache-friendly and avoids the overhead of hashing for typical BFS
/// visited sets (usually < 1000 entries for 2-3 hop traversals).
struct ArenaVisited<'bump> {
    entries: bumpalo::collections::Vec<'bump, NodeId>,
}

impl<'bump> ArenaVisited<'bump> {
    fn new(entries: bumpalo::collections::Vec<'bump, NodeId>) -> Self {
        Self { entries }
    }

    /// Insert a node ID. Returns `true` if the node was not already present.
    fn insert(&mut self, id: NodeId) -> bool {
        // Binary search for O(log n) lookup
        match self.entries.binary_search(&id) {
            Ok(_) => false, // already present
            Err(pos) => {
                self.entries.insert(pos, id);
                true
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Arena-backed score map
// ---------------------------------------------------------------------------

/// A score map backed by an arena-allocated Vec of (NodeId, f32) pairs.
///
/// Uses sorted insertion for O(log n) lookup. For typical BFS result sizes
/// (< 10k entries), this is competitive with HashMap due to cache locality.
struct ArenaScoreMap<'bump> {
    entries: bumpalo::collections::Vec<'bump, (NodeId, f32)>,
}

impl<'bump> ArenaScoreMap<'bump> {
    fn new(entries: bumpalo::collections::Vec<'bump, (NodeId, f32)>) -> Self {
        Self { entries }
    }

    /// Update the score for a node, keeping the maximum.
    fn update(&mut self, id: NodeId, score: f32) {
        match self.entries.binary_search_by_key(&id, |(nid, _)| *nid) {
            Ok(idx) => {
                if score > self.entries[idx].1 {
                    self.entries[idx].1 = score;
                }
            }
            Err(pos) => {
                self.entries.insert(pos, (id, score));
            }
        }
    }

    /// Drain all entries into a heap-allocated Vec (for sorting/returning).
    fn drain(&self) -> Vec<(NodeId, f32)> {
        self.entries.iter().copied().collect()
    }
}

// ---------------------------------------------------------------------------
// Arena-backed traversal for Phase 2 BackendRegistry queries
// ---------------------------------------------------------------------------

use crate::BackendRegistry;

/// Arena-backed traversal for the Phase 2 query DSL.
///
/// Provides the same BFS traversal as [`crate::query::TraversalQuery`] but
/// with arena allocation for visited sets and queues.
pub struct ArenaQueryTraversal {
    bump: Bump,
    queue: VecDeque<(NodeId, u8)>,
}

impl ArenaQueryTraversal {
    /// Create a new arena query traversal.
    pub fn new() -> Self {
        Self {
            bump: Bump::with_capacity(4096),
            queue: VecDeque::with_capacity(256),
        }
    }

    /// Execute a pure graph traversal from the start node using arena allocation.
    pub fn traverse(
        &mut self,
        registry: &BackendRegistry,
        start: NodeId,
        hops: u8,
        node_filter: Option<&dyn Fn(&Node) -> bool>,
    ) -> Result<Vec<Node>> {
        self.bump.reset();
        self.queue.clear();

        let graph = registry.graph();

        let visited_vec = bumpalo::collections::Vec::with_capacity_in(64, &self.bump);
        let mut visited = ArenaVisited::new(visited_vec);
        visited.insert(start);

        self.queue.push_back((start, 0));
        let mut result_nodes = Vec::new();

        while let Some((current_id, depth)) = self.queue.pop_front() {
            if let Some(node) = graph.get_node(current_id)? {
                let passes = node_filter.is_none_or(|f| f(&node));
                if passes {
                    result_nodes.push(node);
                }
            }

            if depth >= hops {
                continue;
            }

            let neighbors = graph.get_neighbors(current_id, 1)?;
            for neighbor in neighbors {
                if visited.insert(neighbor.id) {
                    self.queue.push_back((neighbor.id, depth + 1));
                }
            }
        }

        Ok(result_nodes)
    }

    /// Execute a hybrid query (vector + graph expansion with decay) using arena allocation.
    pub fn hybrid(
        &mut self,
        registry: &BackendRegistry,
        query: &[f32],
        top_k: usize,
        hops: u8,
        hop_decay: f32,
        node_filter: Option<&dyn Fn(&Node) -> bool>,
    ) -> Result<Vec<Node>> {
        self.bump.reset();
        self.queue.clear();

        let vector = registry.vector();
        let graph = registry.graph();

        let vector_results = vector.search(query, top_k)?;

        let scores = bumpalo::collections::Vec::new_in(&self.bump);
        let mut score_map = ArenaScoreMap::new(scores);

        for (seed_id, similarity) in &vector_results {
            score_map.update(*seed_id, *similarity);

            if hops > 0 {
                let visited_vec = bumpalo::collections::Vec::with_capacity_in(64, &self.bump);
                let mut visited = ArenaVisited::new(visited_vec);
                visited.insert(*seed_id);

                self.queue.clear();
                self.queue.push_back((*seed_id, 0u8));

                while let Some((current_id, depth)) = self.queue.pop_front() {
                    if depth >= hops {
                        continue;
                    }

                    let next_depth = depth + 1;
                    let decay = hop_decay.powi(next_depth as i32);
                    let decayed_score = similarity * decay;

                    let neighbors = graph.get_neighbors(current_id, 1)?;
                    for neighbor in &neighbors {
                        if visited.insert(neighbor.id) {
                            score_map.update(neighbor.id, decayed_score);
                            self.queue.push_back((neighbor.id, next_depth));
                        }
                    }
                }
            }
        }

        let mut scored: Vec<(NodeId, f32)> = score_map.drain();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut results = Vec::with_capacity(scored.len());
        for (node_id, _score) in &scored {
            if let Some(node) = graph.get_node(*node_id)? {
                let passes = node_filter.is_none_or(|f| f(&node));
                if passes {
                    results.push(node);
                }
            }
        }

        Ok(results)
    }

    /// Returns the total bytes allocated by the arena.
    pub fn allocated_bytes(&self) -> usize {
        self.bump.allocated_bytes()
    }
}

impl Default for ArenaQueryTraversal {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Config, Edge, InsertStats, Node, NodeType};
    use std::collections::HashMap;

    /// Minimal in-memory StorageEngine for testing arena traversal.
    struct MockEngine {
        nodes: HashMap<NodeId, Node>,
        adj: HashMap<NodeId, Vec<NodeId>>,
    }

    impl MockEngine {
        fn new() -> Self {
            Self {
                nodes: HashMap::new(),
                adj: HashMap::new(),
            }
        }

        fn add_node(&mut self, node: Node) {
            self.nodes.insert(node.id, node);
        }

        fn add_edge_undirected(&mut self, a: NodeId, b: NodeId) {
            self.adj.entry(a).or_default().push(b);
            self.adj.entry(b).or_default().push(a);
        }
    }

    impl StorageEngine for MockEngine {
        fn init(_config: &Config) -> Result<Self> {
            Ok(Self::new())
        }

        fn insert_nodes(&mut self, nodes: &[Node]) -> Result<InsertStats> {
            for node in nodes {
                self.nodes.insert(node.id, node.clone());
            }
            Ok(InsertStats {
                count: nodes.len(),
                duration_us: 0,
            })
        }

        fn insert_edges(&mut self, edges: &[Edge]) -> Result<InsertStats> {
            for edge in edges {
                self.adj.entry(edge.source).or_default().push(edge.target);
                self.adj.entry(edge.target).or_default().push(edge.source);
            }
            Ok(InsertStats {
                count: edges.len(),
                duration_us: 0,
            })
        }

        fn get_node(&self, id: NodeId) -> Result<Option<Node>> {
            Ok(self.nodes.get(&id).cloned())
        }

        fn get_neighbors(&self, id: NodeId, hops: u8) -> Result<Vec<Node>> {
            if hops == 0 {
                return Ok(Vec::new());
            }
            let mut result = Vec::new();
            if let Some(neighbors) = self.adj.get(&id) {
                for &nid in neighbors {
                    if let Some(node) = self.nodes.get(&nid) {
                        result.push(node.clone());
                    }
                }
            }
            Ok(result)
        }

        fn vector_search(&self, query: &[f32], top_k: usize) -> Result<Vec<(NodeId, f32)>> {
            let mut scored: Vec<(NodeId, f32)> = self
                .nodes
                .values()
                .map(|n| {
                    let sim: f32 = n
                        .embedding
                        .iter()
                        .zip(query.iter())
                        .map(|(a, b)| a * b)
                        .sum();
                    (n.id, sim)
                })
                .collect();
            scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            scored.truncate(top_k);
            Ok(scored)
        }

        fn hybrid_search(&self, query: &[f32], top_k: usize, hops: u8) -> Result<Vec<Node>> {
            let mut arena = BfsArena::new();
            arena.find_related(self, query, top_k, hops, DEFAULT_HOP_DECAY)
        }

        fn find_path(
            &self,
            _source: NodeId,
            _target: NodeId,
            _max_depth: u32,
        ) -> Result<Option<Vec<NodeId>>> {
            Ok(None)
        }

        fn shutdown(&mut self) -> Result<()> {
            Ok(())
        }
    }

    fn make_node(id: u64, embedding: Vec<f32>) -> Node {
        Node {
            id,
            content: format!("Node {}", id),
            embedding,
            metadata: HashMap::new(),
            node_type: NodeType::Entity,
            timestamp: 1000 + id,
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

    #[test]
    fn test_arena_find_related_vector_only() {
        let mut engine = MockEngine::new();
        engine.add_node(make_node(0, unit_vec(384, 0)));
        engine.add_node(make_node(1, unit_vec(384, 1)));

        let mut arena = BfsArena::new();
        let query = unit_vec(384, 0);
        let results = arena.find_related(&engine, &query, 2, 0, 0.5).unwrap();

        assert!(results.len() <= 2);
        assert_eq!(results[0].id, 0, "Node 0 should be the top match");
    }

    #[test]
    fn test_arena_find_related_graph_expansion() {
        let mut engine = MockEngine::new();
        engine.add_node(make_node(0, unit_vec(384, 0)));
        engine.add_node(make_node(1, vec![0.0; 384]));
        engine.add_node(make_node(2, vec![0.0; 384]));

        engine.add_edge_undirected(0, 1);
        engine.add_edge_undirected(1, 2);

        let mut arena = BfsArena::new();
        let query = unit_vec(384, 0);
        let results = arena.find_related(&engine, &query, 1, 2, 0.5).unwrap();

        let ids: Vec<u64> = results.iter().map(|n| n.id).collect();
        assert!(ids.contains(&0), "Should include vector match");
        assert!(ids.contains(&1), "Should include 1-hop neighbor");
        assert!(ids.contains(&2), "Should include 2-hop neighbor");
    }

    #[test]
    fn test_arena_find_related_decay_ordering() {
        let mut engine = MockEngine::new();
        engine.add_node(make_node(0, unit_vec(384, 0)));
        engine.add_node(make_node(1, vec![0.0; 384]));
        engine.add_node(make_node(2, vec![0.0; 384]));

        engine.add_edge_undirected(0, 1);
        engine.add_edge_undirected(1, 2);

        let mut arena = BfsArena::new();
        let query = unit_vec(384, 0);
        let results = arena.find_related(&engine, &query, 1, 2, 0.5).unwrap();

        assert_eq!(results[0].id, 0, "Vector match should rank first");
        assert_eq!(results[1].id, 1, "1-hop neighbor should rank second");
        assert_eq!(results[2].id, 2, "2-hop neighbor should rank third");
    }

    #[test]
    fn test_arena_cycle_handling() {
        let mut engine = MockEngine::new();
        engine.add_node(make_node(0, unit_vec(384, 0)));
        engine.add_node(make_node(1, vec![0.0; 384]));
        engine.add_node(make_node(2, vec![0.0; 384]));
        engine.add_edge_undirected(0, 1);
        engine.add_edge_undirected(1, 2);
        engine.add_edge_undirected(2, 0);

        let mut arena = BfsArena::new();
        let query = unit_vec(384, 0);
        let results = arena.find_related(&engine, &query, 1, 10, 0.5).unwrap();

        let ids: Vec<u64> = results.iter().map(|n| n.id).collect();
        assert!(ids.contains(&0));
        assert!(ids.contains(&1));
        assert!(ids.contains(&2));
        assert_eq!(ids.len(), 3, "Should not have duplicates from cycle");
    }

    #[test]
    fn test_arena_reuse_across_calls() {
        let mut engine = MockEngine::new();
        engine.add_node(make_node(0, unit_vec(384, 0)));
        engine.add_node(make_node(1, unit_vec(384, 1)));
        engine.add_edge_undirected(0, 1);

        let mut arena = BfsArena::new();
        let query = unit_vec(384, 0);

        // Call multiple times — arena should reuse memory
        let r1 = arena.find_related(&engine, &query, 1, 1, 0.5).unwrap();
        let bytes1 = arena.allocated_bytes();

        let r2 = arena.find_related(&engine, &query, 1, 1, 0.5).unwrap();
        let _bytes2 = arena.allocated_bytes();

        // Results should be identical
        let ids1: Vec<u64> = r1.iter().map(|n| n.id).collect();
        let ids2: Vec<u64> = r2.iter().map(|n| n.id).collect();
        assert_eq!(ids1, ids2);

        // Arena was reset between calls — bytes should be modest
        assert!(bytes1 < 65536, "Arena should use modest memory: {} bytes", bytes1);
    }

    #[test]
    fn test_arena_best_score_wins() {
        let mut engine = MockEngine::new();

        let mut emb_a = vec![0.0f32; 384];
        emb_a[0] = 0.9;
        emb_a[1] = 0.1;
        let norm = (emb_a[0] * emb_a[0] + emb_a[1] * emb_a[1]).sqrt();
        emb_a[0] /= norm;
        emb_a[1] /= norm;

        engine.add_node(make_node(0, unit_vec(384, 0))); // sim=1.0
        engine.add_node(make_node(1, emb_a));              // sim~=0.994
        engine.add_node(make_node(2, vec![0.0; 384]));     // no vector match

        engine.add_edge_undirected(0, 2);
        engine.add_edge_undirected(1, 2);

        let mut arena = BfsArena::new();
        let query = unit_vec(384, 0);
        let results = arena.find_related(&engine, &query, 2, 1, 0.5).unwrap();

        let ids: Vec<u64> = results.iter().map(|n| n.id).collect();
        assert!(ids.contains(&2), "Node 2 should be discovered via graph");

        let pos_0 = ids.iter().position(|&id| id == 0).unwrap();
        let pos_2 = ids.iter().position(|&id| id == 2).unwrap();
        assert!(pos_0 < pos_2, "Vector match should rank above graph-only");
    }

    #[test]
    fn test_arena_visited_sorted_insert() {
        let bump = Bump::new();
        let entries = bumpalo::collections::Vec::new_in(&bump);
        let mut visited = ArenaVisited::new(entries);

        assert!(visited.insert(5));
        assert!(visited.insert(3));
        assert!(visited.insert(7));
        assert!(visited.insert(1));

        // Duplicates should return false
        assert!(!visited.insert(3));
        assert!(!visited.insert(7));

        // New values should return true
        assert!(visited.insert(4));

        // Internal order should be sorted
        let sorted: Vec<NodeId> = visited.entries.iter().copied().collect();
        assert_eq!(sorted, vec![1, 3, 4, 5, 7]);
    }

    #[test]
    fn test_arena_benchmark_allocation_reduction() {
        // Build a larger graph to measure allocation behavior
        let mut engine = MockEngine::new();
        let dim = 384;
        let num_nodes = 200;

        // Create nodes with semi-random embeddings
        for i in 0..num_nodes {
            let mut emb = vec![0.0f32; dim];
            emb[i % dim] = 1.0;
            engine.add_node(make_node(i as u64, emb));
        }

        // Create edges: linear chain + some cross-links
        for i in 0..(num_nodes - 1) {
            engine.add_edge_undirected(i as u64, (i + 1) as u64);
        }
        // Cross-links every 10 nodes
        for i in (0..num_nodes - 10).step_by(10) {
            engine.add_edge_undirected(i as u64, (i + 10) as u64);
        }

        let query = unit_vec(dim, 0);

        // Run arena version multiple times
        let mut arena = BfsArena::new();
        let start = std::time::Instant::now();
        for _ in 0..100 {
            let _ = arena.find_related(&engine, &query, 5, 3, 0.5).unwrap();
        }
        let arena_duration = start.elapsed();

        // Run standard version multiple times
        let start = std::time::Instant::now();
        for _ in 0..100 {
            let _ = crate::find_related(&engine, &query, 5, 3, 0.5).unwrap();
        }
        let std_duration = start.elapsed();

        // Verify results are equivalent (same node set; order may differ for tied scores)
        let arena_results = arena.find_related(&engine, &query, 5, 3, 0.5).unwrap();
        let std_results = crate::find_related(&engine, &query, 5, 3, 0.5).unwrap();

        let mut arena_ids: Vec<u64> = arena_results.iter().map(|n| n.id).collect();
        let mut std_ids: Vec<u64> = std_results.iter().map(|n| n.id).collect();
        arena_ids.sort();
        std_ids.sort();
        assert_eq!(arena_ids, std_ids, "Arena and standard should discover same nodes");

        // Print timing for manual inspection (not enforced — depends on hardware)
        eprintln!(
            "Arena: {:?}, Standard: {:?}, Arena bytes: {}",
            arena_duration,
            std_duration,
            arena.allocated_bytes()
        );
    }

    #[test]
    fn test_arena_score_map_best_score() {
        let bump = Bump::new();
        let entries = bumpalo::collections::Vec::new_in(&bump);
        let mut scores = ArenaScoreMap::new(entries);

        scores.update(1, 0.5);
        scores.update(2, 0.3);
        scores.update(1, 0.9); // higher → should replace
        scores.update(2, 0.1); // lower → should keep 0.3

        let result = scores.drain();
        let score_1 = result.iter().find(|(id, _)| *id == 1).unwrap().1;
        let score_2 = result.iter().find(|(id, _)| *id == 2).unwrap().1;

        assert!((score_1 - 0.9).abs() < 1e-6, "Should keep highest score for node 1");
        assert!((score_2 - 0.3).abs() < 1e-6, "Should keep highest score for node 2");
    }
}
