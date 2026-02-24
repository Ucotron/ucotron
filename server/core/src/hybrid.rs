//! Generic hybrid search combining vector similarity with graph traversal.
//!
//! The `find_related` function is engine-agnostic and works with any `StorageEngine`.
//! It implements the US-2.3 pipeline:
//!
//! 1. Vector search → top-k most similar nodes
//! 2. For each top-k node → graph traversal of N hops
//! 3. Deduplicate and score by combined relevance (similarity × hop_decay^depth)
//! 4. Return unified sorted list
//!
//! ## Hop Decay
//!
//! Nodes discovered at deeper traversal depths receive exponentially decayed scores:
//! - Depth 0 (vector matches): `score = similarity`
//! - Depth 1 (1-hop neighbors): `score = similarity × hop_decay`
//! - Depth 2 (2-hop neighbors): `score = similarity × hop_decay²`
//!
//! The default `hop_decay` factor is 0.5, meaning each hop halves the relevance.

use crate::arena_traversal::BfsArena;
use crate::{Node, NodeId, StorageEngine};
use anyhow::Result;
use std::collections::{HashMap, HashSet, VecDeque};

/// Default hop decay factor: each additional hop multiplies the score by this value.
pub const DEFAULT_HOP_DECAY: f32 = 0.5;

/// Perform a hybrid search combining vector similarity with graph traversal.
///
/// This is the canonical implementation used by both HelixDB and CozoDB engines.
/// It is generic over `StorageEngine` so it can be tested and used independently.
///
/// # Algorithm
///
/// 1. Run `vector_search(query, top_k)` to find the most semantically similar nodes.
/// 2. For each vector result, perform a BFS traversal up to `hops` levels deep.
///    - At each level, discovered nodes receive a decayed score:
///      `score = vector_similarity × hop_decay^depth`
/// 3. If a node is reachable via multiple paths, keep the **highest** score.
/// 4. Return all discovered nodes sorted by score (descending).
///
/// # Parameters
///
/// - `engine`: Reference to a `StorageEngine` implementation
/// - `query`: Query embedding vector (384-dim, L2-normalized)
/// - `top_k`: Number of vector search results to seed the traversal
/// - `hops`: Maximum graph traversal depth from each seed node
/// - `hop_decay`: Score decay factor per hop (typically 0.5)
pub fn find_related<E: StorageEngine>(
    engine: &E,
    query: &[f32],
    top_k: usize,
    hops: u8,
    hop_decay: f32,
) -> Result<Vec<Node>> {
    // Step 1: Vector search for seed nodes
    let vector_results = engine.vector_search(query, top_k)?;

    // Track best score per node (highest wins if reachable via multiple paths)
    let mut best_scores: HashMap<NodeId, f32> = HashMap::new();

    for (seed_id, similarity) in &vector_results {
        // The seed node itself gets the full vector similarity score
        let entry = best_scores.entry(*seed_id).or_insert(0.0);
        if *similarity > *entry {
            *entry = *similarity;
        }

        // Step 2: BFS traversal from seed, tracking depth for decay calculation
        if hops > 0 {
            let mut visited = HashSet::new();
            let mut queue = VecDeque::new();
            visited.insert(*seed_id);
            queue.push_back((*seed_id, 0u8));

            while let Some((current_id, depth)) = queue.pop_front() {
                if depth >= hops {
                    continue;
                }

                let next_depth = depth + 1;
                let decay = hop_decay.powi(next_depth as i32);
                let decayed_score = similarity * decay;

                // Get direct neighbors (1-hop from current)
                let neighbors = engine.get_neighbors(current_id, 1)?;
                for neighbor in &neighbors {
                    if visited.insert(neighbor.id) {
                        // Update score if this path gives a better score
                        let entry = best_scores.entry(neighbor.id).or_insert(0.0);
                        if decayed_score > *entry {
                            *entry = decayed_score;
                        }
                        queue.push_back((neighbor.id, next_depth));
                    }
                }
            }
        }
    }

    // Step 3: Collect all scored nodes and sort by score descending
    let mut scored: Vec<(NodeId, f32)> = best_scores.into_iter().collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Step 4: Fetch full node data for each result
    let mut results = Vec::with_capacity(scored.len());
    for (node_id, _score) in &scored {
        if let Some(node) = engine.get_node(*node_id)? {
            results.push(node);
        }
    }

    Ok(results)
}

/// Arena-allocated variant of [`find_related`] that reduces heap allocations.
///
/// Uses a [`BfsArena`] to allocate BFS traversal structures from a bump
/// allocator, avoiding per-traversal `HashMap`/`HashSet` creation overhead.
///
/// For single-shot queries, the overhead of arena creation makes this
/// similar to the standard version. For repeated queries (e.g., batch
/// search benchmarks), reuse the arena via [`BfsArena::find_related`].
pub fn arena_find_related<E: StorageEngine>(
    engine: &E,
    query: &[f32],
    top_k: usize,
    hops: u8,
    hop_decay: f32,
) -> Result<Vec<Node>> {
    let mut arena = BfsArena::new();
    arena.find_related(engine, query, top_k, hops, hop_decay)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Config, Edge, InsertStats, Node, NodeType};
    use std::collections::HashMap;

    /// Minimal in-memory StorageEngine for testing hybrid search logic.
    struct MockEngine {
        nodes: HashMap<NodeId, Node>,
        /// Adjacency list: node_id → [(neighbor_id, edge_type)]
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
            // Simple 1-hop only (find_related does its own BFS)
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
            find_related(self, query, top_k, hops, DEFAULT_HOP_DECAY)
        }

        fn find_path(&self, _source: NodeId, _target: NodeId, _max_depth: u32) -> Result<Option<Vec<NodeId>>> {
            Ok(None) // not needed for hybrid search tests
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
        v[idx] = 1.0;
        v
    }

    #[test]
    fn test_find_related_vector_only() {
        // With hops=0, find_related should return only vector search results
        let mut engine = MockEngine::new();
        engine.add_node(make_node(0, unit_vec(384, 0)));
        engine.add_node(make_node(1, unit_vec(384, 1)));
        engine.add_node(make_node(2, unit_vec(384, 2)));

        let query = unit_vec(384, 0);
        let results = find_related(&engine, &query, 2, 0, 0.5).unwrap();

        // Should return top-2 by similarity: node 0 (sim=1.0) then something with sim=0.0
        assert!(results.len() <= 2);
        assert_eq!(results[0].id, 0, "Node 0 should be the top match");
    }

    #[test]
    fn test_find_related_expands_graph() {
        // Verify graph expansion discovers nodes not in vector results
        let mut engine = MockEngine::new();

        // Node 0: strong vector match
        engine.add_node(make_node(0, unit_vec(384, 0)));
        // Node 1: zero vector match, but connected to node 0
        engine.add_node(make_node(1, vec![0.0; 384]));
        // Node 2: zero vector match, 2 hops from node 0
        engine.add_node(make_node(2, vec![0.0; 384]));

        engine.add_edge_undirected(0, 1);
        engine.add_edge_undirected(1, 2);

        let query = unit_vec(384, 0);
        let results = find_related(&engine, &query, 1, 2, 0.5).unwrap();

        let ids: Vec<u64> = results.iter().map(|n| n.id).collect();
        assert!(ids.contains(&0), "Should include vector match (node 0)");
        assert!(ids.contains(&1), "Should include 1-hop neighbor (node 1)");
        assert!(ids.contains(&2), "Should include 2-hop neighbor (node 2)");
    }

    #[test]
    fn test_find_related_decay_ordering() {
        // Verify that closer nodes rank higher than farther ones
        let mut engine = MockEngine::new();

        engine.add_node(make_node(0, unit_vec(384, 0))); // vector match
        engine.add_node(make_node(1, vec![0.0; 384])); // 1-hop
        engine.add_node(make_node(2, vec![0.0; 384])); // 2-hop

        engine.add_edge_undirected(0, 1);
        engine.add_edge_undirected(1, 2);

        let query = unit_vec(384, 0);
        let results = find_related(&engine, &query, 1, 2, 0.5).unwrap();

        // Order should be: node 0 (score=1.0), node 1 (score=0.5), node 2 (score=0.25)
        assert_eq!(results[0].id, 0, "Vector match should rank first");
        assert_eq!(results[1].id, 1, "1-hop neighbor should rank second");
        assert_eq!(results[2].id, 2, "2-hop neighbor should rank third");
    }

    #[test]
    fn test_find_related_deterministic() {
        // Same input should always produce the same output
        let mut engine = MockEngine::new();
        engine.add_node(make_node(0, unit_vec(384, 0)));
        engine.add_node(make_node(1, unit_vec(384, 1)));
        engine.add_node(make_node(2, vec![0.0; 384]));
        engine.add_edge_undirected(0, 2);

        let query = unit_vec(384, 0);

        let r1 = find_related(&engine, &query, 2, 1, 0.5).unwrap();
        let r2 = find_related(&engine, &query, 2, 1, 0.5).unwrap();

        let ids1: Vec<u64> = r1.iter().map(|n| n.id).collect();
        let ids2: Vec<u64> = r2.iter().map(|n| n.id).collect();
        assert_eq!(ids1, ids2, "Repeated queries should return identical results");
    }

    #[test]
    fn test_find_related_best_score_wins() {
        // If a node is reachable via multiple paths, it should keep the highest score
        let mut engine = MockEngine::new();

        // Two vector matches, both connected to node 2
        let mut emb_a = vec![0.0f32; 384];
        emb_a[0] = 0.9;
        emb_a[1] = 0.1;
        let norm = (emb_a[0] * emb_a[0] + emb_a[1] * emb_a[1]).sqrt();
        emb_a[0] /= norm;
        emb_a[1] /= norm;

        engine.add_node(make_node(0, unit_vec(384, 0))); // sim=1.0 to query
        engine.add_node(make_node(1, emb_a));              // sim~=0.994 to query
        engine.add_node(make_node(2, vec![0.0; 384]));     // no vector match

        engine.add_edge_undirected(0, 2); // path 1: 0→2, score = 1.0 * 0.5 = 0.5
        engine.add_edge_undirected(1, 2); // path 2: 1→2, score = 0.994 * 0.5 ≈ 0.497

        let query = unit_vec(384, 0);
        let results = find_related(&engine, &query, 2, 1, 0.5).unwrap();

        let ids: Vec<u64> = results.iter().map(|n| n.id).collect();
        assert!(ids.contains(&2), "Node 2 should be discovered via graph");

        // Node 2 should appear after the vector matches (its best score is 0.5)
        let pos_0 = ids.iter().position(|&id| id == 0).unwrap();
        let pos_2 = ids.iter().position(|&id| id == 2).unwrap();
        assert!(
            pos_0 < pos_2,
            "Vector match (node 0) should rank above graph-only (node 2)"
        );
    }

    #[test]
    fn test_find_related_zero_hops() {
        // hops=0 means vector-only, no graph expansion
        let mut engine = MockEngine::new();
        engine.add_node(make_node(0, unit_vec(384, 0)));
        engine.add_node(make_node(1, vec![0.0; 384]));
        engine.add_edge_undirected(0, 1);

        let query = unit_vec(384, 0);
        let results = find_related(&engine, &query, 1, 0, 0.5).unwrap();

        let ids: Vec<u64> = results.iter().map(|n| n.id).collect();
        assert!(ids.contains(&0), "Vector match should be included");
        assert!(!ids.contains(&1), "No graph expansion should happen with hops=0");
    }

    #[test]
    fn test_find_related_no_vector_matches() {
        // If the query vector matches nothing, result should be empty
        let mut engine = MockEngine::new();
        engine.add_node(make_node(0, unit_vec(384, 0)));

        // Query perpendicular to all embeddings
        let query = unit_vec(384, 100);
        let results = find_related(&engine, &query, 1, 2, 0.5).unwrap();

        // Node 0 has similarity 0.0 to query (orthogonal unit vectors)
        // It may still be returned since top_k=1 and cosine_sim=0 is still a valid result
        // But the score should be 0
        for n in &results {
            // We just check the function doesn't panic
            let _ = n.id;
        }
    }

    #[test]
    fn test_find_related_single_node_no_edges() {
        // Graph with one node and no edges
        let mut engine = MockEngine::new();
        engine.add_node(make_node(0, unit_vec(384, 0)));

        let query = unit_vec(384, 0);
        let results = find_related(&engine, &query, 1, 3, 0.5).unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, 0);
    }

    #[test]
    fn test_find_related_cycle_in_graph() {
        // Cycle: 0→1→2→0 — should not loop infinitely
        let mut engine = MockEngine::new();
        engine.add_node(make_node(0, unit_vec(384, 0)));
        engine.add_node(make_node(1, vec![0.0; 384]));
        engine.add_node(make_node(2, vec![0.0; 384]));
        engine.add_edge_undirected(0, 1);
        engine.add_edge_undirected(1, 2);
        engine.add_edge_undirected(2, 0);

        let query = unit_vec(384, 0);
        let results = find_related(&engine, &query, 1, 10, 0.5).unwrap();

        // Should discover all 3 nodes without infinite loop
        let ids: Vec<u64> = results.iter().map(|n| n.id).collect();
        assert!(ids.contains(&0));
        assert!(ids.contains(&1));
        assert!(ids.contains(&2));
        assert_eq!(ids.len(), 3, "Should not have duplicates from cycle");
    }
}
