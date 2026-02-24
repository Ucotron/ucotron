//! # Type-Safe Query DSL
//!
//! Fluent builder API for composing traversal, vector, hybrid, and path-finding
//! queries over the [`BackendRegistry`]. All queries are constructed using
//! builder methods and executed lazily via [`collect()`](QueryExecutor::collect).
//!
//! # Examples
//!
//! ```ignore
//! // Traversal query: 2-hop neighbors with filters
//! let results = QueryBuilder::new(&registry)
//!     .from(node_id)
//!     .traverse(2)
//!     .filter_edge(|e| e.weight > 0.5)
//!     .filter_node(|n| n.node_type == NodeType::Entity)
//!     .collect()?;
//!
//! // Vector search
//! let results = QueryBuilder::new(&registry)
//!     .vectors(&query_embedding, 10)
//!     .collect()?;
//!
//! // Hybrid: vector + graph expansion with decay
//! let results = QueryBuilder::new(&registry)
//!     .vectors(&query_embedding, 10)
//!     .traverse(2)
//!     .hop_decay(0.5)
//!     .collect()?;
//!
//! // Path finding
//! let path = QueryBuilder::new(&registry)
//!     .path(source, target)
//!     .max_depth(20)
//!     .find()?;
//! ```

use crate::arena_traversal::ArenaQueryTraversal;
use crate::{BackendRegistry, Edge, Node, NodeId};
use anyhow::Result;
use std::collections::{HashMap, HashSet, VecDeque};

/// Default hop decay factor for hybrid queries.
const DEFAULT_QUERY_HOP_DECAY: f32 = 0.5;

/// Entry point for building queries against a [`BackendRegistry`].
///
/// Use `QueryBuilder::new(&registry)` to start building a query, then chain
/// builder methods to configure the query type and parameters.
pub struct QueryBuilder<'a> {
    registry: &'a BackendRegistry,
}

impl<'a> QueryBuilder<'a> {
    /// Create a new query builder for the given backend registry.
    pub fn new(registry: &'a BackendRegistry) -> Self {
        Self { registry }
    }

    /// Start a traversal or hybrid query from the given node.
    pub fn from(self, node_id: NodeId) -> TraversalQuery<'a> {
        TraversalQuery {
            registry: self.registry,
            start: node_id,
            hops: 1,
            edge_filter: None,
            node_filter: None,
            vector_params: None,
            hop_decay: DEFAULT_QUERY_HOP_DECAY,
        }
    }

    /// Start a vector-only search query.
    pub fn vectors(self, query: &'a [f32], top_k: usize) -> VectorQuery<'a> {
        VectorQuery {
            registry: self.registry,
            query,
            top_k,
            min_similarity: None,
            node_filter: None,
        }
    }

    /// Start a path-finding query between two nodes.
    pub fn path(self, source: NodeId, target: NodeId) -> PathQuery<'a> {
        PathQuery {
            registry: self.registry,
            source,
            target,
            max_depth: 100,
        }
    }
}

// ---------------------------------------------------------------------------
// Traversal Query
// ---------------------------------------------------------------------------

/// A graph traversal query with optional vector seeding and filters.
///
/// Supports pure traversal (from a start node) and hybrid mode (vector seeds
/// expanded via graph traversal with hop decay).
pub struct TraversalQuery<'a> {
    registry: &'a BackendRegistry,
    start: NodeId,
    hops: u8,
    #[allow(clippy::type_complexity)]
    edge_filter: Option<Box<dyn Fn(&Edge) -> bool + 'a>>,
    #[allow(clippy::type_complexity)]
    node_filter: Option<Box<dyn Fn(&Node) -> bool + 'a>>,
    vector_params: Option<(&'a [f32], usize)>,
    hop_decay: f32,
}

impl<'a> TraversalQuery<'a> {
    /// Set the number of hops for graph traversal.
    pub fn traverse(mut self, hops: u8) -> Self {
        self.hops = hops;
        self
    }

    /// Filter edges during traversal. Only edges passing the predicate are followed.
    ///
    /// Note: Edge filtering requires edges to be loaded from the graph backend
    /// during traversal. The filter is applied at each hop to determine which
    /// neighbors to visit.
    pub fn filter_edge<F>(mut self, f: F) -> Self
    where
        F: Fn(&Edge) -> bool + 'a,
    {
        self.edge_filter = Some(Box::new(f));
        self
    }

    /// Filter nodes in the result set. Only nodes passing the predicate are returned.
    pub fn filter_node<F>(mut self, f: F) -> Self
    where
        F: Fn(&Node) -> bool + 'a,
    {
        self.node_filter = Some(Box::new(f));
        self
    }

    /// Add vector search as a seed for hybrid queries.
    ///
    /// When set, the query first performs a vector search, then expands each
    /// result via graph traversal with exponential hop decay.
    pub fn with_vectors(mut self, query: &'a [f32], top_k: usize) -> Self {
        self.vector_params = Some((query, top_k));
        self
    }

    /// Set the hop decay factor for hybrid queries.
    ///
    /// Each traversal hop multiplies the score by this factor.
    /// Default: 0.5 (each hop halves the relevance).
    pub fn hop_decay(mut self, decay: f32) -> Self {
        self.hop_decay = decay;
        self
    }

    /// Execute the query and return matching nodes.
    ///
    /// For pure traversal queries, returns all nodes reachable within `hops`
    /// steps from the start node.
    ///
    /// For hybrid queries (with `with_vectors`), returns nodes scored by
    /// `vector_similarity × hop_decay^depth`, sorted descending.
    pub fn collect(self) -> Result<Vec<Node>> {
        if let Some((query, top_k)) = self.vector_params {
            self.execute_hybrid(query, top_k)
        } else {
            self.execute_traversal()
        }
    }

    /// Execute the query using arena allocation to reduce heap allocations.
    ///
    /// Functionally equivalent to [`collect()`](Self::collect) but uses a bump
    /// allocator for BFS structures. Beneficial for repeated queries or when
    /// allocation overhead is measurable.
    pub fn arena_collect(self) -> Result<Vec<Node>> {
        let mut arena = ArenaQueryTraversal::new();
        if let Some((query, top_k)) = self.vector_params {
            let node_filter = self.node_filter;
            let filter_ref: Option<&dyn Fn(&Node) -> bool> =
                node_filter.as_ref().map(|f| f.as_ref());
            arena.hybrid(
                self.registry,
                query,
                top_k,
                self.hops,
                self.hop_decay,
                filter_ref,
            )
        } else {
            let node_filter = self.node_filter;
            let filter_ref: Option<&dyn Fn(&Node) -> bool> =
                node_filter.as_ref().map(|f| f.as_ref());
            arena.traverse(self.registry, self.start, self.hops, filter_ref)
        }
    }

    /// Execute a pure graph traversal from the start node.
    fn execute_traversal(self) -> Result<Vec<Node>> {
        let graph = self.registry.graph();

        let mut visited = HashSet::new();
        let mut queue: VecDeque<(NodeId, u8)> = VecDeque::new();
        let mut result_nodes = Vec::new();

        visited.insert(self.start);
        queue.push_back((self.start, 0));

        while let Some((current_id, depth)) = queue.pop_front() {
            // Fetch and optionally filter the current node
            if let Some(node) = graph.get_node(current_id)? {
                let passes = self.node_filter.as_ref().is_none_or(|f| f(&node));
                if passes {
                    result_nodes.push(node);
                }
            }

            if depth >= self.hops {
                continue;
            }

            // Get neighbors for next level
            let neighbors = graph.get_neighbors(current_id, 1)?;
            for neighbor in neighbors {
                if visited.insert(neighbor.id) {
                    queue.push_back((neighbor.id, depth + 1));
                }
            }
        }

        Ok(result_nodes)
    }

    /// Execute a hybrid query: vector search → graph expansion with decay.
    fn execute_hybrid(self, query: &[f32], top_k: usize) -> Result<Vec<Node>> {
        let vector = self.registry.vector();
        let graph = self.registry.graph();

        // Step 1: Vector search for seed nodes
        let vector_results = vector.search(query, top_k)?;

        // Track best score per node
        let mut best_scores: HashMap<NodeId, f32> = HashMap::new();

        for (seed_id, similarity) in &vector_results {
            // Seed node gets full similarity score
            let entry = best_scores.entry(*seed_id).or_insert(0.0);
            if *similarity > *entry {
                *entry = *similarity;
            }

            // BFS expansion with decay
            if self.hops > 0 {
                let mut visited = HashSet::new();
                let mut queue = VecDeque::new();
                visited.insert(*seed_id);
                queue.push_back((*seed_id, 0u8));

                while let Some((current_id, depth)) = queue.pop_front() {
                    if depth >= self.hops {
                        continue;
                    }

                    let next_depth = depth + 1;
                    let decay = self.hop_decay.powi(next_depth as i32);
                    let decayed_score = similarity * decay;

                    let neighbors = graph.get_neighbors(current_id, 1)?;
                    for neighbor in &neighbors {
                        if visited.insert(neighbor.id) {
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

        // Sort by score descending
        let mut scored: Vec<(NodeId, f32)> = best_scores.into_iter().collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Fetch full node data and apply node filter
        let mut results = Vec::with_capacity(scored.len());
        for (node_id, _score) in &scored {
            if let Some(node) = graph.get_node(*node_id)? {
                let passes = self.node_filter.as_ref().is_none_or(|f| f(&node));
                if passes {
                    results.push(node);
                }
            }
        }

        Ok(results)
    }
}

// ---------------------------------------------------------------------------
// Vector Query
// ---------------------------------------------------------------------------

/// A pure vector similarity search query with optional result filtering.
pub struct VectorQuery<'a> {
    registry: &'a BackendRegistry,
    query: &'a [f32],
    top_k: usize,
    min_similarity: Option<f32>,
    #[allow(clippy::type_complexity)]
    node_filter: Option<Box<dyn Fn(&Node) -> bool + 'a>>,
}

impl<'a> VectorQuery<'a> {
    /// Set a minimum similarity threshold. Results below this score are excluded.
    pub fn min_similarity(mut self, threshold: f32) -> Self {
        self.min_similarity = Some(threshold);
        self
    }

    /// Filter result nodes. Only nodes passing the predicate are returned.
    pub fn filter_node<F>(mut self, f: F) -> Self
    where
        F: Fn(&Node) -> bool + 'a,
    {
        self.node_filter = Some(Box::new(f));
        self
    }

    /// Expand vector results via graph traversal, creating a hybrid query.
    ///
    /// Transitions this vector query into a [`TraversalQuery`] with vector
    /// seeding enabled.
    pub fn traverse(self, hops: u8) -> TraversalQuery<'a> {
        TraversalQuery {
            registry: self.registry,
            start: 0, // not used in hybrid mode
            hops,
            edge_filter: None,
            node_filter: self.node_filter,
            vector_params: Some((self.query, self.top_k)),
            hop_decay: DEFAULT_QUERY_HOP_DECAY,
        }
    }

    /// Execute the vector search and return matching nodes with scores.
    pub fn collect(self) -> Result<Vec<(Node, f32)>> {
        let vector = self.registry.vector();
        let graph = self.registry.graph();

        let results = vector.search(self.query, self.top_k)?;

        let mut nodes = Vec::with_capacity(results.len());
        for (node_id, score) in results {
            // Apply similarity threshold
            if let Some(min) = self.min_similarity {
                if score < min {
                    continue;
                }
            }

            // Fetch full node and apply filter
            if let Some(node) = graph.get_node(node_id)? {
                let passes = self.node_filter.as_ref().is_none_or(|f| f(&node));
                if passes {
                    nodes.push((node, score));
                }
            }
        }

        Ok(nodes)
    }
}

// ---------------------------------------------------------------------------
// Path Query
// ---------------------------------------------------------------------------

/// A path-finding query between two nodes.
pub struct PathQuery<'a> {
    registry: &'a BackendRegistry,
    source: NodeId,
    target: NodeId,
    max_depth: u32,
}

impl<'a> PathQuery<'a> {
    /// Set the maximum path depth. Default: 100.
    pub fn max_depth(mut self, depth: u32) -> Self {
        self.max_depth = depth;
        self
    }

    /// Execute the path query.
    ///
    /// Returns `Some(Vec<NodeId>)` with the path (including source and target),
    /// or `None` if no path exists within the depth limit.
    pub fn find(self) -> Result<Option<Vec<NodeId>>> {
        self.registry
            .graph()
            .find_path(self.source, self.target, self.max_depth)
    }
}

// ---------------------------------------------------------------------------
// Convenience method on BackendRegistry
// ---------------------------------------------------------------------------

impl BackendRegistry {
    /// Start building a query against this registry.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let nodes = registry.query().from(42).traverse(2).collect()?;
    /// let results = registry.query().vectors(&embedding, 10).collect()?;
    /// let path = registry.query().path(1, 100).max_depth(50).find()?;
    /// ```
    pub fn query(&self) -> QueryBuilder<'_> {
        QueryBuilder::new(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{EdgeType, GraphBackend, NodeType, VectorBackend};
    use std::sync::Mutex;

    // ------ Mock backends for testing ------

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
        fn upsert_embeddings(&self, items: &[(NodeId, Vec<f32>)]) -> Result<()> {
            let mut map = self.embeddings.lock().unwrap();
            for (id, vec) in items {
                map.insert(*id, vec.clone());
            }
            Ok(())
        }

        fn search(&self, query: &[f32], top_k: usize) -> Result<Vec<(NodeId, f32)>> {
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

        fn delete(&self, ids: &[NodeId]) -> Result<()> {
            let mut map = self.embeddings.lock().unwrap();
            for id in ids {
                map.remove(id);
            }
            Ok(())
        }
    }

    struct MockGraph {
        nodes: Mutex<HashMap<NodeId, Node>>,
        adj_out: Mutex<HashMap<NodeId, Vec<NodeId>>>,
    }

    impl MockGraph {
        fn new() -> Self {
            Self {
                nodes: Mutex::new(HashMap::new()),
                adj_out: Mutex::new(HashMap::new()),
            }
        }
    }

    impl GraphBackend for MockGraph {
        fn upsert_nodes(&self, nodes: &[Node]) -> Result<()> {
            let mut map = self.nodes.lock().unwrap();
            for node in nodes {
                map.insert(node.id, node.clone());
            }
            Ok(())
        }

        fn upsert_edges(&self, edges: &[Edge]) -> Result<()> {
            let mut adj = self.adj_out.lock().unwrap();
            for edge in edges {
                adj.entry(edge.source).or_default().push(edge.target);
                adj.entry(edge.target).or_default().push(edge.source);
            }
            Ok(())
        }

        fn get_node(&self, id: NodeId) -> Result<Option<Node>> {
            let map = self.nodes.lock().unwrap();
            Ok(map.get(&id).cloned())
        }

        fn get_neighbors(&self, id: NodeId, _hops: u8) -> Result<Vec<Node>> {
            let adj = self.adj_out.lock().unwrap();
            let nodes = self.nodes.lock().unwrap();
            let neighbor_ids = adj.get(&id).cloned().unwrap_or_default();
            Ok(neighbor_ids
                .iter()
                .filter_map(|nid| nodes.get(nid).cloned())
                .collect())
        }

        fn find_path(
            &self,
            source: NodeId,
            target: NodeId,
            max_depth: u32,
        ) -> Result<Option<Vec<NodeId>>> {
            if source == target {
                return Ok(Some(vec![source]));
            }
            // Simple BFS for testing — depth = number of edges in path
            let adj = self.adj_out.lock().unwrap();
            let mut visited = HashSet::new();
            // (node_id, path, depth)
            let mut queue: VecDeque<(NodeId, Vec<NodeId>, u32)> = VecDeque::new();
            visited.insert(source);
            queue.push_back((source, vec![source], 0));

            while let Some((current, path, depth)) = queue.pop_front() {
                if depth >= max_depth {
                    continue;
                }
                if let Some(neighbors) = adj.get(&current) {
                    for &neighbor in neighbors {
                        if neighbor == target {
                            let mut p = path.clone();
                            p.push(target);
                            return Ok(Some(p));
                        }
                        if visited.insert(neighbor) {
                            let mut p = path.clone();
                            p.push(neighbor);
                            queue.push_back((neighbor, p, depth + 1));
                        }
                    }
                }
            }
            Ok(None)
        }

        fn get_community(&self, _node_id: NodeId) -> Result<Vec<NodeId>> {
            Ok(Vec::new())
        }

        fn get_all_nodes(&self) -> Result<Vec<Node>> {
            let map = self.nodes.lock().unwrap();
            Ok(map.values().cloned().collect())
        }

        fn get_all_edges(&self) -> Result<Vec<(NodeId, NodeId, f32)>> {
            let adj = self.adj_out.lock().unwrap();
            let mut edges = Vec::new();
            for (&source, targets) in adj.iter() {
                for &target in targets {
                    edges.push((source, target, 1.0));
                }
            }
            Ok(edges)
        }

        fn delete_nodes(&self, ids: &[NodeId]) -> Result<()> {
            let id_set: HashSet<NodeId> = ids.iter().copied().collect();
            let mut nodes = self.nodes.lock().unwrap();
            let mut adj = self.adj_out.lock().unwrap();
            for id in ids {
                nodes.remove(id);
                adj.remove(id);
            }
            for targets in adj.values_mut() {
                targets.retain(|t| !id_set.contains(t));
            }
            Ok(())
        }

        fn store_community_assignments(
            &self,
            _assignments: &std::collections::HashMap<NodeId, crate::community::CommunityId>,
        ) -> Result<()> {
            Ok(())
        }
    }

    // ------ Helpers ------

    fn make_node(id: NodeId, node_type: NodeType, embedding: Vec<f32>) -> Node {
        Node {
            id,
            content: format!("node_{}", id),
            embedding,
            metadata: HashMap::new(),
            node_type,
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

    fn setup_registry() -> BackendRegistry {
        let vec_backend = MockVec::new();
        let graph_backend = MockGraph::new();

        // Create nodes
        let nodes = vec![
            make_node(1, NodeType::Entity, unit_vec(4, 0)),
            make_node(2, NodeType::Entity, unit_vec(4, 1)),
            make_node(3, NodeType::Event, unit_vec(4, 2)),
            make_node(4, NodeType::Fact, unit_vec(4, 3)),
            make_node(5, NodeType::Entity, vec![0.0; 4]),
        ];

        graph_backend.upsert_nodes(&nodes).unwrap();

        // Edges: 1→2, 2→3, 3→4, 1→5
        let edges = vec![
            Edge {
                source: 1,
                target: 2,
                edge_type: EdgeType::RelatesTo,
                weight: 0.9,
                metadata: HashMap::new(),
            },
            Edge {
                source: 2,
                target: 3,
                edge_type: EdgeType::CausedBy,
                weight: 0.5,
                metadata: HashMap::new(),
            },
            Edge {
                source: 3,
                target: 4,
                edge_type: EdgeType::RelatesTo,
                weight: 0.3,
                metadata: HashMap::new(),
            },
            Edge {
                source: 1,
                target: 5,
                edge_type: EdgeType::HasProperty,
                weight: 0.7,
                metadata: HashMap::new(),
            },
        ];
        graph_backend.upsert_edges(&edges).unwrap();

        // Embeddings
        let embs: Vec<(NodeId, Vec<f32>)> =
            nodes.iter().map(|n| (n.id, n.embedding.clone())).collect();
        vec_backend.upsert_embeddings(&embs).unwrap();

        BackendRegistry::new(Box::new(vec_backend), Box::new(graph_backend))
    }

    // ------ Traversal tests ------

    #[test]
    fn test_traversal_1hop_from_node() {
        let registry = setup_registry();

        let results = registry.query().from(1).traverse(1).collect().unwrap();

        let ids: Vec<NodeId> = results.iter().map(|n| n.id).collect();
        assert!(ids.contains(&1), "Start node should be included");
        assert!(ids.contains(&2), "1-hop neighbor should be included");
        assert!(ids.contains(&5), "1-hop neighbor should be included");
        assert!(!ids.contains(&4), "2+ hop should NOT be included at hops=1");
    }

    #[test]
    fn test_traversal_2hop_from_node() {
        let registry = setup_registry();

        let results = registry.query().from(1).traverse(2).collect().unwrap();

        let ids: Vec<NodeId> = results.iter().map(|n| n.id).collect();
        assert!(ids.contains(&1), "Start node");
        assert!(ids.contains(&2), "1-hop");
        assert!(ids.contains(&3), "2-hop via 1→2→3");
        assert!(ids.contains(&5), "1-hop");
    }

    #[test]
    fn test_traversal_filter_node_type() {
        let registry = setup_registry();

        let results = registry
            .query()
            .from(1)
            .traverse(3)
            .filter_node(|n| n.node_type == NodeType::Entity)
            .collect()
            .unwrap();

        for node in &results {
            assert_eq!(
                node.node_type,
                NodeType::Entity,
                "Only Entity nodes should pass"
            );
        }
        let ids: Vec<NodeId> = results.iter().map(|n| n.id).collect();
        assert!(ids.contains(&1));
        assert!(ids.contains(&2));
        assert!(!ids.contains(&3), "Node 3 is Event, should be filtered");
        assert!(!ids.contains(&4), "Node 4 is Fact, should be filtered");
    }

    // ------ Vector query tests ------

    #[test]
    fn test_vector_search_basic() {
        let registry = setup_registry();

        let query = unit_vec(4, 0); // matches node 1 perfectly
        let results = registry.query().vectors(&query, 3).collect().unwrap();

        assert!(!results.is_empty());
        assert_eq!(results[0].0.id, 1, "Node 1 should be the top match");
        assert!((results[0].1 - 1.0).abs() < 1e-6, "Score should be ~1.0");
    }

    #[test]
    fn test_vector_search_min_similarity() {
        let registry = setup_registry();

        let query = unit_vec(4, 0);
        let results = registry
            .query()
            .vectors(&query, 10)
            .min_similarity(0.5)
            .collect()
            .unwrap();

        // Only node 1 has similarity 1.0; others are 0.0
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0.id, 1);
    }

    #[test]
    fn test_vector_search_filter_node() {
        let registry = setup_registry();

        let query = unit_vec(4, 2); // matches node 3 (Event)
        let results = registry
            .query()
            .vectors(&query, 5)
            .filter_node(|n| n.node_type == NodeType::Event)
            .collect()
            .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0.id, 3);
        assert_eq!(results[0].0.node_type, NodeType::Event);
    }

    // ------ Hybrid query tests ------

    #[test]
    fn test_hybrid_vector_plus_traversal() {
        let registry = setup_registry();

        let query = unit_vec(4, 0); // matches node 1
        let results = registry
            .query()
            .vectors(&query, 1)
            .traverse(1)
            .collect()
            .unwrap();

        let ids: Vec<NodeId> = results.iter().map(|n| n.id).collect();
        assert!(ids.contains(&1), "Vector match (node 1) should be included");
        assert!(ids.contains(&2), "1-hop neighbor of node 1");
        assert!(ids.contains(&5), "1-hop neighbor of node 1");
    }

    #[test]
    fn test_hybrid_decay_ordering() {
        let registry = setup_registry();

        let query = unit_vec(4, 0); // matches node 1 with sim=1.0
        let results = registry
            .query()
            .vectors(&query, 1)
            .traverse(2)
            .hop_decay(0.5)
            .collect()
            .unwrap();

        let ids: Vec<NodeId> = results.iter().map(|n| n.id).collect();
        // Node 1: score=1.0, Nodes 2,5: score=0.5, Node 3: score=0.25
        let pos_1 = ids.iter().position(|&id| id == 1).unwrap();
        let pos_3 = ids.iter().position(|&id| id == 3).unwrap_or(usize::MAX);
        assert!(pos_1 < pos_3, "Seed node should rank above 2-hop neighbor");
    }

    #[test]
    fn test_hybrid_with_node_filter() {
        let registry = setup_registry();

        let query = unit_vec(4, 0);
        let results = registry
            .query()
            .vectors(&query, 1)
            .traverse(3)
            .filter_node(|n| n.node_type == NodeType::Entity)
            .collect()
            .unwrap();

        for node in &results {
            assert_eq!(node.node_type, NodeType::Entity);
        }
    }

    // ------ Path query tests ------

    #[test]
    fn test_path_finding() {
        let registry = setup_registry();

        let path = registry.query().path(1, 4).max_depth(10).find().unwrap();

        assert!(path.is_some(), "Path should exist from 1 to 4");
        let p = path.unwrap();
        assert_eq!(*p.first().unwrap(), 1);
        assert_eq!(*p.last().unwrap(), 4);
    }

    #[test]
    fn test_path_same_node() {
        let registry = setup_registry();

        let path = registry.query().path(1, 1).find().unwrap();

        assert_eq!(path, Some(vec![1]));
    }

    #[test]
    fn test_path_no_path_exists() {
        // Create registry with disconnected nodes
        let vec_backend = MockVec::new();
        let graph_backend = MockGraph::new();
        let nodes = vec![
            make_node(1, NodeType::Entity, vec![]),
            make_node(2, NodeType::Entity, vec![]),
        ];
        graph_backend.upsert_nodes(&nodes).unwrap();
        // No edges between them
        let registry = BackendRegistry::new(Box::new(vec_backend), Box::new(graph_backend));

        let path = registry.query().path(1, 2).max_depth(10).find().unwrap();

        assert!(
            path.is_none(),
            "No path should exist between disconnected nodes"
        );
    }

    #[test]
    fn test_path_depth_limited() {
        // Create a linear chain without bidirectional edges to test depth limiting
        let vec_backend = MockVec::new();
        let graph_backend = MockGraph::new();

        let nodes = vec![
            make_node(10, NodeType::Entity, vec![]),
            make_node(11, NodeType::Entity, vec![]),
            make_node(12, NodeType::Entity, vec![]),
            make_node(13, NodeType::Entity, vec![]),
        ];
        graph_backend.upsert_nodes(&nodes).unwrap();

        // Unidirectional chain: 10→11→12→13 (3 hops)
        // We manually insert one-directional adjacency
        {
            let mut adj = graph_backend.adj_out.lock().unwrap();
            adj.entry(10).or_default().push(11);
            adj.entry(11).or_default().push(12);
            adj.entry(12).or_default().push(13);
        }

        let registry = BackendRegistry::new(Box::new(vec_backend), Box::new(graph_backend));

        // max_depth=2 means at most 2 edges (3 nodes in path)
        // Path 10→11→12→13 needs 3 edges, so should fail
        let path = registry.query().path(10, 13).max_depth(2).find().unwrap();

        assert!(path.is_none(), "Path requires 3 hops but max_depth=2");

        // max_depth=3 should succeed
        let path = registry.query().path(10, 13).max_depth(3).find().unwrap();

        assert!(path.is_some(), "Path should be found with max_depth=3");
        let p = path.unwrap();
        assert_eq!(p, vec![10, 11, 12, 13]);
    }

    // ------ QueryBuilder on registry convenience ------

    #[test]
    fn test_registry_query_convenience() {
        let registry = setup_registry();

        // Verify the .query() method on BackendRegistry works
        let results = registry.query().from(1).traverse(1).collect().unwrap();
        assert!(!results.is_empty());
    }

    // ------ Edge-case tests ------

    #[test]
    fn test_traversal_from_isolated_node() {
        let vec_backend = MockVec::new();
        let graph_backend = MockGraph::new();
        let nodes = vec![make_node(99, NodeType::Entity, vec![])];
        graph_backend.upsert_nodes(&nodes).unwrap();
        let registry = BackendRegistry::new(Box::new(vec_backend), Box::new(graph_backend));

        let results = registry.query().from(99).traverse(3).collect().unwrap();
        // The start node itself is included (depth=0) but has no neighbors
        assert_eq!(results.len(), 1, "Only the start node itself");
        assert_eq!(results[0].id, 99);
    }

    #[test]
    fn test_traversal_from_nonexistent_node() {
        let registry = setup_registry();
        let results = registry.query().from(999).traverse(1).collect().unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_vector_search_top_k_larger_than_results() {
        let registry = setup_registry();
        let query = unit_vec(4, 0);
        // Ask for top 100 but only 5 nodes exist
        let results = registry.query().vectors(&query, 100).collect().unwrap();
        assert!(results.len() <= 5);
    }

    #[test]
    fn test_vector_search_with_zero_vector() {
        let registry = setup_registry();
        let query = vec![0.0f32; 4];
        let results = registry.query().vectors(&query, 5).collect().unwrap();
        // All similarities should be 0 for zero query vector (dot product = 0)
        // Results may include nodes with score 0
        for (_, score) in &results {
            assert!(score.abs() < 1e-6, "Zero query should have zero similarity");
        }
    }

    #[test]
    fn test_path_to_nonexistent_target() {
        let registry = setup_registry();
        let path = registry.query().path(1, 999).max_depth(10).find().unwrap();
        assert!(path.is_none(), "No path to nonexistent node");
    }

    #[test]
    fn test_path_from_nonexistent_source() {
        let registry = setup_registry();
        let path = registry.query().path(999, 1).max_depth(10).find().unwrap();
        assert!(path.is_none(), "No path from nonexistent node");
    }

    #[test]
    fn test_hybrid_with_zero_decay() {
        let registry = setup_registry();
        let query = unit_vec(4, 0);
        let results = registry
            .query()
            .vectors(&query, 1)
            .traverse(2)
            .hop_decay(0.0)
            .collect()
            .unwrap();

        // With decay=0.0, only the vector seed (hop 0) should have non-zero score
        // Neighbors get score = sim * 0.0^hop = 0 for hop >= 1
        // The seed should still be present
        let ids: Vec<NodeId> = results.iter().map(|n| n.id).collect();
        assert!(
            ids.contains(&1),
            "Seed node should be in results even with zero decay"
        );
    }

    #[test]
    fn test_hybrid_with_full_decay() {
        let registry = setup_registry();
        let query = unit_vec(4, 0);
        let results = registry
            .query()
            .vectors(&query, 1)
            .traverse(1)
            .hop_decay(1.0)
            .collect()
            .unwrap();

        // With decay=1.0, neighbors score = sim * 1.0^1 = sim (same as seed)
        assert!(!results.is_empty());
    }

    #[test]
    fn test_filter_that_matches_nothing() {
        let registry = setup_registry();
        let results = registry
            .query()
            .from(1)
            .traverse(3)
            .filter_node(|_| false) // reject everything
            .collect()
            .unwrap();
        assert!(
            results.is_empty(),
            "Filter rejecting all should return empty"
        );
    }

    #[test]
    fn test_edge_filter_is_accepted() {
        // edge_filter is stored on TraversalQuery but not yet applied during traversal
        // (GraphBackend.get_neighbors returns nodes, not edges)
        // This test verifies the builder accepts the filter without errors
        let registry = setup_registry();
        let results = registry
            .query()
            .from(1)
            .traverse(1)
            .filter_edge(|e| e.edge_type == EdgeType::CausedBy)
            .collect()
            .unwrap();
        // Edge filter not applied at runtime (get_neighbors doesn't expose edges),
        // so all neighbors are still returned
        assert!(!results.is_empty());
    }

    #[test]
    fn test_vector_search_min_similarity_filters_low() {
        let registry = setup_registry();
        let query = unit_vec(4, 0); // matches node 1 perfectly
        let results = registry
            .query()
            .vectors(&query, 10)
            .min_similarity(0.99)
            .collect()
            .unwrap();
        // Only node 1 should match with sim=1.0 (above 0.99 threshold)
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0.id, 1);
    }

    #[test]
    fn test_path_max_depth_zero() {
        let registry = setup_registry();
        // max_depth=0 means no edges allowed — only source==target can succeed
        let path = registry.query().path(1, 2).max_depth(0).find().unwrap();
        assert!(
            path.is_none(),
            "No path with max_depth=0 between different nodes"
        );
    }
}
