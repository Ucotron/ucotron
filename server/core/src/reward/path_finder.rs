//! Graph path finder for reward calculation.
//!
//! Finds ALL paths between two nodes up to a configurable hop limit,
//! returning paths annotated with edge types for coherence scoring.
//! Uses iterative DFS with backtracking to enumerate paths without
//! combinatorial explosion (bounded by max_hops).

use crate::backends::GraphBackend;
use crate::types::{EdgeType, NodeId};

use super::PathWithEdges;

/// Configuration for the path finder.
#[derive(Debug, Clone)]
pub struct PathFinderConfig {
    /// Maximum number of hops (edges) in returned paths.
    pub max_hops: u32,
    /// Maximum number of paths to return (prevents explosion on dense graphs).
    /// Default: 100.
    pub max_paths: usize,
}

impl Default for PathFinderConfig {
    fn default() -> Self {
        Self {
            max_hops: 4,
            max_paths: 100,
        }
    }
}

impl PathFinderConfig {
    /// Creates a config with the given hop limit and default max_paths.
    pub fn with_max_hops(max_hops: u32) -> Self {
        Self {
            max_hops,
            ..Default::default()
        }
    }
}

/// Finds all paths between `source` and `target` up to `config.max_hops` edges,
/// using the provided graph backend for adjacency lookups.
///
/// Returns paths annotated with edge types along each path. The result is
/// bounded by `config.max_paths` to prevent combinatorial explosion on
/// highly connected graphs.
///
/// # Algorithm
///
/// Iterative DFS with explicit stack and visited-per-path tracking.
/// Each stack frame holds the current node, the path so far, and the
/// edge types collected. When the target is found, the path is recorded.
/// Nodes already on the current path are skipped to prevent cycles.
pub fn find_paths(
    graph: &dyn GraphBackend,
    source: NodeId,
    target: NodeId,
    config: &PathFinderConfig,
) -> anyhow::Result<Vec<PathWithEdges>> {
    if source == target {
        return Ok(vec![PathWithEdges {
            nodes: vec![source],
            edge_types: vec![],
        }]);
    }

    // Pre-load all edges for efficient lookup
    let all_edges = graph.get_all_edges_full()?;

    // Build adjacency map: node -> Vec<(neighbor, edge_type)>
    // Include both directions since the graph is treated as bidirectional
    let mut adj: std::collections::HashMap<NodeId, Vec<(NodeId, EdgeType)>> =
        std::collections::HashMap::new();
    for edge in &all_edges {
        adj.entry(edge.source)
            .or_default()
            .push((edge.target, edge.edge_type));
        adj.entry(edge.target)
            .or_default()
            .push((edge.source, edge.edge_type));
    }

    let mut results = Vec::new();

    // DFS stack: (current_node, path_so_far, edge_types_so_far)
    let mut stack: Vec<(NodeId, Vec<NodeId>, Vec<EdgeType>)> = Vec::new();
    stack.push((source, vec![source], Vec::new()));

    while let Some((current, path, edge_types)) = stack.pop() {
        if results.len() >= config.max_paths {
            break;
        }

        let depth = edge_types.len() as u32;
        if depth >= config.max_hops {
            continue;
        }

        if let Some(neighbors) = adj.get(&current) {
            for &(neighbor, et) in neighbors {
                // Skip nodes already on this path (cycle prevention)
                if path.contains(&neighbor) {
                    continue;
                }

                let mut new_path = path.clone();
                new_path.push(neighbor);
                let mut new_edge_types = edge_types.clone();
                new_edge_types.push(et);

                if neighbor == target {
                    results.push(PathWithEdges {
                        nodes: new_path,
                        edge_types: new_edge_types,
                    });
                    if results.len() >= config.max_paths {
                        break;
                    }
                } else {
                    stack.push((neighbor, new_path, new_edge_types));
                }
            }
        }
    }

    // Sort by path length (shorter paths first)
    results.sort_by_key(|p| p.hop_count());

    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Edge, Node, NodeType};
    use std::collections::HashMap;
    use std::sync::Mutex;

    /// Minimal mock graph backend for path finding tests.
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

        fn with_chain(node_ids: &[u64], edge_type: EdgeType) -> Self {
            let g = Self::new();
            let mut nodes = g.nodes.lock().unwrap();
            for &id in node_ids {
                nodes.insert(id, make_node(id));
            }
            drop(nodes);

            let mut edges = g.edges.lock().unwrap();
            for window in node_ids.windows(2) {
                edges.push(Edge {
                    source: window[0],
                    target: window[1],
                    edge_type,
                    weight: 1.0,
                    metadata: HashMap::new(),
                });
            }
            drop(edges);
            g
        }
    }

    fn make_node(id: u64) -> Node {
        Node {
            id,
            content: format!("node_{}", id),
            embedding: vec![],
            metadata: HashMap::new(),
            node_type: NodeType::Entity,
            timestamp: 100,
            media_type: None,
            media_uri: None,
            embedding_visual: None,
            timestamp_range: None,
            parent_video_id: None,
        }
    }

    impl GraphBackend for MockGraph {
        fn upsert_nodes(&self, nodes: &[Node]) -> anyhow::Result<()> {
            let mut map = self.nodes.lock().unwrap();
            for n in nodes {
                map.insert(n.id, n.clone());
            }
            Ok(())
        }
        fn upsert_edges(&self, edges: &[Edge]) -> anyhow::Result<()> {
            let mut vec = self.edges.lock().unwrap();
            vec.extend_from_slice(edges);
            Ok(())
        }
        fn get_node(&self, id: NodeId) -> anyhow::Result<Option<Node>> {
            Ok(self.nodes.lock().unwrap().get(&id).cloned())
        }
        fn get_neighbors(&self, _id: NodeId, _hops: u8) -> anyhow::Result<Vec<Node>> {
            Ok(vec![])
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
        fn get_community(&self, _: NodeId) -> anyhow::Result<Vec<NodeId>> {
            Ok(vec![])
        }
        fn get_all_nodes(&self) -> anyhow::Result<Vec<Node>> {
            Ok(self.nodes.lock().unwrap().values().cloned().collect())
        }
        fn get_all_edges(&self) -> anyhow::Result<Vec<(NodeId, NodeId, f32)>> {
            Ok(self
                .edges
                .lock()
                .unwrap()
                .iter()
                .map(|e| (e.source, e.target, e.weight))
                .collect())
        }
        fn get_all_edges_full(&self) -> anyhow::Result<Vec<Edge>> {
            Ok(self.edges.lock().unwrap().clone())
        }
        fn delete_nodes(&self, _ids: &[NodeId]) -> anyhow::Result<()> {
            Ok(())
        }
        fn store_community_assignments(
            &self,
            _: &HashMap<NodeId, crate::community::CommunityId>,
        ) -> anyhow::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn test_source_equals_target() {
        let g = MockGraph::new();
        let config = PathFinderConfig::with_max_hops(4);
        let paths = find_paths(&g, 1, 1, &config).unwrap();
        assert_eq!(paths.len(), 1);
        assert_eq!(paths[0].nodes, vec![1]);
        assert!(paths[0].edge_types.is_empty());
    }

    #[test]
    fn test_simple_chain_path() {
        // 1 -> 2 -> 3
        let g = MockGraph::with_chain(&[1, 2, 3], EdgeType::RelatesTo);
        let config = PathFinderConfig::with_max_hops(4);
        let paths = find_paths(&g, 1, 3, &config).unwrap();
        assert!(!paths.is_empty());
        // Should find the path 1->2->3
        let found = paths.iter().any(|p| p.nodes == vec![1, 2, 3]);
        assert!(found, "Expected path [1,2,3], got {:?}", paths);
    }

    #[test]
    fn test_no_path_exists() {
        // Disconnected: 1->2, 3->4
        let g = MockGraph::new();
        {
            let mut nodes = g.nodes.lock().unwrap();
            for id in [1, 2, 3, 4] {
                nodes.insert(id, make_node(id));
            }
        }
        {
            let mut edges = g.edges.lock().unwrap();
            edges.push(Edge {
                source: 1,
                target: 2,
                edge_type: EdgeType::RelatesTo,
                weight: 1.0,
                metadata: HashMap::new(),
            });
            edges.push(Edge {
                source: 3,
                target: 4,
                edge_type: EdgeType::RelatesTo,
                weight: 1.0,
                metadata: HashMap::new(),
            });
        }
        let config = PathFinderConfig::with_max_hops(10);
        let paths = find_paths(&g, 1, 4, &config).unwrap();
        assert!(paths.is_empty());
    }

    #[test]
    fn test_multiple_paths_diamond() {
        // Diamond graph: 1->2->4, 1->3->4
        let g = MockGraph::new();
        {
            let mut nodes = g.nodes.lock().unwrap();
            for id in [1, 2, 3, 4] {
                nodes.insert(id, make_node(id));
            }
        }
        {
            let mut edges = g.edges.lock().unwrap();
            edges.push(Edge {
                source: 1,
                target: 2,
                edge_type: EdgeType::RelatesTo,
                weight: 1.0,
                metadata: HashMap::new(),
            });
            edges.push(Edge {
                source: 1,
                target: 3,
                edge_type: EdgeType::CausedBy,
                weight: 1.0,
                metadata: HashMap::new(),
            });
            edges.push(Edge {
                source: 2,
                target: 4,
                edge_type: EdgeType::HasProperty,
                weight: 1.0,
                metadata: HashMap::new(),
            });
            edges.push(Edge {
                source: 3,
                target: 4,
                edge_type: EdgeType::Actor,
                weight: 1.0,
                metadata: HashMap::new(),
            });
        }
        let config = PathFinderConfig::with_max_hops(4);
        let paths = find_paths(&g, 1, 4, &config).unwrap();
        assert!(
            paths.len() >= 2,
            "Expected at least 2 paths, got {}",
            paths.len()
        );

        // Verify edge types are correctly tracked
        for p in &paths {
            assert_eq!(p.edge_types.len(), p.nodes.len() - 1);
        }
    }

    #[test]
    fn test_hop_limit_respected() {
        // 1->2->3->4->5 (4 hops)
        let g = MockGraph::with_chain(&[1, 2, 3, 4, 5], EdgeType::RelatesTo);
        // max_hops = 2 should NOT find path from 1 to 5
        let config = PathFinderConfig::with_max_hops(2);
        let paths = find_paths(&g, 1, 5, &config).unwrap();
        assert!(
            paths.is_empty(),
            "Should not find 4-hop path with max_hops=2"
        );

        // max_hops = 4 should find it
        let config = PathFinderConfig::with_max_hops(4);
        let paths = find_paths(&g, 1, 5, &config).unwrap();
        assert!(!paths.is_empty());
    }

    #[test]
    fn test_max_paths_limit() {
        // Create a graph with many paths: complete graph of 5 nodes
        let g = MockGraph::new();
        {
            let mut nodes = g.nodes.lock().unwrap();
            for id in 1..=5 {
                nodes.insert(id, make_node(id));
            }
        }
        {
            let mut edges = g.edges.lock().unwrap();
            for i in 1..=5u64 {
                for j in (i + 1)..=5 {
                    edges.push(Edge {
                        source: i,
                        target: j,
                        edge_type: EdgeType::RelatesTo,
                        weight: 1.0,
                        metadata: HashMap::new(),
                    });
                }
            }
        }
        let config = PathFinderConfig {
            max_hops: 4,
            max_paths: 5,
        };
        let paths = find_paths(&g, 1, 5, &config).unwrap();
        assert!(paths.len() <= 5, "Should respect max_paths limit");
    }

    #[test]
    fn test_bidirectional_traversal() {
        // Edge only goes 1->2, but we treat graph as bidirectional
        let g = MockGraph::new();
        {
            let mut nodes = g.nodes.lock().unwrap();
            nodes.insert(1, make_node(1));
            nodes.insert(2, make_node(2));
        }
        {
            let mut edges = g.edges.lock().unwrap();
            edges.push(Edge {
                source: 1,
                target: 2,
                edge_type: EdgeType::RelatesTo,
                weight: 1.0,
                metadata: HashMap::new(),
            });
        }
        let config = PathFinderConfig::with_max_hops(2);
        // Find path from 2 to 1 (reverse direction)
        let paths = find_paths(&g, 2, 1, &config).unwrap();
        assert!(
            !paths.is_empty(),
            "Should find reverse path via bidirectional edges"
        );
    }

    #[test]
    fn test_cycle_prevention() {
        // Triangle: 1->2->3->1
        let g = MockGraph::new();
        {
            let mut nodes = g.nodes.lock().unwrap();
            for id in [1, 2, 3] {
                nodes.insert(id, make_node(id));
            }
        }
        {
            let mut edges = g.edges.lock().unwrap();
            edges.push(Edge {
                source: 1,
                target: 2,
                edge_type: EdgeType::RelatesTo,
                weight: 1.0,
                metadata: HashMap::new(),
            });
            edges.push(Edge {
                source: 2,
                target: 3,
                edge_type: EdgeType::RelatesTo,
                weight: 1.0,
                metadata: HashMap::new(),
            });
            edges.push(Edge {
                source: 3,
                target: 1,
                edge_type: EdgeType::RelatesTo,
                weight: 1.0,
                metadata: HashMap::new(),
            });
        }
        let config = PathFinderConfig::with_max_hops(10);
        let paths = find_paths(&g, 1, 3, &config).unwrap();
        // All paths should be acyclic (no repeated nodes)
        for p in &paths {
            let unique: std::collections::HashSet<_> = p.nodes.iter().collect();
            assert_eq!(
                unique.len(),
                p.nodes.len(),
                "Path should not contain cycles: {:?}",
                p.nodes
            );
        }
    }

    #[test]
    fn test_paths_sorted_by_length() {
        // Diamond with different path lengths
        let g = MockGraph::new();
        {
            let mut nodes = g.nodes.lock().unwrap();
            for id in [1, 2, 3, 4] {
                nodes.insert(id, make_node(id));
            }
        }
        {
            let mut edges = g.edges.lock().unwrap();
            // Direct: 1->4
            edges.push(Edge {
                source: 1,
                target: 4,
                edge_type: EdgeType::RelatesTo,
                weight: 1.0,
                metadata: HashMap::new(),
            });
            // Via 2: 1->2->4
            edges.push(Edge {
                source: 1,
                target: 2,
                edge_type: EdgeType::CausedBy,
                weight: 1.0,
                metadata: HashMap::new(),
            });
            edges.push(Edge {
                source: 2,
                target: 4,
                edge_type: EdgeType::HasProperty,
                weight: 1.0,
                metadata: HashMap::new(),
            });
            // Via 3: 1->3->4
            edges.push(Edge {
                source: 1,
                target: 3,
                edge_type: EdgeType::Actor,
                weight: 1.0,
                metadata: HashMap::new(),
            });
            edges.push(Edge {
                source: 3,
                target: 4,
                edge_type: EdgeType::Object,
                weight: 1.0,
                metadata: HashMap::new(),
            });
        }
        let config = PathFinderConfig::with_max_hops(4);
        let paths = find_paths(&g, 1, 4, &config).unwrap();
        // Paths should be sorted by hop count
        for window in paths.windows(2) {
            assert!(
                window[0].hop_count() <= window[1].hop_count(),
                "Paths not sorted: {} hops before {} hops",
                window[0].hop_count(),
                window[1].hop_count()
            );
        }
    }

    #[test]
    fn test_edge_types_tracked_correctly() {
        // 1 -[CausedBy]-> 2 -[HasProperty]-> 3
        let g = MockGraph::new();
        {
            let mut nodes = g.nodes.lock().unwrap();
            for id in [1, 2, 3] {
                nodes.insert(id, make_node(id));
            }
        }
        {
            let mut edges = g.edges.lock().unwrap();
            edges.push(Edge {
                source: 1,
                target: 2,
                edge_type: EdgeType::CausedBy,
                weight: 1.0,
                metadata: HashMap::new(),
            });
            edges.push(Edge {
                source: 2,
                target: 3,
                edge_type: EdgeType::HasProperty,
                weight: 1.0,
                metadata: HashMap::new(),
            });
        }
        let config = PathFinderConfig::with_max_hops(4);
        let paths = find_paths(&g, 1, 3, &config).unwrap();
        assert!(!paths.is_empty());
        let p = &paths[0];
        assert_eq!(p.nodes, vec![1, 2, 3]);
        assert_eq!(
            p.edge_types,
            vec![EdgeType::CausedBy, EdgeType::HasProperty]
        );
    }

    #[test]
    fn test_empty_graph() {
        let g = MockGraph::new();
        let config = PathFinderConfig::with_max_hops(4);
        let paths = find_paths(&g, 1, 2, &config).unwrap();
        assert!(paths.is_empty());
    }

    #[test]
    fn test_config_defaults() {
        let config = PathFinderConfig::default();
        assert_eq!(config.max_hops, 4);
        assert_eq!(config.max_paths, 100);
    }
}
