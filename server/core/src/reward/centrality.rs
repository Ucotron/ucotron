//! Path centrality calculation for reward weighting.
//!
//! Computes degree-based node centrality from graph edges and uses it to
//! weight path rewards. Paths through high-degree (hub) nodes receive higher
//! scores, indicating more reliable connections.
//!
//! # Design
//!
//! The [`PathCentralityCalculator`] builds a degree map from all graph edges
//! (counting both directions), then normalizes each node's degree against the
//! global maximum. This provides graph-aware centrality rather than
//! path-local normalization.

use crate::backends::GraphBackend;
use crate::types::NodeId;
use std::collections::HashMap;

use super::{PathRewardCalculator, PathWithEdges, RewardScore};

/// Degree-based centrality calculator for graph paths.
///
/// Pre-computes node degrees from the graph and provides a `degree_map` closure
/// compatible with [`PathRewardCalculator::calculate_reward`]. Uses global max
/// degree normalization so centrality scores are comparable across paths.
#[derive(Debug, Clone)]
pub struct PathCentralityCalculator {
    /// Maps each node ID to its total degree (in-degree + out-degree).
    degrees: HashMap<NodeId, u32>,
    /// Maximum degree across all nodes (for normalization).
    max_degree: u32,
}

impl PathCentralityCalculator {
    /// Builds a centrality calculator from a graph backend.
    ///
    /// Loads all edges and computes the degree (in + out) for every node that
    /// appears as a source or target.
    pub fn from_graph(graph: &dyn GraphBackend) -> anyhow::Result<Self> {
        let edges = graph.get_all_edges()?;
        Ok(Self::from_edges(&edges))
    }

    /// Builds a centrality calculator from an edge list.
    ///
    /// Each edge `(source, target, _weight)` increments both the source and
    /// target node degrees by 1 (undirected degree).
    pub fn from_edges(edges: &[(NodeId, NodeId, f32)]) -> Self {
        let mut degrees: HashMap<NodeId, u32> = HashMap::new();
        for &(src, tgt, _) in edges {
            *degrees.entry(src).or_insert(0) += 1;
            *degrees.entry(tgt).or_insert(0) += 1;
        }
        let max_degree = degrees.values().copied().max().unwrap_or(1).max(1);
        Self {
            degrees,
            max_degree,
        }
    }

    /// Returns the raw degree for a node, or 0 if the node has no edges.
    pub fn degree(&self, node_id: NodeId) -> u32 {
        self.degrees.get(&node_id).copied().unwrap_or(0)
    }

    /// Returns the normalized degree centrality for a node (0.0 to 1.0).
    ///
    /// Normalization is against the global maximum degree, so hub nodes
    /// score close to 1.0 and leaf nodes score close to 0.0.
    pub fn normalized_centrality(&self, node_id: NodeId) -> f32 {
        self.degree(node_id) as f32 / self.max_degree as f32
    }

    /// Returns the maximum degree across all nodes.
    pub fn max_degree(&self) -> u32 {
        self.max_degree
    }

    /// Returns the total number of nodes with at least one edge.
    pub fn node_count(&self) -> usize {
        self.degrees.len()
    }

    /// Computes the average centrality of intermediate nodes in a path.
    ///
    /// Excludes the source and target (endpoints) and returns the mean
    /// normalized degree of the remaining nodes. Returns 0.5 for paths
    /// with no intermediate nodes (2 or fewer nodes).
    pub fn path_centrality(&self, path: &PathWithEdges) -> f32 {
        if path.nodes.len() <= 2 {
            return 0.5;
        }

        let intermediates = &path.nodes[1..path.nodes.len() - 1];
        let sum: f32 = intermediates
            .iter()
            .map(|&n| self.normalized_centrality(n))
            .sum();
        sum / intermediates.len() as f32
    }

    /// Scores a path using the given reward calculator with graph-aware centrality.
    ///
    /// This is a convenience method that provides the centrality-backed degree
    /// map to `PathRewardCalculator::calculate_reward`.
    pub fn score_path(&self, calc: &PathRewardCalculator, path: &PathWithEdges) -> RewardScore {
        calc.calculate_reward(path, &|node_id| self.degree(node_id))
    }

    /// Scores multiple paths and returns results sorted by total reward (descending).
    pub fn score_paths(
        &self,
        calc: &PathRewardCalculator,
        paths: &[PathWithEdges],
    ) -> Vec<(usize, RewardScore)> {
        let mut scored: Vec<(usize, RewardScore)> = paths
            .iter()
            .enumerate()
            .map(|(i, path)| (i, self.score_path(calc, path)))
            .collect();
        scored.sort_by(|a, b| {
            b.1.total
                .partial_cmp(&a.1.total)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        scored
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::EdgeType;

    fn make_star_edges() -> Vec<(NodeId, NodeId, f32)> {
        // Star graph: node 1 is the hub connected to 2,3,4,5
        vec![(1, 2, 1.0), (1, 3, 1.0), (1, 4, 1.0), (1, 5, 1.0)]
    }

    fn make_chain_edges() -> Vec<(NodeId, NodeId, f32)> {
        // Chain: 1-2-3-4-5
        vec![(1, 2, 1.0), (2, 3, 1.0), (3, 4, 1.0), (4, 5, 1.0)]
    }

    #[test]
    fn test_star_graph_hub_has_max_degree() {
        let calc = PathCentralityCalculator::from_edges(&make_star_edges());
        assert_eq!(calc.degree(1), 4); // hub
        assert_eq!(calc.degree(2), 1); // leaf
        assert_eq!(calc.max_degree(), 4);
    }

    #[test]
    fn test_normalized_centrality_hub_is_one() {
        let calc = PathCentralityCalculator::from_edges(&make_star_edges());
        assert!((calc.normalized_centrality(1) - 1.0).abs() < 0.001);
        assert!((calc.normalized_centrality(2) - 0.25).abs() < 0.001);
    }

    #[test]
    fn test_unknown_node_degree_is_zero() {
        let calc = PathCentralityCalculator::from_edges(&make_star_edges());
        assert_eq!(calc.degree(999), 0);
        assert!((calc.normalized_centrality(999) - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_chain_graph_interior_nodes_higher_degree() {
        let calc = PathCentralityCalculator::from_edges(&make_chain_edges());
        // Interior nodes 2,3,4 have degree 2; endpoints 1,5 have degree 1
        assert_eq!(calc.degree(1), 1);
        assert_eq!(calc.degree(2), 2);
        assert_eq!(calc.degree(3), 2);
        assert_eq!(calc.degree(4), 2);
        assert_eq!(calc.degree(5), 1);
        assert_eq!(calc.max_degree(), 2);
    }

    #[test]
    fn test_path_centrality_through_hub() {
        // Star + extra edge: 2-1-3 where 1 is a hub (degree 4)
        let calc = PathCentralityCalculator::from_edges(&make_star_edges());
        let path = PathWithEdges {
            nodes: vec![2, 1, 3],
            edge_types: vec![EdgeType::RelatesTo, EdgeType::RelatesTo],
        };
        let centrality = calc.path_centrality(&path);
        // Intermediate node is 1 (degree 4, max 4) → normalized = 1.0
        assert!((centrality - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_path_centrality_through_leaf() {
        // Chain 1-2-3 where degree(2)=2, max=2
        let calc = PathCentralityCalculator::from_edges(&make_chain_edges());
        let path = PathWithEdges {
            nodes: vec![1, 2, 3],
            edge_types: vec![EdgeType::RelatesTo, EdgeType::RelatesTo],
        };
        let centrality = calc.path_centrality(&path);
        // Intermediate node is 2 (degree 2, max 2) → normalized = 1.0
        assert!((centrality - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_path_centrality_no_intermediates() {
        let calc = PathCentralityCalculator::from_edges(&make_star_edges());
        let path = PathWithEdges {
            nodes: vec![1, 2],
            edge_types: vec![EdgeType::RelatesTo],
        };
        // No intermediate nodes → default 0.5
        assert!((calc.path_centrality(&path) - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_score_path_integrates_with_reward_calculator() {
        let cent = PathCentralityCalculator::from_edges(&make_star_edges());
        let calc = PathRewardCalculator::new(0.8, 0.0, 1.0); // Only centrality

        let path_via_hub = PathWithEdges {
            nodes: vec![2, 1, 3], // via hub node 1
            edge_types: vec![EdgeType::RelatesTo, EdgeType::RelatesTo],
        };
        let score = cent.score_path(&calc, &path_via_hub);
        // Hub (node 1) has degree 4, max 4, so centrality=1.0
        // With centrality_weight=1.0, total should be high
        assert!(score.total > 0.5);
    }

    #[test]
    fn test_hub_path_centrality_higher_than_leaf_path() {
        // Graph: 1 is hub (degree 6), 10 is leaf (degree 2)
        let edges = vec![
            (1, 2, 1.0),
            (1, 3, 1.0),
            (1, 4, 1.0),
            (1, 5, 1.0),
            (1, 6, 1.0),
            (1, 7, 1.0),
            (10, 11, 1.0),
            (2, 10, 1.0),
            (7, 11, 1.0),
        ];
        let cent = PathCentralityCalculator::from_edges(&edges);

        // Path through hub: 2 -> 1 -> 3 (intermediate=1, degree=6)
        let hub_path = PathWithEdges {
            nodes: vec![2, 1, 3],
            edge_types: vec![EdgeType::RelatesTo, EdgeType::RelatesTo],
        };
        // Path through leaf: 2 -> 10 -> 11 (intermediate=10, degree=2)
        let leaf_path = PathWithEdges {
            nodes: vec![2, 10, 11],
            edge_types: vec![EdgeType::RelatesTo, EdgeType::RelatesTo],
        };

        // Global normalization: hub=6/6=1.0, leaf=2/6=0.33
        let hub_cent = cent.path_centrality(&hub_path);
        let leaf_cent = cent.path_centrality(&leaf_path);
        assert!(
            hub_cent > leaf_cent,
            "Hub path centrality ({}) should exceed leaf path centrality ({})",
            hub_cent,
            leaf_cent
        );
        assert!((hub_cent - 1.0).abs() < 0.001);
        assert!((leaf_cent - 2.0 / 6.0).abs() < 0.01);
    }

    #[test]
    fn test_score_paths_sorted_by_total() {
        let edges = vec![
            (1, 2, 1.0),
            (1, 3, 1.0),
            (1, 4, 1.0),
            (1, 5, 1.0),
            (10, 11, 1.0),
        ];
        let cent = PathCentralityCalculator::from_edges(&edges);
        let calc = PathRewardCalculator::new(1.0, 0.0, 1.0);

        let paths = vec![
            PathWithEdges {
                nodes: vec![10, 11], // no intermediates
                edge_types: vec![EdgeType::RelatesTo],
            },
            PathWithEdges {
                nodes: vec![2, 1, 3], // hub intermediate
                edge_types: vec![EdgeType::RelatesTo, EdgeType::RelatesTo],
            },
        ];

        let scored = cent.score_paths(&calc, &paths);
        // Hub path (index 1) should be ranked first
        assert!(scored[0].1.total >= scored[1].1.total);
    }

    #[test]
    fn test_empty_graph_centrality() {
        let calc = PathCentralityCalculator::from_edges(&[]);
        assert_eq!(calc.node_count(), 0);
        assert_eq!(calc.max_degree(), 1); // Clamped to 1 to avoid division by zero
        assert_eq!(calc.degree(1), 0);
    }

    #[test]
    fn test_node_count() {
        let calc = PathCentralityCalculator::from_edges(&make_star_edges());
        assert_eq!(calc.node_count(), 5); // nodes 1,2,3,4,5
    }
}
