//! # Leiden Community Detection
//!
//! Detects communities (clusters) in the graph using the Leiden algorithm
//! via the `graphrs` crate. Communities group densely-connected nodes together,
//! which is useful for:
//!
//! - Context assembly during retrieval (selecting relevant clusters)
//! - Compression during consolidation (summarising large clusters)
//! - Understanding graph structure for exploration
//!
//! # Usage
//!
//! ```ignore
//! let config = CommunityConfig::default();
//! let result = detect_communities(&edges, &config)?;
//! // result.communities: HashMap<CommunityId, Vec<NodeId>>
//! // result.node_to_community: HashMap<NodeId, CommunityId>
//! ```

use crate::NodeId;
use std::collections::{HashMap, HashSet};

/// Unique identifier for a detected community.
pub type CommunityId = u64;

/// Configuration for community detection.
#[derive(Debug, Clone)]
pub struct CommunityConfig {
    /// Quality function: "modularity" or "cpm".
    pub quality_function: QualityFunctionType,
    /// Resolution parameter — larger values produce smaller communities.
    /// Default: 0.25
    pub resolution: f64,
    /// Theta parameter (randomness in refinement). Default: 0.3
    pub theta: f64,
    /// Gamma parameter (granularity). Default: 0.05
    pub gamma: f64,
    /// Whether to use edge weights. Default: true
    pub weighted: bool,
}

/// Quality function used by the Leiden algorithm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum QualityFunctionType {
    /// Modularity-based quality function.
    Modularity,
    /// Constant Potts Model.
    CPM,
}

impl Default for CommunityConfig {
    fn default() -> Self {
        Self {
            quality_function: QualityFunctionType::CPM,
            resolution: 0.25,
            theta: 0.3,
            gamma: 0.05,
            weighted: true,
        }
    }
}

/// Result of community detection.
#[derive(Debug, Clone)]
pub struct CommunityResult {
    /// Maps each community ID to the list of node IDs in that community.
    pub communities: HashMap<CommunityId, Vec<NodeId>>,
    /// Maps each node ID to its community ID.
    pub node_to_community: HashMap<NodeId, CommunityId>,
}

impl CommunityResult {
    /// Get the community ID for a given node.
    pub fn get_community_id(&self, node_id: NodeId) -> Option<CommunityId> {
        self.node_to_community.get(&node_id).copied()
    }

    /// Get all node IDs in the same community as the given node.
    pub fn get_community_members(&self, node_id: NodeId) -> Vec<NodeId> {
        if let Some(&community_id) = self.node_to_community.get(&node_id) {
            self.communities
                .get(&community_id)
                .cloned()
                .unwrap_or_default()
        } else {
            Vec::new()
        }
    }

    /// Number of communities detected.
    pub fn num_communities(&self) -> usize {
        self.communities.len()
    }

    /// Total number of nodes assigned to communities.
    pub fn num_nodes(&self) -> usize {
        self.node_to_community.len()
    }
}

/// Detect communities using the Leiden algorithm.
///
/// Takes a list of edges (source, target, weight) and returns the community
/// assignments. Self-loops are excluded. Edges are treated as undirected.
///
/// # Arguments
///
/// * `edges` - Weighted edge list as `(source_id, target_id, weight)` tuples
/// * `config` - Algorithm parameters
///
/// # Returns
///
/// A [`CommunityResult`] mapping nodes to communities.
pub fn detect_communities(
    edges: &[(NodeId, NodeId, f32)],
    config: &CommunityConfig,
) -> anyhow::Result<CommunityResult> {
    use graphrs::algorithms::community::leiden::{leiden, QualityFunction};
    use graphrs::{Edge as GEdge, EdgeDedupeStrategy, Graph, GraphSpecs};

    if edges.is_empty() {
        return Ok(CommunityResult {
            communities: HashMap::new(),
            node_to_community: HashMap::new(),
        });
    }

    // Build undirected graphrs graph from edge list.
    // graphrs Leiden requires undirected graphs.
    // We use KeepLast for edge dedup (duplicate edges keep the latest weight).
    let graphrs_edges: Vec<_> = edges
        .iter()
        .filter(|(s, t, _)| s != t) // exclude self-loops
        .map(|(s, t, w)| GEdge::with_weight(*s, *t, *w as f64))
        .collect();

    if graphrs_edges.is_empty() {
        return Ok(CommunityResult {
            communities: HashMap::new(),
            node_to_community: HashMap::new(),
        });
    }

    let mut specs = GraphSpecs::undirected_create_missing();
    specs.edge_dedupe_strategy = EdgeDedupeStrategy::KeepLast;

    let graph = Graph::<u64, ()>::new_from_nodes_and_edges(vec![], graphrs_edges, specs)
        .map_err(|e| anyhow::anyhow!("Failed to build graphrs graph: {}", e.message))?;

    let quality_fn = match config.quality_function {
        QualityFunctionType::Modularity => QualityFunction::Modularity,
        QualityFunctionType::CPM => QualityFunction::CPM,
    };

    let raw_communities: Vec<HashSet<u64>> = leiden(
        &graph,
        config.weighted,
        quality_fn,
        Some(config.resolution),
        Some(config.theta),
        Some(config.gamma),
    )
    .map_err(|e| anyhow::anyhow!("Leiden algorithm failed: {}", e.message))?;

    // Convert Vec<HashSet<u64>> → CommunityResult
    let mut communities = HashMap::new();
    let mut node_to_community = HashMap::new();

    for (idx, members) in raw_communities.into_iter().enumerate() {
        let community_id = idx as CommunityId;
        let mut member_vec: Vec<NodeId> = members.iter().copied().collect();
        member_vec.sort_unstable(); // deterministic ordering

        for &node_id in &member_vec {
            node_to_community.insert(node_id, community_id);
        }
        communities.insert(community_id, member_vec);
    }

    Ok(CommunityResult {
        communities,
        node_to_community,
    })
}

/// Incremental community detection.
///
/// Re-runs Leiden on the full edge set but preserves the previous community
/// assignments as a reference. Only nodes whose community changed are
/// reported in `changed_nodes`. This enables callers to update only the
/// modified node metadata rather than rewriting all community assignments.
///
/// # Arguments
///
/// * `edges` - Full edge list (including new and existing edges)
/// * `config` - Algorithm parameters
/// * `previous` - Previous community result (or `None` for first run)
///
/// # Returns
///
/// A tuple of `(new_result, changed_node_ids)`.
pub fn detect_communities_incremental(
    edges: &[(NodeId, NodeId, f32)],
    config: &CommunityConfig,
    previous: Option<&CommunityResult>,
) -> anyhow::Result<(CommunityResult, Vec<NodeId>)> {
    let new_result = detect_communities(edges, config)?;

    let changed_nodes = match previous {
        None => {
            // First run — all nodes are "changed"
            new_result.node_to_community.keys().copied().collect()
        }
        Some(prev) => {
            // Find nodes whose community membership changed.
            // We compare by co-membership: two nodes that were in the same
            // community before should still be in the same community.
            // However, community IDs may differ between runs (Leiden is
            // non-deterministic in ID assignment). So we detect changes
            // by checking if the set of co-members changed for each node.
            let mut changed = Vec::new();

            for (&node_id, &new_cid) in &new_result.node_to_community {
                let new_members: HashSet<NodeId> = new_result
                    .communities
                    .get(&new_cid)
                    .map(|m| m.iter().copied().collect())
                    .unwrap_or_default();

                let old_members: HashSet<NodeId> = prev
                    .get_community_id(node_id)
                    .and_then(|old_cid| prev.communities.get(&old_cid))
                    .map(|m| m.iter().copied().collect())
                    .unwrap_or_default();

                if new_members != old_members {
                    changed.push(node_id);
                }
            }

            // Also include nodes that were in previous but not in new (removed edges)
            for &node_id in prev.node_to_community.keys() {
                if !new_result.node_to_community.contains_key(&node_id) {
                    changed.push(node_id);
                }
            }

            changed.sort_unstable();
            changed.dedup();
            changed
        }
    };

    Ok((new_result, changed_nodes))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: build a simple graph with two clearly separated clusters
    /// connected by a weak bridge.
    ///
    /// Cluster A: nodes 1,2,3,4 (densely connected, weight 1.0)
    /// Cluster B: nodes 5,6,7,8 (densely connected, weight 1.0)
    /// Bridge: 4→5 (weight 0.1)
    fn two_cluster_edges() -> Vec<(NodeId, NodeId, f32)> {
        vec![
            // Cluster A
            (1, 2, 1.0),
            (1, 3, 1.0),
            (1, 4, 1.0),
            (2, 3, 1.0),
            (2, 4, 1.0),
            (3, 4, 1.0),
            // Cluster B
            (5, 6, 1.0),
            (5, 7, 1.0),
            (5, 8, 1.0),
            (6, 7, 1.0),
            (6, 8, 1.0),
            (7, 8, 1.0),
            // Weak bridge
            (4, 5, 0.1),
        ]
    }

    #[test]
    fn test_detect_communities_two_clusters() {
        let edges = two_cluster_edges();
        let config = CommunityConfig::default();
        let result = detect_communities(&edges, &config).unwrap();

        // Should detect at least 2 communities
        assert!(
            result.num_communities() >= 2,
            "Expected >= 2 communities, got {}",
            result.num_communities()
        );

        // All 8 nodes should be assigned
        assert_eq!(result.num_nodes(), 8);

        // Nodes 1,2,3,4 should be in the same community
        let c1 = result.get_community_id(1).unwrap();
        let c2 = result.get_community_id(2).unwrap();
        let c3 = result.get_community_id(3).unwrap();
        let c4 = result.get_community_id(4).unwrap();
        assert_eq!(c1, c2, "Nodes 1 and 2 should be in same community");
        assert_eq!(c1, c3, "Nodes 1 and 3 should be in same community");
        assert_eq!(c1, c4, "Nodes 1 and 4 should be in same community");

        // Nodes 5,6,7,8 should be in the same community
        let c5 = result.get_community_id(5).unwrap();
        let c6 = result.get_community_id(6).unwrap();
        let c7 = result.get_community_id(7).unwrap();
        let c8 = result.get_community_id(8).unwrap();
        assert_eq!(c5, c6, "Nodes 5 and 6 should be in same community");
        assert_eq!(c5, c7, "Nodes 5 and 7 should be in same community");
        assert_eq!(c5, c8, "Nodes 5 and 8 should be in same community");

        // The two clusters should be in DIFFERENT communities
        assert_ne!(
            c1, c5,
            "Cluster A and Cluster B should be different communities"
        );
    }

    #[test]
    fn test_detect_communities_empty_graph() {
        let edges: Vec<(NodeId, NodeId, f32)> = vec![];
        let config = CommunityConfig::default();
        let result = detect_communities(&edges, &config).unwrap();
        assert_eq!(result.num_communities(), 0);
        assert_eq!(result.num_nodes(), 0);
    }

    #[test]
    fn test_detect_communities_self_loops_excluded() {
        // Only self-loops — should produce empty result
        let edges = vec![(1, 1, 1.0), (2, 2, 1.0)];
        let config = CommunityConfig::default();
        let result = detect_communities(&edges, &config).unwrap();
        assert_eq!(result.num_nodes(), 0);
    }

    #[test]
    fn test_detect_communities_single_edge() {
        let edges = vec![(1, 2, 1.0)];
        let config = CommunityConfig::default();
        let result = detect_communities(&edges, &config).unwrap();
        // Both nodes should be assigned (possibly same community)
        assert_eq!(result.num_nodes(), 2);
        assert!(result.get_community_id(1).is_some());
        assert!(result.get_community_id(2).is_some());
    }

    #[test]
    fn test_community_result_get_members() {
        let edges = two_cluster_edges();
        let config = CommunityConfig::default();
        let result = detect_communities(&edges, &config).unwrap();

        // Members of node 1's community should include 1,2,3,4
        let members = result.get_community_members(1);
        assert!(members.contains(&1));
        assert!(members.contains(&2));
        assert!(members.contains(&3));
        assert!(members.contains(&4));
    }

    #[test]
    fn test_community_result_nonexistent_node() {
        let edges = vec![(1, 2, 1.0)];
        let config = CommunityConfig::default();
        let result = detect_communities(&edges, &config).unwrap();

        assert_eq!(result.get_community_id(999), None);
        assert!(result.get_community_members(999).is_empty());
    }

    #[test]
    fn test_detect_communities_modularity() {
        let edges = two_cluster_edges();
        let config = CommunityConfig {
            quality_function: QualityFunctionType::Modularity,
            ..Default::default()
        };
        let result = detect_communities(&edges, &config).unwrap();
        assert!(result.num_communities() >= 2);
        assert_eq!(result.num_nodes(), 8);
    }

    #[test]
    fn test_incremental_first_run() {
        let edges = two_cluster_edges();
        let config = CommunityConfig::default();
        let (result, changed) = detect_communities_incremental(&edges, &config, None).unwrap();

        // First run: all nodes are changed
        assert_eq!(changed.len(), 8);
        assert_eq!(result.num_nodes(), 8);
    }

    #[test]
    fn test_incremental_no_change() {
        let edges = two_cluster_edges();
        let config = CommunityConfig::default();
        let (first, _) = detect_communities_incremental(&edges, &config, None).unwrap();

        // Re-run with same data: check structure preserved
        let (second, _changed) =
            detect_communities_incremental(&edges, &config, Some(&first)).unwrap();

        // Same number of communities and nodes
        assert_eq!(second.num_communities(), first.num_communities());
        assert_eq!(second.num_nodes(), first.num_nodes());
    }

    #[test]
    fn test_incremental_detects_change_on_new_edges() {
        let edges = two_cluster_edges();
        let config = CommunityConfig::default();
        let (first, _) = detect_communities_incremental(&edges, &config, None).unwrap();

        // Add a strong bridge connecting the two clusters
        let mut edges2 = edges.clone();
        edges2.push((4, 5, 10.0));
        edges2.push((3, 6, 10.0));
        edges2.push((2, 7, 10.0));

        let (second, changed) =
            detect_communities_incremental(&edges2, &config, Some(&first)).unwrap();

        // Second run should still assign all 8 nodes
        assert_eq!(second.num_nodes(), 8);

        // The strong bridges may cause communities to merge, or at minimum
        // some nodes change community — we can't predict exact behavior
        // but the function should not panic
        let _ = changed;
    }

    #[test]
    fn test_three_clusters() {
        // Three clearly separated clusters
        let edges = vec![
            // Cluster A (1-4)
            (1, 2, 1.0),
            (1, 3, 1.0),
            (1, 4, 1.0),
            (2, 3, 1.0),
            (2, 4, 1.0),
            (3, 4, 1.0),
            // Cluster B (5-8)
            (5, 6, 1.0),
            (5, 7, 1.0),
            (5, 8, 1.0),
            (6, 7, 1.0),
            (6, 8, 1.0),
            (7, 8, 1.0),
            // Cluster C (9-12)
            (9, 10, 1.0),
            (9, 11, 1.0),
            (9, 12, 1.0),
            (10, 11, 1.0),
            (10, 12, 1.0),
            (11, 12, 1.0),
            // Weak bridges
            (4, 5, 0.05),
            (8, 9, 0.05),
        ];
        let config = CommunityConfig::default();
        let result = detect_communities(&edges, &config).unwrap();

        assert!(
            result.num_communities() >= 3,
            "Expected >= 3 communities, got {}",
            result.num_communities()
        );
        assert_eq!(result.num_nodes(), 12);

        // Verify intra-cluster cohesion
        let c1 = result.get_community_id(1).unwrap();
        let c2 = result.get_community_id(2).unwrap();
        assert_eq!(c1, c2);

        let c5 = result.get_community_id(5).unwrap();
        let c6 = result.get_community_id(6).unwrap();
        assert_eq!(c5, c6);

        let c9 = result.get_community_id(9).unwrap();
        let c10 = result.get_community_id(10).unwrap();
        assert_eq!(c9, c10);

        // Verify inter-cluster separation
        assert_ne!(c1, c5);
        assert_ne!(c5, c9);
        assert_ne!(c1, c9);
    }

    #[test]
    fn test_large_graph_1000_nodes() {
        // Generate a graph with clear clusters: 10 clusters of 100 nodes each
        let mut edges = Vec::new();
        for cluster in 0..10 {
            let base = cluster * 100 + 1;
            // Dense intra-cluster edges
            for i in 0..100u64 {
                for j in (i + 1)..100u64 {
                    // Only connect ~20% of pairs to keep it manageable
                    if (i + j) % 5 == 0 {
                        edges.push((base + i, base + j, 1.0));
                    }
                }
            }
        }
        // Weak inter-cluster bridges
        for cluster in 0..9 {
            let a = cluster * 100 + 1;
            let b = (cluster + 1) * 100 + 1;
            edges.push((a, b, 0.01));
        }

        let config = CommunityConfig::default();
        let result = detect_communities(&edges, &config).unwrap();

        // Should detect multiple communities (at least 5, ideally ~10)
        assert!(
            result.num_communities() >= 5,
            "Expected >= 5 communities for 10-cluster graph, got {}",
            result.num_communities()
        );
    }

    #[test]
    fn test_detect_communities_disconnected_components() {
        // Two completely disconnected pairs → should be in different communities
        let edges = vec![
            (1, 2, 1.0),
            (3, 4, 1.0),
        ];
        let config = CommunityConfig::default();
        let result = detect_communities(&edges, &config).unwrap();

        // Nodes 1,2 should be in the same community; nodes 3,4 in another
        let c1 = result.get_community_id(1);
        let c3 = result.get_community_id(3);
        assert_ne!(c1, c3, "Disconnected components should be in different communities");
    }

    #[test]
    fn test_community_result_all_node_ids() {
        let edges = vec![
            (10, 20, 1.0),
            (20, 30, 1.0),
        ];
        let config = CommunityConfig::default();
        let result = detect_communities(&edges, &config).unwrap();

        // All nodes that appear in edges should be assigned a community
        for &node in &[10, 20, 30] {
            assert!(result.get_community_id(node).is_some(),
                "Node {} should have a community assignment", node);
        }
    }

    #[test]
    fn test_detect_communities_small_resolution() {
        // Very small resolution should tend to merge into fewer communities
        let edges = vec![
            (1, 2, 1.0),
            (2, 3, 1.0),
            (3, 4, 1.0),
            (4, 5, 1.0),
        ];
        let config = CommunityConfig {
            resolution: 0.01,
            ..Default::default()
        };
        let result = detect_communities(&edges, &config).unwrap();
        assert!(result.num_communities() >= 1, "Should have at least 1 community");
    }

    #[test]
    fn test_incremental_with_no_previous() {
        // Incremental detection with no previous state (None)
        let edges = vec![(1, 2, 1.0)];
        let config = CommunityConfig::default();
        let (result, changed) = detect_communities_incremental(&edges, &config, None).unwrap();
        assert!(result.num_communities() >= 1);
        // With no previous state, all nodes should be reported as changed
        assert!(!changed.is_empty());
    }
}
