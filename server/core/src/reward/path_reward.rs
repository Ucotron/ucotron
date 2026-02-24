//! Path-derived reward calculator for knowledge graph retrieval.
//!
//! Scores graph paths based on:
//! - **Hop decay**: Shorter paths receive higher rewards (exponential decay per hop).
//! - **Predicate coherence**: Paths with consistent edge types score higher.
//! - **Node centrality**: Paths through high-degree nodes are weighted higher.
//!
//! Reference: arXiv:2601.15160 (Knowledge Graph Reward Models for LLM Alignment)

use crate::types::{EdgeType, NodeId};
use std::collections::HashMap;

/// A graph path annotated with edge types for coherence scoring.
#[derive(Debug, Clone)]
pub struct PathWithEdges {
    /// Ordered list of node IDs from source to target.
    pub nodes: Vec<NodeId>,
    /// Edge types along the path (length = nodes.len() - 1).
    pub edge_types: Vec<EdgeType>,
}

impl PathWithEdges {
    /// Returns the number of hops (edges) in this path.
    pub fn hop_count(&self) -> usize {
        self.edge_types.len()
    }
}

/// Result of a path reward calculation.
#[derive(Debug, Clone)]
pub struct RewardScore {
    /// Combined reward score (0.0 to 1.0).
    pub total: f32,
    /// Hop decay component.
    pub hop_decay_score: f32,
    /// Predicate coherence component.
    pub coherence_score: f32,
    /// Node centrality component.
    pub centrality_score: f32,
}

/// Configurable predicate similarity mapping for coherence scoring.
///
/// Groups edge types into semantic categories and defines similarity scores
/// for same-type, related (same-group), and unrelated (cross-group) pairs.
///
/// The default configuration uses 4 groups:
/// - Structural: `RelatesTo`, `HasProperty`
/// - Causal/Temporal: `CausedBy`, `Supersedes`, `NextEpisode`
/// - Conflict: `ConflictsWith`
/// - Event roles: `Actor`, `Object`, `Location`, `Companion`
#[derive(Debug, Clone)]
pub struct PredicateCoherenceConfig {
    /// Maps each edge type to its semantic group ID.
    pub edge_groups: HashMap<EdgeType, u8>,
    /// Score for identical predicates (default 1.0).
    pub same_score: f32,
    /// Score for predicates in the same semantic group (default 0.7).
    pub related_score: f32,
    /// Score for predicates in different groups (default 0.3).
    pub unrelated_score: f32,
}

impl Default for PredicateCoherenceConfig {
    fn default() -> Self {
        let mut edge_groups = HashMap::new();
        // Group 0: Structural relationships
        edge_groups.insert(EdgeType::RelatesTo, 0);
        edge_groups.insert(EdgeType::HasProperty, 0);
        // Group 1: Causal/temporal relationships
        edge_groups.insert(EdgeType::CausedBy, 1);
        edge_groups.insert(EdgeType::Supersedes, 1);
        edge_groups.insert(EdgeType::NextEpisode, 1);
        // Group 2: Conflict relationships
        edge_groups.insert(EdgeType::ConflictsWith, 2);
        // Group 3: Event role relationships
        edge_groups.insert(EdgeType::Actor, 3);
        edge_groups.insert(EdgeType::Object, 3);
        edge_groups.insert(EdgeType::Location, 3);
        edge_groups.insert(EdgeType::Companion, 3);

        Self {
            edge_groups,
            same_score: 1.0,
            related_score: 0.7,
            unrelated_score: 0.3,
        }
    }
}

impl PredicateCoherenceConfig {
    /// Returns the similarity score between two edge types.
    pub fn similarity(&self, a: EdgeType, b: EdgeType) -> f32 {
        if a == b {
            return self.same_score;
        }
        let group_a = self.edge_groups.get(&a);
        let group_b = self.edge_groups.get(&b);
        match (group_a, group_b) {
            (Some(ga), Some(gb)) if ga == gb => self.related_score,
            _ => self.unrelated_score,
        }
    }
}

/// Calculates rewards for graph paths to score retrieval quality.
///
/// The final reward is a weighted combination:
/// `reward = hop_decay * coherence_weight * coherence + centrality_weight * centrality`
///
/// where hop_decay provides the base score that decays exponentially with path length.
#[derive(Debug, Clone)]
pub struct PathRewardCalculator {
    /// Decay factor per hop (0.0 to 1.0). A value of 0.8 means each hop
    /// reduces the reward to 80% of the previous hop's value.
    pub hop_decay: f32,
    /// Weight for predicate coherence in the final score (0.0 to 1.0).
    pub coherence_weight: f32,
    /// Weight for node centrality in the final score (0.0 to 1.0).
    pub centrality_weight: f32,
    /// Configurable predicate similarity mapping for coherence scoring.
    pub coherence_config: PredicateCoherenceConfig,
}

impl Default for PathRewardCalculator {
    fn default() -> Self {
        Self {
            hop_decay: 0.8,
            coherence_weight: 0.4,
            centrality_weight: 0.2,
            coherence_config: PredicateCoherenceConfig::default(),
        }
    }
}

impl PathRewardCalculator {
    /// Creates a new calculator with the given parameters and default coherence config.
    pub fn new(hop_decay: f32, coherence_weight: f32, centrality_weight: f32) -> Self {
        Self {
            hop_decay,
            coherence_weight,
            centrality_weight,
            coherence_config: PredicateCoherenceConfig::default(),
        }
    }

    /// Creates a new calculator with custom coherence configuration.
    pub fn with_coherence_config(
        hop_decay: f32,
        coherence_weight: f32,
        centrality_weight: f32,
        coherence_config: PredicateCoherenceConfig,
    ) -> Self {
        Self {
            hop_decay,
            coherence_weight,
            centrality_weight,
            coherence_config,
        }
    }

    /// Calculates the reward for a single path.
    ///
    /// - `path`: The graph path with edge type annotations.
    /// - `degree_map`: A function that returns the degree (number of edges) for a node.
    ///   Used for centrality calculation. If not available, pass `|_| 1`.
    ///
    /// Returns a [`RewardScore`] with individual components and combined total.
    pub fn calculate_reward(
        &self,
        path: &PathWithEdges,
        degree_map: &dyn Fn(NodeId) -> u32,
    ) -> RewardScore {
        if path.nodes.is_empty() {
            return RewardScore {
                total: 0.0,
                hop_decay_score: 0.0,
                coherence_score: 0.0,
                centrality_score: 0.0,
            };
        }

        let hop_decay_score = self.compute_hop_decay(path.hop_count());
        let coherence_score = self.compute_coherence(&path.edge_types);
        let centrality_score = self.compute_centrality(&path.nodes, degree_map);

        // Base score from hop decay, modulated by coherence and centrality
        let base_weight = 1.0 - self.coherence_weight - self.centrality_weight;
        let total = (base_weight * hop_decay_score
            + self.coherence_weight * coherence_score
            + self.centrality_weight * centrality_score)
            .clamp(0.0, 1.0);

        RewardScore {
            total,
            hop_decay_score,
            coherence_score,
            centrality_score,
        }
    }

    /// Computes exponential hop decay: hop_decay ^ num_hops.
    fn compute_hop_decay(&self, hops: usize) -> f32 {
        self.hop_decay.powi(hops as i32)
    }

    /// Computes predicate coherence score using the configurable similarity mapping.
    ///
    /// Matching consecutive predicates = same_score, related = related_score,
    /// unrelated = unrelated_score. Returns average coherence across all consecutive
    /// edge pairs.
    fn compute_coherence(&self, edge_types: &[EdgeType]) -> f32 {
        if edge_types.len() <= 1 {
            return 1.0; // Single edge or no edges = perfectly coherent
        }

        let mut total = 0.0f32;
        let pairs = edge_types.len() - 1;

        for i in 0..pairs {
            total += self
                .coherence_config
                .similarity(edge_types[i], edge_types[i + 1]);
        }

        total / pairs as f32
    }

    /// Computes centrality score based on node degrees along the path.
    ///
    /// Uses normalized degree centrality of intermediate nodes (excludes endpoints).
    fn compute_centrality(&self, nodes: &[NodeId], degree_map: &dyn Fn(NodeId) -> u32) -> f32 {
        if nodes.len() <= 2 {
            // No intermediate nodes — centrality is neutral
            return 0.5;
        }

        let intermediates = &nodes[1..nodes.len() - 1];
        let degrees: Vec<u32> = intermediates.iter().map(|&n| degree_map(n)).collect();
        let max_degree = degrees.iter().copied().max().unwrap_or(1).max(1);

        // Average normalized degree of intermediate nodes
        let avg: f32 = degrees
            .iter()
            .map(|&d| d as f32 / max_degree as f32)
            .sum::<f32>()
            / intermediates.len() as f32;

        avg
    }
}

/// Returns a similarity score for two edge types using the default predicate grouping.
///
/// - Same type: 1.0
/// - Related types (same category): 0.7
/// - Unrelated types: 0.3
///
/// For configurable similarity mapping, use [`PredicateCoherenceConfig::similarity`].
pub fn predicate_similarity(a: EdgeType, b: EdgeType) -> f32 {
    PredicateCoherenceConfig::default().similarity(a, b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_calculator() {
        let calc = PathRewardCalculator::default();
        assert_eq!(calc.hop_decay, 0.8);
        assert_eq!(calc.coherence_weight, 0.4);
        assert_eq!(calc.centrality_weight, 0.2);
    }

    #[test]
    fn test_empty_path_returns_zero() {
        let calc = PathRewardCalculator::default();
        let path = PathWithEdges {
            nodes: vec![],
            edge_types: vec![],
        };
        let score = calc.calculate_reward(&path, &|_| 1);
        assert_eq!(score.total, 0.0);
    }

    #[test]
    fn test_single_hop_path() {
        let calc = PathRewardCalculator::default();
        let path = PathWithEdges {
            nodes: vec![1, 2],
            edge_types: vec![EdgeType::RelatesTo],
        };
        let score = calc.calculate_reward(&path, &|_| 5);
        assert!(score.hop_decay_score > 0.0);
        assert_eq!(score.coherence_score, 1.0); // Single edge = perfect coherence
        assert!(score.total > 0.0);
    }

    #[test]
    fn test_hop_decay_reduces_with_distance() {
        let calc = PathRewardCalculator::new(0.8, 0.0, 0.0);

        let short_path = PathWithEdges {
            nodes: vec![1, 2],
            edge_types: vec![EdgeType::RelatesTo],
        };
        let long_path = PathWithEdges {
            nodes: vec![1, 2, 3, 4, 5],
            edge_types: vec![
                EdgeType::RelatesTo,
                EdgeType::RelatesTo,
                EdgeType::RelatesTo,
                EdgeType::RelatesTo,
            ],
        };

        let short_score = calc.calculate_reward(&short_path, &|_| 1);
        let long_score = calc.calculate_reward(&long_path, &|_| 1);

        assert!(short_score.total > long_score.total);
    }

    #[test]
    fn test_coherent_path_scores_higher() {
        let calc = PathRewardCalculator::new(1.0, 1.0, 0.0); // Only coherence matters

        let coherent = PathWithEdges {
            nodes: vec![1, 2, 3],
            edge_types: vec![EdgeType::RelatesTo, EdgeType::RelatesTo],
        };
        let incoherent = PathWithEdges {
            nodes: vec![1, 2, 3],
            edge_types: vec![EdgeType::RelatesTo, EdgeType::ConflictsWith],
        };

        let coherent_score = calc.calculate_reward(&coherent, &|_| 1);
        let incoherent_score = calc.calculate_reward(&incoherent, &|_| 1);

        assert!(coherent_score.coherence_score > incoherent_score.coherence_score);
    }

    #[test]
    fn test_predicate_similarity_same_type() {
        assert_eq!(predicate_similarity(EdgeType::Actor, EdgeType::Actor), 1.0);
    }

    #[test]
    fn test_predicate_similarity_related_type() {
        assert_eq!(predicate_similarity(EdgeType::Actor, EdgeType::Object), 0.7);
    }

    #[test]
    fn test_predicate_similarity_unrelated_type() {
        assert_eq!(
            predicate_similarity(EdgeType::RelatesTo, EdgeType::ConflictsWith),
            0.3
        );
    }

    #[test]
    fn test_centrality_with_high_degree_nodes() {
        let calc = PathRewardCalculator::new(1.0, 0.0, 1.0); // Only centrality matters

        let path = PathWithEdges {
            nodes: vec![1, 2, 3, 4],
            edge_types: vec![
                EdgeType::RelatesTo,
                EdgeType::RelatesTo,
                EdgeType::RelatesTo,
            ],
        };

        // Intermediate nodes (2, 3) have high degree
        let high_centrality =
            calc.calculate_reward(&path, &|n| if n == 2 || n == 3 { 100 } else { 1 });
        // Intermediate nodes have low degree
        let low_centrality = calc.calculate_reward(&path, &|_| 1);

        assert!(high_centrality.centrality_score >= low_centrality.centrality_score);
    }

    #[test]
    fn test_path_with_edges_hop_count() {
        let path = PathWithEdges {
            nodes: vec![1, 2, 3],
            edge_types: vec![EdgeType::RelatesTo, EdgeType::CausedBy],
        };
        assert_eq!(path.hop_count(), 2);
    }

    // --- Configurable predicate coherence tests (US-27.3) ---

    #[test]
    fn test_default_coherence_config_groups() {
        let config = PredicateCoherenceConfig::default();
        // Same type → 1.0
        assert_eq!(config.similarity(EdgeType::Actor, EdgeType::Actor), 1.0);
        // Related (same group) → 0.7
        assert_eq!(config.similarity(EdgeType::Actor, EdgeType::Object), 0.7);
        assert_eq!(
            config.similarity(EdgeType::CausedBy, EdgeType::Supersedes),
            0.7
        );
        assert_eq!(
            config.similarity(EdgeType::RelatesTo, EdgeType::HasProperty),
            0.7
        );
        // Unrelated (different groups) → 0.3
        assert_eq!(
            config.similarity(EdgeType::RelatesTo, EdgeType::ConflictsWith),
            0.3
        );
        assert_eq!(config.similarity(EdgeType::Actor, EdgeType::CausedBy), 0.3);
    }

    #[test]
    fn test_custom_coherence_config_scores() {
        let config = PredicateCoherenceConfig {
            same_score: 1.0,
            related_score: 0.9,
            unrelated_score: 0.1,
            ..Default::default()
        };

        assert_eq!(config.similarity(EdgeType::Actor, EdgeType::Actor), 1.0);
        assert_eq!(config.similarity(EdgeType::Actor, EdgeType::Object), 0.9);
        assert_eq!(config.similarity(EdgeType::Actor, EdgeType::CausedBy), 0.1);
    }

    #[test]
    fn test_custom_coherence_config_groups() {
        // Create config where CausedBy and Actor are in the same group
        let mut edge_groups = HashMap::new();
        edge_groups.insert(EdgeType::CausedBy, 0);
        edge_groups.insert(EdgeType::Actor, 0);
        edge_groups.insert(EdgeType::RelatesTo, 1);

        let config = PredicateCoherenceConfig {
            edge_groups,
            same_score: 1.0,
            related_score: 0.8,
            unrelated_score: 0.2,
        };

        // CausedBy and Actor are now related (same custom group)
        assert_eq!(config.similarity(EdgeType::CausedBy, EdgeType::Actor), 0.8);
        // RelatesTo is in a different group
        assert_eq!(
            config.similarity(EdgeType::RelatesTo, EdgeType::CausedBy),
            0.2
        );
    }

    #[test]
    fn test_coherence_config_unknown_edge_type() {
        // Config with only some edge types mapped
        let mut edge_groups = HashMap::new();
        edge_groups.insert(EdgeType::Actor, 0);

        let config = PredicateCoherenceConfig {
            edge_groups,
            same_score: 1.0,
            related_score: 0.7,
            unrelated_score: 0.3,
        };

        // Unmapped edge type defaults to unrelated
        assert_eq!(config.similarity(EdgeType::Actor, EdgeType::CausedBy), 0.3);
        // Same type still returns same_score even if unmapped
        assert_eq!(
            config.similarity(EdgeType::CausedBy, EdgeType::CausedBy),
            1.0
        );
    }

    #[test]
    fn test_calculator_with_custom_coherence_config() {
        let config = PredicateCoherenceConfig {
            same_score: 1.0,
            related_score: 0.5,   // Lower than default 0.7
            unrelated_score: 0.0, // Stricter than default 0.3
            ..Default::default()
        };

        let calc = PathRewardCalculator::with_coherence_config(1.0, 1.0, 0.0, config);

        // Path with related edges (Actor→Object, same group)
        let related_path = PathWithEdges {
            nodes: vec![1, 2, 3],
            edge_types: vec![EdgeType::Actor, EdgeType::Object],
        };
        let score = calc.calculate_reward(&related_path, &|_| 1);
        assert!((score.coherence_score - 0.5).abs() < 0.001);

        // Path with unrelated edges → 0.0 with strict config
        let unrelated_path = PathWithEdges {
            nodes: vec![1, 2, 3],
            edge_types: vec![EdgeType::Actor, EdgeType::CausedBy],
        };
        let score = calc.calculate_reward(&unrelated_path, &|_| 1);
        assert!((score.coherence_score - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_coherence_multi_edge_path() {
        let calc = PathRewardCalculator::new(1.0, 1.0, 0.0);

        // 4-edge path: same, related, unrelated → (1.0 + 0.7 + 0.3) / 3 = 0.667
        let path = PathWithEdges {
            nodes: vec![1, 2, 3, 4, 5],
            edge_types: vec![
                EdgeType::Actor,  // → Actor (same)
                EdgeType::Actor,  // → Object (related)
                EdgeType::Object, // → CausedBy (unrelated)
                EdgeType::CausedBy,
            ],
        };
        let score = calc.calculate_reward(&path, &|_| 1);
        // Pairs: (Actor,Actor)=1.0, (Actor,Object)=0.7, (Object,CausedBy)=0.3
        let expected = (1.0 + 0.7 + 0.3) / 3.0;
        assert!((score.coherence_score - expected).abs() < 0.001);
    }
}
