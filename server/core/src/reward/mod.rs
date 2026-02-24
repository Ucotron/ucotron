//! Knowledge graph reward models for retrieval quality scoring.
//!
//! Implements path-derived rewards (arXiv:2601.15160) with hop decay,
//! predicate coherence, and node centrality weighting.
//!
//! # Path Finding
//!
//! The [`path_finder`] module provides [`find_paths`] to enumerate all paths
//! between two nodes up to a configurable hop limit, returning
//! [`PathWithEdges`] for coherence scoring via [`PathRewardCalculator`].
//!
//! # Centrality
//!
//! The [`centrality`] module provides [`PathCentralityCalculator`] for
//! degree-based centrality scoring of paths. It computes node degrees from
//! graph edges and normalizes against the global maximum, enabling
//! graph-aware reward weighting where hub nodes boost path scores.

mod centrality;
mod path_finder;
mod path_reward;

pub use centrality::PathCentralityCalculator;
pub use path_finder::{find_paths, PathFinderConfig};
pub use path_reward::{
    predicate_similarity, PathRewardCalculator, PathWithEdges, PredicateCoherenceConfig,
    RewardScore,
};
