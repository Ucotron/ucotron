//! # Ucotron Core
//!
//! Core traits, types, and data generation for the Ucotron cognitive memory framework.
//!
//! This crate defines the shared abstractions that storage engine implementations
//! must conform to, as well as the data models for the tripartite memory system
//! (Episodic, Semantic, Procedural).
//!
//! # Phase 2 Backend Traits
//!
//! The [`backends`] module defines the pluggable [`VectorBackend`](backends::VectorBackend)
//! and [`GraphBackend`](backends::GraphBackend) traits that decouple vector search from
//! graph storage, enabling independent backend selection via configuration.

pub mod agent;
pub mod arena_traversal;
pub mod backends;
pub mod bench_eval;
pub mod community;
pub mod contradictions;
pub mod data_gen;
pub mod entity_resolution;
pub mod event_nodes;
pub mod hybrid;
pub mod jsonld_export;
pub mod locomo;
pub mod longmemeval;
pub mod mem0_adapter;
pub mod multimodal;
pub mod query;
pub mod retrieval;
pub mod reward;
pub mod types;
pub mod zep_adapter;

pub use agent::{Agent, AgentId, AgentShare, CloneFilter, CloneResult, MergeResult, SharePermission};
pub use backends::{
    BackendRegistry, ExternalGraphBackend, ExternalVectorBackend, GraphBackend, VectorBackend,
    VisualVectorBackend,
};
pub use community::{
    detect_communities, detect_communities_incremental, CommunityConfig, CommunityId,
    CommunityResult, QualityFunctionType,
};
pub use contradictions::{build_conflict_edges, detect_conflict, resolve_conflict};
pub use entity_resolution::{
    resolve_entities, structural_similarity, EntityCluster, EntityResolutionConfig,
    DEFAULT_SIMILARITY_THRESHOLD,
};
pub use arena_traversal::{ArenaQueryTraversal, BfsArena};
pub use hybrid::{arena_find_related, find_related, DEFAULT_HOP_DECAY};
pub use multimodal::{MultimodalNodeBuilder, MultimodalValidationError};
pub use query::{QueryBuilder, TraversalQuery, VectorQuery, PathQuery};
pub use retrieval::{MindsetDetector, MindsetKeyword, MindsetScorer, MindsetWeights};
pub use reward::{
    find_paths, predicate_similarity, PathCentralityCalculator, PathFinderConfig,
    PathRewardCalculator, PathWithEdges, PredicateCoherenceConfig, RewardScore,
};
pub use types::*;
