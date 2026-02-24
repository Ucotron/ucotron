//! Core data types for the Ucotron storage engines.
//!
//! Defines the fundamental types shared across all storage engine implementations:
//! nodes, edges, configuration, the `StorageEngine` trait, and the cognitive data model
//! (Fact, MindsetTag, ResolutionState) for Chain of Mindset conflict resolution.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Unique identifier for a node in the graph.
pub type NodeId = u64;

/// Unique identifier for a fact in the cognitive model.
pub type FactId = u64;

/// Flexible metadata value supporting common JSON-like types.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum Value {
    /// A string value.
    String(String),
    /// An integer value.
    Integer(i64),
    /// A floating-point value.
    Float(f64),
    /// A boolean value.
    Bool(bool),
}

/// Media type classification for multimodal memory nodes.
///
/// Determines which processing pipeline and embedding index a node uses:
/// - **Text**: Standard 384-dim MiniLM text embedding (default)
/// - **Audio**: Whisper transcription → text embedding + optional visual
/// - **Image**: CLIP 512-dim visual embedding + optional text description
/// - **VideoSegment**: Both visual (keyframe) and text (transcript) embeddings
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum MediaType {
    /// Text-only content (default for all existing nodes).
    #[default]
    Text,
    /// Audio content (speech, music, sound effects).
    Audio,
    /// Image content (photos, screenshots, diagrams).
    Image,
    /// Video segment with keyframes and optional transcript.
    VideoSegment,
}

/// Classification of nodes in the tripartite memory model.
///
/// Based on Tulving's memory taxonomy:
/// - **Entity**: Semantic memory — stable knowledge entities (people, places, things)
/// - **Event**: Episodic memory — time-bound experiences with temporal ordering
/// - **Fact**: Semantic memory — extracted knowledge claims with confidence scores
/// - **Skill**: Procedural memory — registered tools and capabilities
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum NodeType {
    /// Semantic memory: stable knowledge entities (60% of synthetic data).
    Entity,
    /// Episodic memory: time-bound experiences (25% of synthetic data).
    Event,
    /// Semantic memory: extracted knowledge claims (15% of synthetic data).
    Fact,
    /// Procedural memory: registered tools/skills.
    Skill,
}

/// Classification of edges (relationships) in the knowledge graph.
///
/// Supports both structural relationships and cognitive/semantic edge types
/// needed for the proto-hypergraph Event Node pattern.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum EdgeType {
    /// General semantic relationship between entities.
    RelatesTo,
    /// Causal relationship: source caused target.
    CausedBy,
    /// Conflict marker: source contradicts target.
    ConflictsWith,
    /// Temporal ordering in episodic memory.
    NextEpisode,
    /// Property relationship: source has property target.
    HasProperty,
    /// Temporal supersession: source replaces target.
    Supersedes,
    /// Event Node role: the acting entity.
    Actor,
    /// Event Node role: the object/patient of an action.
    Object,
    /// Event Node role: where the event occurred.
    Location,
    /// Event Node role: accompanying entity.
    Companion,
}

/// A node in the knowledge graph.
///
/// Represents a single entity, event, fact, or skill in the memory system.
/// Each node carries a 384-dimensional embedding vector compatible with
/// sentence-transformer models.
///
/// # Multimodal Support
///
/// Nodes can optionally carry multimodal metadata:
/// - `media_type`: Classification of the content modality (Text, Audio, Image, VideoSegment)
/// - `media_uri`: URI/path to the original media file
/// - `embedding_visual`: 512-dimensional CLIP embedding for visual content
/// - `timestamp_range`: Start/end timestamps for temporal segments (audio/video)
/// - `parent_video_id`: For VideoSegment nodes, the ID of the parent video node
///
/// All multimodal fields are `Option` with `#[serde(default)]` for backwards
/// compatibility — existing serialized nodes deserialize correctly with `None`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Node {
    /// Unique node identifier (sequential).
    pub id: NodeId,
    /// Text content of the node (50-200 chars in synthetic data).
    pub content: String,
    /// 384-dimensional normalized embedding vector.
    pub embedding: Vec<f32>,
    /// Extensible key-value metadata.
    pub metadata: HashMap<String, Value>,
    /// Classification in the tripartite memory model.
    pub node_type: NodeType,
    /// Unix timestamp (distributed over 1 year in synthetic data).
    pub timestamp: u64,

    // --- Multimodal fields (Phase 3.5) ---

    /// Media type classification. Defaults to `None` (treated as `Text`).
    #[serde(default)]
    pub media_type: Option<MediaType>,
    /// URI or path to the original media file (e.g., `file:///path/to/audio.wav`).
    #[serde(default)]
    pub media_uri: Option<String>,
    /// 512-dimensional CLIP visual embedding (for Image/VideoSegment nodes).
    #[serde(default)]
    pub embedding_visual: Option<Vec<f32>>,
    /// Temporal range as (start_ms, end_ms) for audio/video segments.
    #[serde(default)]
    pub timestamp_range: Option<(u64, u64)>,
    /// For VideoSegment nodes, the ID of the parent video node.
    #[serde(default)]
    pub parent_video_id: Option<NodeId>,
}

/// A directed edge in the knowledge graph.
///
/// Connects two nodes with a typed, weighted relationship. Edges carry
/// metadata for cognitive operations like conflict resolution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Edge {
    /// Source node identifier.
    pub source: NodeId,
    /// Target node identifier.
    pub target: NodeId,
    /// Semantic type of this relationship.
    pub edge_type: EdgeType,
    /// Relationship strength (0.1 to 1.0).
    pub weight: f32,
    /// Extensible key-value metadata.
    pub metadata: HashMap<String, Value>,
}

// ---------------------------------------------------------------------------
// Cognitive Data Model — Chain of Mindset (CoM)
// ---------------------------------------------------------------------------

/// Cognitive processing mode tag, based on Chain of Mindset (CoM).
///
/// Different tasks require different mental modes. Tagging facts with a mindset
/// allows the system to route conflict resolution and retrieval strategies
/// appropriately.
///
/// - **Convergent**: Grouping/resolution mode — seeks consensus among related facts.
/// - **Divergent**: Exploration/contradiction mode — surfaces alternative viewpoints.
/// - **Algorithmic**: Verification/rules mode — applies strict logical checks.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum MindsetTag {
    /// Grouping/resolution mode: seeks consensus among facts.
    Convergent,
    /// Exploration/contradiction mode: surfaces alternatives.
    Divergent,
    /// Verification/rules mode: strict logical checks.
    Algorithmic,
}

/// Resolution state of a [`Fact`] in the cognitive model.
///
/// Tracks whether a fact is currently accepted, in conflict, or has been
/// superseded by newer information. Old facts are **never deleted**; they
/// transition through states to maintain a full audit trail.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum ResolutionState {
    /// The fact is confirmed and currently valid.
    Accepted,
    /// A conflicting fact exists; not yet resolved.
    Contradiction,
    /// Replaced by a more recent or higher-confidence fact.
    Superseded,
    /// Cannot be resolved automatically; requires external intervention.
    Ambiguous,
}

/// A knowledge claim in the cognitive model.
///
/// Facts represent subject–predicate–object triples extracted from episodic
/// memory. Each fact carries a confidence score, a cognitive mindset tag,
/// and a resolution state for conflict tracking.
///
/// # Conflict Resolution
///
/// When two facts share the same `(subject, predicate)` but differ on `object`,
/// the system creates a [`EdgeType::ConflictsWith`] edge between them and
/// applies resolution rules (temporal, confidence, or ambiguity).
///
/// Invariant: old facts are never deleted. Superseded facts keep their data
/// and gain a [`EdgeType::Supersedes`] edge pointing from the newer fact.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Fact {
    /// Unique identifier for this fact.
    pub id: FactId,
    /// The entity this fact is about (references a [`Node`] id).
    pub subject: NodeId,
    /// The relationship type (e.g., `"lives_in"`, `"color_is"`).
    pub predicate: String,
    /// The value (e.g., `"Madrid"`, `"Blue"`).
    pub object: String,
    /// Confidence score from the source (0.0 = no confidence, 1.0 = certain).
    pub source_confidence: f32,
    /// When this fact was recorded (Unix timestamp).
    pub timestamp: u64,
    /// Cognitive processing mode under which this fact was generated.
    pub mindset_tag: MindsetTag,
    /// Current state in the conflict resolution lifecycle.
    pub resolution_state: ResolutionState,
}

/// The outcome of a conflict resolution between two [`Fact`]s.
///
/// Contains both facts and the resolution decision. Neither fact is deleted;
/// the losing fact's [`ResolutionState`] is updated and a
/// [`EdgeType::ConflictsWith`] edge is created.
#[derive(Debug, Clone)]
pub struct Conflict {
    /// The existing fact that was already in the system.
    pub existing: Fact,
    /// The new fact that triggered the conflict.
    pub incoming: Fact,
    /// Which resolution strategy was applied.
    pub strategy: ResolutionStrategy,
}

/// Strategy used to resolve a conflict between two facts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ResolutionStrategy {
    /// Newer fact wins because timestamps differ by more than the temporal threshold.
    Temporal,
    /// Higher-confidence fact wins because confidence difference exceeds threshold.
    Confidence,
    /// Cannot be auto-resolved; both facts marked as contradictions.
    Ambiguous,
}

/// The result of applying conflict resolution rules to two facts.
///
/// Contains the updated facts (with adjusted resolution states) and the
/// strategy that was applied.
#[derive(Debug, Clone)]
pub struct Resolution {
    /// The fact that "won" the resolution (or the existing fact if ambiguous).
    pub winner: Fact,
    /// The fact that "lost" (or the incoming fact if ambiguous).
    pub loser: Fact,
    /// Which rule triggered the resolution.
    pub strategy: ResolutionStrategy,
}

/// Configuration for conflict resolution thresholds.
///
/// These values control when temporal vs. confidence-based resolution applies.
#[derive(Debug, Clone)]
pub struct ConflictConfig {
    /// Timestamp difference (in seconds) above which the temporal rule applies.
    /// Default: 1 year (365.25 days × 86400 seconds).
    pub temporal_threshold_secs: u64,
    /// Minimum confidence difference for the confidence rule to apply.
    /// Default: 0.3.
    pub confidence_threshold: f32,
}

impl Default for ConflictConfig {
    fn default() -> Self {
        Self {
            temporal_threshold_secs: 365 * 24 * 3600 + 6 * 3600, // ~365.25 days
            confidence_threshold: 0.3,
        }
    }
}

impl Fact {
    /// Create a new fact with `Accepted` resolution state and `Convergent` mindset.
    pub fn new(
        id: FactId,
        subject: NodeId,
        predicate: impl Into<String>,
        object: impl Into<String>,
        confidence: f32,
        timestamp: u64,
    ) -> Self {
        Self {
            id,
            subject,
            predicate: predicate.into(),
            object: object.into(),
            source_confidence: confidence.clamp(0.0, 1.0),
            timestamp,
            mindset_tag: MindsetTag::Convergent,
            resolution_state: ResolutionState::Accepted,
        }
    }
}

impl Edge {
    /// Create a `CONFLICTS_WITH` edge between two facts, with standard metadata.
    ///
    /// The metadata includes `detected_at` (timestamp of detection) and
    /// `resolution_strategy` (the rule used to resolve the conflict).
    pub fn conflict(
        source_fact_id: NodeId,
        target_fact_id: NodeId,
        detected_at: u64,
        strategy: ResolutionStrategy,
    ) -> Self {
        let strategy_str = match strategy {
            ResolutionStrategy::Temporal => "temporal",
            ResolutionStrategy::Confidence => "confidence",
            ResolutionStrategy::Ambiguous => "ambiguous",
        };
        let mut metadata = HashMap::new();
        metadata.insert(
            "detected_at".to_string(),
            Value::Integer(detected_at as i64),
        );
        metadata.insert(
            "resolution_strategy".to_string(),
            Value::String(strategy_str.to_string()),
        );
        Self {
            source: source_fact_id,
            target: target_fact_id,
            edge_type: EdgeType::ConflictsWith,
            weight: 1.0,
            metadata,
        }
    }

    /// Create a `SUPERSEDES` edge from a newer fact to the fact it replaces.
    pub fn supersedes(newer_fact_id: NodeId, older_fact_id: NodeId) -> Self {
        Self {
            source: newer_fact_id,
            target: older_fact_id,
            edge_type: EdgeType::Supersedes,
            weight: 1.0,
            metadata: HashMap::new(),
        }
    }
}

/// Configuration for initializing a storage engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    /// Filesystem path for the database storage directory.
    pub data_dir: String,
    /// Maximum database size in bytes (e.g., 10GB for LMDB map_size).
    pub max_db_size: u64,
    /// Batch size for bulk insert operations.
    pub batch_size: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            data_dir: "data".to_string(),
            max_db_size: 10 * 1024 * 1024 * 1024, // 10GB
            batch_size: 10_000,
        }
    }
}

/// Statistics returned from insert operations.
#[derive(Debug, Clone, Default)]
pub struct InsertStats {
    /// Number of items successfully inserted.
    pub count: usize,
    /// Total duration of the insert operation in microseconds.
    pub duration_us: u64,
}

/// Result of a single benchmark measurement.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchResult {
    /// Name of the benchmark.
    pub name: String,
    /// Engine being benchmarked.
    pub engine: String,
    /// Metric name (e.g., "latency_p95", "throughput").
    pub metric: String,
    /// Measured value.
    pub value: f64,
    /// Unit of measurement (e.g., "ms", "docs/s", "MB").
    pub unit: String,
}

/// The core storage engine abstraction.
///
/// All storage backends must implement this trait to be usable within the
/// Ucotron framework. This enables engine-agnostic benchmarking and
/// cognitive logic that works across different storage technologies.
///
/// # Implementors
/// - `ucotron-helix`: HelixDB implementation using Heed/LMDB
/// - `ucotron-cozo`: CozoDB implementation using Datalog/RocksDB
pub trait StorageEngine: Sized {
    /// Initialize the storage engine with the given configuration.
    ///
    /// Creates or opens the database at the configured path.
    fn init(config: &Config) -> anyhow::Result<Self>;

    /// Insert a batch of nodes into the graph.
    ///
    /// Returns statistics about the insert operation.
    fn insert_nodes(&mut self, nodes: &[Node]) -> anyhow::Result<InsertStats>;

    /// Insert a batch of edges into the graph.
    ///
    /// All referenced node IDs should already exist in the graph.
    fn insert_edges(&mut self, edges: &[Edge]) -> anyhow::Result<InsertStats>;

    /// Retrieve a single node by its ID.
    ///
    /// Returns `None` if the node does not exist.
    fn get_node(&self, id: NodeId) -> anyhow::Result<Option<Node>>;

    /// Get all nodes reachable within `hops` traversal steps from the given node.
    ///
    /// Uses BFS/iterative traversal (HelixDB) or recursive Datalog (CozoDB).
    /// Avoids cycles using a visited set.
    fn get_neighbors(&self, id: NodeId, hops: u8) -> anyhow::Result<Vec<Node>>;

    /// Find the `top_k` most similar nodes by cosine similarity to the query vector.
    ///
    /// Returns pairs of (NodeId, similarity_score) sorted by descending similarity.
    fn vector_search(&self, query: &[f32], top_k: usize) -> anyhow::Result<Vec<(NodeId, f32)>>;

    /// Hybrid search combining vector similarity with graph traversal.
    ///
    /// Pipeline:
    /// 1. Vector search for top-k similar nodes
    /// 2. Graph traversal of `hops` steps from each result
    /// 3. Deduplicate and rank by combined score (similarity * hop_decay)
    fn hybrid_search(
        &self,
        query: &[f32],
        top_k: usize,
        hops: u8,
    ) -> anyhow::Result<Vec<Node>>;

    /// Find a path from `source` to `target` in the graph.
    ///
    /// Returns `Some(Vec<NodeId>)` with the full path (including source and target)
    /// if a path exists, or `None` if no path is found. Uses BFS for shortest path
    /// in unweighted graphs (HelixDB) or recursive Datalog (CozoDB).
    ///
    /// The `max_depth` parameter limits the maximum path length to prevent
    /// unbounded traversals on large graphs.
    fn find_path(
        &self,
        source: NodeId,
        target: NodeId,
        max_depth: u32,
    ) -> anyhow::Result<Option<Vec<NodeId>>>;

    /// Gracefully shut down the storage engine.
    ///
    /// Flushes pending writes and releases resources.
    fn shutdown(&mut self) -> anyhow::Result<()>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_node_type_variants() {
        let types = [NodeType::Entity, NodeType::Event, NodeType::Fact, NodeType::Skill];
        assert_eq!(types.len(), 4);
    }

    #[test]
    fn test_edge_type_variants() {
        let types = [
            EdgeType::RelatesTo,
            EdgeType::CausedBy,
            EdgeType::ConflictsWith,
            EdgeType::NextEpisode,
            EdgeType::HasProperty,
            EdgeType::Supersedes,
            EdgeType::Actor,
            EdgeType::Object,
            EdgeType::Location,
            EdgeType::Companion,
        ];
        assert_eq!(types.len(), 10);
    }

    #[test]
    fn test_node_serialization() {
        let node = Node {
            id: 1,
            content: "Test node".to_string(),
            embedding: vec![0.1; 384],
            metadata: HashMap::new(),
            node_type: NodeType::Entity,
            timestamp: 1000,
            media_type: None,
            media_uri: None,
            embedding_visual: None,
            timestamp_range: None,
            parent_video_id: None,
        };

        let serialized = bincode::serialize(&node).expect("serialize");
        let deserialized: Node = bincode::deserialize(&serialized).expect("deserialize");
        assert_eq!(deserialized.id, node.id);
        assert_eq!(deserialized.content, node.content);
        assert_eq!(deserialized.embedding.len(), 384);
        assert_eq!(deserialized.node_type, NodeType::Entity);
    }

    #[test]
    fn test_edge_serialization() {
        let edge = Edge {
            source: 1,
            target: 2,
            edge_type: EdgeType::RelatesTo,
            weight: 0.5,
            metadata: HashMap::new(),
        };

        let serialized = bincode::serialize(&edge).expect("serialize");
        let deserialized: Edge = bincode::deserialize(&serialized).expect("deserialize");
        assert_eq!(deserialized.source, edge.source);
        assert_eq!(deserialized.target, edge.target);
        assert_eq!(deserialized.edge_type, EdgeType::RelatesTo);
    }

    #[test]
    fn test_config_default() {
        let config = Config::default();
        assert_eq!(config.batch_size, 10_000);
        assert_eq!(config.max_db_size, 10 * 1024 * 1024 * 1024);
    }

    #[test]
    fn test_value_variants() {
        let vals = [
            Value::String("test".into()),
            Value::Integer(42),
            Value::Float(3.15),
            Value::Bool(true),
        ];
        assert_eq!(vals.len(), 4);
    }

    // --- Cognitive data model tests ---

    #[test]
    fn test_mindset_tag_variants() {
        let tags = [MindsetTag::Convergent, MindsetTag::Divergent, MindsetTag::Algorithmic];
        assert_eq!(tags.len(), 3);
    }

    #[test]
    fn test_resolution_state_variants() {
        let states = [
            ResolutionState::Accepted,
            ResolutionState::Contradiction,
            ResolutionState::Superseded,
            ResolutionState::Ambiguous,
        ];
        assert_eq!(states.len(), 4);
    }

    #[test]
    fn test_fact_serialization() {
        let fact = Fact {
            id: 42,
            subject: 1,
            predicate: "lives_in".to_string(),
            object: "Madrid".to_string(),
            source_confidence: 0.95,
            timestamp: 1_700_000_000,
            mindset_tag: MindsetTag::Convergent,
            resolution_state: ResolutionState::Accepted,
        };

        let serialized = bincode::serialize(&fact).expect("serialize fact");
        let deserialized: Fact = bincode::deserialize(&serialized).expect("deserialize fact");
        assert_eq!(deserialized.id, 42);
        assert_eq!(deserialized.subject, 1);
        assert_eq!(deserialized.predicate, "lives_in");
        assert_eq!(deserialized.object, "Madrid");
        assert!((deserialized.source_confidence - 0.95).abs() < 1e-6);
        assert_eq!(deserialized.timestamp, 1_700_000_000);
        assert_eq!(deserialized.mindset_tag, MindsetTag::Convergent);
        assert_eq!(deserialized.resolution_state, ResolutionState::Accepted);
    }

    #[test]
    fn test_fact_new_constructor() {
        let fact = Fact::new(1, 100, "color_is", "Blue", 0.8, 1_000_000);
        assert_eq!(fact.id, 1);
        assert_eq!(fact.subject, 100);
        assert_eq!(fact.predicate, "color_is");
        assert_eq!(fact.object, "Blue");
        assert!((fact.source_confidence - 0.8).abs() < 1e-6);
        assert_eq!(fact.mindset_tag, MindsetTag::Convergent);
        assert_eq!(fact.resolution_state, ResolutionState::Accepted);
    }

    #[test]
    fn test_fact_new_clamps_confidence() {
        let too_high = Fact::new(1, 1, "p", "o", 1.5, 0);
        assert!((too_high.source_confidence - 1.0).abs() < 1e-6);

        let too_low = Fact::new(2, 1, "p", "o", -0.5, 0);
        assert!((too_low.source_confidence - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_conflict_edge_metadata() {
        let edge = Edge::conflict(10, 20, 1_700_000_000, ResolutionStrategy::Temporal);
        assert_eq!(edge.source, 10);
        assert_eq!(edge.target, 20);
        assert_eq!(edge.edge_type, EdgeType::ConflictsWith);
        assert_eq!(
            edge.metadata.get("detected_at"),
            Some(&Value::Integer(1_700_000_000))
        );
        assert_eq!(
            edge.metadata.get("resolution_strategy"),
            Some(&Value::String("temporal".to_string()))
        );
    }

    #[test]
    fn test_supersedes_edge() {
        let edge = Edge::supersedes(20, 10);
        assert_eq!(edge.source, 20);
        assert_eq!(edge.target, 10);
        assert_eq!(edge.edge_type, EdgeType::Supersedes);
        assert!(edge.metadata.is_empty());
    }

    #[test]
    fn test_conflict_config_default() {
        let config = ConflictConfig::default();
        // ~365.25 days in seconds
        assert!(config.temporal_threshold_secs > 365 * 24 * 3600 - 1);
        assert!((config.confidence_threshold - 0.3).abs() < 1e-6);
    }

    #[test]
    fn test_resolution_strategy_serialization() {
        for strategy in [
            ResolutionStrategy::Temporal,
            ResolutionStrategy::Confidence,
            ResolutionStrategy::Ambiguous,
        ] {
            let serialized = bincode::serialize(&strategy).expect("serialize");
            let deserialized: ResolutionStrategy =
                bincode::deserialize(&serialized).expect("deserialize");
            assert_eq!(deserialized, strategy);
        }
    }

    #[test]
    fn test_mindset_tag_serialization() {
        for tag in [MindsetTag::Convergent, MindsetTag::Divergent, MindsetTag::Algorithmic] {
            let serialized = bincode::serialize(&tag).expect("serialize");
            let deserialized: MindsetTag =
                bincode::deserialize(&serialized).expect("deserialize");
            assert_eq!(deserialized, tag);
        }
    }

    #[test]
    fn test_resolution_state_serialization() {
        for state in [
            ResolutionState::Accepted,
            ResolutionState::Contradiction,
            ResolutionState::Superseded,
            ResolutionState::Ambiguous,
        ] {
            let serialized = bincode::serialize(&state).expect("serialize");
            let deserialized: ResolutionState =
                bincode::deserialize(&serialized).expect("deserialize");
            assert_eq!(deserialized, state);
        }
    }

    #[test]
    fn test_node_empty_content_and_metadata() {
        let node = Node {
            id: 0,
            content: String::new(),
            embedding: vec![],
            metadata: HashMap::new(),
            node_type: NodeType::Entity,
            timestamp: 0,
            media_type: None,
            media_uri: None,
            embedding_visual: None,
            timestamp_range: None,
            parent_video_id: None,
        };
        let serialized = bincode::serialize(&node).expect("serialize");
        let deserialized: Node = bincode::deserialize(&serialized).expect("deserialize");
        assert_eq!(deserialized.content, "");
        assert!(deserialized.embedding.is_empty());
        assert!(deserialized.metadata.is_empty());
        assert_eq!(deserialized.timestamp, 0);
    }

    #[test]
    fn test_node_unicode_content() {
        let node = Node {
            id: 1,
            content: "Berlín café résumé 日本語 🎉".to_string(),
            embedding: vec![1.0; 384],
            metadata: HashMap::new(),
            node_type: NodeType::Entity,
            timestamp: 100,
            media_type: None,
            media_uri: None,
            embedding_visual: None,
            timestamp_range: None,
            parent_video_id: None,
        };
        let serialized = bincode::serialize(&node).expect("serialize");
        let deserialized: Node = bincode::deserialize(&serialized).expect("deserialize");
        assert_eq!(deserialized.content, "Berlín café résumé 日本語 🎉");
    }

    #[test]
    fn test_node_large_metadata() {
        let mut metadata = HashMap::new();
        for i in 0..100 {
            metadata.insert(format!("key_{}", i), Value::Integer(i));
        }
        let node = Node {
            id: 1,
            content: "test".to_string(),
            embedding: vec![],
            metadata,
            node_type: NodeType::Entity,
            timestamp: 100,
            media_type: None,
            media_uri: None,
            embedding_visual: None,
            timestamp_range: None,
            parent_video_id: None,
        };
        let serialized = bincode::serialize(&node).expect("serialize");
        let deserialized: Node = bincode::deserialize(&serialized).expect("deserialize");
        assert_eq!(deserialized.metadata.len(), 100);
    }

    #[test]
    fn test_edge_all_types_serialization() {
        let all_types = [
            EdgeType::RelatesTo,
            EdgeType::CausedBy,
            EdgeType::ConflictsWith,
            EdgeType::NextEpisode,
            EdgeType::HasProperty,
            EdgeType::Supersedes,
            EdgeType::Actor,
            EdgeType::Object,
            EdgeType::Location,
            EdgeType::Companion,
        ];
        for edge_type in all_types {
            let edge = Edge {
                source: 1,
                target: 2,
                edge_type,
                weight: 0.5,
                metadata: HashMap::new(),
            };
            let serialized = bincode::serialize(&edge).expect("serialize");
            let deserialized: Edge = bincode::deserialize(&serialized).expect("deserialize");
            assert_eq!(deserialized.edge_type, edge_type);
        }
    }

    #[test]
    fn test_conflict_edge_all_strategies() {
        for strategy in [
            ResolutionStrategy::Temporal,
            ResolutionStrategy::Confidence,
            ResolutionStrategy::Ambiguous,
        ] {
            let edge = Edge::conflict(1, 2, 1000, strategy);
            assert_eq!(edge.edge_type, EdgeType::ConflictsWith);
            assert!(edge.metadata.contains_key("resolution_strategy"));
        }
    }

    // --- Multimodal / MediaType tests (Phase 3.5) ---

    #[test]
    fn test_media_type_variants() {
        let types = [
            MediaType::Text,
            MediaType::Audio,
            MediaType::Image,
            MediaType::VideoSegment,
        ];
        assert_eq!(types.len(), 4);
        assert_eq!(MediaType::default(), MediaType::Text);
    }

    #[test]
    fn test_media_type_serialization() {
        for mt in [
            MediaType::Text,
            MediaType::Audio,
            MediaType::Image,
            MediaType::VideoSegment,
        ] {
            let serialized = bincode::serialize(&mt).expect("serialize");
            let deserialized: MediaType =
                bincode::deserialize(&serialized).expect("deserialize");
            assert_eq!(deserialized, mt);
        }
    }

    #[test]
    fn test_node_multimodal_fields_none() {
        let node = Node {
            id: 1,
            content: "text-only node".to_string(),
            embedding: vec![0.1; 384],
            metadata: HashMap::new(),
            node_type: NodeType::Entity,
            timestamp: 1000,
            media_type: None,
            media_uri: None,
            embedding_visual: None,
            timestamp_range: None,
            parent_video_id: None,
        };
        let serialized = bincode::serialize(&node).expect("serialize");
        let deserialized: Node = bincode::deserialize(&serialized).expect("deserialize");
        assert!(deserialized.media_type.is_none());
        assert!(deserialized.media_uri.is_none());
        assert!(deserialized.embedding_visual.is_none());
        assert!(deserialized.timestamp_range.is_none());
        assert!(deserialized.parent_video_id.is_none());
    }

    #[test]
    fn test_node_multimodal_fields_populated() {
        let node = Node {
            id: 42,
            content: "image of a sunset".to_string(),
            embedding: vec![0.5; 384],
            metadata: HashMap::new(),
            node_type: NodeType::Entity,
            timestamp: 1_700_000_000,
            media_type: Some(MediaType::Image),
            media_uri: Some("file:///media/sunset.jpg".to_string()),
            embedding_visual: Some(vec![0.3; 512]),
            timestamp_range: None,
            parent_video_id: None,
        };
        let serialized = bincode::serialize(&node).expect("serialize");
        let deserialized: Node = bincode::deserialize(&serialized).expect("deserialize");
        assert_eq!(deserialized.media_type, Some(MediaType::Image));
        assert_eq!(
            deserialized.media_uri,
            Some("file:///media/sunset.jpg".to_string())
        );
        assert_eq!(deserialized.embedding_visual.as_ref().unwrap().len(), 512);
        assert!(deserialized.timestamp_range.is_none());
        assert!(deserialized.parent_video_id.is_none());
    }

    #[test]
    fn test_node_video_segment_fields() {
        let node = Node {
            id: 100,
            content: "video segment transcript".to_string(),
            embedding: vec![0.2; 384],
            metadata: HashMap::new(),
            node_type: NodeType::Event,
            timestamp: 1_700_000_000,
            media_type: Some(MediaType::VideoSegment),
            media_uri: Some("file:///media/lecture.mp4".to_string()),
            embedding_visual: Some(vec![0.1; 512]),
            timestamp_range: Some((30_000, 60_000)),
            parent_video_id: Some(99),
        };
        let serialized = bincode::serialize(&node).expect("serialize");
        let deserialized: Node = bincode::deserialize(&serialized).expect("deserialize");
        assert_eq!(deserialized.media_type, Some(MediaType::VideoSegment));
        assert_eq!(deserialized.timestamp_range, Some((30_000, 60_000)));
        assert_eq!(deserialized.parent_video_id, Some(99));
    }
}
