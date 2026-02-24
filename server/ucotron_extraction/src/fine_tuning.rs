//! # Fine-Tuning Pipeline
//!
//! Generates training datasets from the Ucotron knowledge graph for
//! domain-specific relation extraction model fine-tuning.
//!
//! ## Training Data Format
//!
//! Output is JSONL (one JSON object per line):
//! ```json
//! {"text": "Juan lives in Madrid...", "entities": [...], "expected_relations": [...]}
//! ```
//!
//! Each line contains the source text, extracted entities, and verified relations
//! from the graph — suitable for supervised fine-tuning (SFT) with TRL.
//!
//! ## Pipeline
//!
//! 1. **Export graph**: Read all Entity nodes and their edges from `BackendRegistry`
//! 2. **Generate samples**: For each entity cluster, build (text, entities, relations) triples
//! 3. **Format for SFT**: Convert to chat-template JSONL for Qwen3/Llama instruction tuning
//! 4. **Split**: Train/validation split with configurable ratio

use std::collections::HashMap;
use std::io::Write;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use ucotron_core::{BackendRegistry, Edge, EdgeType, Node, NodeId, NodeType};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for training dataset generation.
#[derive(Debug, Clone)]
pub struct DatasetConfig {
    /// Maximum number of samples to generate.
    pub max_samples: usize,
    /// Train/validation split ratio (0.0–1.0, fraction used for training).
    pub train_ratio: f32,
    /// Minimum number of relations a node cluster must have to be included.
    pub min_relations: usize,
    /// Maximum text length per sample (characters). Longer texts are truncated.
    pub max_text_length: usize,
    /// Seed for deterministic shuffling of the train/val split.
    pub seed: u64,
}

impl Default for DatasetConfig {
    fn default() -> Self {
        Self {
            max_samples: 10_000,
            train_ratio: 0.8,
            min_relations: 1,
            max_text_length: 2048,
            seed: 42,
        }
    }
}

// ---------------------------------------------------------------------------
// Training sample types
// ---------------------------------------------------------------------------

/// An entity annotation in a training sample.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TrainingEntity {
    pub text: String,
    pub label: String,
    pub start: usize,
    pub end: usize,
}

/// A relation annotation in a training sample.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TrainingRelation {
    pub subject: String,
    pub predicate: String,
    pub object: String,
    pub confidence: f32,
}

/// A single training sample for relation extraction fine-tuning.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingSample {
    /// Source text containing the entities.
    pub text: String,
    /// Entity annotations.
    pub entities: Vec<TrainingEntity>,
    /// Expected relations to extract.
    pub expected_relations: Vec<TrainingRelation>,
}

/// A training sample formatted for SFT chat-template.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SftSample {
    /// The system prompt.
    pub system: String,
    /// The user message (text + entities).
    pub user: String,
    /// The assistant response (JSON relations).
    pub assistant: String,
}

/// Result of dataset generation.
#[derive(Debug)]
pub struct DatasetResult {
    /// Training samples.
    pub train: Vec<TrainingSample>,
    /// Validation samples.
    pub validation: Vec<TrainingSample>,
    /// Total entity nodes processed.
    pub total_entities: usize,
    /// Total edges processed.
    pub total_edges: usize,
    /// Samples skipped (below min_relations threshold).
    pub skipped: usize,
}

// ---------------------------------------------------------------------------
// Dataset generation from graph
// ---------------------------------------------------------------------------

/// Generate training dataset from the Ucotron knowledge graph.
///
/// Reads all Entity nodes and their outgoing edges, then constructs
/// training samples where the text is the entity node content and
/// relations are derived from the graph edges.
pub fn generate_dataset(
    registry: &BackendRegistry,
    config: &DatasetConfig,
) -> Result<DatasetResult> {
    let graph = registry.graph();

    // 1. Read all nodes
    let all_nodes = graph
        .get_all_nodes()
        .context("Failed to read nodes from graph")?;

    // Build node lookup
    let node_map: HashMap<NodeId, &Node> = all_nodes.iter().map(|n| (n.id, n)).collect();

    // 2. Read all edges
    let all_edges = graph
        .get_all_edges_full()
        .context("Failed to read edges from graph")?;

    // Group edges by source node
    let mut edges_by_source: HashMap<NodeId, Vec<&Edge>> = HashMap::new();
    for edge in &all_edges {
        edges_by_source.entry(edge.source).or_default().push(edge);
    }

    // 3. Generate samples from entity nodes with edges
    let mut samples: Vec<TrainingSample> = Vec::new();
    let mut skipped = 0;

    for node in &all_nodes {
        if samples.len() >= config.max_samples {
            break;
        }

        // Only process Entity-type nodes that have meaningful content
        if !matches!(node.node_type, NodeType::Entity) {
            continue;
        }

        if node.content.trim().is_empty() {
            continue;
        }

        let edges = match edges_by_source.get(&node.id) {
            Some(e) => e,
            None => {
                skipped += 1;
                continue;
            }
        };

        // Filter to meaningful relation edges
        let relation_edges: Vec<&&Edge> = edges
            .iter()
            .filter(|e| is_relation_edge(e.edge_type))
            .collect();

        if relation_edges.len() < config.min_relations {
            skipped += 1;
            continue;
        }

        // Build sample
        if let Some(sample) = build_sample(node, &relation_edges, &node_map, config) {
            samples.push(sample);
        } else {
            skipped += 1;
        }
    }

    // 4. Deterministic shuffle and split
    deterministic_shuffle(&mut samples, config.seed);

    let split_idx = (samples.len() as f32 * config.train_ratio).round() as usize;
    let split_idx = split_idx.min(samples.len());

    let validation = samples.split_off(split_idx);
    let train = samples;

    Ok(DatasetResult {
        train,
        validation,
        total_entities: all_nodes
            .iter()
            .filter(|n| matches!(n.node_type, NodeType::Entity))
            .count(),
        total_edges: all_edges.len(),
        skipped,
    })
}

/// Generate training dataset from pre-loaded nodes and edges (no registry needed).
///
/// This is useful for testing or when data is already available in memory.
pub fn generate_dataset_from_data(
    nodes: &[Node],
    edges: &[Edge],
    config: &DatasetConfig,
) -> DatasetResult {
    let node_map: HashMap<NodeId, &Node> = nodes.iter().map(|n| (n.id, n)).collect();

    let mut edges_by_source: HashMap<NodeId, Vec<&Edge>> = HashMap::new();
    for edge in edges {
        edges_by_source.entry(edge.source).or_default().push(edge);
    }

    let mut samples: Vec<TrainingSample> = Vec::new();
    let mut skipped = 0;

    for node in nodes {
        if samples.len() >= config.max_samples {
            break;
        }

        if !matches!(node.node_type, NodeType::Entity) {
            continue;
        }

        if node.content.trim().is_empty() {
            continue;
        }

        let edges = match edges_by_source.get(&node.id) {
            Some(e) => e,
            None => {
                skipped += 1;
                continue;
            }
        };

        let relation_edges: Vec<&&Edge> = edges
            .iter()
            .filter(|e| is_relation_edge(e.edge_type))
            .collect();

        if relation_edges.len() < config.min_relations {
            skipped += 1;
            continue;
        }

        if let Some(sample) = build_sample(node, &relation_edges, &node_map, config) {
            samples.push(sample);
        } else {
            skipped += 1;
        }
    }

    deterministic_shuffle(&mut samples, config.seed);

    let split_idx = (samples.len() as f32 * config.train_ratio).round() as usize;
    let split_idx = split_idx.min(samples.len());

    let validation = samples.split_off(split_idx);
    let train = samples;

    DatasetResult {
        train,
        validation,
        total_entities: nodes
            .iter()
            .filter(|n| matches!(n.node_type, NodeType::Entity))
            .count(),
        total_edges: edges.len(),
        skipped,
    }
}

// ---------------------------------------------------------------------------
// SFT chat formatting
// ---------------------------------------------------------------------------

const SYSTEM_PROMPT: &str = "You are a relation extraction assistant. Given a text and a list of entities found in it, extract all relationships between the entities. Output ONLY a JSON array of objects with fields: subject, predicate, object, confidence.";

/// Convert a training sample to SFT chat format.
pub fn sample_to_sft(sample: &TrainingSample) -> SftSample {
    let entity_list: String = sample
        .entities
        .iter()
        .map(|e| format!("- \"{}\" ({})", e.text, e.label))
        .collect::<Vec<_>>()
        .join("\n");

    let user = format!(
        "Extract all relationships from the following text.\n\nText: \"{}\"\n\nEntities found:\n{}",
        sample.text, entity_list
    );

    let relations_json: Vec<serde_json::Value> = sample
        .expected_relations
        .iter()
        .map(|r| {
            serde_json::json!({
                "subject": r.subject,
                "predicate": r.predicate,
                "object": r.object,
                "confidence": r.confidence,
            })
        })
        .collect();

    let assistant = serde_json::to_string(&relations_json).unwrap_or_else(|_| "[]".to_string());

    SftSample {
        system: SYSTEM_PROMPT.to_string(),
        user,
        assistant,
    }
}

/// Convert a training sample to TRL SFT messages format (OpenAI-compatible).
pub fn sample_to_messages(sample: &TrainingSample) -> serde_json::Value {
    let sft = sample_to_sft(sample);
    serde_json::json!({
        "messages": [
            {"role": "system", "content": sft.system},
            {"role": "user", "content": sft.user},
            {"role": "assistant", "content": sft.assistant},
        ]
    })
}

// ---------------------------------------------------------------------------
// JSONL I/O
// ---------------------------------------------------------------------------

/// Write training samples to a JSONL file (raw format).
pub fn write_jsonl(samples: &[TrainingSample], path: &std::path::Path) -> Result<()> {
    let file = std::fs::File::create(path).context("Failed to create JSONL file")?;
    let mut writer = std::io::BufWriter::new(file);

    for sample in samples {
        let json = serde_json::to_string(sample).context("Failed to serialize sample")?;
        writeln!(writer, "{}", json)?;
    }

    writer.flush()?;
    Ok(())
}

/// Write training samples to a JSONL file in SFT messages format.
pub fn write_sft_jsonl(samples: &[TrainingSample], path: &std::path::Path) -> Result<()> {
    let file = std::fs::File::create(path).context("Failed to create SFT JSONL file")?;
    let mut writer = std::io::BufWriter::new(file);

    for sample in samples {
        let json = sample_to_messages(sample);
        let line = serde_json::to_string(&json).context("Failed to serialize SFT sample")?;
        writeln!(writer, "{}", line)?;
    }

    writer.flush()?;
    Ok(())
}

/// Read training samples from a JSONL file (raw format).
pub fn read_jsonl(path: &std::path::Path) -> Result<Vec<TrainingSample>> {
    let content = std::fs::read_to_string(path).context("Failed to read JSONL file")?;
    parse_jsonl(&content)
}

/// Parse JSONL content into training samples.
pub fn parse_jsonl(content: &str) -> Result<Vec<TrainingSample>> {
    content
        .lines()
        .filter(|l| !l.trim().is_empty())
        .enumerate()
        .map(|(i, line)| {
            serde_json::from_str(line).with_context(|| format!("Failed to parse line {}", i + 1))
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Check if an edge type represents a meaningful relation for training data.
fn is_relation_edge(edge_type: EdgeType) -> bool {
    matches!(
        edge_type,
        EdgeType::RelatesTo
            | EdgeType::CausedBy
            | EdgeType::HasProperty
            | EdgeType::Actor
            | EdgeType::Object
            | EdgeType::Location
            | EdgeType::Companion
    )
}

/// Convert an EdgeType to a predicate string for training data.
fn edge_type_to_predicate(edge_type: EdgeType) -> &'static str {
    match edge_type {
        EdgeType::RelatesTo => "related_to",
        EdgeType::CausedBy => "caused_by",
        EdgeType::HasProperty => "has_property",
        EdgeType::Actor => "actor_in",
        EdgeType::Object => "object_of",
        EdgeType::Location => "located_in",
        EdgeType::Companion => "companion_of",
        EdgeType::ConflictsWith => "conflicts_with",
        EdgeType::NextEpisode => "next_episode",
        EdgeType::Supersedes => "supersedes",
    }
}

/// Infer an entity label from a NodeType.
fn node_type_to_label(node_type: &NodeType) -> &'static str {
    match node_type {
        NodeType::Entity => "entity",
        NodeType::Event => "event",
        NodeType::Fact => "fact",
        NodeType::Skill => "skill",
    }
}

/// Build a training sample from a source node and its outgoing relation edges.
fn build_sample(
    source: &Node,
    edges: &[&&Edge],
    node_map: &HashMap<NodeId, &Node>,
    config: &DatasetConfig,
) -> Option<TrainingSample> {
    let mut text = source.content.clone();
    if text.len() > config.max_text_length {
        // Truncate at a word boundary
        if let Some(pos) = text[..config.max_text_length].rfind(' ') {
            text.truncate(pos);
        } else {
            text.truncate(config.max_text_length);
        }
    }

    let mut entities = Vec::new();
    let mut relations = Vec::new();

    // Source entity is always the first entity
    let source_label = node_type_to_label(&source.node_type);
    let source_name = extract_entity_name(&source.content);

    // Find subject entity position in text
    let source_start = text.find(&source_name).unwrap_or(0);
    let source_end = source_start + source_name.len();

    entities.push(TrainingEntity {
        text: source_name.clone(),
        label: source_label.to_string(),
        start: source_start,
        end: source_end,
    });

    for edge in edges {
        let target = match node_map.get(&edge.target) {
            Some(n) => n,
            None => continue,
        };

        let target_name = extract_entity_name(&target.content);
        let target_label = node_type_to_label(&target.node_type);

        // Find or note target entity position
        let target_start = text.find(&target_name).unwrap_or(0);
        let target_end = target_start + target_name.len();

        // Add target entity if not duplicate
        if !entities.iter().any(|e| e.text == target_name) {
            entities.push(TrainingEntity {
                text: target_name.clone(),
                label: target_label.to_string(),
                start: target_start,
                end: target_end,
            });
        }

        relations.push(TrainingRelation {
            subject: source_name.clone(),
            predicate: edge_type_to_predicate(edge.edge_type).to_string(),
            object: target_name,
            confidence: edge.weight.clamp(0.0, 1.0),
        });
    }

    if relations.is_empty() {
        return None;
    }

    Some(TrainingSample {
        text,
        entities,
        expected_relations: relations,
    })
}

/// Extract a short entity name from node content.
///
/// Uses the first sentence or the first N words as the entity name.
fn extract_entity_name(content: &str) -> String {
    let trimmed = content.trim();
    // Use content up to first period, question mark, or newline
    let end = trimmed.find(['.', '?', '!', '\n']).unwrap_or(trimmed.len());

    let name = &trimmed[..end];
    // Limit to ~60 chars for entity name
    if name.len() > 60 {
        if let Some(pos) = name[..60].rfind(' ') {
            return name[..pos].to_string();
        }
        return name[..60].to_string();
    }
    name.to_string()
}

/// Deterministic shuffle using a simple LCG PRNG.
fn deterministic_shuffle<T>(items: &mut [T], seed: u64) {
    let len = items.len();
    if len <= 1 {
        return;
    }
    // Linear congruential generator (Knuth)
    let mut state = seed;
    for i in (1..len).rev() {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let j = (state >> 33) as usize % (i + 1);
        items.swap(i, j);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use ucotron_core::{Edge, EdgeType, Node, NodeType};

    fn make_entity_node(id: NodeId, content: &str) -> Node {
        Node {
            id,
            content: content.to_string(),
            embedding: vec![0.0; 384],
            metadata: HashMap::new(),
            node_type: NodeType::Entity,
            timestamp: 1_700_000_000,
            media_type: None,
            media_uri: None,
            embedding_visual: None,
            timestamp_range: None,
            parent_video_id: None,
        }
    }

    fn make_event_node(id: NodeId, content: &str) -> Node {
        Node {
            id,
            content: content.to_string(),
            embedding: vec![0.0; 384],
            metadata: HashMap::new(),
            node_type: NodeType::Event,
            timestamp: 1_700_000_000,
            media_type: None,
            media_uri: None,
            embedding_visual: None,
            timestamp_range: None,
            parent_video_id: None,
        }
    }

    fn make_edge(source: NodeId, target: NodeId, edge_type: EdgeType, weight: f32) -> Edge {
        Edge {
            source,
            target,
            edge_type,
            weight,
            metadata: HashMap::new(),
        }
    }

    // --- Dataset generation tests ---

    #[test]
    fn test_generate_dataset_basic() {
        let nodes = vec![
            make_entity_node(1, "Juan is a software engineer from Madrid"),
            make_entity_node(2, "Madrid is the capital of Spain"),
            make_entity_node(3, "Spain is a country in Europe"),
        ];
        let edges = vec![
            make_edge(1, 2, EdgeType::Location, 0.9),
            make_edge(2, 3, EdgeType::HasProperty, 0.8),
        ];
        let config = DatasetConfig::default();
        let result = generate_dataset_from_data(&nodes, &edges, &config);

        assert!(!result.train.is_empty() || !result.validation.is_empty());
        assert_eq!(result.total_entities, 3);
        assert_eq!(result.total_edges, 2);
    }

    #[test]
    fn test_generate_dataset_skips_event_nodes() {
        let nodes = vec![
            make_event_node(1, "Meeting happened yesterday"),
            make_entity_node(2, "Alice works at Google"),
            make_entity_node(3, "Google is a tech company"),
        ];
        let edges = vec![
            make_edge(1, 2, EdgeType::Actor, 0.8),
            make_edge(2, 3, EdgeType::RelatesTo, 0.7),
        ];
        let config = DatasetConfig::default();
        let result = generate_dataset_from_data(&nodes, &edges, &config);

        // Event node should be skipped as source
        let all_sources: Vec<&str> = result
            .train
            .iter()
            .chain(result.validation.iter())
            .flat_map(|s| s.expected_relations.iter())
            .map(|r| r.subject.as_str())
            .collect();

        // "Meeting happened yesterday" should not be a subject
        assert!(!all_sources.contains(&"Meeting happened yesterday"));
    }

    #[test]
    fn test_generate_dataset_respects_min_relations() {
        let nodes = vec![
            make_entity_node(1, "Alice"),
            make_entity_node(2, "Bob"),
            make_entity_node(3, "Charlie"),
        ];
        // Node 1 has no edges, node 2 has 1, node 3 has 0
        let edges = vec![make_edge(2, 1, EdgeType::RelatesTo, 0.5)];

        let config = DatasetConfig {
            min_relations: 2, // Require at least 2 relations
            ..Default::default()
        };

        let result = generate_dataset_from_data(&nodes, &edges, &config);

        // No samples should be generated (max 1 relation per node)
        assert!(result.train.is_empty());
        assert!(result.validation.is_empty());
    }

    #[test]
    fn test_generate_dataset_max_samples() {
        let mut nodes = Vec::new();
        let mut edges = Vec::new();

        for i in 0..20 {
            nodes.push(make_entity_node(i, &format!("Entity number {}", i)));
        }
        for i in 0..19 {
            edges.push(make_edge(i, i + 1, EdgeType::RelatesTo, 0.5));
        }

        let config = DatasetConfig {
            max_samples: 5,
            ..Default::default()
        };

        let result = generate_dataset_from_data(&nodes, &edges, &config);
        let total = result.train.len() + result.validation.len();
        assert!(total <= 5);
    }

    #[test]
    fn test_generate_dataset_train_val_split() {
        let mut nodes = Vec::new();
        let mut edges = Vec::new();

        for i in 0..100 {
            nodes.push(make_entity_node(i, &format!("Entity {}", i)));
        }
        for i in 0..99 {
            edges.push(make_edge(i, i + 1, EdgeType::RelatesTo, 0.5));
        }

        let config = DatasetConfig {
            train_ratio: 0.8,
            ..Default::default()
        };

        let result = generate_dataset_from_data(&nodes, &edges, &config);
        let total = result.train.len() + result.validation.len();

        if total > 0 {
            let actual_ratio = result.train.len() as f32 / total as f32;
            assert!(
                (actual_ratio - 0.8).abs() < 0.15,
                "Train ratio {} is too far from 0.8",
                actual_ratio
            );
        }
    }

    #[test]
    fn test_generate_dataset_empty_graph() {
        let config = DatasetConfig::default();
        let result = generate_dataset_from_data(&[], &[], &config);
        assert!(result.train.is_empty());
        assert!(result.validation.is_empty());
        assert_eq!(result.total_entities, 0);
        assert_eq!(result.total_edges, 0);
    }

    #[test]
    fn test_generate_dataset_skips_conflict_edges() {
        let nodes = vec![
            make_entity_node(1, "The sky is blue"),
            make_entity_node(2, "The sky is red"),
        ];
        let edges = vec![
            make_edge(1, 2, EdgeType::ConflictsWith, 0.9), // Should be filtered out
        ];
        let config = DatasetConfig::default();
        let result = generate_dataset_from_data(&nodes, &edges, &config);

        // ConflictsWith is not a relation edge, so no samples generated
        let total = result.train.len() + result.validation.len();
        assert_eq!(total, 0);
    }

    // --- Sample building tests ---

    #[test]
    fn test_build_sample_with_relations() {
        let source = make_entity_node(1, "Juan lives in Madrid and works at Google");
        let target_madrid = make_entity_node(2, "Madrid");
        let target_google = make_entity_node(3, "Google");

        let edge1 = make_edge(1, 2, EdgeType::Location, 0.9);
        let edge2 = make_edge(1, 3, EdgeType::RelatesTo, 0.8);

        let mut node_map = HashMap::new();
        node_map.insert(1u64, &source);
        node_map.insert(2u64, &target_madrid);
        node_map.insert(3u64, &target_google);

        let edges_vec = [&edge1, &edge2];
        let edges_ref: Vec<&&Edge> = edges_vec.iter().collect();

        let config = DatasetConfig::default();
        let sample = build_sample(&source, &edges_ref, &node_map, &config);

        assert!(sample.is_some());
        let sample = sample.unwrap();
        assert_eq!(sample.expected_relations.len(), 2);
        assert!(sample.entities.len() >= 2);
    }

    #[test]
    fn test_build_sample_no_target_node() {
        let source = make_entity_node(1, "Alice");
        let edge = make_edge(1, 999, EdgeType::RelatesTo, 0.5);

        let mut node_map: HashMap<NodeId, &Node> = HashMap::new();
        node_map.insert(1, &source);
        // Node 999 not in map

        let edges_vec = [&edge];
        let edges_ref: Vec<&&Edge> = edges_vec.iter().collect();

        let config = DatasetConfig::default();
        let sample = build_sample(&source, &edges_ref, &node_map, &config);

        // No valid relations (target not found), should return None
        assert!(sample.is_none());
    }

    // --- SFT formatting tests ---

    #[test]
    fn test_sample_to_sft() {
        let sample = TrainingSample {
            text: "Juan lives in Madrid".to_string(),
            entities: vec![
                TrainingEntity {
                    text: "Juan".to_string(),
                    label: "person".to_string(),
                    start: 0,
                    end: 4,
                },
                TrainingEntity {
                    text: "Madrid".to_string(),
                    label: "location".to_string(),
                    start: 14,
                    end: 20,
                },
            ],
            expected_relations: vec![TrainingRelation {
                subject: "Juan".to_string(),
                predicate: "lives_in".to_string(),
                object: "Madrid".to_string(),
                confidence: 0.95,
            }],
        };

        let sft = sample_to_sft(&sample);
        assert!(sft.system.contains("relation extraction"));
        assert!(sft.user.contains("Juan lives in Madrid"));
        assert!(sft.user.contains("Juan"));
        assert!(sft.user.contains("Madrid"));
        assert!(sft.assistant.contains("lives_in"));
    }

    #[test]
    fn test_sample_to_messages() {
        let sample = TrainingSample {
            text: "Alice works at Google".to_string(),
            entities: vec![
                TrainingEntity {
                    text: "Alice".to_string(),
                    label: "person".to_string(),
                    start: 0,
                    end: 5,
                },
                TrainingEntity {
                    text: "Google".to_string(),
                    label: "organization".to_string(),
                    start: 15,
                    end: 21,
                },
            ],
            expected_relations: vec![TrainingRelation {
                subject: "Alice".to_string(),
                predicate: "works_at".to_string(),
                object: "Google".to_string(),
                confidence: 0.9,
            }],
        };

        let msg = sample_to_messages(&sample);
        let messages = msg.get("messages").unwrap().as_array().unwrap();
        assert_eq!(messages.len(), 3);
        assert_eq!(messages[0]["role"], "system");
        assert_eq!(messages[1]["role"], "user");
        assert_eq!(messages[2]["role"], "assistant");
    }

    // --- JSONL I/O tests ---

    #[test]
    fn test_jsonl_roundtrip() {
        let samples = vec![
            TrainingSample {
                text: "Alice works at Google".to_string(),
                entities: vec![TrainingEntity {
                    text: "Alice".to_string(),
                    label: "person".to_string(),
                    start: 0,
                    end: 5,
                }],
                expected_relations: vec![TrainingRelation {
                    subject: "Alice".to_string(),
                    predicate: "works_at".to_string(),
                    object: "Google".to_string(),
                    confidence: 0.9,
                }],
            },
            TrainingSample {
                text: "Bob lives in Berlin".to_string(),
                entities: vec![TrainingEntity {
                    text: "Bob".to_string(),
                    label: "person".to_string(),
                    start: 0,
                    end: 3,
                }],
                expected_relations: vec![TrainingRelation {
                    subject: "Bob".to_string(),
                    predicate: "lives_in".to_string(),
                    object: "Berlin".to_string(),
                    confidence: 0.85,
                }],
            },
        ];

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.jsonl");

        write_jsonl(&samples, &path).unwrap();
        let loaded = read_jsonl(&path).unwrap();

        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded[0].text, "Alice works at Google");
        assert_eq!(loaded[1].text, "Bob lives in Berlin");
        assert_eq!(loaded[0].expected_relations[0].predicate, "works_at");
        assert_eq!(loaded[1].expected_relations[0].predicate, "lives_in");
    }

    #[test]
    fn test_write_sft_jsonl() {
        let samples = vec![TrainingSample {
            text: "Juan moved to Berlin".to_string(),
            entities: vec![
                TrainingEntity {
                    text: "Juan".to_string(),
                    label: "person".to_string(),
                    start: 0,
                    end: 4,
                },
                TrainingEntity {
                    text: "Berlin".to_string(),
                    label: "location".to_string(),
                    start: 14,
                    end: 20,
                },
            ],
            expected_relations: vec![TrainingRelation {
                subject: "Juan".to_string(),
                predicate: "moved_to".to_string(),
                object: "Berlin".to_string(),
                confidence: 0.88,
            }],
        }];

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("sft.jsonl");

        write_sft_jsonl(&samples, &path).unwrap();

        let content = std::fs::read_to_string(&path).unwrap();
        let line: serde_json::Value =
            serde_json::from_str(content.lines().next().unwrap()).unwrap();

        let messages = line.get("messages").unwrap().as_array().unwrap();
        assert_eq!(messages.len(), 3);
        assert_eq!(messages[0]["role"], "system");
        assert_eq!(messages[1]["role"], "user");
        assert_eq!(messages[2]["role"], "assistant");
    }

    #[test]
    fn test_parse_jsonl_empty() {
        let result = parse_jsonl("").unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_parse_jsonl_with_blank_lines() {
        let content = r#"{"text":"a","entities":[],"expected_relations":[]}

{"text":"b","entities":[],"expected_relations":[]}
"#;
        let result = parse_jsonl(content).unwrap();
        assert_eq!(result.len(), 2);
    }

    // --- Helper tests ---

    #[test]
    fn test_is_relation_edge() {
        assert!(is_relation_edge(EdgeType::RelatesTo));
        assert!(is_relation_edge(EdgeType::Location));
        assert!(is_relation_edge(EdgeType::Actor));
        assert!(is_relation_edge(EdgeType::HasProperty));
        assert!(!is_relation_edge(EdgeType::ConflictsWith));
        assert!(!is_relation_edge(EdgeType::NextEpisode));
        assert!(!is_relation_edge(EdgeType::Supersedes));
    }

    #[test]
    fn test_edge_type_to_predicate() {
        assert_eq!(edge_type_to_predicate(EdgeType::RelatesTo), "related_to");
        assert_eq!(edge_type_to_predicate(EdgeType::Location), "located_in");
        assert_eq!(edge_type_to_predicate(EdgeType::Actor), "actor_in");
        assert_eq!(edge_type_to_predicate(EdgeType::CausedBy), "caused_by");
    }

    #[test]
    fn test_extract_entity_name() {
        assert_eq!(
            extract_entity_name("Juan lives in Madrid. He is 30."),
            "Juan lives in Madrid"
        );
        assert_eq!(extract_entity_name("Short name"), "Short name");
        assert_eq!(extract_entity_name("  Trimmed  "), "Trimmed");
    }

    #[test]
    fn test_extract_entity_name_long() {
        let long = "A".repeat(100);
        let name = extract_entity_name(&long);
        assert!(name.len() <= 60);
    }

    #[test]
    fn test_deterministic_shuffle() {
        let mut a = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let mut b = a.clone();

        deterministic_shuffle(&mut a, 42);
        deterministic_shuffle(&mut b, 42);

        assert_eq!(a, b, "Same seed should produce same shuffle");

        let mut c = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        deterministic_shuffle(&mut c, 99);
        assert_ne!(a, c, "Different seeds should produce different shuffles");
    }

    #[test]
    fn test_deterministic_shuffle_empty() {
        let mut empty: Vec<i32> = vec![];
        deterministic_shuffle(&mut empty, 42);
        assert!(empty.is_empty());
    }

    #[test]
    fn test_deterministic_shuffle_single() {
        let mut single = vec![42];
        deterministic_shuffle(&mut single, 42);
        assert_eq!(single, vec![42]);
    }

    #[test]
    fn test_node_type_to_label() {
        assert_eq!(node_type_to_label(&NodeType::Entity), "entity");
        assert_eq!(node_type_to_label(&NodeType::Event), "event");
        assert_eq!(node_type_to_label(&NodeType::Fact), "fact");
        assert_eq!(node_type_to_label(&NodeType::Skill), "skill");
    }

    #[test]
    fn test_text_truncation() {
        let source = make_entity_node(
            1,
            &"Word ".repeat(500), // 2500 chars
        );
        let target = make_entity_node(2, "Target");
        let edge = make_edge(1, 2, EdgeType::RelatesTo, 0.5);

        let mut node_map = HashMap::new();
        node_map.insert(1u64, &source);
        node_map.insert(2u64, &target);

        let edges_vec = [&edge];
        let edges_ref: Vec<&&Edge> = edges_vec.iter().collect();

        let config = DatasetConfig {
            max_text_length: 100,
            ..Default::default()
        };

        let sample = build_sample(&source, &edges_ref, &node_map, &config);
        assert!(sample.is_some());
        assert!(sample.unwrap().text.len() <= 100);
    }
}
