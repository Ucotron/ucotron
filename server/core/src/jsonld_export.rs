//! # JSON-LD Export/Import for Ucotron Memory Graphs
//!
//! Defines a universal JSON-LD export format for Ucotron memory graphs.
//! Supports full and incremental exports, optional embedding vectors,
//! and optional gzip compression.
//!
//! The format follows JSON-LD conventions with a custom `@context` mapping
//! Ucotron types to schema.org and custom vocabulary.
//!
//! # Example JSON-LD Output
//!
//! ```json
//! {
//!   "@context": { ... },
//!   "@type": "ucotron:MemoryGraph",
//!   "version": "1.0",
//!   "exported_at": 1707700000,
//!   "namespace": "default",
//!   "nodes": [ ... ],
//!   "edges": [ ... ],
//!   "communities": { ... }
//! }
//! ```

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::{Edge, EdgeType, Node, NodeId, NodeType, Value};

/// The JSON-LD context mapping for Ucotron types.
const UCOTRON_CONTEXT_URL: &str = "https://ucotron.com/schema/v1";

// ---------------------------------------------------------------------------
// Export types
// ---------------------------------------------------------------------------

/// Complete JSON-LD export of a Ucotron memory graph.
#[derive(Debug, Serialize, Deserialize)]
pub struct MemoryGraphExport {
    /// JSON-LD context.
    #[serde(rename = "@context")]
    pub context: serde_json::Value,

    /// Type identifier.
    #[serde(rename = "@type")]
    pub graph_type: String,

    /// Export format version.
    pub version: String,

    /// Unix timestamp when the export was created.
    pub exported_at: u64,

    /// Source namespace.
    pub namespace: String,

    /// Exported nodes.
    pub nodes: Vec<ExportNode>,

    /// Exported edges.
    pub edges: Vec<ExportEdge>,

    /// Community assignments (node_id → community_id).
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub communities: HashMap<String, u64>,

    /// Statistics about the export.
    pub stats: ExportStats,
}

/// A node in the JSON-LD export format.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ExportNode {
    /// Node ID (as string for JSON-LD compatibility).
    #[serde(rename = "@id")]
    pub id: String,

    /// Node type (Entity, Event, Fact, Skill).
    #[serde(rename = "@type")]
    pub node_type: String,

    /// Text content.
    pub content: String,

    /// Unix timestamp.
    pub timestamp: u64,

    /// Metadata key-value pairs.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub metadata: HashMap<String, serde_json::Value>,

    /// Embedding vector (optional — can be excluded to reduce export size).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embedding: Option<Vec<f32>>,
}

/// An edge in the JSON-LD export format.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ExportEdge {
    /// Source node ID (as string).
    pub source: String,

    /// Target node ID (as string).
    pub target: String,

    /// Edge type name.
    pub edge_type: String,

    /// Edge weight.
    pub weight: f32,

    /// Edge metadata.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Statistics about an export.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ExportStats {
    pub total_nodes: usize,
    pub total_edges: usize,
    pub has_embeddings: bool,
    pub is_incremental: bool,
    /// If incremental, the timestamp from which nodes/edges were exported.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub from_timestamp: Option<u64>,
}

/// Options controlling what gets included in the export.
#[derive(Debug, Clone)]
pub struct ExportOptions {
    /// Include embedding vectors in the export (can be large).
    pub include_embeddings: bool,
    /// If set, only export nodes/edges with timestamp >= this value.
    pub from_timestamp: Option<u64>,
    /// Namespace to export (nodes without this namespace are skipped).
    pub namespace: String,
}

impl Default for ExportOptions {
    fn default() -> Self {
        Self {
            include_embeddings: true,
            from_timestamp: None,
            namespace: "default".to_string(),
        }
    }
}

// ---------------------------------------------------------------------------
// Conversion: Core types → Export types
// ---------------------------------------------------------------------------

/// Build the JSON-LD `@context` object.
fn build_context() -> serde_json::Value {
    serde_json::json!({
        "ucotron": UCOTRON_CONTEXT_URL,
        "schema": "https://schema.org/",
        "@vocab": UCOTRON_CONTEXT_URL,
        "content": "schema:text",
        "timestamp": "schema:dateCreated",
        "Entity": "schema:Thing",
        "Event": "schema:Event",
        "Fact": "ucotron:Fact",
        "Skill": "ucotron:Skill",
        "RelatesTo": "schema:relatedTo",
        "CausedBy": "schema:causeOf",
        "ConflictsWith": "ucotron:conflictsWith",
        "Supersedes": "ucotron:supersedes",
        "Actor": "schema:actor",
        "Object": "schema:object",
        "Location": "schema:location",
        "Companion": "ucotron:companion"
    })
}

/// Convert a core `NodeType` to its string representation.
fn node_type_to_string(nt: &NodeType) -> String {
    match nt {
        NodeType::Entity => "Entity".to_string(),
        NodeType::Event => "Event".to_string(),
        NodeType::Fact => "Fact".to_string(),
        NodeType::Skill => "Skill".to_string(),
    }
}

/// Convert a string back to a core `NodeType`.
fn string_to_node_type(s: &str) -> NodeType {
    match s {
        "Entity" => NodeType::Entity,
        "Event" => NodeType::Event,
        "Fact" => NodeType::Fact,
        "Skill" => NodeType::Skill,
        _ => NodeType::Entity,
    }
}

/// Convert a core `EdgeType` to its string representation.
fn edge_type_to_string(et: &EdgeType) -> String {
    match et {
        EdgeType::RelatesTo => "RelatesTo".to_string(),
        EdgeType::CausedBy => "CausedBy".to_string(),
        EdgeType::ConflictsWith => "ConflictsWith".to_string(),
        EdgeType::NextEpisode => "NextEpisode".to_string(),
        EdgeType::HasProperty => "HasProperty".to_string(),
        EdgeType::Supersedes => "Supersedes".to_string(),
        EdgeType::Actor => "Actor".to_string(),
        EdgeType::Object => "Object".to_string(),
        EdgeType::Location => "Location".to_string(),
        EdgeType::Companion => "Companion".to_string(),
    }
}

/// Convert a string back to a core `EdgeType`.
fn string_to_edge_type(s: &str) -> EdgeType {
    match s {
        "RelatesTo" => EdgeType::RelatesTo,
        "CausedBy" => EdgeType::CausedBy,
        "ConflictsWith" => EdgeType::ConflictsWith,
        "NextEpisode" => EdgeType::NextEpisode,
        "HasProperty" => EdgeType::HasProperty,
        "Supersedes" => EdgeType::Supersedes,
        "Actor" => EdgeType::Actor,
        "Object" => EdgeType::Object,
        "Location" => EdgeType::Location,
        "Companion" => EdgeType::Companion,
        _ => EdgeType::RelatesTo,
    }
}

/// Convert a core `Value` to JSON.
fn value_to_json(val: &Value) -> serde_json::Value {
    match val {
        Value::String(s) => serde_json::Value::String(s.clone()),
        Value::Integer(i) => serde_json::json!(*i),
        Value::Float(f) => serde_json::json!(*f),
        Value::Bool(b) => serde_json::Value::Bool(*b),
    }
}

/// Convert a JSON value back to a core `Value`.
fn json_to_value(v: &serde_json::Value) -> Value {
    match v {
        serde_json::Value::String(s) => Value::String(s.clone()),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Value::Integer(i)
            } else {
                Value::Float(n.as_f64().unwrap_or(0.0))
            }
        }
        serde_json::Value::Bool(b) => Value::Bool(*b),
        _ => Value::String(v.to_string()),
    }
}

/// Check if a node belongs to the given namespace.
fn node_in_namespace(node: &Node, namespace: &str) -> bool {
    match node.metadata.get("_namespace") {
        Some(Value::String(ns)) => ns == namespace,
        None => namespace == "default",
        _ => namespace == "default",
    }
}

/// Convert a core `Node` to an `ExportNode`.
fn node_to_export(node: &Node, include_embeddings: bool) -> ExportNode {
    let metadata: HashMap<String, serde_json::Value> = node
        .metadata
        .iter()
        .filter(|(k, _)| !k.starts_with('_')) // Skip internal metadata
        .map(|(k, v)| (k.clone(), value_to_json(v)))
        .collect();

    ExportNode {
        id: format!("ucotron:node/{}", node.id),
        node_type: node_type_to_string(&node.node_type),
        content: node.content.clone(),
        timestamp: node.timestamp,
        metadata,
        embedding: if include_embeddings {
            Some(node.embedding.clone())
        } else {
            None
        },
    }
}

/// Convert a core `Edge` to an `ExportEdge`.
fn edge_to_export(edge: &Edge) -> ExportEdge {
    let metadata: HashMap<String, serde_json::Value> = edge
        .metadata
        .iter()
        .map(|(k, v)| (k.clone(), value_to_json(v)))
        .collect();

    ExportEdge {
        source: format!("ucotron:node/{}", edge.source),
        target: format!("ucotron:node/{}", edge.target),
        edge_type: edge_type_to_string(&edge.edge_type),
        weight: edge.weight,
        metadata,
    }
}

// ---------------------------------------------------------------------------
// Export function
// ---------------------------------------------------------------------------

/// Export nodes and edges to a `MemoryGraphExport` JSON-LD structure.
///
/// Filters nodes by namespace and optionally by timestamp.
pub fn export_graph(
    nodes: &[Node],
    edges: &[Edge],
    options: &ExportOptions,
) -> MemoryGraphExport {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    // Filter nodes by namespace and optional timestamp.
    let filtered_nodes: Vec<&Node> = nodes
        .iter()
        .filter(|n| node_in_namespace(n, &options.namespace))
        .filter(|n| {
            if let Some(from_ts) = options.from_timestamp {
                n.timestamp >= from_ts
            } else {
                true
            }
        })
        .collect();

    // Collect node IDs for edge filtering.
    let node_ids: std::collections::HashSet<NodeId> =
        filtered_nodes.iter().map(|n| n.id).collect();

    // Filter edges to only those between exported nodes.
    let filtered_edges: Vec<&Edge> = edges
        .iter()
        .filter(|e| node_ids.contains(&e.source) && node_ids.contains(&e.target))
        .collect();

    let export_nodes: Vec<ExportNode> = filtered_nodes
        .iter()
        .map(|n| node_to_export(n, options.include_embeddings))
        .collect();

    let export_edges: Vec<ExportEdge> = filtered_edges
        .iter()
        .map(|e| edge_to_export(e))
        .collect();

    let stats = ExportStats {
        total_nodes: export_nodes.len(),
        total_edges: export_edges.len(),
        has_embeddings: options.include_embeddings,
        is_incremental: options.from_timestamp.is_some(),
        from_timestamp: options.from_timestamp,
    };

    MemoryGraphExport {
        context: build_context(),
        graph_type: "ucotron:MemoryGraph".to_string(),
        version: "1.0".to_string(),
        exported_at: now,
        namespace: options.namespace.clone(),
        nodes: export_nodes,
        edges: export_edges,
        communities: HashMap::new(),
        stats,
    }
}

/// Serialize a `MemoryGraphExport` to a JSON string.
pub fn export_to_json(export: &MemoryGraphExport) -> anyhow::Result<String> {
    serde_json::to_string_pretty(export).map_err(|e| anyhow::anyhow!("JSON serialization failed: {}", e))
}

// ---------------------------------------------------------------------------
// Import function
// ---------------------------------------------------------------------------

/// Result of importing a JSON-LD export back into core types.
#[derive(Debug)]
pub struct ImportResult {
    /// Imported nodes with remapped IDs.
    pub nodes: Vec<Node>,
    /// Imported edges with remapped IDs.
    pub edges: Vec<Edge>,
    /// Mapping from original export IDs to new IDs.
    pub id_mapping: HashMap<NodeId, NodeId>,
    /// Number of nodes imported.
    pub nodes_imported: usize,
    /// Number of edges imported.
    pub edges_imported: usize,
}

/// Parse a node ID from the JSON-LD `@id` format "ucotron:node/{id}".
fn parse_node_id(id_str: &str) -> Option<NodeId> {
    id_str.strip_prefix("ucotron:node/").and_then(|s| s.parse().ok())
}

/// Import a JSON-LD export, remapping node IDs starting from `next_id`.
///
/// Returns the imported nodes and edges with fresh IDs, plus the ID mapping.
pub fn import_graph(
    export: &MemoryGraphExport,
    next_id: NodeId,
    target_namespace: &str,
) -> anyhow::Result<ImportResult> {
    let mut id_mapping: HashMap<NodeId, NodeId> = HashMap::new();
    let mut current_id = next_id;

    // Build ID mapping.
    for export_node in &export.nodes {
        if let Some(original_id) = parse_node_id(&export_node.id) {
            id_mapping.insert(original_id, current_id);
            current_id += 1;
        }
    }

    // Convert nodes.
    let mut nodes = Vec::with_capacity(export.nodes.len());
    for export_node in &export.nodes {
        let original_id = parse_node_id(&export_node.id)
            .ok_or_else(|| anyhow::anyhow!("Invalid node ID: {}", export_node.id))?;
        let new_id = *id_mapping
            .get(&original_id)
            .ok_or_else(|| anyhow::anyhow!("Missing ID mapping for {}", original_id))?;

        let mut metadata: HashMap<String, Value> = export_node
            .metadata
            .iter()
            .map(|(k, v)| (k.clone(), json_to_value(v)))
            .collect();

        // Tag with target namespace.
        metadata.insert(
            "_namespace".to_string(),
            Value::String(target_namespace.to_string()),
        );

        nodes.push(Node {
            id: new_id,
            content: export_node.content.clone(),
            embedding: export_node.embedding.clone().unwrap_or_else(|| vec![0.0f32; 384]),
            metadata,
            node_type: string_to_node_type(&export_node.node_type),
            timestamp: export_node.timestamp,
            media_type: None,
            media_uri: None,
            embedding_visual: None,
            timestamp_range: None,
            parent_video_id: None,
        });
    }

    // Convert edges.
    let mut edges = Vec::with_capacity(export.edges.len());
    for export_edge in &export.edges {
        let original_source = parse_node_id(&export_edge.source)
            .ok_or_else(|| anyhow::anyhow!("Invalid source ID: {}", export_edge.source))?;
        let original_target = parse_node_id(&export_edge.target)
            .ok_or_else(|| anyhow::anyhow!("Invalid target ID: {}", export_edge.target))?;

        // Skip edges whose nodes were not in the export.
        let new_source = match id_mapping.get(&original_source) {
            Some(id) => *id,
            None => continue,
        };
        let new_target = match id_mapping.get(&original_target) {
            Some(id) => *id,
            None => continue,
        };

        let metadata: HashMap<String, Value> = export_edge
            .metadata
            .iter()
            .map(|(k, v)| (k.clone(), json_to_value(v)))
            .collect();

        edges.push(Edge {
            source: new_source,
            target: new_target,
            edge_type: string_to_edge_type(&export_edge.edge_type),
            weight: export_edge.weight,
            metadata,
        });
    }

    let nodes_imported = nodes.len();
    let edges_imported = edges.len();

    Ok(ImportResult {
        nodes,
        edges,
        id_mapping,
        nodes_imported,
        edges_imported,
    })
}

/// Parse a JSON string into a `MemoryGraphExport`.
pub fn import_from_json(json: &str) -> anyhow::Result<MemoryGraphExport> {
    serde_json::from_str(json).map_err(|e| anyhow::anyhow!("JSON parse failed: {}", e))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_node(id: NodeId, content: &str, node_type: NodeType, namespace: &str) -> Node {
        let mut metadata = HashMap::new();
        metadata.insert(
            "_namespace".to_string(),
            Value::String(namespace.to_string()),
        );
        metadata.insert("tag".to_string(), Value::String("test".to_string()));
        Node {
            id,
            content: content.to_string(),
            embedding: vec![0.1, 0.2, 0.3],
            metadata,
            node_type,
            timestamp: 1700000000 + id,
            media_type: None,
            media_uri: None,
            embedding_visual: None,
            timestamp_range: None,
            parent_video_id: None,
        }
    }

    fn make_test_edge(source: NodeId, target: NodeId, edge_type: EdgeType) -> Edge {
        Edge {
            source,
            target,
            edge_type,
            weight: 0.8,
            metadata: HashMap::new(),
        }
    }

    #[test]
    fn test_export_basic() {
        let nodes = vec![
            make_test_node(1, "Alice", NodeType::Entity, "default"),
            make_test_node(2, "Bob", NodeType::Entity, "default"),
        ];
        let edges = vec![make_test_edge(1, 2, EdgeType::RelatesTo)];
        let options = ExportOptions::default();

        let export = export_graph(&nodes, &edges, &options);

        assert_eq!(export.graph_type, "ucotron:MemoryGraph");
        assert_eq!(export.version, "1.0");
        assert_eq!(export.namespace, "default");
        assert_eq!(export.nodes.len(), 2);
        assert_eq!(export.edges.len(), 1);
        assert_eq!(export.stats.total_nodes, 2);
        assert_eq!(export.stats.total_edges, 1);
        assert!(export.stats.has_embeddings);
        assert!(!export.stats.is_incremental);
    }

    #[test]
    fn test_export_namespace_filtering() {
        let nodes = vec![
            make_test_node(1, "Alice", NodeType::Entity, "ns_a"),
            make_test_node(2, "Bob", NodeType::Entity, "ns_b"),
            make_test_node(3, "Carol", NodeType::Entity, "ns_a"),
        ];
        let edges = vec![
            make_test_edge(1, 3, EdgeType::RelatesTo),
            make_test_edge(1, 2, EdgeType::RelatesTo),
        ];
        let options = ExportOptions {
            namespace: "ns_a".to_string(),
            ..Default::default()
        };

        let export = export_graph(&nodes, &edges, &options);

        assert_eq!(export.nodes.len(), 2);
        assert_eq!(export.edges.len(), 1); // Only edge 1→3 (both in ns_a)
        assert_eq!(export.namespace, "ns_a");
    }

    #[test]
    fn test_export_incremental_timestamp() {
        let nodes = vec![
            make_test_node(1, "Old node", NodeType::Entity, "default"),
            make_test_node(100, "New node", NodeType::Entity, "default"),
        ];
        let edges = vec![];
        let options = ExportOptions {
            from_timestamp: Some(1700000050),
            ..Default::default()
        };

        let export = export_graph(&nodes, &edges, &options);

        assert_eq!(export.nodes.len(), 1);
        assert!(export.stats.is_incremental);
        assert_eq!(export.stats.from_timestamp, Some(1700000050));
    }

    #[test]
    fn test_export_without_embeddings() {
        let nodes = vec![make_test_node(1, "Alice", NodeType::Entity, "default")];
        let options = ExportOptions {
            include_embeddings: false,
            ..Default::default()
        };

        let export = export_graph(&nodes, &[], &options);

        assert!(!export.stats.has_embeddings);
        assert!(export.nodes[0].embedding.is_none());
    }

    #[test]
    fn test_export_with_embeddings() {
        let nodes = vec![make_test_node(1, "Alice", NodeType::Entity, "default")];
        let options = ExportOptions::default();

        let export = export_graph(&nodes, &[], &options);

        assert!(export.stats.has_embeddings);
        assert!(export.nodes[0].embedding.is_some());
        assert_eq!(export.nodes[0].embedding.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn test_export_json_roundtrip() {
        let nodes = vec![
            make_test_node(1, "Alice", NodeType::Entity, "default"),
            make_test_node(2, "Meeting", NodeType::Event, "default"),
        ];
        let edges = vec![make_test_edge(1, 2, EdgeType::Actor)];
        let options = ExportOptions::default();

        let export = export_graph(&nodes, &edges, &options);
        let json = export_to_json(&export).unwrap();

        // Parse back.
        let reimported: MemoryGraphExport = import_from_json(&json).unwrap();

        assert_eq!(reimported.nodes.len(), 2);
        assert_eq!(reimported.edges.len(), 1);
        assert_eq!(reimported.version, "1.0");
        assert_eq!(reimported.namespace, "default");
    }

    #[test]
    fn test_import_remaps_ids() {
        let nodes = vec![
            make_test_node(1, "Alice", NodeType::Entity, "default"),
            make_test_node(2, "Bob", NodeType::Entity, "default"),
        ];
        let edges = vec![make_test_edge(1, 2, EdgeType::RelatesTo)];
        let options = ExportOptions::default();

        let export = export_graph(&nodes, &edges, &options);
        let result = import_graph(&export, 100, "imported").unwrap();

        assert_eq!(result.nodes.len(), 2);
        assert_eq!(result.edges.len(), 1);

        // IDs should be remapped starting from 100.
        assert_eq!(result.nodes[0].id, 100);
        assert_eq!(result.nodes[1].id, 101);
        assert_eq!(result.edges[0].source, 100);
        assert_eq!(result.edges[0].target, 101);

        // Namespace should be overridden.
        assert_eq!(
            result.nodes[0].metadata.get("_namespace"),
            Some(&Value::String("imported".to_string()))
        );
    }

    #[test]
    fn test_import_preserves_content_and_types() {
        let nodes = vec![
            make_test_node(1, "Alice", NodeType::Entity, "default"),
            make_test_node(2, "Meeting at park", NodeType::Event, "default"),
            make_test_node(3, "Sky is blue", NodeType::Fact, "default"),
        ];
        let edges = vec![
            make_test_edge(1, 2, EdgeType::Actor),
            make_test_edge(2, 3, EdgeType::HasProperty),
        ];
        let options = ExportOptions::default();

        let export = export_graph(&nodes, &edges, &options);
        let result = import_graph(&export, 500, "test").unwrap();

        assert_eq!(result.nodes[0].content, "Alice");
        assert_eq!(result.nodes[1].content, "Meeting at park");
        assert_eq!(result.nodes[2].content, "Sky is blue");

        assert!(matches!(result.nodes[0].node_type, NodeType::Entity));
        assert!(matches!(result.nodes[1].node_type, NodeType::Event));
        assert!(matches!(result.nodes[2].node_type, NodeType::Fact));

        assert!(matches!(result.edges[0].edge_type, EdgeType::Actor));
        assert!(matches!(result.edges[1].edge_type, EdgeType::HasProperty));
    }

    #[test]
    fn test_import_preserves_embeddings() {
        let nodes = vec![make_test_node(1, "Alice", NodeType::Entity, "default")];
        let options = ExportOptions::default();

        let export = export_graph(&nodes, &[], &options);
        let result = import_graph(&export, 1, "test").unwrap();

        assert_eq!(result.nodes[0].embedding, vec![0.1, 0.2, 0.3]);
    }

    #[test]
    fn test_import_without_embeddings_gets_zeros() {
        let nodes = vec![make_test_node(1, "Alice", NodeType::Entity, "default")];
        let options = ExportOptions {
            include_embeddings: false,
            ..Default::default()
        };

        let export = export_graph(&nodes, &[], &options);
        let result = import_graph(&export, 1, "test").unwrap();

        assert_eq!(result.nodes[0].embedding.len(), 384);
        assert!(result.nodes[0].embedding.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_internal_metadata_stripped_on_export() {
        let nodes = vec![make_test_node(1, "Alice", NodeType::Entity, "default")];
        let options = ExportOptions::default();

        let export = export_graph(&nodes, &[], &options);

        // _namespace should NOT appear in exported metadata.
        assert!(!export.nodes[0].metadata.contains_key("_namespace"));
        // But user metadata should be preserved.
        assert!(export.nodes[0].metadata.contains_key("tag"));
    }

    #[test]
    fn test_export_context_has_required_fields() {
        let export = export_graph(&[], &[], &ExportOptions::default());
        let ctx = &export.context;

        assert!(ctx.get("ucotron").is_some());
        assert!(ctx.get("schema").is_some());
        assert!(ctx.get("Entity").is_some());
        assert!(ctx.get("Event").is_some());
    }

    #[test]
    fn test_node_type_roundtrip() {
        let types = vec![NodeType::Entity, NodeType::Event, NodeType::Fact, NodeType::Skill];
        for nt in types {
            let s = node_type_to_string(&nt);
            let rt = string_to_node_type(&s);
            assert_eq!(std::mem::discriminant(&nt), std::mem::discriminant(&rt));
        }
    }

    #[test]
    fn test_edge_type_roundtrip() {
        let types = vec![
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
        for et in types {
            let s = edge_type_to_string(&et);
            let rt = string_to_edge_type(&s);
            assert_eq!(std::mem::discriminant(&et), std::mem::discriminant(&rt));
        }
    }

    #[test]
    fn test_export_empty_graph() {
        let export = export_graph(&[], &[], &ExportOptions::default());

        assert_eq!(export.nodes.len(), 0);
        assert_eq!(export.edges.len(), 0);
        assert_eq!(export.stats.total_nodes, 0);
        assert_eq!(export.stats.total_edges, 0);
    }

    #[test]
    fn test_import_empty_export() {
        let export = export_graph(&[], &[], &ExportOptions::default());
        let result = import_graph(&export, 1, "test").unwrap();

        assert_eq!(result.nodes.len(), 0);
        assert_eq!(result.edges.len(), 0);
        assert!(result.id_mapping.is_empty());
    }

    #[test]
    fn test_full_roundtrip_integrity() {
        // Create a graph, export, reimport, and verify data integrity.
        let nodes = vec![
            make_test_node(10, "Alice", NodeType::Entity, "myns"),
            make_test_node(20, "Bob", NodeType::Entity, "myns"),
            make_test_node(30, "Lunch event", NodeType::Event, "myns"),
        ];
        let edges = vec![
            make_test_edge(10, 30, EdgeType::Actor),
            make_test_edge(20, 30, EdgeType::Companion),
            make_test_edge(10, 20, EdgeType::RelatesTo),
        ];
        let options = ExportOptions {
            namespace: "myns".to_string(),
            ..Default::default()
        };

        // Export.
        let export = export_graph(&nodes, &edges, &options);
        let json = export_to_json(&export).unwrap();

        // Reimport from JSON.
        let parsed = import_from_json(&json).unwrap();
        let result = import_graph(&parsed, 1000, "reimported").unwrap();

        // Verify counts.
        assert_eq!(result.nodes_imported, 3);
        assert_eq!(result.edges_imported, 3);

        // Verify content preserved.
        let contents: Vec<&str> = result.nodes.iter().map(|n| n.content.as_str()).collect();
        assert!(contents.contains(&"Alice"));
        assert!(contents.contains(&"Bob"));
        assert!(contents.contains(&"Lunch event"));

        // Verify edges are coherent (source and target are valid new IDs).
        let new_ids: std::collections::HashSet<NodeId> = result.nodes.iter().map(|n| n.id).collect();
        for edge in &result.edges {
            assert!(new_ids.contains(&edge.source));
            assert!(new_ids.contains(&edge.target));
        }
    }
}
