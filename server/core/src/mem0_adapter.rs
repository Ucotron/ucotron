//! # Mem0 Import Adapter
//!
//! Converts Mem0 exported memories into the Ucotron `MemoryGraphExport` format
//! for import via the standard import pipeline.
//!
//! Mem0 stores memories as flat text entries with optional metadata, user/agent IDs,
//! and timestamps. This adapter maps them to Ucotron `ExportNode` entries as
//! `Entity` type nodes with `RelatesTo` edges inferred between memories sharing
//! the same user/agent context.
//!
//! ## Supported Mem0 Formats
//!
//! - **v2 (recommended)**: `{ "results": [...], "total_memories": N }`
//! - **v1 (legacy)**: `[...]` (bare array of memory objects)
//! - **Single file export**: `{ "memories": [...] }`
//!
//! ## Memory Object Fields
//!
//! ```json
//! {
//!   "id": "mem_01JF8ZS4Y0R0SPM13R5R6H32CJ",
//!   "memory": "Likes to play cricket on weekends",
//!   "user_id": "alice",
//!   "agent_id": "assistant_v2",
//!   "hash": "abc123",
//!   "metadata": { "category": "hobbies", "source": "onboarding" },
//!   "created_at": "2024-07-01T12:00:00Z",
//!   "updated_at": "2024-07-01T12:00:00Z"
//! }
//! ```

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::jsonld_export::{ExportEdge, ExportNode, ExportStats, MemoryGraphExport};

// ---------------------------------------------------------------------------
// Mem0 types (deserialization)
// ---------------------------------------------------------------------------

/// A single Mem0 memory object.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Mem0Memory {
    /// Unique identifier (e.g., "mem_01JF8ZS4Y0R0SPM13R5R6H32CJ").
    pub id: String,

    /// The memory text content.
    pub memory: String,

    /// User identifier associated with this memory.
    #[serde(default)]
    pub user_id: Option<String>,

    /// Agent identifier.
    #[serde(default)]
    pub agent_id: Option<String>,

    /// Application identifier.
    #[serde(default)]
    pub app_id: Option<String>,

    /// Run identifier.
    #[serde(default)]
    pub run_id: Option<String>,

    /// Content hash.
    #[serde(default)]
    pub hash: Option<String>,

    /// Custom metadata.
    #[serde(default)]
    pub metadata: Option<serde_json::Value>,

    /// ISO 8601 creation timestamp.
    #[serde(default)]
    pub created_at: Option<String>,

    /// ISO 8601 last-update timestamp.
    #[serde(default)]
    pub updated_at: Option<String>,

    /// Conversation input that generated this memory (Mem0 v2).
    #[serde(default)]
    pub input: Option<serde_json::Value>,

    /// Memory classification type (Mem0 v2).
    #[serde(rename = "type", default)]
    pub memory_type: Option<String>,

    /// Owner identifier (Mem0 v2 platform).
    #[serde(default)]
    pub owner: Option<String>,

    /// Organization identifier (Mem0 v2 platform).
    #[serde(default)]
    pub organization: Option<String>,

    /// Whether this memory is immutable (Mem0 v2).
    #[serde(default)]
    pub immutable: Option<bool>,

    /// Expiration date (Mem0 v2, "YYYY-MM-DD").
    #[serde(default)]
    pub expiration_date: Option<String>,
}

/// Mem0 export envelope — wraps an array of memories.
///
/// Supports three formats:
/// - v2: `{ "results": [...], "total_memories": N }`
/// - File export: `{ "memories": [...] }`
/// - v1 bare array: `[...]` (handled separately)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Mem0Export {
    /// Mem0 v2 format: array under "results".
    #[serde(default)]
    pub results: Option<Vec<Mem0Memory>>,

    /// File export format: array under "memories".
    #[serde(default)]
    pub memories: Option<Vec<Mem0Memory>>,

    /// Total count (v2 format).
    #[serde(default)]
    pub total_memories: Option<usize>,
}

/// Options for converting Mem0 data to Ucotron format.
#[derive(Debug, Clone)]
pub struct Mem0ImportOptions {
    /// Target namespace for imported memories.
    pub namespace: String,
    /// Whether to create `RelatesTo` edges between memories that share the same user_id.
    pub link_same_user: bool,
    /// Whether to create `RelatesTo` edges between memories that share the same agent_id.
    pub link_same_agent: bool,
}

impl Default for Mem0ImportOptions {
    fn default() -> Self {
        Self {
            namespace: "mem0_import".to_string(),
            link_same_user: true,
            link_same_agent: false,
        }
    }
}

/// Result of parsing Mem0 data.
#[derive(Debug)]
pub struct Mem0ParseResult {
    /// Number of memories parsed.
    pub memories_parsed: usize,
    /// Number of edges inferred.
    pub edges_inferred: usize,
    /// The resulting Ucotron export document.
    pub export: MemoryGraphExport,
}

// ---------------------------------------------------------------------------
// Parsing
// ---------------------------------------------------------------------------

/// Parse Mem0 JSON data from a string.
///
/// Supports three formats:
/// 1. v2 object with `results` key
/// 2. Object with `memories` key
/// 3. Bare array of memory objects (v1)
pub fn parse_mem0_json(json: &str) -> anyhow::Result<Vec<Mem0Memory>> {
    // Try v2/object format first.
    if let Ok(envelope) = serde_json::from_str::<Mem0Export>(json) {
        if let Some(results) = envelope.results {
            return Ok(results);
        }
        if let Some(memories) = envelope.memories {
            return Ok(memories);
        }
    }

    // Try bare array (v1 format).
    if let Ok(memories) = serde_json::from_str::<Vec<Mem0Memory>>(json) {
        return Ok(memories);
    }

    // Try single memory object.
    if let Ok(memory) = serde_json::from_str::<Mem0Memory>(json) {
        return Ok(vec![memory]);
    }

    Err(anyhow::anyhow!(
        "Failed to parse Mem0 data: expected v2 object with 'results', \
         object with 'memories', bare array, or single memory object"
    ))
}

// ---------------------------------------------------------------------------
// Conversion
// ---------------------------------------------------------------------------

/// Parse an ISO 8601 timestamp string to a Unix epoch (seconds).
///
/// Handles common formats:
/// - "2024-07-01T12:00:00Z"
/// - "2024-07-01T12:00:00+00:00"
/// - "2024-07-01T12:00:00.000Z"
/// - "2024-07-01" (date only → midnight UTC)
fn parse_iso_timestamp(ts: &str) -> u64 {
    // Try full ISO 8601 with timezone.
    // Simple manual parser to avoid adding chrono dependency.
    let trimmed = ts.trim();

    // Extract date part (YYYY-MM-DD).
    if trimmed.len() < 10 {
        return 0;
    }
    let date_part = &trimmed[..10];
    let parts: Vec<&str> = date_part.split('-').collect();
    if parts.len() != 3 {
        return 0;
    }

    let year: i64 = parts[0].parse().unwrap_or(0);
    let month: u32 = parts[1].parse().unwrap_or(1);
    let day: u32 = parts[2].parse().unwrap_or(1);

    // Days from epoch (1970-01-01) — simplified calculation.
    let mut total_days: i64 = 0;
    for y in 1970..year {
        total_days += if is_leap_year(y as u32) { 366 } else { 365 };
    }
    let days_in_months = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
    for m in 1..month {
        total_days += days_in_months[m as usize] as i64;
        if m == 2 && is_leap_year(year as u32) {
            total_days += 1;
        }
    }
    total_days += (day as i64) - 1;

    let mut epoch = total_days * 86400;

    // Extract time part if present (HH:MM:SS).
    if trimmed.len() > 11 {
        let time_start = if trimmed.as_bytes()[10] == b'T' {
            11
        } else {
            10
        };
        // Take up to 8 chars for HH:MM:SS, ignoring fractional seconds and timezone.
        let time_str: String = trimmed[time_start..]
            .chars()
            .take_while(|c| c.is_ascii_digit() || *c == ':')
            .collect();
        let time_parts: Vec<&str> = time_str.split(':').collect();
        if time_parts.len() >= 2 {
            let hours: i64 = time_parts[0].parse().unwrap_or(0);
            let minutes: i64 = time_parts[1].parse().unwrap_or(0);
            let seconds: i64 = time_parts.get(2).and_then(|s| s.parse().ok()).unwrap_or(0);
            epoch += hours * 3600 + minutes * 60 + seconds;
        }
    }

    if epoch < 0 {
        0
    } else {
        epoch as u64
    }
}

fn is_leap_year(year: u32) -> bool {
    (year.is_multiple_of(4) && !year.is_multiple_of(100)) || year.is_multiple_of(400)
}

/// Convert a serde_json::Value metadata object to a flat HashMap<String, serde_json::Value>.
fn flatten_metadata(val: &serde_json::Value) -> HashMap<String, serde_json::Value> {
    match val {
        serde_json::Value::Object(map) => map.iter().map(|(k, v)| (k.clone(), v.clone())).collect(),
        _ => HashMap::new(),
    }
}

/// Convert a collection of Mem0 memories to a `MemoryGraphExport`.
///
/// Each Mem0 memory becomes an `ExportNode` of type `Entity`.
/// Edges are inferred between memories sharing the same `user_id`
/// (and optionally `agent_id`) using `RelatesTo` edge type.
pub fn mem0_to_ucotron(memories: &[Mem0Memory], options: &Mem0ImportOptions) -> Mem0ParseResult {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let mut nodes: Vec<ExportNode> = Vec::with_capacity(memories.len());
    let mut edges: Vec<ExportEdge> = Vec::new();

    // Track memories by user_id and agent_id for edge inference.
    let mut by_user: HashMap<String, Vec<usize>> = HashMap::new();
    let mut by_agent: HashMap<String, Vec<usize>> = HashMap::new();

    for (idx, mem) in memories.iter().enumerate() {
        // Build metadata.
        let mut metadata: HashMap<String, serde_json::Value> = HashMap::new();

        // Preserve Mem0-specific fields as metadata.
        if let Some(ref uid) = mem.user_id {
            metadata.insert("mem0_user_id".to_string(), serde_json::json!(uid));
        }
        if let Some(ref aid) = mem.agent_id {
            metadata.insert("mem0_agent_id".to_string(), serde_json::json!(aid));
        }
        if let Some(ref app) = mem.app_id {
            metadata.insert("mem0_app_id".to_string(), serde_json::json!(app));
        }
        if let Some(ref run) = mem.run_id {
            metadata.insert("mem0_run_id".to_string(), serde_json::json!(run));
        }
        if let Some(ref hash) = mem.hash {
            metadata.insert("mem0_hash".to_string(), serde_json::json!(hash));
        }
        if let Some(ref owner) = mem.owner {
            metadata.insert("mem0_owner".to_string(), serde_json::json!(owner));
        }
        if let Some(ref org) = mem.organization {
            metadata.insert("mem0_organization".to_string(), serde_json::json!(org));
        }
        if let Some(ref mtype) = mem.memory_type {
            metadata.insert("mem0_type".to_string(), serde_json::json!(mtype));
        }
        if let Some(immutable) = mem.immutable {
            metadata.insert("mem0_immutable".to_string(), serde_json::json!(immutable));
        }
        if let Some(ref exp) = mem.expiration_date {
            metadata.insert("mem0_expiration_date".to_string(), serde_json::json!(exp));
        }
        if let Some(ref updated) = mem.updated_at {
            metadata.insert("mem0_updated_at".to_string(), serde_json::json!(updated));
        }

        // Merge user-provided metadata.
        if let Some(ref meta_val) = mem.metadata {
            for (k, v) in flatten_metadata(meta_val) {
                metadata.insert(format!("mem0_meta_{}", k), v);
            }
        }

        // Origin marker.
        metadata.insert("_import_source".to_string(), serde_json::json!("mem0"));
        metadata.insert("mem0_original_id".to_string(), serde_json::json!(mem.id));

        // Compute timestamp.
        let timestamp = mem
            .created_at
            .as_deref()
            .map(parse_iso_timestamp)
            .unwrap_or(now);

        // Use sequential numeric IDs for the export (import_graph will remap).
        let node_id = idx as u64 + 1;

        nodes.push(ExportNode {
            id: format!("ucotron:node/{}", node_id),
            node_type: "Entity".to_string(),
            content: mem.memory.clone(),
            timestamp,
            metadata,
            embedding: None, // Embeddings will be generated on import by the server.
        });

        // Track by user/agent for edge inference.
        if options.link_same_user {
            if let Some(ref uid) = mem.user_id {
                by_user.entry(uid.clone()).or_default().push(idx);
            }
        }
        if options.link_same_agent {
            if let Some(ref aid) = mem.agent_id {
                by_agent.entry(aid.clone()).or_default().push(idx);
            }
        }
    }

    // Infer edges: connect memories within the same user/agent group.
    // Use a chain pattern: mem[0] → mem[1] → mem[2] → ... (sorted by timestamp).
    let mut infer_chain_edges = |groups: &HashMap<String, Vec<usize>>| {
        for indices in groups.values() {
            if indices.len() < 2 {
                continue;
            }
            // Sort by timestamp to create temporal chains.
            let mut sorted: Vec<usize> = indices.clone();
            sorted.sort_by_key(|&i| {
                memories[i]
                    .created_at
                    .as_deref()
                    .map(parse_iso_timestamp)
                    .unwrap_or(0)
            });

            for window in sorted.windows(2) {
                let src_id = window[0] as u64 + 1;
                let tgt_id = window[1] as u64 + 1;
                edges.push(ExportEdge {
                    source: format!("ucotron:node/{}", src_id),
                    target: format!("ucotron:node/{}", tgt_id),
                    edge_type: "RelatesTo".to_string(),
                    weight: 0.5,
                    metadata: HashMap::new(),
                });
            }
        }
    };

    infer_chain_edges(&by_user);
    infer_chain_edges(&by_agent);

    let memories_parsed = nodes.len();
    let edges_inferred = edges.len();

    let stats = ExportStats {
        total_nodes: nodes.len(),
        total_edges: edges.len(),
        has_embeddings: false,
        is_incremental: false,
        from_timestamp: None,
    };

    let context = serde_json::json!({
        "ucotron": "https://ucotron.com/schema/v1",
        "schema": "https://schema.org/",
        "@vocab": "https://ucotron.com/schema/v1",
        "mem0": "https://docs.mem0.ai/schema/v1"
    });

    let export = MemoryGraphExport {
        context,
        graph_type: "ucotron:MemoryGraph".to_string(),
        version: "1.0".to_string(),
        exported_at: now,
        namespace: options.namespace.clone(),
        nodes,
        edges,
        communities: HashMap::new(),
        stats,
    };

    Mem0ParseResult {
        memories_parsed,
        edges_inferred,
        export,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_v2_format() {
        let json = r#"{
            "results": [
                {
                    "id": "mem_001",
                    "memory": "Likes coffee",
                    "user_id": "alice",
                    "created_at": "2024-07-01T12:00:00Z"
                },
                {
                    "id": "mem_002",
                    "memory": "Works at Google",
                    "user_id": "alice",
                    "created_at": "2024-07-02T10:00:00Z"
                }
            ],
            "total_memories": 2
        }"#;

        let memories = parse_mem0_json(json).unwrap();
        assert_eq!(memories.len(), 2);
        assert_eq!(memories[0].id, "mem_001");
        assert_eq!(memories[0].memory, "Likes coffee");
        assert_eq!(memories[1].id, "mem_002");
    }

    #[test]
    fn test_parse_memories_format() {
        let json = r#"{
            "memories": [
                {
                    "id": "mem_100",
                    "memory": "Prefers dark mode",
                    "user_id": "bob"
                }
            ]
        }"#;

        let memories = parse_mem0_json(json).unwrap();
        assert_eq!(memories.len(), 1);
        assert_eq!(memories[0].memory, "Prefers dark mode");
    }

    #[test]
    fn test_parse_v1_bare_array() {
        let json = r#"[
            {
                "id": "mem_200",
                "memory": "Lives in Austin",
                "user_id": "charlie",
                "hash": "abc123",
                "metadata": {"source": "onboarding"},
                "created_at": "2024-06-15T08:30:00Z",
                "updated_at": "2024-06-15T08:30:00Z"
            }
        ]"#;

        let memories = parse_mem0_json(json).unwrap();
        assert_eq!(memories.len(), 1);
        assert_eq!(memories[0].hash, Some("abc123".to_string()));
        assert!(memories[0].metadata.is_some());
    }

    #[test]
    fn test_parse_single_memory() {
        let json = r#"{
            "id": "mem_solo",
            "memory": "Speaks Spanish fluently"
        }"#;

        let memories = parse_mem0_json(json).unwrap();
        assert_eq!(memories.len(), 1);
        assert_eq!(memories[0].memory, "Speaks Spanish fluently");
    }

    #[test]
    fn test_parse_invalid_json() {
        let result = parse_mem0_json("not valid json at all");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_empty_results() {
        let json = r#"{"results": [], "total_memories": 0}"#;
        let memories = parse_mem0_json(json).unwrap();
        assert!(memories.is_empty());
    }

    #[test]
    fn test_iso_timestamp_parsing() {
        // 2024-07-01T12:00:00Z → 1719835200
        let ts = parse_iso_timestamp("2024-07-01T12:00:00Z");
        assert_eq!(ts, 1719835200);

        // Date-only.
        let ts2 = parse_iso_timestamp("2024-01-01");
        assert_eq!(ts2, 1704067200);

        // With fractional seconds.
        let ts3 = parse_iso_timestamp("2024-07-01T12:00:00.123Z");
        assert_eq!(ts3, 1719835200);
    }

    #[test]
    fn test_iso_timestamp_edge_cases() {
        assert_eq!(parse_iso_timestamp(""), 0);
        assert_eq!(parse_iso_timestamp("invalid"), 0);
        // 1970-01-01 midnight.
        assert_eq!(parse_iso_timestamp("1970-01-01T00:00:00Z"), 0);
    }

    #[test]
    fn test_convert_basic() {
        let memories = vec![Mem0Memory {
            id: "mem_001".to_string(),
            memory: "Likes coffee".to_string(),
            user_id: Some("alice".to_string()),
            agent_id: None,
            app_id: None,
            run_id: None,
            hash: Some("h1".to_string()),
            metadata: Some(serde_json::json!({"category": "preferences"})),
            created_at: Some("2024-07-01T12:00:00Z".to_string()),
            updated_at: Some("2024-07-01T12:00:00Z".to_string()),
            input: None,
            memory_type: None,
            owner: None,
            organization: None,
            immutable: None,
            expiration_date: None,
        }];

        let options = Mem0ImportOptions::default();
        let result = mem0_to_ucotron(&memories, &options);

        assert_eq!(result.memories_parsed, 1);
        assert_eq!(result.export.nodes.len(), 1);
        assert_eq!(result.export.version, "1.0");

        let node = &result.export.nodes[0];
        assert_eq!(node.content, "Likes coffee");
        assert_eq!(node.node_type, "Entity");
        assert_eq!(node.timestamp, 1719835200);
        assert!(node.embedding.is_none());

        // Check metadata preservation.
        assert_eq!(
            node.metadata.get("mem0_user_id"),
            Some(&serde_json::json!("alice"))
        );
        assert_eq!(
            node.metadata.get("mem0_hash"),
            Some(&serde_json::json!("h1"))
        );
        assert_eq!(
            node.metadata.get("mem0_meta_category"),
            Some(&serde_json::json!("preferences"))
        );
        assert_eq!(
            node.metadata.get("_import_source"),
            Some(&serde_json::json!("mem0"))
        );
    }

    #[test]
    fn test_convert_with_edge_inference() {
        let memories = vec![
            Mem0Memory {
                id: "mem_001".to_string(),
                memory: "Likes coffee".to_string(),
                user_id: Some("alice".to_string()),
                created_at: Some("2024-07-01T10:00:00Z".to_string()),
                ..default_mem0_memory()
            },
            Mem0Memory {
                id: "mem_002".to_string(),
                memory: "Works at Google".to_string(),
                user_id: Some("alice".to_string()),
                created_at: Some("2024-07-02T10:00:00Z".to_string()),
                ..default_mem0_memory()
            },
            Mem0Memory {
                id: "mem_003".to_string(),
                memory: "Lives in NYC".to_string(),
                user_id: Some("bob".to_string()),
                created_at: Some("2024-07-01T10:00:00Z".to_string()),
                ..default_mem0_memory()
            },
        ];

        let options = Mem0ImportOptions::default();
        let result = mem0_to_ucotron(&memories, &options);

        assert_eq!(result.memories_parsed, 3);
        assert_eq!(result.export.nodes.len(), 3);
        // Alice has 2 memories → 1 chain edge; Bob has 1 → 0 edges.
        assert_eq!(result.edges_inferred, 1);
        assert_eq!(result.export.edges.len(), 1);

        let edge = &result.export.edges[0];
        assert_eq!(edge.edge_type, "RelatesTo");
        assert_eq!(edge.weight, 0.5);
    }

    #[test]
    fn test_convert_no_edge_inference() {
        let memories = vec![
            Mem0Memory {
                id: "mem_001".to_string(),
                memory: "Fact 1".to_string(),
                user_id: Some("alice".to_string()),
                ..default_mem0_memory()
            },
            Mem0Memory {
                id: "mem_002".to_string(),
                memory: "Fact 2".to_string(),
                user_id: Some("alice".to_string()),
                ..default_mem0_memory()
            },
        ];

        let options = Mem0ImportOptions {
            link_same_user: false,
            link_same_agent: false,
            ..Default::default()
        };
        let result = mem0_to_ucotron(&memories, &options);

        assert_eq!(result.edges_inferred, 0);
        assert!(result.export.edges.is_empty());
    }

    #[test]
    fn test_convert_empty() {
        let result = mem0_to_ucotron(&[], &Mem0ImportOptions::default());
        assert_eq!(result.memories_parsed, 0);
        assert!(result.export.nodes.is_empty());
        assert!(result.export.edges.is_empty());
    }

    #[test]
    fn test_convert_preserves_v2_fields() {
        let mem = Mem0Memory {
            id: "mem_v2".to_string(),
            memory: "V2 memory".to_string(),
            memory_type: Some("episodic".to_string()),
            owner: Some("owner_123".to_string()),
            organization: Some("org_456".to_string()),
            immutable: Some(true),
            expiration_date: Some("2025-12-31".to_string()),
            ..default_mem0_memory()
        };

        let result = mem0_to_ucotron(&[mem], &Mem0ImportOptions::default());
        let node = &result.export.nodes[0];

        assert_eq!(
            node.metadata.get("mem0_type"),
            Some(&serde_json::json!("episodic"))
        );
        assert_eq!(
            node.metadata.get("mem0_owner"),
            Some(&serde_json::json!("owner_123"))
        );
        assert_eq!(
            node.metadata.get("mem0_organization"),
            Some(&serde_json::json!("org_456"))
        );
        assert_eq!(
            node.metadata.get("mem0_immutable"),
            Some(&serde_json::json!(true))
        );
        assert_eq!(
            node.metadata.get("mem0_expiration_date"),
            Some(&serde_json::json!("2025-12-31"))
        );
    }

    #[test]
    fn test_export_has_correct_version() {
        let result = mem0_to_ucotron(&[], &Mem0ImportOptions::default());
        assert_eq!(result.export.version, "1.0");
        assert_eq!(result.export.graph_type, "ucotron:MemoryGraph");
    }

    #[test]
    fn test_node_ids_are_sequential() {
        let memories: Vec<Mem0Memory> = (0..5)
            .map(|i| Mem0Memory {
                id: format!("mem_{}", i),
                memory: format!("Memory {}", i),
                ..default_mem0_memory()
            })
            .collect();

        let result = mem0_to_ucotron(&memories, &Mem0ImportOptions::default());

        for (i, node) in result.export.nodes.iter().enumerate() {
            let expected_id = format!("ucotron:node/{}", i + 1);
            assert_eq!(node.id, expected_id);
        }
    }

    #[test]
    fn test_full_roundtrip_parse_and_convert() {
        let json = r#"{
            "results": [
                {
                    "id": "mem_rt1",
                    "memory": "User prefers dark mode",
                    "user_id": "alice",
                    "hash": "hash_1",
                    "metadata": {"source": "settings"},
                    "created_at": "2024-01-15T10:30:00Z",
                    "updated_at": "2024-01-15T10:30:00Z"
                },
                {
                    "id": "mem_rt2",
                    "memory": "User is a software engineer",
                    "user_id": "alice",
                    "agent_id": "assistant",
                    "created_at": "2024-01-16T14:00:00Z"
                },
                {
                    "id": "mem_rt3",
                    "memory": "User likes TypeScript",
                    "user_id": "bob",
                    "created_at": "2024-01-17T09:00:00Z"
                }
            ],
            "total_memories": 3
        }"#;

        let memories = parse_mem0_json(json).unwrap();
        assert_eq!(memories.len(), 3);

        let result = mem0_to_ucotron(&memories, &Mem0ImportOptions::default());
        assert_eq!(result.memories_parsed, 3);
        assert_eq!(result.export.nodes.len(), 3);
        // Alice: 2 memories → 1 edge; Bob: 1 → 0.
        assert_eq!(result.edges_inferred, 1);

        // Verify the export is valid JSON-LD.
        let json_export = crate::jsonld_export::export_to_json(&result.export).unwrap();
        let reimported = crate::jsonld_export::import_from_json(&json_export).unwrap();
        assert_eq!(reimported.nodes.len(), 3);
        assert_eq!(reimported.edges.len(), 1);
        assert_eq!(reimported.version, "1.0");
    }

    /// Helper to create a default Mem0Memory for tests.
    fn default_mem0_memory() -> Mem0Memory {
        Mem0Memory {
            id: String::new(),
            memory: String::new(),
            user_id: None,
            agent_id: None,
            app_id: None,
            run_id: None,
            hash: None,
            metadata: None,
            created_at: None,
            updated_at: None,
            input: None,
            memory_type: None,
            owner: None,
            organization: None,
            immutable: None,
            expiration_date: None,
        }
    }
}
