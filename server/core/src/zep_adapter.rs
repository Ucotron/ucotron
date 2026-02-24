//! # Zep/Graphiti Import Adapter
//!
//! Converts Zep and Graphiti exported data into the Ucotron `MemoryGraphExport` format
//! for import via the standard import pipeline.
//!
//! Supports two data models:
//!
//! ## Graphiti (Temporal Knowledge Graph)
//!
//! Graphiti exports contain entity nodes, episode nodes, and edges (facts) with
//! temporal metadata (`valid_at`, `invalid_at`). The adapter maps:
//! - Entity nodes → Ucotron `Entity` nodes (preserving labels, summary, attributes)
//! - Episode nodes → Ucotron `Event` nodes (preserving source, description)
//! - Entity edges (facts) → Ucotron edges with `RelatesTo` type + temporal metadata
//! - Episode edges → Ucotron edges linking events to entities
//!
//! ## Zep v2 (Session-Based Memory)
//!
//! Zep v2 exports contain sessions with messages (role, content), facts, and user IDs.
//! The adapter maps:
//! - Messages → Ucotron `Entity` nodes with role metadata
//! - Facts → Ucotron `Fact` nodes
//! - Temporal ordering within sessions → `NextEpisode` edges
//!
//! ## Supported Formats
//!
//! - **Graphiti export**: `{ "entities": [...], "episodes": [...], "edges": [...] }`
//! - **Zep sessions**: `{ "sessions": [...] }` where each session has `messages`
//! - **Zep facts array**: `{ "facts": [...] }`
//! - **Bare array of sessions**: `[{ "session_id": ..., "messages": [...] }]`
//!
//! ## Example Graphiti JSON
//!
//! ```json
//! {
//!   "entities": [
//!     {
//!       "uuid": "ent-001",
//!       "name": "Alice",
//!       "group_id": "g1",
//!       "labels": ["Person"],
//!       "created_at": "2024-07-01T12:00:00Z",
//!       "summary": "Software engineer at Google",
//!       "attributes": {"occupation": "engineer"}
//!     }
//!   ],
//!   "episodes": [
//!     {
//!       "uuid": "ep-001",
//!       "content": "Alice mentioned she likes coffee",
//!       "source": "chat",
//!       "source_description": "Slack conversation",
//!       "reference_time": "2024-07-01T14:00:00Z",
//!       "group_id": "g1"
//!     }
//!   ],
//!   "edges": [
//!     {
//!       "uuid": "edge-001",
//!       "source_node_uuid": "ent-001",
//!       "target_node_uuid": "ent-002",
//!       "fact": "Alice works at Google",
//!       "name": "works_at",
//!       "valid_at": "2024-01-01T00:00:00Z",
//!       "invalid_at": null,
//!       "created_at": "2024-07-01T12:00:00Z",
//!       "group_id": "g1"
//!     }
//!   ]
//! }
//! ```

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::jsonld_export::{ExportEdge, ExportNode, ExportStats, MemoryGraphExport};

// ---------------------------------------------------------------------------
// Graphiti types (deserialization)
// ---------------------------------------------------------------------------

/// A Graphiti entity node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphitiEntity {
    /// Unique identifier.
    pub uuid: String,

    /// Entity name (e.g., "Alice", "Google").
    #[serde(default)]
    pub name: Option<String>,

    /// Group identifier for multi-tenant separation.
    #[serde(default)]
    pub group_id: Option<String>,

    /// Classification labels (e.g., ["Person", "Engineer"]).
    #[serde(default)]
    pub labels: Vec<String>,

    /// ISO 8601 creation timestamp.
    #[serde(default)]
    pub created_at: Option<String>,

    /// Entity summary text.
    #[serde(default)]
    pub summary: Option<String>,

    /// Custom key-value attributes.
    #[serde(default)]
    pub attributes: Option<serde_json::Value>,

    /// Name embedding vector (optional).
    #[serde(default)]
    pub name_embedding: Option<Vec<f32>>,
}

/// A Graphiti episode node — represents a raw data event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphitiEpisode {
    /// Unique identifier.
    pub uuid: String,

    /// Episode content (text or JSON string).
    #[serde(default)]
    pub content: Option<String>,

    /// Source type (e.g., "chat", "document", "json").
    #[serde(default)]
    pub source: Option<String>,

    /// Human-readable source description.
    #[serde(default)]
    pub source_description: Option<String>,

    /// ISO 8601 reference time for this episode.
    #[serde(default)]
    pub reference_time: Option<String>,

    /// Group identifier.
    #[serde(default)]
    pub group_id: Option<String>,

    /// ISO 8601 creation timestamp.
    #[serde(default)]
    pub created_at: Option<String>,

    /// Episode name/title.
    #[serde(default)]
    pub name: Option<String>,
}

/// A Graphiti entity edge (fact) with temporal metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphitiEdge {
    /// Unique identifier.
    pub uuid: String,

    /// Source entity node UUID.
    #[serde(default)]
    pub source_node_uuid: Option<String>,

    /// Target entity node UUID.
    #[serde(default)]
    pub target_node_uuid: Option<String>,

    /// The fact text (e.g., "Alice works at Google").
    #[serde(default)]
    pub fact: Option<String>,

    /// Edge name/type (e.g., "works_at", "lives_in").
    #[serde(default)]
    pub name: Option<String>,

    /// ISO 8601 timestamp when the fact became valid.
    #[serde(default)]
    pub valid_at: Option<String>,

    /// ISO 8601 timestamp when the fact became invalid (null = still valid).
    #[serde(default)]
    pub invalid_at: Option<String>,

    /// ISO 8601 creation timestamp.
    #[serde(default)]
    pub created_at: Option<String>,

    /// Group identifier.
    #[serde(default)]
    pub group_id: Option<String>,

    /// Fact embedding vector (optional).
    #[serde(default)]
    pub fact_embedding: Option<Vec<f32>>,

    /// Episode UUIDs that sourced this edge.
    #[serde(default)]
    pub episodes: Vec<String>,

    /// ISO 8601 timestamp when this edge was expired/superseded.
    #[serde(default)]
    pub expired_at: Option<String>,
}

/// A Graphiti episode edge — links an episode to an entity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphitiEpisodeEdge {
    /// Source node UUID (typically the episode).
    pub source_node_uuid: String,

    /// Target node UUID (typically an entity).
    pub target_node_uuid: String,
}

// ---------------------------------------------------------------------------
// Zep v2 session types (deserialization)
// ---------------------------------------------------------------------------

/// A Zep session message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZepMessage {
    /// Message unique identifier.
    #[serde(default)]
    pub uuid: Option<String>,

    /// Role name (e.g., "user", "assistant").
    #[serde(default)]
    pub role: Option<String>,

    /// Role type (e.g., "human", "ai", "tool").
    #[serde(default)]
    pub role_type: Option<String>,

    /// Message content text.
    #[serde(default)]
    pub content: Option<String>,

    /// ISO 8601 timestamp.
    #[serde(default)]
    pub created_at: Option<String>,

    /// Custom metadata.
    #[serde(default)]
    pub metadata: Option<serde_json::Value>,

    /// Token count.
    #[serde(default)]
    pub token_count: Option<u32>,
}

/// A Zep session with messages.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZepSession {
    /// Session identifier.
    #[serde(default)]
    pub session_id: Option<String>,

    /// Alternative: `id` field.
    #[serde(default)]
    pub id: Option<String>,

    /// Messages in this session.
    #[serde(default)]
    pub messages: Vec<ZepMessage>,

    /// User identifier.
    #[serde(default)]
    pub user_id: Option<String>,

    /// ISO 8601 creation timestamp.
    #[serde(default)]
    pub created_at: Option<String>,

    /// ISO 8601 update timestamp.
    #[serde(default)]
    pub updated_at: Option<String>,

    /// Custom metadata.
    #[serde(default)]
    pub metadata: Option<serde_json::Value>,

    /// Extracted facts for this session.
    #[serde(default)]
    pub facts: Vec<ZepFact>,
}

/// A Zep extracted fact.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZepFact {
    /// Unique identifier.
    #[serde(default)]
    pub uuid: Option<String>,

    /// Fact text content.
    #[serde(default)]
    pub fact: Option<String>,

    /// ISO 8601 timestamp when fact became valid.
    #[serde(default)]
    pub valid_at: Option<String>,

    /// ISO 8601 timestamp when fact became invalid.
    #[serde(default)]
    pub invalid_at: Option<String>,

    /// ISO 8601 creation timestamp.
    #[serde(default)]
    pub created_at: Option<String>,

    /// Rating or confidence score.
    #[serde(default)]
    pub rating: Option<f32>,
}

// ---------------------------------------------------------------------------
// Unified envelope
// ---------------------------------------------------------------------------

/// Zep/Graphiti export envelope — wraps data in multiple supported formats.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZepExport {
    // Graphiti format fields
    /// Graphiti entity nodes.
    #[serde(default)]
    pub entities: Option<Vec<GraphitiEntity>>,

    /// Graphiti episode nodes.
    #[serde(default)]
    pub episodes: Option<Vec<GraphitiEpisode>>,

    /// Graphiti entity edges (facts with temporal metadata).
    #[serde(default)]
    pub edges: Option<Vec<GraphitiEdge>>,

    /// Graphiti episode edges (episode-to-entity links).
    #[serde(default)]
    pub episode_edges: Option<Vec<GraphitiEpisodeEdge>>,

    // Zep v2 format fields
    /// Zep sessions with messages.
    #[serde(default)]
    pub sessions: Option<Vec<ZepSession>>,

    /// Zep standalone facts.
    #[serde(default)]
    pub facts: Option<Vec<ZepFact>>,

    /// Total count metadata.
    #[serde(default)]
    pub total: Option<usize>,
}

/// Options for converting Zep/Graphiti data to Ucotron format.
#[derive(Debug, Clone)]
pub struct ZepImportOptions {
    /// Target namespace for imported data.
    pub namespace: String,
    /// Whether to create edges between items sharing the same user_id.
    pub link_same_user: bool,
    /// Whether to create edges between items sharing the same group_id.
    pub link_same_group: bool,
    /// Whether to preserve expired/invalid edges (default: true).
    pub preserve_expired: bool,
}

impl Default for ZepImportOptions {
    fn default() -> Self {
        Self {
            namespace: "zep_import".to_string(),
            link_same_user: true,
            link_same_group: false,
            preserve_expired: true,
        }
    }
}

/// Result of parsing Zep/Graphiti data.
#[derive(Debug)]
pub struct ZepParseResult {
    /// Number of memories/items parsed.
    pub memories_parsed: usize,
    /// Number of edges imported or inferred.
    pub edges_inferred: usize,
    /// The resulting Ucotron export document.
    pub export: MemoryGraphExport,
}

// ---------------------------------------------------------------------------
// Parsing
// ---------------------------------------------------------------------------

/// Parse Zep/Graphiti JSON data from a string.
///
/// Supports multiple formats:
/// 1. Graphiti export: `{ "entities": [...], "episodes": [...], "edges": [...] }`
/// 2. Zep sessions object: `{ "sessions": [...] }`
/// 3. Zep facts object: `{ "facts": [...] }`
/// 4. Bare array of sessions: `[{ "session_id": ..., "messages": [...] }]`
/// 5. Single session object: `{ "session_id": ..., "messages": [...] }`
pub fn parse_zep_json(json: &str) -> anyhow::Result<ZepExport> {
    // Try the envelope format first (handles Graphiti + Zep object formats).
    if let Ok(envelope) = serde_json::from_str::<ZepExport>(json) {
        // Check if any meaningful data was parsed.
        let has_graphiti = envelope.entities.as_ref().is_some_and(|e| !e.is_empty())
            || envelope.episodes.as_ref().is_some_and(|e| !e.is_empty())
            || envelope.edges.as_ref().is_some_and(|e| !e.is_empty());
        let has_zep = envelope.sessions.as_ref().is_some_and(|s| !s.is_empty())
            || envelope.facts.as_ref().is_some_and(|f| !f.is_empty());

        if has_graphiti || has_zep {
            return Ok(envelope);
        }

        // Empty envelope is still valid.
        if envelope.entities.is_some() || envelope.sessions.is_some() || envelope.facts.is_some() {
            return Ok(envelope);
        }
    }

    // Try bare array of sessions.
    if let Ok(sessions) = serde_json::from_str::<Vec<ZepSession>>(json) {
        return Ok(ZepExport {
            entities: None,
            episodes: None,
            edges: None,
            episode_edges: None,
            sessions: Some(sessions),
            facts: None,
            total: None,
        });
    }

    // Try single session object.
    if let Ok(session) = serde_json::from_str::<ZepSession>(json) {
        if session.session_id.is_some() || session.id.is_some() || !session.messages.is_empty() {
            return Ok(ZepExport {
                entities: None,
                episodes: None,
                edges: None,
                episode_edges: None,
                sessions: Some(vec![session]),
                facts: None,
                total: None,
            });
        }
    }

    Err(anyhow::anyhow!(
        "Failed to parse Zep/Graphiti data: expected Graphiti export with 'entities'/'episodes'/'edges', \
         Zep object with 'sessions'/'facts', bare array of sessions, or single session object"
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
    let trimmed = ts.trim();

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

    if trimmed.len() > 11 {
        let time_start = if trimmed.as_bytes()[10] == b'T' {
            11
        } else {
            10
        };
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

/// Flatten a serde_json::Value object to a HashMap.
fn flatten_metadata(val: &serde_json::Value) -> HashMap<String, serde_json::Value> {
    match val {
        serde_json::Value::Object(map) => map.iter().map(|(k, v)| (k.clone(), v.clone())).collect(),
        _ => HashMap::new(),
    }
}

/// Convert Zep/Graphiti export data to a `MemoryGraphExport`.
///
/// Handles both Graphiti (temporal KG) and Zep v2 (session-based) formats.
/// All temporal metadata is preserved as node/edge metadata with `zep_` prefix.
pub fn zep_to_ucotron(data: &ZepExport, options: &ZepImportOptions) -> ZepParseResult {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let mut nodes: Vec<ExportNode> = Vec::new();
    let mut edges: Vec<ExportEdge> = Vec::new();

    // Map Graphiti UUIDs → Ucotron node indices for edge resolution.
    let mut uuid_to_idx: HashMap<String, usize> = HashMap::new();

    // Track items by user/group for edge inference.
    let mut by_user: HashMap<String, Vec<usize>> = HashMap::new();
    let mut by_group: HashMap<String, Vec<usize>> = HashMap::new();

    // --- Process Graphiti entities ---
    if let Some(ref entities) = data.entities {
        for entity in entities {
            let idx = nodes.len();
            uuid_to_idx.insert(entity.uuid.clone(), idx);

            let mut metadata: HashMap<String, serde_json::Value> = HashMap::new();
            metadata.insert("_import_source".into(), serde_json::json!("zep"));
            metadata.insert("zep_original_id".into(), serde_json::json!(entity.uuid));
            metadata.insert("zep_data_type".into(), serde_json::json!("entity"));

            if let Some(ref name) = entity.name {
                metadata.insert("zep_name".into(), serde_json::json!(name));
            }
            if let Some(ref gid) = entity.group_id {
                metadata.insert("zep_group_id".into(), serde_json::json!(gid));
                if options.link_same_group {
                    by_group.entry(gid.clone()).or_default().push(idx);
                }
            }
            if !entity.labels.is_empty() {
                metadata.insert("zep_labels".into(), serde_json::json!(entity.labels));
            }
            if let Some(ref summary) = entity.summary {
                metadata.insert("zep_summary".into(), serde_json::json!(summary));
            }
            if let Some(ref attrs) = entity.attributes {
                for (k, v) in flatten_metadata(attrs) {
                    metadata.insert(format!("zep_attr_{}", k), v);
                }
            }

            let timestamp = entity
                .created_at
                .as_deref()
                .map(parse_iso_timestamp)
                .unwrap_or(now);

            let content = entity
                .summary
                .clone()
                .unwrap_or_else(|| entity.name.clone().unwrap_or_else(|| entity.uuid.clone()));

            nodes.push(ExportNode {
                id: format!("ucotron:node/{}", idx + 1),
                node_type: "Entity".to_string(),
                content,
                timestamp,
                metadata,
                embedding: entity.name_embedding.clone(),
            });
        }
    }

    // --- Process Graphiti episodes ---
    if let Some(ref episodes) = data.episodes {
        for episode in episodes {
            let idx = nodes.len();
            uuid_to_idx.insert(episode.uuid.clone(), idx);

            let mut metadata: HashMap<String, serde_json::Value> = HashMap::new();
            metadata.insert("_import_source".into(), serde_json::json!("zep"));
            metadata.insert("zep_original_id".into(), serde_json::json!(episode.uuid));
            metadata.insert("zep_data_type".into(), serde_json::json!("episode"));

            if let Some(ref source) = episode.source {
                metadata.insert("zep_source".into(), serde_json::json!(source));
            }
            if let Some(ref desc) = episode.source_description {
                metadata.insert("zep_source_description".into(), serde_json::json!(desc));
            }
            if let Some(ref gid) = episode.group_id {
                metadata.insert("zep_group_id".into(), serde_json::json!(gid));
                if options.link_same_group {
                    by_group.entry(gid.clone()).or_default().push(idx);
                }
            }
            if let Some(ref name) = episode.name {
                metadata.insert("zep_episode_name".into(), serde_json::json!(name));
            }

            let timestamp = episode
                .reference_time
                .as_deref()
                .or(episode.created_at.as_deref())
                .map(parse_iso_timestamp)
                .unwrap_or(now);

            let content = episode
                .content
                .clone()
                .unwrap_or_else(|| episode.uuid.clone());

            nodes.push(ExportNode {
                id: format!("ucotron:node/{}", idx + 1),
                node_type: "Event".to_string(),
                content,
                timestamp,
                metadata,
                embedding: None,
            });
        }
    }

    // --- Process Graphiti entity edges ---
    if let Some(ref graph_edges) = data.edges {
        for ge in graph_edges {
            let source_idx = ge
                .source_node_uuid
                .as_deref()
                .and_then(|u| uuid_to_idx.get(u));
            let target_idx = ge
                .target_node_uuid
                .as_deref()
                .and_then(|u| uuid_to_idx.get(u));

            // Skip edges referencing unknown nodes.
            let (src, tgt) = match (source_idx, target_idx) {
                (Some(s), Some(t)) => (*s, *t),
                _ => continue,
            };

            // Check if this edge is expired and whether we should include it.
            let is_expired = ge.invalid_at.is_some() || ge.expired_at.is_some();
            if is_expired && !options.preserve_expired {
                continue;
            }

            let mut edge_meta: HashMap<String, serde_json::Value> = HashMap::new();
            edge_meta.insert("zep_edge_id".into(), serde_json::json!(ge.uuid));

            if let Some(ref fact) = ge.fact {
                edge_meta.insert("zep_fact".into(), serde_json::json!(fact));
            }
            if let Some(ref name) = ge.name {
                edge_meta.insert("zep_edge_name".into(), serde_json::json!(name));
            }
            if let Some(ref valid) = ge.valid_at {
                edge_meta.insert("zep_valid_at".into(), serde_json::json!(valid));
            }
            if let Some(ref invalid) = ge.invalid_at {
                edge_meta.insert("zep_invalid_at".into(), serde_json::json!(invalid));
            }
            if let Some(ref expired) = ge.expired_at {
                edge_meta.insert("zep_expired_at".into(), serde_json::json!(expired));
            }
            if let Some(ref created) = ge.created_at {
                edge_meta.insert("zep_created_at".into(), serde_json::json!(created));
            }
            if !ge.episodes.is_empty() {
                edge_meta.insert("zep_episodes".into(), serde_json::json!(ge.episodes));
            }

            // Map edge name to Ucotron edge type.
            let edge_type = if is_expired {
                "Supersedes".to_string()
            } else {
                "RelatesTo".to_string()
            };

            let weight = if is_expired { 0.3 } else { 0.7 };

            edges.push(ExportEdge {
                source: format!("ucotron:node/{}", src + 1),
                target: format!("ucotron:node/{}", tgt + 1),
                edge_type,
                weight,
                metadata: edge_meta,
            });
        }
    }

    // --- Process Graphiti episode edges ---
    if let Some(ref ep_edges) = data.episode_edges {
        for ee in ep_edges {
            let source_idx = uuid_to_idx.get(&ee.source_node_uuid);
            let target_idx = uuid_to_idx.get(&ee.target_node_uuid);

            if let (Some(s), Some(t)) = (source_idx, target_idx) {
                edges.push(ExportEdge {
                    source: format!("ucotron:node/{}", s + 1),
                    target: format!("ucotron:node/{}", t + 1),
                    edge_type: "RelatesTo".to_string(),
                    weight: 0.5,
                    metadata: HashMap::new(),
                });
            }
        }
    }

    // --- Process Zep sessions ---
    if let Some(ref sessions) = data.sessions {
        for session in sessions {
            let session_id = session
                .session_id
                .as_deref()
                .or(session.id.as_deref())
                .unwrap_or("unknown");

            let mut session_indices: Vec<usize> = Vec::new();

            for msg in &session.messages {
                let idx = nodes.len();
                session_indices.push(idx);

                let mut metadata: HashMap<String, serde_json::Value> = HashMap::new();
                metadata.insert("_import_source".into(), serde_json::json!("zep"));
                metadata.insert("zep_data_type".into(), serde_json::json!("message"));
                metadata.insert("zep_session_id".into(), serde_json::json!(session_id));

                if let Some(ref uuid) = msg.uuid {
                    metadata.insert("zep_original_id".into(), serde_json::json!(uuid));
                }
                if let Some(ref role) = msg.role {
                    metadata.insert("zep_role".into(), serde_json::json!(role));
                }
                if let Some(ref role_type) = msg.role_type {
                    metadata.insert("zep_role_type".into(), serde_json::json!(role_type));
                }
                if let Some(ref uid) = session.user_id {
                    metadata.insert("zep_user_id".into(), serde_json::json!(uid));
                    if options.link_same_user {
                        by_user.entry(uid.clone()).or_default().push(idx);
                    }
                }
                if let Some(tokens) = msg.token_count {
                    metadata.insert("zep_token_count".into(), serde_json::json!(tokens));
                }
                if let Some(ref meta_val) = msg.metadata {
                    for (k, v) in flatten_metadata(meta_val) {
                        metadata.insert(format!("zep_meta_{}", k), v);
                    }
                }

                let timestamp = msg
                    .created_at
                    .as_deref()
                    .or(session.created_at.as_deref())
                    .map(parse_iso_timestamp)
                    .unwrap_or(now);

                let content = msg.content.clone().unwrap_or_default();

                nodes.push(ExportNode {
                    id: format!("ucotron:node/{}", idx + 1),
                    node_type: "Entity".to_string(),
                    content,
                    timestamp,
                    metadata,
                    embedding: None,
                });
            }

            // Create temporal chain edges within a session (NextEpisode).
            for window in session_indices.windows(2) {
                edges.push(ExportEdge {
                    source: format!("ucotron:node/{}", window[0] + 1),
                    target: format!("ucotron:node/{}", window[1] + 1),
                    edge_type: "NextEpisode".to_string(),
                    weight: 0.8,
                    metadata: {
                        let mut m = HashMap::new();
                        m.insert("zep_session_id".into(), serde_json::json!(session_id));
                        m
                    },
                });
            }

            // Process session-level facts.
            for fact in &session.facts {
                let idx = nodes.len();

                let mut metadata: HashMap<String, serde_json::Value> = HashMap::new();
                metadata.insert("_import_source".into(), serde_json::json!("zep"));
                metadata.insert("zep_data_type".into(), serde_json::json!("fact"));
                metadata.insert("zep_session_id".into(), serde_json::json!(session_id));

                if let Some(ref uuid) = fact.uuid {
                    metadata.insert("zep_original_id".into(), serde_json::json!(uuid));
                }
                if let Some(ref valid) = fact.valid_at {
                    metadata.insert("zep_valid_at".into(), serde_json::json!(valid));
                }
                if let Some(ref invalid) = fact.invalid_at {
                    metadata.insert("zep_invalid_at".into(), serde_json::json!(invalid));
                }
                if let Some(rating) = fact.rating {
                    metadata.insert("zep_rating".into(), serde_json::json!(rating));
                }
                if let Some(ref uid) = session.user_id {
                    metadata.insert("zep_user_id".into(), serde_json::json!(uid));
                }

                let timestamp = fact
                    .valid_at
                    .as_deref()
                    .or(fact.created_at.as_deref())
                    .or(session.created_at.as_deref())
                    .map(parse_iso_timestamp)
                    .unwrap_or(now);

                let content = fact.fact.clone().unwrap_or_default();

                nodes.push(ExportNode {
                    id: format!("ucotron:node/{}", idx + 1),
                    node_type: "Fact".to_string(),
                    content,
                    timestamp,
                    metadata,
                    embedding: None,
                });
            }
        }
    }

    // --- Process standalone Zep facts ---
    if let Some(ref facts) = data.facts {
        for fact in facts {
            let idx = nodes.len();

            let mut metadata: HashMap<String, serde_json::Value> = HashMap::new();
            metadata.insert("_import_source".into(), serde_json::json!("zep"));
            metadata.insert("zep_data_type".into(), serde_json::json!("fact"));

            if let Some(ref uuid) = fact.uuid {
                metadata.insert("zep_original_id".into(), serde_json::json!(uuid));
            }
            if let Some(ref valid) = fact.valid_at {
                metadata.insert("zep_valid_at".into(), serde_json::json!(valid));
            }
            if let Some(ref invalid) = fact.invalid_at {
                metadata.insert("zep_invalid_at".into(), serde_json::json!(invalid));
            }
            if let Some(rating) = fact.rating {
                metadata.insert("zep_rating".into(), serde_json::json!(rating));
            }

            let timestamp = fact
                .valid_at
                .as_deref()
                .or(fact.created_at.as_deref())
                .map(parse_iso_timestamp)
                .unwrap_or(now);

            let content = fact.fact.clone().unwrap_or_default();

            nodes.push(ExportNode {
                id: format!("ucotron:node/{}", idx + 1),
                node_type: "Fact".to_string(),
                content,
                timestamp,
                metadata,
                embedding: None,
            });

            // Ignore idx for user-linking since standalone facts don't have user_id
            let _ = idx;
        }
    }

    // --- Infer group-based edges ---
    infer_chain_edges(&mut edges, &by_group, &nodes);
    infer_chain_edges(&mut edges, &by_user, &nodes);

    let memories_parsed = nodes.len();
    let edges_inferred = edges.len();

    let stats = ExportStats {
        total_nodes: nodes.len(),
        total_edges: edges.len(),
        has_embeddings: nodes.iter().any(|n| n.embedding.is_some()),
        is_incremental: false,
        from_timestamp: None,
    };

    let context = serde_json::json!({
        "ucotron": "https://ucotron.com/schema/v1",
        "schema": "https://schema.org/",
        "@vocab": "https://ucotron.com/schema/v1",
        "zep": "https://help.getzep.com/schema/v1",
        "graphiti": "https://github.com/getzep/graphiti/schema/v1"
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

    ZepParseResult {
        memories_parsed,
        edges_inferred,
        export,
    }
}

/// Create temporal chain edges between items in the same group.
fn infer_chain_edges(
    edges: &mut Vec<ExportEdge>,
    groups: &HashMap<String, Vec<usize>>,
    nodes: &[ExportNode],
) {
    for indices in groups.values() {
        if indices.len() < 2 {
            continue;
        }
        let mut sorted: Vec<usize> = indices.clone();
        sorted.sort_by_key(|&i| nodes[i].timestamp);

        for window in sorted.windows(2) {
            edges.push(ExportEdge {
                source: format!("ucotron:node/{}", window[0] + 1),
                target: format!("ucotron:node/{}", window[1] + 1),
                edge_type: "RelatesTo".to_string(),
                weight: 0.5,
                metadata: HashMap::new(),
            });
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_graphiti_format() {
        let json = r#"{
            "entities": [
                {
                    "uuid": "ent-001",
                    "name": "Alice",
                    "labels": ["Person"],
                    "created_at": "2024-07-01T12:00:00Z",
                    "summary": "Software engineer"
                },
                {
                    "uuid": "ent-002",
                    "name": "Google",
                    "labels": ["Organization"],
                    "created_at": "2024-07-01T12:00:00Z",
                    "summary": "Tech company"
                }
            ],
            "edges": [
                {
                    "uuid": "edge-001",
                    "source_node_uuid": "ent-001",
                    "target_node_uuid": "ent-002",
                    "fact": "Alice works at Google",
                    "name": "works_at",
                    "valid_at": "2024-01-01T00:00:00Z",
                    "created_at": "2024-07-01T12:00:00Z"
                }
            ]
        }"#;

        let export = parse_zep_json(json).unwrap();
        assert_eq!(export.entities.as_ref().unwrap().len(), 2);
        assert_eq!(export.edges.as_ref().unwrap().len(), 1);
        assert!(export.sessions.is_none());
    }

    #[test]
    fn test_parse_zep_sessions_format() {
        let json = r#"{
            "sessions": [
                {
                    "session_id": "sess-001",
                    "user_id": "alice",
                    "messages": [
                        {
                            "uuid": "msg-001",
                            "role": "user",
                            "role_type": "human",
                            "content": "I like coffee",
                            "created_at": "2024-07-01T12:00:00Z"
                        },
                        {
                            "uuid": "msg-002",
                            "role": "assistant",
                            "role_type": "ai",
                            "content": "That's great!",
                            "created_at": "2024-07-01T12:01:00Z"
                        }
                    ]
                }
            ]
        }"#;

        let export = parse_zep_json(json).unwrap();
        assert!(export.entities.is_none());
        let sessions = export.sessions.as_ref().unwrap();
        assert_eq!(sessions.len(), 1);
        assert_eq!(sessions[0].messages.len(), 2);
    }

    #[test]
    fn test_parse_zep_facts_format() {
        let json = r#"{
            "facts": [
                {
                    "uuid": "fact-001",
                    "fact": "Alice works at Google",
                    "valid_at": "2024-01-01T00:00:00Z",
                    "rating": 0.9
                }
            ]
        }"#;

        let export = parse_zep_json(json).unwrap();
        let facts = export.facts.as_ref().unwrap();
        assert_eq!(facts.len(), 1);
        assert_eq!(facts[0].fact, Some("Alice works at Google".to_string()));
        assert_eq!(facts[0].rating, Some(0.9));
    }

    #[test]
    fn test_parse_bare_sessions_array() {
        let json = r#"[
            {
                "session_id": "sess-100",
                "messages": [
                    {"content": "Hello", "role": "user"}
                ]
            }
        ]"#;

        let export = parse_zep_json(json).unwrap();
        let sessions = export.sessions.as_ref().unwrap();
        assert_eq!(sessions.len(), 1);
        assert_eq!(sessions[0].session_id, Some("sess-100".to_string()));
    }

    #[test]
    fn test_parse_single_session() {
        let json = r#"{
            "session_id": "sess-solo",
            "user_id": "bob",
            "messages": [
                {"content": "Test message", "role": "user"}
            ]
        }"#;

        let export = parse_zep_json(json).unwrap();
        let sessions = export.sessions.as_ref().unwrap();
        assert_eq!(sessions.len(), 1);
        assert_eq!(sessions[0].user_id, Some("bob".to_string()));
    }

    #[test]
    fn test_parse_invalid_json() {
        let result = parse_zep_json("not valid json");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_empty_sessions() {
        let json = r#"{"sessions": []}"#;
        let export = parse_zep_json(json).unwrap();
        assert!(export.sessions.as_ref().unwrap().is_empty());
    }

    #[test]
    fn test_convert_graphiti_entities() {
        let data = ZepExport {
            entities: Some(vec![GraphitiEntity {
                uuid: "ent-001".into(),
                name: Some("Alice".into()),
                group_id: Some("g1".into()),
                labels: vec!["Person".into()],
                created_at: Some("2024-07-01T12:00:00Z".into()),
                summary: Some("Software engineer".into()),
                attributes: Some(serde_json::json!({"occupation": "engineer"})),
                name_embedding: None,
            }]),
            episodes: None,
            edges: None,
            episode_edges: None,
            sessions: None,
            facts: None,
            total: None,
        };

        let result = zep_to_ucotron(&data, &ZepImportOptions::default());

        assert_eq!(result.memories_parsed, 1);
        assert_eq!(result.export.nodes.len(), 1);

        let node = &result.export.nodes[0];
        assert_eq!(node.node_type, "Entity");
        assert_eq!(node.content, "Software engineer");
        assert_eq!(node.timestamp, 1719835200);
        assert_eq!(
            node.metadata.get("_import_source"),
            Some(&serde_json::json!("zep"))
        );
        assert_eq!(
            node.metadata.get("zep_name"),
            Some(&serde_json::json!("Alice"))
        );
        assert_eq!(
            node.metadata.get("zep_labels"),
            Some(&serde_json::json!(["Person"]))
        );
        assert_eq!(
            node.metadata.get("zep_attr_occupation"),
            Some(&serde_json::json!("engineer"))
        );
    }

    #[test]
    fn test_convert_graphiti_with_edges() {
        let data = ZepExport {
            entities: Some(vec![
                GraphitiEntity {
                    uuid: "ent-001".into(),
                    name: Some("Alice".into()),
                    summary: Some("Person".into()),
                    ..default_graphiti_entity()
                },
                GraphitiEntity {
                    uuid: "ent-002".into(),
                    name: Some("Google".into()),
                    summary: Some("Company".into()),
                    ..default_graphiti_entity()
                },
            ]),
            edges: Some(vec![GraphitiEdge {
                uuid: "edge-001".into(),
                source_node_uuid: Some("ent-001".into()),
                target_node_uuid: Some("ent-002".into()),
                fact: Some("Alice works at Google".into()),
                name: Some("works_at".into()),
                valid_at: Some("2024-01-01T00:00:00Z".into()),
                invalid_at: None,
                created_at: Some("2024-07-01T12:00:00Z".into()),
                ..default_graphiti_edge()
            }]),
            episodes: None,
            episode_edges: None,
            sessions: None,
            facts: None,
            total: None,
        };

        let result = zep_to_ucotron(&data, &ZepImportOptions::default());

        assert_eq!(result.export.nodes.len(), 2);
        assert_eq!(result.export.edges.len(), 1);

        let edge = &result.export.edges[0];
        assert_eq!(edge.edge_type, "RelatesTo");
        assert_eq!(edge.weight, 0.7);
        assert_eq!(
            edge.metadata.get("zep_fact"),
            Some(&serde_json::json!("Alice works at Google"))
        );
        assert_eq!(
            edge.metadata.get("zep_valid_at"),
            Some(&serde_json::json!("2024-01-01T00:00:00Z"))
        );
    }

    #[test]
    fn test_convert_graphiti_expired_edges() {
        let data = ZepExport {
            entities: Some(vec![
                GraphitiEntity {
                    uuid: "ent-001".into(),
                    name: Some("Alice".into()),
                    summary: Some("Person".into()),
                    ..default_graphiti_entity()
                },
                GraphitiEntity {
                    uuid: "ent-002".into(),
                    name: Some("StartupX".into()),
                    summary: Some("Company".into()),
                    ..default_graphiti_entity()
                },
            ]),
            edges: Some(vec![GraphitiEdge {
                uuid: "edge-exp".into(),
                source_node_uuid: Some("ent-001".into()),
                target_node_uuid: Some("ent-002".into()),
                fact: Some("Alice worked at StartupX".into()),
                invalid_at: Some("2023-06-01T00:00:00Z".into()),
                ..default_graphiti_edge()
            }]),
            episodes: None,
            episode_edges: None,
            sessions: None,
            facts: None,
            total: None,
        };

        // With preserve_expired = true (default).
        let result = zep_to_ucotron(&data, &ZepImportOptions::default());
        assert_eq!(result.export.edges.len(), 1);
        assert_eq!(result.export.edges[0].edge_type, "Supersedes");
        assert_eq!(result.export.edges[0].weight, 0.3);

        // With preserve_expired = false.
        let opts = ZepImportOptions {
            preserve_expired: false,
            ..Default::default()
        };
        let result2 = zep_to_ucotron(&data, &opts);
        assert_eq!(result2.export.edges.len(), 0);
    }

    #[test]
    fn test_convert_zep_sessions() {
        let data = ZepExport {
            sessions: Some(vec![ZepSession {
                session_id: Some("sess-001".into()),
                user_id: Some("alice".into()),
                messages: vec![
                    ZepMessage {
                        uuid: Some("msg-001".into()),
                        role: Some("user".into()),
                        role_type: Some("human".into()),
                        content: Some("I like coffee".into()),
                        created_at: Some("2024-07-01T10:00:00Z".into()),
                        ..default_zep_message()
                    },
                    ZepMessage {
                        uuid: Some("msg-002".into()),
                        role: Some("assistant".into()),
                        role_type: Some("ai".into()),
                        content: Some("Noted!".into()),
                        created_at: Some("2024-07-01T10:01:00Z".into()),
                        ..default_zep_message()
                    },
                ],
                facts: vec![],
                ..default_zep_session()
            }]),
            entities: None,
            episodes: None,
            edges: None,
            episode_edges: None,
            facts: None,
            total: None,
        };

        let result = zep_to_ucotron(&data, &ZepImportOptions::default());

        // 2 messages = 2 nodes.
        assert_eq!(result.export.nodes.len(), 2);

        // 1 NextEpisode edge between messages.
        let next_edges: Vec<_> = result
            .export
            .edges
            .iter()
            .filter(|e| e.edge_type == "NextEpisode")
            .collect();
        assert_eq!(next_edges.len(), 1);
        assert_eq!(next_edges[0].weight, 0.8);

        // Check metadata.
        let node = &result.export.nodes[0];
        assert_eq!(node.content, "I like coffee");
        assert_eq!(
            node.metadata.get("zep_role"),
            Some(&serde_json::json!("user"))
        );
        assert_eq!(
            node.metadata.get("zep_session_id"),
            Some(&serde_json::json!("sess-001"))
        );
        assert_eq!(
            node.metadata.get("zep_user_id"),
            Some(&serde_json::json!("alice"))
        );
    }

    #[test]
    fn test_convert_zep_session_with_facts() {
        let data = ZepExport {
            sessions: Some(vec![ZepSession {
                session_id: Some("sess-facts".into()),
                user_id: Some("alice".into()),
                messages: vec![ZepMessage {
                    content: Some("I work at Google".into()),
                    ..default_zep_message()
                }],
                facts: vec![ZepFact {
                    uuid: Some("fact-001".into()),
                    fact: Some("Alice works at Google".into()),
                    valid_at: Some("2024-01-01T00:00:00Z".into()),
                    invalid_at: None,
                    created_at: Some("2024-07-01T12:00:00Z".into()),
                    rating: Some(0.95),
                }],
                ..default_zep_session()
            }]),
            entities: None,
            episodes: None,
            edges: None,
            episode_edges: None,
            facts: None,
            total: None,
        };

        let result = zep_to_ucotron(&data, &ZepImportOptions::default());

        // 1 message + 1 fact = 2 nodes.
        assert_eq!(result.export.nodes.len(), 2);

        let fact_node = result
            .export
            .nodes
            .iter()
            .find(|n| n.node_type == "Fact")
            .unwrap();
        assert_eq!(fact_node.content, "Alice works at Google");
        // f32 → f64 conversion causes precision difference, so check approximately.
        let rating = fact_node.metadata.get("zep_rating").unwrap();
        let rating_val = rating.as_f64().unwrap();
        assert!((rating_val - 0.95).abs() < 0.001);
        assert_eq!(
            fact_node.metadata.get("zep_valid_at"),
            Some(&serde_json::json!("2024-01-01T00:00:00Z"))
        );
    }

    #[test]
    fn test_convert_empty() {
        let data = ZepExport {
            entities: Some(vec![]),
            episodes: None,
            edges: None,
            episode_edges: None,
            sessions: Some(vec![]),
            facts: None,
            total: None,
        };

        let result = zep_to_ucotron(&data, &ZepImportOptions::default());
        assert_eq!(result.memories_parsed, 0);
        assert!(result.export.nodes.is_empty());
        assert!(result.export.edges.is_empty());
    }

    #[test]
    fn test_convert_preserves_temporal_metadata() {
        let data = ZepExport {
            entities: Some(vec![
                GraphitiEntity {
                    uuid: "ent-t1".into(),
                    name: Some("A".into()),
                    summary: Some("Node A".into()),
                    ..default_graphiti_entity()
                },
                GraphitiEntity {
                    uuid: "ent-t2".into(),
                    name: Some("B".into()),
                    summary: Some("Node B".into()),
                    ..default_graphiti_entity()
                },
            ]),
            edges: Some(vec![GraphitiEdge {
                uuid: "edge-t1".into(),
                source_node_uuid: Some("ent-t1".into()),
                target_node_uuid: Some("ent-t2".into()),
                fact: Some("A relates to B".into()),
                valid_at: Some("2024-01-15T00:00:00Z".into()),
                invalid_at: Some("2024-06-01T00:00:00Z".into()),
                expired_at: Some("2024-06-01T00:00:00Z".into()),
                created_at: Some("2024-01-15T10:00:00Z".into()),
                episodes: vec!["ep-001".into(), "ep-002".into()],
                ..default_graphiti_edge()
            }]),
            episodes: None,
            episode_edges: None,
            sessions: None,
            facts: None,
            total: None,
        };

        let result = zep_to_ucotron(&data, &ZepImportOptions::default());
        let edge = &result.export.edges[0];

        assert!(edge.metadata.contains_key("zep_valid_at"));
        assert!(edge.metadata.contains_key("zep_invalid_at"));
        assert!(edge.metadata.contains_key("zep_expired_at"));
        assert!(edge.metadata.contains_key("zep_created_at"));
        assert_eq!(
            edge.metadata.get("zep_episodes"),
            Some(&serde_json::json!(["ep-001", "ep-002"]))
        );
    }

    #[test]
    fn test_export_has_correct_version() {
        let result = zep_to_ucotron(
            &ZepExport {
                entities: Some(vec![]),
                episodes: None,
                edges: None,
                episode_edges: None,
                sessions: None,
                facts: None,
                total: None,
            },
            &ZepImportOptions::default(),
        );
        assert_eq!(result.export.version, "1.0");
        assert_eq!(result.export.graph_type, "ucotron:MemoryGraph");
    }

    #[test]
    fn test_full_roundtrip_parse_and_convert() {
        let json = r#"{
            "entities": [
                {
                    "uuid": "ent-rt1",
                    "name": "Alice",
                    "labels": ["Person"],
                    "created_at": "2024-01-15T10:00:00Z",
                    "summary": "Software engineer at Google"
                },
                {
                    "uuid": "ent-rt2",
                    "name": "Google",
                    "labels": ["Organization"],
                    "created_at": "2024-01-15T10:00:00Z",
                    "summary": "Tech company"
                }
            ],
            "edges": [
                {
                    "uuid": "edge-rt1",
                    "source_node_uuid": "ent-rt1",
                    "target_node_uuid": "ent-rt2",
                    "fact": "Alice works at Google",
                    "name": "works_at",
                    "valid_at": "2024-01-01T00:00:00Z",
                    "created_at": "2024-01-15T10:00:00Z"
                }
            ],
            "episodes": [
                {
                    "uuid": "ep-rt1",
                    "content": "Alice mentioned she works at Google",
                    "source": "chat",
                    "reference_time": "2024-01-15T14:00:00Z"
                }
            ]
        }"#;

        let export_data = parse_zep_json(json).unwrap();
        let result = zep_to_ucotron(&export_data, &ZepImportOptions::default());

        // 2 entities + 1 episode = 3 nodes.
        assert_eq!(result.export.nodes.len(), 3);
        // 1 entity edge.
        assert!(!result.export.edges.is_empty());

        // Verify JSON-LD roundtrip.
        let json_export = crate::jsonld_export::export_to_json(&result.export).unwrap();
        let reimported = crate::jsonld_export::import_from_json(&json_export).unwrap();
        assert_eq!(reimported.nodes.len(), 3);
        assert_eq!(reimported.version, "1.0");
    }

    #[test]
    fn test_node_ids_are_sequential() {
        let data = ZepExport {
            entities: Some(vec![
                GraphitiEntity {
                    uuid: "e1".into(),
                    name: Some("A".into()),
                    summary: Some("Node A".into()),
                    ..default_graphiti_entity()
                },
                GraphitiEntity {
                    uuid: "e2".into(),
                    name: Some("B".into()),
                    summary: Some("Node B".into()),
                    ..default_graphiti_entity()
                },
            ]),
            episodes: Some(vec![GraphitiEpisode {
                uuid: "ep1".into(),
                content: Some("Episode 1".into()),
                ..default_graphiti_episode()
            }]),
            edges: None,
            episode_edges: None,
            sessions: None,
            facts: None,
            total: None,
        };

        let result = zep_to_ucotron(&data, &ZepImportOptions::default());

        for (i, node) in result.export.nodes.iter().enumerate() {
            assert_eq!(node.id, format!("ucotron:node/{}", i + 1));
        }
    }

    // --- Test helpers ---

    fn default_graphiti_entity() -> GraphitiEntity {
        GraphitiEntity {
            uuid: String::new(),
            name: None,
            group_id: None,
            labels: vec![],
            created_at: None,
            summary: None,
            attributes: None,
            name_embedding: None,
        }
    }

    fn default_graphiti_episode() -> GraphitiEpisode {
        GraphitiEpisode {
            uuid: String::new(),
            content: None,
            source: None,
            source_description: None,
            reference_time: None,
            group_id: None,
            created_at: None,
            name: None,
        }
    }

    fn default_graphiti_edge() -> GraphitiEdge {
        GraphitiEdge {
            uuid: String::new(),
            source_node_uuid: None,
            target_node_uuid: None,
            fact: None,
            name: None,
            valid_at: None,
            invalid_at: None,
            created_at: None,
            group_id: None,
            fact_embedding: None,
            episodes: vec![],
            expired_at: None,
        }
    }

    fn default_zep_message() -> ZepMessage {
        ZepMessage {
            uuid: None,
            role: None,
            role_type: None,
            content: None,
            created_at: None,
            metadata: None,
            token_count: None,
        }
    }

    fn default_zep_session() -> ZepSession {
        ZepSession {
            session_id: None,
            id: None,
            messages: vec![],
            user_id: None,
            created_at: None,
            updated_at: None,
            metadata: None,
            facts: vec![],
        }
    }
}
