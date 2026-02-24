//! Agent management types for multi-agent memory isolation and sharing.
//!
//! Each agent owns an isolated namespace in the knowledge graph. Agents can
//! share read or read-write access to their namespace with other agents,
//! clone their graph into a new namespace (optionally filtered), and merge
//! graphs from other namespaces with entity deduplication.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::types::Value;

/// Unique identifier for an agent (string-based for user-friendly naming).
pub type AgentId = String;

/// An autonomous agent with its own isolated memory namespace.
///
/// Each agent maps to exactly one namespace in the backend. The namespace
/// is auto-created when the agent is first registered and cascade-deleted
/// when the agent is removed.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Agent {
    /// Unique agent identifier.
    pub id: AgentId,
    /// Human-readable display name.
    pub name: String,
    /// Backend namespace this agent owns (typically `"agent_{id}"`).
    pub namespace: String,
    /// Owner identifier (user or service account that created the agent).
    pub owner: String,
    /// Unix timestamp when the agent was created.
    pub created_at: u64,
    /// Agent-specific configuration (model, prompt template, etc.).
    pub config: HashMap<String, Value>,
}

impl Agent {
    /// Create a new agent with the given fields. The namespace defaults to
    /// `"agent_{id}"` if not explicitly provided.
    pub fn new(
        id: impl Into<String>,
        name: impl Into<String>,
        owner: impl Into<String>,
        created_at: u64,
    ) -> Self {
        let id = id.into();
        let namespace = format!("agent_{}", &id);
        Self {
            id,
            name: name.into(),
            namespace,
            owner: owner.into(),
            created_at,
            config: HashMap::new(),
        }
    }

    /// Builder-style setter for a custom namespace.
    pub fn with_namespace(mut self, ns: impl Into<String>) -> Self {
        self.namespace = ns.into();
        self
    }

    /// Builder-style setter for agent configuration.
    pub fn with_config(mut self, config: HashMap<String, Value>) -> Self {
        self.config = config;
        self
    }
}

/// Permission level for shared access between agents.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum SharePermission {
    /// Can read memories but not write.
    ReadOnly,
    /// Can read and write memories.
    ReadWrite,
}

/// A share grant from one agent's namespace to another.
///
/// When agent A shares with agent B, agent B can access agent A's namespace
/// according to the granted [`SharePermission`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AgentShare {
    /// The agent granting access (source namespace owner).
    pub agent_id: AgentId,
    /// The agent receiving access.
    pub target_agent_id: AgentId,
    /// Access level granted to the target agent.
    pub permissions: SharePermission,
    /// Unix timestamp when the share was created.
    pub created_at: u64,
}

impl AgentShare {
    /// Create a new share grant.
    pub fn new(
        agent_id: impl Into<String>,
        target_agent_id: impl Into<String>,
        permissions: SharePermission,
        created_at: u64,
    ) -> Self {
        Self {
            agent_id: agent_id.into(),
            target_agent_id: target_agent_id.into(),
            permissions,
            created_at,
        }
    }
}

/// Filtering criteria for graph clone operations.
///
/// When cloning an agent's graph into a new namespace, these filters
/// control which nodes and edges are included in the copy.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct CloneFilter {
    /// If set, only include nodes of these types.
    pub node_types: Option<Vec<crate::types::NodeType>>,
    /// If set, only include nodes with timestamp >= this value.
    pub time_range_start: Option<u64>,
    /// If set, only include nodes with timestamp <= this value.
    pub time_range_end: Option<u64>,
}

/// Result of a graph clone operation between two agent namespaces.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct CloneResult {
    /// Number of nodes copied to the destination namespace.
    pub nodes_copied: usize,
    /// Number of edges copied to the destination namespace.
    pub edges_copied: usize,
    /// Mapping from original node IDs to new node IDs in the destination.
    pub id_map: std::collections::HashMap<u64, u64>,
}

/// Result of a graph merge operation between two agent namespaces.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct MergeResult {
    /// Number of nodes copied from the source namespace.
    pub nodes_copied: usize,
    /// Number of edges copied from the source namespace.
    pub edges_copied: usize,
    /// Number of duplicate nodes that were deduplicated during merge.
    pub nodes_deduplicated: usize,
    /// Number of node IDs that had to be remapped to avoid conflicts.
    pub ids_remapped: usize,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{NodeType, Value};

    #[test]
    fn test_agent_new_defaults() {
        let agent = Agent::new("bot-1", "My Bot", "user-42", 1700000000);
        assert_eq!(agent.id, "bot-1");
        assert_eq!(agent.name, "My Bot");
        assert_eq!(agent.namespace, "agent_bot-1");
        assert_eq!(agent.owner, "user-42");
        assert_eq!(agent.created_at, 1700000000);
        assert!(agent.config.is_empty());
    }

    #[test]
    fn test_agent_with_namespace() {
        let agent =
            Agent::new("bot-1", "My Bot", "user-42", 1700000000).with_namespace("custom-ns");
        assert_eq!(agent.namespace, "custom-ns");
    }

    #[test]
    fn test_agent_with_config() {
        let mut config = HashMap::new();
        config.insert("model".to_string(), Value::String("qwen-2.5".to_string()));
        config.insert("temperature".to_string(), Value::Float(0.7));

        let agent = Agent::new("bot-1", "Bot", "owner", 0).with_config(config.clone());
        assert_eq!(agent.config.len(), 2);
        assert_eq!(
            agent.config.get("model"),
            Some(&Value::String("qwen-2.5".to_string()))
        );
    }

    #[test]
    fn test_agent_serialization() {
        let agent = Agent::new("test-agent", "Test Agent", "owner-1", 1700000000);
        let serialized = bincode::serialize(&agent).expect("serialize Agent");
        let deserialized: Agent = bincode::deserialize(&serialized).expect("deserialize Agent");
        assert_eq!(agent, deserialized);
    }

    #[test]
    fn test_agent_share_new() {
        let share = AgentShare::new("agent-a", "agent-b", SharePermission::ReadOnly, 1700000000);
        assert_eq!(share.agent_id, "agent-a");
        assert_eq!(share.target_agent_id, "agent-b");
        assert_eq!(share.permissions, SharePermission::ReadOnly);
        assert_eq!(share.created_at, 1700000000);
    }

    #[test]
    fn test_agent_share_serialization() {
        let share = AgentShare::new("a", "b", SharePermission::ReadWrite, 123456);
        let serialized = bincode::serialize(&share).expect("serialize AgentShare");
        let deserialized: AgentShare =
            bincode::deserialize(&serialized).expect("deserialize AgentShare");
        assert_eq!(share, deserialized);
    }

    #[test]
    fn test_share_permission_variants() {
        assert_ne!(SharePermission::ReadOnly, SharePermission::ReadWrite);

        let ro_bytes = bincode::serialize(&SharePermission::ReadOnly).unwrap();
        let rw_bytes = bincode::serialize(&SharePermission::ReadWrite).unwrap();
        assert_ne!(ro_bytes, rw_bytes);
    }

    #[test]
    fn test_clone_filter_default() {
        let filter = CloneFilter::default();
        assert!(filter.node_types.is_none());
        assert!(filter.time_range_start.is_none());
        assert!(filter.time_range_end.is_none());
    }

    #[test]
    fn test_clone_filter_serialization() {
        let filter = CloneFilter {
            node_types: Some(vec![NodeType::Entity, NodeType::Event]),
            time_range_start: Some(1000),
            time_range_end: Some(2000),
        };
        let serialized = bincode::serialize(&filter).expect("serialize CloneFilter");
        let deserialized: CloneFilter =
            bincode::deserialize(&serialized).expect("deserialize CloneFilter");
        assert_eq!(filter, deserialized);
    }

    #[test]
    fn test_merge_result_default() {
        let result = MergeResult::default();
        assert_eq!(result.nodes_copied, 0);
        assert_eq!(result.edges_copied, 0);
        assert_eq!(result.nodes_deduplicated, 0);
        assert_eq!(result.ids_remapped, 0);
    }

    #[test]
    fn test_merge_result_serialization() {
        let result = MergeResult {
            nodes_copied: 100,
            edges_copied: 250,
            nodes_deduplicated: 5,
            ids_remapped: 3,
        };
        let serialized = bincode::serialize(&result).expect("serialize MergeResult");
        let deserialized: MergeResult =
            bincode::deserialize(&serialized).expect("deserialize MergeResult");
        assert_eq!(result, deserialized);
    }
}
