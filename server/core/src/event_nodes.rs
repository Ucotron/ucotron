//! Event Nodes — Proto-hypergraph simulation on property graphs.
//!
//! An Event Node represents a multi-dimensional fact or experience that involves
//! multiple entities. Instead of using true hyperedges (which property graphs
//! don't support), we create a central node of type [`NodeType::Event`] and
//! connect it to participating entities via typed edges:
//!
//! - [`EdgeType::Actor`] — the acting entity (e.g., "Juan")
//! - [`EdgeType::Object`] — the object/patient (e.g., "Pizza")
//! - [`EdgeType::Location`] — where it happened (e.g., "Plaza Mayor")
//! - [`EdgeType::Companion`] — accompanying entity (e.g., "María")
//!
//! The event node carries the **full-sentence embedding** of the complete event
//! description, enabling semantic retrieval. The typed edges enable structural
//! (graph) traversal from any participant back to the event.
//!
//! # Example
//!
//! ```
//! use ucotron_core::event_nodes::{EventNodeBuilder, Participant, ParticipantRole};
//! use ucotron_core::{EdgeType, NodeType};
//!
//! let builder = EventNodeBuilder::new(
//!     100,                          // event node id
//!     "Juan ate pizza at Plaza Mayor with María at 2PM",
//!     vec![0.1f32; 384],           // full-sentence embedding
//!     1_700_000_000,                // timestamp
//! )
//! .participant(Participant { node_id: 1, role: ParticipantRole::Actor })
//! .participant(Participant { node_id: 2, role: ParticipantRole::Object })
//! .participant(Participant { node_id: 3, role: ParticipantRole::Location })
//! .participant(Participant { node_id: 4, role: ParticipantRole::Companion });
//!
//! let (event_node, edges) = builder.build();
//!
//! assert_eq!(event_node.node_type, NodeType::Event);
//! assert_eq!(edges.len(), 4);
//! assert_eq!(edges[0].edge_type, EdgeType::Actor);
//! ```

use crate::{Edge, EdgeType, Node, NodeId, NodeType};
use std::collections::HashMap;

/// Role of an entity participating in an event.
///
/// Maps directly to the semantic edge types used to connect the event node
/// to its participants.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParticipantRole {
    /// The acting entity (maps to [`EdgeType::Actor`]).
    Actor,
    /// The object or patient of the action (maps to [`EdgeType::Object`]).
    Object,
    /// The location where the event occurred (maps to [`EdgeType::Location`]).
    Location,
    /// An accompanying entity (maps to [`EdgeType::Companion`]).
    Companion,
}

impl ParticipantRole {
    /// Convert this role to the corresponding [`EdgeType`].
    pub fn to_edge_type(self) -> EdgeType {
        match self {
            ParticipantRole::Actor => EdgeType::Actor,
            ParticipantRole::Object => EdgeType::Object,
            ParticipantRole::Location => EdgeType::Location,
            ParticipantRole::Companion => EdgeType::Companion,
        }
    }
}

/// A participant in an event, identified by node ID and role.
#[derive(Debug, Clone)]
pub struct Participant {
    /// ID of the entity node participating in this event.
    pub node_id: NodeId,
    /// Role of this entity in the event.
    pub role: ParticipantRole,
}

/// Builder for constructing an Event Node and its connecting edges.
///
/// Creates a central [`NodeType::Event`] node with a full-sentence embedding,
/// connected to participant entity nodes via typed edges.
pub struct EventNodeBuilder {
    id: NodeId,
    content: String,
    embedding: Vec<f32>,
    timestamp: u64,
    participants: Vec<Participant>,
}

impl EventNodeBuilder {
    /// Create a new event node builder.
    ///
    /// - `id`: unique ID for the event node
    /// - `content`: full-sentence description of the event
    /// - `embedding`: 384-dim embedding of the full sentence
    /// - `timestamp`: when the event occurred
    pub fn new(
        id: NodeId,
        content: impl Into<String>,
        embedding: Vec<f32>,
        timestamp: u64,
    ) -> Self {
        Self {
            id,
            content: content.into(),
            embedding,
            timestamp,
            participants: Vec::new(),
        }
    }

    /// Add a participant entity to this event.
    pub fn participant(mut self, p: Participant) -> Self {
        self.participants.push(p);
        self
    }

    /// Build the event node and its connecting edges.
    ///
    /// Returns `(event_node, edges)` where each edge connects the event node
    /// to a participant with the appropriate role-based edge type. Edges are
    /// directed from the event node to the participant.
    pub fn build(self) -> (Node, Vec<Edge>) {
        let event_node = Node {
            id: self.id,
            content: self.content,
            embedding: self.embedding,
            metadata: HashMap::new(),
            node_type: NodeType::Event,
            timestamp: self.timestamp,
            media_type: None,
            media_uri: None,
            embedding_visual: None,
            timestamp_range: None,
            parent_video_id: None,
        };

        let edges: Vec<Edge> = self
            .participants
            .iter()
            .map(|p| Edge {
                source: self.id,
                target: p.node_id,
                edge_type: p.role.to_edge_type(),
                weight: 1.0,
                metadata: HashMap::new(),
            })
            .collect();

        (event_node, edges)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Config, InsertStats, StorageEngine};
    use anyhow::Result;

    // --- MockEngine (same pattern as hybrid.rs) ---

    struct MockEngine {
        nodes: HashMap<NodeId, Node>,
        adj: HashMap<NodeId, Vec<(NodeId, EdgeType)>>,
    }

    impl MockEngine {
        fn new() -> Self {
            Self {
                nodes: HashMap::new(),
                adj: HashMap::new(),
            }
        }
    }

    impl StorageEngine for MockEngine {
        fn init(_config: &Config) -> Result<Self> {
            Ok(Self::new())
        }

        fn insert_nodes(&mut self, nodes: &[Node]) -> Result<InsertStats> {
            for node in nodes {
                self.nodes.insert(node.id, node.clone());
            }
            Ok(InsertStats {
                count: nodes.len(),
                duration_us: 0,
            })
        }

        fn insert_edges(&mut self, edges: &[Edge]) -> Result<InsertStats> {
            for edge in edges {
                self.adj
                    .entry(edge.source)
                    .or_default()
                    .push((edge.target, edge.edge_type));
                self.adj
                    .entry(edge.target)
                    .or_default()
                    .push((edge.source, edge.edge_type));
            }
            Ok(InsertStats {
                count: edges.len(),
                duration_us: 0,
            })
        }

        fn get_node(&self, id: NodeId) -> Result<Option<Node>> {
            Ok(self.nodes.get(&id).cloned())
        }

        fn get_neighbors(&self, id: NodeId, hops: u8) -> Result<Vec<Node>> {
            if hops == 0 {
                return Ok(Vec::new());
            }
            let mut result = Vec::new();
            if let Some(neighbors) = self.adj.get(&id) {
                for &(nid, _) in neighbors {
                    if let Some(node) = self.nodes.get(&nid) {
                        result.push(node.clone());
                    }
                }
            }
            Ok(result)
        }

        fn vector_search(&self, query: &[f32], top_k: usize) -> Result<Vec<(NodeId, f32)>> {
            let mut scored: Vec<(NodeId, f32)> = self
                .nodes
                .values()
                .map(|n| {
                    let sim: f32 = n
                        .embedding
                        .iter()
                        .zip(query.iter())
                        .map(|(a, b)| a * b)
                        .sum();
                    (n.id, sim)
                })
                .collect();
            scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            scored.truncate(top_k);
            Ok(scored)
        }

        fn hybrid_search(&self, query: &[f32], top_k: usize, hops: u8) -> Result<Vec<Node>> {
            crate::find_related(self, query, top_k, hops, crate::DEFAULT_HOP_DECAY)
        }

        fn find_path(&self, _source: NodeId, _target: NodeId, _max_depth: u32) -> Result<Option<Vec<NodeId>>> {
            Ok(None) // not needed for event node tests
        }

        fn shutdown(&mut self) -> Result<()> {
            Ok(())
        }
    }

    fn make_entity(id: NodeId, content: &str, embedding: Vec<f32>) -> Node {
        Node {
            id,
            content: content.to_string(),
            embedding,
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

    /// Create a normalized embedding with a dominant component at `idx`.
    fn dominant_embedding(dim: usize, idx: usize, strength: f32) -> Vec<f32> {
        let mut v = vec![0.01f32; dim];
        v[idx] = strength;
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        v.iter_mut().for_each(|x| *x /= norm);
        v
    }

    // --- Builder tests ---

    #[test]
    fn test_event_node_builder_basic() {
        let embedding = vec![0.1f32; 384];
        let (event, edges) = EventNodeBuilder::new(100, "Juan ate pizza", embedding.clone(), 1000)
            .participant(Participant {
                node_id: 1,
                role: ParticipantRole::Actor,
            })
            .participant(Participant {
                node_id: 2,
                role: ParticipantRole::Object,
            })
            .build();

        assert_eq!(event.id, 100);
        assert_eq!(event.node_type, NodeType::Event);
        assert_eq!(event.content, "Juan ate pizza");
        assert_eq!(event.timestamp, 1000);
        assert_eq!(event.embedding.len(), 384);

        assert_eq!(edges.len(), 2);
        assert_eq!(edges[0].source, 100);
        assert_eq!(edges[0].target, 1);
        assert_eq!(edges[0].edge_type, EdgeType::Actor);
        assert_eq!(edges[1].source, 100);
        assert_eq!(edges[1].target, 2);
        assert_eq!(edges[1].edge_type, EdgeType::Object);
    }

    #[test]
    fn test_event_node_all_roles() {
        let (event, edges) =
            EventNodeBuilder::new(200, "Full event", vec![0.5f32; 384], 2000)
                .participant(Participant {
                    node_id: 10,
                    role: ParticipantRole::Actor,
                })
                .participant(Participant {
                    node_id: 20,
                    role: ParticipantRole::Object,
                })
                .participant(Participant {
                    node_id: 30,
                    role: ParticipantRole::Location,
                })
                .participant(Participant {
                    node_id: 40,
                    role: ParticipantRole::Companion,
                })
                .build();

        assert_eq!(event.node_type, NodeType::Event);
        assert_eq!(edges.len(), 4);

        let edge_types: Vec<EdgeType> = edges.iter().map(|e| e.edge_type).collect();
        assert_eq!(
            edge_types,
            vec![
                EdgeType::Actor,
                EdgeType::Object,
                EdgeType::Location,
                EdgeType::Companion,
            ]
        );

        // All edges have weight 1.0
        assert!(edges.iter().all(|e| (e.weight - 1.0).abs() < 1e-6));
    }

    #[test]
    fn test_event_node_no_participants() {
        let (event, edges) =
            EventNodeBuilder::new(300, "Lonely event", vec![0.1f32; 384], 3000).build();

        assert_eq!(event.id, 300);
        assert_eq!(event.node_type, NodeType::Event);
        assert!(edges.is_empty());
    }

    #[test]
    fn test_participant_role_to_edge_type() {
        assert_eq!(ParticipantRole::Actor.to_edge_type(), EdgeType::Actor);
        assert_eq!(ParticipantRole::Object.to_edge_type(), EdgeType::Object);
        assert_eq!(ParticipantRole::Location.to_edge_type(), EdgeType::Location);
        assert_eq!(
            ParticipantRole::Companion.to_edge_type(),
            EdgeType::Companion
        );
    }

    // --- Integration tests: Event Nodes reachable via both vector search and graph traversal ---

    #[test]
    fn test_event_reachable_via_vector_search() {
        // PRD test: vector search for "comida en plaza" retrieves the event node
        let mut engine = MockEngine::new();

        // Entity nodes with unrelated embeddings
        engine
            .insert_nodes(&[
                make_entity(1, "Juan", dominant_embedding(384, 0, 1.0)),
                make_entity(2, "Pizza", dominant_embedding(384, 1, 1.0)),
                make_entity(3, "Plaza Mayor", dominant_embedding(384, 2, 1.0)),
                make_entity(4, "María", dominant_embedding(384, 3, 1.0)),
            ])
            .unwrap();

        // Event node: embedding encodes "comida en plaza" semantics (dominant at idx 5)
        let event_embedding = dominant_embedding(384, 5, 1.0);
        let (event_node, edges) = EventNodeBuilder::new(
            100,
            "Juan comió pizza en la Plaza Mayor con María a las 2PM",
            event_embedding.clone(),
            1_700_000_000,
        )
        .participant(Participant {
            node_id: 1,
            role: ParticipantRole::Actor,
        })
        .participant(Participant {
            node_id: 2,
            role: ParticipantRole::Object,
        })
        .participant(Participant {
            node_id: 3,
            role: ParticipantRole::Location,
        })
        .participant(Participant {
            node_id: 4,
            role: ParticipantRole::Companion,
        })
        .build();

        engine.insert_nodes(&[event_node]).unwrap();
        engine.insert_edges(&edges).unwrap();

        // Query embedding similar to event node (dominant at idx 5)
        let query = dominant_embedding(384, 5, 1.0);
        let results = engine.vector_search(&query, 3).unwrap();

        // The event node should be the top result
        assert!(!results.is_empty());
        assert_eq!(
            results[0].0, 100,
            "Event node should be the top vector search result"
        );
        assert!(
            results[0].1 > 0.9,
            "Event node similarity should be very high"
        );
    }

    #[test]
    fn test_event_reachable_via_graph_traversal() {
        // PRD test: traversal from "Juan" (node 1) reaches the event node (node 100)
        let mut engine = MockEngine::new();

        engine
            .insert_nodes(&[
                make_entity(1, "Juan", dominant_embedding(384, 0, 1.0)),
                make_entity(2, "Pizza", dominant_embedding(384, 1, 1.0)),
                make_entity(3, "Plaza Mayor", dominant_embedding(384, 2, 1.0)),
                make_entity(4, "María", dominant_embedding(384, 3, 1.0)),
            ])
            .unwrap();

        let (event_node, edges) = EventNodeBuilder::new(
            100,
            "Juan comió pizza en la Plaza Mayor con María a las 2PM",
            dominant_embedding(384, 5, 1.0),
            1_700_000_000,
        )
        .participant(Participant {
            node_id: 1,
            role: ParticipantRole::Actor,
        })
        .participant(Participant {
            node_id: 2,
            role: ParticipantRole::Object,
        })
        .participant(Participant {
            node_id: 3,
            role: ParticipantRole::Location,
        })
        .participant(Participant {
            node_id: 4,
            role: ParticipantRole::Companion,
        })
        .build();

        engine.insert_nodes(&[event_node]).unwrap();
        engine.insert_edges(&edges).unwrap();

        // Traverse 1 hop from Juan → should reach event node 100
        let neighbors = engine.get_neighbors(1, 1).unwrap();
        let neighbor_ids: Vec<NodeId> = neighbors.iter().map(|n| n.id).collect();
        assert!(
            neighbor_ids.contains(&100),
            "Event node should be reachable from Juan via 1-hop traversal"
        );

        // From event node 100 → should reach all participants
        let event_neighbors = engine.get_neighbors(100, 1).unwrap();
        let event_neighbor_ids: Vec<NodeId> = event_neighbors.iter().map(|n| n.id).collect();
        assert!(event_neighbor_ids.contains(&1), "Juan should be reachable from event");
        assert!(event_neighbor_ids.contains(&2), "Pizza should be reachable from event");
        assert!(event_neighbor_ids.contains(&3), "Plaza Mayor should be reachable from event");
        assert!(event_neighbor_ids.contains(&4), "María should be reachable from event");
    }

    #[test]
    fn test_event_reachable_via_both_paths() {
        // Combined test: event is found by both vector search AND graph traversal
        let mut engine = MockEngine::new();

        engine
            .insert_nodes(&[
                make_entity(1, "Juan", dominant_embedding(384, 0, 1.0)),
                make_entity(2, "Pizza", dominant_embedding(384, 1, 1.0)),
                make_entity(3, "Plaza Mayor", dominant_embedding(384, 2, 1.0)),
                make_entity(4, "María", dominant_embedding(384, 3, 1.0)),
            ])
            .unwrap();

        let event_embedding = dominant_embedding(384, 5, 1.0);
        let (event_node, edges) = EventNodeBuilder::new(
            100,
            "Juan comió pizza en la Plaza Mayor con María a las 2PM",
            event_embedding,
            1_700_000_000,
        )
        .participant(Participant {
            node_id: 1,
            role: ParticipantRole::Actor,
        })
        .participant(Participant {
            node_id: 2,
            role: ParticipantRole::Object,
        })
        .participant(Participant {
            node_id: 3,
            role: ParticipantRole::Location,
        })
        .participant(Participant {
            node_id: 4,
            role: ParticipantRole::Companion,
        })
        .build();

        engine.insert_nodes(&[event_node]).unwrap();
        engine.insert_edges(&edges).unwrap();

        // Vector search path: query similar to event embedding
        let query = dominant_embedding(384, 5, 1.0);
        let vector_results = engine.vector_search(&query, 5).unwrap();
        let vector_ids: Vec<NodeId> = vector_results.iter().map(|r| r.0).collect();
        assert!(
            vector_ids.contains(&100),
            "Event node must be reachable via vector search"
        );

        // Graph traversal path: from Juan → event
        let graph_results = engine.get_neighbors(1, 1).unwrap();
        let graph_ids: Vec<NodeId> = graph_results.iter().map(|n| n.id).collect();
        assert!(
            graph_ids.contains(&100),
            "Event node must be reachable via graph traversal from Juan"
        );

        // Hybrid search should also find it
        let hybrid_results = engine.hybrid_search(&query, 3, 1).unwrap();
        let hybrid_ids: Vec<NodeId> = hybrid_results.iter().map(|n| n.id).collect();
        assert!(
            hybrid_ids.contains(&100),
            "Event node must be reachable via hybrid search"
        );
    }

    #[test]
    fn test_event_cross_entity_traversal() {
        // Verify that traversal from Juan can reach María via the event node (2 hops)
        let mut engine = MockEngine::new();

        engine
            .insert_nodes(&[
                make_entity(1, "Juan", dominant_embedding(384, 0, 1.0)),
                make_entity(4, "María", dominant_embedding(384, 3, 1.0)),
            ])
            .unwrap();

        let (event_node, edges) = EventNodeBuilder::new(
            100,
            "Juan y María comieron juntos",
            dominant_embedding(384, 5, 1.0),
            1_700_000_000,
        )
        .participant(Participant {
            node_id: 1,
            role: ParticipantRole::Actor,
        })
        .participant(Participant {
            node_id: 4,
            role: ParticipantRole::Companion,
        })
        .build();

        engine.insert_nodes(&[event_node]).unwrap();
        engine.insert_edges(&edges).unwrap();

        // 1 hop from Juan → event node
        let hop1 = engine.get_neighbors(1, 1).unwrap();
        let hop1_ids: Vec<NodeId> = hop1.iter().map(|n| n.id).collect();
        assert!(hop1_ids.contains(&100), "Event reachable from Juan at 1 hop");
        assert!(
            !hop1_ids.contains(&4),
            "María should NOT be reachable from Juan at 1 hop"
        );

        // From event node → María is reachable at 1 hop
        let from_event = engine.get_neighbors(100, 1).unwrap();
        let from_event_ids: Vec<NodeId> = from_event.iter().map(|n| n.id).collect();
        assert!(
            from_event_ids.contains(&4),
            "María reachable from event at 1 hop"
        );
    }
}
