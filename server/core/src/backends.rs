//! # Pluggable Backend Traits
//!
//! Defines the `VectorBackend` and `GraphBackend` traits for Phase 2's
//! pluggable storage architecture. These traits decouple vector search
//! and graph storage into independent backends that can be swapped via
//! configuration.
//!
//! # Architecture
//!
//! The [`BackendRegistry`] holds boxed trait objects for both backends,
//! providing a unified entry point for all storage operations. Factory
//! functions in the engine crates (e.g., `helix_impl`) create backends
//! from configuration.
//!
//! # Default Backends
//!
//! - `HelixVectorBackend`: LMDB + brute-force SIMD (Phase 1), HNSW (Phase 2)
//! - `HelixGraphBackend`: LMDB adjacency lists (Phase 1)
//!
//! # External Backends (Stubs)
//!
//! - [`ExternalVectorBackend`]: Stub for future Qdrant/external vector backends
//! - FalkorDB (graph, future)

use crate::{Edge, Node, NodeId};

/// Backend for vector similarity search operations.
///
/// Separates vector indexing from graph storage, enabling independent
/// scaling and backend selection. For example, LMDB-based HNSW for
/// embedded mode or Qdrant for external mode.
pub trait VectorBackend: Send + Sync {
    /// Upsert (insert or update) embedding vectors for the given node IDs.
    fn upsert_embeddings(&self, items: &[(NodeId, Vec<f32>)]) -> anyhow::Result<()>;

    /// Search for the top-k most similar vectors to the query.
    ///
    /// Returns `(node_id, similarity_score)` pairs sorted by descending similarity.
    fn search(&self, query: &[f32], top_k: usize) -> anyhow::Result<Vec<(NodeId, f32)>>;

    /// Delete embedding vectors for the given node IDs.
    fn delete(&self, ids: &[NodeId]) -> anyhow::Result<()>;
}

/// Backend for graph storage and traversal operations.
///
/// Separates graph operations from vector search, enabling independent
/// scaling and backend selection. For example, LMDB adjacency lists for
/// embedded mode or FalkorDB for external mode.
pub trait GraphBackend: Send + Sync {
    /// Upsert (insert or update) nodes in the graph.
    fn upsert_nodes(&self, nodes: &[Node]) -> anyhow::Result<()>;

    /// Upsert (insert or update) edges in the graph.
    fn upsert_edges(&self, edges: &[Edge]) -> anyhow::Result<()>;

    /// Retrieve a single node by its ID.
    fn get_node(&self, id: NodeId) -> anyhow::Result<Option<Node>>;

    /// Get all nodes reachable within `hops` steps from the given node.
    fn get_neighbors(&self, id: NodeId, hops: u8) -> anyhow::Result<Vec<Node>>;

    /// Find the shortest path between two nodes, up to `max_depth` hops.
    ///
    /// Returns `None` if no path exists within the depth limit.
    fn find_path(
        &self,
        source: NodeId,
        target: NodeId,
        max_depth: u32,
    ) -> anyhow::Result<Option<Vec<NodeId>>>;

    /// Get all nodes in the same community as the given node.
    ///
    /// Returns an empty vec if communities have not been computed yet.
    fn get_community(&self, node_id: NodeId) -> anyhow::Result<Vec<NodeId>>;

    /// Return all nodes in the graph.
    ///
    /// Used by export operations. This may be expensive for very large graphs;
    /// callers should stream or paginate if possible.
    fn get_all_nodes(&self) -> anyhow::Result<Vec<Node>>;

    /// Return all edges as `(source, target, weight)` tuples.
    ///
    /// Used by graph-wide algorithms like community detection. This may be
    /// expensive for very large graphs; callers should cache results.
    fn get_all_edges(&self) -> anyhow::Result<Vec<(NodeId, NodeId, f32)>>;

    /// Return all edges with full metadata.
    ///
    /// Used by export operations that need edge types and metadata.
    /// Default implementation uses `get_all_edges()` with minimal data.
    fn get_all_edges_full(&self) -> anyhow::Result<Vec<Edge>> {
        // Default: return edges with only source/target/weight
        let edges = self.get_all_edges()?;
        Ok(edges
            .into_iter()
            .map(|(src, tgt, weight)| Edge {
                source: src,
                target: tgt,
                edge_type: crate::EdgeType::RelatesTo,
                weight,
                metadata: std::collections::HashMap::new(),
            })
            .collect())
    }

    /// Delete nodes and all their associated edges from the graph.
    ///
    /// Removes nodes from node storage, adjacency lists, type indices,
    /// community assignments, and all edges where the node is source or target.
    /// This is a hard delete — data is permanently removed.
    fn delete_nodes(&self, ids: &[NodeId]) -> anyhow::Result<()>;

    /// Store community assignments computed by community detection.
    ///
    /// Takes a mapping from node ID to community ID and persists it so that
    /// future calls to `get_community` can return the correct members.
    fn store_community_assignments(
        &self,
        assignments: &std::collections::HashMap<NodeId, crate::community::CommunityId>,
    ) -> anyhow::Result<()>;

    // ----- Agent CRUD (default: not supported) -----

    /// Create or update an agent record.
    ///
    /// Backends that support agent management should persist the agent and
    /// its auto-generated namespace. Default returns "not supported".
    fn create_agent(&self, _agent: &crate::Agent) -> anyhow::Result<()> {
        anyhow::bail!("Agent management not supported by this graph backend")
    }

    /// Retrieve an agent by its ID.
    fn get_agent(&self, _id: &str) -> anyhow::Result<Option<crate::Agent>> {
        anyhow::bail!("Agent management not supported by this graph backend")
    }

    /// List all agents, optionally filtered by owner.
    fn list_agents(&self, _owner: Option<&str>) -> anyhow::Result<Vec<crate::Agent>> {
        anyhow::bail!("Agent management not supported by this graph backend")
    }

    /// Delete an agent and cascade-delete its share grants.
    fn delete_agent(&self, _id: &str) -> anyhow::Result<()> {
        anyhow::bail!("Agent management not supported by this graph backend")
    }

    // ----- Agent Share (default: not supported) -----

    /// Create a share grant from one agent to another.
    fn create_share(&self, _share: &crate::AgentShare) -> anyhow::Result<()> {
        anyhow::bail!("Agent sharing not supported by this graph backend")
    }

    /// Get a specific share grant between two agents.
    fn get_share(
        &self,
        _agent_id: &str,
        _target_id: &str,
    ) -> anyhow::Result<Option<crate::AgentShare>> {
        anyhow::bail!("Agent sharing not supported by this graph backend")
    }

    /// List all shares granted by a specific agent.
    fn list_shares(&self, _agent_id: &str) -> anyhow::Result<Vec<crate::AgentShare>> {
        anyhow::bail!("Agent sharing not supported by this graph backend")
    }

    /// Delete a specific share grant between two agents.
    fn delete_share(&self, _agent_id: &str, _target_id: &str) -> anyhow::Result<()> {
        anyhow::bail!("Agent sharing not supported by this graph backend")
    }

    /// Clone nodes and edges from one namespace to another, applying optional filters.
    ///
    /// Copies all nodes matching the filter from `src_ns` into `dst_ns`, remapping
    /// node IDs to avoid collisions. Edges between cloned nodes are also copied.
    /// Returns a mapping from old node IDs to new node IDs.
    fn clone_graph(
        &self,
        _src_ns: &str,
        _dst_ns: &str,
        _filter: &crate::CloneFilter,
        _id_start: u64,
    ) -> anyhow::Result<crate::CloneResult> {
        anyhow::bail!("Graph clone not supported by this graph backend")
    }

    /// Merge nodes and edges from one namespace into another with entity deduplication.
    ///
    /// Unlike `clone_graph`, merge compares source nodes against destination nodes
    /// by content. Nodes with identical content are considered duplicates — their
    /// edges are redirected to the existing destination node instead of creating
    /// a new copy. Non-duplicate nodes are assigned new IDs starting from `id_start`.
    fn merge_graph(
        &self,
        _src_ns: &str,
        _dst_ns: &str,
        _id_start: u64,
    ) -> anyhow::Result<crate::MergeResult> {
        anyhow::bail!("Graph merge not supported by this graph backend")
    }
}

/// Backend for visual (CLIP) vector similarity search operations.
///
/// Operates on 512-dim CLIP embeddings in a separate HNSW index from the
/// text embeddings (384-dim). This dual-index design enables cross-modal
/// search: text→image via CLIP text encoder, image→text via projection layer.
pub trait VisualVectorBackend: Send + Sync {
    /// Upsert (insert or update) visual embedding vectors for the given node IDs.
    ///
    /// Vectors are expected to be 512-dim CLIP embeddings (L2-normalized).
    fn upsert_visual_embeddings(&self, items: &[(NodeId, Vec<f32>)]) -> anyhow::Result<()>;

    /// Search for the top-k most similar visual vectors to the query.
    ///
    /// Returns `(node_id, similarity_score)` pairs sorted by descending similarity.
    fn search_visual(&self, query: &[f32], top_k: usize) -> anyhow::Result<Vec<(NodeId, f32)>>;

    /// Delete visual embedding vectors for the given node IDs.
    fn delete_visual(&self, ids: &[NodeId]) -> anyhow::Result<()>;
}

/// Holds instantiated backends for the storage layer.
///
/// The registry is the single entry point for all storage operations in
/// Phase 2. It owns boxed trait objects for the vector, graph, and
/// optional visual vector backends, enabling runtime selection via configuration.
///
/// # Example
///
/// ```ignore
/// let registry = BackendRegistry::new(
///     Box::new(helix_vector_backend),
///     Box::new(helix_graph_backend),
/// );
/// registry.vector().search(&query, 10)?;
/// registry.graph().get_node(42)?;
///
/// // With visual backend for multimodal search:
/// let registry = BackendRegistry::with_visual(
///     Box::new(text_backend),
///     Box::new(graph_backend),
///     Box::new(visual_backend),
/// );
/// registry.visual().unwrap().search_visual(&clip_query, 10)?;
/// ```
pub struct BackendRegistry {
    vector: Box<dyn VectorBackend>,
    graph: Box<dyn GraphBackend>,
    visual: Option<Box<dyn VisualVectorBackend>>,
}

impl BackendRegistry {
    /// Create a new registry with text vector and graph backends (no visual).
    pub fn new(vector: Box<dyn VectorBackend>, graph: Box<dyn GraphBackend>) -> Self {
        Self {
            vector,
            graph,
            visual: None,
        }
    }

    /// Create a registry with text vector, graph, and visual vector backends.
    pub fn with_visual(
        vector: Box<dyn VectorBackend>,
        graph: Box<dyn GraphBackend>,
        visual: Box<dyn VisualVectorBackend>,
    ) -> Self {
        Self {
            vector,
            graph,
            visual: Some(visual),
        }
    }

    /// Access the text vector backend (384-dim MiniLM).
    pub fn vector(&self) -> &dyn VectorBackend {
        self.vector.as_ref()
    }

    /// Access the graph backend.
    pub fn graph(&self) -> &dyn GraphBackend {
        self.graph.as_ref()
    }

    /// Access the visual vector backend (512-dim CLIP), if configured.
    ///
    /// Returns `None` when multimodal support is not enabled.
    pub fn visual(&self) -> Option<&dyn VisualVectorBackend> {
        self.visual.as_deref()
    }
}

/// Stub implementation of `VectorBackend` for external vector services.
///
/// This is a placeholder for future Qdrant integration. It stores the
/// configured URL but all operations return `Err` indicating the backend
/// is not yet implemented.
///
/// When implementing a real external backend, replace the method bodies
/// with HTTP/gRPC calls to the external service.
pub struct ExternalVectorBackend {
    /// Service URL (e.g., "http://localhost:6333" for Qdrant).
    pub url: String,
    /// Collection name.
    pub collection: String,
}

impl ExternalVectorBackend {
    /// Create a new external vector backend stub.
    pub fn new(url: String, collection: String) -> Self {
        Self { url, collection }
    }
}

impl VectorBackend for ExternalVectorBackend {
    fn upsert_embeddings(&self, _items: &[(NodeId, Vec<f32>)]) -> anyhow::Result<()> {
        anyhow::bail!(
            "ExternalVectorBackend ({}) not yet implemented — configure storage.vector.backend = \"helix\"",
            self.url
        )
    }

    fn search(&self, _query: &[f32], _top_k: usize) -> anyhow::Result<Vec<(NodeId, f32)>> {
        anyhow::bail!(
            "ExternalVectorBackend ({}) not yet implemented — configure storage.vector.backend = \"helix\"",
            self.url
        )
    }

    fn delete(&self, _ids: &[NodeId]) -> anyhow::Result<()> {
        anyhow::bail!(
            "ExternalVectorBackend ({}) not yet implemented — configure storage.vector.backend = \"helix\"",
            self.url
        )
    }
}

/// Stub implementation of `GraphBackend` for external graph services.
///
/// Placeholder for future FalkorDB integration.
pub struct ExternalGraphBackend {
    /// Service URL (e.g., "redis://localhost:6379" for FalkorDB).
    pub url: String,
}

impl ExternalGraphBackend {
    /// Create a new external graph backend stub.
    pub fn new(url: String) -> Self {
        Self { url }
    }
}

impl GraphBackend for ExternalGraphBackend {
    fn upsert_nodes(&self, _nodes: &[Node]) -> anyhow::Result<()> {
        anyhow::bail!("ExternalGraphBackend ({}) not yet implemented", self.url)
    }

    fn upsert_edges(&self, _edges: &[Edge]) -> anyhow::Result<()> {
        anyhow::bail!("ExternalGraphBackend ({}) not yet implemented", self.url)
    }

    fn get_node(&self, _id: NodeId) -> anyhow::Result<Option<Node>> {
        anyhow::bail!("ExternalGraphBackend ({}) not yet implemented", self.url)
    }

    fn get_neighbors(&self, _id: NodeId, _hops: u8) -> anyhow::Result<Vec<Node>> {
        anyhow::bail!("ExternalGraphBackend ({}) not yet implemented", self.url)
    }

    fn find_path(
        &self,
        _source: NodeId,
        _target: NodeId,
        _max_depth: u32,
    ) -> anyhow::Result<Option<Vec<NodeId>>> {
        anyhow::bail!("ExternalGraphBackend ({}) not yet implemented", self.url)
    }

    fn get_community(&self, _node_id: NodeId) -> anyhow::Result<Vec<NodeId>> {
        anyhow::bail!("ExternalGraphBackend ({}) not yet implemented", self.url)
    }

    fn get_all_nodes(&self) -> anyhow::Result<Vec<Node>> {
        anyhow::bail!("ExternalGraphBackend ({}) not yet implemented", self.url)
    }

    fn get_all_edges(&self) -> anyhow::Result<Vec<(NodeId, NodeId, f32)>> {
        anyhow::bail!("ExternalGraphBackend ({}) not yet implemented", self.url)
    }

    fn delete_nodes(&self, _ids: &[NodeId]) -> anyhow::Result<()> {
        anyhow::bail!("ExternalGraphBackend ({}) not yet implemented", self.url)
    }

    fn store_community_assignments(
        &self,
        _assignments: &std::collections::HashMap<NodeId, crate::community::CommunityId>,
    ) -> anyhow::Result<()> {
        anyhow::bail!("ExternalGraphBackend ({}) not yet implemented", self.url)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{EdgeType, NodeType};
    use std::collections::HashMap;
    use std::sync::Mutex;

    /// In-memory mock vector backend for testing.
    struct MockVectorBackend {
        embeddings: Mutex<HashMap<NodeId, Vec<f32>>>,
    }

    impl MockVectorBackend {
        fn new() -> Self {
            Self {
                embeddings: Mutex::new(HashMap::new()),
            }
        }
    }

    impl VectorBackend for MockVectorBackend {
        fn upsert_embeddings(&self, items: &[(NodeId, Vec<f32>)]) -> anyhow::Result<()> {
            let mut map = self.embeddings.lock().unwrap();
            for (id, vec) in items {
                map.insert(*id, vec.clone());
            }
            Ok(())
        }

        fn search(&self, query: &[f32], top_k: usize) -> anyhow::Result<Vec<(NodeId, f32)>> {
            let map = self.embeddings.lock().unwrap();
            let mut scores: Vec<(NodeId, f32)> = map
                .iter()
                .map(|(id, emb)| {
                    let dot: f32 = query.iter().zip(emb.iter()).map(|(a, b)| a * b).sum();
                    (*id, dot)
                })
                .collect();
            scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            scores.truncate(top_k);
            Ok(scores)
        }

        fn delete(&self, ids: &[NodeId]) -> anyhow::Result<()> {
            let mut map = self.embeddings.lock().unwrap();
            for id in ids {
                map.remove(id);
            }
            Ok(())
        }
    }

    /// In-memory mock graph backend for testing.
    struct MockGraphBackend {
        nodes: Mutex<HashMap<NodeId, Node>>,
        adj: Mutex<HashMap<NodeId, Vec<NodeId>>>,
    }

    impl MockGraphBackend {
        fn new() -> Self {
            Self {
                nodes: Mutex::new(HashMap::new()),
                adj: Mutex::new(HashMap::new()),
            }
        }
    }

    impl GraphBackend for MockGraphBackend {
        fn upsert_nodes(&self, nodes: &[Node]) -> anyhow::Result<()> {
            let mut map = self.nodes.lock().unwrap();
            for node in nodes {
                map.insert(node.id, node.clone());
            }
            Ok(())
        }

        fn upsert_edges(&self, edges: &[Edge]) -> anyhow::Result<()> {
            let mut adj = self.adj.lock().unwrap();
            for edge in edges {
                adj.entry(edge.source)
                    .or_default()
                    .push(edge.target);
            }
            Ok(())
        }

        fn get_node(&self, id: NodeId) -> anyhow::Result<Option<Node>> {
            let map = self.nodes.lock().unwrap();
            Ok(map.get(&id).cloned())
        }

        fn get_neighbors(&self, id: NodeId, _hops: u8) -> anyhow::Result<Vec<Node>> {
            let adj = self.adj.lock().unwrap();
            let nodes = self.nodes.lock().unwrap();
            let neighbor_ids = adj.get(&id).cloned().unwrap_or_default();
            Ok(neighbor_ids
                .iter()
                .filter_map(|nid| nodes.get(nid).cloned())
                .collect())
        }

        fn find_path(
            &self,
            source: NodeId,
            target: NodeId,
            _max_depth: u32,
        ) -> anyhow::Result<Option<Vec<NodeId>>> {
            if source == target {
                return Ok(Some(vec![source]));
            }
            Ok(None) // Simplified for testing
        }

        fn get_community(&self, _node_id: NodeId) -> anyhow::Result<Vec<NodeId>> {
            Ok(Vec::new())
        }

        fn get_all_nodes(&self) -> anyhow::Result<Vec<Node>> {
            let map = self.nodes.lock().unwrap();
            Ok(map.values().cloned().collect())
        }

        fn get_all_edges(&self) -> anyhow::Result<Vec<(NodeId, NodeId, f32)>> {
            let adj = self.adj.lock().unwrap();
            let mut edges = Vec::new();
            for (&source, targets) in adj.iter() {
                for &target in targets {
                    edges.push((source, target, 1.0));
                }
            }
            Ok(edges)
        }

        fn delete_nodes(&self, ids: &[NodeId]) -> anyhow::Result<()> {
            let id_set: std::collections::HashSet<NodeId> = ids.iter().copied().collect();
            let mut nodes = self.nodes.lock().unwrap();
            let mut adj = self.adj.lock().unwrap();
            for id in ids {
                nodes.remove(id);
                adj.remove(id);
            }
            // Remove edges pointing to deleted nodes
            for targets in adj.values_mut() {
                targets.retain(|t| !id_set.contains(t));
            }
            Ok(())
        }

        fn store_community_assignments(
            &self,
            _assignments: &std::collections::HashMap<NodeId, crate::community::CommunityId>,
        ) -> anyhow::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn test_mock_vector_backend_upsert_and_search() {
        let backend = MockVectorBackend::new();

        // Insert 3 vectors
        let items: Vec<(NodeId, Vec<f32>)> = vec![
            (1, vec![1.0, 0.0, 0.0]),
            (2, vec![0.0, 1.0, 0.0]),
            (3, vec![0.7, 0.7, 0.0]),
        ];
        backend.upsert_embeddings(&items).unwrap();

        // Search for vec closest to [1, 0, 0]
        let results = backend.search(&[1.0, 0.0, 0.0], 2).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, 1); // exact match
        assert_eq!(results[1].0, 3); // partial match
    }

    #[test]
    fn test_mock_vector_backend_delete() {
        let backend = MockVectorBackend::new();
        backend
            .upsert_embeddings(&[(1, vec![1.0, 0.0]), (2, vec![0.0, 1.0])])
            .unwrap();

        backend.delete(&[1]).unwrap();

        let results = backend.search(&[1.0, 0.0], 10).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, 2);
    }

    #[test]
    fn test_vector_backend_is_object_safe() {
        // Verify VectorBackend can be used as a trait object
        let backend: Box<dyn VectorBackend> = Box::new(MockVectorBackend::new());
        backend.upsert_embeddings(&[(1, vec![1.0])]).unwrap();
        let results = backend.search(&[1.0], 1).unwrap();
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_graph_backend_is_object_safe() {
        // Verify GraphBackend can be used as a trait object
        let backend: Box<dyn GraphBackend> = Box::new(MockGraphBackend::new());
        let node = Node {
            id: 1,
            content: "test".to_string(),
            embedding: vec![],
            metadata: HashMap::new(),
            node_type: NodeType::Entity,
            timestamp: 100,
            media_type: None,
            media_uri: None,
            embedding_visual: None,
            timestamp_range: None,
            parent_video_id: None,
        };
        backend.upsert_nodes(&[node]).unwrap();
        let retrieved = backend.get_node(1).unwrap();
        assert!(retrieved.is_some());
    }

    #[test]
    fn test_backend_registry_creation() {
        let vec_backend = MockVectorBackend::new();
        let graph_backend = MockGraphBackend::new();

        let registry = BackendRegistry::new(Box::new(vec_backend), Box::new(graph_backend));

        // Use the vector backend via registry
        registry
            .vector()
            .upsert_embeddings(&[(1, vec![1.0, 0.0])])
            .unwrap();
        let results = registry.vector().search(&[1.0, 0.0], 1).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, 1);

        // Use the graph backend via registry
        let node = Node {
            id: 1,
            content: "test".to_string(),
            embedding: vec![],
            metadata: HashMap::new(),
            node_type: NodeType::Entity,
            timestamp: 100,
            media_type: None,
            media_uri: None,
            embedding_visual: None,
            timestamp_range: None,
            parent_video_id: None,
        };
        registry.graph().upsert_nodes(&[node]).unwrap();
        let retrieved = registry.graph().get_node(1).unwrap();
        assert!(retrieved.is_some());
    }

    #[test]
    fn test_backend_registry_combined_operations() {
        let vec_backend = MockVectorBackend::new();
        let graph_backend = MockGraphBackend::new();
        let registry = BackendRegistry::new(Box::new(vec_backend), Box::new(graph_backend));

        // Insert nodes and edges via graph backend
        let nodes = vec![
            Node {
                id: 1,
                content: "A".to_string(),
                embedding: vec![],
                metadata: HashMap::new(),
                node_type: NodeType::Entity,
                timestamp: 100,
                media_type: None,
                media_uri: None,
                embedding_visual: None,
                timestamp_range: None,
                parent_video_id: None,
            },
            Node {
                id: 2,
                content: "B".to_string(),
                embedding: vec![],
                metadata: HashMap::new(),
                node_type: NodeType::Entity,
                timestamp: 200,
                media_type: None,
                media_uri: None,
                embedding_visual: None,
                timestamp_range: None,
                parent_video_id: None,
            },
        ];
        registry.graph().upsert_nodes(&nodes).unwrap();
        registry
            .graph()
            .upsert_edges(&[Edge {
                source: 1,
                target: 2,
                edge_type: EdgeType::RelatesTo,
                weight: 1.0,
                metadata: HashMap::new(),
            }])
            .unwrap();

        // Insert embeddings via vector backend
        registry
            .vector()
            .upsert_embeddings(&[(1, vec![1.0, 0.0]), (2, vec![0.0, 1.0])])
            .unwrap();

        // Search vectors
        let results = registry.vector().search(&[1.0, 0.0], 1).unwrap();
        assert_eq!(results[0].0, 1);

        // Get neighbors
        let neighbors = registry.graph().get_neighbors(1, 1).unwrap();
        assert_eq!(neighbors.len(), 1);
        assert_eq!(neighbors[0].id, 2);
    }

    #[test]
    fn test_external_vector_backend_stub_returns_errors() {
        let backend = ExternalVectorBackend::new(
            "http://localhost:6333".to_string(),
            "ucotron".to_string(),
        );

        let err = backend.upsert_embeddings(&[(1, vec![1.0])]).unwrap_err();
        assert!(err.to_string().contains("not yet implemented"));

        let err = backend.search(&[1.0], 10).unwrap_err();
        assert!(err.to_string().contains("not yet implemented"));

        let err = backend.delete(&[1]).unwrap_err();
        assert!(err.to_string().contains("not yet implemented"));
    }

    #[test]
    fn test_external_graph_backend_stub_returns_errors() {
        let backend = ExternalGraphBackend::new("redis://localhost:6379".to_string());

        let node = Node {
            id: 1,
            content: "test".to_string(),
            embedding: vec![],
            metadata: HashMap::new(),
            node_type: NodeType::Entity,
            timestamp: 100,
            media_type: None,
            media_uri: None,
            embedding_visual: None,
            timestamp_range: None,
            parent_video_id: None,
        };

        let err = backend.upsert_nodes(&[node]).unwrap_err();
        assert!(err.to_string().contains("not yet implemented"));

        let err = backend.get_node(1).unwrap_err();
        assert!(err.to_string().contains("not yet implemented"));

        let err = backend.get_community(1).unwrap_err();
        assert!(err.to_string().contains("not yet implemented"));
    }

    #[test]
    fn test_external_backends_are_object_safe() {
        // Verify external stubs can be used as trait objects
        let _vec: Box<dyn VectorBackend> = Box::new(ExternalVectorBackend::new(
            "http://localhost:6333".to_string(),
            "test".to_string(),
        ));
        let _graph: Box<dyn GraphBackend> =
            Box::new(ExternalGraphBackend::new("redis://localhost:6379".to_string()));
    }

    #[test]
    fn test_vector_backend_empty_upsert() {
        let backend = MockVectorBackend::new();
        backend.upsert_embeddings(&[]).unwrap();
        let results = backend.search(&[1.0], 10).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_vector_backend_search_no_matches() {
        let backend = MockVectorBackend::new();
        // Search on an empty backend
        let results = backend.search(&[1.0, 0.0], 5).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_vector_backend_search_top_k_zero() {
        let backend = MockVectorBackend::new();
        backend
            .upsert_embeddings(&[(1, vec![1.0, 0.0])])
            .unwrap();
        let results = backend.search(&[1.0, 0.0], 0).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_vector_backend_delete_nonexistent() {
        let backend = MockVectorBackend::new();
        // Deleting IDs that don't exist should not error
        backend.delete(&[999, 1000]).unwrap();
    }

    #[test]
    fn test_vector_backend_delete_empty_slice() {
        let backend = MockVectorBackend::new();
        backend
            .upsert_embeddings(&[(1, vec![1.0])])
            .unwrap();
        backend.delete(&[]).unwrap();
        let results = backend.search(&[1.0], 10).unwrap();
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_vector_backend_upsert_overwrites() {
        let backend = MockVectorBackend::new();
        backend
            .upsert_embeddings(&[(1, vec![1.0, 0.0])])
            .unwrap();
        // Overwrite with different embedding
        backend
            .upsert_embeddings(&[(1, vec![0.0, 1.0])])
            .unwrap();
        let results = backend.search(&[0.0, 1.0], 1).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, 1);
        assert!((results[0].1 - 1.0).abs() < 1e-6, "Should match new embedding");
    }

    #[test]
    fn test_graph_backend_empty_upsert() {
        let backend = MockGraphBackend::new();
        backend.upsert_nodes(&[]).unwrap();
        backend.upsert_edges(&[]).unwrap();
    }

    #[test]
    fn test_graph_backend_get_nonexistent_node() {
        let backend = MockGraphBackend::new();
        let result = backend.get_node(999).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_graph_backend_neighbors_isolated_node() {
        let backend = MockGraphBackend::new();
        let node = Node {
            id: 1,
            content: "isolated".to_string(),
            embedding: vec![],
            metadata: HashMap::new(),
            node_type: NodeType::Entity,
            timestamp: 100,
            media_type: None,
            media_uri: None,
            embedding_visual: None,
            timestamp_range: None,
            parent_video_id: None,
        };
        backend.upsert_nodes(&[node]).unwrap();
        let neighbors = backend.get_neighbors(1, 1).unwrap();
        assert!(neighbors.is_empty());
    }

    #[test]
    fn test_graph_backend_get_all_edges_empty() {
        let backend = MockGraphBackend::new();
        let edges = backend.get_all_edges().unwrap();
        assert!(edges.is_empty());
    }

    #[test]
    fn test_graph_backend_store_community_noop() {
        let backend = MockGraphBackend::new();
        let assignments = HashMap::new();
        backend.store_community_assignments(&assignments).unwrap();
    }

    #[test]
    fn test_external_graph_backend_all_methods_error() {
        let backend = ExternalGraphBackend::new("redis://localhost:6379".to_string());

        assert!(backend
            .upsert_edges(&[Edge {
                source: 1,
                target: 2,
                edge_type: EdgeType::RelatesTo,
                weight: 1.0,
                metadata: HashMap::new(),
            }])
            .is_err());
        assert!(backend.get_neighbors(1, 1).is_err());
        assert!(backend.find_path(1, 2, 10).is_err());
        assert!(backend.get_all_edges().is_err());
        assert!(backend
            .store_community_assignments(&HashMap::new())
            .is_err());
    }

    // --- VisualVectorBackend tests ---

    /// In-memory mock visual vector backend for testing (512-dim CLIP space).
    struct MockVisualVectorBackend {
        embeddings: Mutex<HashMap<NodeId, Vec<f32>>>,
    }

    impl MockVisualVectorBackend {
        fn new() -> Self {
            Self {
                embeddings: Mutex::new(HashMap::new()),
            }
        }
    }

    impl VisualVectorBackend for MockVisualVectorBackend {
        fn upsert_visual_embeddings(
            &self,
            items: &[(NodeId, Vec<f32>)],
        ) -> anyhow::Result<()> {
            let mut map = self.embeddings.lock().unwrap();
            for (id, vec) in items {
                map.insert(*id, vec.clone());
            }
            Ok(())
        }

        fn search_visual(
            &self,
            query: &[f32],
            top_k: usize,
        ) -> anyhow::Result<Vec<(NodeId, f32)>> {
            let map = self.embeddings.lock().unwrap();
            let mut scores: Vec<(NodeId, f32)> = map
                .iter()
                .map(|(id, emb)| {
                    let dot: f32 = query.iter().zip(emb.iter()).map(|(a, b)| a * b).sum();
                    (*id, dot)
                })
                .collect();
            scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            scores.truncate(top_k);
            Ok(scores)
        }

        fn delete_visual(&self, ids: &[NodeId]) -> anyhow::Result<()> {
            let mut map = self.embeddings.lock().unwrap();
            for id in ids {
                map.remove(id);
            }
            Ok(())
        }
    }

    #[test]
    fn test_visual_backend_is_object_safe() {
        let backend: Box<dyn VisualVectorBackend> = Box::new(MockVisualVectorBackend::new());
        backend
            .upsert_visual_embeddings(&[(1, vec![1.0; 512])])
            .unwrap();
        let results = backend.search_visual(&[1.0; 512], 1).unwrap();
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_visual_backend_upsert_and_search() {
        let backend = MockVisualVectorBackend::new();

        let items: Vec<(NodeId, Vec<f32>)> = vec![
            (1, vec![1.0, 0.0, 0.0]),
            (2, vec![0.0, 1.0, 0.0]),
            (3, vec![0.7, 0.7, 0.0]),
        ];
        backend.upsert_visual_embeddings(&items).unwrap();

        let results = backend.search_visual(&[1.0, 0.0, 0.0], 2).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, 1);
        assert_eq!(results[1].0, 3);
    }

    #[test]
    fn test_visual_backend_delete() {
        let backend = MockVisualVectorBackend::new();
        backend
            .upsert_visual_embeddings(&[(1, vec![1.0, 0.0]), (2, vec![0.0, 1.0])])
            .unwrap();
        backend.delete_visual(&[1]).unwrap();
        let results = backend.search_visual(&[1.0, 0.0], 10).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, 2);
    }

    #[test]
    fn test_registry_without_visual() {
        let registry = BackendRegistry::new(
            Box::new(MockVectorBackend::new()),
            Box::new(MockGraphBackend::new()),
        );
        assert!(registry.visual().is_none());
    }

    #[test]
    fn test_registry_with_visual() {
        let registry = BackendRegistry::with_visual(
            Box::new(MockVectorBackend::new()),
            Box::new(MockGraphBackend::new()),
            Box::new(MockVisualVectorBackend::new()),
        );

        // Visual backend is available
        let vis = registry.visual().expect("visual backend should be present");
        vis.upsert_visual_embeddings(&[(1, vec![0.5; 512])])
            .unwrap();
        let results = vis.search_visual(&[0.5; 512], 1).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, 1);

        // Text vector and graph still work
        registry
            .vector()
            .upsert_embeddings(&[(1, vec![1.0, 0.0])])
            .unwrap();
        let text_results = registry.vector().search(&[1.0, 0.0], 1).unwrap();
        assert_eq!(text_results.len(), 1);
    }

    #[test]
    fn test_dual_index_independent_search() {
        let registry = BackendRegistry::with_visual(
            Box::new(MockVectorBackend::new()),
            Box::new(MockGraphBackend::new()),
            Box::new(MockVisualVectorBackend::new()),
        );

        // Insert different embeddings into text and visual indices
        registry
            .vector()
            .upsert_embeddings(&[
                (1, vec![1.0, 0.0, 0.0]),
                (2, vec![0.0, 1.0, 0.0]),
            ])
            .unwrap();

        registry
            .visual()
            .unwrap()
            .upsert_visual_embeddings(&[
                (1, vec![0.0, 0.0, 1.0]),
                (2, vec![0.0, 1.0, 0.0]),
            ])
            .unwrap();

        // Text search: query [1,0,0] → node 1 is best
        let text_results = registry.vector().search(&[1.0, 0.0, 0.0], 1).unwrap();
        assert_eq!(text_results[0].0, 1);

        // Visual search: query [0,0,1] → node 1 is best (in visual space)
        let vis_results = registry
            .visual()
            .unwrap()
            .search_visual(&[0.0, 0.0, 1.0], 1)
            .unwrap();
        assert_eq!(vis_results[0].0, 1);

        // Visual search: query [0,1,0] → node 2 is best (in visual space)
        let vis_results2 = registry
            .visual()
            .unwrap()
            .search_visual(&[0.0, 1.0, 0.0], 1)
            .unwrap();
        assert_eq!(vis_results2[0].0, 2);
    }
}
