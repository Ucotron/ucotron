//! # Ucotron Helix
//!
//! HelixDB storage engine implementation using Heed/LMDB.
//!
//! This crate provides a high-performance, zero-copy storage backend built on LMDB
//! (via the `heed` Rust bindings). It implements the `StorageEngine` trait from
//! `ucotron-core` with:
//!
//! - Named databases for nodes, edges, and secondary indices
//! - Batch insert operations using grouped write transactions (10k chunks)
//! - SIMD-optimized brute-force cosine similarity vector search (see below)
//! - Iterative BFS graph traversal with cycle detection
//!
//! ## Vector Search Strategy
//!
//! Two vector search backends are available:
//!
//! 1. **`HelixVectorBackend` (Phase 1, brute-force SIMD)** — O(n) scan with 8-wide
//!    accumulator lanes for LLVM auto-vectorization (NEON `fmla` / AVX `vfmadd`).
//!    Min-heap top-k selection for O(n log k). Suitable for ≤100k vectors.
//!
//! 2. **`HnswVectorBackend` (Phase 2 default, instant-distance)** — HNSW
//!    approximate nearest neighbor index via the `instant-distance` crate.
//!    Configurable `ef_construction` and `ef_search` parameters in `ucotron.toml`.
//!    Index is persisted in LMDB via bincode serialization and loaded on startup.
//!    Rebuild-on-upsert strategy: the full index is rebuilt after each
//!    `upsert_embeddings` call (practical for ≤1M vectors; <1s rebuild time).
//!
//! The backend is selected via `ucotron.toml`:
//! - `hnsw.enabled = true` (default) → `HnswVectorBackend`
//! - `hnsw.enabled = false` → `HelixVectorBackend` (brute-force fallback)
//!
//! ## Database Layout
//!
//! | Database          | Key          | Value                  | Description                |
//! |-------------------|-------------|------------------------|----------------------------|
//! | `nodes`           | `u64`       | `Node`                 | Primary node storage       |
//! | `edges`           | `(u64,u64,u32)` | `Edge`            | Primary edge storage       |
//! | `adj_out`         | `u64`       | `Vec<(u64,u32)>`       | Outgoing adjacency list    |
//! | `adj_in`          | `u64`       | `Vec<(u64,u32)>`       | Incoming adjacency list    |
//! | `nodes_by_type`   | `u8`        | `Vec<u64>`             | node_type → node_ids       |
//! | `agents`          | `String`    | `Agent`                | Agent records              |
//! | `shares`          | `(String,String)` | `AgentShare`    | Share grants               |

pub use ucotron_core;

use anyhow::{Context, Result};
use heed::types::SerdeBincode;
use heed::{Database, Env, EnvOpenOptions};
use ucotron_core::{
    Agent, AgentShare,
    Config, Edge, EdgeType, GraphBackend, InsertStats, Node, NodeId, NodeType, StorageEngine,
    VectorBackend, VisualVectorBackend,
};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet, VecDeque};
use std::fs;
use std::path::Path;
use std::time::Instant;

/// Composite key for the edges database: (source, target, edge_type discriminant).
type EdgeKey = (u64, u64, u32);

/// Wrapper for min-heap top-k selection. Reverses f32 ordering so that
/// `BinaryHeap` (a max-heap) returns the *minimum* scored entry at `.peek()`.
struct MinScored(f32, u64);

impl PartialEq for MinScored {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl Eq for MinScored {}

impl PartialOrd for MinScored {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for MinScored {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reversed: smaller scores come first (min-heap behavior)
        other.0.partial_cmp(&self.0).unwrap_or(Ordering::Equal)
    }
}

/// Entry in an adjacency list: (neighbor_id, edge_type discriminant).
type AdjEntry = (u64, u32);

/// HelixDB storage engine backed by LMDB via Heed.
///
/// Uses adjacency lists stored as `Vec<AdjEntry>` per node for O(degree) neighbor
/// lookups instead of full-table scans.
pub struct HelixEngine {
    env: Env,
    /// Primary node storage: NodeId → Node
    nodes_db: Database<SerdeBincode<u64>, SerdeBincode<Node>>,
    /// Primary edge storage: (source, target, type) → Edge
    edges_db: Database<SerdeBincode<EdgeKey>, SerdeBincode<Edge>>,
    /// Outgoing adjacency: node_id → Vec<(target, edge_type)>
    adj_out: Database<SerdeBincode<u64>, SerdeBincode<Vec<AdjEntry>>>,
    /// Incoming adjacency: node_id → Vec<(source, edge_type)>
    adj_in: Database<SerdeBincode<u64>, SerdeBincode<Vec<AdjEntry>>>,
    /// Node type secondary index: type_discriminant → Vec<node_id>
    nodes_by_type: Database<SerdeBincode<u8>, SerdeBincode<Vec<u64>>>,
    /// Batch size for chunked writes.
    batch_size: usize,
}

fn node_type_to_u8(nt: &NodeType) -> u8 {
    match nt {
        NodeType::Entity => 0,
        NodeType::Event => 1,
        NodeType::Fact => 2,
        NodeType::Skill => 3,
    }
}

fn edge_type_to_u32(et: &EdgeType) -> u32 {
    match et {
        EdgeType::RelatesTo => 0,
        EdgeType::CausedBy => 1,
        EdgeType::ConflictsWith => 2,
        EdgeType::NextEpisode => 3,
        EdgeType::HasProperty => 4,
        EdgeType::Supersedes => 5,
        EdgeType::Actor => 6,
        EdgeType::Object => 7,
        EdgeType::Location => 8,
        EdgeType::Companion => 9,
    }
}

/// Cosine similarity between two L2-normalized vectors (= dot product).
///
/// Uses a multi-accumulator strategy with chunks of 8 to enable LLVM
/// auto-vectorization on both x86 (AVX) and aarch64 (NEON). On Apple
/// Silicon this compiles to `fmla` (fused multiply-add) NEON instructions.
///
/// For 384-dim vectors this processes exactly 48 chunks of 8 with zero remainder.
#[inline]
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    dot_product_simd(a, b)
}

/// SIMD-friendly dot product using 8-wide accumulator lanes.
///
/// The 8 independent accumulators break data dependencies and allow the
/// CPU to pipeline multiple multiply-add operations per cycle. LLVM
/// reliably auto-vectorizes this pattern into SIMD instructions.
#[inline]
fn dot_product_simd(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let chunks = n / 8;
    let remainder = n % 8;

    // 8 independent accumulators — enables instruction-level parallelism
    let mut acc = [0.0f32; 8];

    let a_chunks = &a[..chunks * 8];
    let b_chunks = &b[..chunks * 8];

    // Process 8 elements at a time — LLVM auto-vectorizes this into
    // NEON fmla (aarch64) or AVX vfmadd (x86_64) instructions
    for i in (0..a_chunks.len()).step_by(8) {
        acc[0] += a_chunks[i] * b_chunks[i];
        acc[1] += a_chunks[i + 1] * b_chunks[i + 1];
        acc[2] += a_chunks[i + 2] * b_chunks[i + 2];
        acc[3] += a_chunks[i + 3] * b_chunks[i + 3];
        acc[4] += a_chunks[i + 4] * b_chunks[i + 4];
        acc[5] += a_chunks[i + 5] * b_chunks[i + 5];
        acc[6] += a_chunks[i + 6] * b_chunks[i + 6];
        acc[7] += a_chunks[i + 7] * b_chunks[i + 7];
    }

    // Handle remaining elements (0 for 384-dim vectors since 384 % 8 == 0)
    let tail_start = chunks * 8;
    for i in 0..remainder {
        acc[i % 8] += a[tail_start + i] * b[tail_start + i];
    }

    // Reduce 8 accumulators — pairwise for better numerical stability
    let s01 = acc[0] + acc[1];
    let s23 = acc[2] + acc[3];
    let s45 = acc[4] + acc[5];
    let s67 = acc[6] + acc[7];
    (s01 + s23) + (s45 + s67)
}

impl HelixEngine {
    /// Reconstruct the path from source to target using the parent map.
    fn reconstruct_path(
        parent: &HashMap<NodeId, NodeId>,
        source: NodeId,
        target: NodeId,
    ) -> Vec<NodeId> {
        let mut path = Vec::new();
        let mut current = target;
        while current != source {
            path.push(current);
            current = parent[&current];
        }
        path.push(source);
        path.reverse();
        path
    }
}

impl StorageEngine for HelixEngine {
    fn init(config: &Config) -> Result<Self> {
        let path = Path::new(&config.data_dir);
        fs::create_dir_all(path)
            .with_context(|| format!("Failed to create data directory: {}", config.data_dir))?;

        let env = unsafe {
            EnvOpenOptions::new()
                .map_size(config.max_db_size as usize)
                .max_dbs(5)
                .open(path)
                .with_context(|| {
                    format!("Failed to open LMDB environment at {}", config.data_dir)
                })?
        };

        let mut wtxn = env.write_txn()?;
        let nodes_db = env.create_database(&mut wtxn, Some("nodes"))?;
        let edges_db = env.create_database(&mut wtxn, Some("edges"))?;
        let adj_out = env.create_database(&mut wtxn, Some("adj_out"))?;
        let adj_in = env.create_database(&mut wtxn, Some("adj_in"))?;
        let nodes_by_type = env.create_database(&mut wtxn, Some("nodes_by_type"))?;
        wtxn.commit()?;

        Ok(Self {
            env,
            nodes_db,
            edges_db,
            adj_out,
            adj_in,
            nodes_by_type,
            batch_size: config.batch_size,
        })
    }

    fn insert_nodes(&mut self, nodes: &[Node]) -> Result<InsertStats> {
        let start = Instant::now();
        let mut inserted = 0;

        for chunk in nodes.chunks(self.batch_size) {
            let mut wtxn = self.env.write_txn()?;
            for node in chunk {
                self.nodes_db.put(&mut wtxn, &node.id, node)?;

                // Update type index: read-modify-write the Vec for this type
                let type_key = node_type_to_u8(&node.node_type);
                let mut ids = self
                    .nodes_by_type
                    .get(&wtxn, &type_key)?
                    .unwrap_or_default();
                ids.push(node.id);
                self.nodes_by_type.put(&mut wtxn, &type_key, &ids)?;
            }
            wtxn.commit()?;
            inserted += chunk.len();
        }

        Ok(InsertStats {
            count: inserted,
            duration_us: start.elapsed().as_micros() as u64,
        })
    }

    fn insert_edges(&mut self, edges: &[Edge]) -> Result<InsertStats> {
        let start = Instant::now();
        let mut inserted = 0;

        for chunk in edges.chunks(self.batch_size) {
            let mut wtxn = self.env.write_txn()?;
            for edge in chunk {
                let et = edge_type_to_u32(&edge.edge_type);
                let key: EdgeKey = (edge.source, edge.target, et);

                // Store the full edge
                self.edges_db.put(&mut wtxn, &key, edge)?;

                // Update outgoing adjacency list for source
                let mut out_list = self
                    .adj_out
                    .get(&wtxn, &edge.source)?
                    .unwrap_or_default();
                out_list.push((edge.target, et));
                self.adj_out.put(&mut wtxn, &edge.source, &out_list)?;

                // Update incoming adjacency list for target
                let mut in_list = self
                    .adj_in
                    .get(&wtxn, &edge.target)?
                    .unwrap_or_default();
                in_list.push((edge.source, et));
                self.adj_in.put(&mut wtxn, &edge.target, &in_list)?;
            }
            wtxn.commit()?;
            inserted += chunk.len();
        }

        Ok(InsertStats {
            count: inserted,
            duration_us: start.elapsed().as_micros() as u64,
        })
    }

    fn get_node(&self, id: NodeId) -> Result<Option<Node>> {
        let rtxn = self.env.read_txn()?;
        Ok(self.nodes_db.get(&rtxn, &id)?)
    }

    fn get_neighbors(&self, id: NodeId, hops: u8) -> Result<Vec<Node>> {
        let rtxn = self.env.read_txn()?;

        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();

        visited.insert(id);
        queue.push_back((id, 0u8));

        let mut result = Vec::new();

        while let Some((current_id, depth)) = queue.pop_front() {
            if depth >= hops {
                continue;
            }

            // Outgoing edges
            if let Some(out_list) = self.adj_out.get(&rtxn, &current_id)? {
                for &(target, _et) in &out_list {
                    if visited.insert(target) {
                        if let Some(node) = self.nodes_db.get(&rtxn, &target)? {
                            result.push(node);
                        }
                        queue.push_back((target, depth + 1));
                    }
                }
            }

            // Incoming edges
            if let Some(in_list) = self.adj_in.get(&rtxn, &current_id)? {
                for &(source, _et) in &in_list {
                    if visited.insert(source) {
                        if let Some(node) = self.nodes_db.get(&rtxn, &source)? {
                            result.push(node);
                        }
                        queue.push_back((source, depth + 1));
                    }
                }
            }
        }

        Ok(result)
    }

    fn vector_search(&self, query: &[f32], top_k: usize) -> Result<Vec<(NodeId, f32)>> {
        let rtxn = self.env.read_txn()?;

        if top_k == 0 {
            return Ok(Vec::new());
        }

        // Use a min-heap of size top_k: O(n log k) instead of O(n log n) full sort.
        // The heap holds the *lowest* scoring entry at the top so we can quickly
        // decide whether a new candidate should replace it.
        let mut heap: BinaryHeap<MinScored> = BinaryHeap::with_capacity(top_k + 1);

        let iter = self.nodes_db.iter(&rtxn)?;
        for entry in iter {
            let (_id, node) = entry?;
            if node.embedding.len() == query.len() {
                let sim = cosine_similarity(query, &node.embedding);

                if heap.len() < top_k {
                    heap.push(MinScored(sim, node.id));
                } else if let Some(min_entry) = heap.peek() {
                    if sim > min_entry.0 {
                        heap.pop();
                        heap.push(MinScored(sim, node.id));
                    }
                }
            }
        }

        // Drain heap into sorted Vec (highest similarity first)
        let mut results: Vec<(NodeId, f32)> =
            heap.into_iter().map(|ms| (ms.1, ms.0)).collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

        Ok(results)
    }

    fn find_path(
        &self,
        source: NodeId,
        target: NodeId,
        max_depth: u32,
    ) -> Result<Option<Vec<NodeId>>> {
        if source == target {
            return Ok(Some(vec![source]));
        }

        let rtxn = self.env.read_txn()?;

        // BFS with parent tracking for path reconstruction.
        // HashMap maps each visited node to its parent.
        let mut parent: HashMap<NodeId, NodeId> = HashMap::new();
        let mut queue = VecDeque::new();

        parent.insert(source, source); // sentinel: source's parent is itself
        queue.push_back((source, 0u32));

        while let Some((current, depth)) = queue.pop_front() {
            if depth >= max_depth {
                continue;
            }

            // Outgoing edges
            if let Some(out_list) = self.adj_out.get(&rtxn, &current)? {
                for &(neighbor, _et) in &out_list {
                    if let std::collections::hash_map::Entry::Vacant(e) = parent.entry(neighbor) {
                        e.insert(current);
                        if neighbor == target {
                            return Ok(Some(Self::reconstruct_path(&parent, source, target)));
                        }
                        queue.push_back((neighbor, depth + 1));
                    }
                }
            }

            // Incoming edges (bidirectional traversal)
            if let Some(in_list) = self.adj_in.get(&rtxn, &current)? {
                for &(neighbor, _et) in &in_list {
                    if let std::collections::hash_map::Entry::Vacant(e) = parent.entry(neighbor) {
                        e.insert(current);
                        if neighbor == target {
                            return Ok(Some(Self::reconstruct_path(&parent, source, target)));
                        }
                        queue.push_back((neighbor, depth + 1));
                    }
                }
            }
        }

        Ok(None) // no path found
    }

    fn hybrid_search(&self, query: &[f32], top_k: usize, hops: u8) -> Result<Vec<Node>> {
        ucotron_core::find_related(self, query, top_k, hops, ucotron_core::DEFAULT_HOP_DECAY)
    }

    fn shutdown(&mut self) -> Result<()> {
        // LMDB handles cleanup on drop; Heed's Env flushes automatically.
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Phase 2: Pluggable Backend Implementations
// ---------------------------------------------------------------------------

/// HelixDB vector backend implementing the Phase 2 `VectorBackend` trait.
///
/// Uses the same LMDB environment as `HelixEngine` but only exposes vector
/// search operations. Embeddings are stored alongside nodes in the `nodes`
/// database; vector search uses brute-force SIMD cosine similarity.
pub struct HelixVectorBackend {
    env: Env,
    nodes_db: Database<SerdeBincode<u64>, SerdeBincode<Node>>,
}

impl HelixVectorBackend {
    /// Create a new vector backend from an LMDB environment path.
    ///
    /// Opens (or creates) the LMDB environment and `nodes` database.
    pub fn open(data_dir: &str, max_db_size: u64) -> Result<Self> {
        let path = Path::new(data_dir);
        fs::create_dir_all(path)
            .with_context(|| format!("Failed to create vector data dir: {}", data_dir))?;

        let env = unsafe {
            EnvOpenOptions::new()
                .map_size(max_db_size as usize)
                .max_dbs(5)
                .open(path)
                .with_context(|| format!("Failed to open LMDB at {}", data_dir))?
        };

        let mut wtxn = env.write_txn()?;
        let nodes_db = env.create_database(&mut wtxn, Some("nodes"))?;
        wtxn.commit()?;

        Ok(Self { env, nodes_db })
    }
}

impl VectorBackend for HelixVectorBackend {
    fn upsert_embeddings(&self, items: &[(NodeId, Vec<f32>)]) -> Result<()> {
        let mut wtxn = self.env.write_txn()?;
        for (id, embedding) in items {
            // Read existing node or create a minimal placeholder
            let mut node = self
                .nodes_db
                .get(&wtxn, id)?
                .unwrap_or_else(|| Node {
                    id: *id,
                    content: String::new(),
                    embedding: Vec::new(),
                    metadata: HashMap::new(),
                    node_type: NodeType::Entity,
                    timestamp: 0,
                    media_type: None,
                    media_uri: None,
                    embedding_visual: None,
                    timestamp_range: None,
                    parent_video_id: None,
                });
            node.embedding = embedding.clone();
            self.nodes_db.put(&mut wtxn, id, &node)?;
        }
        wtxn.commit()?;
        Ok(())
    }

    fn search(&self, query: &[f32], top_k: usize) -> Result<Vec<(NodeId, f32)>> {
        let rtxn = self.env.read_txn()?;
        if top_k == 0 {
            return Ok(Vec::new());
        }

        let mut heap: BinaryHeap<MinScored> = BinaryHeap::with_capacity(top_k + 1);
        let iter = self.nodes_db.iter(&rtxn)?;
        for entry in iter {
            let (_id, node) = entry?;
            if node.embedding.len() == query.len() {
                let sim = cosine_similarity(query, &node.embedding);
                if heap.len() < top_k {
                    heap.push(MinScored(sim, node.id));
                } else if let Some(min_entry) = heap.peek() {
                    if sim > min_entry.0 {
                        heap.pop();
                        heap.push(MinScored(sim, node.id));
                    }
                }
            }
        }

        let mut results: Vec<(NodeId, f32)> =
            heap.into_iter().map(|ms| (ms.1, ms.0)).collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        Ok(results)
    }

    fn delete(&self, ids: &[NodeId]) -> Result<()> {
        let mut wtxn = self.env.write_txn()?;
        for id in ids {
            // Zero out the embedding but keep the node
            if let Some(mut node) = self.nodes_db.get(&wtxn, id)? {
                node.embedding.clear();
                self.nodes_db.put(&mut wtxn, id, &node)?;
            }
        }
        wtxn.commit()?;
        Ok(())
    }
}

/// HelixDB graph backend implementing the Phase 2 `GraphBackend` trait.
///
/// Uses the same LMDB environment as `HelixEngine` but only exposes graph
/// storage and traversal operations. Adjacency lists enable O(degree)
/// neighbor lookups.
pub struct HelixGraphBackend {
    env: Env,
    nodes_db: Database<SerdeBincode<u64>, SerdeBincode<Node>>,
    edges_db: Database<SerdeBincode<EdgeKey>, SerdeBincode<Edge>>,
    adj_out: Database<SerdeBincode<u64>, SerdeBincode<Vec<AdjEntry>>>,
    adj_in: Database<SerdeBincode<u64>, SerdeBincode<Vec<AdjEntry>>>,
    nodes_by_type: Database<SerdeBincode<u8>, SerdeBincode<Vec<u64>>>,
    /// Maps NodeId → CommunityId for community detection results.
    community_assignments: Database<SerdeBincode<u64>, SerdeBincode<u64>>,
    /// Maps CommunityId → Vec<NodeId> for community member lookup.
    community_members: Database<SerdeBincode<u64>, SerdeBincode<Vec<u64>>>,
    /// Persistent agent storage keyed by AgentId (String).
    agents_db: Database<SerdeBincode<String>, SerdeBincode<Agent>>,
    /// Agent share grants keyed by (agent_id, target_agent_id).
    shares_db: Database<SerdeBincode<(String, String)>, SerdeBincode<AgentShare>>,
    batch_size: usize,
}

impl HelixGraphBackend {
    /// Create a new graph backend from an LMDB environment path.
    pub fn open(data_dir: &str, max_db_size: u64, batch_size: usize) -> Result<Self> {
        let path = Path::new(data_dir);
        fs::create_dir_all(path)
            .with_context(|| format!("Failed to create graph data dir: {}", data_dir))?;

        let env = unsafe {
            EnvOpenOptions::new()
                .map_size(max_db_size as usize)
                .max_dbs(9)
                .open(path)
                .with_context(|| format!("Failed to open LMDB at {}", data_dir))?
        };

        let mut wtxn = env.write_txn()?;
        let nodes_db = env.create_database(&mut wtxn, Some("nodes"))?;
        let edges_db = env.create_database(&mut wtxn, Some("edges"))?;
        let adj_out = env.create_database(&mut wtxn, Some("adj_out"))?;
        let adj_in = env.create_database(&mut wtxn, Some("adj_in"))?;
        let nodes_by_type = env.create_database(&mut wtxn, Some("nodes_by_type"))?;
        let community_assignments = env.create_database(&mut wtxn, Some("community_assign"))?;
        let community_members = env.create_database(&mut wtxn, Some("community_members"))?;
        let agents_db = env.create_database(&mut wtxn, Some("agents"))?;
        let shares_db = env.create_database(&mut wtxn, Some("shares"))?;
        wtxn.commit()?;

        Ok(Self {
            env,
            nodes_db,
            edges_db,
            adj_out,
            adj_in,
            nodes_by_type,
            community_assignments,
            community_members,
            agents_db,
            shares_db,
            batch_size,
        })
    }

    /// Reconstruct a BFS path from parent map.
    fn reconstruct_path_bfs(
        parent: &HashMap<NodeId, NodeId>,
        source: NodeId,
        target: NodeId,
    ) -> Vec<NodeId> {
        let mut path = Vec::new();
        let mut current = target;
        while current != source {
            path.push(current);
            current = parent[&current];
        }
        path.push(source);
        path.reverse();
        path
    }

    // ----- Agent CRUD -----

    /// Insert or update an agent record.
    pub fn create_agent(&self, agent: &Agent) -> Result<()> {
        let mut wtxn = self.env.write_txn()?;
        self.agents_db.put(&mut wtxn, &agent.id, agent)?;
        wtxn.commit()?;
        Ok(())
    }

    /// Retrieve an agent by its ID.
    pub fn get_agent(&self, id: &str) -> Result<Option<Agent>> {
        let rtxn = self.env.read_txn()?;
        Ok(self.agents_db.get(&rtxn, &id.to_string())?)
    }

    /// List all agents. Optionally filter by owner.
    pub fn list_agents(&self, owner: Option<&str>) -> Result<Vec<Agent>> {
        let rtxn = self.env.read_txn()?;
        let mut agents = Vec::new();
        let iter = self.agents_db.iter(&rtxn)?;
        for item in iter {
            let (_key, agent) = item?;
            if let Some(owner_filter) = owner {
                if agent.owner == owner_filter {
                    agents.push(agent);
                }
            } else {
                agents.push(agent);
            }
        }
        Ok(agents)
    }

    /// Delete an agent and all its share grants (both outgoing and incoming).
    pub fn delete_agent(&self, id: &str) -> Result<()> {
        let mut wtxn = self.env.write_txn()?;
        let id_str = id.to_string();
        self.agents_db.delete(&mut wtxn, &id_str)?;

        // Remove all shares where this agent is source or target.
        let share_keys: Vec<(String, String)> = {
            let iter = self.shares_db.iter(&wtxn)?;
            let mut keys = Vec::new();
            for item in iter {
                let (key, _share) = item?;
                if key.0 == id || key.1 == id {
                    keys.push(key);
                }
            }
            keys
        };
        for key in share_keys {
            self.shares_db.delete(&mut wtxn, &key)?;
        }
        wtxn.commit()?;
        Ok(())
    }

    // ----- AgentShare CRUD -----

    /// Create a share grant from one agent to another.
    pub fn create_share(&self, share: &AgentShare) -> Result<()> {
        let mut wtxn = self.env.write_txn()?;
        let key = (share.agent_id.clone(), share.target_agent_id.clone());
        self.shares_db.put(&mut wtxn, &key, share)?;
        wtxn.commit()?;
        Ok(())
    }

    /// Get a specific share grant.
    pub fn get_share(&self, agent_id: &str, target_id: &str) -> Result<Option<AgentShare>> {
        let rtxn = self.env.read_txn()?;
        let key = (agent_id.to_string(), target_id.to_string());
        Ok(self.shares_db.get(&rtxn, &key)?)
    }

    /// List all shares granted by a specific agent.
    pub fn list_shares(&self, agent_id: &str) -> Result<Vec<AgentShare>> {
        let rtxn = self.env.read_txn()?;
        let mut shares = Vec::new();
        let iter = self.shares_db.iter(&rtxn)?;
        for item in iter {
            let (_key, share) = item?;
            if share.agent_id == agent_id {
                shares.push(share);
            }
        }
        Ok(shares)
    }

    /// Delete a specific share grant.
    pub fn delete_share(&self, agent_id: &str, target_id: &str) -> Result<()> {
        let mut wtxn = self.env.write_txn()?;
        let key = (agent_id.to_string(), target_id.to_string());
        self.shares_db.delete(&mut wtxn, &key)?;
        wtxn.commit()?;
        Ok(())
    }

    /// Clone nodes and edges from one namespace to another with optional filtering.
    ///
    /// 1. Reads all nodes, filters by namespace + CloneFilter criteria
    /// 2. Assigns new IDs starting from `id_start`
    /// 3. Re-tags cloned nodes with `dst_ns` namespace
    /// 4. Copies edges that connect two cloned nodes (remapping IDs)
    pub fn clone_graph_impl(
        &self,
        src_ns: &str,
        dst_ns: &str,
        filter: &ucotron_core::CloneFilter,
        id_start: u64,
    ) -> Result<ucotron_core::CloneResult> {
        // 1. Gather source nodes matching namespace + filters
        let all_nodes = self.get_all_nodes_impl()?;
        let src_nodes: Vec<&Node> = all_nodes
            .iter()
            .filter(|n| {
                // Check namespace
                let ns_match = match n.metadata.get("_namespace") {
                    Some(ucotron_core::Value::String(ns)) => ns == src_ns,
                    None => src_ns == "default",
                    _ => src_ns == "default",
                };
                if !ns_match {
                    return false;
                }
                // Apply node_type filter
                if let Some(ref types) = filter.node_types {
                    if !types.iter().any(|t| std::mem::discriminant(t) == std::mem::discriminant(&n.node_type)) {
                        return false;
                    }
                }
                // Apply time_range_start filter
                if let Some(start) = filter.time_range_start {
                    if n.timestamp < start {
                        return false;
                    }
                }
                // Apply time_range_end filter
                if let Some(end) = filter.time_range_end {
                    if n.timestamp > end {
                        return false;
                    }
                }
                true
            })
            .collect();

        if src_nodes.is_empty() {
            return Ok(ucotron_core::CloneResult::default());
        }

        // 2. Build ID mapping: old_id → new_id
        let mut id_map = HashMap::new();
        let mut next_id = id_start;
        for node in &src_nodes {
            id_map.insert(node.id, next_id);
            next_id += 1;
        }

        // 3. Create cloned nodes with new IDs and dst_ns namespace
        let cloned_nodes: Vec<Node> = src_nodes
            .iter()
            .map(|n| {
                let new_id = id_map[&n.id];
                let mut metadata = n.metadata.clone();
                metadata.insert(
                    "_namespace".into(),
                    ucotron_core::Value::String(dst_ns.to_string()),
                );
                Node {
                    id: new_id,
                    content: n.content.clone(),
                    embedding: n.embedding.clone(),
                    metadata,
                    node_type: n.node_type.clone(),
                    timestamp: n.timestamp,
                    media_type: n.media_type.clone(),
                    media_uri: n.media_uri.clone(),
                    embedding_visual: n.embedding_visual.clone(),
                    timestamp_range: n.timestamp_range,
                    parent_video_id: n.parent_video_id,
                }
            })
            .collect();

        let nodes_copied = cloned_nodes.len();
        self.upsert_nodes_impl(&cloned_nodes)?;

        // 4. Copy edges where both source and target are in the cloned set
        let old_id_set: HashSet<NodeId> = src_nodes.iter().map(|n| n.id).collect();
        let all_edges = self.get_all_edges_full_impl()?;
        let cloned_edges: Vec<Edge> = all_edges
            .into_iter()
            .filter(|e| old_id_set.contains(&e.source) && old_id_set.contains(&e.target))
            .map(|e| Edge {
                source: id_map[&e.source],
                target: id_map[&e.target],
                edge_type: e.edge_type,
                weight: e.weight,
                metadata: e.metadata,
            })
            .collect();

        let edges_copied = cloned_edges.len();
        if !cloned_edges.is_empty() {
            self.upsert_edges_impl(&cloned_edges)?;
        }

        Ok(ucotron_core::CloneResult {
            nodes_copied,
            edges_copied,
            id_map,
        })
    }

    /// Merge nodes and edges from one namespace into another with entity deduplication.
    ///
    /// Unlike clone, merge detects duplicate nodes by matching content strings.
    /// Duplicates are not re-inserted; instead their edges are redirected to the
    /// existing destination node. Non-duplicate nodes are assigned new IDs starting
    /// from `id_start`.
    pub fn merge_graph_impl(
        &self,
        src_ns: &str,
        dst_ns: &str,
        id_start: u64,
    ) -> Result<ucotron_core::MergeResult> {
        let all_nodes = self.get_all_nodes_impl()?;

        // Partition nodes into source and destination by namespace
        let src_nodes: Vec<&Node> = all_nodes
            .iter()
            .filter(|n| match n.metadata.get("_namespace") {
                Some(ucotron_core::Value::String(ns)) => ns == src_ns,
                None => src_ns == "default",
                _ => src_ns == "default",
            })
            .collect();

        if src_nodes.is_empty() {
            return Ok(ucotron_core::MergeResult::default());
        }

        let dst_nodes: Vec<&Node> = all_nodes
            .iter()
            .filter(|n| match n.metadata.get("_namespace") {
                Some(ucotron_core::Value::String(ns)) => ns == dst_ns,
                None => dst_ns == "default",
                _ => dst_ns == "default",
            })
            .collect();

        // Build content→NodeId index for destination nodes (for dedup lookup)
        let mut dst_content_map: HashMap<String, NodeId> = HashMap::new();
        for node in &dst_nodes {
            let key = node.content.trim().to_lowercase();
            dst_content_map.entry(key).or_insert(node.id);
        }

        // Map source node IDs → destination IDs (either existing dedup or new)
        let mut id_map: HashMap<NodeId, NodeId> = HashMap::new();
        let mut next_id = id_start;
        let mut nodes_deduplicated: usize = 0;
        let mut ids_remapped: usize = 0;
        let mut new_nodes: Vec<Node> = Vec::new();

        for node in &src_nodes {
            let key = node.content.trim().to_lowercase();
            if let Some(&existing_id) = dst_content_map.get(&key) {
                // Duplicate: map source ID to existing destination ID
                id_map.insert(node.id, existing_id);
                nodes_deduplicated += 1;
            } else {
                // Non-duplicate: assign new ID and prepare node for insertion
                let new_id = next_id;
                next_id += 1;
                id_map.insert(node.id, new_id);
                ids_remapped += 1;

                let mut metadata = node.metadata.clone();
                metadata.insert(
                    "_namespace".into(),
                    ucotron_core::Value::String(dst_ns.to_string()),
                );
                new_nodes.push(Node {
                    id: new_id,
                    content: node.content.clone(),
                    embedding: node.embedding.clone(),
                    metadata,
                    node_type: node.node_type.clone(),
                    timestamp: node.timestamp,
                    media_type: node.media_type.clone(),
                    media_uri: node.media_uri.clone(),
                    embedding_visual: node.embedding_visual.clone(),
                    timestamp_range: node.timestamp_range,
                    parent_video_id: node.parent_video_id,
                });

                // Add to content map so subsequent source dupes also get deduped
                dst_content_map.insert(key, new_id);
            }
        }

        let nodes_copied = new_nodes.len();
        if !new_nodes.is_empty() {
            self.upsert_nodes_impl(&new_nodes)?;
        }

        // Copy edges from source namespace, remapping IDs
        let old_id_set: HashSet<NodeId> = src_nodes.iter().map(|n| n.id).collect();
        let all_edges = self.get_all_edges_full_impl()?;
        let merged_edges: Vec<Edge> = all_edges
            .into_iter()
            .filter(|e| old_id_set.contains(&e.source) && old_id_set.contains(&e.target))
            .map(|e| Edge {
                source: id_map[&e.source],
                target: id_map[&e.target],
                edge_type: e.edge_type,
                weight: e.weight,
                metadata: e.metadata,
            })
            .collect();

        let edges_copied = merged_edges.len();
        if !merged_edges.is_empty() {
            self.upsert_edges_impl(&merged_edges)?;
        }

        Ok(ucotron_core::MergeResult {
            nodes_copied,
            edges_copied,
            nodes_deduplicated,
            ids_remapped,
        })
    }

    // Helper: get_all_nodes without going through the trait
    fn get_all_nodes_impl(&self) -> Result<Vec<Node>> {
        let rtxn = self.env.read_txn()?;
        let mut nodes = Vec::new();
        let iter = self.nodes_db.iter(&rtxn)?;
        for entry in iter {
            let (_id, node) = entry?;
            nodes.push(node);
        }
        Ok(nodes)
    }

    // Helper: upsert_nodes without going through the trait
    fn upsert_nodes_impl(&self, nodes: &[Node]) -> Result<()> {
        for chunk in nodes.chunks(self.batch_size) {
            let mut wtxn = self.env.write_txn()?;
            for node in chunk {
                self.nodes_db.put(&mut wtxn, &node.id, node)?;
                let type_key = node_type_to_u8(&node.node_type);
                let mut ids = self
                    .nodes_by_type
                    .get(&wtxn, &type_key)?
                    .unwrap_or_default();
                ids.push(node.id);
                self.nodes_by_type.put(&mut wtxn, &type_key, &ids)?;
            }
            wtxn.commit()?;
        }
        Ok(())
    }

    // Helper: upsert_edges without going through the trait
    fn upsert_edges_impl(&self, edges: &[Edge]) -> Result<()> {
        for chunk in edges.chunks(self.batch_size) {
            let mut wtxn = self.env.write_txn()?;
            for edge in chunk {
                let et = edge_type_to_u32(&edge.edge_type);
                let key: EdgeKey = (edge.source, edge.target, et);
                self.edges_db.put(&mut wtxn, &key, edge)?;
                let mut out_list = self
                    .adj_out
                    .get(&wtxn, &edge.source)?
                    .unwrap_or_default();
                out_list.push((edge.target, et));
                self.adj_out.put(&mut wtxn, &edge.source, &out_list)?;
                let mut in_list = self
                    .adj_in
                    .get(&wtxn, &edge.target)?
                    .unwrap_or_default();
                in_list.push((edge.source, et));
                self.adj_in.put(&mut wtxn, &edge.target, &in_list)?;
            }
            wtxn.commit()?;
        }
        Ok(())
    }

    // Helper: get_all_edges_full without going through the trait
    fn get_all_edges_full_impl(&self) -> Result<Vec<Edge>> {
        let rtxn = self.env.read_txn()?;
        let mut edges = Vec::new();
        let iter = self.adj_out.iter(&rtxn)?;
        for item in iter {
            let (source, adj_list) = item?;
            for &(target, et) in &adj_list {
                let key: EdgeKey = (source, target, et);
                if let Some(edge) = self.edges_db.get(&rtxn, &key)? {
                    edges.push(edge);
                }
            }
        }
        Ok(edges)
    }
}

impl GraphBackend for HelixGraphBackend {
    fn upsert_nodes(&self, nodes: &[Node]) -> Result<()> {
        for chunk in nodes.chunks(self.batch_size) {
            let mut wtxn = self.env.write_txn()?;
            for node in chunk {
                self.nodes_db.put(&mut wtxn, &node.id, node)?;
                // Update type index
                let type_key = node_type_to_u8(&node.node_type);
                let mut ids = self
                    .nodes_by_type
                    .get(&wtxn, &type_key)?
                    .unwrap_or_default();
                ids.push(node.id);
                self.nodes_by_type.put(&mut wtxn, &type_key, &ids)?;
            }
            wtxn.commit()?;
        }
        Ok(())
    }

    fn upsert_edges(&self, edges: &[Edge]) -> Result<()> {
        for chunk in edges.chunks(self.batch_size) {
            let mut wtxn = self.env.write_txn()?;
            for edge in chunk {
                let et = edge_type_to_u32(&edge.edge_type);
                let key: EdgeKey = (edge.source, edge.target, et);
                self.edges_db.put(&mut wtxn, &key, edge)?;

                // Update outgoing adjacency
                let mut out_list = self
                    .adj_out
                    .get(&wtxn, &edge.source)?
                    .unwrap_or_default();
                out_list.push((edge.target, et));
                self.adj_out.put(&mut wtxn, &edge.source, &out_list)?;

                // Update incoming adjacency
                let mut in_list = self
                    .adj_in
                    .get(&wtxn, &edge.target)?
                    .unwrap_or_default();
                in_list.push((edge.source, et));
                self.adj_in.put(&mut wtxn, &edge.target, &in_list)?;
            }
            wtxn.commit()?;
        }
        Ok(())
    }

    fn get_node(&self, id: NodeId) -> Result<Option<Node>> {
        let rtxn = self.env.read_txn()?;
        Ok(self.nodes_db.get(&rtxn, &id)?)
    }

    fn get_neighbors(&self, id: NodeId, hops: u8) -> Result<Vec<Node>> {
        let rtxn = self.env.read_txn()?;
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        visited.insert(id);
        queue.push_back((id, 0u8));
        let mut result = Vec::new();

        while let Some((current_id, depth)) = queue.pop_front() {
            if depth >= hops {
                continue;
            }
            if let Some(out_list) = self.adj_out.get(&rtxn, &current_id)? {
                for &(target, _et) in &out_list {
                    if visited.insert(target) {
                        if let Some(node) = self.nodes_db.get(&rtxn, &target)? {
                            result.push(node);
                        }
                        queue.push_back((target, depth + 1));
                    }
                }
            }
            if let Some(in_list) = self.adj_in.get(&rtxn, &current_id)? {
                for &(source, _et) in &in_list {
                    if visited.insert(source) {
                        if let Some(node) = self.nodes_db.get(&rtxn, &source)? {
                            result.push(node);
                        }
                        queue.push_back((source, depth + 1));
                    }
                }
            }
        }
        Ok(result)
    }

    fn find_path(
        &self,
        source: NodeId,
        target: NodeId,
        max_depth: u32,
    ) -> Result<Option<Vec<NodeId>>> {
        if source == target {
            return Ok(Some(vec![source]));
        }
        let rtxn = self.env.read_txn()?;
        let mut parent: HashMap<NodeId, NodeId> = HashMap::new();
        let mut queue = VecDeque::new();
        parent.insert(source, source);
        queue.push_back((source, 0u32));

        while let Some((current, depth)) = queue.pop_front() {
            if depth >= max_depth {
                continue;
            }
            if let Some(out_list) = self.adj_out.get(&rtxn, &current)? {
                for &(neighbor, _et) in &out_list {
                    if let std::collections::hash_map::Entry::Vacant(e) = parent.entry(neighbor) {
                        e.insert(current);
                        if neighbor == target {
                            return Ok(Some(Self::reconstruct_path_bfs(
                                &parent, source, target,
                            )));
                        }
                        queue.push_back((neighbor, depth + 1));
                    }
                }
            }
            if let Some(in_list) = self.adj_in.get(&rtxn, &current)? {
                for &(neighbor, _et) in &in_list {
                    if let std::collections::hash_map::Entry::Vacant(e) = parent.entry(neighbor) {
                        e.insert(current);
                        if neighbor == target {
                            return Ok(Some(Self::reconstruct_path_bfs(
                                &parent, source, target,
                            )));
                        }
                        queue.push_back((neighbor, depth + 1));
                    }
                }
            }
        }
        Ok(None)
    }

    fn get_community(&self, node_id: NodeId) -> Result<Vec<NodeId>> {
        let rtxn = self.env.read_txn()?;
        if let Some(community_id) = self.community_assignments.get(&rtxn, &node_id)? {
            Ok(self
                .community_members
                .get(&rtxn, &community_id)?
                .unwrap_or_default())
        } else {
            Ok(Vec::new())
        }
    }

    fn get_all_nodes(&self) -> Result<Vec<Node>> {
        let rtxn = self.env.read_txn()?;
        let mut nodes = Vec::new();
        let iter = self.nodes_db.iter(&rtxn)?;
        for entry in iter {
            let (_id, node) = entry?;
            nodes.push(node);
        }
        Ok(nodes)
    }

    fn get_all_edges(&self) -> Result<Vec<(NodeId, NodeId, f32)>> {
        let rtxn = self.env.read_txn()?;
        let mut edges = Vec::new();
        let iter = self.adj_out.iter(&rtxn)?;
        for item in iter {
            let (source, adj_list) = item?;
            for &(target, et) in &adj_list {
                // Look up the edge weight from edges_db using the stored edge type.
                let key: EdgeKey = (source, target, et);
                let weight = self
                    .edges_db
                    .get(&rtxn, &key)?
                    .map(|e| e.weight)
                    .unwrap_or(1.0);
                edges.push((source, target, weight));
            }
        }
        Ok(edges)
    }

    fn get_all_edges_full(&self) -> Result<Vec<Edge>> {
        let rtxn = self.env.read_txn()?;
        let mut edges = Vec::new();
        let iter = self.adj_out.iter(&rtxn)?;
        for item in iter {
            let (source, adj_list) = item?;
            for &(target, et) in &adj_list {
                let key: EdgeKey = (source, target, et);
                if let Some(edge) = self.edges_db.get(&rtxn, &key)? {
                    edges.push(edge);
                }
            }
        }
        Ok(edges)
    }

    fn delete_nodes(&self, ids: &[NodeId]) -> Result<()> {
        if ids.is_empty() {
            return Ok(());
        }
        let id_set: std::collections::HashSet<NodeId> = ids.iter().copied().collect();

        for chunk in ids.chunks(self.batch_size) {
            let mut wtxn = self.env.write_txn()?;
            for &node_id in chunk {
                // 1. Get node to know its type for type-index cleanup
                let node_type_key = self
                    .nodes_db
                    .get(&wtxn, &node_id)?
                    .map(|n| node_type_to_u8(&n.node_type));

                // 2. Delete node from primary storage
                self.nodes_db.delete(&mut wtxn, &node_id)?;

                // 3. Remove from type index
                if let Some(type_key) = node_type_key {
                    if let Some(mut type_ids) = self.nodes_by_type.get(&wtxn, &type_key)? {
                        type_ids.retain(|&id| id != node_id);
                        if type_ids.is_empty() {
                            self.nodes_by_type.delete(&mut wtxn, &type_key)?;
                        } else {
                            self.nodes_by_type.put(&mut wtxn, &type_key, &type_ids)?;
                        }
                    }
                }

                // 4. Delete outgoing edges and clean up targets' incoming lists
                if let Some(out_list) = self.adj_out.get(&wtxn, &node_id)? {
                    for &(target, et) in &out_list {
                        // Delete the edge record
                        let key: EdgeKey = (node_id, target, et);
                        self.edges_db.delete(&mut wtxn, &key)?;
                        // Remove from target's incoming list (if target not also being deleted)
                        if !id_set.contains(&target) {
                            if let Some(mut in_list) = self.adj_in.get(&wtxn, &target)? {
                                in_list.retain(|&(src, _)| src != node_id);
                                if in_list.is_empty() {
                                    self.adj_in.delete(&mut wtxn, &target)?;
                                } else {
                                    self.adj_in.put(&mut wtxn, &target, &in_list)?;
                                }
                            }
                        }
                    }
                    self.adj_out.delete(&mut wtxn, &node_id)?;
                }

                // 5. Delete incoming edges and clean up sources' outgoing lists
                if let Some(in_list) = self.adj_in.get(&wtxn, &node_id)? {
                    for &(source, et) in &in_list {
                        // Delete the edge record
                        let key: EdgeKey = (source, node_id, et);
                        self.edges_db.delete(&mut wtxn, &key)?;
                        // Remove from source's outgoing list (if source not also being deleted)
                        if !id_set.contains(&source) {
                            if let Some(mut out_list) = self.adj_out.get(&wtxn, &source)? {
                                out_list.retain(|&(tgt, _)| tgt != node_id);
                                if out_list.is_empty() {
                                    self.adj_out.delete(&mut wtxn, &source)?;
                                } else {
                                    self.adj_out.put(&mut wtxn, &source, &out_list)?;
                                }
                            }
                        }
                    }
                    self.adj_in.delete(&mut wtxn, &node_id)?;
                }

                // 6. Remove from community assignments
                self.community_assignments.delete(&mut wtxn, &node_id)?;
            }
            wtxn.commit()?;
        }

        Ok(())
    }

    fn store_community_assignments(
        &self,
        assignments: &std::collections::HashMap<NodeId, ucotron_core::community::CommunityId>,
    ) -> Result<()> {
        // Build community_id → Vec<NodeId> mapping
        let mut members_map: std::collections::HashMap<u64, Vec<NodeId>> =
            std::collections::HashMap::new();
        for (&node_id, &community_id) in assignments {
            members_map.entry(community_id).or_default().push(node_id);
        }
        // Sort member lists for deterministic ordering
        for members in members_map.values_mut() {
            members.sort_unstable();
        }

        let mut wtxn = self.env.write_txn()?;

        // Clear previous assignments
        self.community_assignments.clear(&mut wtxn)?;
        self.community_members.clear(&mut wtxn)?;

        // Store node → community assignments
        for (&node_id, &community_id) in assignments {
            self.community_assignments
                .put(&mut wtxn, &node_id, &community_id)?;
        }

        // Store community → members
        for (community_id, members) in &members_map {
            self.community_members
                .put(&mut wtxn, community_id, members)?;
        }

        wtxn.commit()?;
        Ok(())
    }

    // ----- Agent CRUD (delegates to inherent methods) -----

    fn create_agent(&self, agent: &ucotron_core::Agent) -> Result<()> {
        HelixGraphBackend::create_agent(self, agent)
    }

    fn get_agent(&self, id: &str) -> Result<Option<ucotron_core::Agent>> {
        HelixGraphBackend::get_agent(self, id)
    }

    fn list_agents(&self, owner: Option<&str>) -> Result<Vec<ucotron_core::Agent>> {
        HelixGraphBackend::list_agents(self, owner)
    }

    fn delete_agent(&self, id: &str) -> Result<()> {
        HelixGraphBackend::delete_agent(self, id)
    }

    fn create_share(&self, share: &ucotron_core::AgentShare) -> Result<()> {
        HelixGraphBackend::create_share(self, share)
    }

    fn get_share(
        &self,
        agent_id: &str,
        target_id: &str,
    ) -> Result<Option<ucotron_core::AgentShare>> {
        HelixGraphBackend::get_share(self, agent_id, target_id)
    }

    fn list_shares(&self, agent_id: &str) -> Result<Vec<ucotron_core::AgentShare>> {
        HelixGraphBackend::list_shares(self, agent_id)
    }

    fn delete_share(&self, agent_id: &str, target_id: &str) -> Result<()> {
        HelixGraphBackend::delete_share(self, agent_id, target_id)
    }

    fn clone_graph(
        &self,
        src_ns: &str,
        dst_ns: &str,
        filter: &ucotron_core::CloneFilter,
        id_start: u64,
    ) -> Result<ucotron_core::CloneResult> {
        self.clone_graph_impl(src_ns, dst_ns, filter, id_start)
    }

    fn merge_graph(
        &self,
        src_ns: &str,
        dst_ns: &str,
        id_start: u64,
    ) -> Result<ucotron_core::MergeResult> {
        self.merge_graph_impl(src_ns, dst_ns, id_start)
    }
}

// ---------------------------------------------------------------------------
// Phase 2: HNSW Vector Backend (instant-distance)
// ---------------------------------------------------------------------------

/// A 384-dimensional point for the HNSW index. Wraps a `NodeId` and its embedding.
///
/// Implements `instant_distance::Point` using cosine distance (1 - cosine_similarity)
/// on L2-normalized embeddings (which equals 1 - dot_product).
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct HnswPoint {
    id: NodeId,
    embedding: Vec<f32>,
}

impl instant_distance::Point for HnswPoint {
    fn distance(&self, other: &Self) -> f32 {
        // Cosine distance = 1 - cosine_similarity
        // For L2-normalized vectors: cosine_similarity = dot_product
        1.0 - dot_product_simd(&self.embedding, &other.embedding)
    }
}

/// HNSW vector index key in LMDB for storing the serialized index.
const HNSW_INDEX_KEY: u64 = u64::MAX;

/// Wrapper for storing binary blobs in LMDB via SerdeBincode.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct BlobValue {
    data: Vec<u8>,
}

/// HNSW-accelerated vector backend implementing the Phase 2 `VectorBackend` trait.
///
/// Uses `instant-distance` for HNSW index construction and search, with LMDB for
/// persistent storage of both embeddings and the serialized HNSW index.
///
/// **Index lifecycle:**
/// - Embeddings are stored in LMDB as the source of truth
/// - The HNSW index is rebuilt from all stored embeddings after each `upsert_embeddings` call
/// - The rebuilt index is serialized with bincode and persisted in LMDB
/// - On startup, the persisted index is loaded (if it exists)
///
/// This rebuild-on-upsert strategy is practical for up to ~1M vectors (rebuild takes <1s).
/// For larger scales, switch to an incrementally-updatable index (e.g., `hnsw_rs`).
pub struct HnswVectorBackend {
    env: Env,
    /// Stores Node objects keyed by NodeId (source of truth for embeddings)
    nodes_db: Database<SerdeBincode<u64>, SerdeBincode<Node>>,
    /// Stores the serialized HNSW index as a single blob
    hnsw_index_db: Database<SerdeBincode<u64>, SerdeBincode<BlobValue>>,
    /// In-memory HNSW index (rebuilt from LMDB on startup or after upserts)
    index: std::sync::RwLock<Option<HnswIndex>>,
    /// HNSW build parameters
    hnsw_config: HnswConfig,
}

/// Wrapper for the in-memory HNSW index and its point-to-NodeId mapping.
struct HnswIndex {
    hnsw: instant_distance::HnswMap<HnswPoint, NodeId>,
}

impl HnswVectorBackend {
    /// Open an HNSW vector backend from an LMDB directory.
    ///
    /// If a persisted HNSW index exists in LMDB, it is loaded into memory.
    pub fn open(data_dir: &str, max_db_size: u64, hnsw_config: HnswConfig) -> Result<Self> {
        let path = Path::new(data_dir);
        fs::create_dir_all(path)
            .with_context(|| format!("Failed to create HNSW data dir: {}", data_dir))?;

        let env = unsafe {
            EnvOpenOptions::new()
                .map_size(max_db_size as usize)
                .max_dbs(5)
                .open(path)
                .with_context(|| format!("Failed to open LMDB at {}", data_dir))?
        };

        let mut wtxn = env.write_txn()?;
        let nodes_db = env.create_database(&mut wtxn, Some("nodes"))?;
        let hnsw_index_db = env.create_database(&mut wtxn, Some("hnsw_index"))?;
        wtxn.commit()?;

        // Try to load persisted index
        let index = {
            let rtxn = env.read_txn()?;
            let maybe_blob: Option<BlobValue> = hnsw_index_db.get(&rtxn, &HNSW_INDEX_KEY)?;
            match maybe_blob {
                Some(blob) => {
                    match bincode::deserialize::<instant_distance::HnswMap<HnswPoint, NodeId>>(blob.data.as_slice()) {
                        Ok(hnsw) => Some(HnswIndex { hnsw }),
                        Err(_) => None, // corrupted index; will be rebuilt on next upsert
                    }
                }
                None => None,
            }
        };

        Ok(Self {
            env,
            nodes_db,
            hnsw_index_db,
            index: std::sync::RwLock::new(index),
            hnsw_config,
        })
    }

    /// Rebuild the HNSW index from all embeddings stored in LMDB.
    ///
    /// Collects all non-empty embeddings, builds the index using instant-distance,
    /// then persists the serialized index back to LMDB.
    fn rebuild_index(&self) -> Result<()> {
        let rtxn = self.env.read_txn()?;

        // Collect all points with non-empty embeddings
        let mut points = Vec::new();
        let mut values = Vec::new();
        let iter = self.nodes_db.iter(&rtxn)?;
        for entry in iter {
            let (_key, node) = entry?;
            if !node.embedding.is_empty() {
                points.push(HnswPoint {
                    id: node.id,
                    embedding: node.embedding.clone(),
                });
                values.push(node.id);
            }
        }
        drop(rtxn);

        if points.is_empty() {
            // Clear index
            let mut guard = self.index.write().map_err(|e| anyhow::anyhow!("Lock poisoned: {}", e))?;
            *guard = None;

            let mut wtxn = self.env.write_txn()?;
            self.hnsw_index_db.delete(&mut wtxn, &HNSW_INDEX_KEY)?;
            wtxn.commit()?;
            return Ok(());
        }

        // Build HNSW index
        let hnsw = instant_distance::Builder::default()
            .ef_construction(self.hnsw_config.ef_construction)
            .ef_search(self.hnsw_config.ef_search)
            .seed(42)
            .build(points, values);

        // Serialize and persist
        let bytes = bincode::serialize(&hnsw)
            .with_context(|| "Failed to serialize HNSW index")?;

        let blob = BlobValue { data: bytes };
        let mut wtxn = self.env.write_txn()?;
        self.hnsw_index_db.put(&mut wtxn, &HNSW_INDEX_KEY, &blob)?;
        wtxn.commit()?;

        // Update in-memory index
        let mut guard = self.index.write().map_err(|e| anyhow::anyhow!("Lock poisoned: {}", e))?;
        *guard = Some(HnswIndex { hnsw });

        Ok(())
    }
}

impl VectorBackend for HnswVectorBackend {
    fn upsert_embeddings(&self, items: &[(NodeId, Vec<f32>)]) -> Result<()> {
        // Store embeddings in LMDB (source of truth)
        let mut wtxn = self.env.write_txn()?;
        for (id, embedding) in items {
            let mut node = self
                .nodes_db
                .get(&wtxn, id)?
                .unwrap_or_else(|| Node {
                    id: *id,
                    content: String::new(),
                    embedding: Vec::new(),
                    metadata: HashMap::new(),
                    node_type: NodeType::Entity,
                    timestamp: 0,
                    media_type: None,
                    media_uri: None,
                    embedding_visual: None,
                    timestamp_range: None,
                    parent_video_id: None,
                });
            node.embedding = embedding.clone();
            self.nodes_db.put(&mut wtxn, id, &node)?;
        }
        wtxn.commit()?;

        // Rebuild HNSW index from all embeddings
        self.rebuild_index()?;

        Ok(())
    }

    fn search(&self, query: &[f32], top_k: usize) -> Result<Vec<(NodeId, f32)>> {
        if top_k == 0 {
            return Ok(Vec::new());
        }

        let guard = self.index.read().map_err(|e| anyhow::anyhow!("Lock poisoned: {}", e))?;

        if let Some(ref idx) = *guard {
            // HNSW search
            let query_point = HnswPoint {
                id: 0, // dummy id for query
                embedding: query.to_vec(),
            };
            let mut search = instant_distance::Search::default();
            let results: Vec<(NodeId, f32)> = idx
                .hnsw
                .search(&query_point, &mut search)
                .take(top_k)
                .map(|item| {
                    // instant-distance returns cosine distance; convert to similarity
                    let similarity = 1.0 - item.distance;
                    (*item.value, similarity)
                })
                .collect();
            Ok(results)
        } else {
            // Fallback to brute-force if no index exists yet
            drop(guard); // release read lock before reading LMDB
            let rtxn = self.env.read_txn()?;
            let mut heap: BinaryHeap<MinScored> = BinaryHeap::with_capacity(top_k + 1);
            let iter = self.nodes_db.iter(&rtxn)?;
            for entry in iter {
                let (_id, node) = entry?;
                if node.embedding.len() == query.len() {
                    let sim = cosine_similarity(query, &node.embedding);
                    if heap.len() < top_k {
                        heap.push(MinScored(sim, node.id));
                    } else if let Some(min_entry) = heap.peek() {
                        if sim > min_entry.0 {
                            heap.pop();
                            heap.push(MinScored(sim, node.id));
                        }
                    }
                }
            }
            let mut results: Vec<(NodeId, f32)> =
                heap.into_iter().map(|ms| (ms.1, ms.0)).collect();
            results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
            Ok(results)
        }
    }

    fn delete(&self, ids: &[NodeId]) -> Result<()> {
        let mut wtxn = self.env.write_txn()?;
        for id in ids {
            if let Some(mut node) = self.nodes_db.get(&wtxn, id)? {
                node.embedding.clear();
                self.nodes_db.put(&mut wtxn, id, &node)?;
            }
        }
        wtxn.commit()?;

        // Rebuild index without deleted embeddings
        self.rebuild_index()?;

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Visual Vector Backend (512-dim CLIP, dual-index design)
// ---------------------------------------------------------------------------

/// LMDB key for the visual HNSW serialized index.
const VISUAL_HNSW_INDEX_KEY: u64 = u64::MAX - 1;

/// HNSW-accelerated visual vector backend for 512-dim CLIP embeddings.
///
/// Mirrors `HnswVectorBackend` but operates in a separate LMDB environment
/// with its own databases and HNSW index, enabling independent text and visual
/// similarity search. Part of the dual-index multimodal architecture.
///
/// **Index lifecycle** (same as `HnswVectorBackend`):
/// - Visual embeddings stored in LMDB `visual_nodes` as source of truth
/// - HNSW index rebuilt after each `upsert_visual_embeddings` call
/// - Serialized index persisted in LMDB `visual_hnsw_index`
/// - On startup, persisted index is loaded if it exists
pub struct HelixVisualVectorBackend {
    env: Env,
    /// Stores visual embeddings keyed by NodeId
    visual_nodes_db: Database<SerdeBincode<u64>, SerdeBincode<Vec<f32>>>,
    /// Stores the serialized visual HNSW index as a single blob
    visual_hnsw_index_db: Database<SerdeBincode<u64>, SerdeBincode<BlobValue>>,
    /// In-memory visual HNSW index
    index: std::sync::RwLock<Option<HnswIndex>>,
    /// HNSW build parameters
    hnsw_config: HnswConfig,
}

impl HelixVisualVectorBackend {
    /// Open a visual vector backend from an LMDB directory.
    ///
    /// Creates a separate LMDB environment for visual embeddings. If a persisted
    /// HNSW index exists, it is loaded into memory.
    pub fn open(data_dir: &str, max_db_size: u64, hnsw_config: HnswConfig) -> Result<Self> {
        let path = Path::new(data_dir);
        fs::create_dir_all(path)
            .with_context(|| format!("Failed to create visual data dir: {}", data_dir))?;

        let env = unsafe {
            EnvOpenOptions::new()
                .map_size(max_db_size as usize)
                .max_dbs(5)
                .open(path)
                .with_context(|| format!("Failed to open visual LMDB at {}", data_dir))?
        };

        let mut wtxn = env.write_txn()?;
        let visual_nodes_db = env.create_database(&mut wtxn, Some("visual_nodes"))?;
        let visual_hnsw_index_db =
            env.create_database(&mut wtxn, Some("visual_hnsw_index"))?;
        wtxn.commit()?;

        // Try to load persisted index
        let index = {
            let rtxn = env.read_txn()?;
            let maybe_blob: Option<BlobValue> =
                visual_hnsw_index_db.get(&rtxn, &VISUAL_HNSW_INDEX_KEY)?;
            match maybe_blob {
                Some(blob) => {
                    match bincode::deserialize::<instant_distance::HnswMap<HnswPoint, NodeId>>(
                        blob.data.as_slice(),
                    ) {
                        Ok(hnsw) => Some(HnswIndex { hnsw }),
                        Err(_) => None, // corrupted; rebuilt on next upsert
                    }
                }
                None => None,
            }
        };

        Ok(Self {
            env,
            visual_nodes_db,
            visual_hnsw_index_db,
            index: std::sync::RwLock::new(index),
            hnsw_config,
        })
    }

    /// Rebuild the visual HNSW index from all embeddings stored in LMDB.
    fn rebuild_index(&self) -> Result<()> {
        let rtxn = self.env.read_txn()?;

        let mut points = Vec::new();
        let mut values = Vec::new();
        let iter = self.visual_nodes_db.iter(&rtxn)?;
        for entry in iter {
            let (node_id, embedding) = entry?;
            if !embedding.is_empty() {
                points.push(HnswPoint {
                    id: node_id,
                    embedding: embedding.clone(),
                });
                values.push(node_id);
            }
        }
        drop(rtxn);

        if points.is_empty() {
            let mut guard = self
                .index
                .write()
                .map_err(|e| anyhow::anyhow!("Lock poisoned: {}", e))?;
            *guard = None;

            let mut wtxn = self.env.write_txn()?;
            self.visual_hnsw_index_db
                .delete(&mut wtxn, &VISUAL_HNSW_INDEX_KEY)?;
            wtxn.commit()?;
            return Ok(());
        }

        let hnsw = instant_distance::Builder::default()
            .ef_construction(self.hnsw_config.ef_construction)
            .ef_search(self.hnsw_config.ef_search)
            .seed(42)
            .build(points, values);

        let bytes =
            bincode::serialize(&hnsw).with_context(|| "Failed to serialize visual HNSW index")?;

        let blob = BlobValue { data: bytes };
        let mut wtxn = self.env.write_txn()?;
        self.visual_hnsw_index_db
            .put(&mut wtxn, &VISUAL_HNSW_INDEX_KEY, &blob)?;
        wtxn.commit()?;

        let mut guard = self
            .index
            .write()
            .map_err(|e| anyhow::anyhow!("Lock poisoned: {}", e))?;
        *guard = Some(HnswIndex { hnsw });

        Ok(())
    }
}

impl VisualVectorBackend for HelixVisualVectorBackend {
    fn upsert_visual_embeddings(&self, items: &[(NodeId, Vec<f32>)]) -> Result<()> {
        let mut wtxn = self.env.write_txn()?;
        for (id, embedding) in items {
            self.visual_nodes_db.put(&mut wtxn, id, embedding)?;
        }
        wtxn.commit()?;

        self.rebuild_index()?;
        Ok(())
    }

    fn search_visual(&self, query: &[f32], top_k: usize) -> Result<Vec<(NodeId, f32)>> {
        if top_k == 0 {
            return Ok(Vec::new());
        }

        let guard = self
            .index
            .read()
            .map_err(|e| anyhow::anyhow!("Lock poisoned: {}", e))?;

        if let Some(ref idx) = *guard {
            let query_point = HnswPoint {
                id: 0,
                embedding: query.to_vec(),
            };
            let mut search = instant_distance::Search::default();
            let results: Vec<(NodeId, f32)> = idx
                .hnsw
                .search(&query_point, &mut search)
                .take(top_k)
                .map(|item| {
                    let similarity = 1.0 - item.distance;
                    (*item.value, similarity)
                })
                .collect();
            Ok(results)
        } else {
            // Fallback to brute-force if no index exists yet
            drop(guard);
            let rtxn = self.env.read_txn()?;
            let mut heap: BinaryHeap<MinScored> = BinaryHeap::with_capacity(top_k + 1);
            let iter = self.visual_nodes_db.iter(&rtxn)?;
            for entry in iter {
                let (node_id, embedding) = entry?;
                if embedding.len() == query.len() {
                    let sim = cosine_similarity(query, &embedding);
                    if heap.len() < top_k {
                        heap.push(MinScored(sim, node_id));
                    } else if let Some(min_entry) = heap.peek() {
                        if sim > min_entry.0 {
                            heap.pop();
                            heap.push(MinScored(sim, node_id));
                        }
                    }
                }
            }
            let mut results: Vec<(NodeId, f32)> =
                heap.into_iter().map(|ms| (ms.1, ms.0)).collect();
            results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
            Ok(results)
        }
    }

    fn delete_visual(&self, ids: &[NodeId]) -> Result<()> {
        let mut wtxn = self.env.write_txn()?;
        for id in ids {
            self.visual_nodes_db.delete(&mut wtxn, id)?;
        }
        wtxn.commit()?;

        self.rebuild_index()?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Config-driven factory functions
// ---------------------------------------------------------------------------

use ucotron_config::{GraphBackendConfig, HnswConfig, VectorBackendConfig};

/// Create a `HelixVectorBackend` from a `VectorBackendConfig`.
///
/// Uses `data_dir` and `max_db_size` from the config section.
/// When HNSW is disabled, returns a brute-force backend.
pub fn create_helix_vector_backend(config: &VectorBackendConfig) -> Result<HelixVectorBackend> {
    HelixVectorBackend::open(&config.data_dir, config.max_db_size)
}

/// Create an `HnswVectorBackend` from a `VectorBackendConfig`.
///
/// Uses HNSW parameters from `config.hnsw` section. This is the default
/// for Phase 2 when `hnsw.enabled = true` (the default).
pub fn create_hnsw_vector_backend(config: &VectorBackendConfig) -> Result<HnswVectorBackend> {
    HnswVectorBackend::open(&config.data_dir, config.max_db_size, config.hnsw.clone())
}

/// Create a `HelixGraphBackend` from a `GraphBackendConfig`.
///
/// Uses `data_dir`, `max_db_size`, and `batch_size` from the config section.
pub fn create_helix_graph_backend(config: &GraphBackendConfig) -> Result<HelixGraphBackend> {
    HelixGraphBackend::open(&config.data_dir, config.max_db_size, config.batch_size)
}

/// Create a `HelixVisualVectorBackend` from a `VectorBackendConfig`.
///
/// Uses a `visual/` subdirectory under the config's `data_dir` to keep
/// visual LMDB data separate from the text vector index.
pub fn create_helix_visual_backend(
    config: &VectorBackendConfig,
) -> Result<HelixVisualVectorBackend> {
    let visual_dir = format!("{}/visual", config.data_dir);
    HelixVisualVectorBackend::open(&visual_dir, config.max_db_size, config.hnsw.clone())
}

/// Create both backends from the full storage config sections, returning
/// boxed trait objects suitable for [`BackendRegistry`](ucotron_core::BackendRegistry).
///
/// Both vector and graph backends must be configured as "helix".
/// When `hnsw.enabled = true` (default), uses HNSW for vector search.
/// When `hnsw.enabled = false`, falls back to brute-force SIMD.
pub fn create_helix_backends(
    vector_config: &VectorBackendConfig,
    graph_config: &GraphBackendConfig,
) -> Result<(Box<dyn VectorBackend>, Box<dyn GraphBackend>)> {
    let vec_backend: Box<dyn VectorBackend> = if vector_config.hnsw.enabled {
        Box::new(create_hnsw_vector_backend(vector_config)?)
    } else {
        Box::new(create_helix_vector_backend(vector_config)?)
    };
    let graph_backend = create_helix_graph_backend(graph_config)?;
    Ok((vec_backend, Box::new(graph_backend)))
}

/// Create text vector, graph, and visual vector backends from configuration.
///
/// Returns all three backends as boxed trait objects for
/// [`BackendRegistry::with_visual`](ucotron_core::BackendRegistry::with_visual).
#[allow(clippy::type_complexity)]
pub fn create_helix_backends_with_visual(
    vector_config: &VectorBackendConfig,
    graph_config: &GraphBackendConfig,
) -> Result<(
    Box<dyn VectorBackend>,
    Box<dyn GraphBackend>,
    Box<dyn VisualVectorBackend>,
)> {
    let (text_backend, graph_backend) = create_helix_backends(vector_config, graph_config)?;
    let visual_backend = create_helix_visual_backend(vector_config)?;
    Ok((text_backend, graph_backend, Box::new(visual_backend)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ucotron_core::Value;
    use std::collections::HashMap;

    fn test_engine() -> (HelixEngine, tempfile::TempDir) {
        let dir = tempfile::tempdir().expect("create temp dir");
        let config = Config {
            data_dir: dir.path().to_string_lossy().to_string(),
            max_db_size: 100 * 1024 * 1024, // 100MB for tests
            batch_size: 100,
        };
        let engine = HelixEngine::init(&config).expect("init engine");
        (engine, dir)
    }

    fn make_node(id: u64, content: &str, node_type: NodeType) -> Node {
        Node {
            id,
            content: content.to_string(),
            embedding: vec![0.0; 384],
            metadata: HashMap::new(),
            node_type,
            timestamp: 1000 + id,
            media_type: None,
            media_uri: None,
            embedding_visual: None,
            timestamp_range: None,
            parent_video_id: None,
        }
    }

    fn make_edge(source: u64, target: u64, edge_type: EdgeType) -> Edge {
        Edge {
            source,
            target,
            edge_type,
            weight: 0.5,
            metadata: HashMap::new(),
        }
    }

    #[test]
    fn test_init_and_shutdown() {
        let (mut engine, _dir) = test_engine();
        engine.shutdown().expect("shutdown");
    }

    #[test]
    fn test_insert_and_get_node() {
        let (mut engine, _dir) = test_engine();
        let node = make_node(1, "Hello world", NodeType::Entity);
        let stats = engine.insert_nodes(&[node]).expect("insert");
        assert_eq!(stats.count, 1);

        let retrieved = engine.get_node(1).expect("get").expect("exists");
        assert_eq!(retrieved.id, 1);
        assert_eq!(retrieved.content, "Hello world");
        assert_eq!(retrieved.node_type, NodeType::Entity);
    }

    #[test]
    fn test_get_nonexistent_node() {
        let (engine, _dir) = test_engine();
        let result = engine.get_node(999).expect("get");
        assert!(result.is_none());
    }

    #[test]
    fn test_insert_multiple_nodes() {
        let (mut engine, _dir) = test_engine();
        let nodes: Vec<Node> = (0..50)
            .map(|i| make_node(i, &format!("Node {}", i), NodeType::Entity))
            .collect();

        let stats = engine.insert_nodes(&nodes).expect("insert");
        assert_eq!(stats.count, 50);

        for i in 0..50 {
            let n = engine.get_node(i).expect("get").expect("exists");
            assert_eq!(n.id, i);
        }
    }

    #[test]
    fn test_insert_edges() {
        let (mut engine, _dir) = test_engine();
        let nodes = vec![
            make_node(0, "A", NodeType::Entity),
            make_node(1, "B", NodeType::Entity),
            make_node(2, "C", NodeType::Entity),
        ];
        engine.insert_nodes(&nodes).expect("insert nodes");

        let edges = vec![
            make_edge(0, 1, EdgeType::RelatesTo),
            make_edge(1, 2, EdgeType::CausedBy),
        ];
        let stats = engine.insert_edges(&edges).expect("insert edges");
        assert_eq!(stats.count, 2);
    }

    #[test]
    fn test_get_neighbors_1hop() {
        let (mut engine, _dir) = test_engine();
        let nodes = vec![
            make_node(0, "A", NodeType::Entity),
            make_node(1, "B", NodeType::Entity),
            make_node(2, "C", NodeType::Entity),
            make_node(3, "D", NodeType::Entity),
        ];
        engine.insert_nodes(&nodes).expect("insert nodes");

        let edges = vec![
            make_edge(0, 1, EdgeType::RelatesTo),
            make_edge(0, 2, EdgeType::RelatesTo),
        ];
        engine.insert_edges(&edges).expect("insert edges");

        let neighbors = engine.get_neighbors(0, 1).expect("neighbors");
        let ids: HashSet<u64> = neighbors.iter().map(|n| n.id).collect();
        assert!(ids.contains(&1), "Should find B");
        assert!(ids.contains(&2), "Should find C");
        assert!(!ids.contains(&3), "Should not find D (no edge)");
    }

    #[test]
    fn test_get_neighbors_2hop() {
        let (mut engine, _dir) = test_engine();
        let nodes = vec![
            make_node(0, "A", NodeType::Entity),
            make_node(1, "B", NodeType::Entity),
            make_node(2, "C", NodeType::Entity),
        ];
        engine.insert_nodes(&nodes).expect("insert nodes");

        // A→B→C
        let edges = vec![
            make_edge(0, 1, EdgeType::RelatesTo),
            make_edge(1, 2, EdgeType::CausedBy),
        ];
        engine.insert_edges(&edges).expect("insert edges");

        let neighbors = engine.get_neighbors(0, 2).expect("neighbors");
        let ids: HashSet<u64> = neighbors.iter().map(|n| n.id).collect();
        assert!(ids.contains(&1), "Should find B (1-hop)");
        assert!(ids.contains(&2), "Should find C (2-hop)");
    }

    #[test]
    fn test_get_neighbors_deep_chain() {
        // Test multi-hop traversal on a longer chain: A→B→C→D→E
        let (mut engine, _dir) = test_engine();
        let nodes: Vec<Node> = (0..5)
            .map(|i| make_node(i, &format!("Node{}", i), NodeType::Entity))
            .collect();
        engine.insert_nodes(&nodes).expect("insert nodes");

        // Chain: 0→1→2→3→4
        let edges = vec![
            make_edge(0, 1, EdgeType::RelatesTo),
            make_edge(1, 2, EdgeType::RelatesTo),
            make_edge(2, 3, EdgeType::CausedBy),
            make_edge(3, 4, EdgeType::HasProperty),
        ];
        engine.insert_edges(&edges).expect("insert edges");

        // 1-hop from 0: should find only node 1
        let n1 = engine.get_neighbors(0, 1).expect("1-hop");
        let ids1: HashSet<u64> = n1.iter().map(|n| n.id).collect();
        assert_eq!(ids1.len(), 1);
        assert!(ids1.contains(&1));

        // 2-hop from 0: should find nodes 1, 2
        let n2 = engine.get_neighbors(0, 2).expect("2-hop");
        let ids2: HashSet<u64> = n2.iter().map(|n| n.id).collect();
        assert_eq!(ids2.len(), 2);
        assert!(ids2.contains(&1));
        assert!(ids2.contains(&2));

        // 3-hop from 0: should find nodes 1, 2, 3
        let n3 = engine.get_neighbors(0, 3).expect("3-hop");
        let ids3: HashSet<u64> = n3.iter().map(|n| n.id).collect();
        assert_eq!(ids3.len(), 3);
        assert!(ids3.contains(&1));
        assert!(ids3.contains(&2));
        assert!(ids3.contains(&3));
        assert!(!ids3.contains(&4), "Node 4 is 4 hops away");

        // 4-hop from 0: should find all nodes 1-4
        let n4 = engine.get_neighbors(0, 4).expect("4-hop");
        let ids4: HashSet<u64> = n4.iter().map(|n| n.id).collect();
        assert_eq!(ids4.len(), 4);
        assert!(ids4.contains(&4));
    }

    #[test]
    fn test_get_neighbors_bidirectional() {
        // Test that traversal follows edges in both directions
        let (mut engine, _dir) = test_engine();
        let nodes: Vec<Node> = (0..4)
            .map(|i| make_node(i, &format!("Node{}", i), NodeType::Entity))
            .collect();
        engine.insert_nodes(&nodes).expect("insert nodes");

        // 1→0 and 0→2 and 2→3
        let edges = vec![
            make_edge(1, 0, EdgeType::RelatesTo), // incoming to 0
            make_edge(0, 2, EdgeType::RelatesTo), // outgoing from 0
            make_edge(2, 3, EdgeType::RelatesTo), // 2-hop from 0 via outgoing
        ];
        engine.insert_edges(&edges).expect("insert edges");

        // 1-hop from 0: should find 1 (via incoming) and 2 (via outgoing)
        let n1 = engine.get_neighbors(0, 1).expect("1-hop");
        let ids: HashSet<u64> = n1.iter().map(|n| n.id).collect();
        assert_eq!(ids.len(), 2);
        assert!(ids.contains(&1), "Should find node 1 via incoming edge");
        assert!(ids.contains(&2), "Should find node 2 via outgoing edge");

        // 2-hop from 0: should also find 3 (via 0→2→3)
        let n2 = engine.get_neighbors(0, 2).expect("2-hop");
        let ids2: HashSet<u64> = n2.iter().map(|n| n.id).collect();
        assert_eq!(ids2.len(), 3);
        assert!(ids2.contains(&3));
    }

    #[test]
    fn test_vector_search() {
        let (mut engine, _dir) = test_engine();

        let mut n1 = make_node(0, "Similar", NodeType::Entity);
        n1.embedding = {
            let mut v = vec![0.0f32; 384];
            v[0] = 1.0;
            v
        };

        let mut n2 = make_node(1, "Different", NodeType::Entity);
        n2.embedding = {
            let mut v = vec![0.0f32; 384];
            v[1] = 1.0;
            v
        };

        let mut n3 = make_node(2, "Also similar", NodeType::Event);
        n3.embedding = {
            let mut v = vec![0.0f32; 384];
            v[0] = 0.9;
            v[1] = 0.1;
            let norm = (v[0] * v[0] + v[1] * v[1]).sqrt();
            v[0] /= norm;
            v[1] /= norm;
            v
        };

        engine.insert_nodes(&[n1, n2, n3]).expect("insert");

        let mut query = vec![0.0f32; 384];
        query[0] = 1.0;

        let results = engine.vector_search(&query, 2).expect("search");
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, 0, "Most similar should be node 0");
        assert_eq!(results[1].0, 2, "Second should be node 2");
        assert!(results[0].1 > results[1].1);
    }

    #[test]
    fn test_hybrid_search() {
        let (mut engine, _dir) = test_engine();

        let mut n0 = make_node(0, "Query match", NodeType::Entity);
        n0.embedding = {
            let mut v = vec![0.0f32; 384];
            v[0] = 1.0;
            v
        };

        let n1 = make_node(1, "Connected neighbor", NodeType::Entity);
        // n1 has zero embedding — won't be found by vector search alone

        engine.insert_nodes(&[n0, n1]).expect("insert nodes");
        engine
            .insert_edges(&[make_edge(0, 1, EdgeType::RelatesTo)])
            .expect("insert edges");

        let mut query = vec![0.0f32; 384];
        query[0] = 1.0;

        let results = engine.hybrid_search(&query, 1, 1).expect("hybrid");
        let ids: Vec<u64> = results.iter().map(|n| n.id).collect();
        assert!(ids.contains(&0), "Should include vector match");
        assert!(ids.contains(&1), "Should include graph neighbor");
    }

    #[test]
    fn test_batch_insert_large() {
        let (mut engine, _dir) = test_engine();

        let nodes: Vec<Node> = (0..500)
            .map(|i| make_node(i, &format!("Node {}", i), NodeType::Entity))
            .collect();

        let stats = engine.insert_nodes(&nodes).expect("insert");
        assert_eq!(stats.count, 500);

        assert!(engine.get_node(0).expect("get").is_some());
        assert!(engine.get_node(250).expect("get").is_some());
        assert!(engine.get_node(499).expect("get").is_some());
    }

    #[test]
    fn test_node_with_metadata() {
        let (mut engine, _dir) = test_engine();

        let mut node = make_node(1, "With metadata", NodeType::Fact);
        node.metadata
            .insert("source".to_string(), Value::String("wikipedia".to_string()));
        node.metadata
            .insert("confidence".to_string(), Value::Float(0.95));

        engine.insert_nodes(&[node]).expect("insert");

        let retrieved = engine.get_node(1).expect("get").expect("exists");
        assert_eq!(
            retrieved.metadata.get("source"),
            Some(&Value::String("wikipedia".to_string()))
        );
    }

    #[test]
    fn test_dot_product_simd_384dim() {
        // Verify SIMD dot product matches naive computation on 384-dim vectors
        let a: Vec<f32> = (0..384).map(|i| (i as f32) * 0.01).collect();
        let b: Vec<f32> = (0..384).map(|i| ((383 - i) as f32) * 0.01).collect();

        let naive: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let simd = dot_product_simd(&a, &b);

        assert!(
            (naive - simd).abs() < 1e-3,
            "SIMD dot product should match naive: naive={naive}, simd={simd}"
        );
    }

    #[test]
    fn test_dot_product_simd_non_multiple_of_8() {
        // Test with length not divisible by 8 (remainder path)
        let a = vec![1.0f32; 13];
        let b = vec![2.0f32; 13];

        let result = dot_product_simd(&a, &b);
        assert!(
            (result - 26.0).abs() < 1e-5,
            "13 elements of 1.0*2.0 = 26.0, got {result}"
        );
    }

    #[test]
    fn test_vector_search_top_k_heap() {
        // Verify min-heap correctly returns top-k from many candidates
        let (mut engine, _dir) = test_engine();

        let nodes: Vec<Node> = (0..20)
            .map(|i| {
                let mut n = make_node(i, &format!("Node {i}"), NodeType::Entity);
                // Each node has embedding with strength proportional to its id
                let mut emb = vec![0.0f32; 384];
                let val = (i as f32 + 1.0) / 20.0;
                emb[0] = val;
                let norm = val; // single non-zero component
                emb[0] /= norm; // normalize to unit length — all become 1.0 in dim 0
                // Actually let's give different directions to get varying similarities
                emb[0] = (i as f32 * 0.1).cos();
                emb[1] = (i as f32 * 0.1).sin();
                let norm = (emb[0] * emb[0] + emb[1] * emb[1]).sqrt();
                if norm > 0.0 {
                    emb[0] /= norm;
                    emb[1] /= norm;
                }
                n.embedding = emb;
                n
            })
            .collect();

        engine.insert_nodes(&nodes).expect("insert");

        // Query aligned with dim 0
        let mut query = vec![0.0f32; 384];
        query[0] = 1.0;

        let results = engine.vector_search(&query, 5).expect("search");
        assert_eq!(results.len(), 5, "Should return exactly top-5");

        // Verify descending similarity order
        for w in results.windows(2) {
            assert!(
                w[0].1 >= w[1].1,
                "Results should be sorted descending: {} >= {}",
                w[0].1,
                w[1].1
            );
        }
    }

    #[test]
    fn test_neighbors_no_cycle() {
        let (mut engine, _dir) = test_engine();
        let nodes = vec![
            make_node(0, "A", NodeType::Entity),
            make_node(1, "B", NodeType::Entity),
        ];
        engine.insert_nodes(&nodes).expect("insert nodes");

        // A↔B (bidirectional)
        let edges = vec![
            make_edge(0, 1, EdgeType::RelatesTo),
            make_edge(1, 0, EdgeType::RelatesTo),
        ];
        engine.insert_edges(&edges).expect("insert edges");

        let neighbors = engine.get_neighbors(0, 3).expect("neighbors");
        // Should only return B once despite cycle
        assert_eq!(neighbors.len(), 1);
        assert_eq!(neighbors[0].id, 1);
    }

    // --- find_path tests ---

    #[test]
    fn test_find_path_simple_chain() {
        // A→B→C→D: path from A to D should be [0, 1, 2, 3]
        let (mut engine, _dir) = test_engine();
        let nodes: Vec<Node> = (0..4)
            .map(|i| make_node(i, &format!("Node{}", i), NodeType::Entity))
            .collect();
        engine.insert_nodes(&nodes).expect("insert nodes");

        let edges = vec![
            make_edge(0, 1, EdgeType::RelatesTo),
            make_edge(1, 2, EdgeType::RelatesTo),
            make_edge(2, 3, EdgeType::RelatesTo),
        ];
        engine.insert_edges(&edges).expect("insert edges");

        let path = engine.find_path(0, 3, 100).expect("find_path").expect("path exists");
        assert_eq!(path.first(), Some(&0), "Path should start at source");
        assert_eq!(path.last(), Some(&3), "Path should end at target");
        assert_eq!(path.len(), 4, "Shortest path is 4 nodes long");
    }

    #[test]
    fn test_find_path_no_path() {
        // Disconnected nodes: no path should exist
        let (mut engine, _dir) = test_engine();
        let nodes = vec![
            make_node(0, "A", NodeType::Entity),
            make_node(1, "B", NodeType::Entity),
        ];
        engine.insert_nodes(&nodes).expect("insert nodes");
        // No edges

        let result = engine.find_path(0, 1, 100).expect("find_path");
        assert!(result.is_none(), "No path between disconnected nodes");
    }

    #[test]
    fn test_find_path_same_node() {
        let (mut engine, _dir) = test_engine();
        let nodes = vec![make_node(0, "A", NodeType::Entity)];
        engine.insert_nodes(&nodes).expect("insert");

        let path = engine.find_path(0, 0, 100).expect("find_path").expect("path exists");
        assert_eq!(path, vec![0], "Source == target gives single-element path");
    }

    #[test]
    fn test_find_path_depth_100_chain() {
        // Chain of 101 nodes: 0→1→2→...→100
        let (mut engine, _dir) = test_engine();
        let depth = 101usize;
        let nodes: Vec<Node> = (0..depth as u64)
            .map(|i| make_node(i, &format!("N{}", i), NodeType::Entity))
            .collect();
        engine.insert_nodes(&nodes).expect("insert nodes");

        let edges: Vec<Edge> = (0..depth as u64 - 1)
            .map(|i| make_edge(i, i + 1, EdgeType::RelatesTo))
            .collect();
        engine.insert_edges(&edges).expect("insert edges");

        let path = engine
            .find_path(0, 100, 200)
            .expect("find_path")
            .expect("path exists in chain of 101");
        assert_eq!(path.len(), 101);
        assert_eq!(path[0], 0);
        assert_eq!(path[100], 100);
        // Verify path is sequential
        for i in 0..101u64 {
            assert_eq!(path[i as usize], i);
        }
    }

    #[test]
    fn test_find_path_max_depth_exceeded() {
        // Chain: 0→1→2→3 with max_depth=2 — path of length 4 should not be found
        let (mut engine, _dir) = test_engine();
        let nodes: Vec<Node> = (0..4)
            .map(|i| make_node(i, &format!("N{}", i), NodeType::Entity))
            .collect();
        engine.insert_nodes(&nodes).expect("insert");

        let edges = vec![
            make_edge(0, 1, EdgeType::RelatesTo),
            make_edge(1, 2, EdgeType::RelatesTo),
            make_edge(2, 3, EdgeType::RelatesTo),
        ];
        engine.insert_edges(&edges).expect("insert edges");

        let result = engine.find_path(0, 3, 2).expect("find_path");
        assert!(result.is_none(), "Path of length 4 exceeds max_depth 2");
    }

    // -----------------------------------------------------------------------
    // Phase 2: HelixVectorBackend tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_helix_vector_backend_upsert_and_search() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let backend = HelixVectorBackend::open(
            &dir.path().to_string_lossy(),
            100 * 1024 * 1024,
        )
        .expect("open vector backend");

        // First insert nodes via a graph backend (since embeddings are stored on nodes)
        // For the vector backend, upsert_embeddings creates placeholder nodes
        let mut emb_a = vec![0.0f32; 384];
        emb_a[0] = 1.0;
        let mut emb_b = vec![0.0f32; 384];
        emb_b[1] = 1.0;

        backend
            .upsert_embeddings(&[(1, emb_a.clone()), (2, emb_b)])
            .expect("upsert");

        let results = backend.search(&emb_a, 2).expect("search");
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, 1, "Exact match should be first");
    }

    #[test]
    fn test_helix_vector_backend_delete() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let backend = HelixVectorBackend::open(
            &dir.path().to_string_lossy(),
            100 * 1024 * 1024,
        )
        .expect("open vector backend");

        let mut emb = vec![0.0f32; 384];
        emb[0] = 1.0;
        backend
            .upsert_embeddings(&[(1, emb.clone()), (2, emb.clone())])
            .expect("upsert");

        backend.delete(&[1]).expect("delete");

        // After delete, searching should not find node 1's embedding
        let results = backend.search(&emb, 10).expect("search");
        // Node 1 still exists but embedding is empty (won't match 384-dim query)
        let ids: Vec<NodeId> = results.iter().map(|r| r.0).collect();
        assert!(!ids.contains(&1), "Deleted embedding should not appear in results");
    }

    // -----------------------------------------------------------------------
    // Phase 2: HelixGraphBackend tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_helix_graph_backend_nodes_and_edges() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let backend = HelixGraphBackend::open(
            &dir.path().to_string_lossy(),
            100 * 1024 * 1024,
            100,
        )
        .expect("open graph backend");

        let nodes = vec![
            make_node(0, "A", NodeType::Entity),
            make_node(1, "B", NodeType::Entity),
            make_node(2, "C", NodeType::Entity),
        ];
        backend.upsert_nodes(&nodes).expect("upsert nodes");

        let edges = vec![
            make_edge(0, 1, EdgeType::RelatesTo),
            make_edge(1, 2, EdgeType::CausedBy),
        ];
        backend.upsert_edges(&edges).expect("upsert edges");

        // Get node
        let node = backend.get_node(1).expect("get").expect("exists");
        assert_eq!(node.content, "B");

        // Get neighbors
        let neighbors = backend.get_neighbors(0, 1).expect("neighbors");
        let ids: HashSet<u64> = neighbors.iter().map(|n| n.id).collect();
        assert!(ids.contains(&1));
    }

    #[test]
    fn test_helix_graph_backend_find_path() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let backend = HelixGraphBackend::open(
            &dir.path().to_string_lossy(),
            100 * 1024 * 1024,
            100,
        )
        .expect("open graph backend");

        let nodes: Vec<Node> = (0..5)
            .map(|i| make_node(i, &format!("N{}", i), NodeType::Entity))
            .collect();
        backend.upsert_nodes(&nodes).expect("upsert nodes");

        let edges: Vec<Edge> = (0..4)
            .map(|i| make_edge(i, i + 1, EdgeType::RelatesTo))
            .collect();
        backend.upsert_edges(&edges).expect("upsert edges");

        let path = backend
            .find_path(0, 4, 100)
            .expect("find_path")
            .expect("path exists");
        assert_eq!(path, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn test_helix_graph_backend_community_stub() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let backend = HelixGraphBackend::open(
            &dir.path().to_string_lossy(),
            100 * 1024 * 1024,
            100,
        )
        .expect("open graph backend");

        let community = backend.get_community(1).expect("get_community");
        assert!(community.is_empty(), "No communities assigned yet");
    }

    #[test]
    fn test_helix_graph_backend_community_detection_end_to_end() {
        use ucotron_core::community::{detect_communities, CommunityConfig};
        use ucotron_core::GraphBackend;

        let dir = tempfile::tempdir().expect("create temp dir");
        let backend = HelixGraphBackend::open(
            &dir.path().to_string_lossy(),
            100 * 1024 * 1024,
            100,
        )
        .expect("open graph backend");

        // Create two well-separated clusters with a weak bridge
        let nodes: Vec<Node> = (1..=8)
            .map(|id| make_node(id, &format!("node_{}", id), NodeType::Entity))
            .collect();
        backend.upsert_nodes(&nodes).expect("upsert nodes");

        let mut edges = Vec::new();
        // Cluster A (1-4): fully connected
        for i in 1..=4u64 {
            for j in (i + 1)..=4u64 {
                edges.push(Edge {
                    source: i,
                    target: j,
                    edge_type: EdgeType::RelatesTo,
                    weight: 1.0,
                    metadata: HashMap::new(),
                });
            }
        }
        // Cluster B (5-8): fully connected
        for i in 5..=8u64 {
            for j in (i + 1)..=8u64 {
                edges.push(Edge {
                    source: i,
                    target: j,
                    edge_type: EdgeType::RelatesTo,
                    weight: 1.0,
                    metadata: HashMap::new(),
                });
            }
        }
        // Weak bridge
        edges.push(Edge {
            source: 4,
            target: 5,
            edge_type: EdgeType::RelatesTo,
            weight: 0.1,
            metadata: HashMap::new(),
        });
        backend.upsert_edges(&edges).expect("upsert edges");

        // Get all edges from backend
        let all_edges = backend.get_all_edges().expect("get_all_edges");
        assert!(!all_edges.is_empty(), "Should have edges");

        // Run community detection
        let config = CommunityConfig::default();
        let result = detect_communities(&all_edges, &config).expect("detect communities");

        assert!(result.num_communities() >= 2, "Should detect >= 2 communities");
        assert_eq!(result.num_nodes(), 8);

        // Store community assignments
        backend
            .store_community_assignments(&result.node_to_community)
            .expect("store communities");

        // Now get_community should return members
        let community_of_1 = backend.get_community(1).expect("get_community");
        assert!(!community_of_1.is_empty(), "Node 1 should have community members");
        assert!(community_of_1.contains(&1));

        // Nodes in same cluster should be in same community
        let community_of_2 = backend.get_community(2).expect("get_community");
        assert_eq!(community_of_1, community_of_2, "Nodes 1 and 2 should share community");

        // Nodes in different clusters should have different communities
        let community_of_5 = backend.get_community(5).expect("get_community");
        assert_ne!(community_of_1, community_of_5, "Nodes 1 and 5 should be in different communities");
    }

    #[test]
    fn test_helix_graph_backend_get_all_edges() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let backend = HelixGraphBackend::open(
            &dir.path().to_string_lossy(),
            100 * 1024 * 1024,
            100,
        )
        .expect("open graph backend");

        let nodes = vec![
            make_node(1, "A", NodeType::Entity),
            make_node(2, "B", NodeType::Entity),
            make_node(3, "C", NodeType::Entity),
        ];
        backend.upsert_nodes(&nodes).expect("upsert nodes");

        let edges = vec![
            Edge {
                source: 1,
                target: 2,
                edge_type: EdgeType::RelatesTo,
                weight: 0.5,
                metadata: HashMap::new(),
            },
            Edge {
                source: 2,
                target: 3,
                edge_type: EdgeType::CausedBy,
                weight: 0.8,
                metadata: HashMap::new(),
            },
        ];
        backend.upsert_edges(&edges).expect("upsert edges");

        let all_edges = backend.get_all_edges().expect("get_all_edges");
        assert_eq!(all_edges.len(), 2);

        // Check weights are preserved
        let e12 = all_edges.iter().find(|(s, t, _)| *s == 1 && *t == 2);
        assert!(e12.is_some());
        assert!((e12.unwrap().2 - 0.5).abs() < 0.01);

        let e23 = all_edges.iter().find(|(s, t, _)| *s == 2 && *t == 3);
        assert!(e23.is_some());
        assert!((e23.unwrap().2 - 0.8).abs() < 0.01);
    }

    // -----------------------------------------------------------------------
    // Phase 2: Factory function tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_create_helix_vector_backend_from_config() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let config = ucotron_config::VectorBackendConfig {
            backend: "helix".to_string(),
            data_dir: dir.path().to_string_lossy().to_string(),
            max_db_size: 100 * 1024 * 1024,
            url: None,
            hnsw: ucotron_config::HnswConfig { enabled: false, ..Default::default() },
        };
        let backend = create_helix_vector_backend(&config).expect("create from config");
        let mut emb = vec![0.0f32; 384];
        emb[0] = 1.0;
        backend.upsert_embeddings(&[(1, emb)]).expect("upsert");
    }

    #[test]
    fn test_create_helix_graph_backend_from_config() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let config = ucotron_config::GraphBackendConfig {
            backend: "helix".to_string(),
            data_dir: dir.path().to_string_lossy().to_string(),
            max_db_size: 100 * 1024 * 1024,
            batch_size: 100,
            url: None,
        };
        let backend = create_helix_graph_backend(&config).expect("create from config");
        let node = make_node(1, "test", NodeType::Entity);
        backend.upsert_nodes(&[node]).expect("upsert");
        let retrieved = backend.get_node(1).expect("get").expect("exists");
        assert_eq!(retrieved.content, "test");
    }

    #[test]
    fn test_create_helix_backends_boxed() {
        let vec_dir = tempfile::tempdir().expect("create temp dir");
        let graph_dir = tempfile::tempdir().expect("create temp dir");

        let vec_config = ucotron_config::VectorBackendConfig {
            backend: "helix".to_string(),
            data_dir: vec_dir.path().to_string_lossy().to_string(),
            max_db_size: 100 * 1024 * 1024,
            url: None,
            hnsw: Default::default(),
        };
        let graph_config = ucotron_config::GraphBackendConfig {
            backend: "helix".to_string(),
            data_dir: graph_dir.path().to_string_lossy().to_string(),
            max_db_size: 100 * 1024 * 1024,
            batch_size: 100,
            url: None,
        };

        let (vec_backend, graph_backend) =
            create_helix_backends(&vec_config, &graph_config).expect("create backends");

        // Use via BackendRegistry
        let registry =
            ucotron_core::BackendRegistry::new(vec_backend, graph_backend);

        let node = make_node(1, "registry test", NodeType::Entity);
        registry.graph().upsert_nodes(&[node]).expect("upsert");
        let retrieved = registry.graph().get_node(1).expect("get").expect("exists");
        assert_eq!(retrieved.content, "registry test");

        let mut emb = vec![0.0f32; 384];
        emb[0] = 1.0;
        registry
            .vector()
            .upsert_embeddings(&[(1, emb.clone())])
            .expect("upsert embedding");
        let results = registry.vector().search(&emb, 1).expect("search");
        assert_eq!(results.len(), 1);
    }

    // -----------------------------------------------------------------------
    // Phase 2: HnswVectorBackend tests
    // -----------------------------------------------------------------------

    fn default_hnsw_config() -> ucotron_config::HnswConfig {
        ucotron_config::HnswConfig {
            ef_construction: 200,
            ef_search: 200,
            enabled: true,
        }
    }

    #[test]
    fn test_hnsw_upsert_and_search() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let backend = HnswVectorBackend::open(
            &dir.path().to_string_lossy(),
            100 * 1024 * 1024,
            default_hnsw_config(),
        )
        .expect("open hnsw backend");

        // Create two orthogonal embeddings
        let mut emb_a = vec![0.0f32; 384];
        emb_a[0] = 1.0;
        let mut emb_b = vec![0.0f32; 384];
        emb_b[1] = 1.0;

        backend
            .upsert_embeddings(&[(1, emb_a.clone()), (2, emb_b)])
            .expect("upsert");

        // Search for embedding aligned with dim 0
        let results = backend.search(&emb_a, 2).expect("search");
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, 1, "Exact match should be first");
        assert!(results[0].1 > results[1].1, "First result should have higher similarity");
    }

    #[test]
    fn test_hnsw_search_top_k_ordering() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let backend = HnswVectorBackend::open(
            &dir.path().to_string_lossy(),
            100 * 1024 * 1024,
            default_hnsw_config(),
        )
        .expect("open hnsw backend");

        // Insert 10 vectors with varying angles from dim 0
        let items: Vec<(NodeId, Vec<f32>)> = (0..10)
            .map(|i| {
                let angle = (i as f32) * 0.15; // spread across angles
                let mut emb = vec![0.0f32; 384];
                emb[0] = angle.cos();
                emb[1] = angle.sin();
                let norm = (emb[0] * emb[0] + emb[1] * emb[1]).sqrt();
                emb[0] /= norm;
                emb[1] /= norm;
                (i as u64, emb)
            })
            .collect();

        backend.upsert_embeddings(&items).expect("upsert");

        // Query aligned with dim 0
        let mut query = vec![0.0f32; 384];
        query[0] = 1.0;

        let results = backend.search(&query, 5).expect("search");
        assert_eq!(results.len(), 5);

        // Verify descending similarity
        for w in results.windows(2) {
            assert!(
                w[0].1 >= w[1].1,
                "Results should be sorted descending: {} >= {}",
                w[0].1,
                w[1].1
            );
        }

        // Node 0 (angle=0) should be most similar to dim-0 query
        assert_eq!(results[0].0, 0, "Node 0 should be closest to dim-0 query");
    }

    #[test]
    fn test_hnsw_delete_and_rebuild() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let backend = HnswVectorBackend::open(
            &dir.path().to_string_lossy(),
            100 * 1024 * 1024,
            default_hnsw_config(),
        )
        .expect("open hnsw backend");

        let mut emb = vec![0.0f32; 384];
        emb[0] = 1.0;

        backend
            .upsert_embeddings(&[(1, emb.clone()), (2, emb.clone()), (3, emb.clone())])
            .expect("upsert");

        // Delete node 1
        backend.delete(&[1]).expect("delete");

        // Search should not return node 1
        let results = backend.search(&emb, 10).expect("search");
        let ids: Vec<NodeId> = results.iter().map(|r| r.0).collect();
        assert!(!ids.contains(&1), "Deleted node should not appear");
        assert!(ids.contains(&2), "Node 2 should still be found");
        assert!(ids.contains(&3), "Node 3 should still be found");
    }

    #[test]
    fn test_hnsw_persistence_across_reopen() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let data_dir = dir.path().to_string_lossy().to_string();

        // Create and populate
        {
            let backend = HnswVectorBackend::open(
                &data_dir,
                100 * 1024 * 1024,
                default_hnsw_config(),
            )
            .expect("open hnsw backend");

            let mut emb_a = vec![0.0f32; 384];
            emb_a[0] = 1.0;
            let mut emb_b = vec![0.0f32; 384];
            emb_b[1] = 1.0;

            backend
                .upsert_embeddings(&[(1, emb_a), (2, emb_b)])
                .expect("upsert");
        }

        // Reopen and verify index was loaded from LMDB
        {
            let backend = HnswVectorBackend::open(
                &data_dir,
                100 * 1024 * 1024,
                default_hnsw_config(),
            )
            .expect("reopen hnsw backend");

            let mut query = vec![0.0f32; 384];
            query[0] = 1.0;

            let results = backend.search(&query, 2).expect("search after reopen");
            assert_eq!(results.len(), 2, "Should find both nodes after reopen");
            assert_eq!(results[0].0, 1, "Node 1 should be most similar to dim-0 query");
        }
    }

    #[test]
    fn test_hnsw_incremental_upsert() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let backend = HnswVectorBackend::open(
            &dir.path().to_string_lossy(),
            100 * 1024 * 1024,
            default_hnsw_config(),
        )
        .expect("open hnsw backend");

        // First batch
        let mut emb_a = vec![0.0f32; 384];
        emb_a[0] = 1.0;
        backend.upsert_embeddings(&[(1, emb_a.clone())]).expect("first upsert");

        // Second batch (incremental)
        let mut emb_b = vec![0.0f32; 384];
        emb_b[1] = 1.0;
        backend.upsert_embeddings(&[(2, emb_b)]).expect("second upsert");

        // Both should be searchable
        let results = backend.search(&emb_a, 2).expect("search");
        assert_eq!(results.len(), 2);
        let ids: Vec<NodeId> = results.iter().map(|r| r.0).collect();
        assert!(ids.contains(&1));
        assert!(ids.contains(&2));
    }

    #[test]
    fn test_hnsw_empty_search() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let backend = HnswVectorBackend::open(
            &dir.path().to_string_lossy(),
            100 * 1024 * 1024,
            default_hnsw_config(),
        )
        .expect("open hnsw backend");

        let query = vec![0.0f32; 384];
        let results = backend.search(&query, 10).expect("search empty");
        assert!(results.is_empty(), "Empty index should return empty results");
    }

    #[test]
    fn test_hnsw_from_config_factory() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let config = ucotron_config::VectorBackendConfig {
            backend: "helix".to_string(),
            data_dir: dir.path().to_string_lossy().to_string(),
            max_db_size: 100 * 1024 * 1024,
            url: None,
            hnsw: ucotron_config::HnswConfig {
                ef_construction: 100,
                ef_search: 50,
                enabled: true,
            },
        };
        let backend = create_hnsw_vector_backend(&config).expect("create from config");

        let mut emb = vec![0.0f32; 384];
        emb[0] = 1.0;
        backend.upsert_embeddings(&[(1, emb.clone())]).expect("upsert");

        let results = backend.search(&emb, 1).expect("search");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, 1);
        assert!((results[0].1 - 1.0).abs() < 0.01, "Exact match should have ~1.0 similarity");
    }

    #[test]
    fn test_create_helix_backends_uses_hnsw_by_default() {
        // Verify that create_helix_backends returns HNSW backend when enabled (default)
        let vec_dir = tempfile::tempdir().expect("create temp dir");
        let graph_dir = tempfile::tempdir().expect("create temp dir");

        let vec_config = ucotron_config::VectorBackendConfig {
            backend: "helix".to_string(),
            data_dir: vec_dir.path().to_string_lossy().to_string(),
            max_db_size: 100 * 1024 * 1024,
            url: None,
            hnsw: ucotron_config::HnswConfig {
                ef_construction: 200,
                ef_search: 200,
                enabled: true,
            },
        };
        let graph_config = ucotron_config::GraphBackendConfig {
            backend: "helix".to_string(),
            data_dir: graph_dir.path().to_string_lossy().to_string(),
            max_db_size: 100 * 1024 * 1024,
            batch_size: 100,
            url: None,
        };

        let (vec_backend, _graph_backend) =
            create_helix_backends(&vec_config, &graph_config).expect("create backends");

        // Verify it works as a VectorBackend
        let mut emb = vec![0.0f32; 384];
        emb[0] = 1.0;
        vec_backend.upsert_embeddings(&[(1, emb.clone())]).expect("upsert");
        let results = vec_backend.search(&emb, 1).expect("search");
        assert_eq!(results.len(), 1);
    }

    // -----------------------------------------------------------------------
    // Edge-case tests (US-12.2)
    // -----------------------------------------------------------------------

    #[test]
    fn test_graph_backend_upsert_node_overwrites() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let backend = HelixGraphBackend::open(
            &dir.path().to_string_lossy(),
            100 * 1024 * 1024,
            100,
        )
        .expect("open");

        let node_v1 = Node {
            id: 1,
            content: "version 1".to_string(),
            embedding: vec![],
            metadata: std::collections::HashMap::new(),
            node_type: NodeType::Entity,
            timestamp: 100,
            media_type: None,
            media_uri: None,
            embedding_visual: None,
            timestamp_range: None,
            parent_video_id: None,
        };
        backend.upsert_nodes(&[node_v1]).expect("upsert v1");

        let node_v2 = Node {
            id: 1,
            content: "version 2".to_string(),
            embedding: vec![],
            metadata: std::collections::HashMap::new(),
            node_type: NodeType::Event,
            timestamp: 200,
            media_type: None,
            media_uri: None,
            embedding_visual: None,
            timestamp_range: None,
            parent_video_id: None,
        };
        backend.upsert_nodes(&[node_v2]).expect("upsert v2");

        let retrieved = backend.get_node(1).expect("get").expect("exists");
        assert_eq!(retrieved.content, "version 2", "Upsert should overwrite");
        assert_eq!(retrieved.node_type, NodeType::Event);
    }

    #[test]
    fn test_vector_backend_search_empty() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let backend = HelixVectorBackend::open(
            &dir.path().to_string_lossy(),
            100 * 1024 * 1024,
        )
        .expect("open");

        let query = vec![1.0f32; 384];
        let results = backend.search(&query, 10).expect("search");
        assert!(results.is_empty(), "Empty backend should return no results");
    }

    // -----------------------------------------------------------------------
    // Phase 3.5: HelixVisualVectorBackend tests (512-dim CLIP)
    // -----------------------------------------------------------------------

    #[test]
    fn test_visual_backend_upsert_and_search() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let backend = HelixVisualVectorBackend::open(
            &dir.path().to_string_lossy(),
            100 * 1024 * 1024,
            HnswConfig::default(),
        )
        .expect("open visual backend");

        // Insert 3 visual embeddings (512-dim)
        let mut emb1 = vec![0.0f32; 512];
        emb1[0] = 1.0;
        let mut emb2 = vec![0.0f32; 512];
        emb2[1] = 1.0;
        let mut emb3 = vec![0.0f32; 512];
        emb3[0] = 0.7;
        emb3[1] = 0.7;

        backend
            .upsert_visual_embeddings(&[(1, emb1.clone()), (2, emb2), (3, emb3)])
            .expect("upsert");

        let results = backend.search_visual(&emb1, 2).expect("search");
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, 1, "Exact match should be first");
    }

    #[test]
    fn test_visual_backend_delete() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let backend = HelixVisualVectorBackend::open(
            &dir.path().to_string_lossy(),
            100 * 1024 * 1024,
            HnswConfig::default(),
        )
        .expect("open");

        let mut emb1 = vec![0.0f32; 512];
        emb1[0] = 1.0;
        let mut emb2 = vec![0.0f32; 512];
        emb2[1] = 1.0;

        backend
            .upsert_visual_embeddings(&[(1, emb1.clone()), (2, emb2)])
            .expect("upsert");

        backend.delete_visual(&[1]).expect("delete");

        let results = backend.search_visual(&emb1, 10).expect("search");
        assert_eq!(results.len(), 1, "One embedding should remain after delete");
        assert_eq!(results[0].0, 2);
    }

    #[test]
    fn test_visual_backend_empty_search() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let backend = HelixVisualVectorBackend::open(
            &dir.path().to_string_lossy(),
            100 * 1024 * 1024,
            HnswConfig::default(),
        )
        .expect("open");

        let query = vec![1.0f32; 512];
        let results = backend.search_visual(&query, 10).expect("search");
        assert!(results.is_empty());
    }

    #[test]
    fn test_visual_backend_top_k_zero() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let backend = HelixVisualVectorBackend::open(
            &dir.path().to_string_lossy(),
            100 * 1024 * 1024,
            HnswConfig::default(),
        )
        .expect("open");

        let mut emb = vec![0.0f32; 512];
        emb[0] = 1.0;
        backend
            .upsert_visual_embeddings(&[(1, emb)])
            .expect("upsert");

        let results = backend.search_visual(&[1.0f32; 512], 0).expect("search");
        assert!(results.is_empty());
    }

    #[test]
    fn test_visual_backend_persistence() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let dir_str = dir.path().to_string_lossy().to_string();

        let mut emb = vec![0.0f32; 512];
        emb[0] = 1.0;

        // Insert and drop
        {
            let backend = HelixVisualVectorBackend::open(
                &dir_str,
                100 * 1024 * 1024,
                HnswConfig::default(),
            )
            .expect("open");

            backend
                .upsert_visual_embeddings(&[(1, emb.clone())])
                .expect("upsert");
        }

        // Reopen and verify
        {
            let backend = HelixVisualVectorBackend::open(
                &dir_str,
                100 * 1024 * 1024,
                HnswConfig::default(),
            )
            .expect("reopen");

            let results = backend.search_visual(&emb, 1).expect("search");
            assert_eq!(results.len(), 1, "Persisted data should survive reopen");
            assert_eq!(results[0].0, 1);
        }
    }

    #[test]
    fn test_dual_index_independence() {
        // Verify text and visual backends have separate storage
        let text_dir = tempfile::tempdir().expect("create text dir");
        let vis_dir = tempfile::tempdir().expect("create visual dir");

        let text_backend = HnswVectorBackend::open(
            &text_dir.path().to_string_lossy(),
            100 * 1024 * 1024,
            HnswConfig::default(),
        )
        .expect("open text");

        let vis_backend = HelixVisualVectorBackend::open(
            &vis_dir.path().to_string_lossy(),
            100 * 1024 * 1024,
            HnswConfig::default(),
        )
        .expect("open visual");

        // Insert 384-dim into text backend
        let text_emb = vec![0.5f32; 384];
        text_backend
            .upsert_embeddings(&[(1, text_emb.clone())])
            .expect("upsert text");

        // Insert 512-dim into visual backend
        let mut vis_emb = vec![0.0f32; 512];
        vis_emb[0] = 1.0;
        vis_backend
            .upsert_visual_embeddings(&[(2, vis_emb.clone())])
            .expect("upsert visual");

        // Text search: finds node 1 only
        let text_results = text_backend.search(&text_emb, 10).expect("text search");
        assert_eq!(text_results.len(), 1);
        assert_eq!(text_results[0].0, 1);

        // Visual search: finds node 2 only
        let vis_results = vis_backend.search_visual(&vis_emb, 10).expect("visual search");
        assert_eq!(vis_results.len(), 1);
        assert_eq!(vis_results[0].0, 2);
    }

    #[test]
    fn test_visual_backend_upsert_overwrites() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let backend = HelixVisualVectorBackend::open(
            &dir.path().to_string_lossy(),
            100 * 1024 * 1024,
            HnswConfig::default(),
        )
        .expect("open");

        let mut emb1 = vec![0.0f32; 512];
        emb1[0] = 1.0;
        backend
            .upsert_visual_embeddings(&[(1, emb1)])
            .expect("upsert v1");

        // Overwrite with different embedding
        let mut emb2 = vec![0.0f32; 512];
        emb2[1] = 1.0;
        backend
            .upsert_visual_embeddings(&[(1, emb2.clone())])
            .expect("upsert v2");

        let results = backend.search_visual(&emb2, 1).expect("search");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, 1);
        assert!(
            results[0].1 > 0.9,
            "Should match overwritten embedding, got {}",
            results[0].1
        );
    }

    #[test]
    fn test_create_visual_backend_factory() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let config = VectorBackendConfig {
            backend: "helix".to_string(),
            data_dir: dir.path().to_string_lossy().to_string(),
            max_db_size: 100 * 1024 * 1024,
            url: None,
            hnsw: HnswConfig::default(),
        };
        let backend = create_helix_visual_backend(&config).expect("create visual");

        let mut emb = vec![0.0f32; 512];
        emb[0] = 1.0;
        backend
            .upsert_visual_embeddings(&[(1, emb.clone())])
            .expect("upsert");

        let results = backend.search_visual(&emb, 1).expect("search");
        assert_eq!(results.len(), 1);
    }

    // ----- Agent CRUD tests -----

    fn open_graph_backend() -> (tempfile::TempDir, HelixGraphBackend) {
        let dir = tempfile::tempdir().expect("create temp dir");
        let backend = HelixGraphBackend::open(
            &dir.path().to_string_lossy(),
            100 * 1024 * 1024,
            1000,
        )
        .expect("open graph backend");
        (dir, backend)
    }

    #[test]
    fn test_agent_create_and_get() {
        let (_dir, backend) = open_graph_backend();
        let agent = Agent::new("bot-1", "My Bot", "user-42", 1700000000);
        backend.create_agent(&agent).expect("create agent");

        let retrieved = backend.get_agent("bot-1").expect("get agent");
        assert_eq!(retrieved, Some(agent));
    }

    #[test]
    fn test_agent_get_nonexistent() {
        let (_dir, backend) = open_graph_backend();
        let result = backend.get_agent("nonexistent").expect("get agent");
        assert_eq!(result, None);
    }

    #[test]
    fn test_agent_list_all() {
        let (_dir, backend) = open_graph_backend();
        let a1 = Agent::new("bot-1", "Bot One", "alice", 1000);
        let a2 = Agent::new("bot-2", "Bot Two", "bob", 2000);
        let a3 = Agent::new("bot-3", "Bot Three", "alice", 3000);
        backend.create_agent(&a1).expect("create a1");
        backend.create_agent(&a2).expect("create a2");
        backend.create_agent(&a3).expect("create a3");

        let all = backend.list_agents(None).expect("list all");
        assert_eq!(all.len(), 3);
    }

    #[test]
    fn test_agent_list_by_owner() {
        let (_dir, backend) = open_graph_backend();
        let a1 = Agent::new("bot-1", "Bot One", "alice", 1000);
        let a2 = Agent::new("bot-2", "Bot Two", "bob", 2000);
        let a3 = Agent::new("bot-3", "Bot Three", "alice", 3000);
        backend.create_agent(&a1).expect("create a1");
        backend.create_agent(&a2).expect("create a2");
        backend.create_agent(&a3).expect("create a3");

        let alice_agents = backend.list_agents(Some("alice")).expect("list alice");
        assert_eq!(alice_agents.len(), 2);
        assert!(alice_agents.iter().all(|a| a.owner == "alice"));

        let bob_agents = backend.list_agents(Some("bob")).expect("list bob");
        assert_eq!(bob_agents.len(), 1);
        assert_eq!(bob_agents[0].id, "bot-2");
    }

    #[test]
    fn test_agent_update() {
        let (_dir, backend) = open_graph_backend();
        let agent = Agent::new("bot-1", "Original", "owner", 1000);
        backend.create_agent(&agent).expect("create");

        let updated = Agent::new("bot-1", "Updated Name", "owner", 1000);
        backend.create_agent(&updated).expect("update");

        let retrieved = backend.get_agent("bot-1").expect("get").unwrap();
        assert_eq!(retrieved.name, "Updated Name");
    }

    #[test]
    fn test_agent_delete() {
        let (_dir, backend) = open_graph_backend();
        let agent = Agent::new("bot-1", "Bot", "owner", 1000);
        backend.create_agent(&agent).expect("create");
        backend.delete_agent("bot-1").expect("delete");

        let result = backend.get_agent("bot-1").expect("get after delete");
        assert_eq!(result, None);
    }

    #[test]
    fn test_agent_delete_cascades_shares() {
        let (_dir, backend) = open_graph_backend();
        let a1 = Agent::new("bot-1", "Bot 1", "owner", 1000);
        let a2 = Agent::new("bot-2", "Bot 2", "owner", 2000);
        backend.create_agent(&a1).expect("create a1");
        backend.create_agent(&a2).expect("create a2");

        use ucotron_core::SharePermission;
        let share = AgentShare::new("bot-1", "bot-2", SharePermission::ReadOnly, 3000);
        backend.create_share(&share).expect("create share");

        // Delete bot-1 should cascade-remove the share
        backend.delete_agent("bot-1").expect("delete agent");
        let shares = backend.list_shares("bot-1").expect("list shares");
        assert!(shares.is_empty());
    }

    // ----- AgentShare CRUD tests -----

    #[test]
    fn test_share_create_and_get() {
        let (_dir, backend) = open_graph_backend();
        use ucotron_core::SharePermission;
        let share = AgentShare::new("a", "b", SharePermission::ReadWrite, 1000);
        backend.create_share(&share).expect("create share");

        let retrieved = backend.get_share("a", "b").expect("get share");
        assert_eq!(retrieved, Some(share));
    }

    #[test]
    fn test_share_get_nonexistent() {
        let (_dir, backend) = open_graph_backend();
        let result = backend.get_share("x", "y").expect("get share");
        assert_eq!(result, None);
    }

    #[test]
    fn test_share_list() {
        let (_dir, backend) = open_graph_backend();
        use ucotron_core::SharePermission;

        let s1 = AgentShare::new("a", "b", SharePermission::ReadOnly, 1000);
        let s2 = AgentShare::new("a", "c", SharePermission::ReadWrite, 2000);
        let s3 = AgentShare::new("b", "a", SharePermission::ReadOnly, 3000);
        backend.create_share(&s1).expect("s1");
        backend.create_share(&s2).expect("s2");
        backend.create_share(&s3).expect("s3");

        let a_shares = backend.list_shares("a").expect("list a shares");
        assert_eq!(a_shares.len(), 2);
        assert!(a_shares.iter().all(|s| s.agent_id == "a"));

        let b_shares = backend.list_shares("b").expect("list b shares");
        assert_eq!(b_shares.len(), 1);
        assert_eq!(b_shares[0].target_agent_id, "a");
    }

    #[test]
    fn test_share_delete() {
        let (_dir, backend) = open_graph_backend();
        use ucotron_core::SharePermission;
        let share = AgentShare::new("a", "b", SharePermission::ReadOnly, 1000);
        backend.create_share(&share).expect("create");
        backend.delete_share("a", "b").expect("delete");

        let result = backend.get_share("a", "b").expect("get after delete");
        assert_eq!(result, None);
    }

    #[test]
    fn test_agent_persistence_across_reopen() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let path = dir.path().to_string_lossy().to_string();

        // Create agent in first session
        {
            let backend = HelixGraphBackend::open(&path, 100 * 1024 * 1024, 1000)
                .expect("open");
            let agent = Agent::new("bot-1", "Persistent Bot", "owner", 1000);
            backend.create_agent(&agent).expect("create");
        }

        // Reopen and verify agent persists
        {
            let backend = HelixGraphBackend::open(&path, 100 * 1024 * 1024, 1000)
                .expect("reopen");
            let agent = backend.get_agent("bot-1").expect("get").unwrap();
            assert_eq!(agent.name, "Persistent Bot");
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// US-30.5: DUP_SORT Adjacency Index Evaluation
// ─────────────────────────────────────────────────────────────────────────────

/// DUP_SORT-based adjacency backend for benchmarking against Vec-based approach.
///
/// Instead of storing adjacency as `NodeId → Vec<AdjEntry>`, this uses LMDB's
/// native DUP_SORT flag where each `(NodeId, AdjEntry)` is a separate sorted
/// duplicate under the same key. This eliminates the read-modify-write cycle
/// for edge insertions.
#[cfg(test)]
mod dup_sort_eval {
    use super::*;
    use heed::DatabaseFlags;
    use heed::types::SerdeBincode;
    use std::time::Instant;

    /// Adjacency backend using Vec<AdjEntry> per node (current production approach).
    struct VecAdjBackend {
        env: Env,
        adj_out: Database<SerdeBincode<u64>, SerdeBincode<Vec<AdjEntry>>>,
        adj_in: Database<SerdeBincode<u64>, SerdeBincode<Vec<AdjEntry>>>,
    }

    impl VecAdjBackend {
        fn open(path: &str) -> Result<Self> {
            let p = Path::new(path);
            fs::create_dir_all(p)?;
            let env = unsafe {
                EnvOpenOptions::new()
                    .map_size(1_073_741_824) // 1GB
                    .max_dbs(2)
                    .open(p)?
            };
            let mut wtxn = env.write_txn()?;
            let adj_out = env.create_database(&mut wtxn, Some("vec_adj_out"))?;
            let adj_in = env.create_database(&mut wtxn, Some("vec_adj_in"))?;
            wtxn.commit()?;
            Ok(Self { env, adj_out, adj_in })
        }

        fn insert_edges(&self, edges: &[(u64, u64, u32)]) -> Result<()> {
            let mut wtxn = self.env.write_txn()?;
            for &(source, target, et) in edges {
                // Read-modify-write for outgoing
                let mut out_list = self.adj_out.get(&wtxn, &source)?.unwrap_or_default();
                out_list.push((target, et));
                self.adj_out.put(&mut wtxn, &source, &out_list)?;
                // Read-modify-write for incoming
                let mut in_list = self.adj_in.get(&wtxn, &target)?.unwrap_or_default();
                in_list.push((source, et));
                self.adj_in.put(&mut wtxn, &target, &in_list)?;
            }
            wtxn.commit()?;
            Ok(())
        }

        fn get_neighbors_out(&self, node_id: u64) -> Result<Vec<AdjEntry>> {
            let rtxn = self.env.read_txn()?;
            Ok(self.adj_out.get(&rtxn, &node_id)?.unwrap_or_default())
        }

        fn get_neighbors_bfs(&self, start: u64, hops: u8) -> Result<Vec<u64>> {
            let rtxn = self.env.read_txn()?;
            let mut visited = HashSet::new();
            let mut queue = VecDeque::new();
            visited.insert(start);
            queue.push_back((start, 0u8));
            let mut result = Vec::new();
            while let Some((current, depth)) = queue.pop_front() {
                if depth >= hops {
                    continue;
                }
                if let Some(out_list) = self.adj_out.get(&rtxn, &current)? {
                    for &(target, _et) in &out_list {
                        if visited.insert(target) {
                            result.push(target);
                            queue.push_back((target, depth + 1));
                        }
                    }
                }
                if let Some(in_list) = self.adj_in.get(&rtxn, &current)? {
                    for &(source, _et) in &in_list {
                        if visited.insert(source) {
                            result.push(source);
                            queue.push_back((source, depth + 1));
                        }
                    }
                }
            }
            Ok(result)
        }
    }

    /// Adjacency backend using LMDB DUP_SORT (one key per node, sorted duplicate values).
    ///
    /// Each adjacency entry `(neighbor_id, edge_type)` is stored as a separate
    /// sorted value under the node's key. No read-modify-write needed for inserts.
    struct DupSortAdjBackend {
        env: Env,
        adj_out: Database<SerdeBincode<u64>, SerdeBincode<AdjEntry>>,
        adj_in: Database<SerdeBincode<u64>, SerdeBincode<AdjEntry>>,
    }

    impl DupSortAdjBackend {
        fn open(path: &str) -> Result<Self> {
            let p = Path::new(path);
            fs::create_dir_all(p)?;
            let env = unsafe {
                EnvOpenOptions::new()
                    .map_size(1_073_741_824) // 1GB
                    .max_dbs(2)
                    .open(p)?
            };
            let mut wtxn = env.write_txn()?;
            let adj_out = env
                .database_options()
                .types::<SerdeBincode<u64>, SerdeBincode<AdjEntry>>()
                .flags(DatabaseFlags::DUP_SORT)
                .name("dup_adj_out")
                .create(&mut wtxn)?;
            let adj_in = env
                .database_options()
                .types::<SerdeBincode<u64>, SerdeBincode<AdjEntry>>()
                .flags(DatabaseFlags::DUP_SORT)
                .name("dup_adj_in")
                .create(&mut wtxn)?;
            wtxn.commit()?;
            Ok(Self { env, adj_out, adj_in })
        }

        fn insert_edges(&self, edges: &[(u64, u64, u32)]) -> Result<()> {
            let mut wtxn = self.env.write_txn()?;
            for &(source, target, et) in edges {
                // Direct put — LMDB handles sorted duplicate insertion natively
                self.adj_out.put(&mut wtxn, &source, &(target, et))?;
                self.adj_in.put(&mut wtxn, &target, &(source, et))?;
            }
            wtxn.commit()?;
            Ok(())
        }

        fn get_neighbors_out(&self, node_id: u64) -> Result<Vec<AdjEntry>> {
            let rtxn = self.env.read_txn()?;
            let mut result = Vec::new();
            let maybe_iter = self.adj_out.get_duplicates(&rtxn, &node_id)?;
            if let Some(iter) = maybe_iter {
                for item in iter {
                    let (_key, value) = item?;
                    result.push(value);
                }
            }
            Ok(result)
        }

        fn get_neighbors_bfs(&self, start: u64, hops: u8) -> Result<Vec<u64>> {
            let rtxn = self.env.read_txn()?;
            let mut visited = HashSet::new();
            let mut queue = VecDeque::new();
            visited.insert(start);
            queue.push_back((start, 0u8));
            let mut result = Vec::new();
            while let Some((current, depth)) = queue.pop_front() {
                if depth >= hops {
                    continue;
                }
                // Collect outgoing neighbors
                let out_neighbors: Vec<(u64, u32)> = {
                    let mut v = Vec::new();
                    let maybe_iter = self.adj_out.get_duplicates(&rtxn, &current)?;
                    if let Some(iter) = maybe_iter {
                        for item in iter {
                            let (_key, entry) = item?;
                            v.push(entry);
                        }
                    }
                    v
                };
                for (target, _et) in out_neighbors {
                    if visited.insert(target) {
                        result.push(target);
                        queue.push_back((target, depth + 1));
                    }
                }
                // Collect incoming neighbors
                let in_neighbors: Vec<(u64, u32)> = {
                    let mut v = Vec::new();
                    let maybe_iter = self.adj_in.get_duplicates(&rtxn, &current)?;
                    if let Some(iter) = maybe_iter {
                        for item in iter {
                            let (_key, entry) = item?;
                            v.push(entry);
                        }
                    }
                    v
                };
                for (source, _et) in in_neighbors {
                    if visited.insert(source) {
                        result.push(source);
                        queue.push_back((source, depth + 1));
                    }
                }
            }
            Ok(result)
        }
    }

    /// Generate a power-law edge set for benchmarking.
    fn generate_bench_edges(node_count: u64, edge_count: usize, seed: u64) -> Vec<(u64, u64, u32)> {
        use rand::prelude::*;
        use rand_chacha::ChaCha8Rng;
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut edges = Vec::with_capacity(edge_count);
        for _ in 0..edge_count {
            let source = rng.gen_range(0..node_count);
            // Zipf-like: bias toward lower-numbered targets for hub effect
            let target = {
                let r: f64 = rng.gen();
                ((r.powf(2.0) * node_count as f64) as u64).min(node_count - 1)
            };
            if source != target {
                let et = rng.gen_range(0..10u32);
                edges.push((source, target, et));
            }
        }
        edges
    }

    /// Run the full DUP_SORT vs Vec benchmark and return structured results.
    pub fn run_benchmark(
        node_count: u64,
        edge_count: usize,
    ) -> Result<DupSortEvalResult> {
        let edges = generate_bench_edges(node_count, edge_count, 42);

        // --- Vec-based benchmark ---
        let vec_dir = tempfile::tempdir()?;
        let vec_path = vec_dir.path().to_string_lossy().to_string();
        let vec_backend = VecAdjBackend::open(&vec_path)?;

        let t0 = Instant::now();
        vec_backend.insert_edges(&edges)?;
        let vec_insert_us = t0.elapsed().as_micros() as u64;

        // Read benchmark: get neighbors for 100 random hub nodes (low IDs = high degree)
        let read_nodes: Vec<u64> = (0..100).collect();
        let t0 = Instant::now();
        for &node_id in &read_nodes {
            let _ = vec_backend.get_neighbors_out(node_id)?;
        }
        let vec_read_us = t0.elapsed().as_micros() as u64;

        // BFS benchmark: 2-hop traversal from 10 hub nodes
        let bfs_nodes: Vec<u64> = (0..10).collect();
        let t0 = Instant::now();
        for &node_id in &bfs_nodes {
            let _ = vec_backend.get_neighbors_bfs(node_id, 2)?;
        }
        let vec_bfs_us = t0.elapsed().as_micros() as u64;

        // Disk size
        let vec_disk = dir_size_bytes(vec_dir.path());

        // --- DUP_SORT benchmark ---
        let dup_dir = tempfile::tempdir()?;
        let dup_path = dup_dir.path().to_string_lossy().to_string();
        let dup_backend = DupSortAdjBackend::open(&dup_path)?;

        let t0 = Instant::now();
        dup_backend.insert_edges(&edges)?;
        let dup_insert_us = t0.elapsed().as_micros() as u64;

        // Read benchmark: same nodes
        let t0 = Instant::now();
        for &node_id in &read_nodes {
            let _ = dup_backend.get_neighbors_out(node_id)?;
        }
        let dup_read_us = t0.elapsed().as_micros() as u64;

        // BFS benchmark: same nodes
        let t0 = Instant::now();
        for &node_id in &bfs_nodes {
            let _ = dup_backend.get_neighbors_bfs(node_id, 2)?;
        }
        let dup_bfs_us = t0.elapsed().as_micros() as u64;

        // Disk size
        let dup_disk = dir_size_bytes(dup_dir.path());

        // Verify correctness: same neighbor sets
        // Verify correctness: same unique neighbor sets (Vec allows dups, DUP_SORT doesn't)
        for &node_id in &read_nodes {
            let mut vec_neighbors = vec_backend.get_neighbors_out(node_id)?;
            let mut dup_neighbors = dup_backend.get_neighbors_out(node_id)?;
            vec_neighbors.sort();
            vec_neighbors.dedup();
            dup_neighbors.sort();
            assert_eq!(vec_neighbors, dup_neighbors, "Mismatch for node {}", node_id);
        }

        Ok(DupSortEvalResult {
            node_count,
            edge_count: edges.len(),
            vec_insert_us,
            vec_read_us,
            vec_bfs_us,
            vec_disk_bytes: vec_disk,
            dup_insert_us,
            dup_read_us,
            dup_bfs_us,
            dup_disk_bytes: dup_disk,
        })
    }

    fn dir_size_bytes(path: &Path) -> u64 {
        let mut size = 0u64;
        if let Ok(entries) = fs::read_dir(path) {
            for entry in entries.flatten() {
                let meta = entry.metadata().unwrap_or_else(|_| {
                    fs::metadata(entry.path()).expect("metadata")
                });
                if meta.is_file() {
                    size += meta.len();
                } else if meta.is_dir() {
                    size += dir_size_bytes(&entry.path());
                }
            }
        }
        size
    }

    /// Results of the DUP_SORT evaluation benchmark.
    #[derive(Debug)]
    pub struct DupSortEvalResult {
        pub node_count: u64,
        pub edge_count: usize,
        pub vec_insert_us: u64,
        pub vec_read_us: u64,
        pub vec_bfs_us: u64,
        pub vec_disk_bytes: u64,
        pub dup_insert_us: u64,
        pub dup_read_us: u64,
        pub dup_bfs_us: u64,
        pub dup_disk_bytes: u64,
    }

    impl DupSortEvalResult {
        pub fn to_markdown(&self) -> String {
            let edge_tp_vec = self.edge_count as f64 / (self.vec_insert_us as f64 / 1_000_000.0);
            let edge_tp_dup = self.edge_count as f64 / (self.dup_insert_us as f64 / 1_000_000.0);
            let insert_speedup = self.vec_insert_us as f64 / self.dup_insert_us as f64;
            let read_speedup = self.vec_read_us as f64 / self.dup_read_us as f64;
            let bfs_speedup = self.vec_bfs_us as f64 / self.dup_bfs_us as f64;
            let disk_ratio = self.dup_disk_bytes as f64 / self.vec_disk_bytes as f64;

            format!(
                "| Metric | Vec<AdjEntry> | DUP_SORT | Speedup |\n\
                 |--------|---------------|----------|:-------:|\n\
                 | Edge insertion ({} edges) | {:.2}ms | {:.2}ms | {:.2}x |\n\
                 | Edge throughput | {:.0} edges/s | {:.0} edges/s | — |\n\
                 | Neighbor read (100 nodes) | {:.2}ms | {:.2}ms | {:.2}x |\n\
                 | BFS 2-hop (10 nodes) | {:.2}ms | {:.2}ms | {:.2}x |\n\
                 | Disk size | {:.2} MB | {:.2} MB | {:.2}x |",
                self.edge_count,
                self.vec_insert_us as f64 / 1000.0,
                self.dup_insert_us as f64 / 1000.0,
                insert_speedup,
                edge_tp_vec,
                edge_tp_dup,
                self.vec_read_us as f64 / 1000.0,
                self.dup_read_us as f64 / 1000.0,
                read_speedup,
                self.vec_bfs_us as f64 / 1000.0,
                self.dup_bfs_us as f64 / 1000.0,
                bfs_speedup,
                self.vec_disk_bytes as f64 / (1024.0 * 1024.0),
                self.dup_disk_bytes as f64 / (1024.0 * 1024.0),
                disk_ratio,
            )
        }
    }

    // ─── Tests ────────────────────────────────────────────────────────────────

    #[test]
    fn test_dup_sort_basic_insert_and_read() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let path = dir.path().to_string_lossy().to_string();
        let backend = DupSortAdjBackend::open(&path).expect("open");

        // Insert edges: 1→2, 1→3, 1→4
        backend
            .insert_edges(&[(1, 2, 0), (1, 3, 1), (1, 4, 2)])
            .expect("insert");

        // Read outgoing neighbors of node 1
        let neighbors = backend.get_neighbors_out(1).expect("read");
        assert_eq!(neighbors.len(), 3);
        let target_ids: Vec<u64> = neighbors.iter().map(|e| e.0).collect();
        assert!(target_ids.contains(&2));
        assert!(target_ids.contains(&3));
        assert!(target_ids.contains(&4));
    }

    #[test]
    fn test_dup_sort_empty_node() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let path = dir.path().to_string_lossy().to_string();
        let backend = DupSortAdjBackend::open(&path).expect("open");

        let neighbors = backend.get_neighbors_out(999).expect("read");
        assert!(neighbors.is_empty());
    }

    #[test]
    fn test_dup_sort_bfs_traversal() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let path = dir.path().to_string_lossy().to_string();
        let backend = DupSortAdjBackend::open(&path).expect("open");

        // Chain: 1→2→3→4→5
        backend
            .insert_edges(&[(1, 2, 0), (2, 3, 0), (3, 4, 0), (4, 5, 0)])
            .expect("insert");

        // 1-hop from node 1: should find node 2
        let neighbors_1 = backend.get_neighbors_bfs(1, 1).expect("bfs 1-hop");
        assert_eq!(neighbors_1, vec![2]);

        // 2-hop from node 1: should find nodes 2, 3
        let mut neighbors_2 = backend.get_neighbors_bfs(1, 2).expect("bfs 2-hop");
        neighbors_2.sort();
        assert_eq!(neighbors_2, vec![2, 3]);
    }

    #[test]
    fn test_dup_sort_correctness_vs_vec() {
        let dir_vec = tempfile::tempdir().expect("create vec dir");
        let dir_dup = tempfile::tempdir().expect("create dup dir");
        let vec_backend =
            VecAdjBackend::open(&dir_vec.path().to_string_lossy()).expect("open vec");
        let dup_backend =
            DupSortAdjBackend::open(&dir_dup.path().to_string_lossy()).expect("open dup");

        let edges = generate_bench_edges(100, 500, 123);
        vec_backend.insert_edges(&edges).expect("vec insert");
        dup_backend.insert_edges(&edges).expect("dup insert");

        // Verify all nodes produce the same neighbor sets.
        // Note: Vec approach allows duplicate (target, edge_type) entries while
        // DUP_SORT deduplicates identical values natively. We dedup Vec results
        // to compare the unique neighbor sets.
        for node_id in 0..100 {
            let mut vec_out = vec_backend.get_neighbors_out(node_id).expect("vec read");
            let mut dup_out = dup_backend.get_neighbors_out(node_id).expect("dup read");
            vec_out.sort();
            vec_out.dedup(); // Remove duplicates that Vec allows but DUP_SORT prevents
            dup_out.sort();
            assert_eq!(vec_out, dup_out, "Mismatch for node {}", node_id);
        }
    }

    #[test]
    fn test_dup_sort_benchmark_small() {
        // Small benchmark: 1k nodes, 5k edges — verifies benchmark harness works
        let result = run_benchmark(1_000, 5_000).expect("benchmark");
        assert!(result.vec_insert_us > 0);
        assert!(result.dup_insert_us > 0);
        assert!(result.vec_read_us > 0);
        assert!(result.dup_read_us > 0);
        assert!(result.edge_count > 0);
        // Print results for manual inspection
        eprintln!("\n--- DUP_SORT Evaluation (1k nodes, 5k edges) ---");
        eprintln!("{}", result.to_markdown());
    }

    #[test]
    fn test_dup_sort_high_degree_node() {
        let dir_vec = tempfile::tempdir().expect("create vec dir");
        let dir_dup = tempfile::tempdir().expect("create dup dir");
        let vec_backend =
            VecAdjBackend::open(&dir_vec.path().to_string_lossy()).expect("open vec");
        let dup_backend =
            DupSortAdjBackend::open(&dir_dup.path().to_string_lossy()).expect("open dup");

        // Create a hub node 0 with 1000 outgoing edges
        let edges: Vec<(u64, u64, u32)> = (1..=1000).map(|t| (0, t, 0)).collect();

        let t0 = Instant::now();
        vec_backend.insert_edges(&edges).expect("vec insert");
        let vec_us = t0.elapsed().as_micros();

        let t0 = Instant::now();
        dup_backend.insert_edges(&edges).expect("dup insert");
        let dup_us = t0.elapsed().as_micros();

        // DUP_SORT should be faster for high-degree node insertion
        // because Vec approach does read-modify-write 1000 times (growing list each time)
        eprintln!("\n--- High-degree node (1000 edges) ---");
        eprintln!("Vec insert: {}us, DUP_SORT insert: {}us", vec_us, dup_us);

        // Verify correctness
        let mut vec_out = vec_backend.get_neighbors_out(0).expect("vec read");
        let mut dup_out = dup_backend.get_neighbors_out(0).expect("dup read");
        vec_out.sort();
        dup_out.sort();
        assert_eq!(vec_out.len(), 1000);
        assert_eq!(vec_out, dup_out);
    }

    #[test]
    #[ignore] // Run with --ignored for full benchmark
    fn test_dup_sort_benchmark_10k() {
        // Medium benchmark: 10k nodes, 50k edges
        let result = run_benchmark(10_000, 50_000).expect("benchmark");
        eprintln!("\n--- DUP_SORT Evaluation (10k nodes, 50k edges) ---");
        eprintln!("{}", result.to_markdown());
    }

    #[test]
    #[ignore] // Run with --ignored for full benchmark
    fn test_dup_sort_benchmark_100k() {
        // Large benchmark: 100k nodes, 500k edges (PRD minimum benchmark)
        let result = run_benchmark(100_000, 500_000).expect("benchmark");
        eprintln!("\n--- DUP_SORT Evaluation (100k nodes, 500k edges) ---");
        eprintln!("{}", result.to_markdown());
    }
}
