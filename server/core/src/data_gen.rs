//! Deterministic synthetic data generator for Ucotron benchmarks.
//!
//! Generates reproducible nodes and edges using seeded RNG (`rand_chacha`).
//! All outputs are deterministic for a given seed, enabling reproducible benchmarks.
//!
//! # Data Characteristics
//! - **Nodes**: Sequential IDs, lorem-ipsum-style content (50-200 chars),
//!   normalized 384-dim embeddings, type mix (60% Entity, 25% Event, 15% Fact),
//!   timestamps distributed over 1 year.
//! - **Edges**: Power-law degree distribution (few hubs, many low-degree nodes),
//!   mixed edge types, weights between 0.1 and 1.0.

use crate::{Edge, EdgeType, Node, NodeType, Value};
use rand::distributions::{Distribution, Uniform};
use rand::Rng;
use rand_chacha::ChaCha8Rng;
use rand::SeedableRng;
use std::collections::HashMap;
use std::io::{BufReader, BufWriter};
use std::path::Path;

/// Word pool for generating pseudo-lorem-ipsum content.
const WORDS: &[&str] = &[
    "memoria", "cognitiva", "grafo", "nodo", "arista", "vector", "semántico",
    "episódico", "procedimental", "entidad", "evento", "hecho", "relación",
    "causal", "temporal", "conflicto", "resolución", "confianza", "embedding",
    "travesía", "búsqueda", "índice", "latencia", "rendimiento", "datos",
    "sistema", "modelo", "algoritmo", "inferencia", "consulta", "resultado",
    "profundidad", "amplitud", "ciclo", "ruta", "peso", "umbral", "cluster",
    "comunidad", "decaimiento", "similitud", "distancia", "dimensión",
    "normalización", "serialización", "benchmark", "ingesta", "lectura",
    "escritura", "operación", "transacción", "lote", "concurrencia",
    "estructura", "propiedad", "atributo", "contexto", "conocimiento",
    "experiencia", "habilidad", "herramienta", "aprendizaje", "adaptación",
];

/// Serializable container for generated benchmark data.
#[derive(serde::Serialize, serde::Deserialize)]
pub struct DataSet {
    pub nodes: Vec<Node>,
    pub edges: Vec<Edge>,
}

/// Generate `count` deterministic nodes using the given `seed`.
///
/// Each node has:
/// - Sequential ID starting at 0
/// - Lorem-ipsum-style content (50-200 characters)
/// - Normalized 384-dim f32 embedding (simulating sentence-transformers)
/// - Type distribution: 60% Entity, 25% Event, 15% Fact
/// - Timestamps uniformly distributed over 1 year (from a base epoch)
///
/// # Examples
/// ```
/// let nodes = ucotron_core::data_gen::generate_nodes(100, 42);
/// assert_eq!(nodes.len(), 100);
/// assert_eq!(nodes[0].id, 0);
/// ```
pub fn generate_nodes(count: usize, seed: u64) -> Vec<Node> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let word_dist = Uniform::new(0, WORDS.len());
    let embed_dist = Uniform::new(-1.0f32, 1.0f32);
    // Base timestamp: 2025-01-01 00:00:00 UTC
    let base_ts: u64 = 1_735_689_600;
    let one_year_secs: u64 = 365 * 24 * 3600;
    let ts_dist = Uniform::new(base_ts, base_ts + one_year_secs);

    let mut nodes = Vec::with_capacity(count);

    for id in 0..count {
        // Generate content: random word count targeting 50-200 chars
        let word_count = rng.gen_range(8..30);
        let content = generate_content(&mut rng, &word_dist, word_count);

        // Generate and normalize embedding (384-dim)
        let embedding = generate_embedding(&mut rng, &embed_dist);

        // Determine node type by distribution: 60% Entity, 25% Event, 15% Fact
        let type_roll: f32 = rng.gen();
        let node_type = if type_roll < 0.60 {
            NodeType::Entity
        } else if type_roll < 0.85 {
            NodeType::Event
        } else {
            NodeType::Fact
        };

        let timestamp = ts_dist.sample(&mut rng);

        let mut metadata = HashMap::new();
        metadata.insert("gen_seed".to_string(), Value::Integer(seed as i64));

        nodes.push(Node {
            id: id as u64,
            content,
            embedding,
            metadata,
            node_type,
            timestamp,
            media_type: None,
            media_uri: None,
            embedding_visual: None,
            timestamp_range: None,
            parent_video_id: None,
        });
    }

    nodes
}

/// Generate content from the word pool.
fn generate_content(rng: &mut ChaCha8Rng, word_dist: &Uniform<usize>, word_count: usize) -> String {
    let mut words = Vec::with_capacity(word_count);
    for _ in 0..word_count {
        let idx = word_dist.sample(rng);
        words.push(WORDS[idx]);
    }
    words.join(" ")
}

/// Generate a normalized 384-dimensional embedding vector.
fn generate_embedding(rng: &mut ChaCha8Rng, dist: &Uniform<f32>) -> Vec<f32> {
    let mut vec: Vec<f32> = (0..384).map(|_| dist.sample(rng)).collect();
    // L2 normalize
    let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for v in vec.iter_mut() {
            *v /= norm;
        }
    }
    vec
}

/// Generate `count` deterministic edges for the given nodes using a power-law distribution.
///
/// The power-law distribution creates a realistic graph topology where a few "hub" nodes
/// have many connections while most nodes have few. This is achieved by sampling source
/// nodes uniformly but biasing target selection with a Zipf-like distribution.
///
/// Edge type distribution (for structural realism):
/// - 50% RelatesTo (general semantic links)
/// - 15% CausedBy (causal chains)
/// - 10% HasProperty (attribute links)
/// - 10% NextEpisode (temporal chains)
/// - 5% ConflictsWith (contradiction markers)
/// - 5% Actor (event-entity links)
/// - 3% Object (event-entity links)
/// - 2% Location (event-place links)
///
/// # Panics
/// Panics if `nodes` is empty.
pub fn generate_edges(nodes: &[Node], count: usize, seed: u64) -> Vec<Edge> {
    assert!(!nodes.is_empty(), "Cannot generate edges for empty node set");

    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let n = nodes.len();
    let weight_dist = Uniform::new(0.1f32, 1.0f32);

    // Pre-compute power-law weights for target selection (Zipf-like)
    // Weight[i] = 1 / (i + 1)^0.8  — gives a heavy-tailed distribution
    let zipf_weights: Vec<f64> = (0..n).map(|i| 1.0 / ((i + 1) as f64).powf(0.8)).collect();
    let zipf_total: f64 = zipf_weights.iter().sum();

    let mut edges = Vec::with_capacity(count);

    for _ in 0..count {
        // Source: uniform random
        let source = rng.gen_range(0..n) as u64;

        // Target: power-law (Zipf-like) via inverse CDF sampling
        let target = sample_zipf(&mut rng, &zipf_weights, zipf_total, n);

        // Avoid self-loops
        let target = if target == source {
            (target + 1) % (n as u64)
        } else {
            target
        };

        // Edge type by distribution
        let type_roll: f32 = rng.gen();
        let edge_type = if type_roll < 0.50 {
            EdgeType::RelatesTo
        } else if type_roll < 0.65 {
            EdgeType::CausedBy
        } else if type_roll < 0.75 {
            EdgeType::HasProperty
        } else if type_roll < 0.85 {
            EdgeType::NextEpisode
        } else if type_roll < 0.90 {
            EdgeType::ConflictsWith
        } else if type_roll < 0.95 {
            EdgeType::Actor
        } else if type_roll < 0.98 {
            EdgeType::Object
        } else {
            EdgeType::Location
        };

        let weight = weight_dist.sample(&mut rng);

        edges.push(Edge {
            source,
            target,
            edge_type,
            weight,
            metadata: HashMap::new(),
        });
    }

    edges
}

/// Sample from a Zipf-like distribution using inverse CDF.
fn sample_zipf(rng: &mut ChaCha8Rng, weights: &[f64], total: f64, n: usize) -> u64 {
    let u: f64 = rng.gen::<f64>() * total;
    let mut cumulative = 0.0;
    for (i, weight) in weights.iter().enumerate().take(n) {
        cumulative += weight;
        if u <= cumulative {
            return i as u64;
        }
    }
    // Fallback (shouldn't happen due to floating point, but be safe)
    (n - 1) as u64
}

/// Generate a linear chain graph of the given `depth`.
///
/// Creates a chain: node_0 → node_1 → node_2 → ... → node_{depth-1}
/// with `depth` nodes and `depth - 1` edges (all `RELATES_TO` with weight 1.0).
///
/// Each node has a normalized 384-dim embedding and Entity type.
/// Useful for benchmarking deep traversal and path-finding queries.
///
/// # Arguments
/// * `depth` - Number of nodes in the chain (must be >= 1)
/// * `seed` - RNG seed for deterministic embedding generation
///
/// # Returns
/// A tuple of `(nodes, edges)` representing the chain.
///
/// # Panics
/// Panics if `depth` is 0.
pub fn generate_chain(depth: usize, seed: u64) -> (Vec<Node>, Vec<Edge>) {
    assert!(depth >= 1, "Chain depth must be at least 1");

    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let embed_dist = Uniform::new(-1.0f32, 1.0f32);
    let base_ts: u64 = 1_735_689_600;

    let mut nodes = Vec::with_capacity(depth);
    let mut edges = Vec::with_capacity(depth.saturating_sub(1));

    for i in 0..depth {
        let embedding = generate_embedding(&mut rng, &embed_dist);
        nodes.push(Node {
            id: i as u64,
            content: format!("chain_node_{}", i),
            embedding,
            metadata: HashMap::new(),
            node_type: NodeType::Entity,
            timestamp: base_ts + i as u64,
            media_type: None,
            media_uri: None,
            embedding_visual: None,
            timestamp_range: None,
            parent_video_id: None,
        });

        if i > 0 {
            edges.push(Edge {
                source: (i - 1) as u64,
                target: i as u64,
                edge_type: EdgeType::RelatesTo,
                weight: 1.0,
                metadata: HashMap::new(),
            });
        }
    }

    (nodes, edges)
}

/// Generate a tree graph with the given `depth` and `branching` factor.
///
/// Creates a rooted tree where each non-leaf node has exactly `branching` children.
/// Edges are directed parent → child, all `RELATES_TO` with weight 1.0.
///
/// Total nodes = (branching^depth - 1) / (branching - 1) for branching > 1,
/// or `depth` for branching == 1 (degenerates to a chain).
///
/// Each node has a normalized 384-dim embedding and Entity type.
/// Useful for benchmarking worst-case traversal (exponential blowup).
///
/// # Arguments
/// * `depth` - Depth of the tree (root is at depth 0, leaves at depth `depth - 1`)
/// * `branching` - Number of children per non-leaf node (must be >= 1)
/// * `seed` - RNG seed for deterministic embedding generation
///
/// # Returns
/// A tuple of `(nodes, edges)` representing the tree.
///
/// # Panics
/// Panics if `depth` is 0 or `branching` is 0.
pub fn generate_tree(depth: usize, branching: usize, seed: u64) -> (Vec<Node>, Vec<Edge>) {
    assert!(depth >= 1, "Tree depth must be at least 1");
    assert!(branching >= 1, "Branching factor must be at least 1");

    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let embed_dist = Uniform::new(-1.0f32, 1.0f32);
    let base_ts: u64 = 1_735_689_600;

    // Pre-calculate total node count to size vectors
    let total_nodes = if branching == 1 {
        depth
    } else {
        // Geometric series: (b^depth - 1) / (b - 1)
        (branching.pow(depth as u32) - 1) / (branching - 1)
    };

    let mut nodes = Vec::with_capacity(total_nodes);
    let mut edges = Vec::with_capacity(total_nodes.saturating_sub(1));

    // BFS-style level-order construction
    // Start with root (id 0)
    let mut next_id: u64 = 0;
    let root_embedding = generate_embedding(&mut rng, &embed_dist);
    nodes.push(Node {
        id: next_id,
        content: "tree_node_0".to_string(),
        embedding: root_embedding,
        metadata: HashMap::new(),
        node_type: NodeType::Entity,
        timestamp: base_ts,
        media_type: None,
        media_uri: None,
        embedding_visual: None,
        timestamp_range: None,
        parent_video_id: None,
    });
    next_id += 1;

    // current_level holds IDs of nodes at the current depth level
    let mut current_level = vec![0u64];

    for _level in 1..depth {
        let mut next_level = Vec::with_capacity(current_level.len() * branching);

        for &parent_id in &current_level {
            for _ in 0..branching {
                let child_id = next_id;
                let embedding = generate_embedding(&mut rng, &embed_dist);
                nodes.push(Node {
                    id: child_id,
                    content: format!("tree_node_{}", child_id),
                    embedding,
                    metadata: HashMap::new(),
                    node_type: NodeType::Entity,
                    timestamp: base_ts + child_id,
                    media_type: None,
                    media_uri: None,
                    embedding_visual: None,
                    timestamp_range: None,
                    parent_video_id: None,
                });
                edges.push(Edge {
                    source: parent_id,
                    target: child_id,
                    edge_type: EdgeType::RelatesTo,
                    weight: 1.0,
                    metadata: HashMap::new(),
                });
                next_level.push(child_id);
                next_id += 1;
            }
        }

        current_level = next_level;
    }

    (nodes, edges)
}

/// Save nodes and edges to a binary file using bincode serialization.
///
/// The file format is a bincode-encoded `DataSet` struct containing both
/// nodes and edges. This separates data generation time from benchmark time.
///
/// # Errors
/// Returns an error if the file cannot be created or serialization fails.
pub fn save_to_bin<P: AsRef<Path>>(path: P, nodes: &[Node], edges: &[Edge]) -> anyhow::Result<()> {
    let file = std::fs::File::create(path)?;
    let writer = BufWriter::new(file);
    let dataset = DataSet {
        nodes: nodes.to_vec(),
        edges: edges.to_vec(),
    };
    bincode::serialize_into(writer, &dataset)?;
    Ok(())
}

/// Load nodes and edges from a binary file previously saved with [`save_to_bin`].
///
/// # Errors
/// Returns an error if the file cannot be read or deserialization fails.
pub fn load_from_bin<P: AsRef<Path>>(path: P) -> anyhow::Result<DataSet> {
    let file = std::fs::File::open(path)?;
    let reader = BufReader::new(file);
    let dataset: DataSet = bincode::deserialize_from(reader)?;
    Ok(dataset)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_nodes_count_and_ids() {
        let nodes = generate_nodes(100, 42);
        assert_eq!(nodes.len(), 100);
        // IDs should be sequential starting at 0
        for (i, node) in nodes.iter().enumerate() {
            assert_eq!(node.id, i as u64);
        }
    }

    #[test]
    fn test_generate_nodes_deterministic() {
        let nodes_a = generate_nodes(50, 42);
        let nodes_b = generate_nodes(50, 42);
        for (a, b) in nodes_a.iter().zip(nodes_b.iter()) {
            assert_eq!(a.id, b.id);
            assert_eq!(a.content, b.content);
            assert_eq!(a.embedding, b.embedding);
            assert_eq!(a.node_type, b.node_type);
            assert_eq!(a.timestamp, b.timestamp);
        }
    }

    #[test]
    fn test_generate_nodes_different_seeds() {
        let nodes_a = generate_nodes(10, 1);
        let nodes_b = generate_nodes(10, 2);
        // With different seeds, content should differ (statistically guaranteed)
        let same_content = nodes_a
            .iter()
            .zip(nodes_b.iter())
            .filter(|(a, b)| a.content == b.content)
            .count();
        assert!(same_content < nodes_a.len(), "Different seeds should produce different data");
    }

    #[test]
    fn test_generate_nodes_embedding_normalized() {
        let nodes = generate_nodes(20, 42);
        for node in &nodes {
            assert_eq!(node.embedding.len(), 384);
            let norm: f32 = node.embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!((norm - 1.0).abs() < 1e-5, "Embedding should be L2 normalized, got norm={}", norm);
        }
    }

    #[test]
    fn test_generate_nodes_type_distribution() {
        let nodes = generate_nodes(10_000, 42);
        let entity_count = nodes.iter().filter(|n| n.node_type == NodeType::Entity).count();
        let event_count = nodes.iter().filter(|n| n.node_type == NodeType::Event).count();
        let fact_count = nodes.iter().filter(|n| n.node_type == NodeType::Fact).count();

        // With 10k nodes, distributions should be close to target (within 5%)
        let entity_pct = entity_count as f64 / 10_000.0;
        let event_pct = event_count as f64 / 10_000.0;
        let fact_pct = fact_count as f64 / 10_000.0;

        assert!((entity_pct - 0.60).abs() < 0.05, "Entity should be ~60%, got {:.1}%", entity_pct * 100.0);
        assert!((event_pct - 0.25).abs() < 0.05, "Event should be ~25%, got {:.1}%", event_pct * 100.0);
        assert!((fact_pct - 0.15).abs() < 0.05, "Fact should be ~15%, got {:.1}%", fact_pct * 100.0);
    }

    #[test]
    fn test_generate_nodes_timestamp_range() {
        let nodes = generate_nodes(100, 42);
        let base_ts: u64 = 1_735_689_600;
        let one_year_secs: u64 = 365 * 24 * 3600;
        for node in &nodes {
            assert!(node.timestamp >= base_ts, "Timestamp below base");
            assert!(node.timestamp < base_ts + one_year_secs, "Timestamp above range");
        }
    }

    #[test]
    fn test_generate_edges_count() {
        let nodes = generate_nodes(100, 42);
        let edges = generate_edges(&nodes, 500, 99);
        assert_eq!(edges.len(), 500);
    }

    #[test]
    fn test_generate_edges_deterministic() {
        let nodes = generate_nodes(50, 42);
        let edges_a = generate_edges(&nodes, 200, 99);
        let edges_b = generate_edges(&nodes, 200, 99);
        for (a, b) in edges_a.iter().zip(edges_b.iter()) {
            assert_eq!(a.source, b.source);
            assert_eq!(a.target, b.target);
            assert_eq!(a.edge_type, b.edge_type);
            assert_eq!(a.weight, b.weight);
        }
    }

    #[test]
    fn test_generate_edges_valid_ids() {
        let nodes = generate_nodes(100, 42);
        let edges = generate_edges(&nodes, 1000, 99);
        let max_id = nodes.len() as u64;
        for edge in &edges {
            assert!(edge.source < max_id, "Source ID out of range: {}", edge.source);
            assert!(edge.target < max_id, "Target ID out of range: {}", edge.target);
            assert_ne!(edge.source, edge.target, "Self-loop detected");
        }
    }

    #[test]
    fn test_generate_edges_no_self_loops() {
        let nodes = generate_nodes(100, 42);
        let edges = generate_edges(&nodes, 5000, 99);
        for edge in &edges {
            assert_ne!(edge.source, edge.target, "Self-loop: {} -> {}", edge.source, edge.target);
        }
    }

    #[test]
    fn test_generate_edges_weight_range() {
        let nodes = generate_nodes(50, 42);
        let edges = generate_edges(&nodes, 500, 99);
        for edge in &edges {
            assert!(edge.weight >= 0.1, "Weight below 0.1: {}", edge.weight);
            assert!(edge.weight < 1.0, "Weight above 1.0: {}", edge.weight);
        }
    }

    #[test]
    fn test_generate_edges_power_law() {
        // With power-law, some nodes should have significantly more connections than others
        let nodes = generate_nodes(1000, 42);
        let edges = generate_edges(&nodes, 10_000, 99);

        let mut target_counts = vec![0u32; 1000];
        for edge in &edges {
            target_counts[edge.target as usize] += 1;
        }

        let max_count = *target_counts.iter().max().unwrap();
        let min_count = *target_counts.iter().min().unwrap();
        // Power-law should create significant variance
        assert!(
            max_count > min_count * 3,
            "Expected power-law distribution: max={} min={}",
            max_count,
            min_count
        );
    }

    #[test]
    fn test_save_load_roundtrip() {
        let nodes = generate_nodes(50, 42);
        let edges = generate_edges(&nodes, 100, 99);

        let tmp = std::env::temp_dir().join("ucotron_test_data.bin");
        save_to_bin(&tmp, &nodes, &edges).expect("save failed");
        let loaded = load_from_bin(&tmp).expect("load failed");

        assert_eq!(loaded.nodes.len(), nodes.len());
        assert_eq!(loaded.edges.len(), edges.len());

        // Verify first node fields match
        assert_eq!(loaded.nodes[0].id, nodes[0].id);
        assert_eq!(loaded.nodes[0].content, nodes[0].content);
        assert_eq!(loaded.nodes[0].embedding, nodes[0].embedding);
        assert_eq!(loaded.nodes[0].node_type, nodes[0].node_type);

        // Verify first edge fields match
        assert_eq!(loaded.edges[0].source, edges[0].source);
        assert_eq!(loaded.edges[0].target, edges[0].target);
        assert_eq!(loaded.edges[0].edge_type, edges[0].edge_type);

        // Cleanup
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    #[should_panic(expected = "Cannot generate edges for empty node set")]
    fn test_generate_edges_empty_nodes_panics() {
        let empty: Vec<Node> = vec![];
        generate_edges(&empty, 10, 42);
    }

    // --- Chain generator tests ---

    #[test]
    fn test_generate_chain_basic() {
        let (nodes, edges) = generate_chain(10, 42);
        assert_eq!(nodes.len(), 10);
        assert_eq!(edges.len(), 9);
        // Sequential IDs
        for (i, node) in nodes.iter().enumerate() {
            assert_eq!(node.id, i as u64);
        }
        // Each edge connects consecutive nodes
        for (i, edge) in edges.iter().enumerate() {
            assert_eq!(edge.source, i as u64);
            assert_eq!(edge.target, (i + 1) as u64);
            assert_eq!(edge.edge_type, EdgeType::RelatesTo);
            assert_eq!(edge.weight, 1.0);
        }
    }

    #[test]
    fn test_generate_chain_single_node() {
        let (nodes, edges) = generate_chain(1, 42);
        assert_eq!(nodes.len(), 1);
        assert_eq!(edges.len(), 0);
        assert_eq!(nodes[0].id, 0);
    }

    #[test]
    fn test_generate_chain_deterministic() {
        let (nodes_a, edges_a) = generate_chain(20, 42);
        let (nodes_b, edges_b) = generate_chain(20, 42);
        for (a, b) in nodes_a.iter().zip(nodes_b.iter()) {
            assert_eq!(a.id, b.id);
            assert_eq!(a.embedding, b.embedding);
        }
        for (a, b) in edges_a.iter().zip(edges_b.iter()) {
            assert_eq!(a.source, b.source);
            assert_eq!(a.target, b.target);
        }
    }

    #[test]
    fn test_generate_chain_embeddings_normalized() {
        let (nodes, _) = generate_chain(5, 42);
        for node in &nodes {
            assert_eq!(node.embedding.len(), 384);
            let norm: f32 = node.embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!((norm - 1.0).abs() < 1e-5, "Expected L2 norm ~1.0, got {}", norm);
        }
    }

    #[test]
    fn test_generate_chain_depth_100() {
        let (nodes, edges) = generate_chain(100, 42);
        assert_eq!(nodes.len(), 100);
        assert_eq!(edges.len(), 99);
        // Verify first→last path exists
        assert_eq!(edges[0].source, 0);
        assert_eq!(edges[98].target, 99);
    }

    #[test]
    #[should_panic(expected = "Chain depth must be at least 1")]
    fn test_generate_chain_zero_depth_panics() {
        generate_chain(0, 42);
    }

    // --- Tree generator tests ---

    #[test]
    fn test_generate_tree_binary_depth_3() {
        // branching=2, depth=3 → 1 + 2 + 4 = 7 nodes, 6 edges
        let (nodes, edges) = generate_tree(3, 2, 42);
        assert_eq!(nodes.len(), 7);
        assert_eq!(edges.len(), 6);
        // Root is node 0
        assert_eq!(nodes[0].id, 0);
        // All IDs are sequential
        for (i, node) in nodes.iter().enumerate() {
            assert_eq!(node.id, i as u64);
        }
    }

    #[test]
    fn test_generate_tree_branching_3_depth_4() {
        // branching=3, depth=4 → (3^4 - 1) / (3 - 1) = 80/2 = 40 nodes, 39 edges
        let (nodes, edges) = generate_tree(4, 3, 42);
        assert_eq!(nodes.len(), 40);
        assert_eq!(edges.len(), 39);
    }

    #[test]
    fn test_generate_tree_branching_1_is_chain() {
        // branching=1 degenerates to a chain
        let (nodes, edges) = generate_tree(5, 1, 42);
        assert_eq!(nodes.len(), 5);
        assert_eq!(edges.len(), 4);
        for (i, edge) in edges.iter().enumerate() {
            assert_eq!(edge.source, i as u64);
            assert_eq!(edge.target, (i + 1) as u64);
        }
    }

    #[test]
    fn test_generate_tree_single_node() {
        let (nodes, edges) = generate_tree(1, 3, 42);
        assert_eq!(nodes.len(), 1);
        assert_eq!(edges.len(), 0);
    }

    #[test]
    fn test_generate_tree_deterministic() {
        let (nodes_a, edges_a) = generate_tree(3, 2, 42);
        let (nodes_b, edges_b) = generate_tree(3, 2, 42);
        for (a, b) in nodes_a.iter().zip(nodes_b.iter()) {
            assert_eq!(a.id, b.id);
            assert_eq!(a.embedding, b.embedding);
        }
        for (a, b) in edges_a.iter().zip(edges_b.iter()) {
            assert_eq!(a.source, b.source);
            assert_eq!(a.target, b.target);
        }
    }

    #[test]
    fn test_generate_tree_branching_3_depth_10() {
        // PRD target: branching=3, depth=10 = (3^10 - 1)/2 = 29524 nodes
        let (nodes, edges) = generate_tree(10, 3, 42);
        assert_eq!(nodes.len(), 29524);
        assert_eq!(edges.len(), 29523);
    }

    #[test]
    fn test_generate_tree_edges_valid() {
        let (nodes, edges) = generate_tree(4, 3, 42);
        let max_id = nodes.len() as u64;
        for edge in &edges {
            assert!(edge.source < max_id);
            assert!(edge.target < max_id);
            assert_ne!(edge.source, edge.target, "Self-loop in tree");
            assert_eq!(edge.edge_type, EdgeType::RelatesTo);
            assert_eq!(edge.weight, 1.0);
        }
    }

    #[test]
    #[should_panic(expected = "Tree depth must be at least 1")]
    fn test_generate_tree_zero_depth_panics() {
        generate_tree(0, 2, 42);
    }

    #[test]
    #[should_panic(expected = "Branching factor must be at least 1")]
    fn test_generate_tree_zero_branching_panics() {
        generate_tree(3, 0, 42);
    }
}
