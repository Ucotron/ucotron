//! # Ucotron Bench Runner
//!
//! CLI binary for benchmarking Ucotron storage engines.
//!
//! Provides subcommands for different benchmark scenarios:
//! - `ingest`: Bulk data ingestion benchmarks
//! - `search`: Vector, graph, and hybrid search latency benchmarks
//! - `recursion`: Deep graph traversal and path-finding benchmarks
//!
//! # CozoDB Support
//!
//! CozoDB benchmarks are available behind the `cozo` feature flag.
//! Enable with: `cargo run --features cozo --bin bench_runner`

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use ucotron_core::data_gen;
use ucotron_core::{Config, Edge, Node, NodeId, StorageEngine};
#[cfg(feature = "cozo")]
use ucotron_cozo::CozoEngine;
use ucotron_helix::HelixEngine;
use rand::SeedableRng;
use rand::Rng;
use rand_chacha::ChaCha8Rng;
use std::path::{Path, PathBuf};
use std::time::Instant;

/// Ucotron Storage Engine Benchmark Runner
#[derive(Parser)]
#[command(name = "bench_runner")]
#[command(about = "Benchmark runner for Ucotron storage engines")]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Benchmark bulk data ingestion on both engines
    Ingest {
        /// Number of nodes to generate and ingest
        #[arg(long, default_value_t = 100_000)]
        count: usize,

        /// Number of edges to generate and ingest
        #[arg(long, default_value_t = 500_000)]
        edges: usize,

        /// Only benchmark HelixDB
        #[arg(long, default_value_t = false)]
        helix_only: bool,

        /// Only benchmark CozoDB (requires --features cozo)
        #[arg(long, default_value_t = false)]
        cozo_only: bool,

        /// Path to pre-generated .bin data file (skip generation if provided)
        #[arg(long)]
        data_path: Option<PathBuf>,

        /// RNG seed for data generation (default: 42)
        #[arg(long, default_value_t = 42)]
        seed: u64,
    },

    /// Benchmark vector, graph, and hybrid search latency
    Search {
        /// Number of search queries to run
        #[arg(long, default_value_t = 1000)]
        queries: usize,

        /// Number of top results for vector search
        #[arg(long, default_value_t = 10)]
        top_k: usize,

        /// Number of graph hops for traversal
        #[arg(long, default_value_t = 2)]
        hops: u8,

        /// Number of nodes to ingest before searching
        #[arg(long, default_value_t = 10_000)]
        count: usize,

        /// Number of edges to ingest before searching
        #[arg(long, default_value_t = 50_000)]
        edges: usize,

        /// Only benchmark HelixDB
        #[arg(long, default_value_t = false)]
        helix_only: bool,

        /// Only benchmark CozoDB (requires --features cozo)
        #[arg(long, default_value_t = false)]
        cozo_only: bool,

        /// RNG seed for data generation (default: 42)
        #[arg(long, default_value_t = 42)]
        seed: u64,
    },

    /// Benchmark deep graph traversal and path-finding
    Recursion {
        /// Comma-separated list of depths to test
        #[arg(long, default_value = "10,20,50,100", value_delimiter = ',')]
        depths: Vec<usize>,

        /// Number of iterations per depth
        #[arg(long, default_value_t = 100)]
        iterations: usize,

        /// Only benchmark HelixDB
        #[arg(long, default_value_t = false)]
        helix_only: bool,

        /// Only benchmark CozoDB (requires --features cozo)
        #[arg(long, default_value_t = false)]
        cozo_only: bool,

        /// RNG seed for data generation (default: 42)
        #[arg(long, default_value_t = 42)]
        seed: u64,

        /// Include tree benchmark
        #[arg(long, default_value_t = true)]
        include_tree: bool,

        /// Tree branching factor (default: 3)
        #[arg(long, default_value_t = 3)]
        tree_branching: usize,

        /// Tree depth for tree benchmark (default: 10)
        #[arg(long, default_value_t = 10)]
        tree_depth: usize,
    },

    /// Estimate monthly cloud deployment costs for AWS, GCP, and Azure
    Cost {
        /// Number of memory nodes to store
        #[arg(long, default_value_t = 100_000)]
        nodes: u64,

        /// Number of search queries per day
        #[arg(long, default_value_t = 10_000)]
        queries_per_day: u64,

        /// Number of tenants (namespaces)
        #[arg(long, default_value_t = 1)]
        namespaces: u64,

        /// Enable multimodal processing (dual HNSW index)
        #[arg(long, default_value_t = false)]
        multimodal: bool,

        /// Enable multi-instance mode (writer + readers)
        #[arg(long, default_value_t = false)]
        multi_instance: bool,

        /// Number of reader replicas (only used with --multi-instance)
        #[arg(long, default_value_t = 2)]
        readers: u64,

        /// Output format: markdown (default) or json
        #[arg(long, default_value = "markdown")]
        format: String,
    },
}

// ---------------------------------------------------------------------------
// macOS RAM measurement via mach_task_info
// ---------------------------------------------------------------------------

#[cfg(target_os = "macos")]
fn get_resident_memory_bytes() -> Option<u64> {
    use std::mem;

    #[repr(C)]
    #[allow(non_camel_case_types)]
    struct mach_task_basic_info {
        virtual_size: u64,
        resident_size: u64,
        resident_size_max: u64,
        user_time: [u32; 2],
        system_time: [u32; 2],
        policy: i32,
        suspend_count: i32,
    }

    const MACH_TASK_BASIC_INFO: u32 = 20;

    unsafe {
        let mut info: mach_task_basic_info = mem::zeroed();
        let mut count = (mem::size_of::<mach_task_basic_info>() / mem::size_of::<u32>()) as u32;

        extern "C" {
            fn mach_task_self() -> u32;
            fn task_info(
                target_task: u32,
                flavor: u32,
                task_info_out: *mut mach_task_basic_info,
                task_info_count: *mut u32,
            ) -> i32;
        }

        let ret = task_info(
            mach_task_self(),
            MACH_TASK_BASIC_INFO,
            &mut info as *mut _,
            &mut count as *mut _,
        );

        if ret == 0 {
            Some(info.resident_size)
        } else {
            None
        }
    }
}

#[cfg(target_os = "linux")]
fn get_resident_memory_bytes() -> Option<u64> {
    use std::fs;
    let status = fs::read_to_string("/proc/self/status").ok()?;
    for line in status.lines() {
        if line.starts_with("VmRSS:") {
            let kb: u64 = line
                .split_whitespace()
                .nth(1)?
                .parse()
                .ok()?;
            return Some(kb * 1024);
        }
    }
    None
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
fn get_resident_memory_bytes() -> Option<u64> {
    None
}

// ---------------------------------------------------------------------------
// Disk size measurement
// ---------------------------------------------------------------------------

/// Recursively compute the total size of all files in a directory.
fn dir_size_bytes(path: &Path) -> u64 {
    if !path.exists() {
        return 0;
    }
    let mut total: u64 = 0;
    if path.is_file() {
        return path.metadata().map(|m| m.len()).unwrap_or(0);
    }
    if let Ok(entries) = std::fs::read_dir(path) {
        for entry in entries.flatten() {
            let p = entry.path();
            if p.is_dir() {
                total += dir_size_bytes(&p);
            } else {
                total += p.metadata().map(|m| m.len()).unwrap_or(0);
            }
        }
    }
    total
}

// ---------------------------------------------------------------------------
// Helper formatting
// ---------------------------------------------------------------------------

fn format_bytes(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;
    if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.2} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} B", bytes)
    }
}

fn format_duration_ms(us: u64) -> String {
    if us >= 1_000_000 {
        format!("{:.2}s", us as f64 / 1_000_000.0)
    } else {
        format!("{:.2}ms", us as f64 / 1_000.0)
    }
}

/// Check if CozoDB support is compiled in. Prints a warning if --cozo-only is
/// used without the feature flag.
fn check_cozo_available(cozo_only: bool) -> Result<()> {
    if cozo_only && !cfg!(feature = "cozo") {
        anyhow::bail!(
            "CozoDB benchmarks require the 'cozo' feature flag. \
             Rebuild with: cargo run --features cozo --bin bench_runner"
        );
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Ingestion metrics for one engine
// ---------------------------------------------------------------------------

struct IngestMetrics {
    engine_name: String,
    cold_start_us: u64,
    node_ingest_us: u64,
    edge_ingest_us: u64,
    total_ingest_us: u64,
    node_count: usize,
    edge_count: usize,
    ram_bytes: Option<u64>,
    disk_bytes: u64,
}

impl IngestMetrics {
    fn throughput_docs_per_sec(&self) -> f64 {
        let total_items = (self.node_count + self.edge_count) as f64;
        let secs = self.total_ingest_us as f64 / 1_000_000.0;
        if secs > 0.0 {
            total_items / secs
        } else {
            0.0
        }
    }

    fn node_throughput(&self) -> f64 {
        let secs = self.node_ingest_us as f64 / 1_000_000.0;
        if secs > 0.0 {
            self.node_count as f64 / secs
        } else {
            0.0
        }
    }

    fn edge_throughput(&self) -> f64 {
        let secs = self.edge_ingest_us as f64 / 1_000_000.0;
        if secs > 0.0 {
            self.edge_count as f64 / secs
        } else {
            0.0
        }
    }
}

// ---------------------------------------------------------------------------
// Engine benchmarking
// ---------------------------------------------------------------------------

fn benchmark_engine<E: StorageEngine>(
    engine_name: &str,
    data_dir: &Path,
    nodes: &[Node],
    edges: &[Edge],
) -> Result<IngestMetrics> {
    // Measure RAM before engine init
    let ram_before = get_resident_memory_bytes().unwrap_or(0);

    // Cold start: time from init call to ready
    let cold_start = Instant::now();
    let config = Config {
        data_dir: data_dir.to_string_lossy().to_string(),
        max_db_size: 10 * 1024 * 1024 * 1024, // 10GB
        batch_size: 10_000,
    };
    let mut engine = E::init(&config).with_context(|| format!("{}: init failed", engine_name))?;
    let cold_start_us = cold_start.elapsed().as_micros() as u64;

    // Insert nodes
    let node_start = Instant::now();
    let node_stats = engine
        .insert_nodes(nodes)
        .with_context(|| format!("{}: insert_nodes failed", engine_name))?;
    let node_ingest_us = node_start.elapsed().as_micros() as u64;

    // Insert edges
    let edge_start = Instant::now();
    let edge_stats = engine
        .insert_edges(edges)
        .with_context(|| format!("{}: insert_edges failed", engine_name))?;
    let edge_ingest_us = edge_start.elapsed().as_micros() as u64;

    let total_ingest_us = node_ingest_us + edge_ingest_us;

    // Measure RAM after ingestion
    let ram_after = get_resident_memory_bytes().unwrap_or(0);
    let ram_bytes = if ram_after > ram_before {
        Some(ram_after - ram_before)
    } else {
        get_resident_memory_bytes()
    };

    // Measure disk
    let disk_bytes = dir_size_bytes(data_dir);

    // Shutdown
    engine
        .shutdown()
        .with_context(|| format!("{}: shutdown failed", engine_name))?;

    println!(
        "  {} complete: {} nodes, {} edges in {}",
        engine_name,
        node_stats.count,
        edge_stats.count,
        format_duration_ms(total_ingest_us)
    );

    Ok(IngestMetrics {
        engine_name: engine_name.to_string(),
        cold_start_us,
        node_ingest_us,
        edge_ingest_us,
        total_ingest_us,
        node_count: node_stats.count,
        edge_count: edge_stats.count,
        ram_bytes,
        disk_bytes,
    })
}

// ---------------------------------------------------------------------------
// Markdown table output
// ---------------------------------------------------------------------------

fn print_comparison_table(results: &[IngestMetrics]) {
    println!();
    println!("## Ingestion Benchmark Results");
    println!();

    // Build header
    let mut header = "| Metric                  ".to_string();
    let mut separator = "|-------------------------".to_string();
    for r in results {
        header.push_str(&format!("| {:>16} ", r.engine_name));
        separator.push_str("|------------------");
    }
    header.push('|');
    separator.push('|');
    println!("{}", header);
    println!("{}", separator);

    // Cold Start
    print!("| Cold Start              ");
    for r in results {
        print!("| {:>16} ", format_duration_ms(r.cold_start_us));
    }
    println!("|");

    // Node Ingestion
    print!("| Node Ingestion          ");
    for r in results {
        print!("| {:>16} ", format_duration_ms(r.node_ingest_us));
    }
    println!("|");

    // Edge Ingestion
    print!("| Edge Ingestion          ");
    for r in results {
        print!("| {:>16} ", format_duration_ms(r.edge_ingest_us));
    }
    println!("|");

    // Total Ingestion
    print!("| Total Ingestion         ");
    for r in results {
        print!("| {:>16} ", format_duration_ms(r.total_ingest_us));
    }
    println!("|");

    // Throughput (combined)
    print!("| Throughput (total)      ");
    for r in results {
        print!("| {:>13.0} d/s ", r.throughput_docs_per_sec());
    }
    println!("|");

    // Node throughput
    print!("| Throughput (nodes)      ");
    for r in results {
        print!("| {:>13.0} n/s ", r.node_throughput());
    }
    println!("|");

    // Edge throughput
    print!("| Throughput (edges)      ");
    for r in results {
        print!("| {:>13.0} e/s ", r.edge_throughput());
    }
    println!("|");

    // RAM
    print!("| Peak RAM (delta)        ");
    for r in results {
        let ram_str = match r.ram_bytes {
            Some(b) => format_bytes(b),
            None => "N/A".to_string(),
        };
        print!("| {:>16} ", ram_str);
    }
    println!("|");

    // Disk
    print!("| Disk Size               ");
    for r in results {
        print!("| {:>16} ", format_bytes(r.disk_bytes));
    }
    println!("|");

    println!();
    println!("---");
    println!();
}

// ---------------------------------------------------------------------------
// Ingest subcommand
// ---------------------------------------------------------------------------

#[allow(unused_variables)]
fn run_ingest(
    count: usize,
    edge_count: usize,
    helix_only: bool,
    cozo_only: bool,
    data_path: Option<PathBuf>,
    seed: u64,
) -> Result<()> {
    check_cozo_available(cozo_only)?;

    println!("=== Ucotron Ingestion Benchmark ===");
    println!();

    // Step 1: Generate or load data
    let (nodes, edges) = if let Some(ref path) = data_path {
        println!("Loading pre-generated data from {:?}...", path);
        let t = Instant::now();
        let dataset = data_gen::load_from_bin(path)
            .with_context(|| format!("Failed to load data from {:?}", path))?;
        println!(
            "  Loaded {} nodes, {} edges in {:.2}ms",
            dataset.nodes.len(),
            dataset.edges.len(),
            t.elapsed().as_millis()
        );
        (dataset.nodes, dataset.edges)
    } else {
        println!(
            "Generating {} nodes and {} edges (seed={})...",
            count, edge_count, seed
        );
        let t = Instant::now();
        let nodes = data_gen::generate_nodes(count, seed);
        let edges = data_gen::generate_edges(&nodes, edge_count, seed);
        println!(
            "  Generated {} nodes, {} edges in {:.2}ms",
            nodes.len(),
            edges.len(),
            t.elapsed().as_millis()
        );
        (nodes, edges)
    };

    println!();

    // Step 2: Create temp dirs for each engine (unique per run to avoid parallel test collisions)
    let unique_id = format!(
        "ucotron_bench_ingest_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos()
    );
    let base_dir = std::env::temp_dir().join(unique_id);
    let _ = std::fs::remove_dir_all(&base_dir);
    std::fs::create_dir_all(&base_dir)?;

    let mut results: Vec<IngestMetrics> = Vec::new();

    // Step 3: Benchmark HelixDB
    if !cozo_only {
        let helix_dir = base_dir.join("helix");
        std::fs::create_dir_all(&helix_dir)?;
        println!("Benchmarking HelixDB...");
        let metrics = benchmark_engine::<HelixEngine>("HelixDB", &helix_dir, &nodes, &edges)?;
        results.push(metrics);
    }

    // Step 4: Benchmark CozoDB (feature-gated)
    #[cfg(feature = "cozo")]
    if !helix_only {
        let cozo_dir = base_dir.join("cozo");
        std::fs::create_dir_all(&cozo_dir)?;
        println!("Benchmarking CozoDB...");
        let metrics = benchmark_engine::<CozoEngine>("CozoDB", &cozo_dir, &nodes, &edges)?;
        results.push(metrics);
    }

    // Step 5: Print comparison table
    print_comparison_table(&results);

    // Step 6: Clean up
    let _ = std::fs::remove_dir_all(&base_dir);

    Ok(())
}

// ---------------------------------------------------------------------------
// Percentile calculation
// ---------------------------------------------------------------------------

/// Calculate a percentile value from a sorted slice of latencies (in microseconds).
/// `p` is a value between 0 and 100. The slice must be sorted ascending.
fn percentile(sorted: &[u64], p: f64) -> u64 {
    if sorted.is_empty() {
        return 0;
    }
    if sorted.len() == 1 {
        return sorted[0];
    }
    let idx = (p / 100.0) * (sorted.len() - 1) as f64;
    let lower = idx.floor() as usize;
    let upper = idx.ceil() as usize;
    if lower == upper {
        sorted[lower]
    } else {
        let frac = idx - lower as f64;
        ((1.0 - frac) * sorted[lower] as f64 + frac * sorted[upper] as f64).round() as u64
    }
}

// ---------------------------------------------------------------------------
// Search benchmark metrics
// ---------------------------------------------------------------------------

/// Latency measurements for one benchmark category on one engine.
struct SearchLatencies {
    /// All latency measurements in microseconds, sorted ascending.
    sorted_us: Vec<u64>,
}

impl SearchLatencies {
    fn from_raw(mut raw: Vec<u64>) -> Self {
        raw.sort_unstable();
        Self { sorted_us: raw }
    }

    fn p50(&self) -> u64 {
        percentile(&self.sorted_us, 50.0)
    }

    fn p95(&self) -> u64 {
        percentile(&self.sorted_us, 95.0)
    }

    fn p99(&self) -> u64 {
        percentile(&self.sorted_us, 99.0)
    }
}

/// All latency categories for one engine.
struct SearchMetrics {
    engine_name: String,
    vector: SearchLatencies,
    graph_1hop: SearchLatencies,
    graph_2hop: SearchLatencies,
    hybrid: SearchLatencies,
}

// ---------------------------------------------------------------------------
// Generate random query vectors
// ---------------------------------------------------------------------------

/// Generate `count` random L2-normalized 384-dim query vectors.
fn generate_query_vectors(count: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed.wrapping_add(9999));
    let dim = 384;
    let mut queries = Vec::with_capacity(count);
    for _ in 0..count {
        let mut v: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0f32..1.0f32)).collect();
        // L2 normalize
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in &mut v {
                *x /= norm;
            }
        }
        queries.push(v);
    }
    queries
}

// ---------------------------------------------------------------------------
// Benchmark search queries on a single engine
// ---------------------------------------------------------------------------

fn benchmark_search<E: StorageEngine>(
    engine: &E,
    engine_name: &str,
    queries: &[Vec<f32>],
    top_k: usize,
    hops: u8,
) -> Result<SearchMetrics> {
    let num_queries = queries.len();

    // Vector-only search
    println!("  {}: Running {} vector searches...", engine_name, num_queries);
    let mut vector_latencies = Vec::with_capacity(num_queries);
    for q in queries {
        let start = Instant::now();
        let _ = engine.vector_search(q, top_k)?;
        vector_latencies.push(start.elapsed().as_micros() as u64);
    }

    // Graph 1-hop: pick a node from vector results and do 1-hop traversal
    println!("  {}: Running {} graph 1-hop traversals...", engine_name, num_queries);
    let mut graph_1hop_latencies = Vec::with_capacity(num_queries);
    for q in queries {
        // Find a seed node first (not timed)
        let seeds = engine.vector_search(q, 1)?;
        let seed_id = seeds.first().map(|(id, _)| *id).unwrap_or(0);
        let start = Instant::now();
        let _ = engine.get_neighbors(seed_id, 1)?;
        graph_1hop_latencies.push(start.elapsed().as_micros() as u64);
    }

    // Graph 2-hop
    println!("  {}: Running {} graph 2-hop traversals...", engine_name, num_queries);
    let mut graph_2hop_latencies = Vec::with_capacity(num_queries);
    for q in queries {
        let seeds = engine.vector_search(q, 1)?;
        let seed_id = seeds.first().map(|(id, _)| *id).unwrap_or(0);
        let start = Instant::now();
        let _ = engine.get_neighbors(seed_id, 2)?;
        graph_2hop_latencies.push(start.elapsed().as_micros() as u64);
    }

    // Hybrid search (vector + graph)
    println!("  {}: Running {} hybrid searches (top_k={}, hops={})...", engine_name, num_queries, top_k, hops);
    let mut hybrid_latencies = Vec::with_capacity(num_queries);
    for q in queries {
        let start = Instant::now();
        let _ = engine.hybrid_search(q, top_k, hops)?;
        hybrid_latencies.push(start.elapsed().as_micros() as u64);
    }

    Ok(SearchMetrics {
        engine_name: engine_name.to_string(),
        vector: SearchLatencies::from_raw(vector_latencies),
        graph_1hop: SearchLatencies::from_raw(graph_1hop_latencies),
        graph_2hop: SearchLatencies::from_raw(graph_2hop_latencies),
        hybrid: SearchLatencies::from_raw(hybrid_latencies),
    })
}

// ---------------------------------------------------------------------------
// Search comparison table
// ---------------------------------------------------------------------------

fn print_search_table(results: &[SearchMetrics]) {
    println!();
    println!("## Search Benchmark Results");
    println!();

    // Build header
    let mut header = "| Metric                  ".to_string();
    let mut separator = "|-------------------------".to_string();
    for r in results {
        header.push_str(&format!("| {:>16} ", r.engine_name));
        separator.push_str("|------------------");
    }
    header.push('|');
    separator.push('|');
    println!("{}", header);
    println!("{}", separator);

    // Helper to print a row
    let print_row = |label: &str, f: &dyn Fn(&SearchMetrics) -> u64| {
        print!("| {:<24}", label);
        for r in results {
            print!("| {:>16} ", format_duration_ms(f(r)));
        }
        println!("|");
    };

    print_row("Vector P50", &|m| m.vector.p50());
    print_row("Vector P95", &|m| m.vector.p95());
    print_row("Vector P99", &|m| m.vector.p99());
    print_row("Graph 1-hop P50", &|m| m.graph_1hop.p50());
    print_row("Graph 1-hop P95", &|m| m.graph_1hop.p95());
    print_row("Graph 1-hop P99", &|m| m.graph_1hop.p99());
    print_row("Graph 2-hop P50", &|m| m.graph_2hop.p50());
    print_row("Graph 2-hop P95", &|m| m.graph_2hop.p95());
    print_row("Graph 2-hop P99", &|m| m.graph_2hop.p99());
    print_row("Hybrid P50", &|m| m.hybrid.p50());
    print_row("Hybrid P95", &|m| m.hybrid.p95());
    print_row("Hybrid P99", &|m| m.hybrid.p99());

    println!();
    println!("---");
    println!();
}

// ---------------------------------------------------------------------------
// Search subcommand
// ---------------------------------------------------------------------------

#[allow(unused_variables, clippy::too_many_arguments)]
fn run_search(
    query_count: usize,
    top_k: usize,
    hops: u8,
    node_count: usize,
    edge_count: usize,
    helix_only: bool,
    cozo_only: bool,
    seed: u64,
) -> Result<()> {
    check_cozo_available(cozo_only)?;

    println!("=== Ucotron Search Benchmark ===");
    println!();

    // Step 1: Generate data
    println!(
        "Generating {} nodes and {} edges (seed={})...",
        node_count, edge_count, seed
    );
    let t = Instant::now();
    let nodes = data_gen::generate_nodes(node_count, seed);
    let edges = data_gen::generate_edges(&nodes, edge_count, seed);
    println!(
        "  Generated {} nodes, {} edges in {:.2}ms",
        nodes.len(),
        edges.len(),
        t.elapsed().as_millis()
    );

    // Step 2: Generate query vectors
    println!("Generating {} query vectors...", query_count);
    let queries = generate_query_vectors(query_count, seed);
    println!();

    // Step 3: Setup temp dirs (use PID + timestamp for uniqueness in parallel tests)
    let unique_id = format!(
        "ucotron_bench_search_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos()
    );
    let base_dir = std::env::temp_dir().join(unique_id);
    let _ = std::fs::remove_dir_all(&base_dir);
    std::fs::create_dir_all(&base_dir)?;

    let mut results: Vec<SearchMetrics> = Vec::new();

    // Step 4: Benchmark HelixDB
    if !cozo_only {
        let helix_dir = base_dir.join("helix");
        std::fs::create_dir_all(&helix_dir)?;
        println!("Setting up HelixDB...");
        let config = Config {
            data_dir: helix_dir.to_string_lossy().to_string(),
            max_db_size: 10 * 1024 * 1024 * 1024,
            batch_size: 10_000,
        };
        let mut engine = HelixEngine::init(&config).context("HelixDB init failed")?;
        engine.insert_nodes(&nodes).context("HelixDB insert_nodes failed")?;
        engine.insert_edges(&edges).context("HelixDB insert_edges failed")?;
        println!("  HelixDB ready with {} nodes, {} edges", nodes.len(), edges.len());

        let metrics = benchmark_search(&engine, "HelixDB", &queries, top_k, hops)?;
        results.push(metrics);

        engine.shutdown().context("HelixDB shutdown failed")?;
    }

    // Step 5: Benchmark CozoDB (feature-gated)
    #[cfg(feature = "cozo")]
    if !helix_only {
        let cozo_dir = base_dir.join("cozo");
        std::fs::create_dir_all(&cozo_dir)?;
        println!("Setting up CozoDB...");
        let config = Config {
            data_dir: cozo_dir.to_string_lossy().to_string(),
            max_db_size: 10 * 1024 * 1024 * 1024,
            batch_size: 10_000,
        };
        let mut engine = CozoEngine::init(&config).context("CozoDB init failed")?;
        engine.insert_nodes(&nodes).context("CozoDB insert_nodes failed")?;
        engine.insert_edges(&edges).context("CozoDB insert_edges failed")?;
        println!("  CozoDB ready with {} nodes, {} edges", nodes.len(), edges.len());

        let metrics = benchmark_search(&engine, "CozoDB", &queries, top_k, hops)?;
        results.push(metrics);

        engine.shutdown().context("CozoDB shutdown failed")?;
    }

    // Step 6: Print comparison table
    print_search_table(&results);

    // Step 7: Clean up
    let _ = std::fs::remove_dir_all(&base_dir);

    Ok(())
}

// ---------------------------------------------------------------------------
// Recursion benchmark metrics
// ---------------------------------------------------------------------------

/// Latency and memory measurements for one depth on one engine.
#[allow(dead_code)]
struct RecursionMetrics {
    engine_name: String,
    depth: usize,
    /// Number of nodes in the graph (depth for chains, calculated for trees).
    node_count: usize,
    /// Median (P50) latency for find_path in microseconds.
    p50_us: u64,
    /// P95 latency in microseconds.
    p95_us: u64,
    /// P99 latency in microseconds.
    p99_us: u64,
    /// Mean latency in microseconds.
    mean_us: u64,
    /// Peak RAM during traversal (delta from baseline), if measurable.
    ram_bytes: Option<u64>,
}

/// Benchmark find_path on one engine for a given chain/tree graph.
#[allow(clippy::too_many_arguments)]
fn benchmark_recursion_engine<E: StorageEngine>(
    engine_name: &str,
    data_dir: &Path,
    nodes: &[Node],
    edges: &[Edge],
    source: NodeId,
    target: NodeId,
    max_depth: u32,
    depth_label: usize,
    iterations: usize,
) -> Result<RecursionMetrics> {
    let config = Config {
        data_dir: data_dir.to_string_lossy().to_string(),
        max_db_size: 10 * 1024 * 1024 * 1024,
        batch_size: 10_000,
    };
    let mut engine = E::init(&config)
        .with_context(|| format!("{}: init failed", engine_name))?;
    engine.insert_nodes(nodes)
        .with_context(|| format!("{}: insert_nodes failed", engine_name))?;
    engine.insert_edges(edges)
        .with_context(|| format!("{}: insert_edges failed", engine_name))?;

    // Warm up: run one traversal to prime caches
    let _ = engine.find_path(source, target, max_depth)?;

    // Measure RAM baseline
    let ram_before = get_resident_memory_bytes().unwrap_or(0);

    // Run iterations
    let mut latencies = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let start = Instant::now();
        let path = engine.find_path(source, target, max_depth)?;
        let elapsed_us = start.elapsed().as_micros() as u64;
        latencies.push(elapsed_us);

        // Verify correctness: path should exist and connect source to target
        if let Some(ref p) = path {
            debug_assert_eq!(*p.first().unwrap(), source);
            debug_assert_eq!(*p.last().unwrap(), target);
        }
    }

    // Measure RAM after traversals
    let ram_after = get_resident_memory_bytes().unwrap_or(0);
    let ram_bytes = if ram_after > ram_before {
        Some(ram_after - ram_before)
    } else {
        get_resident_memory_bytes()
    };

    // Calculate stats
    latencies.sort_unstable();
    let mean_us = if latencies.is_empty() {
        0
    } else {
        (latencies.iter().sum::<u64>()) / latencies.len() as u64
    };

    engine.shutdown()
        .with_context(|| format!("{}: shutdown failed", engine_name))?;

    Ok(RecursionMetrics {
        engine_name: engine_name.to_string(),
        depth: depth_label,
        node_count: nodes.len(),
        p50_us: percentile(&latencies, 50.0),
        p95_us: percentile(&latencies, 95.0),
        p99_us: percentile(&latencies, 99.0),
        mean_us,
        ram_bytes,
    })
}

// ---------------------------------------------------------------------------
// Recursion comparison table
// ---------------------------------------------------------------------------

fn print_recursion_table(title: &str, helix_results: &[RecursionMetrics], cozo_results: &[RecursionMetrics]) {
    println!();
    println!("## {}", title);
    println!();

    // Determine which engines are present
    let has_helix = !helix_results.is_empty();
    let has_cozo = !cozo_results.is_empty();

    if has_helix && has_cozo {
        println!("| Depth | Nodes | HelixDB P50 | HelixDB P95 | HelixDB P99 | HelixDB Mean | HelixDB RAM | CozoDB P50 | CozoDB P95 | CozoDB P99 | CozoDB Mean | CozoDB RAM |");
        println!("|-------|-------|-------------|-------------|-------------|--------------|-------------|------------|------------|------------|-------------|------------|");
        for (h, c) in helix_results.iter().zip(cozo_results.iter()) {
            let h_ram = h.ram_bytes.map_or("N/A".to_string(), format_bytes);
            let c_ram = c.ram_bytes.map_or("N/A".to_string(), format_bytes);
            println!(
                "| {:>5} | {:>5} | {:>11} | {:>11} | {:>11} | {:>12} | {:>11} | {:>10} | {:>10} | {:>10} | {:>11} | {:>10} |",
                h.depth, h.node_count,
                format_duration_ms(h.p50_us), format_duration_ms(h.p95_us), format_duration_ms(h.p99_us),
                format_duration_ms(h.mean_us), h_ram,
                format_duration_ms(c.p50_us), format_duration_ms(c.p95_us), format_duration_ms(c.p99_us),
                format_duration_ms(c.mean_us), c_ram,
            );
        }
    } else {
        let results = if has_helix { helix_results } else { cozo_results };
        let name = if has_helix { "HelixDB" } else { "CozoDB" };
        println!("| Depth | Nodes | {} P50 | {} P95 | {} P99 | {} Mean | {} RAM |", name, name, name, name, name);
        println!("|-------|-------|-------------|-------------|-------------|--------------|-------------|");
        for r in results {
            let ram = r.ram_bytes.map_or("N/A".to_string(), format_bytes);
            println!(
                "| {:>5} | {:>5} | {:>11} | {:>11} | {:>11} | {:>12} | {:>11} |",
                r.depth, r.node_count,
                format_duration_ms(r.p50_us), format_duration_ms(r.p95_us), format_duration_ms(r.p99_us),
                format_duration_ms(r.mean_us), ram,
            );
        }
    }

    println!();
    println!("---");
    println!();
}

// ---------------------------------------------------------------------------
// Recursion subcommand
// ---------------------------------------------------------------------------

#[allow(unused_variables, clippy::too_many_arguments)]
fn run_recursion(
    depths: &[usize],
    iterations: usize,
    helix_only: bool,
    cozo_only: bool,
    seed: u64,
    include_tree: bool,
    tree_branching: usize,
    tree_depth: usize,
) -> Result<()> {
    check_cozo_available(cozo_only)?;

    println!("=== Ucotron Recursion Benchmark ===");
    println!();
    println!("Depths: {:?}", depths);
    println!("Iterations per depth: {}", iterations);
    println!();

    // Unique temp dir for this run
    let unique_id = format!(
        "ucotron_bench_recursion_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos()
    );
    let base_dir = std::env::temp_dir().join(unique_id);
    let _ = std::fs::remove_dir_all(&base_dir);
    std::fs::create_dir_all(&base_dir)?;

    // --- Chain benchmarks ---
    let mut helix_chain_results: Vec<RecursionMetrics> = Vec::new();
    #[allow(unused_mut)]
    let mut cozo_chain_results: Vec<RecursionMetrics> = Vec::new();

    for &depth in depths {
        println!("Generating chain graph (depth={})...", depth);
        let (nodes, edges) = data_gen::generate_chain(depth, seed);
        let source = nodes.first().unwrap().id;
        let target = nodes.last().unwrap().id;
        let max_depth = depth as u32 + 10; // allow some slack beyond exact chain length
        println!("  Chain: {} nodes, {} edges (source={}, target={})", nodes.len(), edges.len(), source, target);

        // HelixDB
        if !cozo_only {
            let helix_dir = base_dir.join(format!("helix_chain_{}", depth));
            std::fs::create_dir_all(&helix_dir)?;
            println!("  Benchmarking HelixDB (chain depth={})...", depth);
            let metrics = benchmark_recursion_engine::<HelixEngine>(
                "HelixDB", &helix_dir, &nodes, &edges,
                source, target, max_depth, depth, iterations,
            )?;
            println!("    P50={}, P95={}, P99={}, Mean={}",
                format_duration_ms(metrics.p50_us),
                format_duration_ms(metrics.p95_us),
                format_duration_ms(metrics.p99_us),
                format_duration_ms(metrics.mean_us));
            helix_chain_results.push(metrics);
        }

        // CozoDB (feature-gated)
        #[cfg(feature = "cozo")]
        if !helix_only {
            let cozo_dir = base_dir.join(format!("cozo_chain_{}", depth));
            std::fs::create_dir_all(&cozo_dir)?;
            println!("  Benchmarking CozoDB (chain depth={})...", depth);
            let metrics = benchmark_recursion_engine::<CozoEngine>(
                "CozoDB", &cozo_dir, &nodes, &edges,
                source, target, max_depth, depth, iterations,
            )?;
            println!("    P50={}, P95={}, P99={}, Mean={}",
                format_duration_ms(metrics.p50_us),
                format_duration_ms(metrics.p95_us),
                format_duration_ms(metrics.p99_us),
                format_duration_ms(metrics.mean_us));
            cozo_chain_results.push(metrics);
        }
    }

    print_recursion_table("Chain Traversal (find_path start→end)", &helix_chain_results, &cozo_chain_results);

    // --- Tree benchmark ---
    if include_tree {
        println!("Generating tree graph (branching={}, depth={})...", tree_branching, tree_depth);
        let (nodes, edges) = data_gen::generate_tree(tree_depth, tree_branching, seed);
        // Source = root (first node), target = a deep leaf (last node)
        let source = nodes.first().unwrap().id;
        let target = nodes.last().unwrap().id;
        let max_depth = tree_depth as u32 + 10;
        println!("  Tree: {} nodes, {} edges (source={}, target={})", nodes.len(), edges.len(), source, target);

        let mut helix_tree_results: Vec<RecursionMetrics> = Vec::new();
        #[allow(unused_mut)]
        let mut cozo_tree_results: Vec<RecursionMetrics> = Vec::new();

        if !cozo_only {
            let helix_dir = base_dir.join("helix_tree");
            std::fs::create_dir_all(&helix_dir)?;
            println!("  Benchmarking HelixDB (tree branching={}, depth={})...", tree_branching, tree_depth);
            let metrics = benchmark_recursion_engine::<HelixEngine>(
                "HelixDB", &helix_dir, &nodes, &edges,
                source, target, max_depth, tree_depth, iterations,
            )?;
            println!("    P50={}, P95={}, P99={}, Mean={} ({} nodes)",
                format_duration_ms(metrics.p50_us),
                format_duration_ms(metrics.p95_us),
                format_duration_ms(metrics.p99_us),
                format_duration_ms(metrics.mean_us),
                metrics.node_count);
            helix_tree_results.push(metrics);
        }

        #[cfg(feature = "cozo")]
        if !helix_only {
            let cozo_dir = base_dir.join("cozo_tree");
            std::fs::create_dir_all(&cozo_dir)?;
            println!("  Benchmarking CozoDB (tree branching={}, depth={})...", tree_branching, tree_depth);
            let metrics = benchmark_recursion_engine::<CozoEngine>(
                "CozoDB", &cozo_dir, &nodes, &edges,
                source, target, max_depth, tree_depth, iterations,
            )?;
            println!("    P50={}, P95={}, P99={}, Mean={} ({} nodes)",
                format_duration_ms(metrics.p50_us),
                format_duration_ms(metrics.p95_us),
                format_duration_ms(metrics.p99_us),
                format_duration_ms(metrics.mean_us),
                metrics.node_count);
            cozo_tree_results.push(metrics);
        }

        print_recursion_table(
            &format!("Tree Traversal (branching={}, depth={}, {} nodes)", tree_branching, tree_depth, nodes.len()),
            &helix_tree_results,
            &cozo_tree_results,
        );
    }

    // Clean up
    let _ = std::fs::remove_dir_all(&base_dir);

    println!("Recursion benchmark complete.");
    Ok(())
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Ingest {
            count,
            edges,
            helix_only,
            cozo_only,
            data_path,
            seed,
        } => {
            run_ingest(count, edges, helix_only, cozo_only, data_path, seed)?;
        }
        Commands::Search {
            queries,
            top_k,
            hops,
            count,
            edges,
            helix_only,
            cozo_only,
            seed,
        } => {
            run_search(queries, top_k, hops, count, edges, helix_only, cozo_only, seed)?;
        }
        Commands::Recursion {
            depths,
            iterations,
            helix_only,
            cozo_only,
            seed,
            include_tree,
            tree_branching,
            tree_depth,
        } => {
            run_recursion(&depths, iterations, helix_only, cozo_only, seed, include_tree, tree_branching, tree_depth)?;
        }
        Commands::Cost {
            nodes,
            queries_per_day,
            namespaces,
            multimodal,
            multi_instance,
            readers,
            format,
        } => {
            run_cost_estimation(nodes, queries_per_day, namespaces, multimodal, multi_instance, readers, &format)?;
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// US-29.6: Cloud deployment cost estimation
// ---------------------------------------------------------------------------

/// Compute sizing: estimate RAM, disk, and vCPU needs from workload parameters.
fn estimate_resource_requirements(
    nodes: u64,
    queries_per_day: u64,
    multimodal: bool,
) -> (f64, f64, u32) {
    // From benchmarks: 100k nodes + 500k edges = 320.58 MB RAM, 426.16 MB disk
    let ram_per_100k_nodes_mb: f64 = 320.0;
    let disk_per_100k_nodes_mb: f64 = 430.0;

    let node_factor = nodes as f64 / 100_000.0;

    // RAM: base + nodes + ONNX models (~200MB for MiniLM+GLiNER) + HNSW index overhead
    let model_ram_mb: f64 = if multimodal { 600.0 } else { 200.0 };
    // 512MB for OS/runtime/tokio/buffers overhead
    let ram_mb = ram_per_100k_nodes_mb * node_factor + model_ram_mb + 512.0;

    // Disk: nodes + edges (5x ratio) + HNSW index
    let disk_mb = disk_per_100k_nodes_mb * node_factor * if multimodal { 1.5 } else { 1.0 };

    // vCPU: 1 core handles ~5000 queries/s (from P95 18ms benchmark)
    // Each query takes ~20ms → 50 qps per core → 4.3M queries/day per core
    let qps_per_core: f64 = 50.0;
    let required_qps = queries_per_day as f64 / 86400.0;
    let cpu_cores = (required_qps / qps_per_core).ceil().max(2.0) as u32; // min 2 cores

    (ram_mb, disk_mb, cpu_cores)
}

/// Select the smallest instance type that fits the resource requirements.
fn select_instance(provider: &str, ram_mb: f64, cpu_cores: u32) -> (&'static str, f64, u32, f64) {
    // Returns (instance_type, monthly_cost_usd, vcpus, ram_gb)
    let ram_gb = ram_mb / 1024.0;

    match provider {
        "aws" => {
            // us-east-1 pricing (on-demand, Linux)
            let instances = [
                ("t3.medium",    2, 4.0,   30.37),
                ("t3.large",     2, 8.0,   60.74),
                ("t3.xlarge",    4, 16.0, 121.47),
                ("t3.2xlarge",   8, 32.0, 242.94),
                ("m6i.xlarge",   4, 16.0, 138.70),
                ("m6i.2xlarge",  8, 32.0, 277.40),
                ("m6i.4xlarge", 16, 64.0, 554.80),
                ("r6i.xlarge",   4, 32.0, 181.54),
                ("r6i.2xlarge",  8, 64.0, 363.07),
            ];
            for &(name, vcpus, ram, cost) in &instances {
                if vcpus >= cpu_cores && ram >= ram_gb {
                    return (name, cost, vcpus, ram);
                }
            }
            ("r6i.2xlarge", 363.07, 8, 64.0)
        }
        "gcp" => {
            // us-central1 pricing (on-demand)
            let instances = [
                ("e2-medium",      2,  4.0,   24.27),
                ("e2-standard-2",  2,  8.0,   48.55),
                ("e2-standard-4",  4, 16.0,   97.09),
                ("e2-standard-8",  8, 32.0,  194.18),
                ("n2-standard-4",  4, 16.0,  116.58),
                ("n2-standard-8",  8, 32.0,  233.16),
                ("n2-highmem-4",   4, 32.0,  156.92),
                ("n2-highmem-8",   8, 64.0,  313.84),
            ];
            for &(name, vcpus, ram, cost) in &instances {
                if vcpus >= cpu_cores && ram >= ram_gb {
                    return (name, cost, vcpus, ram);
                }
            }
            ("n2-highmem-8", 313.84, 8, 64.0)
        }
        "azure" => {
            // East US pricing (pay-as-you-go)
            let instances = [
                ("Standard_B2s",    2,  4.0,   30.37),
                ("Standard_D2s_v5", 2,  8.0,   70.08),
                ("Standard_D4s_v5", 4, 16.0,  140.16),
                ("Standard_D8s_v5", 8, 32.0,  280.32),
                ("Standard_E4s_v5", 4, 32.0,  182.50),
                ("Standard_E8s_v5", 8, 64.0,  365.00),
            ];
            for &(name, vcpus, ram, cost) in &instances {
                if vcpus >= cpu_cores && ram >= ram_gb {
                    return (name, cost, vcpus, ram);
                }
            }
            ("Standard_E8s_v5", 365.00, 8, 64.0)
        }
        _ => ("unknown", 0.0, 0, 0.0),
    }
}

/// Provider cost breakdown for a deployment.
struct ProviderCost {
    provider: &'static str,
    region: &'static str,
    instance_type: &'static str,
    instance_vcpus: u32,
    instance_ram_gb: f64,
    instance_count: u64,
    compute_monthly: f64,
    control_plane_monthly: f64,
    storage_monthly: f64,
    network_monthly: f64,
    total_monthly: f64,
    disk_size_gb: u64,
    notes: Vec<String>,
}

fn estimate_provider_cost(
    provider: &str,
    nodes: u64,
    queries_per_day: u64,
    multimodal: bool,
    multi_instance: bool,
    readers: u64,
) -> ProviderCost {
    let (ram_mb, disk_mb, cpu_cores) = estimate_resource_requirements(nodes, queries_per_day, multimodal);
    let (instance_type, instance_cost, vcpus, ram_gb) = select_instance(provider, ram_mb, cpu_cores);

    let instance_count = if multi_instance { 1 + readers } else { 1 };
    let compute_monthly = instance_cost * instance_count as f64;

    let disk_gb = ((disk_mb / 1024.0).ceil() as u64).max(20);

    // Storage pricing per GB/month
    let storage_price_per_gb = match provider {
        "aws" => 0.08,     // gp3
        "gcp" => 0.17,     // pd-ssd
        "azure" => 0.132,  // Premium SSD P6 effective
        _ => 0.10,
    };
    let storage_monthly = disk_gb as f64 * storage_price_per_gb * instance_count as f64;

    // Control plane
    let control_plane_monthly = match provider {
        "aws" => 73.00,   // EKS
        "gcp" => 73.00,   // GKE Standard
        "azure" => 0.0,   // AKS free tier
        _ => 0.0,
    };

    // Network: NAT + load balancer + estimated egress
    // ~1KB per query response → 1 GB per 1M queries
    let egress_gb_per_month = (queries_per_day as f64 * 30.0 / 1_000_000.0).ceil();
    let network_monthly = match provider {
        "aws" => 32.0 + 16.0 + egress_gb_per_month * 0.09,  // NAT + NLB + egress
        "gcp" => 32.0 + egress_gb_per_month * 0.12,           // Cloud NAT + egress
        "azure" => 18.0 + egress_gb_per_month * 0.087,        // LB + egress
        _ => 0.0,
    };

    let total_monthly = compute_monthly + control_plane_monthly + storage_monthly + network_monthly;

    let mut notes = Vec::new();
    if multi_instance {
        notes.push(format!("1 writer + {} reader replicas", readers));
    }
    if multimodal {
        notes.push("Dual HNSW index (384-dim text + 512-dim visual)".to_string());
    }
    if nodes > 1_000_000 {
        notes.push("Consider memory-optimized instances for >1M nodes".to_string());
    }

    let (provider_name, region) = match provider {
        "aws" => ("AWS (EKS)", "us-east-1"),
        "gcp" => ("GCP (GKE)", "us-central1"),
        "azure" => ("Azure (AKS)", "eastus"),
        _ => ("Unknown", "unknown"),
    };

    ProviderCost {
        provider: provider_name,
        region,
        instance_type,
        instance_vcpus: vcpus,
        instance_ram_gb: ram_gb,
        instance_count,
        compute_monthly,
        control_plane_monthly,
        storage_monthly,
        network_monthly,
        total_monthly,
        disk_size_gb: disk_gb,
        notes,
    }
}

fn format_cost_markdown(
    nodes: u64,
    queries_per_day: u64,
    namespaces: u64,
    multimodal: bool,
    multi_instance: bool,
    readers: u64,
    costs: &[ProviderCost],
) -> String {
    let (ram_mb, disk_mb, cpu_cores) = estimate_resource_requirements(nodes, queries_per_day, multimodal);

    let mut out = String::new();
    out.push_str("# Ucotron Cloud Cost Estimation\n\n");
    out.push_str(&format!("**Generated:** {}\n\n", chrono_lite_date()));

    out.push_str("## Workload Parameters\n\n");
    out.push_str("| Parameter | Value |\n");
    out.push_str("|-----------|-------|\n");
    out.push_str(&format!("| Memory nodes | {} |\n", format_number(nodes)));
    out.push_str(&format!("| Queries/day | {} |\n", format_number(queries_per_day)));
    out.push_str(&format!("| Namespaces | {} |\n", namespaces));
    out.push_str(&format!("| Multimodal | {} |\n", if multimodal { "Yes" } else { "No" }));
    out.push_str(&format!("| Multi-instance | {} |\n", if multi_instance { format!("Yes (1W + {}R)", readers) } else { "No (single)".to_string() }));
    out.push_str("\n");

    out.push_str("## Estimated Resource Requirements\n\n");
    out.push_str("| Resource | Estimate | Basis |\n");
    out.push_str("|----------|----------|-------|\n");
    out.push_str(&format!("| RAM | {:.1} GB | ~320 MB per 100k nodes + {} MB models |\n",
        ram_mb / 1024.0,
        if multimodal { 600 } else { 200 }));
    out.push_str(&format!("| Disk | {:.1} GB | ~430 MB per 100k nodes{} |\n",
        disk_mb / 1024.0,
        if multimodal { " × 1.5 (dual index)" } else { "" }));
    out.push_str(&format!("| vCPUs | {} | ~50 qps/core at P95 <20ms |\n", cpu_cores));
    out.push_str("\n");

    out.push_str("## Cost Comparison\n\n");
    out.push_str("| Component | AWS (EKS) | GCP (GKE) | Azure (AKS) |\n");
    out.push_str("|-----------|-----------|-----------|-------------|\n");
    out.push_str(&format!("| Region | {} | {} | {} |\n",
        costs[0].region, costs[1].region, costs[2].region));
    out.push_str(&format!("| Instance | {} | {} | {} |\n",
        costs[0].instance_type, costs[1].instance_type, costs[2].instance_type));
    out.push_str(&format!("| vCPUs × RAM | {}×{:.0}GB | {}×{:.0}GB | {}×{:.0}GB |\n",
        costs[0].instance_vcpus, costs[0].instance_ram_gb,
        costs[1].instance_vcpus, costs[1].instance_ram_gb,
        costs[2].instance_vcpus, costs[2].instance_ram_gb));
    out.push_str(&format!("| Instances | {} | {} | {} |\n",
        costs[0].instance_count, costs[1].instance_count, costs[2].instance_count));
    out.push_str(&format!("| Disk | {} GB | {} GB | {} GB |\n",
        costs[0].disk_size_gb, costs[1].disk_size_gb, costs[2].disk_size_gb));
    out.push_str(&format!("| **Compute** | **${:.0}** | **${:.0}** | **${:.0}** |\n",
        costs[0].compute_monthly, costs[1].compute_monthly, costs[2].compute_monthly));
    out.push_str(&format!("| Control plane | ${:.0} | ${:.0} | ${:.0} |\n",
        costs[0].control_plane_monthly, costs[1].control_plane_monthly, costs[2].control_plane_monthly));
    out.push_str(&format!("| Storage | ${:.0} | ${:.0} | ${:.0} |\n",
        costs[0].storage_monthly, costs[1].storage_monthly, costs[2].storage_monthly));
    out.push_str(&format!("| Network | ${:.0} | ${:.0} | ${:.0} |\n",
        costs[0].network_monthly, costs[1].network_monthly, costs[2].network_monthly));
    out.push_str(&format!("| **Total/month** | **${:.0}** | **${:.0}** | **${:.0}** |\n",
        costs[0].total_monthly, costs[1].total_monthly, costs[2].total_monthly));
    out.push_str("\n");

    // Notes
    if costs.iter().any(|c| !c.notes.is_empty()) {
        out.push_str("## Notes\n\n");
        for cost in costs {
            if !cost.notes.is_empty() {
                out.push_str(&format!("**{}:**\n", cost.provider));
                for note in &cost.notes {
                    out.push_str(&format!("- {}\n", note));
                }
                out.push_str("\n");
            }
        }
    }

    // Assumptions
    out.push_str("## Assumptions\n\n");
    out.push_str("- Pricing is on-demand (no reserved instances or spot/preemptible)\n");
    out.push_str("- Kubernetes control plane included (EKS $73, GKE $73, AKS free tier)\n");
    out.push_str("- Storage uses SSD (gp3, pd-ssd, Premium SSD)\n");
    out.push_str("- Network includes NAT gateway and load balancer base costs\n");
    out.push_str("- Egress estimated at ~1KB per query response\n");
    out.push_str("- No LLM API costs (Ucotron uses local ONNX models)\n");
    out.push_str("- RAM sizing based on Phase 1 benchmarks: 320 MB per 100k nodes\n");
    out.push_str("- Query throughput based on P95 hybrid search: ~50 qps per vCPU core\n");
    out.push_str("\n");

    // Savings tips
    out.push_str("## Cost Optimization Tips\n\n");
    out.push_str("| Strategy | Savings | Notes |\n");
    out.push_str("|----------|---------|-------|\n");
    out.push_str("| Reserved instances (1yr) | 30-40% | Commit to steady-state workload |\n");
    out.push_str("| Spot/preemptible readers | 60-80% | Reader replicas tolerate interruption |\n");
    out.push_str("| GKE Autopilot | varies | Pay-per-pod, good for bursty traffic |\n");
    out.push_str("| ARM instances (Graviton/Ampere) | 20-30% | Ucotron compiles for ARM |\n");
    out.push_str("| Smaller disk + compression | 10-20% | LMDB data is compressible |\n");

    out
}

fn format_cost_json(
    nodes: u64,
    queries_per_day: u64,
    namespaces: u64,
    multimodal: bool,
    multi_instance: bool,
    readers: u64,
    costs: &[ProviderCost],
) -> String {
    let (ram_mb, disk_mb, cpu_cores) = estimate_resource_requirements(nodes, queries_per_day, multimodal);

    let mut providers = String::from("[");
    for (i, cost) in costs.iter().enumerate() {
        if i > 0 { providers.push(','); }
        providers.push_str(&format!(
            r#"{{"provider":"{}","region":"{}","instance_type":"{}","instance_vcpus":{},"instance_ram_gb":{},"instance_count":{},"disk_size_gb":{},"compute_monthly":{:.2},"control_plane_monthly":{:.2},"storage_monthly":{:.2},"network_monthly":{:.2},"total_monthly":{:.2}}}"#,
            cost.provider, cost.region, cost.instance_type,
            cost.instance_vcpus, cost.instance_ram_gb,
            cost.instance_count, cost.disk_size_gb,
            cost.compute_monthly, cost.control_plane_monthly,
            cost.storage_monthly, cost.network_monthly,
            cost.total_monthly
        ));
    }
    providers.push(']');

    format!(
        r#"{{"workload":{{"nodes":{},"queries_per_day":{},"namespaces":{},"multimodal":{},"multi_instance":{},"readers":{}}},"resources":{{"ram_mb":{:.0},"disk_mb":{:.0},"cpu_cores":{}}},"providers":{}}}"#,
        nodes, queries_per_day, namespaces, multimodal, multi_instance, readers,
        ram_mb, disk_mb, cpu_cores,
        providers
    )
}

/// Simple date string (no chrono dependency)
fn chrono_lite_date() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    // Approximate date from epoch seconds
    let days = secs / 86400;
    let years = (days as f64 / 365.25) as u64;
    let year = 1970 + years;
    let remaining_days = days - (years as f64 * 365.25) as u64;
    let month_days: [u64; 12] = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
    let mut month = 1u64;
    let mut d = remaining_days;
    for &md in &month_days {
        if d < md { break; }
        d -= md;
        month += 1;
    }
    format!("{}-{:02}-{:02}", year, month.min(12), (d + 1).min(31))
}

/// Format large numbers with commas
fn format_number(n: u64) -> String {
    let s = n.to_string();
    let mut result = String::new();
    for (i, c) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 { result.push(','); }
        result.push(c);
    }
    result.chars().rev().collect()
}

fn run_cost_estimation(
    nodes: u64,
    queries_per_day: u64,
    namespaces: u64,
    multimodal: bool,
    multi_instance: bool,
    readers: u64,
    format: &str,
) -> Result<()> {
    let costs: Vec<ProviderCost> = ["aws", "gcp", "azure"]
        .iter()
        .map(|p| estimate_provider_cost(p, nodes, queries_per_day, multimodal, multi_instance, readers))
        .collect();

    let output = match format {
        "json" => format_cost_json(nodes, queries_per_day, namespaces, multimodal, multi_instance, readers, &costs),
        _ => format_cost_markdown(nodes, queries_per_day, namespaces, multimodal, multi_instance, readers, &costs),
    };

    println!("{}", output);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(500), "500 B");
        assert_eq!(format_bytes(1024), "1.00 KB");
        assert_eq!(format_bytes(1024 * 1024), "1.00 MB");
        assert_eq!(format_bytes(2 * 1024 * 1024 * 1024), "2.00 GB");
    }

    #[test]
    fn test_format_duration_ms() {
        assert_eq!(format_duration_ms(500), "0.50ms");
        assert_eq!(format_duration_ms(5_000), "5.00ms");
        assert_eq!(format_duration_ms(1_500_000), "1.50s");
    }

    #[test]
    fn test_dir_size_bytes_empty() {
        let dir = std::env::temp_dir().join("ucotron_test_dir_size");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        assert_eq!(dir_size_bytes(&dir), 0);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_dir_size_bytes_with_file() {
        let dir = std::env::temp_dir().join("ucotron_test_dir_size_file");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let file_path = dir.join("test.bin");
        std::fs::write(&file_path, vec![0u8; 1024]).unwrap();
        assert_eq!(dir_size_bytes(&dir), 1024);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_get_resident_memory() {
        // On macOS/Linux this should return Some(>0), on other platforms None
        let mem = get_resident_memory_bytes();
        if cfg!(any(target_os = "macos", target_os = "linux")) {
            assert!(mem.is_some());
            assert!(mem.unwrap() > 0);
        }
    }

    #[test]
    fn test_ingest_metrics_throughput() {
        let m = IngestMetrics {
            engine_name: "test".to_string(),
            cold_start_us: 1_000,
            node_ingest_us: 1_000_000, // 1 second
            edge_ingest_us: 1_000_000, // 1 second
            total_ingest_us: 2_000_000,
            node_count: 10_000,
            edge_count: 50_000,
            ram_bytes: Some(100 * 1024 * 1024),
            disk_bytes: 50 * 1024 * 1024,
        };
        // 60k items / 2 seconds = 30k/s
        assert!((m.throughput_docs_per_sec() - 30_000.0).abs() < 1.0);
        // 10k nodes / 1 second = 10k/s
        assert!((m.node_throughput() - 10_000.0).abs() < 1.0);
        // 50k edges / 1 second = 50k/s
        assert!((m.edge_throughput() - 50_000.0).abs() < 1.0);
    }

    #[test]
    fn test_ingest_small_helix_only() {
        // Integration test: run a small ingest benchmark with HelixDB only
        let result = run_ingest(100, 200, true, false, None, 42);
        assert!(result.is_ok(), "Small Helix ingest failed: {:?}", result.err());
    }

    #[cfg(feature = "cozo")]
    #[test]
    fn test_ingest_small_cozo_only() {
        // Integration test: run a small ingest benchmark with CozoDB only
        let result = run_ingest(100, 200, false, true, None, 42);
        assert!(result.is_ok(), "Small Cozo ingest failed: {:?}", result.err());
    }

    // -----------------------------------------------------------------------
    // US-2.4: Search benchmark tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_percentile_single_element() {
        assert_eq!(percentile(&[100], 50.0), 100);
        assert_eq!(percentile(&[100], 99.0), 100);
    }

    #[test]
    fn test_percentile_empty() {
        assert_eq!(percentile(&[], 50.0), 0);
    }

    #[test]
    fn test_percentile_basic() {
        // 100 values: 1, 2, 3, ..., 100 (indices 0..99)
        let data: Vec<u64> = (1..=100).collect();
        // P50: index 49.5 → interpolate data[49]=50, data[50]=51 → 51
        assert_eq!(percentile(&data, 50.0), 51);
        // P95: index 94.05 → interpolate data[94]=95, data[95]=96 → 95
        assert_eq!(percentile(&data, 95.0), 95);
        // P99: index 98.01 → interpolate data[98]=99, data[99]=100 → 99
        assert_eq!(percentile(&data, 99.0), 99);
        assert_eq!(percentile(&data, 0.0), 1);
        assert_eq!(percentile(&data, 100.0), 100);
    }

    #[test]
    fn test_percentile_interpolation() {
        // 10 values: 10, 20, 30, ..., 100
        let data: Vec<u64> = (1..=10).map(|x| x * 10).collect();
        // P50 = index 4.5 → interpolate between 50 and 60 → 55
        assert_eq!(percentile(&data, 50.0), 55);
    }

    #[test]
    fn test_search_latencies_percentiles() {
        let raw: Vec<u64> = (1..=1000).collect();
        let lat = SearchLatencies::from_raw(raw);
        // P50 of 1..=1000 at index 499.5 → ~500
        assert!(lat.p50() >= 499 && lat.p50() <= 501);
        // P95 at index 949.05 → ~950
        assert!(lat.p95() >= 949 && lat.p95() <= 951);
        // P99 at index 989.01 → ~990
        assert!(lat.p99() >= 989 && lat.p99() <= 991);
    }

    #[test]
    fn test_generate_query_vectors() {
        let queries = generate_query_vectors(10, 42);
        assert_eq!(queries.len(), 10);
        for q in &queries {
            assert_eq!(q.len(), 384);
            // Check L2 normalization: magnitude should be ~1.0
            let mag: f32 = q.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!((mag - 1.0).abs() < 1e-5, "Query not normalized: mag={}", mag);
        }
    }

    #[test]
    fn test_generate_query_vectors_deterministic() {
        let q1 = generate_query_vectors(5, 42);
        let q2 = generate_query_vectors(5, 42);
        for (a, b) in q1.iter().zip(q2.iter()) {
            assert_eq!(a, b, "Query vectors should be deterministic for same seed");
        }
    }

    #[test]
    fn test_search_small_helix_only() {
        // Integration: run a small search benchmark on HelixDB
        let result = run_search(10, 5, 1, 100, 200, true, false, 42);
        assert!(result.is_ok(), "Small Helix search failed: {:?}", result.err());
    }

    #[cfg(feature = "cozo")]
    #[test]
    fn test_search_small_cozo_only() {
        // Integration: run a small search benchmark on CozoDB
        let result = run_search(10, 5, 1, 100, 200, false, true, 42);
        assert!(result.is_ok(), "Small Cozo search failed: {:?}", result.err());
    }

    // -----------------------------------------------------------------------
    // US-4.3: Recursion benchmark tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_recursion_small_helix_chain() {
        // Integration: run recursion benchmark on small chains with HelixDB only
        let result = run_recursion(&[5, 10], 3, true, false, 42, false, 3, 10);
        assert!(result.is_ok(), "Small Helix chain recursion failed: {:?}", result.err());
    }

    #[cfg(feature = "cozo")]
    #[test]
    fn test_recursion_small_cozo_chain() {
        // Integration: run recursion benchmark on small chains with CozoDB only
        let result = run_recursion(&[5, 10], 3, false, true, 42, false, 3, 10);
        assert!(result.is_ok(), "Small Cozo chain recursion failed: {:?}", result.err());
    }

    #[test]
    fn test_recursion_with_tree_helix() {
        // Integration: run recursion benchmark including tree test with HelixDB
        // Uses branching=2, depth=5 (31 nodes) for fast test execution
        let result = run_recursion(&[5], 2, true, false, 42, true, 2, 5);
        assert!(result.is_ok(), "Helix tree recursion failed: {:?}", result.err());
    }

    #[cfg(feature = "cozo")]
    #[test]
    fn test_recursion_with_tree_cozo() {
        // Integration: run recursion benchmark including tree test with CozoDB
        // Uses branching=2, depth=5 (31 nodes) for fast test execution
        let result = run_recursion(&[5], 2, false, true, 42, true, 2, 5);
        assert!(result.is_ok(), "Cozo tree recursion failed: {:?}", result.err());
    }

    #[test]
    fn test_recursion_metrics_correctness() {
        // Verify that benchmark_recursion_engine produces valid metrics
        let (nodes, edges) = data_gen::generate_chain(20, 42);
        let source = nodes.first().unwrap().id;
        let target = nodes.last().unwrap().id;

        let dir = std::env::temp_dir().join(format!(
            "ucotron_test_recursion_metrics_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        let metrics = benchmark_recursion_engine::<HelixEngine>(
            "HelixDB", &dir, &nodes, &edges,
            source, target, 30, 20, 5,
        ).unwrap();

        assert_eq!(metrics.engine_name, "HelixDB");
        assert_eq!(metrics.depth, 20);
        assert_eq!(metrics.node_count, 20);
        assert!(metrics.p50_us > 0, "P50 should be > 0");
        assert!(metrics.p95_us >= metrics.p50_us, "P95 should be >= P50");
        assert!(metrics.p99_us >= metrics.p95_us, "P99 should be >= P95");
        assert!(metrics.mean_us > 0, "Mean should be > 0");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_check_cozo_available() {
        // When cozo feature is not enabled, cozo_only should fail
        if !cfg!(feature = "cozo") {
            let result = check_cozo_available(true);
            assert!(result.is_err());
        }
        // helix_only (cozo_only=false) should always pass
        let result = check_cozo_available(false);
        assert!(result.is_ok());
    }

    // -----------------------------------------------------------------------
    // US-29.6: Cloud cost estimation tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_estimate_resources_small_workload() {
        let (ram_mb, disk_mb, cpu_cores) = estimate_resource_requirements(100_000, 10_000, false);
        // 100k nodes: ~320 MB data + 200 MB models + 512 MB base = ~1032 MB
        assert!(ram_mb > 900.0 && ram_mb < 1200.0, "RAM estimate: {} MB", ram_mb);
        // 100k nodes: ~430 MB disk
        assert!(disk_mb > 300.0 && disk_mb < 600.0, "Disk estimate: {} MB", disk_mb);
        // 10k queries/day = 0.12 qps → min 2 cores
        assert_eq!(cpu_cores, 2, "CPU cores should be minimum 2");
    }

    #[test]
    fn test_estimate_resources_large_workload() {
        let (ram_mb, disk_mb, cpu_cores) = estimate_resource_requirements(1_000_000, 1_000_000, true);
        // 1M nodes multimodal: much more RAM
        assert!(ram_mb > 3000.0, "Large workload RAM: {} MB", ram_mb);
        // 1M nodes multimodal: ~6.45 GB disk
        assert!(disk_mb > 5000.0, "Large workload disk: {} MB", disk_mb);
        // 1M queries/day = 11.6 qps → still 2 min cores
        assert!(cpu_cores >= 2, "CPU cores: {}", cpu_cores);
    }

    #[test]
    fn test_estimate_resources_multimodal_increases_disk() {
        let (_, disk_no_mm, _) = estimate_resource_requirements(100_000, 10_000, false);
        let (_, disk_mm, _) = estimate_resource_requirements(100_000, 10_000, true);
        assert!(disk_mm > disk_no_mm, "Multimodal should increase disk: {} vs {}", disk_mm, disk_no_mm);
        // Multimodal adds 1.5x factor
        let ratio = disk_mm / disk_no_mm;
        assert!((ratio - 1.5).abs() < 0.01, "Multimodal disk ratio: {}", ratio);
    }

    #[test]
    fn test_select_instance_aws() {
        let (name, cost, vcpus, ram) = select_instance("aws", 4096.0, 2);
        assert!(!name.is_empty());
        assert!(cost > 0.0);
        assert!(vcpus >= 2);
        assert!(ram >= 4.0);
    }

    #[test]
    fn test_select_instance_gcp() {
        let (name, cost, vcpus, ram) = select_instance("gcp", 2048.0, 2);
        assert!(!name.is_empty());
        assert!(cost > 0.0);
        assert!(vcpus >= 2);
        assert!(ram >= 2.0);
    }

    #[test]
    fn test_select_instance_azure() {
        let (name, cost, vcpus, ram) = select_instance("azure", 8192.0, 4);
        assert!(!name.is_empty());
        assert!(cost > 0.0);
        assert!(vcpus >= 4);
        assert!(ram >= 8.0);
    }

    #[test]
    fn test_provider_cost_all_components_positive() {
        for provider in &["aws", "gcp", "azure"] {
            let cost = estimate_provider_cost(provider, 100_000, 10_000, false, false, 0);
            assert!(cost.compute_monthly > 0.0, "{} compute", provider);
            assert!(cost.storage_monthly > 0.0, "{} storage", provider);
            assert!(cost.total_monthly > 0.0, "{} total", provider);
            assert_eq!(cost.instance_count, 1, "{} single instance", provider);
        }
    }

    #[test]
    fn test_multi_instance_increases_cost() {
        let single = estimate_provider_cost("aws", 100_000, 10_000, false, false, 0);
        let multi = estimate_provider_cost("aws", 100_000, 10_000, false, true, 2);
        assert_eq!(multi.instance_count, 3); // 1 writer + 2 readers
        assert!(multi.compute_monthly > single.compute_monthly, "Multi should cost more");
    }

    #[test]
    fn test_format_number() {
        assert_eq!(format_number(100), "100");
        assert_eq!(format_number(1_000), "1,000");
        assert_eq!(format_number(1_000_000), "1,000,000");
        assert_eq!(format_number(10_000_000), "10,000,000");
    }

    #[test]
    fn test_cost_markdown_output_contains_providers() {
        let costs: Vec<ProviderCost> = ["aws", "gcp", "azure"]
            .iter()
            .map(|p| estimate_provider_cost(p, 100_000, 10_000, false, false, 0))
            .collect();
        let md = format_cost_markdown(100_000, 10_000, 1, false, false, 0, &costs);
        assert!(md.contains("AWS (EKS)"), "Should contain AWS");
        assert!(md.contains("GCP (GKE)"), "Should contain GCP");
        assert!(md.contains("Azure (AKS)"), "Should contain Azure");
        assert!(md.contains("Total/month"), "Should contain total");
        assert!(md.contains("100,000"), "Should format node count");
    }

    #[test]
    fn test_cost_json_output_valid() {
        let costs: Vec<ProviderCost> = ["aws", "gcp", "azure"]
            .iter()
            .map(|p| estimate_provider_cost(p, 100_000, 10_000, false, false, 0))
            .collect();
        let json = format_cost_json(100_000, 10_000, 1, false, false, 0, &costs);
        // Basic JSON structure check
        assert!(json.starts_with('{'), "Should be JSON object");
        assert!(json.ends_with('}'), "Should end with brace");
        assert!(json.contains("\"workload\""), "Should contain workload");
        assert!(json.contains("\"providers\""), "Should contain providers");
        assert!(json.contains("\"total_monthly\""), "Should contain total_monthly");
    }

    #[test]
    fn test_run_cost_estimation_markdown() {
        let result = run_cost_estimation(100_000, 10_000, 1, false, false, 0, "markdown");
        assert!(result.is_ok());
    }

    #[test]
    fn test_run_cost_estimation_json() {
        let result = run_cost_estimation(100_000, 10_000, 1, false, false, 0, "json");
        assert!(result.is_ok());
    }

    #[test]
    fn test_azure_free_control_plane() {
        let cost = estimate_provider_cost("azure", 100_000, 10_000, false, false, 0);
        assert_eq!(cost.control_plane_monthly, 0.0, "Azure AKS free tier has no control plane cost");
    }
}
