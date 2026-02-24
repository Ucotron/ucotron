//! Benchmark: HNSW rebuild-on-upsert performance at various scales.
//!
//! Measures the cost of rebuilding the entire instant-distance HNSW index
//! after each upsert batch, which is the current strategy in HnswVectorBackend.
//!
//! Run with: cargo bench -p ucotron-helix --bench hnsw_rebuild
//!
//! Results are used in ADR-001 (Incremental HNSW Insert Decision).

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use ucotron_config::HnswConfig;
use ucotron_core::backends::VectorBackend;
use ucotron_helix::HnswVectorBackend;
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use tempfile::TempDir;

/// Generate a random L2-normalized 384-dim embedding vector.
fn random_embedding(rng: &mut ChaCha8Rng) -> Vec<f32> {
    let raw: Vec<f32> = (0..384).map(|_| rng.gen::<f32>() - 0.5).collect();
    let norm: f32 = raw.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm < 1e-8 {
        vec![1.0 / (384.0_f32).sqrt(); 384]
    } else {
        raw.iter().map(|x| x / norm).collect()
    }
}

/// Benchmark: Full rebuild cost with N existing vectors + 100-vector batch upsert.
///
/// Workflow:
/// 1. Pre-populate index with `base_size` vectors
/// 2. Measure time to upsert 100 new vectors (which triggers full rebuild of base_size + 100)
fn bench_rebuild_upsert(c: &mut Criterion) {
    let mut group = c.benchmark_group("hnsw_rebuild_upsert");
    group.sample_size(10); // Reduced samples for larger sizes

    for base_size in [1_000u64, 10_000, 50_000, 100_000] {
        let dir = TempDir::new().unwrap();
        let config = HnswConfig {
            ef_construction: 200,
            ef_search: 200,
            enabled: true,
        };

        let backend = HnswVectorBackend::open(
            dir.path().to_str().unwrap(),
            2 * 1024 * 1024 * 1024,
            config,
        )
        .unwrap();

        // Pre-populate with base_size vectors (single batch to minimize setup time)
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let base_items: Vec<(u64, Vec<f32>)> = (0..base_size)
            .map(|id| (id, random_embedding(&mut rng)))
            .collect();
        backend.upsert_embeddings(&base_items).unwrap();

        // Prepare the batch to insert during benchmarking
        let new_batch: Vec<(u64, Vec<f32>)> = (base_size..base_size + 100)
            .map(|id| (id, random_embedding(&mut rng)))
            .collect();

        group.bench_with_input(
            BenchmarkId::new("batch_100_on", base_size),
            &base_size,
            |b, _| {
                b.iter(|| {
                    backend.upsert_embeddings(&new_batch).unwrap();
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_rebuild_upsert);
criterion_main!(benches);
