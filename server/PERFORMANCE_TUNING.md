# Performance Tuning Guide

This guide covers tuning Ucotron for different workloads: high-throughput ingestion, low-latency search, and resource-constrained environments.

---

## Quick Reference

| Workload | Key Parameters | Recommended Values |
|----------|---------------|-------------------|
| High ingestion throughput | `batch_size`, `enable_entity_resolution`, `enable_contradiction_detection` | 50,000 / false / false |
| Low-latency search | `ef_search`, `vector_top_k`, `graph_expansion_hops` | 100 / 20 / 1 |
| High-recall search | `ef_search`, `vector_top_k`, `graph_expansion_hops` | 400 / 100 / 2 |
| Low memory (<256MB) | `max_db_size`, `batch_size`, `embedding_batch_size` | 2GB / 5000 / 16 |
| Large dataset (>500k nodes) | `max_db_size`, HNSW `enabled`, `trigger_interval` | 50GB / true / 500 |

---

## 1. HNSW Vector Index Tuning

The HNSW index (via instant-distance) controls the speed/recall trade-off for vector search.

### Configuration

```toml
[storage.vector.hnsw]
enabled = true         # false = brute-force SIMD (adequate for <100k vectors)
ef_construction = 200  # Build-time quality (higher = better recall, slower builds)
ef_search = 200        # Search-time quality (higher = better recall, slower queries)
```

### Parameter Guide

**`ef_construction`** (default: 200)
- Controls the number of candidate neighbors evaluated during index construction.
- Higher values produce a denser, higher-quality graph but increase build time.
- Recommended range: 100-400.
- Cannot be changed after index construction without rebuilding.

| ef_construction | Recall@10 (approx.) | Build Time (100k vectors) |
|-----------------|---------------------|---------------------------|
| 100 | ~0.92 | ~0.3s |
| 200 | ~0.97 | ~0.6s |
| 400 | ~0.99 | ~1.2s |

**`ef_search`** (default: 200)
- Controls the number of candidates evaluated during each search.
- Can be tuned at query time without rebuilding.
- Lower values = faster but lower recall.

| ef_search | Recall@10 (approx.) | Search P95 |
|-----------|---------------------|------------|
| 50 | ~0.85 | ~1ms |
| 100 | ~0.93 | ~2ms |
| 200 | ~0.97 | ~4ms |
| 400 | ~0.99 | ~8ms |

**`enabled`** (default: true)
- When `false`, vector search falls back to brute-force SIMD cosine similarity.
- Brute-force is faster for small datasets (<50k vectors) due to zero overhead.
- HNSW becomes essential above 100k vectors where brute-force exceeds 100ms.

### When to Disable HNSW

Disable HNSW (`enabled = false`) when:
- Dataset has fewer than 50k vectors
- You need exact (non-approximate) results
- Write-heavy workload where index rebuild overhead dominates

### Rebuild Strategy

Ucotron uses rebuild-on-upsert: the entire HNSW index is rebuilt after each batch insert. This is because the underlying instant-distance crate does not support incremental insertion.

- Rebuild time at 100k vectors: <1s
- Rebuild time at 500k vectors: ~3-5s
- For datasets exceeding 500k vectors, consider batching ingestion into larger chunks to reduce rebuild frequency.

---

## 2. LMDB Storage Configuration

Ucotron uses LMDB (via heed) for both vector and graph storage. LMDB uses memory-mapped I/O for zero-copy reads.

### Configuration

```toml
[storage.vector]
path = "data/vector_db"
max_db_size = 10737418240  # 10 GB in bytes

[storage.graph]
path = "data/graph_db"
max_db_size = 10737418240  # 10 GB in bytes
batch_size = 10000
```

### `max_db_size`

- **Must be set before first run.** LMDB pre-allocates the memory map.
- Set this larger than your expected dataset size.
- On Linux, the virtual memory is lazy-allocated (no actual disk usage until written).
- On macOS, the file may be pre-allocated on some filesystems.

**Sizing guidelines:**

| Node Count | Approximate Disk Usage | Recommended max_db_size |
|------------|----------------------|------------------------|
| 10k | ~50 MB | 1 GB |
| 100k | ~430 MB | 5 GB |
| 500k | ~2.1 GB | 10 GB |
| 1M | ~4.2 GB | 20 GB |

If you see `MDB_MAP_FULL` errors, increase `max_db_size` and restart.

### `batch_size` (graph)

- Controls how many nodes/edges are inserted per LMDB write transaction.
- Default: 10,000 — good for most workloads.
- Larger batches reduce transaction overhead but increase memory usage during writes.
- LMDB uses single-writer concurrency: writes block other writers (but not readers).

| batch_size | Throughput (nodes/s) | Write Memory Overhead |
|------------|---------------------|----------------------|
| 1,000 | ~120k | Low |
| 10,000 | ~168k | Moderate |
| 50,000 | ~175k | High |

---

## 3. Ingestion Pipeline Tuning

The ingestion pipeline has 8 steps. Several can be disabled to trade accuracy for throughput.

### Configuration (code-level, via IngestionConfig)

```rust
IngestionConfig {
    ner_labels: vec!["person", "location", "organization", "date", "concept"],
    enable_relations: true,
    enable_entity_resolution: true,
    enable_contradiction_detection: true,
    entity_resolution_threshold: 0.5,
    embedding_batch_size: 32,  // ONNX embedding sub-batch size
    ner_batch_size: 8,         // GLiNER NER sub-batch size
}
```

### Optimization Strategies

**Maximize throughput (bulk import):**
- Disable `enable_entity_resolution` and `enable_contradiction_detection` — these query the existing graph per chunk.
- Increase `embedding_batch_size` to 64 or 128 (uses more GPU/CPU memory).
- Increase graph `batch_size` to 50,000.
- Run consolidation worker after bulk import to catch duplicates.

**Minimize latency (real-time ingestion):**
- Keep all steps enabled for immediate consistency.
- Use default `embedding_batch_size` (32) to limit per-request memory.
- Keep `ner_batch_size` at 8 (GLiNER memory scales with batch size).

**Reduce NER noise:**
- Narrow `ner_labels` to only the types you care about.
- Fewer labels = faster NER inference + fewer false positives.
- Example for a people-focused app: `["person", "organization"]`

### Pipeline Step Costs

| Step | Approximate Cost | Can Disable? |
|------|-----------------|-------------|
| Chunking | <1ms per chunk | No |
| Embedding (ONNX) | ~5ms per chunk (batched) | No |
| NER (GLiNER) | ~8ms per chunk (batched) | No (core feature) |
| Relation Extraction | ~1ms per chunk | Yes (`enable_relations`) |
| Entity Resolution | ~2-10ms per entity (graph query) | Yes (`enable_entity_resolution`) |
| Contradiction Detection | ~2-5ms per fact (graph query) | Yes (`enable_contradiction_detection`) |
| Graph Upsert | <1ms per node | No |
| Vector Index Update | Rebuild time (see HNSW section) | No |

---

## 4. Retrieval Pipeline Tuning

### Configuration (code-level, via RetrievalConfig)

```rust
RetrievalConfig {
    vector_top_k: 50,
    min_similarity: 0.0,
    graph_expansion_hops: 1,
    enable_entity_extraction: true,
    enable_community_expansion: true,
    max_community_members: 20,
    vector_sim_weight: 0.5,
    graph_centrality_weight: 0.3,
    recency_weight: 0.2,
    temporal_decay_half_life_secs: 2_592_000,  // 30 days
    final_top_k: 10,
}
```

### Key Parameters

**`vector_top_k`** (default: 50)
- Number of initial vector search candidates.
- Increasing this improves recall but increases graph expansion time (each candidate is expanded).
- For precision-focused use: 20-30.
- For recall-focused use: 100-200.

**`graph_expansion_hops`** (default: 1)
- Number of hops to traverse from each vector result.
- 1-hop: <1ms per expansion (adjacency list lookup).
- 2-hop: ~1-2ms per expansion (fan-out depends on graph density).
- 3-hop: not recommended for latency-sensitive paths (exponential fan-out).

**`max_community_members`** (default: 20)
- Caps the number of nodes fetched per community during community expansion.
- Lower values reduce candidate set size and speed up re-ranking.

**`temporal_decay_half_life_secs`** (default: 2,592,000 = 30 days)
- Controls how aggressively old memories are down-ranked.
- Shorter half-life = more emphasis on recent memories.
- Set to `u64::MAX` to effectively disable temporal decay.

### Re-ranking Weights

The final score is: `vector_sim_weight * sim + graph_centrality_weight * centrality + recency_weight * recency`

Adjust weights based on your use case:

| Use Case | vector_sim | graph_centrality | recency |
|----------|-----------|-----------------|---------|
| General (default) | 0.5 | 0.3 | 0.2 |
| Semantic search focus | 0.7 | 0.2 | 0.1 |
| Knowledge graph focus | 0.3 | 0.5 | 0.2 |
| Recent events focus | 0.3 | 0.2 | 0.5 |

---

## 5. Consolidation Worker Tuning

The background consolidation worker runs three phases: community detection, entity merge, and memory decay.

### Configuration

```toml
[consolidation]
trigger_interval = 100
enable_decay = true
decay_halflife_secs = 2592000   # 30 days
enable_community_detection = true
enable_entity_merge = true
entity_merge_threshold = 0.8
```

**Environment overrides:**
- `UCOTRON_CONSOLIDATION_TRIGGER_INTERVAL`
- `UCOTRON_CONSOLIDATION_ENABLE_DECAY`
- `UCOTRON_CONSOLIDATION_DECAY_HALFLIFE_SECS`

### `trigger_interval` (default: 100)
- Number of ingestion operations between consolidation runs.
- Lower values = more frequent consolidation = more consistent graph, but higher CPU overhead.
- For high-throughput ingestion, set to 500-1000 to reduce overhead.
- For interactive use with small batches, 50-100 is fine.

### `entity_merge_threshold` (default: 0.8)
- Cosine similarity threshold for merging duplicate entities.
- Higher threshold = fewer false merges but more duplicates.
- Lower threshold = more aggressive merging, risk of false merges.
- Recommended range: 0.7-0.9.

### `decay_halflife_secs` (default: 2,592,000 = 30 days)
- Memory decay formula: `score = 0.5 ^ (age_secs / half_life_secs)`
- A 30-day half-life means memories older than 30 days have <50% of their original score.
- Set to higher values (90 days, 365 days) for long-term knowledge bases.
- Disable with `enable_decay = false` for permanent memory storage.

---

## 6. ONNX Model Thread Tuning

All ONNX pipelines (embeddings, NER, CLIP, Whisper) use configurable intra-op parallelism.

### Default: 4 threads per model

For a machine with N CPU cores:
- **Single model active at a time:** Set `intra_threads = N` (use all cores).
- **Multiple models concurrent:** Set `intra_threads = N / active_models` (split cores).
- **Memory-constrained:** Set `intra_threads = 1-2` (reduces memory, slower inference).

### Server Workers

```toml
[server]
workers = 4  # HTTP server worker threads
```

**Environment override:** `UCOTRON_SERVER_WORKERS`

- Default of 4 is good for most single-instance deployments.
- For CPU-bound workloads (lots of embedding/NER), match to physical core count.
- For I/O-bound workloads (mostly search), can exceed core count (e.g., 2x cores).

---

## 7. Benchmark Reference Data

Baseline measurements on Apple M-series (single instance, debug/release builds):

### Ingestion (100k nodes, 500k edges, release build)

| Metric | HelixDB |
|--------|---------|
| Cold start | 5.25ms |
| Node ingestion | 6.99s (168k nodes/s) |
| Edge ingestion | 2.93s (170k edges/s) |
| Total throughput | 60,464 docs/s |
| Peak RAM | 320.58 MB |
| Disk size | 426.16 MB |

### Search (10k nodes, 1000 queries, release build)

| Query Type | P50 | P95 | P99 |
|-----------|-----|-----|-----|
| Vector search | 3.47ms | 5.26ms | 7.89ms |
| 1-hop traversal | 0.01ms | 0.02ms | 0.03ms |
| 2-hop traversal | 0.85ms | 1.62ms | 2.41ms |
| Hybrid search | 10.24ms | 18.37ms | 25.12ms |

### PRD Targets

| Metric | Target | Achieved |
|--------|--------|----------|
| Read latency (1-hop) | <10ms | 0.02ms |
| Read latency (2-hop) | <50ms | 1.62ms |
| Write throughput | >5,000 docs/s | 60,464 docs/s |
| Cold start | <200ms | 5.25ms |
| RAM (100k nodes) | <500MB | 320.58 MB |

---

## 8. Common Tuning Scenarios

### Scenario: Bulk Data Import

Maximize throughput when importing a large dataset for the first time.

```toml
[storage.graph]
batch_size = 50000

[consolidation]
trigger_interval = 1000  # Consolidate less frequently during import
```

In code, disable per-item checks:
```rust
let config = IngestionConfig {
    enable_entity_resolution: false,
    enable_contradiction_detection: false,
    enable_relations: false,  // Extract relations in post-processing
    embedding_batch_size: 64,
    ..Default::default()
};
```

After import, run a single consolidation pass to detect duplicates and communities.

### Scenario: Real-Time Chat Memory

Optimize for low-latency ingestion and retrieval in a conversational agent.

```toml
[storage.vector.hnsw]
ef_search = 100   # Slightly lower recall for faster search

[consolidation]
trigger_interval = 50      # More frequent consolidation
decay_halflife_secs = 604800  # 7-day half-life for chat context
```

Use `final_top_k = 5` and `vector_top_k = 20` for fast, focused retrieval.

### Scenario: Long-Term Knowledge Base

Optimize for high recall and permanent storage.

```toml
[storage.vector.hnsw]
ef_construction = 400  # Higher quality index
ef_search = 300        # Higher recall

[consolidation]
enable_decay = false           # No memory decay
entity_merge_threshold = 0.85  # Conservative merging
```

Use `vector_top_k = 100` and `graph_expansion_hops = 2` for thorough retrieval.

### Scenario: Resource-Constrained Edge Deployment

Minimize memory and CPU usage.

```toml
[storage.vector]
max_db_size = 2147483648  # 2 GB

[storage.vector.hnsw]
enabled = false  # Brute-force for small datasets

[storage.graph]
max_db_size = 2147483648  # 2 GB
batch_size = 5000

[server]
workers = 2

[consolidation]
trigger_interval = 500  # Less frequent consolidation
```

Set ONNX `intra_threads = 2` and `embedding_batch_size = 16`.

---

## 9. Monitoring Performance

### Server Metrics

The `/api/v1/metrics` endpoint exposes:
- `request_count`: Total requests handled
- Per-endpoint latency (via tracing middleware)

### Structured Logging

Enable JSON logging for performance analysis:

```toml
[server]
log_format = "json"
```

Each log entry includes `trace_id` and `span_id` for correlating ingestion/retrieval latency.

### OTLP Tracing

When OpenTelemetry is configured, traces include spans for:
- HTTP request handling
- Embedding generation
- NER extraction
- Graph queries
- Vector search
- Re-ranking

Use these spans to identify bottlenecks in your specific workload.
