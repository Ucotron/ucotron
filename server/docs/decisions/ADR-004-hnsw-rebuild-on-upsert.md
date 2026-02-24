# ADR-004: HNSW Rebuild-on-Upsert Strategy

**Status:** Accepted
**Date:** 2026-02-13
**Decision Makers:** Phase 2 vector index design

## Context

After selecting HelixDB (ADR-001), Ucotron needed an HNSW vector index for sub-linear similarity search. The Phase 1 brute-force SIMD approach works up to ~100k vectors but doesn't scale to the 1M+ target.

The chosen crate, `instant-distance` (v0.6), provides a fast Rust HNSW implementation but does **not** support incremental insertion. Options:

1. **Rebuild-on-upsert** - Rebuild the entire HNSW index from stored embeddings after each batch upsert
2. **Incremental HNSW** - Use a different crate (e.g., `hnsw_rs`) that supports incremental inserts
3. **Hybrid** - Use instant-distance for reads, queue writes for periodic batch rebuilds

## Decision

**Rebuild-on-upsert**: after each `upsert_embeddings()` call, load all embeddings from LMDB, rebuild the HNSW index, and persist the serialized index back to LMDB.

## Rationale

### Performance is acceptable at target scale

- Rebuild time for 100k vectors (384-dim): <1 second
- Rebuild time for 1M vectors: estimated ~10 seconds (acceptable for batch ingestion)
- Read queries are unaffected (HNSW search remains O(log n))

### Simplicity over complexity

- LMDB stores embeddings as the single source of truth
- The HNSW index is a derived, disposable structure (can be rebuilt from embeddings at any time)
- No complex write-ahead log or incremental update logic needed
- Crash recovery is trivial: rebuild from stored embeddings on startup

### Why not incremental?

- `hnsw_rs` exists but has fewer users and less mature Serde support
- Incremental HNSW inserts can degrade recall over time without periodic rebalancing
- Full rebuilds guarantee optimal graph quality for every query

### Persistence

The serialized HNSW index is stored in a dedicated LMDB database:
- Key: fixed `b"hnsw_index"`
- Value: bincode-serialized `HnswMap` from instant-distance
- Loaded on backend initialization; rebuilt on upsert

## Consequences

- Write latency includes full index rebuild (O(n log n) per batch)
- Not suitable for real-time single-document inserts at very large scale (>10M vectors)
- For >1M vectors, consider migrating to incremental HNSW or batch-queued rebuilds
- The `with-serde` feature of instant-distance is required for persistence

## References

- `helix_impl/src/lib.rs` - `HelixVectorBackend::upsert_embeddings()` and `rebuild_hnsw_index()`
- `Cargo.toml` - `instant-distance = { version = "0.6", features = ["with-serde"] }`
- ADR-001 - HelixDB selection rationale
