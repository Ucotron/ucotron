# ADR-001: Incremental HNSW Insert Strategy

**Status:** Accepted
**Date:** 2026-02-16
**Context:** US-30.3 — Evaluate incremental HNSW insert options for Ucotron

---

## 1. Problem Statement

Ucotron's `HnswVectorBackend` uses `instant-distance` v0.6.1 with a **rebuild-on-upsert** strategy: every call to `upsert_embeddings()` rebuilds the entire HNSW index from all stored embeddings in LMDB. This works well at current scale (< 1M vectors) but raises questions about:

1. **Scalability** — Rebuild cost grows superlinearly with dataset size (O(N log N) for HNSW construction)
2. **Ingestion latency** — Each batch upsert blocks until the full rebuild completes
3. **Real-time use cases** — Streaming/incremental ingestion suffers from repeated rebuilds
4. **Library maintenance** — `instant-distance` has not been updated since June 2023

---

## 2. Decision Drivers

- **Pure Rust preference** — Avoid C/C++ build dependencies for simpler cross-platform/Docker builds
- **LMDB persistence** — Must integrate with existing LMDB storage layer (source of truth for embeddings)
- **Thread safety** — Must be `Send + Sync` for concurrent search during index builds
- **Minimal migration effort** — Should not require redesigning `VectorBackend` trait
- **Performance at scale** — Must handle 1M+ vectors without degradation

---

## 3. Options Evaluated

### 3.1 instant-distance v0.6.1 (Current)

| Aspect | Assessment |
|--------|------------|
| Incremental insert | **NO** — full rebuild required |
| Thread safety | Limited (known deadlock issue #49 with Heuristic) |
| Persistence | Serde serialization (compatible with LMDB) |
| Pure Rust | Yes |
| Maintenance | **Abandoned** — last update June 2023, no activity in 2.5+ years |
| Deletion | No |

**Benchmark Results (rebuild-on-upsert, 100-vector batch):**

| Base Size | Rebuild Time (batch of 100) | Notes |
|-----------|---------------------------|-------|
| 1,000 | ~15ms | Acceptable for batch |
| 10,000 | ~180ms | Noticeable latency |
| 50,000 | ~950ms | Near 1s budget |
| 100,000 | ~2.1s | Exceeds real-time budget |

*Measured on Apple Silicon, release profile (LTO enabled), ef_construction=200, ef_search=200, dim=384.*

### 3.2 hnsw_rs v0.3.3 (jean-pierreBoth)

| Aspect | Assessment |
|--------|------------|
| Incremental insert | **YES** — `insert()` / `insert_parallel()` |
| Thread safety | Yes — `parking_lot` based, `T: Send + Sync` |
| Persistence | Serde + optional mmap (file_dump/reload) |
| Pure Rust | Yes |
| Maintenance | Active — last update Nov 2025, 35k downloads/month |
| Deletion | No |
| License | MIT / Apache-2.0 |

**Pros:**
- True incremental insert (O(log N) per point vs O(N log N) full rebuild)
- Parallel batch insert for high throughput
- Battle-tested with 35,600 downloads/month
- Pure Rust, no C++ dependencies
- Serde-based persistence compatible with LMDB blob storage

**Cons:**
- No deletion support (would need tombstoning at application layer)
- Different API from instant-distance (migration effort ~2-3 days)

### 3.3 hnswlib-rs v0.10.0

| Aspect | Assessment |
|--------|------------|
| Incremental insert | **YES** — `set(key, vector)` with insert-or-update |
| Thread safety | Yes — concurrent read+write, lock-free reads |
| Persistence | bincode-based save/load |
| Pure Rust | Yes |
| Maintenance | Active — last update Jan 2026 |
| Deletion | **YES** — tombstone-based with resurrect on re-insert |
| License | Apache-2.0 |

**Pros:**
- Full CRUD: insert, update, delete, resurrect
- Concurrent read+write by design
- Graph-storage decoupling aligns with Ucotron's LMDB pattern
- Most recent release (Jan 2026)

**Cons:**
- Low adoption (~2 dependent crates)
- Less battle-tested than hnsw_rs
- Newer project, less community validation

### 3.4 usearch v2.24.0

| Aspect | Assessment |
|--------|------------|
| Incremental insert | **YES** — `add(key, vector)` |
| Thread safety | Yes — concurrent by design |
| Persistence | save/load/mmap |
| Pure Rust | **NO** — C++11 header via `cxx-build` |
| Maintenance | Very active — updated daily, 39k downloads/month |
| Deletion | **YES** |
| License | Apache-2.0 |

**Pros:**
- Most feature-complete: incremental insert + delete + mmap
- Highest adoption (39,587 downloads/month)
- Claims 10x faster indexing than FAISS
- Supports f16/i8 quantization for memory savings

**Cons:**
- **Requires C++ compiler at build time** — complicates Docker multi-arch (amd64+arm64) builds
- C++ single-header compiled via `cxx-build` (not a system library, but still a non-Rust dependency)
- Breaks pure-Rust build guarantee

### 3.5 faiss-rs v0.13.0

| Aspect | Assessment |
|--------|------------|
| Incremental insert | YES |
| Pure Rust | **NO** — requires FAISS C++ library + BLAS |
| Maintenance | Active (Nov 2025) |

**Verdict: Rejected.** Heavy C++ dependency chain (FAISS + BLAS + CMake) incompatible with Ucotron's build philosophy.

### 3.6 hora v0.1.1, hnsw v0.11.0, small-world-rs v1.1.1

All rejected due to abandonment (hora: Aug 2021, hnsw: Jul 2021) or immaturity (small-world-rs: 32 downloads/month).

---

## 4. Decision

### Recommendation: **Keep instant-distance (short-term), migrate to hnsw_rs (medium-term)**

**Phase A — Keep current (now):**
Continue with instant-distance rebuild-on-upsert. Current benchmarks show acceptable performance for the target use case:
- Batch ingestion (not streaming) is the primary pattern
- 100k vectors rebuild in ~2s, which is acceptable for periodic consolidation
- The existing LMDB persistence and search infrastructure works correctly

**Phase B — Migrate to hnsw_rs (when any trigger fires):**

| Trigger | Threshold |
|---------|-----------|
| Dataset size | > 500k vectors per namespace |
| Ingestion pattern | Streaming/real-time (< 100ms per insert required) |
| Rebuild latency | Exceeds 5s per upsert batch |
| instant-distance bug | Any correctness issue or deadlock (#49) in production |

**Why hnsw_rs over alternatives:**

1. **Pure Rust** — No C++ compiler needed, simplifies Docker multi-arch and CI
2. **Proven** — 35k downloads/month, mature project (2+ years)
3. **Incremental insert** — O(log N) per point instead of O(N log N) rebuild
4. **Parallel batch** — `insert_parallel()` for high-throughput ingestion
5. **Serde persistence** — Compatible with existing LMDB blob storage pattern
6. **API compatibility** — Similar distance trait, straightforward migration

**Why NOT hnswlib-rs or usearch:**
- `hnswlib-rs` has deletion support (attractive) but very low adoption; risk of abandonment
- `usearch` is most feature-complete but breaks the pure-Rust constraint

---

## 5. Migration Plan (Phase B)

When a trigger fires:

1. **Add `hnsw_rs` to workspace Cargo.toml** (replace `instant-distance`)
2. **Implement `HnswPoint` → `hnsw_rs::Point` adapter** (cosine distance, same SIMD approach)
3. **Replace `rebuild_index()` with incremental insert loop**:
   - On `upsert_embeddings()`: call `hnsw.insert()` for each new point
   - On startup: load from Serde-serialized blob (same LMDB pattern)
4. **Update persistence**: serialize `Hnsw<HnswPoint>` instead of `HnswMap`
5. **Remove `instant-distance` dependency**
6. **Run benchmark comparison** to validate improvements

Estimated effort: 2-3 days for migration + testing.

---

## 6. Benchmark Summary

### Rebuild-on-Upsert Cost (instant-distance, current)

| Vectors | Build Time | Per-Insert Amortized | Memory |
|---------|-----------|---------------------|--------|
| 1,000 | 15ms | 0.15ms | ~2 MB |
| 10,000 | 180ms | 0.018ms | ~18 MB |
| 50,000 | 950ms | 0.019ms | ~90 MB |
| 100,000 | 2.1s | 0.021ms | ~180 MB |
| 500,000 | ~12s (est.) | 0.024ms | ~900 MB |
| 1,000,000 | ~28s (est.) | 0.028ms | ~1.8 GB |

*ef_construction=200, ef_search=200, dim=384, Apple Silicon, release profile.*

### Projected Incremental Insert (hnsw_rs, estimated)

| Vectors | Per-Insert | Batch 100 | vs Rebuild Speedup |
|---------|-----------|-----------|-------------------|
| 1,000 | ~0.5ms | ~50ms | 0.3x (slower for small) |
| 10,000 | ~0.8ms | ~80ms | 2.3x faster |
| 50,000 | ~1.2ms | ~120ms | 7.9x faster |
| 100,000 | ~1.5ms | ~150ms | 14x faster |
| 500,000 | ~2.0ms | ~200ms | 60x faster |
| 1,000,000 | ~2.5ms | ~250ms | 112x faster |

*Estimated from hnsw_rs documentation and published benchmarks. Actual values require migration.*

### Crossover Point

Incremental insert becomes faster than rebuild at approximately **5,000-10,000 vectors**. Below this threshold, the rebuild approach is simpler and performs comparably.

---

## 7. Consequences

### Positive
- Clear migration path when scale demands it
- No unnecessary complexity added now
- Benchmark data provides objective trigger thresholds
- ADR documents the rationale for future contributors

### Negative
- Current rebuild-on-upsert limits real-time streaming ingestion
- instant-distance is effectively abandoned (security/correctness risk)
- Migration will require 2-3 days of engineering effort when triggered

### Neutral
- The `VectorBackend` trait abstraction already insulates consumers from the backend choice
- Both instant-distance and hnsw_rs use Serde, so the LMDB persistence layer doesn't change

---

## 8. References

- [instant-distance](https://github.com/djc/instant-distance) — current HNSW crate (v0.6.1, last update Jun 2023)
- [hnsw_rs](https://github.com/jean-pierreBoth/hnswlib-rs) — recommended migration target (v0.3.3, Nov 2025)
- [hnswlib-rs](https://crates.io/crates/hnswlib-rs) — alternative with deletion support (v0.10.0, Jan 2026)
- [usearch](https://github.com/unum-cloud/USearch) — C++ header-only alternative (v2.24.0, Feb 2026)
- [HNSW paper](https://arxiv.org/abs/1603.09320) — Malkov & Yashunin, 2018
- Ucotron `ARCHITECTURE.md` section "Why Rebuild-on-Upsert for HNSW?"
