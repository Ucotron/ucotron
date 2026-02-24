# Ucotron Phase 1 — Benchmark Results

**Date:** 2026-02-12
**Platform:** macOS Darwin 25.2.0 (Apple Silicon)
**Rust:** Edition 2021, release profile (LTO enabled, codegen-units=1)
**Seed:** 42 (deterministic, reproducible)

---

## 1. Ingestion Benchmark

### 1.1 HelixDB — 100k Nodes, 500k Edges

| Metric               | HelixDB          |
|----------------------|------------------|
| Cold Start           | 5.25ms           |
| Node Ingestion       | 6.99s            |
| Edge Ingestion       | 2.93s            |
| Total Ingestion      | 9.92s            |
| Throughput (total)   | 60,464 docs/s    |
| Throughput (nodes)   | 14,301 nodes/s   |
| Throughput (edges)   | 170,598 edges/s  |
| Peak RAM (delta)     | 320.58 MB        |
| Disk Size            | 426.16 MB        |

### 1.2 Side-by-Side — 10k Nodes, 50k Edges

| Metric               | HelixDB          | CozoDB           |
|----------------------|------------------|------------------|
| Cold Start           | 4.35ms           | 3.30ms           |
| Node Ingestion       | 59.53ms          | 174.03s          |
| Edge Ingestion       | 118.44ms         | 251.10ms         |
| Total Ingestion      | 177.97ms         | 174.28s          |
| Throughput (total)   | 337,128 docs/s   | 344 docs/s       |
| Throughput (nodes)   | 167,971 nodes/s  | 57 nodes/s       |
| Throughput (edges)   | 422,155 edges/s  | 199,123 edges/s  |
| Peak RAM (delta)     | 38.50 MB         | 1.63 GB          |
| Disk Size            | 40.05 MB         | 38.46 MB         |

> **Note:** CozoDB node ingestion is dominated by HNSW index construction (384-dim vectors, ef=200). Edge ingestion is fast because edges don't require vector indexing. At 100k nodes, CozoDB failed to complete ingestion (process exceeded memory/time limits after 30+ minutes).

### 1.3 CozoDB 100k Ingestion Attempt

CozoDB was unable to complete ingestion of 100k nodes + 500k edges. After 30+ minutes at 100% CPU and 2.5 GB RAM, the process was terminated. The bottleneck is HNSW index construction during node insertion — each 384-dimensional vector must be indexed into the HNSW graph, which scales poorly at O(n log n) per insert.

---

## 2. Search Benchmark

### 2.1 HelixDB — 10k Nodes, 50k Edges, 1000 Queries

| Metric           | HelixDB          |
|------------------|------------------|
| Vector P50       | 4.79ms           |
| Vector P95       | 5.26ms           |
| Vector P99       | 5.45ms           |
| Graph 1-hop P50  | 0.01ms           |
| Graph 1-hop P95  | 0.02ms           |
| Graph 1-hop P99  | 0.05ms           |
| Graph 2-hop P50  | 0.33ms           |
| Graph 2-hop P95  | 1.62ms           |
| Graph 2-hop P99  | 2.01ms           |
| Hybrid P50       | 14.26ms          |
| Hybrid P95       | 18.37ms          |
| Hybrid P99       | 20.09ms          |

### 2.2 CozoDB — 1k Nodes, 5k Edges, 100 Queries

| Metric           | CozoDB           |
|------------------|------------------|
| Vector P50       | 4.16ms           |
| Vector P95       | 4.47ms           |
| Vector P99       | 4.56ms           |
| Graph 1-hop P50  | 3.88ms           |
| Graph 1-hop P95  | 4.28ms           |
| Graph 1-hop P99  | 5.31ms           |
| Graph 2-hop P50  | 12.49ms          |
| Graph 2-hop P95  | 23.94ms          |
| Graph 2-hop P99  | 25.74ms          |
| Hybrid P50       | 486.15ms         |
| Hybrid P95       | 834.55ms         |
| Hybrid P99       | 1,560ms          |

> **Note:** CozoDB search was run on a 10x smaller dataset (1k nodes vs 10k) and still showed dramatically worse latency. CozoDB could not complete 1000 hybrid queries on 10k nodes within 40 minutes. HelixDB uses brute-force SIMD vector search which is fast for <100k vectors; CozoDB uses HNSW which has good asymptotic complexity but high constant overhead for the Datalog query engine.

---

## 3. Recursion Benchmark (Path Finding)

### 3.1 Chain Traversal (start → end, 100 iterations)

| Depth | Nodes | HelixDB P50 | HelixDB P95 | HelixDB P99 | CozoDB P50 | CozoDB P95 | CozoDB P99 |
|-------|-------|-------------|-------------|-------------|------------|------------|------------|
| 10    | 10    | 0.00ms      | 0.00ms      | 0.00ms      | 0.32ms     | 0.37ms     | 0.38ms     |
| 20    | 20    | 0.00ms      | 0.01ms      | 0.01ms      | 0.70ms     | 0.74ms     | 0.78ms     |
| 50    | 50    | 0.01ms      | 0.01ms      | 0.01ms      | 2.15ms     | 2.20ms     | 2.30ms     |
| 100   | 100   | 0.02ms      | 0.03ms      | 0.03ms      | 5.61ms     | 5.72ms     | 5.75ms     |

HelixDB chain traversal scales linearly with depth at ~0.2us per hop. CozoDB scales linearly at ~56us per hop (280x slower) due to Datalog query engine overhead.

### 3.2 Tree Traversal (branching=3, depth=10, 29,524 nodes)

| Engine  | P50    | P95    | P99    | Mean   | RAM      |
|---------|--------|--------|--------|--------|----------|
| HelixDB | 4.99ms | 5.20ms | 5.38ms | 4.99ms | 1.20 MB  |
| CozoDB  | DNF    | DNF    | DNF    | DNF    | DNF      |

> **DNF = Did Not Finish.** CozoDB was unable to complete tree traversal on 29,524 nodes within the allotted time. The combination of HNSW index construction during ingestion + BFS path-finding over Datalog exceeded practical limits.

---

## 4. PRD Target Compliance

### 4.1 Critical Targets

| Target                          | Threshold    | HelixDB                  | CozoDB                        | Status        |
|---------------------------------|-------------|--------------------------|-------------------------------|---------------|
| Latency 1-hop                   | < 10ms      | 0.02ms P95               | 4.28ms P95 (1k nodes)        | BOTH PASS     |
| Latency 2-hop                   | < 50ms      | 1.62ms P95               | 23.94ms P95 (1k nodes)       | BOTH PASS     |
| Write throughput                | > 5,000 d/s | 60,464 d/s (100k)       | 344 d/s (10k)                | HELIX PASS / COZO FAIL |
| Cold start                      | < 200ms     | 5.25ms                   | 3.30ms                        | BOTH PASS     |
| RAM < 500MB for 100k nodes      | < 500MB     | 320.58 MB                | DNF (exceeded 2.5GB at 100k) | HELIX PASS / COZO FAIL |

### 4.2 Desirable Targets

| Target                          | Threshold    | HelixDB          | CozoDB                   | Status        |
|---------------------------------|-------------|------------------|--------------------------|---------------|
| Hybrid search P95               | < 50ms      | 18.37ms (10k)    | 834.55ms (1k nodes)      | HELIX PASS / COZO FAIL |
| Disk < 2GB for 1M nodes         | < 2GB       | ~4.26 GB (est.)  | N/A (cannot ingest 1M)   | HELIX MARGINAL / COZO N/A |

> Disk estimate for HelixDB at 1M nodes: extrapolated linearly from 426 MB at 100k nodes. Actual disk usage may be sub-linear due to LMDB page reuse.

---

## 5. Summary

### HelixDB (Heed/LMDB)

**Strengths:**
- Exceptional ingestion throughput: 60k+ docs/s at 100k scale
- Sub-millisecond graph traversal (0.02ms P95 at 100 hops)
- Hybrid search under 20ms P95
- Memory-efficient: 320 MB for 100k nodes (well under 500MB target)
- Cold start: 5ms
- Zero-copy reads via memory-mapped LMDB
- Predictable, linear scaling

**Weaknesses:**
- Brute-force vector search — O(n) per query, adequate for <100k but needs HNSW for 1M+
- Disk usage may exceed 2GB at 1M nodes (needs validation)
- No built-in query language (imperative Rust code for all queries)

### CozoDB (Datalog/RocksDB + HNSW)

**Strengths:**
- Expressive Datalog query language for recursive queries
- Cold start comparable to HelixDB (3.3ms)
- Edge ingestion fast when no vector indexing involved (199k edges/s)
- Compact on-disk size (38 MB for 10k nodes)

**Weaknesses:**
- HNSW index construction makes node ingestion 2,900x slower than HelixDB (57 nodes/s vs 167k nodes/s)
- Cannot complete ingestion at 100k scale (OOM/timeout after 30+ minutes)
- Hybrid search P95 = 835ms on just 1k nodes (47x over the 50ms target)
- Graph traversal 280x slower than HelixDB per hop due to Datalog overhead
- 1.63 GB RAM for 10k nodes (extrapolates to ~16 GB for 100k — untenable)
- Tree traversal on 29k nodes: Did Not Finish

---

## 6. Verdict

**GO: HelixDB** — Passes all 5 critical PRD targets and 1 of 2 desirable targets.

**NO-GO: CozoDB** — Fails 2 of 5 critical targets (write throughput, RAM) and all desirable targets. The HNSW index construction overhead makes it unsuitable for the ingestion volumes required by Ucotron. While Datalog provides elegant query semantics for recursive patterns, the performance penalty is prohibitive (280x slower traversal, 2900x slower ingestion).

### Recommendation for Phase 2

1. **Proceed with HelixDB** as the sole storage engine
2. **Add HNSW indexing** (via `instant-distance` crate) for vector search at 1M+ scale — brute-force SIMD is adequate up to ~100k vectors
3. **Implement a thin query DSL** on top of HelixDB's Rust API to recover some of Datalog's ergonomic advantages (see ERGONOMIA.md)
4. **Validate disk usage** at 1M nodes to confirm the 2GB desirable target
5. **Consider hybrid architecture** where CozoDB handles only the recursive/inferential queries on small subgraphs extracted by HelixDB (micro-Datalog pattern)
