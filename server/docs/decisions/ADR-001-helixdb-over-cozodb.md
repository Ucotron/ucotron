# ADR-001: HelixDB (LMDB) Selection Over CozoDB

**Status:** Accepted
**Date:** 2026-02-12
**Decision Makers:** Phase 1 benchmark evaluation

## Context

Ucotron needed a storage engine for its cognitive memory graph supporting:
- High-throughput node/edge ingestion (>5,000 docs/s)
- Sub-10ms single-hop graph traversal
- Sub-50ms two-hop traversal
- <500 MB RAM for 100,000 nodes
- Vector similarity search (384-dim embeddings)

Two candidates were evaluated end-to-end:
- **HelixDB**: Heed bindings over LMDB (memory-mapped B+ tree)
- **CozoDB**: Datalog engine over RocksDB with built-in HNSW

## Decision

**Use HelixDB (Heed/LMDB)** as the sole storage backend for Phase 2+.

## Rationale

### Weighted Scoring (9.45 vs 3.95)

| Criterion            | Weight | HelixDB | CozoDB |
|----------------------|--------|---------|--------|
| Read latency         | 30%    | 10/10   | 4/10   |
| Write throughput     | 20%    | 10/10   | 1/10   |
| Query ergonomics     | 15%    | 8/10    | 6/10   |
| Memory usage         | 15%    | 9/10    | 1/10   |
| Cold start           | 10%    | 10/10   | 10/10  |
| Maturity/Stability   | 10%    | 9/10    | 5/10   |

### Key Benchmark Results (100k nodes, 500k edges)

- **Throughput**: HelixDB 60,464 docs/s vs CozoDB 344 docs/s (10k scale; CozoDB DNF at 100k)
- **RAM**: HelixDB 321 MB vs CozoDB 1.63 GB (at 10k nodes only)
- **Hybrid P95**: HelixDB 18ms vs CozoDB 835ms (at 1k nodes)

CozoDB's bottleneck was HNSW index construction during node insertion (57 nodes/s), which made it unable to complete the 100k-node benchmark within 30 minutes.

### CozoDB's Advantage

Recursive Datalog provided elegant multi-hop traversal queries. This advantage was partially captured by implementing a type-safe QueryBuilder DSL on top of HelixDB in Phase 2.

## Consequences

- CozoDB crate archived to `_archive/cozo_impl/`
- All Phase 2+ development targets HelixDB exclusively
- HNSW vector indexing must be added separately (see ADR-004)
- Query DSL built to compensate for loss of Datalog expressiveness

## References

- `RESULTS.md` - Full benchmark data
- `DECISION.md` - Weighted scoring matrix
- `ERGONOMIA.md` - Developer ergonomics comparison
