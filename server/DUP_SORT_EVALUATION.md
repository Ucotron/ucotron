# LMDB DUP_SORT Adjacency Index Evaluation (US-30.5)

## Summary

This document evaluates whether LMDB's `DUP_SORT` flag would improve adjacency index
performance compared to the current `Vec<AdjEntry>` approach used in `HelixGraphBackend`.

**Recommendation: Keep the current Vec-based approach.** DUP_SORT improves write
throughput but degrades read performance and increases disk usage. Since Ucotron's
workload is read-heavy (BFS traversal, multi-hop queries), the Vec approach is the
better trade-off.

---

## Approaches Compared

### Current: Vec-based Adjacency

```
adj_out[NodeId] → Vec<(neighbor_id, edge_type)>   (single LMDB value)
adj_in[NodeId]  → Vec<(neighbor_id, edge_type)>   (single LMDB value)
```

- **Write**: Read-modify-write per edge (get Vec, push, put Vec back)
- **Read**: Single `get()` deserializes the entire Vec at once
- **Dedup**: Manual (caller must handle duplicate edges)

### Alternative: DUP_SORT Adjacency

```
adj_out[NodeId] → (neighbor_id, edge_type)   (one LMDB DUP_SORT entry per edge)
adj_in[NodeId]  → (neighbor_id, edge_type)   (one LMDB DUP_SORT entry per edge)
```

- **Write**: Direct `put()` per edge (LMDB handles sorted insertion natively)
- **Read**: Cursor iteration via `get_duplicates()` collects entries one by one
- **Dedup**: Automatic (LMDB DUP_SORT deduplicates identical key+value pairs)

---

## Benchmark Results

All benchmarks run in `--release` mode on Apple Silicon (M-series), LMDB `map_size=1GB`.
Edge distribution: power-law (Zipf-like) with 10 edge types.

### Small Scale: 1k nodes, 5k edges

| Metric | Vec | DUP_SORT | Speedup |
|--------|-----|----------|:-------:|
| Edge insertion (4,994 edges) | 15.06ms | 12.91ms | **1.17x** |
| Edge throughput | 331,629 edges/s | 386,862 edges/s | -- |
| Neighbor read (100 nodes) | 0.10ms | 0.10ms | 1.00x |
| BFS 2-hop (10 nodes) | 1.02ms | 1.13ms | 0.90x |
| Disk size | 0.38 MB | 0.59 MB | 1.53x |

### Medium Scale: 10k nodes, 50k edges

| Metric | Vec | DUP_SORT | Speedup |
|--------|-----|----------|:-------:|
| Edge insertion (49,994 edges) | 87.11ms | 49.96ms | **1.74x** |
| Edge throughput | 573,898 edges/s | 1,000,761 edges/s | -- |
| Neighbor read (100 nodes) | 0.08ms | 0.11ms | 0.77x |
| BFS 2-hop (10 nodes) | 1.81ms | 1.83ms | 0.99x |
| Disk size | 2.15 MB | 4.15 MB | 1.93x |

### Large Scale: 100k nodes, 500k edges (PRD minimum benchmark)

| Metric | Vec | DUP_SORT | Speedup |
|--------|-----|----------|:-------:|
| Edge insertion (499,994 edges) | 771.99ms | 594.98ms | **1.30x** |
| Edge throughput | 647,671 edges/s | 840,356 edges/s | -- |
| Neighbor read (100 nodes) | 0.17ms | 0.19ms | 0.90x |
| BFS 2-hop (10 nodes) | 6.86ms | 9.06ms | 0.76x |
| Disk size | 24.87 MB | 37.05 MB | 1.49x |

### High-Degree Node: 1 node, 1000 outgoing edges

| Metric | Vec | DUP_SORT |
|--------|-----|----------|
| Insert 1000 edges to hub | 5,951us | 5,017us |

---

## Analysis

### Write Performance (DUP_SORT wins)

DUP_SORT is **1.2-1.7x faster** for edge insertion because it eliminates the
read-modify-write cycle. The Vec approach must:

1. Read the existing Vec from LMDB
2. Deserialize with bincode
3. Append the new entry
4. Re-serialize the growing Vec
5. Write the entire Vec back

For high-degree nodes, this is O(degree) per insert as the Vec grows. DUP_SORT
does a single B-tree sorted insertion per edge.

The advantage is most pronounced at medium scale (1.74x at 50k edges) where hub
nodes accumulate enough edges to make the read-modify-write costly. At 500k edges,
the advantage narrows to 1.30x because LMDB's DUP_SORT B-tree itself grows.

### Read Performance (Vec wins)

Vec is **10-24% faster** for single-node neighbor reads because:

- **Vec**: Single `get()` → one bincode deserialization → all neighbors in memory
- **DUP_SORT**: `get_duplicates()` → cursor iteration → per-entry deserialization

The Vec approach benefits from spatial locality: the entire adjacency list is stored
contiguously in a single LMDB page (or span of pages), enabling sequential reads.
DUP_SORT entries may span multiple B-tree leaf pages.

### BFS Traversal (Vec wins)

For multi-hop BFS traversal, Vec is **up to 24% faster** at scale. Each BFS step
does one `get()` per node. With Vec, this is one LMDB lookup + one deserialization.
With DUP_SORT, this is one LMDB cursor open + N individual reads per node. The
overhead compounds across multiple hops.

### Disk Usage (Vec wins)

DUP_SORT uses **49-93% more disk** because LMDB stores each duplicate entry with
its own B-tree overhead (internal node pointers, alignment padding). The Vec approach
packs all neighbors into a single bincode-serialized blob, which is more space-efficient.

### Deduplication (DUP_SORT wins)

DUP_SORT automatically prevents duplicate edges (identical `(target, edge_type)` pairs
under the same source key). The current Vec approach silently allows duplicates, which
can waste memory and produce incorrect BFS results. However, this can be addressed by
adding a dedup check in `upsert_edges()` without changing the storage strategy.

---

## Migration Complexity

Switching to DUP_SORT would require changes to:

| Area | Effort | Risk |
|------|--------|------|
| `HelixGraphBackend` adjacency databases | Medium | Low |
| `get_neighbors()` and `find_path()` | Medium | Medium |
| `get_all_edges()` iteration | Low | Low |
| `store_community_assignments()` | None | None |
| Phase 1 `HelixEngine` (backward compat) | High | High |
| Test updates | Medium | Low |
| **Total** | **~2 days** | **Medium** |

The main risk is BFS traversal regression: DUP_SORT's cursor-based iteration changes
the iterator lifetime semantics (iterators borrow the read transaction, requiring
careful scoping to avoid "does not live long enough" errors).

---

## Decision

**Keep Vec-based adjacency lists.** Rationale:

1. **Read-heavy workload**: Ucotron's primary operations are retrieval (BFS, multi-hop,
   hybrid search). Write throughput is already ~648k edges/s, well above the PRD target
   of 5,000 docs/s. Read performance matters more.

2. **Disk efficiency**: At 100k nodes, Vec uses 25MB vs DUP_SORT's 37MB (48% less).
   For the 1M node target, Vec would use ~250MB vs ~370MB.

3. **Code simplicity**: The Vec approach uses standard `get()`/`put()` operations with
   no cursor lifetime management. DUP_SORT requires careful scoping of iterators to
   satisfy Rust's borrow checker.

4. **Dedup fix**: The one advantage of DUP_SORT (automatic deduplication) can be
   addressed by adding a `contains()` check in `upsert_edges()` — a one-line fix
   to the Vec approach.

5. **Migration risk**: Changing the adjacency storage format would require updates to
   Phase 1 `HelixEngine`, Phase 2 `HelixGraphBackend`, all BFS code, and the test
   suite. The marginal write improvement does not justify this effort.

### When to Reconsider

DUP_SORT would be worth revisiting if:
- Write throughput becomes a bottleneck (e.g., real-time streaming ingestion)
- Individual hub nodes exceed 100k edges (read-modify-write becomes prohibitive)
- Disk usage is not a constraint (e.g., NVMe with abundant space)

---

## Benchmark Code

The benchmark implementation is in `helix_impl/src/lib.rs`, module `dup_sort_eval`.
Run with:

```bash
# Quick test (1k nodes, 5k edges)
cargo test --package ucotron-helix --release dup_sort_benchmark_small -- --nocapture

# Full benchmark (10k + 100k)
cargo test --package ucotron-helix --release dup_sort_benchmark -- --ignored --nocapture
```
