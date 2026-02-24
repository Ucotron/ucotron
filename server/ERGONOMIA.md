# Ergonomics Report: HelixDB (Rust/LMDB) vs CozoDB (Datalog)

**Author:** Implementer (automated analysis)
**Date:** 2026-02-11
**Context:** Ucotron Phase 1 — US-4.4

This document compares the developer ergonomics of writing queries against HelixDB (imperative Rust over Heed/LMDB) versus CozoDB (Datalog + Rust glue). Comparisons cover lines of code, clarity, ease of modification, and debugging.

---

## 1. Query Comparison: Lines of Code

| Query Type              | HelixDB (Rust) | CozoDB (Datalog+Rust) | Ratio |
|-------------------------|---------------:|----------------------:|------:|
| `get_neighbors` (N-hop) |       43 lines |             44 lines  |  ~1:1 |
| `vector_search` (top-k) |       36 lines |             25 lines  | 1.4:1 |
| `find_path` (BFS)       |       54 lines |             65 lines  | 0.8:1 |
| `insert_nodes`          |       27 lines |             47 lines  | 0.6:1 |
| `insert_edges`          |       38 lines |             33 lines  | 1.2:1 |
| `hybrid_search`         |        3 lines |              3 lines  |  1:1  |

**Total query implementation:** HelixDB ~201 lines vs CozoDB ~217 lines.

The difference is small. Neither engine requires dramatically more code. CozoDB's overhead comes from string-building CozoScript, while HelixDB's comes from manual adjacency list management.

---

## 2. Multi-Hop Traversal (`get_neighbors`)

### HelixDB — Imperative BFS (43 lines)

```rust
let mut visited = HashSet::new();
let mut queue = VecDeque::new();
visited.insert(id);
queue.push_back((id, 0u8));
while let Some((current_id, depth)) = queue.pop_front() {
    if depth >= hops { continue; }
    if let Some(out_list) = self.adj_out.get(&rtxn, &current_id)? {
        for &(target, _et) in &out_list {
            if visited.insert(target) {
                result.push(self.nodes_db.get(&rtxn, &target)?.unwrap());
                queue.push_back((target, depth + 1));
            }
        }
    }
    // ... same for adj_in ...
}
```

**Strengths:** Explicit control flow; easy to step through in a debugger; zero-copy LMDB reads; can inspect state at every point.

**Weaknesses:** Must manually manage visited set, queue, and depth tracking; duplicated logic for outgoing and incoming edges.

### CozoDB — Recursive Datalog (44 lines, but 5 lines of Datalog core)

```datalog
reachable[neighbor, depth] := *edges{source: start, target: neighbor}, depth = 1
reachable[neighbor, depth] := *edges{source: neighbor, target: start}, depth = 1
reachable[neighbor, depth] := reachable[prev, prev_depth],
    *edges{source: prev, target: neighbor}, depth = prev_depth + 1, depth <= max
reachable[neighbor, depth] := reachable[prev, prev_depth],
    *edges{source: neighbor, target: prev}, depth = prev_depth + 1, depth <= max
?[neighbor] := reachable[neighbor, _depth], neighbor != start
```

**Strengths:** Declarative intent is clear — "find all nodes reachable within N hops"; the recursion semantics are handled by the engine; set deduplication is automatic.

**Weaknesses:** The Rust glue code (35+ lines) for building the query string and parsing `DataValue` results dwarfs the Datalog itself; debugging Datalog query errors requires reading CozoDB error messages, which are often cryptic; bidirectional traversal requires 4 rules (easy to forget one).

### Verdict: Traversal

**Tie**, with different trade-offs. HelixDB is more transparent and debuggable. CozoDB is more concise *in the query language* but the surrounding Rust is boilerplate-heavy.

---

## 3. Vector Search

### HelixDB — Brute-Force SIMD (36 lines)

```rust
let mut heap: BinaryHeap<MinScored> = BinaryHeap::with_capacity(top_k + 1);
let iter = self.nodes_db.iter(&rtxn)?;
for entry in iter {
    let (_id, node) = entry?;
    let sim = cosine_similarity(query, &node.embedding);
    if heap.len() < top_k || sim > heap.peek().unwrap().0 {
        heap.push(MinScored(sim, node.id));
        if heap.len() > top_k { heap.pop(); }
    }
}
```

**Strengths:** Full control over the search algorithm; can profile and optimize (SIMD, parallelism); O(n log k) min-heap is explicitly visible; no hidden overhead.

**Weaknesses:** Must implement the similarity function, min-heap logic, and iterator handling manually; brute-force doesn't scale past ~1M vectors without further work.

### CozoDB — HNSW Index Query (25 lines)

```rust
let script = format!(
    "?[dist, node_id] := ~embeddings:vec_idx{{ node_id, embedding \
     | query: {}, k: {}, ef: 200, bind_distance: dist }}",
    query_vec, top_k
);
```

**Strengths:** One-line Datalog query; HNSW index provides sub-linear search (O(log n) amortized); no manual similarity calculation; index maintenance is automatic.

**Weaknesses:** Must convert embeddings to CozoDB string format (`vec([0.1, 0.2, ...])`) for every query; HNSW returns *cosine distance* (0–2 range), requiring conversion to similarity (`1 - dist`); index rebuild during ingestion is very slow (bottleneck for batch inserts); tuning `ef` requires experimentation.

### Verdict: Vector Search

**CozoDB wins on ergonomics** for querying (one declarative line vs 36 lines of heap logic). **HelixDB wins on transparency and control.** For production, CozoDB's HNSW index is the better choice unless you need custom scoring functions.

---

## 4. Path Finding (`find_path`)

### HelixDB — BFS with Parent Tracking (54 lines)

```rust
let mut parent: HashMap<NodeId, NodeId> = HashMap::new();
let mut queue = VecDeque::new();
parent.insert(source, source);
queue.push_back((source, 0u32));
while let Some((current, depth)) = queue.pop_front() {
    if depth >= max_depth { continue; }
    for &(neighbor, _) in &out_list {
        if !parent.contains_key(&neighbor) {
            parent.insert(neighbor, current);
            if neighbor == target {
                return Ok(Some(reconstruct_path(&parent, source, target)));
            }
            queue.push_back((neighbor, depth + 1));
        }
    }
}
```

**Strengths:** Early termination on target found; explicit path reconstruction; easy to add visited-node logging; straightforward to extend with weighted shortest path (Dijkstra).

**Weaknesses:** Boilerplate for parent tracking and path reconstruction (~15 extra lines); duplicated for outgoing/incoming edges.

### CozoDB — Rust-Side BFS over Datalog Edges (65 lines)

```rust
// For each BFS level, query CozoDB for neighbors
let script = format!(
    "?[neighbor] := *edges{{ source: {cur}, target: neighbor }}\n\
     ?[neighbor] := *edges{{ source: neighbor, target: {cur} }}",
    cur = current,
);
let result = self.run_query(&script)?;
// Then BFS in Rust with parent tracking...
```

**Note:** We deliberately *did not* use recursive Datalog with `path = append(p1, [dst])` because it causes combinatorial path explosion on bidirectional graphs. The hybrid approach (Rust BFS + per-hop Datalog queries) was necessary for correctness.

**Strengths:** Leverages CozoDB's edge storage without manual adjacency management.

**Weaknesses:** **Worst of both worlds** — must maintain BFS state in Rust *and* issue a Datalog query per BFS expansion step; each hop requires a full round-trip to CozoDB's query engine; more lines than pure Rust BFS; no benefit from Datalog's recursive capabilities (they were unusable here).

### Verdict: Path Finding

**HelixDB wins clearly.** CozoDB's Datalog recursion—its main ergonomic advantage—was inapplicable for path finding on bidirectional graphs. The fallback (Rust BFS + per-hop queries) is slower and more complex than HelixDB's direct adjacency-list BFS.

---

## 5. Data Insertion

### HelixDB — Direct Key-Value Puts (27 lines for nodes)

```rust
for chunk in nodes.chunks(self.batch_size) {
    let mut wtxn = self.env.write_txn()?;
    for node in chunk {
        self.nodes_db.put(&mut wtxn, &node.id, node)?;
        // Update type index: read-modify-write
        let mut ids = self.nodes_by_type.get(&wtxn, &type_key)?
            .unwrap_or_default();
        ids.push(node.id);
        self.nodes_by_type.put(&mut wtxn, &type_key, &ids)?;
    }
    wtxn.commit()?;
}
```

**Strengths:** Bincode serialization is automatic via `SerdeBincode`; transaction semantics are explicit; batch size is configurable.

**Weaknesses:** Must manually maintain secondary indices (adjacency lists, type index) via read-modify-write pattern; the adjacency list append pattern (`get → push → put`) is error-prone if a put is missed.

### CozoDB — CozoScript String Building (47 lines for nodes)

```rust
for node in chunk {
    let content_esc = node.content.replace('\'', "''");
    node_rows.push(format!(
        "[{}, '{}', '{}', {}, '{}']",
        node.id, content_esc, node_type_str, node.timestamp, meta_esc
    ));
    emb_rows.push(format!("[{}, {}]", node.id, embedding_to_cozo_vec(&node.embedding)));
}
let script = format!(
    "?[id, content, node_type, timestamp, metadata] <- [{}]\n\
     :put nodes {{ id => content, node_type, timestamp, metadata }}",
    node_rows.join(", ")
);
```

**Strengths:** Schema is declarative (Datalog relations); HNSW index updates happen automatically on insert; edge relations don't need manual adjacency lists.

**Weaknesses:** **String-building CozoScript is extremely error-prone:** must escape single quotes, format vectors as `vec([...])`, serialize metadata as a custom delimited string; embedding serialization (384 floats → comma-separated string) adds significant overhead; any formatting error produces cryptic parser failures.

### Verdict: Insertion

**HelixDB wins on ergonomics.** The type-safe `put()` API with automatic bincode serialization is far cleaner than string-building CozoScript queries. CozoDB's main insertion advantage (automatic index maintenance) is offset by the fragility and performance cost of string-based query construction.

---

## 6. Ease of Modification

**Scenario: Add a weight filter to neighbor traversal** (e.g., only follow edges with weight > 0.5)

### HelixDB

```rust
// Change in get_neighbors: add a filter inside the loop
for &(target, et) in &out_list {
    // Fetch the full edge to check weight
    let edge_key: EdgeKey = (current_id, target, et);
    if let Some(edge) = self.edges_db.get(&rtxn, &edge_key)? {
        if edge.weight > 0.5 && visited.insert(target) {
            // ... proceed
        }
    }
}
```

**Effort:** ~5 lines added, but requires fetching the full edge (additional LMDB read per neighbor). The adjacency list only stores `(NodeId, EdgeType)`, not weights.

### CozoDB

```datalog
-- Just add a condition to the Datalog rules:
reachable[neighbor, depth] := *edges{source: start, target: neighbor, weight},
    weight > 0.5, depth = 1
```

**Effort:** ~1 condition per rule (4 rules to update). The change is local and obvious. Edge weight is already available in the relation without additional lookups.

### Verdict: Modification

**CozoDB wins.** Datalog's declarative filters are more natural for query modification. Adding conditions to Datalog rules is trivial. In HelixDB, structural changes to the adjacency list or additional lookups are needed.

---

## 7. Debugging Experience

### HelixDB

- **Rust debugger (lldb/gdb/IDE):** Full breakpoint support; inspect `visited`, `queue`, `parent` at any step
- **Logging:** `eprintln!()` at any point in the loop
- **Error messages:** Standard Rust `Result` with `anyhow` context
- **Stack traces:** Meaningful, point to exact line of failure
- **Tooling:** `cargo test`, `cargo clippy`, IDE autocomplete all work seamlessly

### CozoDB

- **Datalog debugging:** CozoDB provides no query debugger, explain plan, or step-through
- **Error messages:** Often cryptic; e.g., a missing comma in a rule produces a parser error that doesn't point to the correct location
- **String-building bugs:** Formatting errors (missing quotes, wrong escaping) are caught at runtime, not compile time
- **Data extraction:** `DataValue` enum requires manual pattern matching (`get_int()`, `get_str()`, etc.) — type mismatches are silent (return `None`)
- **Workaround:** `eprintln!("{}", script)` to print generated queries, then test them in CozoDB REPL

### Verdict: Debugging

**HelixDB wins decisively.** Compile-time type safety, standard Rust tooling, and the ability to step through traversal logic make debugging straightforward. CozoDB's runtime string generation and opaque Datalog engine make debugging significantly harder.

---

## 8. Summary Scorecard

| Criterion          | Weight | HelixDB | CozoDB | Notes |
|--------------------|--------|---------|--------|-------|
| Code conciseness   | 15%    | 7/10    | 7/10   | Similar total LoC; CozoDB has shorter queries but more glue code |
| Query clarity      | 20%    | 7/10    | 8/10   | Datalog's declarative style is more readable for relational queries |
| Ease of modification | 15%  | 6/10    | 9/10   | Adding filters/conditions to Datalog is trivial |
| Debugging          | 25%    | 9/10    | 4/10   | CozoDB has no debugger; string-built queries are error-prone |
| Type safety        | 15%    | 9/10    | 3/10   | CozoDB uses strings and runtime DataValue extraction |
| API ergonomics     | 10%    | 8/10    | 5/10   | Heed's typed API vs CozoDB's `run_script` with BTreeMap params |

**Weighted scores:**
- **HelixDB:** 0.15×7 + 0.20×7 + 0.15×6 + 0.25×9 + 0.15×9 + 0.10×8 = **7.70**
- **CozoDB:** 0.15×7 + 0.20×8 + 0.15×9 + 0.25×4 + 0.15×3 + 0.10×5 = **5.90**

---

## 9. Implementer's Subjective Opinion

HelixDB (Heed/LMDB) is the more ergonomic choice for this project. The main reasons:

1. **CozoDB's Datalog power is undercut by its Rust API.** The beauty of Datalog is in its declarative recursion — but the string-building, manual escaping, and `DataValue` extraction required by the Rust API eliminate most of that advantage. A first-class Rust Datalog DSL (compile-time query checking) would change this calculus entirely.

2. **Path finding exposed Datalog's limitations.** Recursive Datalog with `append()` caused combinatorial explosion on bidirectional graphs, forcing a hybrid Rust-BFS approach that was worse than either pure strategy. This is a fundamental limitation for graph problems where you need shortest-path semantics rather than reachability sets.

3. **Debugging CozoDB is painful.** When a query fails, you're debugging a string at runtime. When a `DataValue` extraction returns `None`, you get silent failures. These issues compound on complex queries.

4. **Where CozoDB shines:** Automatic HNSW index management, declarative schema, and filter ergonomics are genuine advantages. If the project were query-heavy with complex relational patterns (joins, aggregations, community detection), CozoDB's Datalog would be far more valuable.

**Recommendation:** Use HelixDB as the primary engine for Ucotron. Consider CozoDB as a reference implementation for complex analytical queries that benefit from Datalog's expressiveness, but don't use it as the production runtime.
