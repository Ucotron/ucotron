# Ucotron Pipelines

> Detailed documentation of the ingestion, retrieval, and consolidation data flows.

## Table of Contents

- [Ingestion Pipeline](#ingestion-pipeline)
- [Retrieval Pipeline](#retrieval-pipeline)
- [Consolidation Worker](#consolidation-worker)

---

## Ingestion Pipeline

The `IngestionOrchestrator` converts raw text into a knowledge graph through an 8-step pipeline. Each step is instrumented with OpenTelemetry tracing spans and records timing metrics in microseconds.

**Source:** `ucotron_extraction/src/ingestion.rs`

### Data Flow Diagram

```
                         ┌──────────────┐
                         │  Raw Text    │
                         └──────┬───────┘
                                │
                    ┌───────────▼───────────┐
                    │  Step 1: Chunking     │
                    │  Split into sentences │
                    └───────────┬───────────┘
                                │
                    ┌───────────▼───────────┐
                    │  Step 2: Embedding    │
                    │  384-dim vectors      │
                    └───────────┬───────────┘
                                │
               ┌────────────────┼────────────────┐
               │                │                 │
   ┌───────────▼──────────┐     │    ┌────────────▼───────────┐
   │  Step 3: NER         │     │    │  Step 4: Relations     │
   │  Entity extraction   │─────┼───▶│  Subject-Predicate-Obj │
   └───────────┬──────────┘     │    └────────────┬───────────┘
               │                │                 │
               └────────────────┼─────────────────┘
                                │
                    ┌───────────▼───────────┐
                    │  Step 5: Entity       │
                    │  Resolution & Dedup   │
                    └───────────┬───────────┘
                                │
                    ┌───────────▼───────────┐
                    │  Step 6: Contradiction│
                    │  Detection            │
                    └───────────┬───────────┘
                                │
                    ┌───────────▼───────────┐
                    │  Step 7: Graph Update │
                    │  Upsert nodes + edges │
                    └───────────┬───────────┘
                                │
                    ┌───────────▼───────────┐
                    │  Step 8: Vector Index │
                    │  Upsert embeddings    │
                    └──────────────────────┘
```

### Step Details

#### Step 1: Chunking

Splits input text into sentence-level chunks using rule-based splitting on `.`, `!`, `?` followed by whitespace or end-of-line.

- **Function:** `chunk_text(text)`
- **Span:** `ucotron.chunk`
- **Input:** Raw text string
- **Output:** `Vec<String>` of trimmed sentences
- **Example:** `"Juan lives in Madrid. He works at SAP."` becomes `["Juan lives in Madrid.", "He works at SAP."]`

#### Step 2: Embedding

Generates 384-dimensional normalized vectors for each chunk via the ONNX all-MiniLM-L6-v2 model. Processes in sub-batches for memory control.

- **Trait:** `EmbeddingPipeline::embed_batch(&[&str])`
- **Span:** `ucotron.embed`
- **Batch size:** Configurable via `embedding_batch_size` (default: 32)
- **Failure mode:** Hard failure -- embedding is required for vector search, so the entire pipeline aborts on error.

#### Step 3: Named Entity Recognition (NER)

Extracts named entities from each chunk using GLiNER (zero-shot ONNX). Processes in batches with per-chunk fallback on batch failure.

- **Trait:** `NerPipeline::extract_entities_batch(&[&str], &[&str])`
- **Span:** `ucotron.ner`
- **Labels:** Configurable via `ner_labels` (default: `["person", "location", "organization", "date", "concept"]`)
- **Batch size:** Configurable via `ner_batch_size` (default: 8)
- **Failure mode:** Soft -- failures are logged, pipeline continues with empty entities.
- **Output per entity:** `{ text, label, start, end, confidence }`

#### Step 4: Relation Extraction

Infers semantic relations between entities within each chunk. Only runs on chunks with 2+ entities.

- **Trait:** `RelationExtractor::extract_relations(&str, &[ExtractedEntity])`
- **Span:** `ucotron.relations`
- **Requires:** Step 3 output (entity list per chunk)
- **Failure mode:** Soft -- failures are logged, empty Vec returned for failed chunks.
- **Output per relation:** `{ subject, predicate, object, confidence }`
- **Example:** `("Juan", "lives_in", "Madrid", 0.85)`

#### Step 5: Entity Resolution

Deduplicates entities across chunks and merges with existing graph nodes.

- **Function:** `resolve_and_create_entities()`
- **Span:** `ucotron.entity_resolution`
- **Algorithm:**
  1. **Intra-batch dedup:** Normalize names (lowercase, trim), keep highest confidence per unique name.
  2. **Graph resolution:** For each entity, embed its text, search the vector backend for top-5 similar nodes, filter candidates by `NodeType::Entity` and `entity_resolution_threshold`.
  3. **Structural similarity:** Confirm matches using `structural_similarity()` (0.6 Jaccard + 0.4 cosine on neighbors).
  4. **Decision:** If match found, reuse existing node ID (merge). Otherwise, allocate a new node ID.
- **Failure mode:** Soft -- disabled via `enable_entity_resolution=false`.

#### Step 6: Contradiction Detection

Detects conflicting facts and resolves them using temporal and confidence rules.

- **Function:** `detect_contradictions()`
- **Span:** `ucotron.contradiction_detection`
- **Algorithm:**
  1. Convert each relation to a `Fact(subject_node_id, predicate, object, confidence, timestamp)`.
  2. For each new fact, call `detect_conflict(&new_fact, &known_facts, &conflict_config)`.
  3. If conflict found, call `resolve_conflict()` to determine winner.
  4. Build `CONFLICTS_WITH` + `SUPERSEDES` edges.
- **Resolution rules (ordered):**
  1. **Temporal:** If timestamps differ by > `temporal_threshold_secs` (default: 1 year), newer wins.
  2. **Confidence:** If timestamps are close but confidence diff > `confidence_threshold` (default: 0.3), higher wins.
  3. **Ambiguous:** Both conditions fail -- marked as `Contradiction` for human review.
- **Failure mode:** Soft -- disabled via `enable_contradiction_detection=false`.

#### Step 7: Graph Update

Upserts all nodes and edges to the graph backend in a single batch.

- **Span:** `ucotron.graph_update`
- **Node types created:**
  - **Chunk nodes** (`NodeType::Event`): One per sentence, stores raw text + embedding. Metadata: `{ chunk_index, source_type: "ingestion" }`.
  - **Entity nodes** (`NodeType::Entity`): Only newly-created entities (merged entities reuse existing nodes).
- **Edge types created:**
  - **Relation edges:** Entity-to-Entity from Step 4 relations, with `edge_type` mapped from predicate.
  - **Chunk-Entity links:** Chunk-to-Entity (`EdgeType::RelatesTo`), weighted by entity confidence.
  - **Conflict edges:** `CONFLICTS_WITH` + `SUPERSEDES` from Step 6.
- **Backend calls:** `graph().upsert_nodes()` + `graph().upsert_edges()`

#### Step 8: Vector Index Update

Upserts all node embeddings to the HNSW vector index.

- **Backend call:** `vector().upsert_embeddings(&[(node_id, embedding)])`
- **Includes:** Both chunk node and entity node embeddings.

### Ingestion Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ner_labels` | `Vec<String>` | `["person", "location", "organization", "date", "concept"]` | Zero-shot NER labels |
| `enable_relations` | `bool` | `true` | Toggle relation extraction |
| `enable_entity_resolution` | `bool` | `true` | Toggle entity dedup against graph |
| `enable_contradiction_detection` | `bool` | `true` | Toggle conflict detection |
| `entity_resolution_threshold` | `f32` | `0.5` | Cosine similarity threshold for merge |
| `conflict_config.temporal_threshold_secs` | `u64` | `31,557,600` (1 year) | Max age diff for temporal rule |
| `conflict_config.confidence_threshold` | `f32` | `0.3` | Min confidence diff for confidence rule |
| `next_node_id` | `Option<u64>` | `None` (starts at 1,000,000) | Starting ID for new nodes |
| `ner_batch_size` | `usize` | `8` | Chunks per NER batch call |
| `embedding_batch_size` | `usize` | `32` | Sub-batch size for embeddings |

### Ingestion Metrics

The pipeline records 17 metrics per invocation:

| Metric | Type | Description |
|--------|------|-------------|
| `chunks_processed` | count | Sentences extracted |
| `chunks_failed` | count | NER failures (fallback) |
| `entities_extracted` | count | Named entities found |
| `relations_extracted` | count | Relations inferred |
| `entity_nodes_created` | count | New entity nodes |
| `entity_merges` | count | Entities merged with graph |
| `edges_created` | count | Total edges persisted |
| `contradictions_detected` | count | Fact conflicts found |
| `chunking_us` | timing | Step 1 duration |
| `embedding_us` | timing | Step 2 duration |
| `ner_us` | timing | Step 3 duration |
| `relation_extraction_us` | timing | Step 4 duration |
| `entity_resolution_us` | timing | Step 5 duration |
| `contradiction_detection_us` | timing | Step 6 duration |
| `graph_update_us` | timing | Steps 7-8 duration |
| `total_us` | timing | End-to-end duration |

---

## Retrieval Pipeline

The `RetrievalOrchestrator` implements a LazyGraphRAG variant that blends vector similarity, graph structure, temporal recency, and cognitive mindset scoring. Results are assembled into context text for LLM augmentation.

**Source:** `ucotron_extraction/src/retrieval.rs`

### Data Flow Diagram

```
                         ┌──────────────┐
                         │  Query Text  │
                         └──────┬───────┘
                                │
                    ┌───────────▼───────────┐
                    │  Step 0: Mindset      │
                    │  Auto-detection       │
                    └───────────┬───────────┘
                                │
                    ┌───────────▼───────────┐
                    │  Step 1: Query        │
                    │  Embedding (384-dim)  │
                    └───────────┬───────────┘
                                │
               ┌────────────────┼────────────────┐
               │                                  │
   ┌───────────▼──────────┐          ┌────────────▼───────────┐
   │  Step 2: Vector      │          │  Step 3: Entity        │
   │  Search (HNSW)       │          │  Extraction from Query │
   └───────────┬──────────┘          └────────────┬───────────┘
               │                                  │
               └────────────────┬─────────────────┘
                                │ seed nodes
                    ┌───────────▼───────────┐
                    │  Step 4: Graph        │
                    │  Expansion (N-hop)    │
                    └───────────┬───────────┘
                                │
                    ┌───────────▼───────────┐
                    │  Step 5: Community    │
                    │  Selection (Leiden)   │
                    └───────────┬───────────┘
                                │
                    ┌───────────▼───────────┐
                    │  Step 6: Re-ranking   │
                    │  Vector+Graph+Recency │
                    └───────────┬───────────┘
                                │
                    ┌───────────▼───────────┐
                    │  Steps 7-8: Context   │
                    │  Assembly (Markdown)  │
                    └──────────────────────┘
```

### Step Details

#### Step 0: Mindset Auto-Detection (Optional)

If `query_mindset` is not explicitly set but a `MindsetDetector` is configured, scans the query for keyword patterns to auto-detect one of: Convergent, Divergent, or Algorithmic.

- **Function:** `MindsetDetector::detect(query)`
- **Output:** `Option<MindsetTag>`

#### Step 1: Query Embedding

Converts the query text into a 384-dimensional vector using the same embedding model as ingestion.

- **Trait:** `EmbeddingPipeline::embed_text(query)`
- **Span:** `ucotron.query_embed`

#### Step 2: Vector Search

Performs HNSW approximate nearest-neighbor search to find the top-K most semantically similar nodes.

- **Trait:** `VectorBackend::search(&embedding, top_k)`
- **Span:** `ucotron.vector_search`
- **Post-filter:** Results below `min_similarity` threshold are discarded.
- **Output:** `HashMap<NodeId, f32>` mapping node IDs to similarity scores.

#### Step 3: Entity Extraction from Query

Runs NER on the query to identify entities, then searches the graph for matching entity nodes.

- **Span:** `ucotron.query_ner`
- **Algorithm:**
  1. Run GLiNER NER on query text.
  2. For each extracted entity, embed the entity text.
  3. Search vector backend for top-5 similar nodes.
  4. Filter: must be `NodeType::Entity` and match by name containment or similarity > 0.7.
- **Output:** `Vec<Node>` of matched entity nodes + their IDs.

#### Step 4: Graph Expansion

Expands the seed set (vector results + entity matches) by traversing N-hop neighbors.

- **Trait:** `GraphBackend::get_neighbors(id, hops)`
- **Span:** `ucotron.graph_traverse`
- **Similarity propagation:** Neighbor similarity decays exponentially: `seed_sim * 0.5^hops`
- **Output:** `HashMap<NodeId, Node>` of all expanded nodes with their decayed similarities.

#### Step 5: Community Selection

For each seed node, fetches its Leiden community members and adds nodes not already in the expansion set.

- **Trait:** `GraphBackend::get_community(node_id)`
- **Span:** `ucotron.community`
- **Limit:** Up to `max_community_members` (default: 20) new nodes per seed.
- **Default similarity:** Community-sourced nodes get similarity score of `0.1`.

#### Step 6: Re-ranking

Computes a weighted composite score for each candidate node, applying filters and temporal decay.

- **Span:** `ucotron.rerank`
- **Filters applied:**
  - `time_range`: Exclude nodes outside `(min_timestamp, max_timestamp)`.
  - `entity_type_filter`: Keep only nodes matching a specific `NodeType`.
- **Scoring components:**
  - **Vector similarity** (weight: 0.5): From HNSW search or decay propagation.
  - **Graph centrality** (weight: 0.3): `num_neighbors / max_degree` across all candidates.
  - **Recency** (weight: 0.2): Exponential decay `0.5^(age_secs / half_life)`.
- **Mindset scoring (optional):** If `query_mindset` is set, computes a `mindset_score` and blends: `final = base * 0.85 + mindset * 0.15`.
- **Output:** `Vec<ScoredMemory>` sorted by score descending, truncated to `final_top_k`.

#### Steps 7-8: Context Assembly

Formats scored memories and matched entities into markdown-style context text for LLM consumption.

- **Function:** `assemble_context(&memories, &entities)`
- **Output format:**
  ```
  ## Relevant Memories
  [1] (score: 0.85) "Memory content here..."

  ## Known Entities
  - EntityName (type: Entity)
  ```

### Retrieval Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `vector_top_k` | `usize` | `50` | Initial vector search candidates |
| `min_similarity` | `f32` | `0.0` | Minimum similarity threshold |
| `graph_expansion_hops` | `u8` | `1` | N-hop neighbor traversal depth |
| `enable_entity_extraction` | `bool` | `true` | Toggle query NER |
| `ner_labels` | `Vec<String>` | `["person", "location", ...]` | NER labels for query entities |
| `enable_community_expansion` | `bool` | `true` | Toggle Leiden community expansion |
| `max_community_members` | `usize` | `20` | Max community nodes per seed |
| `vector_sim_weight` | `f32` | `0.5` | Re-ranking weight for vector sim |
| `graph_centrality_weight` | `f32` | `0.3` | Re-ranking weight for centrality |
| `recency_weight` | `f32` | `0.2` | Re-ranking weight for recency |
| `temporal_decay_half_life_secs` | `u64` | `2,592,000` (30 days) | Half-life for temporal decay |
| `namespace` | `Option<String>` | `None` | Namespace filter |
| `time_range` | `Option<(u64, u64)>` | `None` | Timestamp range filter |
| `entity_type_filter` | `Option<NodeType>` | `None` | Node type filter |
| `query_mindset` | `Option<MindsetTag>` | `None` | Cognitive mindset hint |
| `final_top_k` | `usize` | `10` | Final result count |

### Retrieval Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `query_embedding_us` | timing | Step 1 duration |
| `vector_search_us` | timing | Step 2 duration |
| `entity_extraction_us` | timing | Step 3 duration |
| `graph_expansion_us` | timing | Step 4 duration |
| `community_selection_us` | timing | Step 5 duration |
| `reranking_us` | timing | Step 6 duration |
| `context_assembly_us` | timing | Steps 7-8 duration |
| `total_us` | timing | End-to-end duration |

---

## Consolidation Worker

The consolidation subsystem runs as a background task that maintains graph health through three phases: community re-detection, entity deduplication, and memory decay. It uses a two-tier architecture: a synchronous `ConsolidationOrchestrator` for single-cycle logic and an async `ConsolidationWorker` that runs the orchestrator on a timer.

**Source:** `ucotron_extraction/src/consolidation.rs`

### Data Flow Diagram

```
                    ┌───────────────────────────┐
                    │  ConsolidationWorker      │
                    │  (tokio background task)  │
                    │                           │
                    │  ┌─ interval timer ──┐    │
                    │  │  tick every N sec │    │  ◄── shutdown_rx
                    │  └────────┬──────────┘    │      (watch channel)
                    └───────────┼────────────────┘
                                │
                    ┌───────────▼───────────────┐
                    │  ConsolidationOrchestrator │
                    │  consolidate()             │
                    └───────────┬───────────────┘
                                │
          ┌─────────────────────┼─────────────────────┐
          │                     │                      │
┌─────────▼──────────┐ ┌───────▼────────────┐ ┌───────▼─────────┐
│  Phase 1:          │ │  Phase 2:          │ │  Phase 3:       │
│  Community         │ │  Entity Merge      │ │  Memory Decay   │
│  Detection         │ │  (Dedup)           │ │  (Exponential)  │
│  (Leiden)          │ │                    │ │                 │
└────────────────────┘ └────────────────────┘ └─────────────────┘
```

### Phase Details

#### Phase 1: Community Detection (Incremental Leiden)

Re-runs the Leiden community detection algorithm on the full edge set, using previous community assignments for incremental optimization.

- **Function:** `run_community_detection()`
- **Algorithm:**
  1. Fetch all edges via `graph().get_all_edges()`.
  2. Call `detect_communities_incremental()` with previous community result.
  3. Persist updated assignments via `graph().store_community_assignments()`.
- **Incremental support:** Previous `CommunityResult` is stored between cycles, so only changed nodes are recomputed.
- **Metrics:** `communities_detected`, `communities_changed_nodes`

#### Phase 2: Entity Merge

Finds and merges duplicate entity nodes based on name similarity and embedding cosine similarity.

- **Function:** `run_entity_merge()`
- **Algorithm:**
  1. Collect all node IDs from edges.
  2. Filter to `NodeType::Entity` nodes with non-empty embeddings.
  3. Group by normalized name (lowercase, trimmed).
  4. Within each group, compute pairwise cosine similarity.
  5. If similarity >= `entity_merge_threshold` (default: 0.8), merge: redirect edges from duplicate to survivor (lower ID wins).
- **Scope:** Only `Entity` nodes are eligible -- `Event` nodes are never merged.
- **Metrics:** `entity_duplicates_found`, `entity_merges_performed`

#### Phase 3: Memory Decay

Applies exponential temporal decay to node metadata, enabling the retrieval pipeline to deprioritize stale memories.

- **Function:** `run_memory_decay()`
- **Decay formula:** `decay_factor = 0.5 ^ (age_secs / half_life_secs)`
- **Algorithm:**
  1. Compute current timestamp (or use `current_time` config for testing).
  2. Fetch all nodes from edges.
  3. For each node with a valid timestamp:
     - Skip if `timestamp == 0` or `timestamp > now`.
     - Compute decay factor.
     - Skip if `decay_factor >= 0.99` (recent nodes, avoids unnecessary writes).
     - Store `decay_factor` in `node.metadata["decay_factor"]`.
  4. Batch upsert all decayed nodes.
- **Metrics:** `nodes_decayed`

### Error Handling

Each phase is independently fault-tolerant:

- Phase failures are logged with `warn!` but do **not** abort the consolidation cycle.
- If Phase 1 fails, Phases 2 and 3 still run.
- This ensures maximum graph maintenance even under partial failure.

### Graceful Shutdown

The `ConsolidationWorker` uses a `tokio::sync::watch` channel for shutdown coordination:

1. `spawn()` returns a `watch::Sender<bool>`.
2. The run loop uses `tokio::select!` to wait on either a timer tick or a shutdown signal.
3. On shutdown signal (`true`), the current cycle completes before the loop exits.
4. No abrupt termination -- all in-flight phases finish naturally.

```rust
let shutdown_tx = worker.spawn(Arc::clone(&registry));
// ... later ...
shutdown_tx.send(true).unwrap(); // graceful shutdown
```

### Consolidation Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_community_detection` | `bool` | `true` | Toggle Leiden re-detection |
| `enable_entity_merge` | `bool` | `true` | Toggle duplicate entity merge |
| `enable_decay` | `bool` | `true` | Toggle memory decay |
| `decay_halflife_secs` | `u64` | `2,592,000` (30 days) | Half-life for exponential decay |
| `entity_merge_threshold` | `f32` | `0.8` | Cosine similarity threshold for merge |
| `community_config` | `CommunityConfig` | Leiden defaults | Algorithm parameters |
| `current_time` | `Option<u64>` | `None` (uses system time) | Override for testing |
| `trigger_interval` | `usize` | `100` | Messages between worker runs (config file) |

### Consolidation Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `communities_detected` | count | Leiden communities found |
| `communities_changed_nodes` | count | Nodes with updated assignments |
| `entity_duplicates_found` | count | Duplicate entity pairs |
| `entity_merges_performed` | count | Merges executed |
| `nodes_decayed` | count | Nodes with decay applied |
| `community_detection_us` | timing | Phase 1 duration |
| `entity_merge_us` | timing | Phase 2 duration |
| `decay_us` | timing | Phase 3 duration |
| `total_us` | timing | Full cycle duration |
