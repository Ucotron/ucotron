# Ucotron Architecture

> Cognitive memory framework for LLMs — system architecture deep-dive.

## Table of Contents

- [Workspace Structure](#workspace-structure)
- [Crate Responsibilities](#crate-responsibilities)
- [Dependency Graph](#dependency-graph)
- [Core Data Model](#core-data-model)
- [Backend Trait Hierarchy](#backend-trait-hierarchy)
- [Data Flow: Ingestion Pipeline](#data-flow-ingestion-pipeline)
- [Data Flow: Retrieval Pipeline](#data-flow-retrieval-pipeline)
- [Data Flow: Consolidation Worker](#data-flow-consolidation-worker)
- [Multimodal Architecture](#multimodal-architecture)
- [Server Architecture](#server-architecture)
- [Storage Engine (HelixDB / LMDB)](#storage-engine-helixdb--lmdb)
- [ML Pipeline (ONNX)](#ml-pipeline-onnx)
- [Configuration System](#configuration-system)
- [Key Design Decisions](#key-design-decisions)

---

## Workspace Structure

```
memory_arena/
├── Cargo.toml              # Workspace root (LTO, 3 codegen-units)
├── Cargo.lock
├── Dockerfile              # Multi-stage build (rust → model-downloader → runtime)
├── docker-compose.yml      # Single-instance dev setup
├── ucotron.toml            # Default configuration
│
├── core/                   # ucotron-core
├── helix_impl/             # ucotron-helix
├── ucotron_extraction/     # ML extraction pipelines
├── ucotron_server/         # REST API + MCP server
├── ucotron_sdk/            # Rust client SDK
├── ucotron_config/         # TOML configuration
├── ucotron_connectors/     # External connectors (Slack, GitHub, Notion, etc.)
├── bench_runner/           # Phase 1 benchmark CLI
│
├── sdks/
│   ├── typescript/         # @ucotron/sdk (npm)
│   ├── python/             # ucotron-sdk (PyPI)
│   ├── go/                 # ucotron-go (Go module)
│   ├── java/               # JVM + Android SDK
│   └── php/                # Packagist SDK
│
├── models/                 # ONNX models (gitignored)
│   ├── all-MiniLM-L6-v2/  # Embedding (384-dim)
│   ├── gliner_small-v2.1/  # Zero-shot NER
│   ├── clip-vit-base-patch32/  # Visual embedding (512-dim)
│   ├── whisper-tiny/       # Speech-to-text
│   └── projection_layer.onnx  # CLIP→MiniLM bridge
│
├── scripts/                # Model downloads, dataset generation, training
├── templates/              # Framework integration templates (14 total)
├── deploy/                 # Helm charts, Grafana dashboards, Terraform
└── _archive/
    └── cozo_impl/          # Archived CozoDB (Phase 1 only, rejected)
```

---

## Crate Responsibilities

| Crate | Purpose | Key Exports |
|-------|---------|-------------|
| **core** | Shared traits, types, algorithms | `Node`, `Edge`, `BackendRegistry`, `VectorBackend`, `GraphBackend`, `QueryBuilder` |
| **helix_impl** | LMDB storage implementation | `HelixEngine`, `HelixGraphBackend`, `HnswVectorBackend` |
| **ucotron_extraction** | ML pipelines (embedding, NER, relations, audio, image, video) | `IngestionOrchestrator`, `RetrievalOrchestrator`, `ConsolidationOrchestrator` |
| **ucotron_server** | HTTP REST API (Axum) + MCP server (rmcp) | 2 binaries: `ucotron_server`, `ucotron_mcp` |
| **ucotron_sdk** | Rust client library | `UcotronClient` (12 async + 7 sync methods) |
| **ucotron_config** | TOML configuration with env overrides | `UcotronConfig`, `StorageConfig`, `ModelsConfig` |
| **ucotron_connectors** | External data source connectors | Slack, GitHub, Notion, Discord, Google Drive connectors |
| **bench_runner** | Phase 1 benchmark CLI (Clap 4) | `ingest`, `search`, `recursion` subcommands |

---

## Dependency Graph

```
                    ┌──────────────┐
                    │  ucotron_sdk │  (standalone client)
                    └──────────────┘

┌──────────────────────────────────────────────────────────┐
│                    ucotron_server                         │
│  (axum, rmcp, tower, tokio, opentelemetry, prometheus)   │
└────────────┬──────────────┬───────────────┬──────────────┘
             │              │               │
             ▼              ▼               ▼
    ┌────────────────┐  ┌──────────┐  ┌──────────────┐
    │ ucotron_extract │  │helix_impl│  │ucotron_config│
    │ (ort, tokenizers│  │(heed,    │  │(toml, serde) │
    │  ndarray, hound │  │ instant- │  └──────┬───────┘
    │  image, ffmpeg) │  │ distance,│         │
    └────────┬───────┘  │ graphrs) │         │
             │          └────┬─────┘         │
             │               │               │
             ▼               ▼               ▼
         ┌───────────────────────────────────────┐
         │               core                     │
         │  (serde, bincode v1, graphrs types)    │
         └───────────────────────────────────────┘
```

**No circular dependencies.** The core crate sits at the bottom, depending only on serialization and math libraries. All other crates depend on core for shared types and traits.

---

## Core Data Model

### Node

```rust
struct Node {
    id: NodeId,                              // u64, sequential
    content: String,                         // 50-200 chars text
    embedding: Vec<f32>,                     // 384-dim (MiniLM L2-normalized)
    metadata: HashMap<String, Value>,        // Extensible key-value
    node_type: NodeType,                     // Entity | Event | Fact | Skill
    timestamp: u64,                          // Unix timestamp

    // Multimodal fields (all Option)
    media_type: Option<MediaType>,           // Text | Audio | Image | VideoSegment
    media_uri: Option<String>,               // Relative path to media file
    embedding_visual: Option<Vec<f32>>,      // 512-dim (CLIP space)
    timestamp_range: Option<(u64, u64)>,     // Segment boundaries (ms)
    parent_video_id: Option<NodeId>,         // Links video segments to parent
}
```

### Edge

```rust
struct Edge {
    source: NodeId,
    target: NodeId,
    edge_type: EdgeType,    // RelatesTo, CausedBy, ConflictsWith, Supersedes,
                            // Actor, Object, Location, Companion, ...
    weight: f32,            // 0.1–1.0
    metadata: HashMap<String, Value>,
}
```

### Cognitive Extensions

| Type | Purpose |
|------|---------|
| `Fact` | Subject-predicate-object triple with confidence and resolution state |
| `MindsetTag` | Convergent / Divergent / Algorithmic (Chain of Mindset) |
| `ResolutionState` | Accepted / Contradiction / Superseded / Ambiguous |
| `Agent` | Multi-agent identity with namespace isolation |
| `AgentShare` | Cross-agent graph sharing with ReadOnly / ReadWrite permissions |

---

## Backend Trait Hierarchy

The pluggable backend system separates vector search from graph operations:

```
┌─────────────────────┐   ┌─────────────────────┐   ┌──────────────────────────┐
│   VectorBackend     │   │    GraphBackend      │   │  VisualVectorBackend     │
│                     │   │                      │   │  (optional)              │
│ upsert_embeddings() │   │ upsert_nodes()       │   │ upsert_visual_embed..() │
│ search()            │   │ upsert_edges()       │   │ search_visual()          │
│ delete()            │   │ get_node()           │   │ delete_visual()          │
│                     │   │ get_neighbors()      │   └──────────────────────────┘
│                     │   │ find_path()          │
│                     │   │ get_community()      │
│                     │   │ create_agent()       │
│                     │   │ clone_graph()        │
│                     │   │ merge_graph()        │
└────────┬────────────┘   └──────────┬───────────┘
         │                           │
         ▼                           ▼
┌────────────────────────────────────────────────────┐
│              BackendRegistry                        │
│                                                     │
│  vector: Box<dyn VectorBackend>                     │
│  graph:  Box<dyn GraphBackend>                      │
│  visual: Option<Box<dyn VisualVectorBackend>>       │
└─────────────────────────────────────────────────────┘
```

**Implementations:**
- `HnswVectorBackend` — HNSW index via instant-distance, persisted in LMDB
- `HelixVectorBackend` — Brute-force SIMD cosine similarity (Phase 1 fallback)
- `HelixGraphBackend` — LMDB adjacency lists with 7+ named databases
- `ExternalVectorBackend` / `ExternalGraphBackend` — Stubs for Qdrant / FalkorDB

---

## Data Flow: Ingestion Pipeline

The `IngestionOrchestrator` processes text through 8 stages:

```
User Input (text: String)
    │
    ▼
┌───────────────────┐
│ 1. Chunking       │  Split text into sentences
└───────┬───────────┘
        │ chunks: Vec<String>
        ▼
┌───────────────────┐
│ 2. Embedding      │  OnnxEmbeddingPipeline.embed_batch()
└───────┬───────────┘  (batched, 384-dim per chunk)
        │
        ▼
┌───────────────────┐
│ 3. NER            │  GlinerNerPipeline.extract_entities_batch()
└───────┬───────────┘  (zero-shot, configurable labels)
        │
        ▼
┌───────────────────┐
│ 4. Relations      │  CooccurrenceRelationExtractor
└───────┬───────────┘  (or LLM-based via feature flag)
        │
        ▼
┌───────────────────┐
│ 5. Entity Resol.  │  Jaccard neighborhood (0.6) + cosine (0.4)
└───────┬───────────┘  Merge with existing entities or create new
        │
        ▼
┌───────────────────┐
│ 6. Contradiction  │  Temporal + confidence conflict detection
└───────┬───────────┘  Creates CONFLICTS_WITH edges
        │
        ▼
┌───────────────────┐
│ 7. Graph Update   │  BackendRegistry
│                   │  ├─ vector.upsert_embeddings()
│                   │  ├─ graph.upsert_nodes()
│                   │  └─ graph.upsert_edges()
└───────┬───────────┘
        │
        ▼
┌───────────────────┐
│ 8. Store Chunks   │  Persist raw text for retrieval
└───────────────────┘
        │
        ▼
  IngestionResult { metrics, chunk_node_ids, entity_node_ids, edges_created }
```

---

## Data Flow: Retrieval Pipeline

The `RetrievalOrchestrator` implements an 8-step LazyGraphRAG pipeline:

```
User Query (query: String)
    │
    ▼
┌───────────────────┐
│ 1. Query Embed    │  OnnxEmbeddingPipeline.embed_text() → 384-dim
└───────┬───────────┘
        │
        ▼
┌───────────────────┐
│ 2. Vector Search  │  HNSW top-N for semantic similarity
└───────┬───────────┘
        │
        ▼
┌───────────────────┐
│ 3. Entity Extract │  GLiNER NER on query text
└───────┬───────────┘  (find entities mentioned in the query)
        │
        ▼
┌───────────────────┐
│ 4. Graph Expand   │  1-hop BFS from matched nodes
└───────┬───────────┘  Score decay: score *= decay^hop
        │
        ▼
┌───────────────────┐
│ 5. Community Sel. │  Leiden clusters containing matched nodes
└───────┬───────────┘  (max K members per community)
        │
        ▼
┌───────────────────┐
│ 6. Re-ranking     │  Combined score:
│                   │    vector_sim × 0.5
│                   │  + centrality × 0.3 (degree-based)
│                   │  + recency   × 0.2 (timestamp-based)
└───────┬───────────┘
        │
        ▼
┌───────────────────┐
│ 7. Temporal Decay │  decay = 0.5 ^ ((now - last_access) / half_life)
└───────┬───────────┘
        │
        ▼
┌───────────────────┐
│ 8. Context Assem. │  Format ranked memories as structured text
└───────────────────┘  for LLM context injection
        │
        ▼
  RetrievalResult { memories: Vec<ScoredMemory>, entities, context_text }
```

---

## Data Flow: Consolidation Worker

A background async tokio task that runs periodically:

```
┌──────────────────────────────────────────────────────┐
│  ConsolidationWorker (tokio::spawn, watch shutdown)  │
│                                                      │
│  Every trigger_interval:                             │
│                                                      │
│  ┌─────────────────────────────┐                     │
│  │ 1. Leiden Community Detect. │                     │
│  │    get_all_nodes/edges()    │                     │
│  │    detect_communities()     │                     │
│  │    store_assignments()      │                     │
│  └─────────────┬───────────────┘                     │
│                ▼                                     │
│  ┌─────────────────────────────┐                     │
│  │ 2. Entity Merge             │                     │
│  │    Find duplicates by       │                     │
│  │    name + embedding sim     │                     │
│  │    Redirect edges, delete   │                     │
│  └─────────────┬───────────────┘                     │
│                ▼                                     │
│  ┌─────────────────────────────┐                     │
│  │ 3. Memory Decay             │                     │
│  │    decay = 0.5^(age/half)   │                     │
│  │    Update metadata          │                     │
│  └─────────────────────────────┘                     │
│                                                      │
│  Graceful shutdown via watch::channel                │
└──────────────────────────────────────────────────────┘
```

---

## Multimodal Architecture

Dual-index design with cross-modal search:

```
┌──────────────────────────────────┐
│          Text Index              │
│  HNSW 384-dim (MiniLM space)    │
│                                  │
│  Contains:                       │
│  - Text node embeddings          │
│  - Audio transcription embeddings│
│  - Projected image embeddings    │
└──────────────────────────────────┘

┌──────────────────────────────────┐
│        Visual Index              │
│  HNSW 512-dim (CLIP space)      │
│                                  │
│  Contains:                       │
│  - Image embeddings              │
│  - Video keyframe embeddings     │
│  - CLIP text query embeddings    │
└──────────────────────────────────┘

Cross-Modal Bridge:
  ProjectionLayer (MLP: 512 → 1024 → 512 → 384)
  - image → text: CLIP visual → ProjectionLayer → Text Index
  - text → image: CLIP text encoder → Visual Index
  - audio → text: Whisper STT → MiniLM embed → Text Index
  - video: dual search (keyframes in Visual + transcript in Text)
```

### Processing by Media Type

| Input | Pipeline | Output |
|-------|----------|--------|
| Text | MiniLM embed | 384-dim → Text Index |
| Audio | Whisper STT → MiniLM embed | 384-dim → Text Index |
| Image | CLIP image encoder | 512-dim → Visual Index |
| Video | FFmpeg frames → CLIP + Whisper → MiniLM | Both indices |

---

## Server Architecture

### Binaries

| Binary | Transport | Purpose |
|--------|-----------|---------|
| `ucotron_server` | HTTP (port 8420) | REST API with Axum |
| `ucotron_mcp` | stdio | MCP server with rmcp 0.15 |

### REST API Endpoints

**Core Operations:**
- `POST /api/v1/memories` — Ingest text (runs full ingestion pipeline)
- `POST /api/v1/memories/search` — Semantic vector search
- `POST /api/v1/augment` — Full retrieval pipeline (LazyGraphRAG)
- `POST /api/v1/learn` — Convenience wrapper for ingestion

**CRUD:**
- `GET/PUT/DELETE /api/v1/memories/{id}` — Memory management
- `GET /api/v1/entities` / `GET /api/v1/entities/{id}` — Entity browsing

**Multimodal:**
- `POST /api/v1/memories/audio` — Audio ingestion (Whisper + embed)
- `POST /api/v1/memories/image` — Image ingestion (CLIP + embed)
- `POST /api/v1/memories/video` — Video ingestion (FFmpeg + CLIP + Whisper)
- `POST /api/v1/search/multimodal` — Cross-modal search
- `GET /api/v1/media/{id}` — Serve stored media files (Range support)

**Agents:**
- `POST /api/v1/agents` — Create agent with isolated namespace
- `POST /api/v1/agents/{id}/clone` — Clone agent graph
- `POST /api/v1/agents/{id}/merge` — Merge from another agent
- `POST /api/v1/agents/{id}/shares` — Share graph access

**Admin:**
- `GET /api/v1/health` / `GET /api/v1/metrics` — Observability
- `POST /api/v1/auth/api-keys` — API key management
- `GET /api/v1/audit` / `GET /api/v1/audit/export` — Audit log access
- `POST /api/v1/import`, `/api/v1/import/mem0`, `/api/v1/import/zep` — Data import

### MCP Tools (6)

| Tool | Description |
|------|-------------|
| `input_memory` | Ingest text through pipeline |
| `search` | Retrieval with ranked results |
| `augment` | Full LazyGraphRAG context generation |
| `list_entities` | Browse entity nodes |
| `get_graph` | Export as JSON-LD or Graphviz DOT |
| `consolidate` | Trigger background consolidation |

### Middleware Stack

```
Request → CORS → Tracing (OpenTelemetry spans) → Request Counter
        → Auth (API key validation, RBAC check, namespace extraction)
        → Audit Logging (state-mutating operations)
        → Handler → Response
```

### Authentication & RBAC

4 roles with escalating permissions:

| Role | Read | Write | Admin | Audit |
|------|------|-------|-------|-------|
| Viewer | Yes | No | No | No |
| Editor | Yes | Yes | No | No |
| Admin | Yes | Yes | Yes | Yes |
| ApiClient | Yes | Yes | No | No |

Multi-tenancy via `X-Ucotron-Namespace` header. Each namespace has isolated graph data.

---

## Storage Engine (HelixDB / LMDB)

### Database Schema

| LMDB Database | Key Type | Value Type | Purpose |
|---------------|----------|------------|---------|
| `nodes` | `u64` | `Node` (bincode) | Primary node storage |
| `edges` | `(u64,u64,u32)` | `Edge` (bincode) | Primary edge storage |
| `adj_out` | `u64` | `Vec<(u64,u32)>` | Outgoing adjacency lists |
| `adj_in` | `u64` | `Vec<(u64,u32)>` | Incoming adjacency lists |
| `nodes_by_type` | `u8` | `Vec<u64>` | Secondary type index |
| `community_assign` | `u64` | `CommunityId` | Node → community mapping |
| `community_members` | `CommunityId` | `Vec<u64>` | Community → member nodes |
| `agents` | `String` | `Agent` | Agent records |
| `shares` | `(String,String)` | `AgentShare` | Agent sharing grants |

### Vector Index

**HNSW** via `instant-distance` crate:
- Parameters: `ef_construction=200`, `ef_search=100`, `max_m=16`
- Persisted in LMDB via bincode serialization
- **Rebuild-on-upsert**: entire index rebuilt after each batch insert (practical for up to 1M vectors, <1s rebuild)

**Brute-force SIMD** (Phase 1 fallback):
- 8-wide accumulator lanes for pipeline parallelism
- Auto-vectorized to NEON (ARM) or AVX (x86)
- O(n) scan with O(n log k) min-heap selection

### Performance (Phase 1 Benchmarks)

| Operation | HelixDB | Target | Status |
|-----------|---------|--------|--------|
| Write throughput | 168k nodes/s | >5k docs/s | Pass |
| Read latency (1-hop) | <1ms | <10ms | Pass |
| Read latency (2-hop) | <5ms | <50ms | Pass |
| Cold start | <50ms | <200ms | Pass |
| RAM (100k nodes) | ~150MB | <500MB | Pass |
| Hybrid search P95 | <5ms | <50ms | Pass |

---

## ML Pipeline (ONNX)

All models run via ONNX Runtime (`ort` crate) for portable, dependency-free inference:

| Model | Task | Dimensions | Crate |
|-------|------|------------|-------|
| all-MiniLM-L6-v2 | Text embedding | 384 | ort + tokenizers |
| gliner_small-v2.1 | Zero-shot NER | N/A | ort (custom GLiNER impl) |
| CLIP ViT-B/32 | Image/text visual embedding | 512 | ort + image |
| Whisper Tiny | Speech-to-text | N/A | ort + hound |
| Projection Layer | CLIP→MiniLM bridge | 512→384 | ort |

**Thread Safety:** `ort::Session::run()` requires `&mut Session`; all pipelines wrap sessions in `Mutex<Session>` for concurrent access.

---

## Configuration System

TOML-based configuration (`ucotron.toml`) with environment variable overrides:

```toml
[server]
host = "0.0.0.0"
port = 8420

[storage]
mode = "embedded"        # embedded | shared | external

[storage.vector]
backend = "helix"        # helix | qdrant | custom
hnsw.ef_construction = 200
hnsw.ef_search = 100

[storage.graph]
backend = "helix"        # helix | falkordb | custom

[models]
embedding_model = "all-MiniLM-L6-v2"
ner_enabled = true
image_embedding_enabled = false
audio_transcription_enabled = false

[consolidation]
enable_decay = true
decay_halflife_secs = 2592000   # 30 days

[auth]
api_key_enabled = true

[telemetry]
tracing_enabled = true
otlp_endpoint = "http://localhost:4317"
```

Every field is overridable via `UCOTRON_<SECTION>_<FIELD>` environment variables (e.g., `UCOTRON_SERVER_PORT=9000`).

---

## Key Design Decisions

### Why LMDB?

- **Zero-copy reads** — Memory-mapped I/O eliminates deserialization on read path
- **ACID transactions** — Crash-safe for embedded databases
- **Single-writer, multi-reader** — Safe concurrent access without locks on reads
- **Proven** — Used by Filecoin, OpenLDAP, many production systems
- **Embedded** — No separate server process; ideal for agent-local deployment

### Why ONNX Runtime?

- **Model portability** — Same `.onnx` files across Rust, Python, Node.js, Go
- **Inference only** — No training framework bloat in production binary
- **Hardware dispatch** — CPU/GPU acceleration transparent to application code
- **Open standard** — Not locked to PyTorch, TensorFlow, or any framework

### Why Dual-Index (Text + Visual)?

- **Different dimensionalities** — MiniLM (384-dim) vs CLIP (512-dim) require separate indices
- **Independent scaling** — Text-heavy workloads don't pay visual index overhead
- **Cross-modal bridge** — Trained MLP projection maps between embedding spaces

### Why Rebuild-on-Upsert for HNSW?

The `instant-distance` crate does not support incremental insertion. Rebuilding the entire index after each batch is acceptable because:
- Rebuild time is <1s for up to 1M vectors
- Ingestion is typically batch/periodic, not real-time streaming
- The alternative (custom HNSW with incremental insert) adds significant complexity

For the full evaluation of incremental HNSW alternatives (hnsw_rs, hnswlib-rs, usearch, faiss-rs) with benchmarks and migration plan, see [ADR-001](docs/adr/ADR-001-incremental-hnsw-insert.md). Decision: keep instant-distance short-term, migrate to `hnsw_rs` when dataset exceeds 500k vectors or real-time insert is required.

### Why Leiden over Louvain?

- **Better modularity** — Provably optimal community assignment
- **Deterministic** — Seeded for reproducibility
- **Incremental** — Can initialize from previous community assignments

### Why MCP (Model Context Protocol)?

- **LLM tool standard** — Claude and other LLMs can call Ucotron tools natively
- **Stdio transport** — Secure subprocess integration, no network exposure
- **Self-describing** — Tools declare their input/output schemas

### Why Bincode v1 (not v3)?

- **Positional encoding** — Fast, compact, no field names in serialized data
- **Caveat:** `#[serde(skip_serializing_if)]` must NOT be used on `Node` fields; bincode v1 relies on field order, and skipping fields breaks deserialization

### Why No LLM Calls During Indexing?

Following the **LazyGraphRAG** philosophy:
- Indexing uses only local models (embedding, NER, co-occurrence)
- LLM inference is deferred to query time (retrieval pipeline)
- This keeps indexing fast, deterministic, and cost-free
