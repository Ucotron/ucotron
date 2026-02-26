# Ucotron - Memory Arena

## Build & Run

```bash
cd memory_arena

# Build all crates
cargo build

# Run all tests
cargo test

# Run bench_runner CLI
cargo run --bin bench_runner -- <command>

# Build with release optimizations (for benchmarking)
cargo build --release
```

## Workspace Structure

- `core/` — Shared traits (`StorageEngine`, `VectorBackend`, `GraphBackend`), types (`Node`, `Edge`, `Config`), data generation, cognitive logic
- `helix_impl/` — HelixDB (Heed/LMDB) StorageEngine implementation
- `ucotron_config/` — TOML configuration system (`UcotronConfig`, `ServerConfig`, `StorageConfig`, `ModelsConfig`)
- `ucotron_extraction/` — Extraction pipeline traits (`EmbeddingPipeline`, `NerPipeline`, `RelationExtractor`)
- `ucotron_server/` — Axum HTTP server binary (El Hipocampo) with REST API
- `ucotron_sdk/` — Rust client SDK (`UcotronClient`, `AugmentResult`, `LearnResult`)
- `bench_runner/` — CLI binary for running benchmarks (clap 4.x, CozoDB gated behind `cozo` feature)
- `_archive/cozo_impl/` — Archived CozoDB implementation (NO-GO from Phase 1, available via `cargo test -p bench-runner --features cozo`)

## Patterns

- **StorageEngine trait**: All engine implementations must implement `ucotron_core::StorageEngine`. This enables engine-agnostic benchmarking and cognitive logic.
- **Serialization**: Use `bincode` for internal storage (fast, compact), `serde` traits for public interfaces.
- **Deterministic data**: All synthetic data generation uses `rand_chacha` with fixed seeds for reproducibility.
- **Embedding dimension**: All vectors are 384-dimensional `Vec<f32>` (sentence-transformers compatible).

## HelixDB (helix_impl) Patterns

- **Heed 0.22 + LMDB**: Uses `heed` crate with `serde-bincode` feature for typed databases.
- **Adjacency lists**: Edges are stored as `Vec<(target, edge_type)>` per node in `adj_out`/`adj_in` databases. This avoids LMDB's single-value-per-key limitation without needing DUP_SORT.
- **Read-modify-write for indices**: Secondary indices (`nodes_by_type`, `adj_out`, `adj_in`) store `Vec` values that must be read, appended, and re-written. This is correct but can be slow at scale — future optimization could use DUP_SORT or byte-prefix keys.
- **5 named databases**: `nodes`, `edges`, `adj_out`, `adj_in`, `nodes_by_type`. Must set `max_dbs(5)` in EnvOpenOptions.
- **Unsafe `open()`**: Heed requires `unsafe` for `EnvOpenOptions::open()` due to memory-mapping.
- **Graph traversal**: BFS with `HashSet<NodeId>` visited set, follows both outgoing and incoming edges.
- **Vector search**: SIMD-optimized brute-force cosine similarity. Uses 8-wide accumulator lanes in `dot_product_simd()` for LLVM auto-vectorization (NEON on ARM, AVX on x86). Top-k selection uses `BinaryHeap<MinScored>` min-heap for O(n log k).
- **MinScored pattern**: `MinScored(f32, u64)` with inverted `Ord` turns `BinaryHeap` (max-heap) into a min-heap. The lowest-scoring entry is at `.peek()` for efficient replacement.

## Dependencies Between Crates

- `helix_impl` depends on `core`, `ucotron_config`, `heed` (0.22), `tempfile` (dev)
- `ucotron_config` depends on `serde`, `toml` (0.8), `anyhow`, `thiserror`
- `ucotron_extraction` depends on `core`, `ucotron_config`, `serde`, `anyhow`, `thiserror`
- `ucotron_server` depends on `core`, `ucotron_config`, `ucotron_extraction`, `helix_impl`, `tokio`, `axum` (0.8), `tower`, `tower-http`, `tracing`
- `ucotron_sdk` depends on `core`, `serde`, `anyhow`, `thiserror`
- `bench_runner` depends on `core`, `helix_impl`; optionally `ucotron-cozo` (feature `cozo`)
- `_archive/cozo_impl` depends on `core` (not in default workspace build)

## Feature Flags

- `bench_runner` has a `cozo` feature flag that enables CozoDB benchmarks
  - All CozoDB imports, benchmark code, and tests are gated behind `#[cfg(feature = "cozo")]`
  - Build with: `cargo test -p bench-runner --features cozo`
  - Without the flag, only HelixDB benchmarks compile and run

## Phase 2 Backend Traits

- `core/src/backends.rs` defines pluggable backend traits for Phase 2:
  - `VectorBackend`: `upsert_embeddings`, `search`, `delete` (Send + Sync, object-safe)
  - `GraphBackend`: `upsert_nodes`, `upsert_edges`, `get_node`, `get_neighbors`, `find_path`, `get_community` (Send + Sync, object-safe)
  - `BackendRegistry`: Holds `Box<dyn VectorBackend>` + `Box<dyn GraphBackend>`, single entry point for storage
  - `ExternalVectorBackend` / `ExternalGraphBackend`: Stub impls for future Qdrant/FalkorDB (return errors)
- These are separate from the Phase 1 `StorageEngine` trait (which combines both concerns)
- Exported from core: `ucotron_core::{VectorBackend, GraphBackend, BackendRegistry, ExternalVectorBackend, ExternalGraphBackend}`

## Phase 2 Helix Backend Implementations

- `helix_impl` provides `HelixVectorBackend` and `HelixGraphBackend` implementing the Phase 2 traits
  - `HelixVectorBackend::open(data_dir, max_db_size)` — vector-only ops over LMDB
  - `HelixGraphBackend::open(data_dir, max_db_size, batch_size)` — graph-only ops over LMDB
  - Both share the same LMDB database layout as `HelixEngine` (can coexist on same data dir)
- Config-driven factory functions:
  - `create_helix_vector_backend(&VectorBackendConfig)` — creates from TOML config
  - `create_helix_graph_backend(&GraphBackendConfig)` — creates from TOML config
  - `create_helix_backends(vec_config, graph_config)` — creates both as boxed trait objects
- Pattern: Use `BackendRegistry::new(vec_box, graph_box)` to combine backends into unified storage layer

## TypeScript SDK (sdks/typescript/)

- `@ucotron/sdk` npm package — pure TypeScript HTTP wrapper using native `fetch`
- Requires Node.js 18+ (uses `AbortSignal.timeout()`)
- Types in `src/types.ts` use snake_case to match server JSON (not camelCase)
- Error hierarchy: `UcotronError` → `UcotronServerError` / `UcotronConnectionError` / `UcotronRetriesExhaustedError`
- Retry logic: only retries 5xx and connection errors, never 4xx
- Multi-tenancy via `X-Ucotron-Namespace` header (configurable default + per-request override)
- Tests mock `global.fetch` — no running server needed
- Build: `npm run build` (outputs to `dist/`), Test: `npm test`
- Works in Node.js, Deno, Bun, and browsers — no native bindings

## CI/CD (.github/workflows/)

- **ci.yml**: Runs on PR to main/ralph/ucotron and push to main
  - `rust-lint`: clippy + rustfmt check
  - `rust-test`: matrix build on linux-x64, linux-arm64, macos-arm64
  - `typescript-sdk`: npm ci + build + jest
  - `python-sdk`: pip install + pytest
  - `go-sdk`: go vet + go test
  - `docs-build`: npm ci + next build
  - `integration`: starts server, runs `scripts/cross_language_tests.sh`
- **release.yml**: Triggered on `v*` tag push
  - `validate`: clippy + test on linux-x64, macos-arm64
  - `docker`: Multi-arch (amd64+arm64) image → GHCR (+ optional Docker Hub via `DOCKERHUB_TOKEN` secret)
    - Semver tags: `latest`, `vX`, `vX.Y`, `vX.Y.Z` (latest+major only for non-prerelease)
    - OCI image labels (title, description, version, created, source, revision, licenses, vendor)
  - `publish-rust`: Syncs workspace version from tag, publishes ucotron-core → ucotron-config → ucotron-sdk with 30s delays
  - `publish-typescript`: Syncs version via `npm version`, builds, publishes `@ucotron/sdk` to npm
  - `publish-python`: Syncs version in pyproject.toml, builds, publishes `ucotron-sdk` to PyPI
  - `publish-go`: Validates Go module (vet + test), documents proxy.golang.org availability
  - `docs`: Builds Next.js docs site, uploads as artifact
  - `release`: Creates GitHub Release with conventional-commit changelog (grouped by feat/fix/other) + install commands table
- **scripts/sync_versions.sh**: Local script to synchronize versions across all SDKs before tagging
  - Updates workspace Cargo.toml, package.json, pyproject.toml; Go uses git tags
- Caching: Cargo registry+build, npm, Go modules all cached via `actions/cache@v4`
- Concurrency: same-branch runs cancel in-progress CI runs

## Multimodal (CLIP Image Pipeline)

- **CLIP models**: Must be at `models/clip-vit-base-patch32/` with `visual_model.onnx`, `text_model.onnx`, `tokenizer.json`
  - Download from jmzzomg/clip-vit-base-patch32-{vision,text}-onnx (official HF repo has no ONNX exports)
  - Or run `scripts/download_multimodal_models.sh` (may need URL updates if HF paths change)
- **Visual vector backend**: CLIP embeddings (512-dim) are stored in a separate `HelixVisualVectorBackend` — NOT in the text vector index (384-dim MiniLM)
  - Without visual backend: image indexing falls back to text index, but `/images/search` returns 501
  - Server auto-creates visual backend if `visual_model.onnx` exists in CLIP model dir
- **Dual embedding storage**: `POST /memories/image` stores CLIP embedding in visual index + text embedding in text index (if description provided)
- **Cross-modal search**: `POST /images/search` encodes text query with CLIP text encoder (512-dim), searches visual index
- **Pipeline initialization**: `main.rs` calls `try_init_clip()` to load ClipImagePipeline + ClipTextPipeline, creates `BackendRegistry::with_visual()` for visual backend

## Gotchas

- Workspace uses `edition = "2021"` — do not mix editions
- Release profile has `lto = true` and `codegen-units = 1` — release builds are slow but optimized
- `bincode` v1 (not v3) is used for stability — v3 has breaking API changes
- `rand` v0.8 and `rand_chacha` v0.3 are pinned for reproducibility across runs
- **LMDB key uniqueness**: Standard LMDB databases map each key to ONE value. For multi-edge adjacency, use `Vec` values (read-modify-write) or composite keys, NOT simple key→value with same key repeated.
- **SerdeBincode key ordering**: `SerdeBincode<u64>` encodes u64 as little-endian via bincode, so LMDB's byte-order iteration does NOT match numeric order. Use adjacency lists or manual big-endian encoding for range queries.
