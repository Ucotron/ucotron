# ADR-002: Dual-Index Multimodal Architecture

**Status:** Accepted
**Date:** 2026-02-14
**Decision Makers:** Phase 3.5 architecture review

## Context

Ucotron needed cross-modal search (text-to-image, image-to-text, audio-to-text) while maintaining the existing 384-dim MiniLM text embedding index. Options considered:

1. **Single unified index** - Project all modalities into one embedding space (e.g., 384-dim)
2. **Dual-index** - Separate HNSW indices for text (384-dim MiniLM) and visual (512-dim CLIP)
3. **Multi-index** - Separate index per modality (text, image, audio, video)

## Decision

**Use a dual-index design**: one 384-dim HNSW index for text (MiniLM space) and one 512-dim HNSW index for visual (CLIP space).

## Rationale

### Why not a single index?

MiniLM (384-dim) and CLIP (512-dim) produce embeddings in fundamentally different vector spaces. Naive concatenation or dimension reduction would destroy semantic relationships. A trained projection layer bridges the spaces at query time instead.

### Why dual and not multi?

- Audio (Whisper) produces text transcripts, which map to the text index
- Video produces both frames (visual index) and transcripts (text index)
- Two indices cover all four modalities with minimal overhead

### Architecture

```
Text/Audio → MiniLM encoder → Text Index (384-dim HNSW)
Images     → CLIP encoder   → Visual Index (512-dim HNSW)
Video      → frames → Visual Index + transcript → Text Index

Cross-modal bridge:
  text→image: CLIP text encoder → Visual Index
  image→text: Projection layer (512→384 MLP) → Text Index
```

### Implementation

- `VectorBackend` trait: text embeddings (384-dim) in `helix_impl/src/lib.rs`
- `VisualVectorBackend` trait: visual embeddings (512-dim) in `core/src/backends.rs`
- Each backend has its own LMDB environment and independent HNSW index
- Projection layer trained separately (`scripts/train_projection_layer.py`)

## Consequences

- Two HNSW indices to maintain and rebuild on upsert
- Cross-modal accuracy depends on projection layer quality (target: cosine loss < 0.15)
- Visual index adds ~50% storage overhead for image-heavy workloads
- Clean separation means text-only deployments pay zero overhead for visual features

## References

- `core/src/backends.rs` - `VisualVectorBackend` trait definition
- `helix_impl/src/lib.rs` - `HelixVisualVectorBackend` implementation
- `scripts/train_projection_layer.py` - Projection layer training
