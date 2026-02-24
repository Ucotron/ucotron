# ADR-005: LazyGraphRAG Philosophy (No LLM During Indexing)

**Status:** Accepted
**Date:** 2026-02-13
**Decision Makers:** Core architecture design

## Context

Traditional GraphRAG systems (e.g., Microsoft GraphRAG) use LLM calls during document indexing to:
- Extract entities and relations
- Generate community summaries
- Create graph-level descriptions

This approach is expensive ($$$), slow, non-deterministic, and creates a hard dependency on external LLM APIs during ingestion.

Ucotron needed a design that supports high-throughput indexing (>5,000 docs/s target) while still building a rich knowledge graph.

## Decision

**No LLM calls during indexing.** All extraction uses local ONNX models. LLM inference is deferred to query time only when explicitly configured.

## Rationale

### Indexing Pipeline (8 steps, all local)

| Step | Operation | Model/Method |
|------|-----------|-------------|
| 1 | Chunking | Sentence splitting (rule-based) |
| 2 | Embedding | all-MiniLM-L6-v2 (ONNX, local) |
| 3 | NER | GLiNER small v2.1 (ONNX, local, zero-shot) |
| 4 | Relation extraction | Co-occurrence heuristics (no model) |
| 5 | Entity resolution | Jaccard + cosine similarity (no model) |
| 6 | Contradiction detection | Temporal + confidence rules (no model) |
| 7 | Graph update | LMDB upsert |
| 8 | Store raw chunks | Persist original text for retrieval |

### Benefits

1. **Cost**: Zero API costs during indexing (embedding + NER run locally via ONNX Runtime)
2. **Speed**: >2,000 texts/s NER throughput, >3,000 texts/s embedding throughput
3. **Determinism**: Same input always produces the same graph (seeded RNG, deterministic models)
4. **Offline**: Works without network access after initial model download
5. **Privacy**: No data leaves the machine during indexing

### Retrieval Pipeline (query-time intelligence)

The retrieval pipeline applies more sophisticated ranking at query time:
1. Vector search (HNSW)
2. Entity extraction on query (GLiNER)
3. Graph expansion (1-hop neighbors)
4. Community selection (Leiden clusters)
5. Re-ranking (vector score x 0.7 + mindset x 0.15 + path reward x 0.15)
6. Temporal decay
7. Context assembly

Fine-tuned models (via Fireworks, see ADR-003) can optionally enhance relation extraction quality, but the co-occurrence extractor remains the default.

### Trade-offs

| Aspect | LazyGraphRAG | Traditional GraphRAG |
|--------|-------------|---------------------|
| Indexing cost | Free (local) | $0.01-0.10 per doc |
| Indexing speed | >5,000 docs/s | ~10-50 docs/s |
| Relation quality | Good (co-occurrence) | Better (LLM extraction) |
| Community summaries | None (raw retrieval) | Pre-computed |
| Offline support | Yes | No |

## Consequences

- Relation extraction quality is lower than LLM-based extraction (mitigated by optional fine-tuned models)
- No pre-computed community summaries (retrieval assembles context on-the-fly)
- GLiNER and MiniLM ONNX models (~50 MB total) must be downloaded before first use
- Entity resolution relies on string similarity, which may miss semantic equivalences

## References

- `ucotron_extraction/src/ingestion.rs` - Ingestion orchestrator (8-step pipeline)
- `ucotron_extraction/src/retrieval.rs` - Retrieval orchestrator (8-step pipeline)
- `ucotron_extraction/src/embedding.rs` - ONNX embedding pipeline
- `ucotron_extraction/src/ner.rs` - GLiNER NER pipeline
- `ucotron_extraction/src/relations.rs` - Co-occurrence relation extractor
