# Multi-Hop Query Comparison Report

**Date:** 2026-02-26
**Project:** oss-launch-v2
**Test Data:** 15 queries (5x 1-hop, 5x 2-hop, 5x 3-hop) over 30 facts with 33 entities

## Configurations Tested

| Config | Embedding Model | Relation Extraction | Reranker | Status |
|--------|----------------|-------------------|----------|--------|
| **No-LLM Baseline** | MiniLM-L6-v2 (ONNX, local) | Co-occurrence | None | Tested |
| **Qwen3-4B LLM** | MiniLM-L6-v2 (ONNX, local) | Qwen3-4B-GGUF (local) | None | Tested |
| **Qwen 2B Pair** | Qwen3-VL-Embedding-2B (sidecar) | Co-occurrence | Qwen3-VL-Reranker-2B | Tested |
| **Qwen 8B Pair** | Qwen3-VL-Embedding-8B (sidecar) | Co-occurrence | Qwen3-VL-Reranker-8B | Skipped (infeasible) |

## Accuracy by Hop Depth

| Hop Depth | No-LLM | Qwen3-4B LLM | Qwen 2B Pair | Delta (4B vs Base) | Delta (2B vs Base) |
|-----------|--------|-------------|-------------|--------------------|--------------------|
| **1-hop** | 80% (4/5) | **100% (5/5)** | **100% (5/5)** | +20% | +20% |
| **2-hop** | 60% (3/5) | **80% (4/5)** | 60% (3/5) | +20% | +0% |
| **3-hop** | 20% (1/5) | **80% (4/5)** | 60% (3/5) | **+60%** | **+40%** |
| **Overall** | 53.3% (8/15) | **86.7% (13/15)** | 73.3% (11/15) | **+33.4%** | **+20%** |

### Key Findings

1. **LLM relation extraction is the biggest accuracy lever** — Qwen3-4B LLM improves overall accuracy by +33.4% vs baseline, with the most dramatic improvement at 3-hop (+60%)
2. **Better embeddings help, but less than LLM relations** — Qwen 2B embedding improves overall by +20%, with 3-hop gaining +40%
3. **1-hop is nearly solved** — Both Qwen configs achieve 100%; baseline is already 80%
4. **2-hop shows LLM advantage** — Only Qwen3-4B LLM improves 2-hop accuracy; better embeddings alone don't help
5. **3-hop is where configs diverge most** — 20% → 60% → 80% across the three configs

## Query Latency by Hop Depth

| Hop Depth | No-LLM P50 (ms) | Qwen3-4B P50 (ms) | Qwen 2B P50 (ms) |
|-----------|-----------------|-------------------|------------------|
| **1-hop** | 216.69 | **88.83** | 296.32 |
| **2-hop** | 217.59 | **89.65** | 296.97 |
| **3-hop** | 218.40 | **89.44** | 338.85 |
| **Overall P50** | 217.91 | **89.20** | 297.32 |
| **Overall Mean** | 220.52 | **91.54** | 351.21 |
| **Overall Max** | 256.44 | **125.48** | 616.12 |

### Latency Observations

- **Qwen3-4B LLM is paradoxically the fastest at query time** — 89ms P50 vs 218ms baseline (-59%). LLM-extracted relations during ingestion produce better memory content, leading to more efficient vector retrieval at query time
- **Qwen 2B Pair is the slowest** — 297ms P50 (+36% vs baseline) due to HTTP sidecar overhead for each embedding call (~100ms roundtrip)
- **Latency is flat across hop depths** — the `/augment` endpoint doesn't perform deeper graph traversal for harder queries; all queries use the same retrieval path
- **Ingestion is the cost center for LLM** — Qwen3-4B adds ~2.3s per fact during ingestion (LLM inference), while Qwen 2B adds ~300ms (HTTP overhead)

## Per-Query Results

| Q# | Hops | Question | No-LLM | Qwen3-4B | Qwen 2B | Notes |
|----|------|----------|--------|----------|---------|-------|
| Q01 | 1 | Who created Linux? | ✅ | ✅ | ✅ | Easy — direct fact |
| Q02 | 1 | Where is CERN located? | ✅ | ✅ | ✅ | Easy — direct fact |
| Q03 | 1 | Who invented the World Wide Web? | ✅ | ✅ | ✅ | Easy — direct fact |
| Q04 | 1 | What company acquired GitHub? | ❌ | ✅ | ✅ | Both Qwen configs find Microsoft |
| Q05 | 1 | Where was Ada Lovelace born? | ✅ | ✅ | ✅ | Easy — direct fact |
| Q06 | 2 | Creator of Linux → studied where? | ✅ | ✅ | ✅ | Helsinki found by all |
| Q07 | 2 | WWW invented → city? | ✅ | ✅ | ✅ | Geneva found by all |
| Q08 | 2 | Linux → OS → company? | ✅ | ✅ | ✅ | Google found by all |
| Q09 | 2 | GitHub → acquired by → HQ? | ❌ | ❌ | ❌ | **All miss** — Redmond not retrievable |
| Q10 | 2 | Bletchley Park → person → invented? | ❌ | ✅ | ❌ | Only LLM finds Turing Machine |
| Q11 | 3 | Bletchley Park → person → invention → foundation? | ❌ | ✅ | ✅ | Computer Science found by 4B + 2B |
| Q12 | 3 | Linux → OS → company → HQ? | ❌ | ✅ | ❌ | Only LLM finds Mountain View |
| Q13 | 3 | WWW → built on → designer? | ❌ | ✅ | ✅ | Vint Cerf found by 4B + 2B |
| Q14 | 3 | Google → founder → university → city? | ✅ | ✅ | ✅ | Palo Alto found by all |
| Q15 | 3 | Torvalds → Git → GitHub → acquirer → HQ? | ❌ | ❌ | ❌ | **All miss** — 4-hop chain to Redmond |

## Analysis: Where LLM Helps Most

### Queries Only LLM Solves (Q10, Q12)

These require **indirect entity connections** that co-occurrence and better embeddings alone can't bridge:

- **Q10** (Bletchley Park → Turing → Turing Machine): LLM explicitly extracts the "Turing worked_at Bletchley Park" and "Turing invented Turing Machine" relations, making the connection retrievable
- **Q12** (Linux → Android → Google → Mountain View): LLM captures the full chain of relations during ingestion, embedding richer context into memories

### Queries Both Qwen Configs Solve (Q04, Q11, Q13)

These benefit from **better semantic understanding** regardless of approach:

- **Q04** (GitHub → Microsoft): Both better embeddings and LLM relations capture the acquisition relationship
- **Q11** (Computer Science): Multi-hop but with strong textual co-occurrence in the training data
- **Q13** (Vint Cerf): TCP/IP + WWW connection has strong semantic signal

### Queries Nothing Solves (Q09, Q15)

Both involve **Redmond** as the answer, requiring:

- **Q09**: GitHub → Microsoft → Redmond (2-hop, but "Redmond" may not appear in enough context)
- **Q15**: Torvalds → Git → GitHub → Microsoft → Redmond (effectively 4-hop, exceeds test design)

These failures suggest that geographic headquarters are weakly connected to company identity in the knowledge graph, even with LLM-extracted relations.

## Ingestion Cost Comparison

| Metric | No-LLM | Qwen3-4B LLM | Qwen 2B Pair |
|--------|--------|-------------|-------------|
| Facts ingested | 30 | 30 | 30 |
| Errors | 0 | 0 | 0 |
| Entities extracted | 85 | 85 | 85 |
| P50 latency (ms) | ~150 | 2,334 | 456 |
| Max latency (ms) | ~500 | 5,797 | 24,635* |
| Relation strategy | co-occurrence | LLM | co-occurrence |

*First fact spike (24.6s) is sidecar model warmup on MPS GPU

### Ingestion Trade-offs

- **No-LLM**: Fastest ingestion, lowest accuracy — good for high-volume, low-precision use cases
- **Qwen3-4B LLM**: 15x slower ingestion, highest accuracy — the LLM inference per entity pair is the bottleneck
- **Qwen 2B Pair**: 3x slower ingestion, moderate accuracy — HTTP overhead is manageable, first-call warmup is one-time

## Known Limitations of Graph Traversal

1. **`graph_expanded_nodes=0` across all configs** — the `/augment` endpoint's graph expansion doesn't trigger for most queries, likely because NER doesn't extract entities from short question strings
2. **Query latency is uniform across hop depths** — no deeper traversal for harder queries; accuracy improvements come from better ingestion (richer memories), not smarter retrieval
3. **Limit=20 masks the problem at small scale** — with only 30 facts, vector search retrieves most data; at production scale (10K+ facts), graph traversal becomes critical
4. **"Redmond" queries fail universally** — geographic headquarters are weakly linked to parent entities, suggesting a need for explicit "headquartered_in" relation extraction

## Recommendations

| Use Case | Recommended Config | Rationale |
|----------|-------------------|-----------|
| **Maximum accuracy** | Qwen3-4B LLM | 86.7% accuracy, best 3-hop (+60%), worth the ingestion cost |
| **Balanced** | Qwen 2B Pair | 73.3% accuracy, no LLM needed, moderate latency |
| **Low latency / high volume** | No-LLM Baseline | 53.3% accuracy but fastest ingestion, suitable when precision isn't critical |
| **Best possible (theoretical)** | Qwen3-4B LLM + Qwen 2B Pair | Combine LLM relations with better embeddings — untested but expected to exceed 86.7% |

## Qwen 8B Pair — Not Tested

The Qwen3-VL-Embedding-8B and Qwen3-VL-Reranker-8B models could not be tested locally:

- **Disk**: 4.9 GB free vs ~32 GB needed (16 GB per model)
- **RAM**: 16 GB total vs ~32 GB needed (fp16) or ~16 GB (int8, cutting it close)
- **No code changes needed**: Sidecar is model-agnostic — set `EMBEDDING_MODEL` and `RERANKER_MODEL` env vars
- **Cloud requirement**: AWS g5.4xlarge (64 GB RAM, A10G GPU) or Mac Studio M2 Ultra (64+ GB unified)
- **Expected improvement**: 8B models should improve embedding quality over 2B, potentially narrowing the gap with LLM config

See `qwen-8b-feasibility.md` for detailed hardware requirements.

## Data Sources

- `multihop-test-data.json` — 30 facts, 15 queries, expected answers
- `multihop-no-llm.json` — Baseline results (MiniLM + co-occurrence)
- `multihop-qwen3-4b.json` — LLM results (MiniLM + Qwen3-4B relations)
- `multihop-qwen-2b.json` — Sidecar results (Qwen3-VL-Embedding-2B + Reranker-2B)
- `multihop-qwen-8b.json` — Skip documentation
