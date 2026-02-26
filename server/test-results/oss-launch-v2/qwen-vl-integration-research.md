# Qwen3-VL-Embedding/Reranker Integration Research

## Models

| Model | Type | Params | Dim | Format |
|-------|------|--------|-----|--------|
| Qwen3-VL-Embedding-2B | Bi-encoder embedding | 2B | 64-2048 (MRL) | HF Transformers (safetensors) |
| Qwen3-VL-Reranker-2B | Cross-encoder reranker | 2B | N/A (scores) | HF Transformers (safetensors) |
| Qwen3-VL-Embedding-8B | Bi-encoder embedding | 8B | 64-2048 (MRL) | HF Transformers (safetensors) |
| Qwen3-VL-Reranker-8B | Cross-encoder reranker | 8B | N/A (scores) | HF Transformers (safetensors) |

## Integration Options Evaluated

### Option 1: ONNX Export (rejected)
- **Pros**: Keep everything in Rust, fast inference
- **Cons**: No documented ONNX export for Qwen3-VL models, 2B+ params produce very large ONNX files, custom architecture may not export cleanly
- **Verdict**: Not viable — models use custom `trust_remote_code=True` and Qwen3VL architecture not in ONNX exporters

### Option 2: Candle (rejected)
- **Pros**: Pure Rust, no Python dependency
- **Cons**: Candle does not support Qwen3-VL architecture, would need custom implementation of the entire model
- **Verdict**: Too much work, not practical

### Option 3: Python Sidecar (chosen)
- **Pros**: Models designed for HF Transformers, uses official code, supports all features (MRL, instruction-aware, multimodal), easiest to maintain
- **Cons**: Adds Python dependency, HTTP overhead (~1-2ms per request), additional process to manage
- **Verdict**: Best tradeoff — models work out of the box, minimal overhead

## Architecture

```
┌─────────────────────┐     HTTP      ┌─────────────────────────┐
│   ucotron_server     │◄────────────►│    Python Sidecar        │
│   (Rust, :8420)      │              │    (FastAPI, :8421)       │
│                      │              │                           │
│  SidecarEmbedding    │  /embed      │  Qwen3-VL-Embedding-2B   │
│  Pipeline            │──────────────│  (mean pool + MRL)        │
│                      │              │                           │
│  SidecarReranker     │  /rerank     │  Qwen3-VL-Reranker-2B    │
│                      │──────────────│  (cross-encoder scores)   │
│                      │              │                           │
│                      │  /health     │  Health check             │
└─────────────────────┘              └─────────────────────────┘
```

## Dimension Compatibility

- Current HNSW index: 384 dimensions (all-MiniLM-L6-v2)
- Qwen3-VL-Embedding supports MRL: output dimensions 64-2048
- Default sidecar config: 384 dimensions (matches existing index)
- For fresh deployments: can use up to 2048 for better quality

**Important**: Switching embedding model on existing data requires re-indexing. The HNSW index dimension must match the embedding dimension.

## Configuration

```toml
[models]
# Switch to sidecar embeddings
embedding_provider = "sidecar"
sidecar_url = "http://localhost:8421"
sidecar_embedding_dim = 384  # Match existing HNSW, or 2048 for fresh
enable_reranker = true
```

Or via environment variables:
```bash
UCOTRON_MODELS_EMBEDDING_PROVIDER=sidecar
UCOTRON_MODELS_SIDECAR_URL=http://localhost:8421
UCOTRON_MODELS_SIDECAR_EMBEDDING_DIM=384
UCOTRON_MODELS_ENABLE_RERANKER=true
```

## Sidecar Setup

```bash
cd server/sidecar
pip install -r requirements.txt
# Download models on first run (auto from HuggingFace)
EMBEDDING_DIM=384 uvicorn main:app --host 0.0.0.0 --port 8421
```

## Requirements

- Python 3.10+
- `transformers>=4.57.0`
- `torch>=2.1.0`
- ~4-6GB RAM per model (2B), ~16GB per model (8B)
- MPS (Apple Silicon), CUDA, or CPU

## Limitations

1. **Latency**: HTTP overhead of ~1-2ms per request vs in-process ONNX
2. **Memory**: 2B models need ~4-6GB each; running both embedding + reranker needs ~8-12GB
3. **Startup**: Model download on first run (~4GB per model from HuggingFace)
4. **Multimodal**: Image/video embedding via sidecar not yet wired (text-only for now)
5. **8B models**: May not fit on 16GB RAM machines — need 32GB+ or cloud GPU

## Files Modified

- `ucotron_extraction/src/embeddings.rs` — Added `SidecarEmbeddingPipeline`
- `ucotron_extraction/src/reranker.rs` — New module: `RerankerPipeline` trait + `SidecarReranker`
- `ucotron_extraction/src/lib.rs` — Registered `reranker` module
- `ucotron_config/src/lib.rs` — Added sidecar config fields + env overrides
- `ucotron_server/src/main.rs` — `try_init_embedder` supports sidecar, added `try_init_reranker`
- `ucotron_server/src/state.rs` — Added `reranker` field to `AppState`
- `ucotron_server/src/types.rs` — Added `embedding_provider`, `reranker_loaded` to health
- `ucotron_server/src/handlers/mod.rs` — Health handler populates new fields
- `ucotron.toml` — Documented new config options
- `sidecar/main.py` — Python FastAPI sidecar service
- `sidecar/requirements.txt` — Python dependencies
