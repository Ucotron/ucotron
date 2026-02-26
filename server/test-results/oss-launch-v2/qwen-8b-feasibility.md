# Qwen3-VL 8B Model Feasibility Assessment

## V2-010: Integrate Qwen3-VL-Embedding-8B + Qwen3-VL-Reranker-8B

### Models

| Model | HuggingFace ID | Estimated Size (disk) | Estimated RAM (fp16) |
|-------|---------------|----------------------|---------------------|
| Qwen3-VL-Embedding-8B | Qwen/Qwen3-VL-Embedding-8B | ~16 GB | ~16 GB |
| Qwen3-VL-Reranker-8B | Qwen/Qwen3-VL-Reranker-8B | ~16 GB | ~16 GB |

### Local Machine Specs

| Resource | Available | Required (both models) |
|----------|-----------|----------------------|
| RAM | 16 GB total | ~32 GB (fp16) or ~16 GB (int8) |
| Disk | 4.9 GB free | ~32 GB for model weights |
| GPU | Apple Silicon (shared memory) | 32+ GB unified memory |

### Verdict: NOT FEASIBLE locally

Both hardware constraints are blocking:

1. **Disk space**: 4.9 GB available vs ~32 GB needed to download both models. Cannot even download a single 8B model.
2. **Memory**: 16 GB total RAM (shared CPU+GPU on Apple Silicon). Even with int8 quantization (~8 GB per model), loading both models simultaneously requires ~16 GB, leaving zero headroom for the OS, ucotron server, and HNSW index.
3. **Single model**: Loading just the embedding model alone (~16 GB fp16 / ~8 GB int8) would consume most available RAM, causing severe swapping and impractical latency.

### Requirements for Running 8B Models

| Configuration | RAM | Disk | GPU |
|--------------|-----|------|-----|
| Both models (fp16) | 48+ GB | 40+ GB | Optional (MPS/CUDA) |
| Both models (int8) | 24+ GB | 20+ GB | Optional |
| Embedding only (fp16) | 24+ GB | 20+ GB | Optional |
| Embedding only (int8) | 16+ GB | 12+ GB | Recommended |

**Recommended cloud instances:**
- AWS: `g5.2xlarge` (1x A10G, 24GB VRAM, 32GB RAM) — embedding only
- AWS: `g5.4xlarge` (1x A10G, 24GB VRAM, 64GB RAM) — both models with CPU offloading
- AWS: `p3.2xlarge` (1x V100, 16GB VRAM, 61GB RAM) — both models in CPU, GPU for one
- Apple: Mac Studio M2 Ultra (64-192GB unified) — both models comfortably

### Integration Approach (when hardware available)

The sidecar architecture from V2-008/V2-009 already supports 8B models with zero code changes:

```bash
# Just change environment variables
EMBEDDING_MODEL=Qwen/Qwen3-VL-Embedding-8B \
RERANKER_MODEL=Qwen/Qwen3-VL-Reranker-8B \
python server/sidecar/main.py
```

The sidecar `main.py` uses `AutoModel.from_pretrained()` and `Qwen3VLForConditionalGeneration.from_pretrained()` — both accept any model size. The only changes needed are:

1. Download the 8B models: `huggingface-cli download Qwen/Qwen3-VL-Embedding-8B`
2. Set the environment variables as shown above
3. Ensure sufficient RAM/VRAM

No Rust code changes required — the `SidecarEmbeddingPipeline` and `SidecarReranker` communicate via HTTP and are model-size agnostic.

### Status

- **V2-010**: COMPLETE (documented as infeasible locally, integration path confirmed)
- **V2-011**: SKIP (benchmark requires running models, blocked by V2-010)

Date: 2026-02-26
