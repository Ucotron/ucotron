# ADR-003: Fireworks.ai for Fine-Tuning (Not Local)

**Status:** Accepted
**Date:** 2026-02-14
**Decision Makers:** Phase 3 architecture review

## Context

Ucotron needs fine-tuned models for:
- Relation extraction (RE) between entities
- Preference alignment (DPO) for retrieval ranking
- Contradiction detection

Options considered:

1. **Local fine-tuning** - PyTorch/TRL on local GPU (requires NVIDIA hardware)
2. **Cloud fine-tuning (Fireworks.ai)** - OpenAI-compatible API, serverless GPUs
3. **Cloud fine-tuning (AWS SageMaker / GCP Vertex)** - Full cloud ML platforms
4. **No fine-tuning** - Use co-occurrence heuristics only

## Decision

**Use Fireworks.ai** for remote fine-tuning via their OpenAI-compatible API.

## Rationale

### Why not local?

- Apple Silicon Macs (primary dev environment) lack CUDA support
- Local fine-tuning requires significant GPU VRAM (7B model needs ~16 GB)
- Development team shouldn't need to maintain GPU infrastructure

### Why Fireworks over SageMaker/Vertex?

- **Simpler API**: OpenAI-compatible endpoints, no cloud-specific SDKs
- **Cost**: Pay-per-job pricing, no persistent instance costs
- **Speed**: No cluster provisioning; jobs start within minutes
- **Model variety**: Direct access to Qwen, Llama, and other open models

### Why not skip fine-tuning?

Co-occurrence relation extraction achieves ~60% precision. Fine-tuned models target >85% precision for production quality, especially for complex predicates like temporal relations.

### Pipeline Architecture

```
1. Dataset Generation (Rust + Python)
   └── 7 generators produce JSONL training data from knowledge graph

2. Fine-Tuning (Python → Fireworks API)
   ├── SFT (Supervised Fine-Tuning) for RE
   ├── DPO (Direct Preference Optimization) for ranking
   └── Three tiers: SLM (0.5B), Small (1.5B), Medium (7B)

3. Export (Optional ONNX)
   └── Convert for local inference without API dependency
```

### Model Tiers

| Tier   | Model      | Use Case           | Latency |
|--------|------------|--------------------| --------|
| SLM    | Qwen 0.5B  | Edge/mobile        | <50ms   |
| Small  | Qwen 1.5B  | Default server     | <200ms  |
| Medium | Qwen 7B    | Maximum accuracy   | <1s     |

## Consequences

- Runtime dependency on `FIREWORKS_API_KEY` environment variable for fine-tuning
- No fine-tuning possible in air-gapped environments (mitigated by ONNX export)
- Training costs are variable (per-job pricing)
- Co-occurrence extractor remains as fallback when no fine-tuned model is configured

## References

- `scripts/fine_tune/train_slm.py` - Fine-tuning client
- `scripts/finetune_pipeline.sh` - End-to-end pipeline orchestrator
- `scripts/generate_training_data/` - Dataset generators
- `ucotron_config/src/lib.rs` - `models.fine_tuned_re_model` configuration
