# Runbook: Model Update

## Overview

Ucotron uses ONNX models for NLP tasks (embedding, NER, relation extraction) and optionally multimodal models (CLIP, Whisper). Models are stored in the `models/` directory and loaded at server startup.

---

## Model Inventory

| Model | Directory | Size | Purpose |
|-------|-----------|------|---------|
| all-MiniLM-L6-v2 | `models/all-MiniLM-L6-v2/` | ~90 MB | Text embedding (384-dim) |
| GLiNER small v2.1 | `models/gliner_small-v2.1/` | ~583 MB | Named entity recognition |
| Qwen3-4B-GGUF | `models/Qwen3-4B-GGUF/` | ~2.5 GB | LLM (optional, for relation extraction) |
| CLIP ViT-B/32 | `models/clip-vit-base-patch32/` | ~350 MB | Image embedding (optional) |
| Whisper base | `models/whisper-base/` | ~150 MB | Audio transcription (optional) |

---

## Downloading Models

### Initial setup

```bash
# Download core models (embedding + NER)
./scripts/download_models.sh

# Download multimodal models (CLIP + Whisper)
./scripts/download_multimodal_models.sh
```

### Manual download (HuggingFace)

```bash
# Example: update embedding model
pip install huggingface_hub
huggingface-cli download sentence-transformers/all-MiniLM-L6-v2 \
  --include "model.onnx" "tokenizer.json" \
  --local-dir models/all-MiniLM-L6-v2/
```

---

## Updating a Model (Zero-Downtime)

To update a model without dropping requests:

```bash
# 1. Download new model to a staging directory
mkdir -p models/all-MiniLM-L6-v2-new/
huggingface-cli download sentence-transformers/all-MiniLM-L6-v2 \
  --include "model.onnx" "tokenizer.json" \
  --local-dir models/all-MiniLM-L6-v2-new/

# 2. Verify the new model loads
# (Quick smoke test: ensure ONNX file is valid)
python3 -c "import onnxruntime; onnxruntime.InferenceSession('models/all-MiniLM-L6-v2-new/model.onnx')"

# 3. Atomic swap
mv models/all-MiniLM-L6-v2 models/all-MiniLM-L6-v2-old
mv models/all-MiniLM-L6-v2-new models/all-MiniLM-L6-v2

# 4. Restart server to pick up new model
kill -TERM <PID>
ucotron_server --config /etc/ucotron/ucotron.toml

# 5. Verify via health endpoint
curl -s http://localhost:8420/api/v1/health | jq '.models'

# 6. Clean up old model (after confirming new one works)
rm -rf models/all-MiniLM-L6-v2-old
```

### Docker model update

```bash
# Rebuild image (models are baked in during build stage)
docker compose build --no-cache ucotron
docker compose up -d
```

---

## Configuration

Models are configured in `ucotron.toml`:

```toml
[models]
embedding_model = "all-MiniLM-L6-v2"
ner_model = "gliner-multi-v2.1"
llm_model = "Qwen3-4B-GGUF"
llm_backend = "candle"           # or "llama_cpp"
clip_model = "clip-vit-base-patch32"
models_dir = "models"
enable_ocr = true
ocr_language = "eng"
tesseract_path = "tesseract"
```

Override models directory: `UCOTRON_MODELS_DIR=/custom/path`

---

## Embedding Dimension Compatibility

Changing the embedding model requires re-indexing all vectors:

1. The HNSW index is built for a specific dimension (384 for MiniLM)
2. If you switch to a model with different dimensions, existing vectors become incompatible
3. **You must re-ingest all data** after changing the embedding model

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `Failed to load ONNX session` | Corrupt or missing model file | Re-download with `scripts/download_models.sh` |
| `Dimension mismatch` on search | Model changed without re-indexing | Re-ingest all data with new model |
| `ORT error: invalid protobuf` | Incomplete download | Delete and re-download the model |
| Health shows `embedder_loaded: false` | Model path misconfigured | Check `models.models_dir` and `models.embedding_model` in config |
| High memory usage after model load | Large LLM model loaded | Use quantized GGUF variant or disable LLM |
