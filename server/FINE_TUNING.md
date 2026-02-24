# Fine-Tuning Guide

Internal guide for generating training datasets and fine-tuning relation extraction models for Ucotron.

---

## Overview

Ucotron uses fine-tuned small language models (SLMs) for relation extraction instead of relying solely on co-occurrence heuristics. The pipeline:

1. **Generate training data** from the Ucotron knowledge graph (Rust) or via LLM synthesis (Python + Fireworks.ai)
2. **Fine-tune** a Qwen model using TRL SFT + optional DPO alignment
3. **Export** the fine-tuned model to ONNX for inference in `ucotron_extraction`

---

## Quick Start

```bash
# Full pipeline (synthetic data -> train -> ONNX export)
./scripts/finetune_pipeline.sh

# Dry run (validate environment without training)
./scripts/finetune_pipeline.sh --dry-run

# Skip dataset generation (reuse existing data)
./scripts/finetune_pipeline.sh --skip-generate --data-dir data/finetune

# Custom model and epochs
./scripts/finetune_pipeline.sh --model-name Qwen/Qwen3-1.7B --epochs 5 --max-samples 20000
```

---

## Prerequisites

### Rust (dataset generation from graph)

- Rust toolchain via rustup (`cargo build -p ucotron-extraction` must succeed)

### Python (training)

```bash
pip install torch transformers trl datasets peft accelerate
```

### Fireworks.ai (remote fine-tuning & LLM dataset generation)

```bash
export FIREWORKS_API_KEY="fw-..."
export FIREWORKS_ACCOUNT_ID="your-account-id"
```

Optional: `export WANDB_API_KEY="..."` for Weights & Biases experiment tracking.

---

## Architecture

### Model Tiers

| Tier | Model | Params | Use Case | SFT LR | LoRA Rank |
|------|-------|--------|----------|--------|-----------|
| Edge (SLM) | Qwen 2.5 0.5B | 0.5B | On-device, low latency | 2e-4 | 8 |
| Balanced | Qwen 2.5 1.5B | 1.5B | General purpose | 2e-4 | 16 |
| Cloud | Qwen 2.5 7B | 7B | High quality | 1e-4 | 16 |

The generation model for synthetic datasets is **GLM-4 Plus** via Fireworks.ai.

### Training Paradigms

- **SFT (Supervised Fine-Tuning)**: Chat-formatted JSONL with system/user/assistant messages. Uses LoRA for parameter-efficient training.
- **DPO (Direct Preference Optimization)**: Chosen (thorough extraction, temp=0.0) vs rejected (incomplete, temp=0.7) pairs for alignment.
- **Projection MLP**: A 3-layer MLP (512->1024->512->384) bridging CLIP embeddings to MiniLM space for cross-modal search.

---

## Step 1: Dataset Generation

### Option A: From Ucotron Knowledge Graph (Rust)

The `ucotron_extraction::fine_tuning` module generates training samples directly from the knowledge graph via `BackendRegistry`.

```rust
use ucotron_extraction::fine_tuning::{DatasetConfig, generate_dataset};

let config = DatasetConfig {
    max_samples: 10_000,
    train_ratio: 0.8,
    min_relations: 1,
    max_text_length: 2048,
    seed: 42,
};

let result = generate_dataset(&registry, &config)?;
// result.train_samples, result.val_samples
```

The REST API also exposes this at `POST /api/v1/finetune/generate-dataset` (admin role required).

### Option B: LLM Synthesis via Fireworks.ai (Python)

Six specialized generators in `scripts/generate_training_data/`:

| Generator | Target Samples | Output |
|-----------|---------------|--------|
| `re_dataset_generator.py` | 10,000 | Relation extraction (text -> entities + relations) |
| `preference_generator.py` | 5,000 | DPO pairs (chosen vs rejected extractions) |
| `contradiction_generator.py` | 3,000 | Contradiction detection (temporal/confidence/ambiguous) |
| `entity_resolution_generator.py` | 2,000 | Entity dedup (canonical + variant + is_duplicate) |
| `path_reward_generator.py` | 500 | Path reward scores for retrieval models |
| `projection_dataset_generator.py` | 50,000 | CLIP/MiniLM embedding pairs for projection training |

Example:

```bash
cd memory_arena
export FIREWORKS_API_KEY="fw-..."

# Generate relation extraction dataset
python -m scripts.generate_training_data.re_dataset_generator \
    --count 10000 --output data/re_dataset.jsonl

# Generate DPO preference pairs
python -m scripts.generate_training_data.preference_generator \
    --count 5000 --output data/dpo_pairs.jsonl
```

### Training Data Format

**SFT (chat messages JSONL)**:

```json
{
  "messages": [
    {"role": "system", "content": "You are a relation extraction assistant..."},
    {"role": "user", "content": "Extract all relationships from...\nText: \"Alice works at Google\"\nEntities:\n- \"Alice\" (person)\n- \"Google\" (organization)"},
    {"role": "assistant", "content": "[{\"subject\": \"Alice\", \"predicate\": \"works_at\", \"object\": \"Google\", \"confidence\": 0.95}]"}
  ]
}
```

**DPO (preference pairs JSONL)**:

```json
{
  "prompt": "Extract relations from: Alice works at Google...",
  "chosen": "[{\"subject\": \"Alice\", \"predicate\": \"works_at\", \"object\": \"Google\"}]",
  "rejected": "[{\"subject\": \"Alice\", \"predicate\": \"related_to\", \"object\": \"Google\"}]"
}
```

---

## Step 2: Fine-Tuning

### Local Training (TRL SFT)

```bash
python scripts/finetune_relations.py \
    --train-data data/finetune/train_sft.jsonl \
    --val-data data/finetune/val_sft.jsonl \
    --model-name Qwen/Qwen3-1.7B \
    --output-dir models/finetuned-relations \
    --epochs 3 \
    --batch-size 4 \
    --learning-rate 2e-4 \
    --max-seq-length 2048
```

Requires a GPU with enough VRAM (8GB+ for 0.5B, 16GB+ for 1.5B, 24GB+ for 7B). Training uses LoRA and merges adapter weights into the base model before saving.

### Remote Training (Fireworks.ai)

```bash
cd memory_arena

# Upload dataset
python -c "
from scripts.fine_tune.train_slm import FireworksFineTuner
ft = FireworksFineTuner()
ft.upload_file('data/finetune/train_sft.jsonl')
"

# Create SFT job
python -c "
from scripts.fine_tune.train_slm import FireworksFineTuner, SftJobConfig
ft = FireworksFineTuner()
config = SftJobConfig(epochs=3, learning_rate=2e-4, lora_rank=16)
job = ft.create_sft_job(dataset_id='...', model_id='...', config=config)
ft.wait_for_job(job)
"

# Create DPO alignment job (after SFT)
python -c "
from scripts.fine_tune.train_slm import FireworksFineTuner, DpoJobConfig
ft = FireworksFineTuner()
config = DpoJobConfig(epochs=2, learning_rate=5e-5, lora_rank=16)
job = ft.create_dpo_job(dataset_id='...', model_id='...', config=config)
ft.wait_for_job(job)
"
```

### Projection Layer Training

For cross-modal search (CLIP -> MiniLM bridge):

```bash
# Generate paired embeddings
python -m scripts.generate_training_data.projection_dataset_generator \
    --count 50000 --output data/projection_dataset.jsonl

# Train projection MLP
python -m scripts.fine_tune.train_projection_mlp train \
    --input data/projection_dataset.jsonl \
    --output models/projection_layer.pt \
    --epochs 10

# Evaluate
python -m scripts.fine_tune.train_projection_mlp evaluate \
    --input data/projection_dataset.jsonl \
    --checkpoint models/projection_layer.pt
```

Target: validation cosine loss < 0.15.

---

## Step 3: ONNX Export

After training, convert the model to ONNX for use in `ucotron_extraction`:

```bash
python scripts/export_to_onnx.py \
    --model-dir models/finetuned-relations \
    --output-dir models/finetuned-relations-onnx
```

Deploy to Ucotron:

```bash
cp models/finetuned-relations-onnx/* models/
```

Update `ucotron.toml`:

```toml
[models]
llm_model = "finetuned-relations"
model_dir = "models/finetuned-relations-onnx"
```

---

## Configuration Reference

All Fireworks.ai settings are in `scripts/config/fireworks_config.yaml`.

### Key Sections

| Section | Purpose |
|---------|---------|
| `api` | Connection URLs, timeout (120s), retries (3x exponential backoff) |
| `models.generation` | GLM-4 Plus for dataset synthesis |
| `models.fine_tuning` | Qwen tiers (SLM/small/medium) |
| `training.sft` | SFT hyperparameters per model tier |
| `training.dpo` | DPO hyperparameters per model tier |
| `generation.tasks` | Per-task generation settings and target sample counts |
| `output` | Data and model output directories |
| `polling` | Job status polling interval (30s) |
| `tracking` | Optional W&B integration |

### SFT Hyperparameters by Tier

| Parameter | SLM (0.5B) | Small (1.5B) | Medium (7B) |
|-----------|-----------|------------|------------|
| Epochs | 3 | 3 | 2 |
| Learning rate | 2e-4 | 2e-4 | 1e-4 |
| LoRA rank | 8 | 16 | 16 |
| Context length | 2048 | 2048 | 4096 |

### DPO Hyperparameters by Tier

| Parameter | SLM (0.5B) | Small (1.5B) | Medium (7B) |
|-----------|-----------|------------|------------|
| Epochs | 2 | 2 | 1 |
| Learning rate | 5e-5 | 5e-5 | 3e-5 |
| LoRA rank | 8 | 16 | 16 |
| Context length | 2048 | 2048 | 4096 |

---

## Cost and Timing Estimates

### Fireworks.ai Remote Training

| Model Tier | SFT (10k samples, 3 epochs) | DPO (5k samples, 2 epochs) |
|------------|----------------------------|---------------------------|
| SLM (0.5B) | ~$2-5, ~15 min | ~$1-3, ~10 min |
| Small (1.5B) | ~$5-15, ~30 min | ~$3-8, ~20 min |
| Medium (7B) | ~$15-40, ~2 hr | ~$8-20, ~1 hr |

### Local Training (GPU)

| Model Tier | GPU VRAM | Time (10k samples, 3 epochs) |
|------------|----------|------------------------------|
| SLM (0.5B) | 8 GB | ~20-40 min |
| Small (1.5B) | 16 GB | ~1-2 hr |
| Medium (7B) | 24 GB | ~4-8 hr |

### Dataset Generation (Fireworks GLM-4 Plus)

| Dataset | Samples | Estimated Cost | Time |
|---------|---------|---------------|------|
| Relation extraction | 10,000 | ~$2-5 | ~30 min |
| DPO preference pairs | 5,000 | ~$2-4 | ~20 min |
| Contradiction detection | 3,000 | ~$1-2 | ~10 min |
| Entity resolution | 2,000 | ~$0.5-1 | ~5 min |
| Projection embeddings | 50,000 | ~$5-10 | ~1 hr |

Total for a full pipeline run: ~$25-75 depending on model tier.

---

## Model Evaluation

### Test Prompts

The `train_slm.py` client includes 5 default evaluation prompts:

1. "Alice works at Google" (person -> works_at -> org)
2. "Dr. Smith diagnoses patient with diabetes" (person -> diagnoses -> condition)
3. "Eiffel Tower designed by Gustave Eiffel" (landmark -> designed_by -> person)
4. "Maria lives in Berlin, studies at TU Berlin" (multi-relation)
5. "Meeting at New York office with John, Sarah, Mike" (multi-entity)

### Running Evaluation

```bash
python -c "
from scripts.fine_tune.train_slm import FireworksFineTuner
ft = FireworksFineTuner()
ft.test_model(model_id='accounts/YOUR_ACCOUNT/models/ucotron-re-slm')
ft.compare_models(model_ids=['base-model', 'finetuned-model'])
"
```

---

## Environment Variables

| Variable | Required | Purpose |
|----------|----------|---------|
| `FIREWORKS_API_KEY` | For LLM generation & remote training | Fireworks.ai API authentication |
| `FIREWORKS_ACCOUNT_ID` | For remote fine-tuning jobs | Fireworks account for job management |
| `WANDB_API_KEY` | No | Weights & Biases experiment tracking |
| `UCOTRON_SERVER_URL` | No | Live server URL for path reward generation (default: localhost:8420) |
| `UCOTRON_API_KEY` | No | API key for Ucotron server access |
| `UCOTRON_NAMESPACE` | No | Default namespace (fallback: "default") |

---

## File Reference

| Path | Purpose |
|------|---------|
| `scripts/finetune_pipeline.sh` | End-to-end pipeline orchestrator |
| `scripts/finetune_relations.py` | Local TRL SFT training script |
| `scripts/export_to_onnx.py` | ONNX model export |
| `scripts/config/fireworks_config.yaml` | Fireworks.ai configuration |
| `scripts/fine_tune/train_slm.py` | Fireworks.ai remote fine-tuning client |
| `scripts/fine_tune/train_projection_mlp.py` | Cross-modal projection layer trainer |
| `scripts/generate_training_data/*.py` | LLM-powered dataset generators |
| `ucotron_extraction/src/fine_tuning.rs` | Rust dataset generation from graph |
| `models/` | ONNX model directory (gitignored) |
| `data/` | Training data directory (gitignored) |

---

## Troubleshooting

**"Missing Python dependencies for training"**: Install ML stack with `pip install torch transformers trl datasets peft accelerate`.

**"Rust/cargo not found"**: Install Rust from https://rustup.rs. On macOS, use `$HOME/.cargo/bin/cargo` directly.

**Fireworks job stuck in PENDING**: Check `ft.get_job(job_name)` for status. Jobs can take several minutes to start depending on queue depth. Use `ft.cancel_job(job_name)` to abort.

**ONNX export fails**: Ensure the fine-tuned model directory contains `config.json`. If using LoRA, the training script should have merged adapter weights before saving.

**Training loss not decreasing**: Try reducing learning rate (halve it), increasing LoRA rank, or checking that training data format matches the expected chat messages schema.
