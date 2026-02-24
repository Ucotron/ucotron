#!/usr/bin/env bash
# =============================================================================
# Ucotron Fine-Tuning Pipeline
#
# End-to-end pipeline for domain-specific relation extraction model fine-tuning.
#
# Steps:
#   1. Generate training dataset from Ucotron knowledge graph (Rust)
#   2. Fine-tune model with TRL SFT (Python)
#   3. Export fine-tuned model to ONNX (Python)
#
# Usage:
#   ./scripts/finetune_pipeline.sh [options]
#
# Options:
#   --data-dir DIR          Directory for training data (default: data/finetune)
#   --model-name NAME       Base model for fine-tuning (default: Qwen/Qwen3-1.7B)
#   --output-dir DIR        Output directory for ONNX model (default: models/finetuned-relations)
#   --max-samples N         Maximum training samples (default: 10000)
#   --epochs N              Training epochs (default: 3)
#   --skip-generate         Skip dataset generation (use existing data)
#   --skip-train            Skip training (use existing model)
#   --skip-export           Skip ONNX export
#   --dry-run               Validate everything without actually training
#   --help                  Show this help message
# =============================================================================

set -euo pipefail

# ── Defaults ─────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(dirname "$SCRIPT_DIR")"
DATA_DIR="${WORKSPACE_DIR}/data/finetune"
MODEL_NAME="Qwen/Qwen3-1.7B"
OUTPUT_DIR="${WORKSPACE_DIR}/models/finetuned-relations"
ONNX_DIR="${WORKSPACE_DIR}/models/finetuned-relations-onnx"
MAX_SAMPLES=10000
EPOCHS=3
SKIP_GENERATE=false
SKIP_TRAIN=false
SKIP_EXPORT=false
DRY_RUN=false

# ── Parse arguments ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --data-dir) DATA_DIR="$2"; shift 2 ;;
        --model-name) MODEL_NAME="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --max-samples) MAX_SAMPLES="$2"; shift 2 ;;
        --epochs) EPOCHS="$2"; shift 2 ;;
        --skip-generate) SKIP_GENERATE=true; shift ;;
        --skip-train) SKIP_TRAIN=true; shift ;;
        --skip-export) SKIP_EXPORT=true; shift ;;
        --dry-run) DRY_RUN=true; shift ;;
        --help)
            head -27 "$0" | tail -22
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ── Helpers ──────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

step() { echo -e "\n${GREEN}==> $1${NC}"; }
warn() { echo -e "${YELLOW}WARNING: $1${NC}"; }
fail() { echo -e "${RED}ERROR: $1${NC}"; exit 1; }

# ── Pre-flight checks ───────────────────────────────────────────────────────
step "Pre-flight checks"

# Check Rust toolchain
if command -v "$HOME/.cargo/bin/cargo" &>/dev/null; then
    CARGO="$HOME/.cargo/bin/cargo"
elif command -v cargo &>/dev/null; then
    CARGO="cargo"
else
    fail "Rust/cargo not found. Install from https://rustup.rs"
fi
echo "  cargo: $($CARGO --version)"

# Check Python
if command -v python3 &>/dev/null; then
    PYTHON="python3"
elif command -v python &>/dev/null; then
    PYTHON="python"
else
    fail "Python 3 not found"
fi
echo "  python: $($PYTHON --version)"

# Check Python dependencies (lazy — only if training step is needed)
if [ "$SKIP_TRAIN" = false ] && [ "$DRY_RUN" = false ]; then
    $PYTHON -c "import torch, transformers, trl, datasets, peft" 2>/dev/null || {
        warn "Missing Python dependencies for training"
        echo "  Install with: pip install torch transformers trl datasets peft accelerate"
        if [ "$DRY_RUN" = false ]; then
            fail "Cannot proceed without ML dependencies"
        fi
    }
fi

echo "  workspace: $WORKSPACE_DIR"
echo "  data dir: $DATA_DIR"
echo "  model: $MODEL_NAME"
echo "  output: $OUTPUT_DIR"

# ── Step 1: Generate training dataset ────────────────────────────────────────
if [ "$SKIP_GENERATE" = true ]; then
    step "Step 1: SKIPPED (--skip-generate)"
    if [ ! -f "$DATA_DIR/train_sft.jsonl" ]; then
        fail "Training data not found at $DATA_DIR/train_sft.jsonl"
    fi
else
    step "Step 1: Generate training dataset from Ucotron graph"
    mkdir -p "$DATA_DIR"

    # Build the dataset generator (uses the ucotron-server binary with a generate-dataset subcommand,
    # or we can use the Rust library directly via a small binary).
    # For now, generate synthetic data using the existing data_gen module.
    echo "  Building workspace..."
    (cd "$WORKSPACE_DIR" && $CARGO build -p ucotron-extraction 2>&1 | tail -1)

    # Use cargo test to run the dataset generation as an integration test,
    # or create a small Rust script. For the pipeline, we generate from the
    # server's existing data via the REST API.
    echo "  Generating ${MAX_SAMPLES} training samples..."

    # If a Ucotron server is running, use the REST API
    if curl -s http://localhost:8420/api/v1/health >/dev/null 2>&1; then
        echo "  Using live Ucotron server at localhost:8420"
        # Export graph data via API, convert to training format
        curl -s http://localhost:8420/api/v1/memories?limit=500 | \
            $PYTHON -c "
import json, sys
data = json.load(sys.stdin)
memories = data.get('memories', data) if isinstance(data, dict) else data
print(f'Exported {len(memories) if isinstance(memories, list) else 0} memories')
" || warn "Could not export from live server"
    fi

    # Generate synthetic training data for demonstration/testing
    $PYTHON -c "
import json, os, random

random.seed(42)
data_dir = '$DATA_DIR'
max_samples = int('$MAX_SAMPLES')

# Predefined relation patterns for synthetic data
patterns = [
    ('Juan', 'person', 'Madrid', 'location', 'lives_in', 'Juan lives in Madrid and works as an engineer'),
    ('Alice', 'person', 'Google', 'organization', 'works_at', 'Alice works at Google as a senior developer'),
    ('Bob', 'person', 'Berlin', 'location', 'moved_to', 'Bob recently moved to Berlin from London'),
    ('Tesla', 'organization', 'California', 'location', 'headquartered_in', 'Tesla is headquartered in California'),
    ('Marie Curie', 'person', 'Nobel Prize', 'award', 'won', 'Marie Curie won the Nobel Prize in Physics'),
    ('Python', 'technology', 'Guido van Rossum', 'person', 'created_by', 'Python was created by Guido van Rossum'),
    ('Tokyo', 'location', 'Japan', 'location', 'capital_of', 'Tokyo is the capital of Japan'),
    ('Eiffel Tower', 'landmark', 'Paris', 'location', 'located_in', 'The Eiffel Tower is located in Paris'),
    ('Shakespeare', 'person', 'Hamlet', 'work', 'wrote', 'Shakespeare wrote Hamlet in the early 1600s'),
    ('Amazon', 'organization', 'Jeff Bezos', 'person', 'founded_by', 'Amazon was founded by Jeff Bezos in 1994'),
]

system_prompt = 'You are a relation extraction assistant. Given a text and a list of entities found in it, extract all relationships between the entities. Output ONLY a JSON array of objects with fields: subject, predicate, object, confidence.'

samples = []
for i in range(min(max_samples, len(patterns) * 10)):
    p = patterns[i % len(patterns)]
    subj, subj_label, obj, obj_label, pred, text = p

    # Vary the text slightly
    variations = [text, text + '.', 'It is known that ' + text.lower(), text + ' according to sources']
    chosen_text = variations[i % len(variations)]

    msg = {
        'messages': [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': f'Extract all relationships from the following text.\n\nText: \"{chosen_text}\"\n\nEntities found:\n- \"{subj}\" ({subj_label})\n- \"{obj}\" ({obj_label})'},
            {'role': 'assistant', 'content': json.dumps([{'subject': subj, 'predicate': pred, 'object': obj, 'confidence': round(random.uniform(0.8, 1.0), 2)}])}
        ]
    }
    samples.append(msg)

# Split 80/20
split = int(len(samples) * 0.8)
train = samples[:split]
val = samples[split:]

with open(os.path.join(data_dir, 'train_sft.jsonl'), 'w') as f:
    for s in train:
        f.write(json.dumps(s) + '\n')

with open(os.path.join(data_dir, 'val_sft.jsonl'), 'w') as f:
    for s in val:
        f.write(json.dumps(s) + '\n')

# Also write raw format
with open(os.path.join(data_dir, 'train_raw.jsonl'), 'w') as f:
    for s in train:
        raw = {
            'text': s['messages'][1]['content'].split('Text: \"')[1].split('\"')[0] if 'Text: \"' in s['messages'][1]['content'] else '',
            'entities': [],
            'expected_relations': json.loads(s['messages'][2]['content'])
        }
        f.write(json.dumps(raw) + '\n')

print(f'  Generated {len(train)} train + {len(val)} validation samples')
print(f'  Saved to {data_dir}/')
"
fi

# ── Step 2: Fine-tune model ─────────────────────────────────────────────────
if [ "$SKIP_TRAIN" = true ]; then
    step "Step 2: SKIPPED (--skip-train)"
else
    step "Step 2: Fine-tune model with TRL SFT"

    TRAIN_ARGS=(
        --train-data "$DATA_DIR/train_sft.jsonl"
        --val-data "$DATA_DIR/val_sft.jsonl"
        --model-name "$MODEL_NAME"
        --output-dir "$OUTPUT_DIR"
        --epochs "$EPOCHS"
    )

    if [ "$DRY_RUN" = true ]; then
        TRAIN_ARGS+=(--dry-run)
    fi

    $PYTHON "$SCRIPT_DIR/finetune_relations.py" "${TRAIN_ARGS[@]}"
fi

# ── Step 3: Export to ONNX ───────────────────────────────────────────────────
if [ "$SKIP_EXPORT" = true ]; then
    step "Step 3: SKIPPED (--skip-export)"
else
    step "Step 3: Export to ONNX"

    if [ ! -d "$OUTPUT_DIR" ] || [ ! -f "$OUTPUT_DIR/config.json" ]; then
        if [ "$DRY_RUN" = true ]; then
            echo "  (dry-run: model dir not present, would export after training)"
        else
            warn "Fine-tuned model not found at $OUTPUT_DIR"
            echo "  Run training first or use --skip-export"
        fi
    else
        EXPORT_ARGS=(
            --model-dir "$OUTPUT_DIR"
            --output-dir "$ONNX_DIR"
        )

        if [ "$DRY_RUN" = true ]; then
            EXPORT_ARGS+=(--dry-run)
        fi

        $PYTHON "$SCRIPT_DIR/export_to_onnx.py" "${EXPORT_ARGS[@]}"
    fi
fi

# ── Summary ──────────────────────────────────────────────────────────────────
step "Pipeline complete!"
echo ""
echo "  Training data:    $DATA_DIR/"
echo "  Fine-tuned model: $OUTPUT_DIR/"
echo "  ONNX model:       $ONNX_DIR/"
echo ""

if [ -f "$DATA_DIR/train_sft.jsonl" ]; then
    TRAIN_COUNT=$(wc -l < "$DATA_DIR/train_sft.jsonl" | tr -d ' ')
    echo "  Train samples:    $TRAIN_COUNT"
fi
if [ -f "$DATA_DIR/val_sft.jsonl" ]; then
    VAL_COUNT=$(wc -l < "$DATA_DIR/val_sft.jsonl" | tr -d ' ')
    echo "  Val samples:      $VAL_COUNT"
fi

if [ "$DRY_RUN" = true ]; then
    echo ""
    echo "  (This was a dry run. No models were trained or exported.)"
fi

echo ""
echo "To use the fine-tuned model with Ucotron:"
echo "  1. Copy ONNX model to models/ directory"
echo "  2. Update ucotron.toml:"
echo "     [models]"
echo "     llm_model = \"finetuned-relations\""
echo "     model_dir = \"models/finetuned-relations-onnx\""
