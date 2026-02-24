#!/usr/bin/env bash
# Download multimodal ONNX models (Whisper + CLIP) required by ucotron_extraction.
# These are cached separately from base models (MiniLM, GLiNER) in CI.
# Run from the workspace root (memory_arena/).

set -euo pipefail

MODELS_DIR="$(cd "$(dirname "$0")/.." && pwd)/models"

echo "=== Ucotron Multimodal Model Downloader ==="
echo "Models directory: $MODELS_DIR"

# ─── Whisper tiny (audio transcription) ─────────────────────────────────────
WHISPER_DIR="$MODELS_DIR/whisper-tiny"

echo ""
echo "=== Whisper tiny ONNX (audio transcription) ==="

mkdir -p "$WHISPER_DIR"

# Download Whisper encoder
if [ -f "$WHISPER_DIR/encoder.onnx" ]; then
    echo "[SKIP] whisper-tiny/encoder.onnx already exists"
else
    echo "[DOWNLOAD] Whisper tiny encoder (~36MB)..."
    curl -sL -o "$WHISPER_DIR/encoder.onnx" \
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/tiny-encoder.onnx"
    echo "[OK] encoder.onnx downloaded"
fi

# Download Whisper decoder
if [ -f "$WHISPER_DIR/decoder.onnx" ]; then
    echo "[SKIP] whisper-tiny/decoder.onnx already exists"
else
    echo "[DOWNLOAD] Whisper tiny decoder (~185MB)..."
    curl -sL -o "$WHISPER_DIR/decoder.onnx" \
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/tiny-decoder.onnx"
    echo "[OK] decoder.onnx downloaded"
fi

# Download Whisper tokens
if [ -f "$WHISPER_DIR/tokens.txt" ]; then
    echo "[SKIP] whisper-tiny/tokens.txt already exists"
else
    echo "[DOWNLOAD] Whisper tiny tokens.txt..."
    curl -sL -o "$WHISPER_DIR/tokens.txt" \
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/tiny-tokens.txt"
    echo "[OK] tokens.txt downloaded"
fi

# ─── CLIP ViT-B/32 (image embedding + cross-modal search) ──────────────────
CLIP_DIR="$MODELS_DIR/clip-vit-base-patch32"

echo ""
echo "=== CLIP ViT-B/32 ONNX (image embedding + cross-modal search) ==="

mkdir -p "$CLIP_DIR"

# Download CLIP visual encoder
if [ -f "$CLIP_DIR/visual_model.onnx" ]; then
    echo "[SKIP] clip-vit-base-patch32/visual_model.onnx already exists"
else
    echo "[DOWNLOAD] CLIP ViT-B/32 visual encoder (~350MB)..."
    curl -sL -o "$CLIP_DIR/visual_model.onnx" \
        "https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/onnx/visual_model.onnx"
    echo "[OK] visual_model.onnx downloaded"
fi

# Download CLIP text encoder
if [ -f "$CLIP_DIR/text_model.onnx" ]; then
    echo "[SKIP] clip-vit-base-patch32/text_model.onnx already exists"
else
    echo "[DOWNLOAD] CLIP ViT-B/32 text encoder (~250MB)..."
    curl -sL -o "$CLIP_DIR/text_model.onnx" \
        "https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/onnx/text_model.onnx"
    echo "[OK] text_model.onnx downloaded"
fi

# Download CLIP tokenizer
if [ -f "$CLIP_DIR/tokenizer.json" ]; then
    echo "[SKIP] clip-vit-base-patch32/tokenizer.json already exists"
else
    echo "[DOWNLOAD] CLIP tokenizer.json..."
    curl -sL -o "$CLIP_DIR/tokenizer.json" \
        "https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/tokenizer.json"
    echo "[OK] tokenizer.json downloaded"
fi

echo ""
echo "=== Multimodal model download complete ==="

# Show downloaded files
if [ -d "$WHISPER_DIR" ] && [ "$(ls -A "$WHISPER_DIR" 2>/dev/null)" ]; then
    echo "Whisper:"
    ls -lh "$WHISPER_DIR/"
fi
echo ""
if [ -d "$CLIP_DIR" ] && [ "$(ls -A "$CLIP_DIR" 2>/dev/null)" ]; then
    echo "CLIP:"
    ls -lh "$CLIP_DIR/"
fi

# Auto-generate checksums if the checksum file has no entries
SCRIPTS_DIR="$(cd "$(dirname "$0")" && pwd)"
CHECKSUM_FILE="$SCRIPTS_DIR/multimodal_checksums.sha256"

# Count non-comment, non-empty lines with 64-char hashes
HASH_COUNT=$(grep -cE '^[0-9a-f]{64}' "$CHECKSUM_FILE" 2>/dev/null || echo 0)

if [ "$HASH_COUNT" -eq 0 ]; then
    echo ""
    echo "=== Auto-generating multimodal checksums ==="
    bash "$SCRIPTS_DIR/generate_checksums.sh" --multimodal
fi
