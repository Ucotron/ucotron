#!/usr/bin/env bash
# Download base ONNX models (MiniLM, GLiNER) required by ucotron_extraction.
# For multimodal models (Whisper, CLIP), use download_multimodal_models.sh.
# Run from the workspace root (memory_arena/).

set -euo pipefail

MODELS_DIR="$(cd "$(dirname "$0")/.." && pwd)/models"
MINILM_DIR="$MODELS_DIR/all-MiniLM-L6-v2"

echo "=== Ucotron Model Downloader ==="
echo "Models directory: $MODELS_DIR"

mkdir -p "$MINILM_DIR"

# Download all-MiniLM-L6-v2 ONNX model (~90MB)
if [ -f "$MINILM_DIR/model.onnx" ]; then
    echo "[SKIP] model.onnx already exists"
else
    echo "[DOWNLOAD] all-MiniLM-L6-v2 ONNX model (~90MB)..."
    curl -sL -o "$MINILM_DIR/model.onnx" \
        "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/onnx/model.onnx"
    echo "[OK] model.onnx downloaded"
fi

# Download tokenizer.json (~466KB)
if [ -f "$MINILM_DIR/tokenizer.json" ]; then
    echo "[SKIP] tokenizer.json already exists"
else
    echo "[DOWNLOAD] tokenizer.json..."
    curl -sL -o "$MINILM_DIR/tokenizer.json" \
        "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/tokenizer.json"
    echo "[OK] tokenizer.json downloaded"
fi

# --- GLiNER small-v2.1 (NER model) ---
GLINER_DIR="$MODELS_DIR/gliner_small-v2.1"
GLINER_ONNX_DIR="$GLINER_DIR/onnx"

mkdir -p "$GLINER_ONNX_DIR"

# Download GLiNER ONNX model
if [ -f "$GLINER_ONNX_DIR/model.onnx" ]; then
    echo "[SKIP] gliner_small-v2.1/onnx/model.onnx already exists"
else
    echo "[DOWNLOAD] GLiNER small-v2.1 ONNX model (this may take a while)..."
    curl -sL -o "$GLINER_ONNX_DIR/model.onnx" \
        "https://huggingface.co/onnx-community/gliner_small-v2.1/resolve/main/onnx/model.onnx"
    echo "[OK] gliner model.onnx downloaded"
fi

# Download GLiNER tokenizer.json
if [ -f "$GLINER_DIR/tokenizer.json" ]; then
    echo "[SKIP] gliner_small-v2.1/tokenizer.json already exists"
else
    echo "[DOWNLOAD] GLiNER tokenizer.json..."
    curl -sL -o "$GLINER_DIR/tokenizer.json" \
        "https://huggingface.co/onnx-community/gliner_small-v2.1/resolve/main/tokenizer.json"
    echo "[OK] gliner tokenizer.json downloaded"
fi

# Download GLiNER config files
if [ -f "$GLINER_DIR/gliner_config.json" ]; then
    echo "[SKIP] gliner_config.json already exists"
else
    echo "[DOWNLOAD] gliner_config.json..."
    curl -sL -o "$GLINER_DIR/gliner_config.json" \
        "https://huggingface.co/onnx-community/gliner_small-v2.1/resolve/main/gliner_config.json"
    echo "[OK] gliner_config.json downloaded"
fi

# --- Qwen3-4B GGUF (LLM for relation extraction — optional) ---
QWEN_DIR="$MODELS_DIR/Qwen3-4B-GGUF"

echo ""
echo "=== Qwen3-4B GGUF (optional, for LLM relation extraction) ==="
echo "The Qwen3-4B model (~2.5GB) is NOT downloaded automatically."
echo "To enable LLM-based relation extraction, download it manually:"
echo ""
echo "  mkdir -p $QWEN_DIR"
echo "  curl -L -o $QWEN_DIR/qwen3-4b-q4_k_m.gguf \\"
echo "    'https://huggingface.co/Qwen/Qwen3-4B-GGUF/resolve/main/qwen3-4b-q4_k_m.gguf'"
echo ""
echo "Without this model, Ucotron uses co-occurrence-based relation extraction (no LLM needed)."

echo ""
echo "=== Download complete ==="
echo "Model files:"
ls -lh "$MINILM_DIR/"
echo ""
ls -lhR "$GLINER_DIR/"
echo ""
echo "For multimodal models (Whisper, CLIP), run: bash scripts/download_multimodal_models.sh"
