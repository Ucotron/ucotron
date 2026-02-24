#!/usr/bin/env bash
# Download benchmark datasets for external evaluations.
# Run from the workspace root (memory_arena/).

set -euo pipefail

DATA_DIR="$(cd "$(dirname "$0")/.." && pwd)/data/benchmarks"

echo "=== Ucotron Benchmark Data Downloader ==="
echo "Data directory: $DATA_DIR"

mkdir -p "$DATA_DIR/longmemeval"
mkdir -p "$DATA_DIR/locomo"

# Download LongMemEval (ICLR 2025) — cleaned version
LONGMEMEVAL_BASE="https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main"

for file in longmemeval_oracle.json longmemeval_s_cleaned.json; do
    if [ -f "$DATA_DIR/longmemeval/$file" ]; then
        echo "[SKIP] $file already exists"
    else
        echo "[DOWNLOAD] $file..."
        curl -sL -o "$DATA_DIR/longmemeval/$file" "$LONGMEMEVAL_BASE/$file"
        echo "[OK] $file downloaded"
    fi
done

# Download LoCoMo (ACL 2024) — multi-session conversational memory
LOCOMO_BASE="https://raw.githubusercontent.com/snap-research/locomo/main/data"

for file in locomo10.json; do
    if [ -f "$DATA_DIR/locomo/$file" ]; then
        echo "[SKIP] $file already exists"
    else
        echo "[DOWNLOAD] $file..."
        curl -sL -o "$DATA_DIR/locomo/$file" "$LOCOMO_BASE/$file"
        echo "[OK] $file downloaded"
    fi
done

echo ""
echo "=== Download Summary ==="
echo "LongMemEval files:"
ls -lh "$DATA_DIR/longmemeval/" 2>/dev/null || echo "  (empty)"
echo "LoCoMo files:"
ls -lh "$DATA_DIR/locomo/" 2>/dev/null || echo "  (empty)"
echo ""
echo "Done! Run benchmarks with:"
echo "  cargo test -p ucotron-core -- longmemeval"
echo "  cargo test -p ucotron-core -- locomo"
echo "  # Or with real data:"
echo "  cargo run -p bench-runner -- eval --dataset data/benchmarks/longmemeval/longmemeval_oracle.json"
echo "  cargo run -p bench-runner -- eval --dataset data/benchmarks/locomo/locomo10.json"
