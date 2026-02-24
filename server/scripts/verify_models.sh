#!/usr/bin/env bash
# Verify ONNX model integrity via SHA-256 checksums.
# Returns 0 if all models present and valid, 1 otherwise.
# Run from the workspace root (memory_arena/).
#
# Usage:
#   bash scripts/verify_models.sh              # Verify base models (MiniLM, GLiNER)
#   bash scripts/verify_models.sh --multimodal  # Verify multimodal models (Whisper, CLIP)
#   bash scripts/verify_models.sh --all         # Verify all models

set -euo pipefail

SCRIPTS_DIR="$(cd "$(dirname "$0")" && pwd)"
MODELS_DIR="$(cd "$(dirname "$0")/.." && pwd)/models"
MODE="${1:-base}"
FAILED=0

# Determine which checksum file to use
case "$MODE" in
    --multimodal)
        CHECKSUM_FILE="$SCRIPTS_DIR/multimodal_checksums.sha256"
        ;;
    --all)
        # Verify both
        echo "Verifying base models..."
        bash "$SCRIPTS_DIR/verify_models.sh" || FAILED=1
        echo ""
        echo "Verifying multimodal models..."
        bash "$SCRIPTS_DIR/verify_models.sh" --multimodal || FAILED=1
        exit $FAILED
        ;;
    *)
        CHECKSUM_FILE="$SCRIPTS_DIR/model_checksums.sha256"
        ;;
esac

if [ ! -f "$CHECKSUM_FILE" ]; then
    echo "[ERROR] Checksum file not found: $CHECKSUM_FILE"
    exit 1
fi

echo "=== Ucotron Model Integrity Verification ==="
echo "Models directory: $MODELS_DIR"
echo "Checksum file:    $CHECKSUM_FILE"
echo ""

ENTRY_COUNT=0

while IFS='  ' read -r expected_hash rel_path; do
    # Skip comments and empty lines
    [[ -z "$rel_path" || "$expected_hash" == \#* ]] && continue

    # Skip lines without actual hashes (manifest entries without checksums)
    if [[ ${#expected_hash} -ne 64 ]]; then
        continue
    fi

    ENTRY_COUNT=$((ENTRY_COUNT + 1))
    full_path="$MODELS_DIR/$rel_path"

    if [ ! -f "$full_path" ]; then
        echo "[MISSING] $rel_path"
        FAILED=1
        continue
    fi

    # Compute SHA-256 (macOS uses shasum, Linux uses sha256sum)
    if command -v sha256sum &>/dev/null; then
        actual_hash=$(sha256sum "$full_path" | awk '{print $1}')
    else
        actual_hash=$(shasum -a 256 "$full_path" | awk '{print $1}')
    fi

    if [ "$actual_hash" = "$expected_hash" ]; then
        echo "[OK]      $rel_path"
    else
        echo "[MISMATCH] $rel_path"
        echo "  Expected: $expected_hash"
        echo "  Actual:   $actual_hash"
        FAILED=1
    fi
done < "$CHECKSUM_FILE"

echo ""
if [ "$ENTRY_COUNT" -eq 0 ]; then
    echo "=== No checksum entries found (models not yet downloaded) ==="
    echo "Run scripts/download_multimodal_models.sh then scripts/generate_checksums.sh --multimodal"
    # In CI, fail if UCOTRON_MODELS_DIR is set (models should be available)
    if [ -n "${UCOTRON_MODELS_DIR:-}" ] || [ -n "${CI:-}" ] || [ -n "${GITHUB_ACTIONS:-}" ]; then
        echo "[WARN] Running in CI but no checksums found — generate checksums after downloading models"
        exit 1
    fi
    exit 0
fi

if [ "$FAILED" -eq 0 ]; then
    echo "=== All models verified successfully ($ENTRY_COUNT files) ==="
    exit 0
else
    echo "=== Model verification FAILED ==="
    exit 1
fi
