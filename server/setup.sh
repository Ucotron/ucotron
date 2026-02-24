#!/usr/bin/env bash
# =============================================================================
# Ucotron Setup Script
# =============================================================================
# Downloads required models and prepares the environment for running Ucotron.
# Run this before `docker compose up` if you want to use local models directory
# (faster rebuilds — avoids re-downloading models on every Docker build).
#
# Usage:
#   ./setup.sh              # Download models and create directories
#   ./setup.sh --docker     # Just build and start with Docker Compose
#   ./setup.sh --clean      # Remove data and models directories
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODELS_DIR="$SCRIPT_DIR/models"
DATA_DIR="$SCRIPT_DIR/data"

echo "============================================"
echo "  Ucotron Setup"
echo "============================================"
echo ""

# Parse arguments
case "${1:-}" in
    --docker)
        echo "Building and starting Ucotron with Docker Compose..."
        echo ""
        docker compose -f "$SCRIPT_DIR/docker-compose.yml" up --build -d
        echo ""
        echo "Ucotron is starting up. Check status with:"
        echo "  docker compose logs -f ucotron"
        echo "  curl http://localhost:8420/api/v1/health"
        exit 0
        ;;
    --clean)
        echo "Cleaning up data and models directories..."
        rm -rf "$DATA_DIR" "$MODELS_DIR"
        echo "Done. Run ./setup.sh to re-download models."
        exit 0
        ;;
    --help|-h)
        echo "Usage: ./setup.sh [OPTIONS]"
        echo ""
        echo "Options:"
        echo "  (none)       Download models and create directories"
        echo "  --docker     Build and start with Docker Compose"
        echo "  --clean      Remove data and models directories"
        echo "  --help       Show this help message"
        exit 0
        ;;
esac

# Create directories
echo "Creating directories..."
mkdir -p "$MODELS_DIR"
mkdir -p "$DATA_DIR"
echo "  Models: $MODELS_DIR"
echo "  Data:   $DATA_DIR"
echo ""

# Download models using existing script
echo "Downloading ML models..."
echo ""
bash "$SCRIPT_DIR/scripts/download_models.sh"
echo ""

# Generate example config if none exists
if [ ! -f "$SCRIPT_DIR/ucotron.toml" ]; then
    echo "No ucotron.toml found. You can generate an example with:"
    echo "  cargo run --release --bin ucotron_server -- --init-config > ucotron.toml"
    echo ""
fi

echo "============================================"
echo "  Setup Complete!"
echo "============================================"
echo ""
echo "Quick start:"
echo ""
echo "  # Option 1: Docker Compose (recommended)"
echo "  docker compose up --build"
echo ""
echo "  # Option 2: Run locally (requires Rust toolchain)"
echo "  cargo run --release --bin ucotron_server"
echo ""
echo "  # Option 3: Docker Compose with local models (faster rebuilds)"
echo "  # Uncomment the models volume in docker-compose.yml, then:"
echo "  docker compose up --build"
echo ""
echo "Endpoints:"
echo "  REST API:     http://localhost:8420/api/v1/"
echo "  Health check: http://localhost:8420/api/v1/health"
echo "  Metrics:      http://localhost:8420/api/v1/metrics"
echo ""
