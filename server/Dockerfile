# =============================================================================
# Ucotron Server — Multi-stage Docker Build
# =============================================================================
# Stage 1: Build Rust binaries (ucotron_server + ucotron_mcp)
# Stage 2: Download ML models (all-MiniLM-L6-v2 + GLiNER)
# Stage 3: Minimal runtime image (debian:trixie-slim)
#
# Target image size: <2GB (binaries ~30MB + models ~673MB + runtime ~80MB)
# Platforms: linux/amd64, linux/arm64
# =============================================================================

# ---------------------------------------------------------------------------
# Stage 1: Builder — compile Rust workspace in release mode
# ---------------------------------------------------------------------------
FROM rust:1.91-trixie AS builder

WORKDIR /build

# Install build dependencies for LMDB and ONNX Runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config \
    libssl-dev \
    cmake \
    libavutil-dev \
    libavformat-dev \
    libavcodec-dev \
    libavfilter-dev \
    libavdevice-dev \
    libswscale-dev \
    libswresample-dev \
    clang \
    && rm -rf /var/lib/apt/lists/*

# Copy workspace manifests first for dependency caching
COPY Cargo.toml Cargo.lock* ./
COPY core/Cargo.toml core/Cargo.toml
COPY helix_impl/Cargo.toml helix_impl/Cargo.toml
COPY bench_runner/Cargo.toml bench_runner/Cargo.toml
COPY ucotron_config/Cargo.toml ucotron_config/Cargo.toml
COPY ucotron_extraction/Cargo.toml ucotron_extraction/Cargo.toml
COPY ucotron_server/Cargo.toml ucotron_server/Cargo.toml
COPY ucotron_sdk/Cargo.toml ucotron_sdk/Cargo.toml

# Create dummy source files for dependency caching
RUN mkdir -p core/src helix_impl/src bench_runner/src ucotron_config/src \
    ucotron_extraction/src ucotron_server/src ucotron_sdk/src && \
    echo "pub fn dummy() {}" > core/src/lib.rs && \
    echo "pub fn dummy() {}" > helix_impl/src/lib.rs && \
    echo "fn main() {}" > bench_runner/src/main.rs && \
    echo "pub fn dummy() {}" > ucotron_config/src/lib.rs && \
    echo "pub fn dummy() {}" > ucotron_extraction/src/lib.rs && \
    echo "pub fn dummy() {}" > ucotron_server/src/lib.rs && \
    echo "fn main() {}" > ucotron_server/src/main.rs && \
    echo "fn main() {}" > ucotron_server/src/mcp_main.rs && \
    echo "pub fn dummy() {}" > ucotron_sdk/src/lib.rs

# Build dependencies only (cached layer)
RUN cargo build --release --bin ucotron_server --bin ucotron_mcp 2>/dev/null || true

# Copy actual source code
COPY core/ core/
COPY helix_impl/ helix_impl/
COPY bench_runner/ bench_runner/
COPY ucotron_config/ ucotron_config/
COPY ucotron_extraction/ ucotron_extraction/
COPY ucotron_server/ ucotron_server/
COPY ucotron_sdk/ ucotron_sdk/

# Touch source files to invalidate the cached dummy build
RUN find . -name "*.rs" -exec touch {} +

# Build release binaries
RUN cargo build --release --bin ucotron_server --bin ucotron_mcp

# ---------------------------------------------------------------------------
# Stage 2: Model downloader — fetch ONNX models from HuggingFace
# ---------------------------------------------------------------------------
FROM debian:trixie-slim AS models

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /models

# Download all-MiniLM-L6-v2 embedding model (~90MB)
RUN mkdir -p all-MiniLM-L6-v2 && \
    curl -sL -o all-MiniLM-L6-v2/model.onnx \
        "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/onnx/model.onnx" && \
    curl -sL -o all-MiniLM-L6-v2/tokenizer.json \
        "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/tokenizer.json"

# Download GLiNER small-v2.1 NER model (~583MB)
RUN mkdir -p gliner_small-v2.1/onnx && \
    curl -sL -o gliner_small-v2.1/onnx/model.onnx \
        "https://huggingface.co/onnx-community/gliner_small-v2.1/resolve/main/onnx/model.onnx" && \
    curl -sL -o gliner_small-v2.1/tokenizer.json \
        "https://huggingface.co/onnx-community/gliner_small-v2.1/resolve/main/tokenizer.json" && \
    curl -sL -o gliner_small-v2.1/gliner_config.json \
        "https://huggingface.co/onnx-community/gliner_small-v2.1/resolve/main/gliner_config.json"

# ---------------------------------------------------------------------------
# Stage 3: Runtime — minimal production image
# ---------------------------------------------------------------------------
FROM debian:trixie-slim AS runtime

# OCI image labels (overridden by --label in docker build-push-action)
LABEL org.opencontainers.image.title="Ucotron Server" \
      org.opencontainers.image.description="Cognitive trust infrastructure for AI" \
      org.opencontainers.image.vendor="Ucotron" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.source="https://github.com/ucotron/ucotron"

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    libavutil59 \
    libavformat61 \
    libavcodec61 \
    libswscale8 \
    libswresample5 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r ucotron && useradd -r -g ucotron -d /app ucotron

WORKDIR /app

# Copy binaries from builder
COPY --from=builder /build/target/release/ucotron_server /usr/local/bin/ucotron_server
COPY --from=builder /build/target/release/ucotron_mcp /usr/local/bin/ucotron_mcp

# Copy models from downloader (can be overridden with volume mount)
COPY --from=models /models /app/models

# Copy model download script for manual re-download if needed
COPY scripts/download_models.sh /app/scripts/download_models.sh
RUN chmod +x /app/scripts/download_models.sh

# Create data directory for LMDB storage
RUN mkdir -p /app/data /app/models && \
    chown -R ucotron:ucotron /app

# Default configuration via environment variables
ENV UCOTRON_SERVER_HOST=0.0.0.0 \
    UCOTRON_SERVER_PORT=8420 \
    UCOTRON_STORAGE_MODE=embedded \
    UCOTRON_MODELS_DIR=/app/models \
    UCOTRON_STORAGE_VECTOR_DATA_DIR=/app/data \
    UCOTRON_STORAGE_GRAPH_DATA_DIR=/app/data \
    RUST_LOG=info

# Expose REST API port
EXPOSE 8420

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -sf http://localhost:8420/api/v1/health || exit 1

USER ucotron

# Run the REST API server by default
ENTRYPOINT ["ucotron_server"]
