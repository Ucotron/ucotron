# Ucotron

The open-source trust layer for AI. High-performance graph + vector memory layer that enables AI agents to remember, reason, and learn across conversations.

## Features

- **Graph + Vector Storage** — LMDB-backed HNSW vector index with property graph traversal
- **LazyGraphRAG Pipeline** — 8-step ingestion and retrieval without LLM calls at index time
- **NER + Relation Extraction** — Zero-shot entity recognition via GLiNER ONNX, co-occurrence relation extraction
- **Community Detection** — Leiden algorithm for automatic topic clustering
- **Cognitive Layer** — Entity resolution, contradiction detection, temporal decay
- **Multimodal** — Audio (Whisper), image (CLIP), video (FFmpeg), and document (OCR) ingestion
- **Multi-Tenancy** — Namespace isolation with RBAC (admin, writer, reader, viewer)
- **REST API + MCP** — Axum HTTP server and Model Context Protocol (stdio) server
- **Dashboard** — Next.js admin panel for managing memories, entities, graph visualization, and tracing
- **SDKs** — TypeScript, Python, Go, Java, PHP

## Project Structure

```
ucotron/
├── server/          # Rust server (Axum REST API + MCP)
│   ├── core/        # Core types, backends, adapters
│   ├── ucotron_server/  # HTTP handlers, auth, metrics
│   ├── ucotron_config/  # Configuration management
│   ├── ucotron_extraction/  # NER, embeddings, OCR, transcription
│   ├── ucotron_connectors/  # External data source connectors
│   └── ucotron_helix/  # Storage backend initialization
├── dashboard/       # Next.js admin dashboard
│   └── src/
│       ├── app/     # App Router pages
│       ├── components/  # UI components
│       └── lib/     # API client, auth, utilities
└── README.md
```

## Quick Start

### Server

```bash
cd server

# Download ONNX models
bash scripts/download_models.sh

# Build and run
cargo run --bin ucotron_server
```

The server starts at `http://localhost:8420`. Swagger UI is available at `/swagger-ui`.

### Dashboard

```bash
cd dashboard

# Install dependencies
npm install

# Configure environment
cp .env.example .env.local
# Edit .env.local with your settings

# Run dev server
npm run dev
```

The dashboard starts at `http://localhost:3000`.

### Configuration

Server configuration via `ucotron.toml`:

```bash
# Generate example config with documentation
cargo run --bin ucotron_server -- --init-config > ucotron.toml

# Start with config
cargo run --bin ucotron_server -- --config ucotron.toml
```

Environment variable overrides are also supported (e.g., `UCOTRON_SERVER_PORT=9000`).

## API Overview

| Endpoint | Description |
|----------|-------------|
| `POST /api/v1/memories` | Create a memory node |
| `POST /api/v1/memories/search` | Semantic search |
| `POST /api/v1/augment` | Augment LLM context with relevant memories |
| `POST /api/v1/learn` | Extract and store memories from agent output |
| `GET /api/v1/entities` | List knowledge graph entities |
| `GET /api/v1/graph` | Graph visualization data |
| `POST /api/v1/memories/audio` | Audio ingestion (Whisper) |
| `POST /api/v1/memories/image` | Image ingestion (CLIP) |
| `POST /api/v1/memories/video` | Video ingestion (FFmpeg) |
| `POST /api/v1/ocr` | Document OCR |

Full API docs at `/swagger-ui` when the server is running.

## Docker

```bash
docker build -t ucotron -f server/Dockerfile server/
docker run -p 8420:8420 ucotron
```

## Documentation

- [Architecture](server/ARCHITECTURE.md)
- [Pipelines](server/PIPELINES.md)
- [Benchmarks](server/BENCHMARKS.md)
- [Contributing](server/CONTRIBUTING.md)

## License

MIT OR Apache-2.0
