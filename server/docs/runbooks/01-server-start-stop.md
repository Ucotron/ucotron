# Runbook: Server Start / Stop

## Overview

Ucotron server (`ucotron_server`) is an Axum-based REST API that binds on port 8420 by default. It manages LMDB storage, ONNX models, connectors, and a background consolidation worker.

---

## Prerequisites

- Rust toolchain installed (or Docker image available)
- ONNX models downloaded (`scripts/download_models.sh`)
- Config file `ucotron.toml` prepared (generate template with `--init-config`)

---

## Starting the Server

### Binary (bare-metal)

```bash
# Generate example config (first time only)
ucotron_server --init-config

# Start with config file
ucotron_server --config /etc/ucotron/ucotron.toml

# Or use environment variable
UCOTRON_CONFIG=/etc/ucotron/ucotron.toml ucotron_server
```

### Docker (single instance)

```bash
docker compose -f docker-compose.yml up -d
```

### Docker (multi-instance: 1 writer + 2 readers)

```bash
docker compose -f docker-compose.multi.yml up -d
```

### Verify Startup

```bash
# Health check (no auth required)
curl -sf http://localhost:8420/api/v1/health | jq .

# Expected: {"status":"ok","version":"0.1.0",...}
```

---

## Stopping the Server

### Binary

Send SIGTERM or SIGINT (Ctrl-C). The server performs graceful shutdown:

1. Stops accepting new connections
2. Signals the consolidation worker to finish via `tokio::sync::watch`
3. Flushes pending LMDB writes
4. Exits cleanly

```bash
# Graceful stop
kill -TERM <PID>

# Or if running in foreground
Ctrl-C
```

### Docker

```bash
# Graceful stop (sends SIGTERM, waits 30s)
docker compose -f docker-compose.yml down

# Force stop if hung
docker compose -f docker-compose.yml down --timeout 10
```

---

## Key Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `UCOTRON_CONFIG` | — | Path to ucotron.toml |
| `UCOTRON_SERVER_HOST` | `0.0.0.0` | Bind address |
| `UCOTRON_SERVER_PORT` | `8420` | HTTP port |
| `UCOTRON_MODELS_DIR` | `models/` | ONNX models directory |
| `UCOTRON_STORAGE_MODE` | `embedded` | `embedded` or `shared` |
| `UCOTRON_STORAGE_VECTOR_DATA_DIR` | `data/` | Vector LMDB path |
| `UCOTRON_STORAGE_GRAPH_DATA_DIR` | `data/` | Graph LMDB path |
| `RUST_LOG` | `info` | Log verbosity (`debug`, `info`, `warn`, `error`) |

---

## Troubleshooting Startup Failures

| Symptom | Cause | Fix |
|---------|-------|-----|
| `Address already in use` | Port 8420 occupied | Change port via `UCOTRON_SERVER_PORT` or stop the other process |
| `No such file: models/...` | Models not downloaded | Run `scripts/download_models.sh` |
| `LMDB: permission denied` | Data dir not writable | Check file ownership on `data/` directory |
| `LMDB: map full` | DB exceeded max size | Increase `storage.vector.max_db_size` (default 10 GB) |
| Health endpoint returns unhealthy | Models failed to load | Check `models` section in health response; re-download corrupted model files |
