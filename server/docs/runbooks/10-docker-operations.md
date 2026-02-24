# Runbook: Docker Operations

## Overview

Ucotron is packaged as a multi-stage Docker image containing compiled Rust binaries and ONNX models. Two compose files are provided: single-instance and multi-instance with shared LMDB storage.

---

## Image Details

- **Base:** `debian:bookworm-slim`
- **Size:** ~2 GB (includes ONNX models)
- **Binaries:** `ucotron_server`, `ucotron_mcp`
- **Build stages:** Builder (Rust compile) → Models (HuggingFace download) → Runtime

---

## Single Instance

### Start

```bash
docker compose -f docker-compose.yml up -d
```

### Configuration

| Setting | Value |
|---------|-------|
| Port | 8420:8420 |
| Volume | `ucotron_data:/app/data` |
| Health check | `curl -sf http://localhost:8420/api/v1/health` every 30s |
| Start period | 15s (grace period for model loading) |
| Retries | 3 consecutive failures = unhealthy |

### Logs

```bash
docker compose logs -f ucotron
docker compose logs --tail 100 ucotron
```

### Restart

```bash
docker compose restart ucotron
```

### Stop and remove

```bash
docker compose down           # Stop containers, keep volumes
docker compose down -v        # Stop containers AND delete volumes (DATA LOSS)
```

---

## Multi-Instance (Shared Storage)

### Architecture

```
┌──────────────┐  ┌───────────────┐  ┌───────────────┐
│ ucotron-writer│  │ucotron-reader-1│  │ucotron-reader-2│
│  port 8420    │  │  port 8421     │  │  port 8422     │
└──────┬───────┘  └───────┬────────┘  └───────┬────────┘
       │                  │                    │
       └──────────────────┼────────────────────┘
                          │
              ┌───────────▼──────────┐
              │  ucotron_shared_data  │
              │  (LMDB single-writer, │
              │   multi-reader)       │
              └──────────────────────┘
```

### Start

```bash
docker compose -f docker-compose.multi.yml up -d
```

### Instance Environment Variables

| Variable | Writer | Reader |
|----------|--------|--------|
| `UCOTRON_INSTANCE_ID` | `writer-1` | `reader-1`, `reader-2` |
| `UCOTRON_INSTANCE_ROLE` | `writer` | `reader` |
| `UCOTRON_STORAGE_MODE` | `shared` | `shared` |
| `UCOTRON_STORAGE_SHARED_DATA_DIR` | `/app/data` | `/app/data` |
| `UCOTRON_INSTANCE_ID_RANGE_START` | `1000000` | — |
| `UCOTRON_INSTANCE_ID_RANGE_SIZE` | `1000000` | — |

### Check all instances

```bash
for port in 8420 8421 8422; do
  echo "=== Port $port ==="
  curl -sf http://localhost:$port/api/v1/health | jq '{status, instance_id, instance_role}'
done
```

### Scale readers

```bash
docker compose -f docker-compose.multi.yml up -d --scale ucotron-reader=3
```

---

## Building the Image

```bash
# Standard build
docker compose build

# No-cache rebuild (after model or code changes)
docker compose build --no-cache

# Build with specific tag
docker build -t ucotron:v1.0.0 .
```

---

## Volume Management

```bash
# List volumes
docker volume ls | grep ucotron

# Inspect volume
docker volume inspect ucotron_data

# Backup volume (see LMDB backup runbook)
docker run --rm -v ucotron_data:/data:ro -v $(pwd):/backup \
  alpine tar czf /backup/ucotron-data.tar.gz -C /data .

# Restore volume
docker run --rm -v ucotron_data:/data -v $(pwd):/backup \
  alpine sh -c "rm -rf /data/* && tar xzf /backup/ucotron-data.tar.gz -C /data/"
```

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| Container keeps restarting | Health check failing | Check logs: `docker compose logs ucotron` |
| Port already in use | Another container or process on 8420 | `lsof -i :8420` to find conflicting process |
| Volume data lost | Used `docker compose down -v` | Restore from backup; avoid `-v` flag |
| Image build fails at models stage | HuggingFace rate limit or network issue | Retry build, or pre-download models and COPY into image |
| `exec format error` | Wrong architecture (arm64 vs amd64) | Rebuild for correct platform: `docker buildx build --platform linux/amd64` |
| Readers see stale data | LMDB read transaction not refreshed | Restart reader instances |
