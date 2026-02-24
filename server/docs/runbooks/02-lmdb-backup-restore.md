# Runbook: LMDB Backup & Restore

## Overview

Ucotron uses LMDB (via the `heed` crate) for both vector and graph storage. LMDB is a memory-mapped B-tree database that supports concurrent readers with a single writer. This enables hot backups without stopping the server.

---

## Data Layout

| Component | Default Path | Config Key |
|-----------|-------------|------------|
| Vector LMDB | `data/` | `storage.vector.data_dir` |
| Graph LMDB | `data/` | `storage.graph.data_dir` |
| Media files | `data/media/` | `storage.media_dir` |
| Shared mode | `data/` | `storage.shared_data_dir` |

LMDB stores data in two files per environment:
- `data.mdb` — The actual database (can be up to `max_db_size`, default 10 GB)
- `lock.mdb` — Reader lock table (small, regenerated on open)

---

## Hot Backup (Server Running)

LMDB guarantees read consistency. You can safely copy the data files while the server is running.

### Using mdb_copy (recommended)

```bash
# Install lmdb-utils (provides mdb_copy)
# macOS: brew install lmdb
# Ubuntu: apt install lmdb-utils

# Compact copy (reclaims free pages, smaller file)
mdb_copy -c /path/to/data/ /backups/ucotron-$(date +%Y%m%d).mdb

# Direct copy (faster, preserves free space)
mdb_copy /path/to/data/ /backups/ucotron-$(date +%Y%m%d)/
```

### Using filesystem copy

```bash
# Copy entire data directory
cp -r /path/to/data/ /backups/ucotron-$(date +%Y%m%d)/

# Do NOT copy lock.mdb — it is regenerated on open
rm -f /backups/ucotron-$(date +%Y%m%d)/lock.mdb
```

### Docker backup

```bash
# Copy from named volume
docker run --rm \
  -v ucotron_data:/data:ro \
  -v /backups:/backups \
  alpine tar czf /backups/ucotron-$(date +%Y%m%d).tar.gz -C /data .
```

---

## Cold Backup (Server Stopped)

For maximum consistency, stop the server first:

```bash
# 1. Stop server
docker compose down  # or: kill -TERM <PID>

# 2. Archive data directory
tar czf /backups/ucotron-$(date +%Y%m%d).tar.gz -C /path/to data/

# 3. Restart server
docker compose up -d
```

---

## Restore from Backup

### Bare-metal

```bash
# 1. Stop server
kill -TERM <PID>

# 2. Remove current data
rm -rf /path/to/data/

# 3. Restore from backup
tar xzf /backups/ucotron-20260101.tar.gz -C /path/to/

# 4. Remove stale lock file
rm -f /path/to/data/lock.mdb

# 5. Start server
ucotron_server --config /etc/ucotron/ucotron.toml
```

### Docker

```bash
# 1. Stop services
docker compose down

# 2. Remove old volume data and restore
docker run --rm \
  -v ucotron_data:/data \
  -v /backups:/backups \
  alpine sh -c "rm -rf /data/* && tar xzf /backups/ucotron-20260101.tar.gz -C /data/"

# 3. Start services
docker compose up -d
```

---

## Automated Backup Schedule

Example cron job for daily hot backup with 7-day retention:

```bash
# /etc/cron.d/ucotron-backup
0 2 * * * root mdb_copy -c /app/data/ /backups/ucotron-$(date +\%Y\%m\%d).mdb && find /backups -name 'ucotron-*.mdb' -mtime +7 -delete
```

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `MDB_MAP_FULL` after restore | Backup larger than configured max | Increase `storage.*.max_db_size` in config |
| `MDB_INVALID` on startup | Corrupted data file | Restore from last known-good backup |
| Stale `lock.mdb` prevents open | Unclean shutdown | Delete `lock.mdb`; LMDB regenerates it |
| Backup file is 0 bytes | Disk full during copy | Free disk space and retry |
| Data directory missing after Docker restart | Volume not mounted | Verify `volumes:` in docker-compose.yml |
