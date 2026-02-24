# Runbook: Disaster Recovery

## Overview

This runbook covers recovery procedures for critical failure scenarios including data corruption, server crash, disk failure, and multi-instance failover.

---

## Recovery Scenarios

### 1. Server Crash (Process Dies)

**Symptoms:** Health endpoint unreachable, Docker container in "Exited" state.

**Steps:**
```bash
# 1. Check what happened
docker compose logs --tail 100 ucotron
dmesg | grep -i oom    # Check for OOM kill

# 2. Restart
docker compose up -d

# 3. Verify health
curl -sf http://localhost:8420/api/v1/health | jq .

# 4. Check data integrity via graph size
curl -s http://localhost:8420/api/v1/metrics | jq '.total_ingestions'
```

**Notes:** LMDB is crash-safe. Committed transactions survive process crashes. Uncommitted writes (in-flight ingestions) are lost but the database remains consistent.

---

### 2. LMDB Data Corruption

**Symptoms:** Server fails to start with `MDB_INVALID` or `MDB_CORRUPTED`, or returns garbled data.

**Steps:**
```bash
# 1. Stop server
docker compose down

# 2. Move corrupted data aside
mv /path/to/data /path/to/data-corrupted-$(date +%Y%m%d)

# 3. Restore from last backup
tar xzf /backups/ucotron-YYYYMMDD.tar.gz -C /path/to/
rm -f /path/to/data/lock.mdb

# 4. Restart server
docker compose up -d

# 5. Verify data
curl -s http://localhost:8420/api/v1/metrics | jq .
```

**Prevention:**
- Run daily backups (see LMDB backup runbook)
- Monitor `ucotron_lmdb_map_usage_bytes` — map-full can lead to corruption if disk fills
- Use reliable storage (EBS, persistent SSD) — avoid network filesystems for LMDB

---

### 3. Disk Full

**Symptoms:** Ingestion fails with 500 errors, LMDB writes fail, logs show `ENOSPC`.

**Steps:**
```bash
# 1. Check disk usage
df -h /path/to/data

# 2. Free space immediately
# - Delete old backups
# - Delete old audit exports
# - Truncate logs: docker compose logs --no-log-prefix ucotron > /dev/null

# 3. If LMDB map is full, increase max_db_size in config
# Edit ucotron.toml:
#   storage.vector.max_db_size = 21474836480  # 20 GB

# 4. Restart server
docker compose restart ucotron
```

---

### 4. Model Files Missing or Corrupted

**Symptoms:** Health endpoint shows `embedder_loaded: false` or `ner_loaded: false`.

**Steps:**
```bash
# 1. Check which models failed
curl -s http://localhost:8420/api/v1/health | jq '.models'

# 2. Re-download models
./scripts/download_models.sh
./scripts/download_multimodal_models.sh  # if using multimodal

# 3. Verify files exist
ls -la models/all-MiniLM-L6-v2/model.onnx
ls -la models/gliner_small-v2.1/onnx/model.onnx

# 4. Restart server
docker compose restart ucotron
```

---

### 5. Multi-Instance Writer Failure

**Symptoms:** Writer instance is down; reader instances return stale data.

**Steps:**
```bash
# 1. Check writer status
docker compose -f docker-compose.multi.yml ps ucotron-writer

# 2. Restart writer
docker compose -f docker-compose.multi.yml restart ucotron-writer

# 3. If writer data is corrupted, restore from backup
docker compose -f docker-compose.multi.yml down
# Restore shared volume (see LMDB backup runbook)
docker compose -f docker-compose.multi.yml up -d

# 4. Verify all instances
for port in 8420 8421 8422; do
  echo "Port $port:"
  curl -sf http://localhost:$port/api/v1/health | jq '.status'
done
```

**Notes:** LMDB supports single-writer, multi-reader. Readers are not affected by writer crashes (they see the last committed state). However, no new writes can occur until the writer is back.

---

### 6. Full Re-ingestion (Nuclear Option)

If data is irrecoverably lost and no backup exists:

```bash
# 1. Stop server
docker compose down

# 2. Delete all data
rm -rf /path/to/data/*

# 3. Start fresh
docker compose up -d

# 4. Re-trigger all connector syncs
for connector in slack github notion; do
  curl -X POST http://localhost:8420/api/v1/connectors/$connector/sync \
    -H "Authorization: Bearer <admin-key>"
done

# 5. Monitor re-ingestion progress
watch -n 5 'curl -s http://localhost:8420/api/v1/metrics | jq .total_ingestions'
```

---

## Recovery Checklist

After any recovery, verify:

- [ ] Health endpoint returns `"status": "ok"`
- [ ] All models show as loaded in health response
- [ ] `total_ingestions` count is reasonable
- [ ] Graph node/edge counts match expectations
- [ ] Search returns relevant results for test queries
- [ ] Connectors show correct schedule status
- [ ] No error spikes in `ucotron_errors_total`

---

## RPO / RTO Targets

| Scenario | RPO (Data Loss) | RTO (Recovery Time) |
|----------|-----------------|---------------------|
| Process crash | 0 (LMDB is ACID) | < 1 minute |
| Data corruption | Last backup interval | 5–15 minutes |
| Disk failure | Last backup interval | 15–30 minutes |
| Full re-ingestion | N/A | Hours (depends on data volume) |

**Recommendation:** Run daily backups with `mdb_copy -c` to keep RPO under 24 hours. For stricter RPO, increase backup frequency or use filesystem snapshots (ZFS/LVM).
