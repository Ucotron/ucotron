# Runbook: Monitoring & Alert Response

## Overview

Ucotron exposes Prometheus metrics at `/metrics` and a JSON snapshot at `/api/v1/metrics`. This runbook covers metric definitions, recommended alert rules, and response procedures.

---

## Metrics Endpoint

```bash
# Prometheus format (for scraping)
curl -s http://localhost:8420/metrics

# JSON snapshot (for dashboards)
curl -s http://localhost:8420/api/v1/metrics | jq .
```

---

## Metric Reference

### Counters

| Metric | Labels | Description |
|--------|--------|-------------|
| `ucotron_http_requests_total` | method, path, status | Total HTTP requests |
| `ucotron_ingestions_total` | — | Total memory ingestions |
| `ucotron_searches_total` | — | Total search operations |
| `ucotron_errors_total` | error_type | Errors (server_error, client_error) |

### Histograms

| Metric | Labels | Description |
|--------|--------|-------------|
| `ucotron_http_request_duration_seconds` | method, path, status | Request latency |
| `ucotron_ingestion_duration_seconds` | — | Ingestion pipeline latency |
| `ucotron_search_duration_seconds` | — | Search latency |
| `ucotron_model_inference_duration_seconds` | model_name | Model inference time (ner, embedding) |

Histogram buckets: 1ms, 2.5ms, 5ms, 10ms, 25ms, 50ms, 100ms, 250ms, 500ms, 1s, 2.5s, 5s, 10s

### Gauges

| Metric | Description |
|--------|-------------|
| `ucotron_uptime_seconds` | Server uptime |
| `ucotron_graph_nodes_total` | Total nodes in knowledge graph |
| `ucotron_graph_edges_total` | Total edges in knowledge graph |
| `ucotron_process_rss_bytes` | Process resident memory |
| `ucotron_lmdb_map_usage_bytes` | LMDB memory-map utilization |

---

## Recommended Alert Rules

### High Error Rate

```yaml
# Prometheus alert rule
- alert: UcotronHighErrorRate
  expr: rate(ucotron_errors_total{error_type="server_error"}[5m]) > 0.1
  for: 5m
  labels:
    severity: critical
  annotations:
    summary: "Ucotron server error rate > 0.1/s for 5 minutes"
```

**Response procedure:**
1. Check server logs: `docker compose logs --tail 100 ucotron`
2. Look for panic traces or LMDB errors
3. Check disk space: `df -h` (LMDB map-full causes 500s)
4. Check model health: `curl localhost:8420/api/v1/health | jq '.models'`
5. If persistent, restart server and check if errors clear

### High Search Latency

```yaml
- alert: UcotronHighSearchLatency
  expr: histogram_quantile(0.95, rate(ucotron_search_duration_seconds_bucket[5m])) > 0.5
  for: 10m
  labels:
    severity: warning
  annotations:
    summary: "Ucotron P95 search latency > 500ms"
```

**Response procedure:**
1. Check graph size: `ucotron_graph_nodes_total` — large graphs slow traversal
2. Check LMDB map usage: if near `max_db_size`, increase it
3. Check consolidation status — if communities are stale, trigger manual consolidation
4. Check CPU and memory: high RSS may indicate swap pressure

### LMDB Near Capacity

```yaml
- alert: UcotronLmdbNearFull
  expr: ucotron_lmdb_map_usage_bytes / (10 * 1024 * 1024 * 1024) > 0.8
  for: 1m
  labels:
    severity: warning
  annotations:
    summary: "Ucotron LMDB usage > 80% of max map size"
```

**Response procedure:**
1. Check current usage vs configured max in config
2. Increase `storage.vector.max_db_size` or `storage.graph.max_db_size`
3. Restart server for new map size to take effect
4. Consider archiving old data or increasing memory decay rate

### Server Down

```yaml
- alert: UcotronDown
  expr: up{job="ucotron"} == 0
  for: 1m
  labels:
    severity: critical
  annotations:
    summary: "Ucotron server is unreachable"
```

**Response procedure:**
1. Check if process is running: `docker compose ps` or `pgrep ucotron_server`
2. Check for OOM kill: `dmesg | grep -i oom`
3. Check logs for panic: `docker compose logs --tail 50 ucotron`
4. Restart: `docker compose up -d`
5. If repeated crashes, check data integrity (see backup/restore runbook)

### High Ingestion Latency

```yaml
- alert: UcotronHighIngestionLatency
  expr: histogram_quantile(0.95, rate(ucotron_ingestion_duration_seconds_bucket[5m])) > 2
  for: 10m
  labels:
    severity: warning
  annotations:
    summary: "Ucotron P95 ingestion latency > 2s"
```

**Response procedure:**
1. Check model inference times: `ucotron_model_inference_duration_seconds` — NER and embedding are the bottleneck
2. Check if HNSW index is rebuilding (instant-distance rebuilds entire index on upsert)
3. Consider batching ingestions or reducing NER model complexity
4. Check disk I/O — LMDB writes may be bottlenecked on slow storage

---

## OpenTelemetry (Optional)

```toml
[telemetry]
enabled = true
otlp_endpoint = "http://localhost:4317"
service_name = "ucotron"
export_traces = true
export_metrics = true
sample_rate = 0.1
```

Traces include spans for: HTTP handler, ingestion pipeline steps, model inference, LMDB operations, and graph traversal.

---

## Grafana Dashboard Queries

Key panels for a Ucotron dashboard:

```promql
# Request rate
rate(ucotron_http_requests_total[5m])

# Error rate percentage
rate(ucotron_errors_total[5m]) / rate(ucotron_http_requests_total[5m]) * 100

# P95 search latency
histogram_quantile(0.95, rate(ucotron_search_duration_seconds_bucket[5m]))

# Ingestion throughput
rate(ucotron_ingestions_total[5m])

# Graph growth
ucotron_graph_nodes_total
ucotron_graph_edges_total

# Memory usage
ucotron_process_rss_bytes / 1024 / 1024  # in MB
```
