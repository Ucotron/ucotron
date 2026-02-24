# Runbook: Audit Logging

## Overview

Ucotron maintains an append-only audit log of all API operations. The audit log is stored in memory with configurable retention and maximum entry count. It is accessible only to admin users.

---

## Configuration

```toml
[audit]
enabled = true
retention_secs = 7776000    # 90 days
max_entries = 100000         # Max entries in memory
```

Environment overrides:
- `UCOTRON_AUDIT_ENABLED=true|false`
- `UCOTRON_AUDIT_RETENTION_SECS=<seconds>`
- `UCOTRON_AUDIT_MAX_ENTRIES=<number>`

---

## Audit Entry Fields

| Field | Type | Description |
|-------|------|-------------|
| `timestamp` | u64 | Unix timestamp (seconds) |
| `method` | string | HTTP method (GET, POST, DELETE) |
| `path` | string | Request path |
| `action` | string | Derived category (e.g., "memories.create", "search") |
| `status` | u16 | HTTP status code |
| `duration_us` | u64 | Request duration in microseconds |
| `user` | string? | API key name (if authenticated) |
| `role` | string | Caller's role |
| `namespace` | string | Target namespace |
| `resource_id` | string? | Target resource ID |

---

## Querying the Audit Log

### List entries with filters (admin only)

```bash
curl -s "http://localhost:8420/api/v1/audit?action=memories.create&from_timestamp=1700000000" \
  -H "Authorization: Bearer <admin-key>" | jq .
```

Available query parameters:
- `from_timestamp` — Start of time range (Unix seconds)
- `to_timestamp` — End of time range
- `action` — Filter by action type
- `user` — Filter by API key name
- `role` — Filter by role
- `namespace` — Filter by namespace
- `status` — Filter by HTTP status code

### Export full audit log (admin only)

```bash
curl -s http://localhost:8420/api/v1/audit/export \
  -H "Authorization: Bearer <admin-key>" > audit-export-$(date +%Y%m%d).json
```

---

## Skipped Endpoints

These endpoints are not audited to reduce noise:
- `GET /api/v1/health`
- `GET /metrics`
- `POST /api/v1/webhooks/*`

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| Audit endpoint returns 403 | Non-admin key | Use admin API key |
| Audit entries missing | `audit.enabled = false` | Enable in config |
| Old entries disappearing | Retention exceeded | Increase `retention_secs` or export before expiry |
| Audit query slow | Too many entries matching filter | Add more specific filters (time range, action) |
| Memory usage growing | `max_entries` too high | Lower `max_entries` or reduce retention |

---

## Compliance Workflows

### Periodic export for external storage

```bash
# Cron job: export audit log daily, upload to S3
0 3 * * * curl -s http://localhost:8420/api/v1/audit/export \
  -H "Authorization: Bearer <admin-key>" \
  | gzip > /tmp/audit-$(date +\%Y\%m\%d).json.gz \
  && aws s3 cp /tmp/audit-$(date +\%Y\%m\%d).json.gz s3://ucotron-audit/
```

### Investigate a specific incident

```bash
# Find all operations by a specific user in a time window
curl -s "http://localhost:8420/api/v1/audit?user=ml-service&from_timestamp=1700000000&to_timestamp=1700003600" \
  -H "Authorization: Bearer <admin-key>" | jq '.entries[] | {timestamp, action, path, status}'
```
