# Runbook: Connector Sync

## Overview

Ucotron connectors synchronize data from external sources (Slack, GitHub, Notion, etc.) into the knowledge graph. Sync can be triggered manually via API, on a cron schedule, or via incoming webhooks.

---

## Supported Connectors

| Connector | Auth Type | Webhook Support |
|-----------|-----------|----------------|
| Slack | OAuth token | Yes |
| GitHub | Personal access token | Yes |
| GitLab | Personal access token | Yes |
| Bitbucket | App password | Yes |
| Notion | Integration token | No |
| Google Docs | Service account JSON | No |
| Confluence | API token | No |
| Jira | API token | Yes |
| Linear | API key | Yes |
| Discord | Bot token | No |
| Email (IMAP) | App password | No |
| RSS/Atom | — | No |
| Filesystem | — | No |
| S3 | Access key + secret | No |

---

## Configuration

```toml
[connectors]
enabled = true
check_interval_secs = 60    # How often the scheduler checks cron jobs

[[connectors.schedules]]
connector_id = "my-slack"
cron_expression = "0 */6 * * * *"   # 6-field: sec min hour day month weekday
timeout_secs = 300
max_retries = 3
enabled = true
```

Environment overrides:
- `UCOTRON_CONNECTORS_ENABLED=true|false`
- `UCOTRON_CONNECTORS_CHECK_INTERVAL_SECS=<seconds>`

---

## Manual Sync

Trigger a one-off sync for a connector:

```bash
curl -X POST http://localhost:8420/api/v1/connectors/my-slack/sync \
  -H "Authorization: Bearer <writer-or-admin-key>"
```

Response:

```json
{
  "triggered": true,
  "connector_id": "my-slack",
  "message": "Sync started"
}
```

---

## Checking Schedules

```bash
# List all scheduled connectors
curl -s http://localhost:8420/api/v1/connectors/schedules \
  -H "Authorization: Bearer <reader-key>" | jq .
```

Response includes `next_fire_at` for each schedule.

---

## Sync History

```bash
# View sync history for a connector
curl -s http://localhost:8420/api/v1/connectors/my-slack/history \
  -H "Authorization: Bearer <reader-key>" | jq .
```

---

## Webhook Ingestion

External services can push events directly:

```bash
# Webhook endpoint (no auth — validated via connector-specific signatures)
POST /api/v1/webhooks/{connector_id}

# Max payload: 1 MB
# Response:
{
  "accepted": true,
  "connector_id": "my-slack",
  "items_processed": 3,
  "sync_triggered": true
}
```

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| Sync returns 403 | Insufficient role | Use writer or admin API key |
| Sync timeout | Source API slow or large dataset | Increase `timeout_secs` in schedule config |
| Webhook returns 400 | Invalid payload or unknown connector | Check connector_id exists in config |
| Schedule not firing | `enabled: false` or bad cron expression | Verify with `GET /connectors/schedules`; check `enabled` and cron syntax |
| Duplicate data after sync | Connector re-ingests existing items | Entity dedup in ingestion pipeline handles this; check dedup thresholds |
| Auth error from source | Expired token | Rotate credentials in connector config and restart |
| `max_retries` exhausted | Persistent source failure | Check source API status; increase `max_retries` or fix connectivity |
