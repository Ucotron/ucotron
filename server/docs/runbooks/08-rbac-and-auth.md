# Runbook: RBAC & Authentication

## Overview

Ucotron uses API key-based authentication with four privilege levels. When auth is disabled, all requests are treated as admin. Namespace scoping restricts keys to specific data partitions.

---

## Roles

| Role | Level | Permissions |
|------|-------|-------------|
| `admin` | 3 | Full access: read, write, delete, key management, audit |
| `writer` | 2 | Read + write: ingest, learn, update, delete memories |
| `reader` | 1 | Read-only: search, augment, get, list |
| `viewer` | 0 | Health + metrics only |

---

## Configuration

```toml
[auth]
enabled = true
api_key = "legacy-admin-key"      # Single-key mode (grants admin role)
api_keys = [
  { name = "backend", key = "sk-prod-abc123", role = "writer", namespace = "prod", active = true },
  { name = "dashboard", key = "sk-dash-xyz789", role = "reader", namespace = null, active = true },
]
```

Environment overrides:
- `UCOTRON_AUTH_ENABLED=true|false`
- `UCOTRON_AUTH_API_KEY=<key>` (legacy single key)

---

## API Key Management

### List keys (admin only)

```bash
curl -s http://localhost:8420/api/v1/auth/keys \
  -H "Authorization: Bearer <admin-key>" | jq .
```

### Create a new key (admin only)

```bash
curl -X POST http://localhost:8420/api/v1/auth/keys \
  -H "Authorization: Bearer <admin-key>" \
  -H "Content-Type: application/json" \
  -d '{"name": "ml-service", "role": "writer", "namespace": "ml"}'
```

**The key value is returned only once in the response. Store it securely.**

### Revoke a key (admin only)

```bash
curl -X DELETE http://localhost:8420/api/v1/auth/keys/ml-service \
  -H "Authorization: Bearer <admin-key>"
```

This sets `active = false`; the key name remains in the list.

### Check current identity

```bash
curl -s http://localhost:8420/api/v1/auth/whoami \
  -H "Authorization: Bearer <any-key>" | jq .
# {"role":"writer","namespace_scope":"prod","key_name":"backend"}
```

---

## Namespace Scoping

- Keys with `namespace = "prod"` can only access the `prod` namespace
- Keys with `namespace = null` can access all namespaces
- The target namespace is set via header: `X-Ucotron-Namespace: prod`
- Default namespace if header is absent: `"default"`

```bash
# Query a specific namespace
curl -s http://localhost:8420/api/v1/memories \
  -H "Authorization: Bearer <key>" \
  -H "X-Ucotron-Namespace: staging"
```

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| 401 Unauthorized | Missing or invalid API key | Check `Authorization: Bearer <key>` header |
| 403 Forbidden | Insufficient role for endpoint | Use a key with higher privilege level |
| 403 on namespace | Key scoped to different namespace | Use key matching target namespace, or use unscoped admin key |
| All requests succeed without auth | `auth.enabled = false` | Set `auth.enabled = true` in config |
| Key creation returns 403 | Using non-admin key | Only admin keys can manage other keys |
| Key not working after creation | Key was revoked | Check `active` field via `GET /auth/keys` |

---

## Security Best Practices

1. **Enable auth in production** — never run with `auth.enabled = false` in exposed environments
2. **Use named keys** — avoid the legacy single `api_key` field; use `api_keys[]` array for auditability
3. **Scope namespaces** — give each service a key scoped to its namespace
4. **Principle of least privilege** — use `reader` unless writes are needed
5. **Rotate keys periodically** — revoke old keys and create new ones via the API
6. **Store keys in secrets manager** — never commit API keys to source control
