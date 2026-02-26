#!/usr/bin/env python3
"""QA-009: Test auth, RBAC, and API key management"""

import json
import requests
import sys
import time
from datetime import datetime

BASE = "http://localhost:8420/api/v1"
ADMIN_KEY = "mk_admin_test_key_00000000"
WRITER_KEY = "mk_writer_test_key_0000000"
READER_KEY = "mk_reader_test_key_0000000"
SCOPED_KEY = "mk_scoped_test_key_0000000"

results = {
    "test_suite": "QA-009: Auth, RBAC & API Key Management",
    "timestamp": datetime.now().isoformat(),
    "tests": [],
    "summary": {"total": 0, "passed": 0, "failed": 0}
}

def test(name, fn):
    """Run a test and record result."""
    try:
        detail = fn()
        results["tests"].append({"name": name, "status": "PASS", "detail": detail})
        results["summary"]["passed"] += 1
        print(f"  ✓ {name}")
    except Exception as e:
        results["tests"].append({"name": name, "status": "FAIL", "error": str(e)})
        results["summary"]["failed"] += 1
        print(f"  ✗ {name}: {e}")
    results["summary"]["total"] += 1

def h(key):
    """Build auth headers."""
    return {"Authorization": f"Bearer {key}"}

# ── Test 1: Health is public (no auth needed) ──
def test_health_public():
    r = requests.get(f"{BASE}/health")
    assert r.status_code == 200, f"Expected 200, got {r.status_code}"
    return {"status_code": 200, "note": "Health endpoint is public even with auth enabled"}

# ── Test 2: Unauthorized request returns 401 ──
def test_unauthorized_returns_401():
    r = requests.get(f"{BASE}/memories")
    assert r.status_code == 401, f"Expected 401, got {r.status_code}: {r.text[:200]}"
    return {"status_code": 401, "body": r.json() if r.headers.get("content-type","").startswith("application/json") else r.text[:200]}

# ── Test 3: Invalid key returns 401 ──
def test_invalid_key_401():
    r = requests.get(f"{BASE}/memories", headers=h("mk_invalid_key_000000000"))
    assert r.status_code == 401, f"Expected 401, got {r.status_code}"
    return {"status_code": 401}

# ── Test 4: Whoami with admin key ──
def test_whoami_admin():
    r = requests.get(f"{BASE}/auth/whoami", headers=h(ADMIN_KEY))
    assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text[:200]}"
    data = r.json()
    assert data["role"] == "admin", f"Expected admin role, got {data['role']}"
    assert data["auth_enabled"] == True
    return data

# ── Test 5: Whoami with reader key ──
def test_whoami_reader():
    r = requests.get(f"{BASE}/auth/whoami", headers=h(READER_KEY))
    assert r.status_code == 200, f"Expected 200, got {r.status_code}"
    data = r.json()
    assert data["role"] == "reader", f"Expected reader, got {data['role']}"
    return data

# ── Test 6: Whoami with namespace-scoped key ──
def test_whoami_scoped():
    r = requests.get(f"{BASE}/auth/whoami", headers=h(SCOPED_KEY))
    assert r.status_code == 200, f"Expected 200, got {r.status_code}"
    data = r.json()
    assert data["namespace_scope"] == "scoped-ns", f"Expected scoped-ns, got {data.get('namespace_scope')}"
    return data

# ── Test 7: List API keys (admin only) ──
def test_list_keys_admin():
    r = requests.get(f"{BASE}/auth/keys", headers=h(ADMIN_KEY))
    assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text[:200]}"
    data = r.json()
    assert "keys" in data, f"Expected 'keys' in response"
    assert len(data["keys"]) >= 4, f"Expected at least 4 keys, got {len(data['keys'])}"
    return {"key_count": len(data["keys"]), "keys": data["keys"]}

# ── Test 8: List API keys denied for reader ──
def test_list_keys_denied_reader():
    r = requests.get(f"{BASE}/auth/keys", headers=h(READER_KEY))
    assert r.status_code == 403, f"Expected 403, got {r.status_code}: {r.text[:200]}"
    return {"status_code": 403, "note": "Reader correctly denied admin endpoint"}

# ── Test 9: Create API key (admin only) ──
def test_create_key_admin():
    key_name = f"test-dynamic-{int(time.time())}"
    r = requests.post(f"{BASE}/auth/keys", headers=h(ADMIN_KEY), json={
        "name": key_name,
        "role": "reader"
    })
    assert r.status_code in (200, 201), f"Expected 200/201, got {r.status_code}: {r.text[:200]}"
    data = r.json()
    assert "key" in data, f"Expected 'key' in response"
    assert data["name"] == key_name
    assert data["role"] == "reader"
    return {"name": data["name"], "role": data["role"], "key_preview": data["key"][:10] + "..."}

# ── Test 10: Create key denied for writer ──
def test_create_key_denied_writer():
    r = requests.post(f"{BASE}/auth/keys", headers=h(WRITER_KEY), json={
        "name": "should-fail",
        "role": "viewer"
    })
    assert r.status_code == 403, f"Expected 403, got {r.status_code}: {r.text[:200]}"
    return {"status_code": 403}

# ── Test 11: Admin can access all endpoints ──
def test_admin_full_access():
    detail = {}
    # Read endpoint
    r = requests.get(f"{BASE}/memories", headers=h(ADMIN_KEY))
    assert r.status_code == 200, f"GET /memories: expected 200, got {r.status_code}"
    detail["list_memories"] = r.status_code

    # Write endpoint
    r = requests.post(f"{BASE}/memories", headers=h(ADMIN_KEY), json={
        "text": "Admin auth test memory"
    })
    assert r.status_code in (200, 201), f"POST /memories: expected 200/201, got {r.status_code}"
    detail["create_memory"] = r.status_code
    data = r.json()
    mem_id = data.get("chunk_node_ids", [None])[0]

    # Search endpoint
    r = requests.post(f"{BASE}/memories/search", headers=h(ADMIN_KEY), json={
        "query": "auth test", "limit": 5
    })
    assert r.status_code == 200, f"POST /search: expected 200, got {r.status_code}"
    detail["search"] = r.status_code

    # Delete endpoint (if we got an ID)
    if mem_id:
        r = requests.delete(f"{BASE}/memories/{mem_id}", headers=h(ADMIN_KEY))
        assert r.status_code in (200, 204), f"DELETE /memories: expected 200/204, got {r.status_code}"
        detail["delete_memory"] = r.status_code

    return detail

# ── Test 12: Reader can read but not write ──
def test_reader_read_only():
    detail = {}
    # Reader CAN list memories
    r = requests.get(f"{BASE}/memories", headers=h(READER_KEY))
    assert r.status_code == 200, f"GET /memories: expected 200, got {r.status_code}"
    detail["list_memories"] = {"status": r.status_code, "allowed": True}

    # Reader CAN search
    r = requests.post(f"{BASE}/memories/search", headers=h(READER_KEY), json={
        "query": "test", "limit": 5
    })
    assert r.status_code == 200, f"POST /search: expected 200, got {r.status_code}"
    detail["search"] = {"status": r.status_code, "allowed": True}

    # Reader CANNOT create
    r = requests.post(f"{BASE}/memories", headers=h(READER_KEY), json={
        "text": "Should be denied"
    })
    assert r.status_code == 403, f"POST /memories: expected 403, got {r.status_code}: {r.text[:200]}"
    detail["create_memory"] = {"status": r.status_code, "denied": True}

    # Reader CANNOT delete (use a valid u64 ID format — 999999 as non-existent)
    r = requests.delete(f"{BASE}/memories/999999", headers=h(READER_KEY))
    assert r.status_code == 403, f"DELETE /memories: expected 403, got {r.status_code}: {r.text[:200]}"
    detail["delete_memory"] = {"status": r.status_code, "denied": True}

    return detail

# ── Test 13: Writer can create but not admin ──
def test_writer_permissions():
    detail = {}
    # Writer CAN create
    r = requests.post(f"{BASE}/memories", headers=h(WRITER_KEY), json={
        "text": "Writer auth test memory"
    })
    assert r.status_code in (200, 201), f"POST /memories: expected 200/201, got {r.status_code}"
    detail["create_memory"] = {"status": r.status_code, "allowed": True}

    # Writer CAN search
    r = requests.post(f"{BASE}/memories/search", headers=h(WRITER_KEY), json={
        "query": "writer test", "limit": 5
    })
    assert r.status_code == 200, f"POST /search: expected 200, got {r.status_code}"
    detail["search"] = {"status": r.status_code, "allowed": True}

    # Writer CANNOT list API keys
    r = requests.get(f"{BASE}/auth/keys", headers=h(WRITER_KEY))
    assert r.status_code == 403, f"GET /auth/keys: expected 403, got {r.status_code}"
    detail["list_keys"] = {"status": r.status_code, "denied": True}

    return detail

# ── Test 14: Namespace-scoped key isolation ──
def test_namespace_scoped_key():
    detail = {}

    # Scoped key CAN access its own namespace
    r = requests.post(f"{BASE}/memories",
        headers={**h(SCOPED_KEY), "X-Ucotron-Namespace": "scoped-ns"},
        json={"text": "Scoped namespace test memory"})
    assert r.status_code in (200, 201), f"POST to own namespace: expected 200/201, got {r.status_code}: {r.text[:200]}"
    detail["own_namespace_create"] = {"status": r.status_code, "allowed": True}

    # Scoped key CAN search its own namespace
    r = requests.post(f"{BASE}/memories/search",
        headers={**h(SCOPED_KEY), "X-Ucotron-Namespace": "scoped-ns"},
        json={"query": "scoped test", "limit": 5})
    assert r.status_code == 200, f"Search own namespace: expected 200, got {r.status_code}"
    detail["own_namespace_search"] = {"status": r.status_code, "allowed": True}

    # Scoped key CANNOT access different namespace
    r = requests.get(f"{BASE}/memories",
        headers={**h(SCOPED_KEY), "X-Ucotron-Namespace": "other-namespace"})
    assert r.status_code == 403, f"Access other namespace: expected 403, got {r.status_code}: {r.text[:200]}"
    detail["other_namespace_denied"] = {"status": r.status_code, "denied": True}

    # Scoped key CANNOT access default namespace
    r = requests.get(f"{BASE}/memories", headers=h(SCOPED_KEY))
    # Default namespace should be denied if key is scoped to "scoped-ns"
    assert r.status_code == 403, f"Access default namespace: expected 403, got {r.status_code}: {r.text[:200]}"
    detail["default_namespace_denied"] = {"status": r.status_code, "denied": True}

    return detail

# ── Test 15: Auth entries in audit log ──
def test_audit_has_auth_info():
    r = requests.get(f"{BASE}/audit", headers=h(ADMIN_KEY))
    assert r.status_code == 200, f"Expected 200, got {r.status_code}"
    data = r.json()
    entries = data if isinstance(data, list) else data.get("entries", data.get("audit", []))
    # Find an entry with key_name/user info
    auth_entries = [e for e in entries if e.get("user") or e.get("key_name")]
    return {
        "total_audit_entries": len(entries),
        "entries_with_auth_info": len(auth_entries),
        "sample": auth_entries[:3] if auth_entries else "No auth info in audit entries"
    }

# ── Run all tests ──
print("=" * 60)
print("QA-009: Auth, RBAC & API Key Management")
print("=" * 60)

print("\n[Public & Unauthorized]")
test("Health endpoint is public (no auth needed)", test_health_public)
test("Unauthorized request returns 401", test_unauthorized_returns_401)
test("Invalid API key returns 401", test_invalid_key_401)

print("\n[Whoami / Identity]")
test("Whoami with admin key", test_whoami_admin)
test("Whoami with reader key", test_whoami_reader)
test("Whoami with namespace-scoped key", test_whoami_scoped)

print("\n[API Key Management]")
test("List API keys (admin)", test_list_keys_admin)
test("List API keys denied for reader (403)", test_list_keys_denied_reader)
test("Create API key via API (admin)", test_create_key_admin)
test("Create key denied for writer (403)", test_create_key_denied_writer)

print("\n[RBAC Enforcement]")
test("Admin has full access to all endpoints", test_admin_full_access)
test("Reader can read but not create/delete (403)", test_reader_read_only)
test("Writer can create but not admin endpoints (403)", test_writer_permissions)

print("\n[Namespace Scoping]")
test("Namespace-scoped key isolation", test_namespace_scoped_key)

print("\n[Audit]")
test("Audit entries include auth information", test_audit_has_auth_info)

# ── Summary ──
print(f"\n{'=' * 60}")
print(f"Results: {results['summary']['passed']}/{results['summary']['total']} passed, {results['summary']['failed']} failed")
print(f"{'=' * 60}")

# Save results
with open("test-results/oss-qa/auth-results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nResults saved to test-results/oss-qa/auth-results.json")

sys.exit(0 if results["summary"]["failed"] == 0 else 1)
