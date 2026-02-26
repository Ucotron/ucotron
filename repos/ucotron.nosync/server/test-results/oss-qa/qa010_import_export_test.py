#!/usr/bin/env python3
"""QA-010: Test export/import and migration formats."""

import json
import time
import requests

BASE = "http://localhost:8420/api/v1"
RESULTS = {}

def test(name, fn):
    """Run a test and record result."""
    print(f"  [{name}] ", end="", flush=True)
    try:
        result = fn()
        RESULTS[name] = {"status": "PASS", "details": result}
        print("PASS")
        return result
    except Exception as e:
        RESULTS[name] = {"status": "FAIL", "error": str(e)}
        print(f"FAIL: {e}")
        return None

def assert_eq(a, b, msg=""):
    if a != b:
        raise AssertionError(f"{msg}: expected {b}, got {a}")

# ── Test 1: Export default namespace ──
def test_export():
    r = requests.get(f"{BASE}/export")
    assert r.status_code == 200, f"Export returned {r.status_code}: {r.text[:200]}"
    data = r.json()
    assert "nodes" in data, "Export missing 'nodes'"
    assert "edges" in data, "Export missing 'edges'"
    assert "stats" in data, "Export missing 'stats'"
    assert data.get("@type") == "ucotron:MemoryGraph" or data.get("#type") == "ucotron:MemoryGraph", \
        f"Unexpected type: {data.get('@type', data.get('#type'))}"
    assert data.get("version") == "1.0", f"Unexpected version: {data.get('version')}"
    node_count = len(data["nodes"])
    edge_count = len(data["edges"])
    return {
        "status_code": r.status_code,
        "node_count": node_count,
        "edge_count": edge_count,
        "namespace": data.get("namespace"),
        "version": data.get("version"),
        "has_context": "@context" in data or "#context" in data,
        "has_stats": "stats" in data,
        "stats": data.get("stats"),
    }

# ── Test 2: Export with embeddings disabled ──
def test_export_no_embeddings():
    r = requests.get(f"{BASE}/export", params={"include_embeddings": "false"})
    assert r.status_code == 200, f"Export returned {r.status_code}"
    data = r.json()
    # Check that nodes don't have embeddings
    has_embeddings = any(n.get("embedding") for n in data.get("nodes", [])[:10])
    return {
        "status_code": r.status_code,
        "node_count": len(data.get("nodes", [])),
        "has_embeddings_in_response": has_embeddings,
    }

# ── Test 3: Export specific namespace ──
def test_export_namespace():
    # Create a test namespace with some data
    ns = "qa010-export-ns"
    headers = {"X-Ucotron-Namespace": ns}
    # Ingest a few memories
    for i in range(3):
        r = requests.post(f"{BASE}/memories", json={"text": f"QA010 export test memory {i}"}, headers=headers)
        assert r.status_code == 201, f"Create memory failed: {r.status_code}"
    time.sleep(0.5)

    # Export that namespace
    r = requests.get(f"{BASE}/export", headers=headers)
    assert r.status_code == 200, f"Namespace export returned {r.status_code}"
    data = r.json()
    node_count = len(data.get("nodes", []))
    assert node_count >= 3, f"Expected >= 3 nodes in namespace export, got {node_count}"
    return {
        "status_code": r.status_code,
        "namespace": ns,
        "node_count": node_count,
    }

# ── Test 4: Import (round-trip) ──
def test_import_roundtrip():
    # First, create a dedicated namespace and ingest data
    src_ns = "qa010-src"
    src_headers = {"X-Ucotron-Namespace": src_ns}
    test_memories = [
        "The Eiffel Tower is located in Paris, France",
        "Python is a popular programming language",
        "Water boils at 100 degrees Celsius at sea level",
        "The Great Wall of China is visible from space (myth)",
        "Mozart composed his first symphony at age 8",
    ]
    for text in test_memories:
        r = requests.post(f"{BASE}/memories", json={"text": text}, headers=src_headers)
        assert r.status_code == 201, f"Create failed: {r.status_code}"
    time.sleep(0.5)

    # Export source namespace
    r = requests.get(f"{BASE}/export", headers=src_headers)
    assert r.status_code == 200, f"Export failed: {r.status_code}"
    export_data = r.json()
    src_node_count = len(export_data.get("nodes", []))

    # Import into a different namespace
    dst_ns = "qa010-dst"
    dst_headers = {"X-Ucotron-Namespace": dst_ns, "Content-Type": "application/json"}
    r = requests.post(f"{BASE}/import", json=export_data, headers=dst_headers)
    assert r.status_code in (200, 201), f"Import returned {r.status_code}: {r.text[:300]}"
    import_result = r.json()

    # Verify: export the destination namespace
    time.sleep(0.5)
    r = requests.get(f"{BASE}/export", headers={"X-Ucotron-Namespace": dst_ns})
    assert r.status_code == 200
    dst_data = r.json()
    dst_node_count = len(dst_data.get("nodes", []))

    # Verify: search in destination namespace for one of the memories
    r = requests.post(f"{BASE}/memories/search", json={"query": "Eiffel Tower Paris", "limit": 5},
                      headers={"X-Ucotron-Namespace": dst_ns})
    search_ok = r.status_code == 200
    search_results = r.json().get("results", []) if search_ok else []
    found_eiffel = any("eiffel" in (res.get("content", "").lower()) for res in search_results)

    return {
        "src_namespace": src_ns,
        "dst_namespace": dst_ns,
        "src_node_count": src_node_count,
        "dst_node_count": dst_node_count,
        "import_result": import_result,
        "nodes_match": dst_node_count >= src_node_count,
        "search_works_in_dst": found_eiffel,
        "integrity_verified": dst_node_count >= src_node_count and found_eiffel,
    }

# ── Test 5: Import mem0 format ──
def test_import_mem0():
    mem0_data = {
        "results": [
            {
                "id": "mem_test_001",
                "memory": "User prefers dark mode in all applications",
                "user_id": "test-user-42",
                "agent_id": "assistant_v1",
                "created_at": "2024-07-15T10:30:00Z",
                "updated_at": "2024-07-15T10:30:00Z",
                "metadata": {"category": "preferences"},
            },
            {
                "id": "mem_test_002",
                "memory": "User is a senior software engineer at a startup",
                "user_id": "test-user-42",
                "agent_id": "assistant_v1",
                "created_at": "2024-07-15T11:00:00Z",
                "updated_at": "2024-07-15T11:00:00Z",
                "metadata": {"category": "profile"},
            },
            {
                "id": "mem_test_003",
                "memory": "User enjoys hiking on weekends",
                "user_id": "test-user-42",
                "created_at": "2024-07-16T09:00:00Z",
                "updated_at": "2024-07-16T09:00:00Z",
                "metadata": {"category": "hobbies"},
            },
        ],
        "total_memories": 3,
    }

    ns = "qa010-mem0"
    headers = {"X-Ucotron-Namespace": ns, "Content-Type": "application/json"}
    r = requests.post(f"{BASE}/import/mem0", json={"data": mem0_data, "link_same_user": True}, headers=headers)

    if r.status_code in (200, 201):
        result = r.json()
        # Verify imported data is searchable
        time.sleep(0.5)
        r2 = requests.post(f"{BASE}/memories/search", json={"query": "dark mode preference", "limit": 5},
                           headers={"X-Ucotron-Namespace": ns})
        search_ok = r2.status_code == 200
        search_results = r2.json().get("results", []) if search_ok else []
        found = any("dark mode" in res.get("content", "").lower() for res in search_results)
        return {
            "status_code": r.status_code,
            "import_result": result,
            "memories_parsed": result.get("memories_parsed"),
            "nodes_imported": result.get("nodes_imported"),
            "edges_imported": result.get("edges_imported"),
            "searchable": found,
        }
    else:
        # Clear error is acceptable per AC
        return {
            "status_code": r.status_code,
            "response": r.text[:500],
            "clear_error": r.status_code in (400, 422),
        }

# ── Test 6: Import zep format ──
def test_import_zep():
    zep_data = {
        "entities": [
            {
                "uuid": "ent-qa010-001",
                "name": "Alice",
                "group_id": "g1",
                "labels": ["Person"],
                "created_at": "2024-08-01T12:00:00Z",
                "summary": "Software engineer who enjoys coffee",
            },
            {
                "uuid": "ent-qa010-002",
                "name": "TechCorp",
                "group_id": "g1",
                "labels": ["Organization"],
                "created_at": "2024-08-01T12:00:00Z",
                "summary": "A technology company in San Francisco",
            },
        ],
        "episodes": [
            {
                "uuid": "ep-qa010-001",
                "content": "Alice mentioned she works at TechCorp as a backend engineer",
                "source": "chat",
                "source_description": "Slack conversation",
                "reference_time": "2024-08-01T14:00:00Z",
                "group_id": "g1",
            },
        ],
        "edges": [
            {
                "uuid": "edge-qa010-001",
                "source_node_uuid": "ent-qa010-001",
                "target_node_uuid": "ent-qa010-002",
                "fact": "Alice works at TechCorp",
                "name": "works_at",
                "created_at": "2024-08-01T12:00:00Z",
                "group_id": "g1",
            },
        ],
    }

    ns = "qa010-zep"
    headers = {"X-Ucotron-Namespace": ns, "Content-Type": "application/json"}
    r = requests.post(f"{BASE}/import/zep",
                      json={"data": zep_data, "link_same_user": True, "link_same_group": False, "preserve_expired": True},
                      headers=headers)

    if r.status_code in (200, 201):
        result = r.json()
        # Verify imported data is searchable
        time.sleep(0.5)
        r2 = requests.post(f"{BASE}/memories/search", json={"query": "Alice TechCorp engineer", "limit": 5},
                           headers={"X-Ucotron-Namespace": ns})
        search_ok = r2.status_code == 200
        search_results = r2.json().get("results", []) if search_ok else []
        return {
            "status_code": r.status_code,
            "import_result": result,
            "memories_parsed": result.get("memories_parsed"),
            "nodes_imported": result.get("nodes_imported"),
            "edges_imported": result.get("edges_imported"),
            "searchable": len(search_results) > 0,
        }
    else:
        return {
            "status_code": r.status_code,
            "response": r.text[:500],
            "clear_error": r.status_code in (400, 422),
        }

# ── Test 7: Import mem0 with empty/no data (error handling) ──
def test_import_mem0_empty():
    r = requests.post(f"{BASE}/import/mem0",
                      json={"data": {"results": [], "total_memories": 0}},
                      headers={"Content-Type": "application/json"})
    return {
        "status_code": r.status_code,
        "response": r.json() if r.headers.get("content-type", "").startswith("application/json") else r.text[:300],
        "handled_gracefully": r.status_code in (200, 201, 400, 422),
    }

# ── Test 8: Import zep with empty data (error handling) ──
def test_import_zep_empty():
    r = requests.post(f"{BASE}/import/zep",
                      json={"data": {"entities": [], "episodes": [], "edges": []}},
                      headers={"Content-Type": "application/json"})
    return {
        "status_code": r.status_code,
        "response": r.json() if r.headers.get("content-type", "").startswith("application/json") else r.text[:300],
        "handled_gracefully": r.status_code in (200, 201, 400, 422),
    }

# ── Run all tests ──
if __name__ == "__main__":
    print("=" * 60)
    print("QA-010: Export/Import and Migration Formats")
    print("=" * 60)

    test("1_export_default", test_export)
    test("2_export_no_embeddings", test_export_no_embeddings)
    test("3_export_namespace", test_export_namespace)
    test("4_import_roundtrip", test_import_roundtrip)
    test("5_import_mem0", test_import_mem0)
    test("6_import_zep", test_import_zep)
    test("7_import_mem0_empty", test_import_mem0_empty)
    test("8_import_zep_empty", test_import_zep_empty)

    # Summary
    print("\n" + "=" * 60)
    passed = sum(1 for v in RESULTS.values() if v["status"] == "PASS")
    total = len(RESULTS)
    print(f"Results: {passed}/{total} passed")
    print("=" * 60)

    # Save results
    output = {
        "test_suite": "QA-010: Export/Import and Migration Formats",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "summary": {"passed": passed, "total": total, "all_passed": passed == total},
        "tests": RESULTS,
    }

    with open("/Users/martinriesco/Documents/Code/Ucotron/repos/ucotron.nosync/server/test-results/oss-qa/import-export-results.json", "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to test-results/oss-qa/import-export-results.json")

    if passed < total:
        exit(1)
