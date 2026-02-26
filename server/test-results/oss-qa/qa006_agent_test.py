#!/usr/bin/env python3
"""QA-006: Test agent clone/merge operations"""

import json
import requests
import sys
import time

BASE = "http://localhost:8420/api/v1"
results = {"test_id": "QA-006", "description": "Agent clone/merge operations", "tests": [], "summary": {}}
agent_ids = {}

def test(name, fn):
    try:
        passed, detail = fn()
        results["tests"].append({"name": name, "passed": passed, "detail": detail})
        print(f"  {'PASS' if passed else 'FAIL'}: {name}")
        if not passed:
            print(f"        {detail}")
        return passed
    except Exception as e:
        results["tests"].append({"name": name, "passed": False, "detail": str(e)})
        print(f"  FAIL: {name} — {e}")
        return False

# ── Test 1: Create agent A ──
def test_create_agent_a():
    r = requests.post(f"{BASE}/agents", json={"name": "agent-alpha", "config": {"model": "test", "temperature": 0.5}})
    if r.status_code != 201:
        return False, f"Expected 201, got {r.status_code}: {r.text}"
    body = r.json()
    agent_ids["A"] = body["id"]
    has_fields = all(k in body for k in ["id", "name", "namespace", "owner", "created_at"])
    return has_fields and body["name"] == "agent-alpha", body

# ── Test 2: Create agent B ──
def test_create_agent_b():
    r = requests.post(f"{BASE}/agents", json={"name": "agent-beta", "config": {"purpose": "merge-test"}})
    if r.status_code != 201:
        return False, f"Expected 201, got {r.status_code}: {r.text}"
    body = r.json()
    agent_ids["B"] = body["id"]
    return body["name"] == "agent-beta", body

# ── Test 3: List agents ──
def test_list_agents():
    r = requests.get(f"{BASE}/agents")
    if r.status_code != 200:
        return False, f"Expected 200, got {r.status_code}: {r.text}"
    body = r.json()
    ids_found = [a["id"] for a in body["agents"]]
    has_a = agent_ids["A"] in ids_found
    has_b = agent_ids["B"] in ids_found
    return has_a and has_b, {"total": body["total"], "found_A": has_a, "found_B": has_b}

# ── Test 4: Get agent A details ──
def test_get_agent():
    r = requests.get(f"{BASE}/agents/{agent_ids['A']}")
    if r.status_code != 200:
        return False, f"Expected 200, got {r.status_code}: {r.text}"
    body = r.json()
    return body["id"] == agent_ids["A"] and body["name"] == "agent-alpha", body

# ── Test 5: Ingest memories into agent A's namespace ──
def test_ingest_memories_agent_a():
    ns = f"agent_{agent_ids['A']}"  # agent namespace pattern from create response
    # Actually use the namespace from the create response
    r = requests.get(f"{BASE}/agents/{agent_ids['A']}")
    ns = r.json()["namespace"]

    memories = [
        "Rust is a systems programming language focused on safety and performance",
        "Python is widely used for data science and machine learning applications",
        "TypeScript adds static types to JavaScript for better developer experience",
        "Go is designed for concurrent programming and cloud infrastructure",
        "Java remains popular for enterprise applications and Android development",
    ]
    created = []
    for text in memories:
        r = requests.post(f"{BASE}/memories", json={"text": text}, headers={"X-Ucotron-Namespace": ns})
        if r.status_code == 201:
            created.append(r.json().get("chunk_node_ids", []))
    return len(created) == 5, {"namespace": ns, "memories_created": len(created)}

# ── Test 6: Ingest memories into agent B's namespace ──
def test_ingest_memories_agent_b():
    r = requests.get(f"{BASE}/agents/{agent_ids['B']}")
    ns = r.json()["namespace"]

    memories = [
        "Docker containers package applications with their dependencies",
        "Kubernetes orchestrates container deployment at scale",
        "Terraform manages infrastructure as code across cloud providers",
    ]
    created = []
    for text in memories:
        r = requests.post(f"{BASE}/memories", json={"text": text}, headers={"X-Ucotron-Namespace": ns})
        if r.status_code == 201:
            created.append(r.json().get("chunk_node_ids", []))
    return len(created) == 3, {"namespace": ns, "memories_created": len(created)}

# ── Test 7: Clone agent A ──
def test_clone_agent():
    r = requests.post(f"{BASE}/agents/{agent_ids['A']}/clone", json={})
    if r.status_code not in (200, 201):
        return False, f"Expected 200/201, got {r.status_code}: {r.text}"
    body = r.json()
    agent_ids["clone_ns"] = body.get("target_namespace", "")
    has_fields = all(k in body for k in ["source_agent_id", "target_namespace", "nodes_copied", "edges_copied"])
    return has_fields and body["nodes_copied"] >= 0, body

# ── Test 8: Verify cloned data is independent ──
def test_clone_independence():
    clone_ns = agent_ids.get("clone_ns", "")
    if not clone_ns:
        return False, "No clone namespace available"

    # Search in clone namespace for programming language content
    r = requests.post(f"{BASE}/memories/search",
                      json={"query": "programming language", "limit": 5},
                      headers={"X-Ucotron-Namespace": clone_ns})
    if r.status_code != 200:
        return False, f"Search failed: {r.status_code}: {r.text}"

    clone_results = r.json().get("results", [])

    # Also search in original namespace
    r2 = requests.get(f"{BASE}/agents/{agent_ids['A']}")
    orig_ns = r2.json()["namespace"]
    r3 = requests.post(f"{BASE}/memories/search",
                       json={"query": "programming language", "limit": 5},
                       headers={"X-Ucotron-Namespace": orig_ns})
    orig_results = r3.json().get("results", [])

    # Both should have results (clone has independent copy)
    return len(clone_results) > 0 and len(orig_results) > 0, {
        "clone_ns": clone_ns, "clone_results": len(clone_results),
        "orig_ns": orig_ns, "orig_results": len(orig_results)
    }

# ── Test 9: Merge agent B into agent A ──
def test_merge_agents():
    r = requests.post(f"{BASE}/agents/{agent_ids['A']}/merge",
                      json={"source_agent_id": agent_ids["B"]})
    if r.status_code not in (200, 201):
        return False, f"Expected 200/201, got {r.status_code}: {r.text}"
    body = r.json()
    has_fields = all(k in body for k in ["source_namespace", "target_namespace", "nodes_copied", "edges_copied"])
    return has_fields, body

# ── Test 10: Verify merged data in agent A ──
def test_merged_data():
    r = requests.get(f"{BASE}/agents/{agent_ids['A']}")
    ns = r.json()["namespace"]

    # Search for B's content (Docker/Kubernetes) in A's namespace
    r2 = requests.post(f"{BASE}/memories/search",
                       json={"query": "container orchestration kubernetes", "limit": 5},
                       headers={"X-Ucotron-Namespace": ns})
    if r2.status_code != 200:
        return False, f"Search failed: {r2.status_code}: {r2.text}"

    results_list = r2.json().get("results", [])
    # Should find B's content after merge
    return len(results_list) > 0, {"namespace": ns, "results_count": len(results_list),
                                    "top_result": results_list[0]["content"][:80] if results_list else "none"}

# ── Test 11: Delete agent B ──
def test_delete_agent():
    r = requests.delete(f"{BASE}/agents/{agent_ids['B']}")
    if r.status_code not in (200, 204):
        return False, f"Expected 200/204, got {r.status_code}: {r.text}"

    # Verify it's gone
    r2 = requests.get(f"{BASE}/agents/{agent_ids['B']}")
    return r2.status_code == 404, {"delete_status": r.status_code, "get_after_delete": r2.status_code}

# ── Test 12: Delete agent A (cleanup) ──
def test_delete_agent_a():
    r = requests.delete(f"{BASE}/agents/{agent_ids['A']}")
    if r.status_code not in (200, 204):
        return False, f"Expected 200/204, got {r.status_code}: {r.text}"

    r2 = requests.get(f"{BASE}/agents/{agent_ids['A']}")
    return r2.status_code == 404, {"delete_status": r.status_code, "get_after_delete": r2.status_code}

# ── Run all tests ──
print("QA-006: Agent Clone/Merge Operations")
print("=" * 50)

passed = 0
total = 0

for name, fn in [
    ("Create agent A", test_create_agent_a),
    ("Create agent B", test_create_agent_b),
    ("List agents shows both", test_list_agents),
    ("Get agent A details", test_get_agent),
    ("Ingest memories into agent A", test_ingest_memories_agent_a),
    ("Ingest memories into agent B", test_ingest_memories_agent_b),
    ("Clone agent A", test_clone_agent),
    ("Cloned agent has independent data", test_clone_independence),
    ("Merge agent B into agent A", test_merge_agents),
    ("Merged data searchable in agent A", test_merged_data),
    ("Delete agent B", test_delete_agent),
    ("Delete agent A (cleanup)", test_delete_agent_a),
]:
    total += 1
    if test(name, fn):
        passed += 1

results["summary"] = {"passed": passed, "failed": total - passed, "total": total}
print(f"\n{'=' * 50}")
print(f"Results: {passed}/{total} passed")

with open("test-results/oss-qa/agent-results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"Results saved to test-results/oss-qa/agent-results.json")

sys.exit(0 if passed == total else 1)
