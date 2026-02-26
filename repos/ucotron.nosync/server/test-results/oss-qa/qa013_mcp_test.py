#!/usr/bin/env python3
"""QA-013: Test MCP server and conversations."""
import json
import requests
import sys
import time

BASE = "http://localhost:8420"
API = f"{BASE}/api/v1"
NAMESPACE = "qa013-final"
RESULTS = {"test_name": "QA-013: MCP Server and Conversations", "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"), "tests": []}

def record(name, passed, details=None):
    r = {"name": name, "passed": passed}
    if details:
        r["details"] = details
    RESULTS["tests"].append(r)
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {name}")
    return passed

total = 0
passed_count = 0

# === MCP Tests ===

# Test 1: MCP endpoint is accessible (initialize)
print("\n=== MCP Tests ===")
total += 1
try:
    resp = requests.post(f"{BASE}/mcp",
        headers={"Content-Type": "application/json", "Accept": "application/json, text/event-stream"},
        json={"jsonrpc": "2.0", "id": 1, "method": "initialize",
              "params": {"protocolVersion": "2024-11-05", "capabilities": {},
                         "clientInfo": {"name": "qa013-test", "version": "1.0.0"}}},
        stream=True, timeout=10)
    
    session_id = resp.headers.get("mcp-session-id", "")
    # Parse SSE response
    raw = resp.text
    init_data = None
    for line in raw.split("\n"):
        if line.startswith("data: ") and line.strip() != "data:":
            try:
                init_data = json.loads(line[6:])
            except:
                pass
    
    ok = resp.status_code == 200 and session_id != "" and init_data is not None
    server_info = init_data.get("result", {}).get("serverInfo", {}) if init_data else {}
    details = {
        "status_code": resp.status_code,
        "session_id": session_id,
        "server_name": server_info.get("name"),
        "server_version": server_info.get("version"),
        "protocol_version": init_data.get("result", {}).get("protocolVersion") if init_data else None,
        "capabilities": init_data.get("result", {}).get("capabilities") if init_data else None,
        "instructions": init_data.get("result", {}).get("instructions", "")[:100] if init_data else None
    }
    if record("MCP endpoint accessible (initialize)", ok, details):
        passed_count += 1
except Exception as e:
    record("MCP endpoint accessible (initialize)", False, {"error": str(e)})

# Test 2: MCP lists available tools
total += 1
try:
    # Send initialized notification first
    requests.post(f"{BASE}/mcp",
        headers={"Content-Type": "application/json", "Accept": "application/json, text/event-stream",
                 "Mcp-Session-Id": session_id},
        json={"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}},
        timeout=5)
    
    # List tools
    resp = requests.post(f"{BASE}/mcp",
        headers={"Content-Type": "application/json", "Accept": "application/json, text/event-stream",
                 "Mcp-Session-Id": session_id},
        json={"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
        stream=True, timeout=10)
    
    raw = resp.text
    tools_data = None
    for line in raw.split("\n"):
        if line.startswith("data: ") and line.strip() != "data:":
            try:
                parsed = json.loads(line[6:])
                if "result" in parsed and "tools" in parsed.get("result", {}):
                    tools_data = parsed
            except:
                pass
    
    tools = tools_data.get("result", {}).get("tools", []) if tools_data else []
    tool_names = [t["name"] for t in tools]
    expected_tools = ["ucotron_add_memory", "ucotron_search", "ucotron_get_entity", 
                      "ucotron_list_entities", "ucotron_augment", "ucotron_learn"]
    all_present = all(t in tool_names for t in expected_tools)
    
    details = {
        "tool_count": len(tools),
        "tool_names": tool_names,
        "expected_tools_present": all_present,
        "tools_with_schemas": [{
            "name": t["name"],
            "description": t["description"][:80],
            "has_input_schema": "inputSchema" in t
        } for t in tools]
    }
    if record("MCP lists available tools", all_present and len(tools) == 6, details):
        passed_count += 1
except Exception as e:
    record("MCP lists available tools", False, {"error": str(e)})

# Test 3: MCP rejects requests without proper headers
total += 1
try:
    resp = requests.post(f"{BASE}/mcp",
        headers={"Content-Type": "application/json"},  # Missing Accept header
        json={"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
        timeout=5)
    ok = resp.status_code == 406
    if record("MCP rejects without proper Accept header (406)", ok, {"status_code": resp.status_code}):
        passed_count += 1
except Exception as e:
    record("MCP rejects without proper Accept header", False, {"error": str(e)})

# === Conversation Tests ===
print("\n=== Conversation Tests ===")

# Create test conversations via /learn
total += 1
try:
    conv_messages = [
        {"output": "We should use PostgreSQL for the production database because of its robustness.", "conversation_id": "conv-final-001"},
        {"output": "Actually Redis might be better for caching layer alongside PostgreSQL.", "conversation_id": "conv-final-001"},
        {"output": "Good point. Let's use both - PostgreSQL as primary, Redis for caching.", "conversation_id": "conv-final-001"},
        {"output": "The CI pipeline should run on GitHub Actions with matrix builds.", "conversation_id": "conv-final-002"},
        {"output": "We need to add linting and type checking to the CI pipeline.", "conversation_id": "conv-final-002"},
    ]
    
    learn_results = []
    for msg in conv_messages:
        resp = requests.post(f"{API}/learn",
            headers={"Content-Type": "application/json", "X-Ucotron-Namespace": NAMESPACE},
            json=msg, timeout=15)
        learn_results.append({"status": resp.status_code, "body": resp.json()})
    
    all_201 = all(r["status"] == 201 for r in learn_results)
    all_created = all(r["body"]["memories_created"] >= 1 for r in learn_results)
    ok = all_201 and all_created
    if record("Create conversation memories via /learn", ok, 
              {"messages_sent": len(conv_messages), "learn_results": learn_results}):
        passed_count += 1
except Exception as e:
    record("Create conversation memories via /learn", False, {"error": str(e)})

# Test 4: List conversations
total += 1
try:
    resp = requests.get(f"{API}/conversations",
        headers={"X-Ucotron-Namespace": NAMESPACE}, timeout=10)
    convs = resp.json()
    
    conv_ids = [c["conversation_id"] for c in convs]
    has_conv1 = "conv-final-001" in conv_ids
    has_conv2 = "conv-final-002" in conv_ids
    
    details = {
        "status_code": resp.status_code,
        "conversation_count": len(convs),
        "conversations": convs,
        "has_conv_001": has_conv1,
        "has_conv_002": has_conv2
    }
    
    ok = resp.status_code == 200 and has_conv1 and has_conv2
    if record("GET /conversations lists tracked conversations", ok, details):
        passed_count += 1
except Exception as e:
    record("GET /conversations lists tracked conversations", False, {"error": str(e)})

# Test 5: Get conversation messages (conv-001)
total += 1
try:
    resp = requests.get(f"{API}/conversations/conv-final-001/messages",
        headers={"X-Ucotron-Namespace": NAMESPACE}, timeout=10)
    detail = resp.json()
    
    # Note: chunking may split messages, so check >= 3 (not ==3)
    ok = (resp.status_code == 200 and
          detail["conversation_id"] == "conv-final-001" and
          detail["namespace"] == NAMESPACE and
          len(detail["messages"]) >= 3)
    
    details = {
        "status_code": resp.status_code,
        "conversation_id": detail.get("conversation_id"),
        "namespace": detail.get("namespace"),
        "message_count": len(detail.get("messages", [])),
        "messages": [{"id": m["id"], "content": m["content"][:60], "entities": m["entities"]} 
                     for m in detail.get("messages", [])]
    }
    if record("GET /conversations/conv-final-001/messages returns messages", ok, details):
        passed_count += 1
except Exception as e:
    record("GET /conversations/conv-final-001/messages", False, {"error": str(e)})

# Test 6: Get conversation messages (conv-002)
total += 1
try:
    resp = requests.get(f"{API}/conversations/conv-final-002/messages",
        headers={"X-Ucotron-Namespace": NAMESPACE}, timeout=10)
    detail = resp.json()
    
    ok = (resp.status_code == 200 and 
          detail["conversation_id"] == "conv-final-002" and
          len(detail["messages"]) == 2)
    
    details = {
        "status_code": resp.status_code,
        "message_count": len(detail.get("messages", [])),
        "messages": [{"id": m["id"], "content": m["content"][:60]} 
                     for m in detail.get("messages", [])]
    }
    if record("GET /conversations/conv-final-002/messages returns messages", ok, details):
        passed_count += 1
except Exception as e:
    record("GET /conversations/conv-final-002/messages", False, {"error": str(e)})

# Test 7: Non-existent conversation returns 404
total += 1
try:
    resp = requests.get(f"{API}/conversations/non-existent-conv/messages",
        headers={"X-Ucotron-Namespace": NAMESPACE}, timeout=10)
    ok = resp.status_code == 404
    details = {"status_code": resp.status_code, "body": resp.json()}
    if record("Non-existent conversation returns 404", ok, details):
        passed_count += 1
except Exception as e:
    record("Non-existent conversation returns 404", False, {"error": str(e)})

# Test 8: Conversations respect namespace boundaries
total += 1
try:
    # List conversations in a different namespace - should not see qa013-final convos
    resp = requests.get(f"{API}/conversations",
        headers={"X-Ucotron-Namespace": "other-namespace-empty"}, timeout=10)
    convs = resp.json()
    ok = resp.status_code == 200 and len(convs) == 0
    details = {"status_code": resp.status_code, "conversations_in_other_ns": len(convs)}
    if record("Conversations respect namespace boundaries", ok, details):
        passed_count += 1
except Exception as e:
    record("Conversations respect namespace boundaries", False, {"error": str(e)})

# Test 9: Conversations support pagination
total += 1
try:
    resp = requests.get(f"{API}/conversations?limit=1&offset=0",
        headers={"X-Ucotron-Namespace": NAMESPACE}, timeout=10)
    convs = resp.json()
    ok = resp.status_code == 200 and len(convs) == 1
    
    # Check limit=10 returns both conversations
    resp_all = requests.get(f"{API}/conversations?limit=10&offset=0",
        headers={"X-Ucotron-Namespace": NAMESPACE}, timeout=10)
    convs_all = resp_all.json()
    ok = ok and len(convs_all) >= 2

    details = {"page1_limit1": convs, "all_convs": len(convs_all)}
    if record("Conversations support pagination", ok, details):
        passed_count += 1
except Exception as e:
    record("Conversations support pagination", False, {"error": str(e)})

# === Summary ===
RESULTS["summary"] = {
    "total": total,
    "passed": passed_count,
    "failed": total - passed_count,
    "pass_rate": f"{passed_count}/{total}"
}

print(f"\n=== Results: {passed_count}/{total} passed ===")

# Save results
with open("test-results/oss-qa/mcp-results.json", "w") as f:
    json.dump(RESULTS, f, indent=2)
print("Results saved to test-results/oss-qa/mcp-results.json")

sys.exit(0 if passed_count == total else 1)
