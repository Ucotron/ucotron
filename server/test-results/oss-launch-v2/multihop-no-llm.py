#!/usr/bin/env python3
"""V2-013: Multi-hop tests without LLM (co-occurrence baseline).

Ingests 30 facts from multihop-test-data.json, then runs 15 multi-hop
queries (5x 1-hop, 5x 2-hop, 5x 3-hop) using the /augment endpoint.

Evaluates whether the expected answer entity appears in returned context.
Records correctness (0/1), hops traversed, and latency for each query.

Server must be running WITHOUT --features llm (co-occurrence baseline).
"""

import json
import time
import urllib.request
import urllib.error
import sys
import os

BASE_URL = "http://localhost:8420/api/v1"
NAMESPACE = "v2013-multihop"
HEADERS = {"X-Ucotron-Namespace": NAMESPACE}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DATA_FILE = os.path.join(SCRIPT_DIR, "multihop-test-data.json")
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "multihop-no-llm.json")


def api_request(method, path, data=None):
    """Make an API request and return (response_dict, latency_ms)."""
    url = f"{BASE_URL}{path}"
    headers = {"Content-Type": "application/json"}
    headers.update(HEADERS)
    body = json.dumps(data).encode() if data else None
    req = urllib.request.Request(url, data=body, headers=headers, method=method)

    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req) as resp:
            result = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        result = {"error": e.code, "body": e.read().decode()[:500]}
    except Exception as e:
        result = {"error": str(e)}
    elapsed_ms = (time.perf_counter() - start) * 1000
    return result, elapsed_ms


def ingest_facts(facts):
    """Ingest all facts via /learn endpoint."""
    print(f"\nIngesting {len(facts)} facts...")
    ingested = []
    for i, fact in enumerate(facts):
        context = fact["context"]
        result, ms = api_request("POST", "/learn", {
            "output": context,
            "conversation_id": f"multihop-fact-{fact['id']}",
        })
        is_error = isinstance(result, dict) and "error" in result
        memories = result.get("memories_created", 0) if not is_error else 0
        entities = result.get("entities_found", 0) if not is_error else 0
        status = f"ERROR({result.get('error', '')})" if is_error else f"OK (mem={memories}, ent={entities})"
        print(f"  [{fact['id']}] {ms:.0f}ms — {status}")
        ingested.append({
            "fact_id": fact["id"],
            "subject": fact["subject"],
            "relation": fact["relation"],
            "object": fact["object"],
            "latency_ms": round(ms, 2),
            "memories_created": memories,
            "entities_found": entities,
            "error": str(result.get("error", "")) if is_error else None,
        })
    return ingested


def evaluate_query(query, augment_result):
    """Check if the expected answer appears in the augment response."""
    expected = query["expected_answer"].lower()

    # Check in memories content
    memories = augment_result.get("memories", [])
    for mem in memories:
        content = (mem.get("content") or "").lower()
        if expected in content:
            return True, "found_in_memories"

    # Check in entities
    entities = augment_result.get("entities", [])
    for ent in entities:
        content = (ent.get("content") or "").lower()
        if expected in content:
            return True, "found_in_entities"

    # Check in context_text
    context_text = (augment_result.get("context_text") or "").lower()
    if expected in context_text:
        return True, "found_in_context"

    return False, "not_found"


def run_queries(queries):
    """Run all multi-hop queries and evaluate results."""
    print(f"\nRunning {len(queries)} multi-hop queries...")
    results = []

    for query in queries:
        qid = query["id"]
        hops = query["hops"]
        question = query["question"]
        expected = query["expected_answer"]

        # Use /augment with debug and explain for graph traversal info
        result, ms = api_request("POST", f"/augment?explain=true", {
            "context": question,
            "limit": 20,
            "debug": True,
        })

        is_error = isinstance(result, dict) and "error" in result
        if is_error:
            print(f"  [{qid}] {hops}-hop: ERROR — {result.get('error', '')}")
            results.append({
                "query_id": qid,
                "hops": hops,
                "question": question,
                "expected_answer": expected,
                "correct": 0,
                "found_in": "error",
                "latency_ms": round(ms, 2),
                "memories_returned": 0,
                "entities_returned": 0,
                "error": str(result.get("error", "")),
            })
            continue

        correct, found_in = evaluate_query(query, result)
        memories_returned = len(result.get("memories", []))
        entities_returned = len(result.get("entities", []))

        # Extract debug info if available
        debug = result.get("debug", {})
        pipeline_ms = debug.get("pipeline_duration_ms", None)
        vector_count = debug.get("vector_results_count", None)
        query_entities = debug.get("query_entities_count", None)

        # Extract explain info if available
        explain = result.get("explain", {})
        graph_expanded = len(explain.get("graph_expanded_ids", [])) if explain else 0

        status = "CORRECT" if correct else "MISS"
        print(f"  [{qid}] {hops}-hop: {status} — {ms:.0f}ms — '{expected}' {found_in} — mem={memories_returned} ent={entities_returned} graph_exp={graph_expanded}")

        results.append({
            "query_id": qid,
            "hops": hops,
            "question": question,
            "expected_answer": expected,
            "correct": 1 if correct else 0,
            "found_in": found_in,
            "latency_ms": round(ms, 2),
            "memories_returned": memories_returned,
            "entities_returned": entities_returned,
            "graph_expanded_nodes": graph_expanded,
            "vector_results_count": vector_count,
            "query_entities_count": query_entities,
            "pipeline_duration_ms": pipeline_ms,
        })

    return results


def compute_summary(query_results):
    """Compute accuracy and latency summaries by hop depth."""
    summary = {"overall": {}, "by_hops": {}}

    # Overall
    total = len(query_results)
    correct = sum(q["correct"] for q in query_results)
    latencies = [q["latency_ms"] for q in query_results]
    summary["overall"] = {
        "total_queries": total,
        "correct": correct,
        "accuracy": round(correct / total, 4) if total > 0 else 0,
        "latency_p50_ms": round(sorted(latencies)[len(latencies) // 2], 2) if latencies else 0,
        "latency_mean_ms": round(sum(latencies) / len(latencies), 2) if latencies else 0,
        "latency_max_ms": round(max(latencies), 2) if latencies else 0,
    }

    # By hop depth
    for hop in [1, 2, 3]:
        hop_results = [q for q in query_results if q["hops"] == hop]
        if not hop_results:
            continue
        hop_correct = sum(q["correct"] for q in hop_results)
        hop_latencies = [q["latency_ms"] for q in hop_results]
        summary["by_hops"][f"{hop}-hop"] = {
            "total": len(hop_results),
            "correct": hop_correct,
            "accuracy": round(hop_correct / len(hop_results), 4),
            "latency_p50_ms": round(sorted(hop_latencies)[len(hop_latencies) // 2], 2),
            "latency_mean_ms": round(sum(hop_latencies) / len(hop_latencies), 2),
        }

    return summary


def cleanup():
    """Delete all memories in our namespace."""
    print("\nCleaning up test data...")
    result, _ = api_request("GET", "/memories?limit=1000")
    if isinstance(result, list):
        for mem in result:
            mid = mem.get("id")
            if mid:
                api_request("DELETE", f"/memories/{mid}")
        print(f"  Deleted {len(result)} memories.")
    elif isinstance(result, dict) and "results" in result:
        for mem in result["results"]:
            mid = mem.get("id")
            if mid:
                api_request("DELETE", f"/memories/{mid}")
        print(f"  Deleted {len(result['results'])} memories.")
    else:
        print(f"  Could not list memories for cleanup: {str(result)[:200]}")


def main():
    print("=" * 60)
    print("V2-013: Multi-hop Tests WITHOUT LLM (Co-occurrence Baseline)")
    print("=" * 60)

    # Verify server health
    print("\nChecking server health...")
    result, _ = api_request("GET", "/health")
    if not isinstance(result, dict) or result.get("status") != "ok":
        print(f"ERROR: Server not healthy: {result}")
        sys.exit(1)

    models = result.get("models", {})
    strategy = models.get("relation_strategy", "unknown")
    llm_loaded = models.get("llm_loaded", False)
    print(f"  Status: {result['status']}")
    print(f"  Relation strategy: {strategy}")
    print(f"  LLM loaded: {llm_loaded}")
    print(f"  Embedder: {models.get('embedding_model', 'unknown')}")
    print(f"  NER: {models.get('ner_loaded', False)}")

    if llm_loaded:
        print("\n  WARNING: LLM is loaded — this should be a no-LLM baseline test.")
        print("  Results will be tagged as llm_active=true.")

    server_config = {
        "storage_mode": result.get("storage_mode"),
        "vector_backend": result.get("vector_backend"),
        "llm_loaded": llm_loaded,
        "relation_strategy": strategy,
        "embedding_model": models.get("embedding_model"),
        "ner_loaded": models.get("ner_loaded"),
        "embedding_provider": models.get("embedding_provider"),
    }

    # Load test data
    print(f"\nLoading test data from {TEST_DATA_FILE}...")
    with open(TEST_DATA_FILE) as f:
        test_data = json.load(f)
    facts = test_data["facts"]
    queries = test_data["queries"]
    print(f"  Loaded {len(facts)} facts and {len(queries)} queries")

    # Phase 1: Ingest all facts
    print("\n" + "=" * 60)
    print("PHASE 1: Ingesting facts")
    print("=" * 60)
    ingestion_results = ingest_facts(facts)
    total_memories = sum(r["memories_created"] for r in ingestion_results)
    total_entities = sum(r["entities_found"] for r in ingestion_results)
    ingestion_errors = sum(1 for r in ingestion_results if r["error"])
    print(f"\n  Ingestion complete: {total_memories} memories, {total_entities} entities, {ingestion_errors} errors")

    # Brief pause to let indices settle
    time.sleep(1)

    # Phase 2: Run multi-hop queries
    print("\n" + "=" * 60)
    print("PHASE 2: Running multi-hop queries")
    print("=" * 60)
    query_results = run_queries(queries)

    # Phase 3: Compute summary
    summary = compute_summary(query_results)

    # Print summary
    print("\n" + "=" * 60)
    print("MULTI-HOP RESULTS SUMMARY — No LLM (Co-occurrence Baseline)")
    print("=" * 60)
    overall = summary["overall"]
    print(f"  Overall: {overall['correct']}/{overall['total_queries']} correct ({overall['accuracy']*100:.1f}%)")
    print(f"  Latency: P50={overall['latency_p50_ms']}ms, mean={overall['latency_mean_ms']}ms, max={overall['latency_max_ms']}ms")
    print()
    for hop_label, hop_data in summary["by_hops"].items():
        print(f"  {hop_label}: {hop_data['correct']}/{hop_data['total']} correct ({hop_data['accuracy']*100:.1f}%) — P50={hop_data['latency_p50_ms']}ms")
    print("=" * 60)

    # Build output
    output = {
        "benchmark": "multihop-no-llm",
        "description": "Multi-hop reasoning test without LLM (co-occurrence relation extraction)",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "server_config": server_config,
        "ingestion": {
            "total_facts": len(facts),
            "total_memories_created": total_memories,
            "total_entities_found": total_entities,
            "ingestion_errors": ingestion_errors,
            "details": ingestion_results,
        },
        "queries": {
            "total": len(queries),
            "results": query_results,
        },
        "summary": summary,
    }

    # Cleanup
    cleanup()

    # Save
    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {OUTPUT_FILE}")

    accuracy = overall["accuracy"]
    print(f"\nOverall accuracy: {accuracy*100:.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
