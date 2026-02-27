#!/usr/bin/env python3
"""V2-014: Multi-hop tests with Qwen3-4B LLM.

Ingests 30 facts from multihop-test-data.json, then runs 15 multi-hop
queries (5x 1-hop, 5x 2-hop, 5x 3-hop) using the /augment endpoint.

Server must be running WITH --features llm and Qwen3-4B-GGUF loaded.
Uses ucotron-multihop-llm.toml with isolated data dirs.

Compares results against no-LLM baseline (multihop-no-llm.json).
"""

import json
import time
import urllib.request
import urllib.error
import sys
import os

BASE_URL = "http://localhost:8420/api/v1"
NAMESPACE = "v2014-multihop-llm"
HEADERS = {"X-Ucotron-Namespace": NAMESPACE}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DATA_FILE = os.path.join(SCRIPT_DIR, "multihop-test-data.json")
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "multihop-qwen3-4b.json")
BASELINE_FILE = os.path.join(SCRIPT_DIR, "multihop-no-llm.json")


def api_request(method, path, data=None):
    """Make an API request and return (response_dict, latency_ms)."""
    url = f"{BASE_URL}{path}"
    headers = {"Content-Type": "application/json"}
    headers.update(HEADERS)
    body = json.dumps(data).encode() if data else None
    req = urllib.request.Request(url, data=body, headers=headers, method=method)

    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
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
            "conversation_id": f"multihop-llm-fact-{fact['id']}",
        })
        is_error = isinstance(result, dict) and "error" in result
        memories = result.get("memories_created", 0) if not is_error else 0
        entities = result.get("entities_found", 0) if not is_error else 0
        relations = result.get("relations_found", 0) if not is_error else 0
        status = f"ERROR({result.get('error', '')})" if is_error else f"OK (mem={memories}, ent={entities}, rel={relations})"
        print(f"  [{fact['id']}] {ms:.0f}ms — {status}")
        ingested.append({
            "fact_id": fact["id"],
            "subject": fact["subject"],
            "relation": fact["relation"],
            "object": fact["object"],
            "latency_ms": round(ms, 2),
            "memories_created": memories,
            "entities_found": entities,
            "relations_found": relations,
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


def load_baseline():
    """Load baseline results for comparison."""
    if not os.path.exists(BASELINE_FILE):
        return None
    with open(BASELINE_FILE) as f:
        return json.load(f)


def compare_with_baseline(summary, baseline):
    """Generate comparison with no-LLM baseline."""
    if not baseline:
        return None

    bl_summary = baseline.get("summary", {})
    bl_overall = bl_summary.get("overall", {})
    our_overall = summary["overall"]

    comparison = {
        "baseline_accuracy": bl_overall.get("accuracy", 0),
        "llm_accuracy": our_overall["accuracy"],
        "accuracy_delta": round(our_overall["accuracy"] - bl_overall.get("accuracy", 0), 4),
        "baseline_latency_p50": bl_overall.get("latency_p50_ms", 0),
        "llm_latency_p50": our_overall["latency_p50_ms"],
        "latency_delta_p50": round(our_overall["latency_p50_ms"] - bl_overall.get("latency_p50_ms", 0), 2),
        "by_hops": {},
    }

    bl_by_hops = bl_summary.get("by_hops", {})
    for hop_label in ["1-hop", "2-hop", "3-hop"]:
        bl_hop = bl_by_hops.get(hop_label, {})
        our_hop = summary["by_hops"].get(hop_label, {})
        if bl_hop and our_hop:
            comparison["by_hops"][hop_label] = {
                "baseline_accuracy": bl_hop.get("accuracy", 0),
                "llm_accuracy": our_hop["accuracy"],
                "accuracy_delta": round(our_hop["accuracy"] - bl_hop.get("accuracy", 0), 4),
                "baseline_latency_p50": bl_hop.get("latency_p50_ms", 0),
                "llm_latency_p50": our_hop["latency_p50_ms"],
            }

    # Per-query comparison
    bl_results = {r["query_id"]: r for r in baseline.get("queries", {}).get("results", [])}
    per_query = []
    for qr in summary.get("_query_results", []):
        qid = qr.get("query_id") if isinstance(qr, dict) else None
        if qid and qid in bl_results:
            bl_q = bl_results[qid]
            per_query.append({
                "query_id": qid,
                "baseline_correct": bl_q.get("correct", 0),
                "llm_correct": qr.get("correct", 0),
                "changed": bl_q.get("correct", 0) != qr.get("correct", 0),
            })
    comparison["per_query_changes"] = per_query

    return comparison


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
    print("V2-014: Multi-hop Tests WITH Qwen3-4B LLM")
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
    llm_model = models.get("llm_model", "unknown")
    print(f"  Status: {result['status']}")
    print(f"  Relation strategy: {strategy}")
    print(f"  LLM loaded: {llm_loaded}")
    print(f"  LLM model: {llm_model}")
    print(f"  Embedder: {models.get('embedding_model', 'unknown')}")
    print(f"  NER: {models.get('ner_loaded', False)}")

    if not llm_loaded:
        print("\n  ERROR: LLM is NOT loaded. This test requires --features llm with Qwen3-4B-GGUF.")
        print("  Start server with: ./target/release/ucotron_server --config ucotron-multihop-llm.toml")
        print("  (Built with: cargo build --release --features llm)")
        sys.exit(1)

    if strategy != "llm":
        print(f"\n  WARNING: relation_strategy is '{strategy}', expected 'llm'.")

    server_config = {
        "storage_mode": result.get("storage_mode"),
        "vector_backend": result.get("vector_backend"),
        "llm_loaded": llm_loaded,
        "llm_model": llm_model,
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

    # Phase 1: Ingest all facts (with LLM relation extraction)
    print("\n" + "=" * 60)
    print("PHASE 1: Ingesting facts (LLM relation extraction active)")
    print("=" * 60)
    ingestion_results = ingest_facts(facts)
    total_memories = sum(r["memories_created"] for r in ingestion_results)
    total_entities = sum(r["entities_found"] for r in ingestion_results)
    total_relations = sum(r["relations_found"] for r in ingestion_results)
    ingestion_errors = sum(1 for r in ingestion_results if r["error"])
    ingestion_latencies = [r["latency_ms"] for r in ingestion_results]
    ingestion_p50 = sorted(ingestion_latencies)[len(ingestion_latencies) // 2] if ingestion_latencies else 0
    print(f"\n  Ingestion complete: {total_memories} memories, {total_entities} entities, {total_relations} relations, {ingestion_errors} errors")
    print(f"  Ingestion latency P50: {ingestion_p50:.0f}ms")

    # Brief pause to let indices settle
    time.sleep(2)

    # Phase 2: Run multi-hop queries
    print("\n" + "=" * 60)
    print("PHASE 2: Running multi-hop queries")
    print("=" * 60)
    query_results = run_queries(queries)

    # Phase 3: Compute summary
    summary = compute_summary(query_results)

    # Phase 4: Compare with baseline
    print("\n" + "=" * 60)
    print("BASELINE COMPARISON")
    print("=" * 60)
    baseline = load_baseline()
    summary["_query_results"] = query_results  # temp for comparison
    comparison = compare_with_baseline(summary, baseline)
    del summary["_query_results"]

    if comparison:
        print(f"  Baseline accuracy: {comparison['baseline_accuracy']*100:.1f}%")
        print(f"  LLM accuracy:     {comparison['llm_accuracy']*100:.1f}%")
        delta = comparison['accuracy_delta']
        direction = "+" if delta >= 0 else ""
        print(f"  Delta:            {direction}{delta*100:.1f}%")
        print(f"  Latency P50: baseline={comparison['baseline_latency_p50']:.0f}ms, llm={comparison['llm_latency_p50']:.0f}ms (delta={comparison['latency_delta_p50']:+.0f}ms)")
        print()
        for hop_label, hop_comp in comparison["by_hops"].items():
            bl_acc = hop_comp["baseline_accuracy"] * 100
            llm_acc = hop_comp["llm_accuracy"] * 100
            acc_delta = hop_comp["accuracy_delta"] * 100
            print(f"  {hop_label}: baseline={bl_acc:.0f}% → llm={llm_acc:.0f}% ({acc_delta:+.0f}%)")

        # Show per-query changes
        changes = [q for q in comparison.get("per_query_changes", []) if q.get("changed")]
        if changes:
            print(f"\n  Queries that changed:")
            for c in changes:
                direction = "GAINED" if c["llm_correct"] else "LOST"
                print(f"    {c['query_id']}: {direction}")
    else:
        print("  No baseline found for comparison.")

    # Print summary
    print("\n" + "=" * 60)
    print("MULTI-HOP RESULTS SUMMARY — Qwen3-4B LLM")
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
        "benchmark": "multihop-qwen3-4b",
        "description": "Multi-hop reasoning test with Qwen3-4B LLM relation extraction",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "server_config": server_config,
        "ingestion": {
            "total_facts": len(facts),
            "total_memories_created": total_memories,
            "total_entities_found": total_entities,
            "total_relations_found": total_relations,
            "ingestion_errors": ingestion_errors,
            "ingestion_latency_p50_ms": round(ingestion_p50, 2),
            "details": ingestion_results,
        },
        "queries": {
            "total": len(queries),
            "results": query_results,
        },
        "summary": summary,
        "comparison_vs_baseline": comparison,
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
