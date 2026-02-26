#!/usr/bin/env python3
"""QA-004: Benchmark performance without LLM model.

Measures P50/P95/P99 latency for:
- Memory creation (100 operations)
- Vector search (100 queries)
- Augment (100 queries)

Target: Vector search P95 < 25ms
"""

import json
import time
import urllib.request
import urllib.error
import statistics
import random
import sys

BASE_URL = "http://localhost:8420/api/v1"
NUM_OPS = 100
OUTPUT_FILE = "benchmark-no-llm.json"

# Diverse text corpus for creates
TOPICS = [
    "Machine learning algorithms optimize predictions using gradient descent.",
    "Kubernetes orchestrates containers across distributed cloud infrastructure.",
    "Photosynthesis converts solar energy into glucose in plant chloroplasts.",
    "The Renaissance period transformed European art and intellectual thought.",
    "Quantum entanglement enables instantaneous state correlation between particles.",
    "TCP/IP protocol stack enables reliable internet communication worldwide.",
    "Mitochondria generate ATP through oxidative phosphorylation in eukaryotic cells.",
    "GraphQL provides flexible API queries with client-specified data shapes.",
    "The French Revolution fundamentally restructured European political systems.",
    "Neural networks learn hierarchical feature representations from raw data.",
    "Docker containers isolate applications with lightweight OS-level virtualization.",
    "DNA double helix stores genetic information using base pair encoding.",
    "Rust programming language guarantees memory safety without garbage collection.",
    "The Industrial Revolution mechanized production and urbanized populations.",
    "CRISPR-Cas9 enables precise genome editing at specific DNA locations.",
    "WebAssembly enables near-native performance for web applications.",
    "Blockchain distributed ledger technology ensures tamper-proof transaction records.",
    "RNA polymerase transcribes DNA sequences into messenger RNA molecules.",
    "Microservices architecture decomposes monoliths into independently deployable services.",
    "The Cold War shaped global geopolitics through nuclear deterrence strategies.",
]

SEARCH_QUERIES = [
    "machine learning optimization",
    "container orchestration",
    "cellular energy production",
    "European history transformation",
    "quantum physics particles",
    "network communication protocols",
    "programming language memory",
    "web application performance",
    "genetic engineering technology",
    "distributed systems architecture",
    "biological processes energy",
    "data structures algorithms",
    "cloud infrastructure deployment",
    "scientific discoveries biology",
    "political revolution history",
    "API design patterns",
    "database query optimization",
    "artificial intelligence neural",
    "security cryptography blockchain",
    "software engineering practices",
]


def api_request(method, path, data=None):
    """Make an API request and return (response_dict, latency_ms)."""
    url = f"{BASE_URL}{path}"
    headers = {"Content-Type": "application/json"}
    body = json.dumps(data).encode() if data else None
    req = urllib.request.Request(url, data=body, headers=headers, method=method)

    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req) as resp:
            result = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        result = {"error": e.code, "body": e.read().decode()[:200]}
    except Exception as e:
        result = {"error": str(e)}
    elapsed_ms = (time.perf_counter() - start) * 1000
    return result, elapsed_ms


def percentiles(latencies):
    """Calculate P50, P95, P99, min, max, mean."""
    s = sorted(latencies)
    n = len(s)
    return {
        "count": n,
        "min_ms": round(s[0], 2),
        "max_ms": round(s[-1], 2),
        "mean_ms": round(statistics.mean(s), 2),
        "p50_ms": round(s[int(n * 0.50)], 2),
        "p95_ms": round(s[int(n * 0.95)], 2),
        "p99_ms": round(s[int(n * 0.99)], 2),
    }


def benchmark_create(n=NUM_OPS):
    """Benchmark memory creation."""
    print(f"  Creating {n} memories...")
    latencies = []
    created_ids = []
    for i in range(n):
        text = f"{random.choice(TOPICS)} (bench-{i})"
        result, ms = api_request("POST", "/memories", {"text": text})
        latencies.append(ms)
        if "chunk_node_ids" in result:
            created_ids.extend(result["chunk_node_ids"])
        if (i + 1) % 25 == 0:
            print(f"    {i+1}/{n} done (last: {ms:.1f}ms)")
    stats = percentiles(latencies)
    print(f"  Create P50={stats['p50_ms']}ms P95={stats['p95_ms']}ms P99={stats['p99_ms']}ms")
    return stats, latencies, created_ids


def benchmark_search(n=NUM_OPS):
    """Benchmark vector search."""
    print(f"  Running {n} search queries...")
    latencies = []
    for i in range(n):
        query = random.choice(SEARCH_QUERIES)
        result, ms = api_request("POST", "/memories/search", {
            "query": query,
            "limit": 5,
        })
        latencies.append(ms)
        if (i + 1) % 25 == 0:
            print(f"    {i+1}/{n} done (last: {ms:.1f}ms)")
    stats = percentiles(latencies)
    print(f"  Search P50={stats['p50_ms']}ms P95={stats['p95_ms']}ms P99={stats['p99_ms']}ms")
    return stats, latencies


def benchmark_augment(n=NUM_OPS):
    """Benchmark augment endpoint."""
    print(f"  Running {n} augment queries...")
    latencies = []
    for i in range(n):
        query = random.choice(SEARCH_QUERIES)
        result, ms = api_request("POST", "/augment", {
            "context": query,
            "limit": 5,
        })
        latencies.append(ms)
        if (i + 1) % 25 == 0:
            print(f"    {i+1}/{n} done (last: {ms:.1f}ms)")
    stats = percentiles(latencies)
    print(f"  Augment P50={stats['p50_ms']}ms P95={stats['p95_ms']}ms P99={stats['p99_ms']}ms")
    return stats, latencies


def main():
    # Verify server health
    print("Checking server health...")
    result, _ = api_request("GET", "/health")
    if result.get("status") != "ok":
        print(f"ERROR: Server not healthy: {result}")
        sys.exit(1)
    print(f"  Server OK — storage: {result.get('storage_mode')}, embedder: {result.get('models',{}).get('embedder_loaded')}")

    # Verify no LLM model
    llm_model = result.get("models", {}).get("llm_model", "")
    print(f"  LLM model: '{llm_model}' (should be empty/disabled)")

    results = {
        "benchmark": "no-llm",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "num_operations": NUM_OPS,
        "server_config": {
            "storage_mode": result.get("storage_mode"),
            "vector_backend": result.get("vector_backend"),
            "llm_model": llm_model or "disabled",
        },
    }

    # Warmup: 5 creates + 5 searches
    print("\nWarmup (5 creates + 5 searches)...")
    for i in range(5):
        api_request("POST", "/memories", {"text": f"warmup text {i}"})
        api_request("POST", "/memories/search", {"query": "warmup query", "limit": 3})

    # Benchmark create
    print(f"\n[1/3] Benchmark: Memory Create ({NUM_OPS} ops)")
    create_stats, create_latencies, created_ids = benchmark_create()
    results["create"] = create_stats

    # Benchmark search
    print(f"\n[2/3] Benchmark: Vector Search ({NUM_OPS} ops)")
    search_stats, search_latencies = benchmark_search()
    results["search"] = search_stats

    # Check P95 target
    search_p95_pass = search_stats["p95_ms"] < 25.0
    results["search"]["p95_target_25ms"] = "PASS" if search_p95_pass else "FAIL"
    print(f"  P95 target <25ms: {'PASS' if search_p95_pass else 'FAIL'} ({search_stats['p95_ms']}ms)")

    # Benchmark augment
    print(f"\n[3/3] Benchmark: Augment ({NUM_OPS} ops)")
    augment_stats, augment_latencies = benchmark_augment()
    results["augment"] = augment_stats

    # Summary
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY (No LLM)")
    print("=" * 60)
    for op in ["create", "search", "augment"]:
        s = results[op]
        print(f"  {op:>8}: P50={s['p50_ms']:>8.2f}ms  P95={s['p95_ms']:>8.2f}ms  P99={s['p99_ms']:>8.2f}ms  mean={s['mean_ms']:>8.2f}ms")
    print("=" * 60)

    # Cleanup benchmark memories
    print(f"\nCleaning up {len(created_ids)} benchmark memories...")
    for mid in created_ids:
        api_request("DELETE", f"/memories/{mid}")
    print("  Cleanup done.")

    # Save results
    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUTPUT_FILE}")

    # Overall pass/fail
    all_pass = search_p95_pass
    print(f"\nOverall: {'PASS' if all_pass else 'FAIL'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
