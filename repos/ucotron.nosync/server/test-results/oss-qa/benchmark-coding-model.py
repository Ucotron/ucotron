#!/usr/bin/env python3
"""QA-014: Benchmark with small coding model (Qwen3-0.6B) and compare all benchmarks.

Measures P50/P95/P99 latency for:
- Memory creation (100 operations)
- Vector search (100 queries)
- Learn (20 operations)
- Augment (20 operations)

Compares results with no-LLM and default-model (Qwen3-4B) benchmarks.
"""

import json
import time
import urllib.request
import urllib.error
import statistics
import random
import sys
import os

BASE_URL = "http://localhost:8420/api/v1"
NUM_OPS = 100
NUM_LLM_OPS = 20
OUTPUT_FILE = "benchmark-coding-model.json"

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

# Conversations for learn benchmarks
LEARN_CONVERSATIONS = [
    "User asked about Rust async patterns. I explained that Rust uses futures and async/await for concurrency. The tokio runtime is the most popular async executor. Key patterns include select!, join!, and spawn for concurrent tasks.",
    "User wanted to understand Docker networking. I described bridge networks, host networking, and overlay networks. Port mapping uses -p flag. Docker Compose simplifies multi-container networking with service names as DNS.",
    "User asked about database indexing strategies. I covered B-tree indexes for range queries, hash indexes for equality, GIN indexes for full-text search in PostgreSQL, and composite indexes for multi-column queries.",
    "User inquired about CI/CD best practices. I recommended trunk-based development, automated testing gates, blue-green deployments, and canary releases. GitHub Actions and GitLab CI are popular choices.",
    "User asked about WebSocket vs SSE. I explained WebSocket provides full-duplex communication while SSE is server-to-client only. SSE reconnects automatically and works over HTTP/2. WebSocket needs its own protocol upgrade.",
    "User wanted to know about Kubernetes resource limits. I described requests vs limits, QoS classes (Guaranteed, Burstable, BestEffort), LimitRanges for namespace defaults, and ResourceQuotas for aggregate limits.",
    "User asked about GraphQL vs REST. I explained GraphQL eliminates over-fetching with client-specified queries, supports subscriptions for real-time data, and uses a type system for API contracts. REST is simpler for CRUD operations.",
    "User inquired about memory management in Go. I described the garbage collector (concurrent, tri-color mark-sweep), stack vs heap allocation, escape analysis, and the GOGC tuning parameter.",
    "User asked about event-driven architecture. I covered event sourcing, CQRS pattern, message brokers (Kafka, RabbitMQ), eventual consistency, and saga pattern for distributed transactions.",
    "User wanted to understand TLS/SSL handshake. I explained certificate verification, key exchange (ECDHE), cipher suite negotiation, session resumption with tickets, and the differences between TLS 1.2 and 1.3.",
]


def api_request(method, path, data=None, namespace=None):
    """Make an API request and return (response_dict, latency_ms)."""
    url = f"{BASE_URL}{path}"
    headers = {"Content-Type": "application/json"}
    if namespace:
        headers["X-Ucotron-Namespace"] = namespace
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
        "p99_ms": round(s[max(0, int(n * 0.99))], 2),
    }


def benchmark_create(n=NUM_OPS, namespace="qa014-bench"):
    """Benchmark memory creation."""
    print(f"  Creating {n} memories...")
    latencies = []
    created_ids = []
    for i in range(n):
        text = f"{random.choice(TOPICS)} (qa014-bench-{i})"
        result, ms = api_request("POST", "/memories", {"text": text}, namespace=namespace)
        latencies.append(ms)
        if "chunk_node_ids" in result:
            created_ids.extend(result["chunk_node_ids"])
        if (i + 1) % 25 == 0:
            print(f"    {i+1}/{n} done (last: {ms:.1f}ms)")
    stats = percentiles(latencies)
    print(f"  Create P50={stats['p50_ms']}ms P95={stats['p95_ms']}ms P99={stats['p99_ms']}ms")
    return stats, latencies, created_ids


def benchmark_search(n=NUM_OPS, namespace="qa014-bench"):
    """Benchmark vector search."""
    print(f"  Running {n} search queries...")
    latencies = []
    for i in range(n):
        query = random.choice(SEARCH_QUERIES)
        result, ms = api_request("POST", "/memories/search", {
            "query": query,
            "limit": 5,
        }, namespace=namespace)
        latencies.append(ms)
        if (i + 1) % 25 == 0:
            print(f"    {i+1}/{n} done (last: {ms:.1f}ms)")
    stats = percentiles(latencies)
    print(f"  Search P50={stats['p50_ms']}ms P95={stats['p95_ms']}ms P99={stats['p99_ms']}ms")
    return stats, latencies


def benchmark_learn(n=NUM_LLM_OPS, namespace="qa014-bench"):
    """Benchmark learn endpoint (LLM-dependent)."""
    print(f"  Running {n} learn operations...")
    latencies = []
    for i in range(n):
        conversation = random.choice(LEARN_CONVERSATIONS)
        result, ms = api_request("POST", "/learn", {
            "output": conversation,
            "conversation_id": f"qa014-learn-{i}",
        }, namespace=namespace)
        latencies.append(ms)
        if (i + 1) % 5 == 0:
            print(f"    {i+1}/{n} done (last: {ms:.1f}ms)")
    stats = percentiles(latencies)
    print(f"  Learn P50={stats['p50_ms']}ms P95={stats['p95_ms']}ms")
    return stats, latencies


def benchmark_augment(n=NUM_LLM_OPS, namespace="qa014-bench"):
    """Benchmark augment endpoint (LLM-dependent)."""
    print(f"  Running {n} augment queries...")
    latencies = []
    for i in range(n):
        query = random.choice(SEARCH_QUERIES)
        result, ms = api_request("POST", "/augment", {
            "context": query,
            "limit": 5,
            "debug": True,
        }, namespace=namespace)
        latencies.append(ms)
        if (i + 1) % 5 == 0:
            print(f"    {i+1}/{n} done (last: {ms:.1f}ms)")
    stats = percentiles(latencies)
    print(f"  Augment P50={stats['p50_ms']}ms P95={stats['p95_ms']}ms")
    return stats, latencies


def load_previous_benchmarks():
    """Load no-LLM and default-model benchmarks for comparison."""
    previous = {}
    for name, filename in [("no_llm", "benchmark-no-llm.json"), ("default_model", "benchmark-default-model.json")]:
        path = os.path.join(os.path.dirname(OUTPUT_FILE) or ".", filename)
        if os.path.exists(path):
            with open(path) as f:
                previous[name] = json.load(f)
            print(f"  Loaded {name} benchmark from {filename}")
        else:
            print(f"  WARNING: {filename} not found, comparison will be partial")
    return previous


def compare_benchmarks(current, previous):
    """Generate comparison table between all benchmark runs."""
    comparison = {}
    ops = ["create", "search", "augment", "learn"]

    for op in ops:
        comp = {}
        if op in current:
            comp["coding_model"] = {
                "p50_ms": current[op]["p50_ms"],
                "p95_ms": current[op]["p95_ms"],
                "mean_ms": current[op]["mean_ms"],
            }

        for bench_name, bench_data in previous.items():
            if op in bench_data:
                comp[bench_name] = {
                    "p50_ms": bench_data[op]["p50_ms"],
                    "p95_ms": bench_data[op]["p95_ms"],
                    "mean_ms": bench_data[op]["mean_ms"],
                }

        if comp:
            comparison[op] = comp

    return comparison


def main():
    # Verify server health
    print("Checking server health...")
    result, _ = api_request("GET", "/health")
    if result.get("status") != "ok":
        print(f"ERROR: Server not healthy: {result}")
        sys.exit(1)
    print(f"  Server OK — storage: {result.get('storage_mode')}, embedder: {result.get('models', {}).get('embedder_loaded')}")

    # Load previous benchmarks
    print("\nLoading previous benchmarks for comparison...")
    previous = load_previous_benchmarks()

    results = {
        "benchmark": "coding-model",
        "model": "Qwen3-0.6B-GGUF (Q8_0, 610MB, 0.6B params)",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "num_create_ops": NUM_OPS,
        "num_search_ops": NUM_OPS,
        "num_learn_ops": NUM_LLM_OPS,
        "num_augment_ops": NUM_LLM_OPS,
        "server_config": {
            "storage_mode": result.get("storage_mode"),
            "vector_backend": result.get("vector_backend"),
            "llm_model": "Qwen3-0.6B-GGUF",
            "llm_backend": "candle",
        },
        "notes": {
            "llm_feature_compiled": False,
            "llm_inference_active": False,
            "explanation": "The 'llm' cargo feature is not compiled into the binary. "
                          "RelationStrategy::Llm falls back to co-occurrence extraction. "
                          "/augment is pure retrieval (vector search + graph expansion), never uses LLM. "
                          "/learn runs ingestion pipeline where relation extraction falls back to co-occurrence. "
                          "Therefore, benchmarks with different LLM models show equivalent performance "
                          "since LLM inference is never invoked.",
            "model_comparison": "Qwen3-0.6B (0.6B params, 610MB Q8_0) vs Qwen3-4B (4B params, 2.3GB Q4_K_M) — "
                               "no observable performance difference because neither model is actually used for inference.",
        },
    }

    # Warmup: 5 creates + 5 searches in qa014-bench namespace
    print("\nWarmup (5 creates + 5 searches)...")
    for i in range(5):
        api_request("POST", "/memories", {"text": f"qa014 warmup text {i}"}, namespace="qa014-bench")
        api_request("POST", "/memories/search", {"query": "warmup query", "limit": 3}, namespace="qa014-bench")

    # Benchmark create
    print(f"\n[1/4] Benchmark: Memory Create ({NUM_OPS} ops)")
    create_stats, create_latencies, created_ids = benchmark_create()
    results["create"] = create_stats

    # Benchmark search
    print(f"\n[2/4] Benchmark: Vector Search ({NUM_OPS} ops)")
    search_stats, search_latencies = benchmark_search()
    results["search"] = search_stats

    # Check P95 target
    search_p95_pass = search_stats["p95_ms"] < 25.0
    results["search"]["p95_target_25ms"] = "PASS" if search_p95_pass else "FAIL"
    print(f"  P95 target <25ms: {'PASS' if search_p95_pass else 'FAIL'} ({search_stats['p95_ms']}ms)")

    # Benchmark learn
    print(f"\n[3/4] Benchmark: Learn ({NUM_LLM_OPS} ops)")
    learn_stats, learn_latencies = benchmark_learn()
    results["learn"] = learn_stats

    # Benchmark augment
    print(f"\n[4/4] Benchmark: Augment ({NUM_LLM_OPS} ops)")
    augment_stats, augment_latencies = benchmark_augment()
    results["augment"] = augment_stats

    # Generate comparison
    print("\n" + "=" * 80)
    print("BENCHMARK COMPARISON: No-LLM vs Qwen3-4B (default) vs Qwen3-0.6B (coding)")
    print("=" * 80)

    comparison = compare_benchmarks(results, previous)
    results["comparison"] = comparison

    # Print comparison table
    print(f"\n{'Operation':<10} | {'Metric':<8} | {'No-LLM':<12} | {'Qwen3-4B':<12} | {'Qwen3-0.6B':<12}")
    print("-" * 70)

    for op in ["create", "search", "augment", "learn"]:
        if op not in comparison:
            continue
        comp = comparison[op]
        for metric in ["p50_ms", "p95_ms", "mean_ms"]:
            label = metric.replace("_ms", "").upper()
            no_llm = f"{comp.get('no_llm', {}).get(metric, '-')}"
            default = f"{comp.get('default_model', {}).get(metric, '-')}"
            coding = f"{comp.get('coding_model', {}).get(metric, '-')}"
            print(f"{op:<10} | {label:<8} | {no_llm:<12} | {default:<12} | {coding:<12}")
        print("-" * 70)

    # Quality comparison notes
    results["quality_comparison"] = {
        "conclusion": "No observable quality difference between models because LLM inference is not active.",
        "qwen3_4b": {
            "params": "4B",
            "quantization": "Q4_K_M",
            "file_size_mb": 2300,
            "effective_impact": "None (model loaded but not used for inference)",
        },
        "qwen3_0_6b": {
            "params": "0.6B",
            "quantization": "Q8_0",
            "file_size_mb": 610,
            "effective_impact": "None (model configured but not used for inference)",
        },
        "recommendation": "When the 'llm' cargo feature is implemented and compiled, "
                         "re-run these benchmarks to measure actual LLM inference overhead. "
                         "Expected: Qwen3-4B will produce higher quality relation extraction "
                         "but with higher latency; Qwen3-0.6B will be faster but potentially "
                         "lower quality for complex relation extraction tasks.",
    }

    # Summary
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY (Coding Model: Qwen3-0.6B-GGUF)")
    print("=" * 80)
    for op_name, op_key in [("Create", "create"), ("Search", "search"), ("Learn", "learn"), ("Augment", "augment")]:
        if op_key in results:
            s = results[op_key]
            print(f"  {op_name:>8}: P50={s['p50_ms']:>8.2f}ms  P95={s['p95_ms']:>8.2f}ms  mean={s['mean_ms']:>8.2f}ms  (n={s['count']})")
    print("=" * 80)
    print(f"\n  NOTE: LLM feature NOT compiled — all models fall back to co-occurrence.")
    print(f"  Response quality is identical across all three configurations.")

    # Cleanup benchmark memories
    print(f"\nCleaning up {len(created_ids)} benchmark memories...")
    for mid in created_ids:
        api_request("DELETE", f"/memories/{mid}", namespace="qa014-bench")
    print("  Cleanup done.")

    # Delete benchmark namespace
    api_request("DELETE", "/admin/namespaces/qa014-bench")

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
