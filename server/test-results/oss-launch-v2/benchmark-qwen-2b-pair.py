#!/usr/bin/env python3
"""V2-009: Benchmark with Qwen3-VL-Embedding-2B + Qwen3-VL-Reranker-2B.

Server running with sidecar embedding provider (Qwen3-VL-Embedding-2B)
and reranker (Qwen3-VL-Reranker-2B). Benchmarks:
- Memory creation: 100 ops (P50/P95/P99) — uses sidecar for embedding
- Vector search: 100 ops (P50/P95/P99) — uses sidecar for query embedding + reranker
- Augment: 20 ops (P50/P95)
- Learn: 20 ops (P50/P95) — co-occurrence relations (no LLM)

Compares against no-LLM baseline and Qwen3-4B LLM benchmark.
"""

import json
import time
import urllib.request
import urllib.error
import statistics
import sys
import os

BASE_URL = "http://localhost:8420/api/v1"
NUM_CRUD_OPS = 100
NUM_LLM_OPS = 20
NAMESPACE = "v2009-bench"
HEADERS = {"X-Ucotron-Namespace": NAMESPACE}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "benchmark-qwen-2b-pair.json")
BASELINE_FILE = os.path.join(SCRIPT_DIR, "..", "oss-qa", "benchmark-no-llm.json")
LLM_BENCH_FILE = os.path.join(SCRIPT_DIR, "benchmark-qwen3-4b-llm.json")

TOPICS = [
    "Machine learning algorithms optimize predictions using gradient descent and backpropagation.",
    "Kubernetes orchestrates containers across distributed cloud infrastructure with self-healing.",
    "Photosynthesis converts solar energy into glucose in plant chloroplasts via Calvin cycle.",
    "The Renaissance period transformed European art, science, and intellectual thought profoundly.",
    "Quantum entanglement enables instantaneous state correlation between distant particles.",
    "TCP/IP protocol stack enables reliable internet communication across heterogeneous networks.",
    "Mitochondria generate ATP through oxidative phosphorylation in eukaryotic cells.",
    "GraphQL provides flexible API queries letting clients specify exact data shapes needed.",
    "The French Revolution restructured European political systems and inspired democracy worldwide.",
    "Neural networks learn hierarchical feature representations from raw unstructured data.",
    "Docker containers isolate applications with lightweight OS-level virtualization and namespaces.",
    "DNA double helix stores genetic information using complementary base pair encoding.",
    "Rust programming language guarantees memory safety at compile time without garbage collection.",
    "The Industrial Revolution mechanized production, urbanized populations, and reshaped economics.",
    "CRISPR-Cas9 enables precise genome editing at specific DNA locations for gene therapy.",
    "WebAssembly enables near-native performance for web applications compiled from multiple languages.",
    "Blockchain distributed ledger technology ensures tamper-proof transaction records via consensus.",
    "RNA polymerase transcribes DNA sequences into messenger RNA for protein synthesis.",
    "Microservices architecture decomposes monoliths into independently deployable services with APIs.",
    "The Cold War shaped global geopolitics through nuclear deterrence and proxy conflicts.",
]

SEARCH_QUERIES = [
    "machine learning optimization gradient",
    "container orchestration kubernetes",
    "cellular energy ATP production",
    "European history transformation revolution",
    "quantum physics entanglement particles",
    "network communication TCP protocols",
    "programming language memory safety rust",
    "web application performance native",
    "genetic engineering CRISPR technology",
    "distributed systems microservices architecture",
    "biological processes photosynthesis energy",
    "data structures algorithms neural",
    "cloud infrastructure deployment containers",
    "scientific discoveries biology DNA",
    "political revolution democracy history",
    "API design patterns GraphQL",
    "database query optimization indexing",
    "artificial intelligence deep learning",
    "security cryptography blockchain consensus",
    "software engineering practices testing",
]

LEARN_CONVERSATIONS = [
    "User asked about deploying ML models to production. I explained TensorFlow Serving and TorchServe. We discussed model versioning, A/B testing, and gradual rollout strategies for production systems.",
    "Discussion about microservices architecture. Covered service mesh with Istio, circuit breakers with Hystrix, and event-driven communication via Apache Kafka. User interested in saga pattern for distributed transactions.",
    "User needed database optimization help. Covered query execution plans, index strategies including partial and covering indexes, connection pooling with pgBouncer for PostgreSQL performance.",
    "Conversation about CI/CD pipelines. Explained GitHub Actions workflows, Docker multi-stage builds, ArgoCD for GitOps. User wanted canary releases with Flagger on Kubernetes.",
    "Talked about observability stack. Prometheus for metrics, Grafana dashboards, Jaeger for distributed tracing, Loki for log aggregation. Discussed SLOs and error budgets.",
    "User asked about real-time data processing. Covered Apache Flink stream processing, Kafka Streams for transformations, ClickHouse for real-time analytics queries.",
    "Discussion about API security. OAuth2 flows, JWT token rotation, rate limiting with Redis, API gateway patterns with Kong and Envoy proxy.",
    "Cloud-native storage conversation. Object storage S3, block storage EBS, file systems EFS. Covered data lifecycle policies and cost optimization.",
    "User needed testing strategy guidance. Test pyramid, contract testing with Pact, chaos engineering with Litmus, load testing with k6 and Locust.",
    "Infrastructure as code discussion. Terraform vs Pulumi vs CDK comparison. State management, drift detection, module composition patterns.",
    "Edge computing architectures. CDN strategies, edge functions, WebSocket gateways, geo-distributed databases like CockroachDB and Spanner.",
    "Data pipeline orchestration. Apache Airflow DAGs, Prefect flows, Dagster software-defined assets, dbt transformations. Data quality validation.",
    "Kubernetes networking deep dive. Service types ClusterIP LoadBalancer, Ingress controllers, NetworkPolicies, mTLS with service mesh.",
    "LLM deployment patterns. GGUF quantization for efficiency, vLLM serving for throughput, RAG pipelines for grounding, prompt engineering techniques.",
    "Event sourcing architecture. Event stores, projections, snapshots, CQRS pattern. Eventual consistency and compensating transactions.",
    "WebAssembly beyond browsers. WASI specification, Wasmtime runtime, component model, use cases in serverless, plugins, and edge computing.",
    "Graph databases comparison. Neo4j, DGraph, and embedded solutions. Property graphs, knowledge graphs, graph traversal algorithms.",
    "Feature flags and progressive delivery. LaunchDarkly patterns, OpenFeature standard, custom flag evaluation and experimentation.",
    "Secrets management. HashiCorp Vault, AWS Secrets Manager, SOPS encryption, sealed secrets in Kubernetes. Rotation strategies.",
    "Performance profiling techniques. Flamegraphs, perf events, eBPF tracing, continuous profiling with Pyroscope and Parca tools.",
]


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
        result = {"error": e.code, "body": e.read().decode()[:200]}
    except Exception as e:
        result = {"error": str(e)}
    elapsed_ms = (time.perf_counter() - start) * 1000
    return result, elapsed_ms


def percentiles(latencies):
    """Calculate P50, P95, P99, min, max, mean."""
    if not latencies:
        return {"count": 0, "error": "no successful operations"}
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


def benchmark_create(n):
    """Benchmark memory creation (100 ops) — sidecar embedding."""
    print(f"  Creating {n} memories...")
    latencies = []
    created_ids = []
    for i in range(n):
        text = f"{TOPICS[i % len(TOPICS)]} (qwen2b-bench-{i})"
        result, ms = api_request("POST", "/memories", {"text": text})
        latencies.append(ms)
        if isinstance(result, dict) and "chunk_node_ids" in result:
            created_ids.extend(result["chunk_node_ids"])
        if (i + 1) % 25 == 0:
            print(f"    {i+1}/{n} done (last: {ms:.1f}ms)")
    stats = percentiles(latencies)
    print(f"  Create P50={stats['p50_ms']}ms P95={stats['p95_ms']}ms P99={stats['p99_ms']}ms")
    return stats, created_ids


def benchmark_search(n):
    """Benchmark vector search (100 ops) — sidecar embedding + optional reranking."""
    print(f"  Running {n} search queries...")
    latencies = []
    for i in range(n):
        query = SEARCH_QUERIES[i % len(SEARCH_QUERIES)]
        result, ms = api_request("POST", "/memories/search", {"query": query, "limit": 5})
        latencies.append(ms)
        if (i + 1) % 25 == 0:
            print(f"    {i+1}/{n} done (last: {ms:.1f}ms)")
    stats = percentiles(latencies)
    stats["p95_target_25ms"] = "PASS" if stats["p95_ms"] < 25.0 else "FAIL"
    print(f"  Search P50={stats['p50_ms']}ms P95={stats['p95_ms']}ms P99={stats['p99_ms']}ms [{stats['p95_target_25ms']}]")
    return stats


def benchmark_augment(n):
    """Benchmark augment endpoint (20 ops)."""
    print(f"  Running {n} augment queries...")
    latencies = []
    for i in range(n):
        query = SEARCH_QUERIES[i % len(SEARCH_QUERIES)]
        result, ms = api_request("POST", "/augment", {"context": query, "limit": 5})
        latencies.append(ms)
        if (i + 1) % 5 == 0:
            print(f"    {i+1}/{n} done (last: {ms:.1f}ms)")
    stats = percentiles(latencies)
    print(f"  Augment P50={stats['p50_ms']}ms P95={stats['p95_ms']}ms")
    return stats


def benchmark_learn(n):
    """Benchmark /learn endpoint (20 ops) — co-occurrence only (no LLM)."""
    print(f"  Running {n} learn operations (co-occurrence relations, sidecar embedding)...")
    latencies = []
    for i in range(n):
        conv = LEARN_CONVERSATIONS[i % len(LEARN_CONVERSATIONS)]
        result, ms = api_request("POST", "/learn", {
            "output": conv,
            "conversation_id": f"v2009-conv-{i}",
        })
        latencies.append(ms)
        is_error = isinstance(result, dict) and "error" in result
        status = f"ERROR({result.get('error','')})" if is_error else "OK"
        print(f"    Learn {i+1}/{n}: {ms:.0f}ms [{status}]")
    stats = percentiles(latencies)
    print(f"  Learn P50={stats['p50_ms']}ms P95={stats['p95_ms']}ms")
    return stats


def load_json_file(path, label):
    """Load a JSON file for comparison."""
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"  Warning: {label} not found at {path}")
        return None


def compare_benchmarks(results, baseline, llm_bench):
    """Generate comparison metrics against baseline and LLM benchmark."""
    comp = {}

    if baseline:
        comp["vs_no_llm_baseline"] = {}
        for op in ["create", "search", "augment", "learn"]:
            if op in results and op in baseline:
                comp["vs_no_llm_baseline"][f"{op}_p50_delta_ms"] = round(
                    results[op]["p50_ms"] - baseline[op]["p50_ms"], 2
                )
                comp["vs_no_llm_baseline"][f"{op}_p95_delta_ms"] = round(
                    results[op]["p95_ms"] - baseline[op]["p95_ms"], 2
                )
        comp["vs_no_llm_baseline"]["note"] = "Delta = Qwen-2B minus baseline (positive = slower)"

    if llm_bench:
        comp["vs_qwen3_4b_llm"] = {}
        for op in ["create", "search", "augment", "learn"]:
            if op in results and op in llm_bench:
                comp["vs_qwen3_4b_llm"][f"{op}_p50_delta_ms"] = round(
                    results[op]["p50_ms"] - llm_bench[op]["p50_ms"], 2
                )
                comp["vs_qwen3_4b_llm"][f"{op}_p95_delta_ms"] = round(
                    results[op]["p95_ms"] - llm_bench[op]["p95_ms"], 2
                )
        comp["vs_qwen3_4b_llm"]["note"] = "Delta = Qwen-2B minus Qwen3-4B-LLM (negative = faster)"

    return comp


def check_sidecar_health():
    """Check if the sidecar is running and healthy."""
    try:
        req = urllib.request.Request("http://localhost:8421/health")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
            return data
    except Exception as e:
        return None


def main():
    print("=" * 65)
    print("V2-009: Benchmark with Qwen3-VL-Embedding-2B + Reranker-2B")
    print("=" * 65)

    # Verify sidecar health
    print("\nChecking sidecar health...")
    sidecar_health = check_sidecar_health()
    if sidecar_health is None:
        print("ERROR: Sidecar not running at http://localhost:8421")
        print("Start it with: cd server/sidecar && source .venv/bin/activate && python main.py")
        sys.exit(1)

    print(f"  Sidecar status: {sidecar_health.get('status')}")
    print(f"  Embedding model: {sidecar_health.get('embedding_model')}")
    print(f"  Reranker model: {sidecar_health.get('reranker_model')}")
    print(f"  Embedding loaded: {sidecar_health.get('embedding_loaded')}")
    print(f"  Reranker loaded: {sidecar_health.get('reranker_loaded')}")
    print(f"  Device: {sidecar_health.get('device')}")
    print(f"  Embedding dim: {sidecar_health.get('embedding_dim')}")

    if not sidecar_health.get("embedding_loaded"):
        print("ERROR: Embedding model not loaded in sidecar!")
        sys.exit(1)

    # Verify server health
    print("\nChecking server health...")
    result, _ = api_request("GET", "/health")
    if not isinstance(result, dict) or result.get("status") != "ok":
        print(f"ERROR: Server not healthy: {result}")
        sys.exit(1)

    models = result.get("models", {})
    embedding_provider = models.get("embedding_provider", "unknown")
    reranker_loaded = models.get("reranker_loaded", False)

    print(f"  Status: {result['status']}")
    print(f"  Storage: {result.get('storage_mode')}")
    print(f"  Embedding provider: {embedding_provider}")
    print(f"  Reranker loaded: {reranker_loaded}")
    print(f"  Relation strategy: {models.get('relation_strategy', 'unknown')}")

    if embedding_provider != "sidecar":
        print(f"\nWARNING: Server embedding_provider is '{embedding_provider}', expected 'sidecar'")
        print("Server should be configured with embedding_provider = 'sidecar' in ucotron.toml")

    server_config = {
        "storage_mode": result.get("storage_mode"),
        "vector_backend": result.get("vector_backend"),
        "embedding_provider": embedding_provider,
        "reranker_loaded": reranker_loaded,
        "embedder_loaded": models.get("embedder_loaded"),
        "ner_loaded": models.get("ner_loaded"),
        "llm_loaded": models.get("llm_loaded", False),
        "relation_strategy": models.get("relation_strategy", "co_occurrence"),
    }

    sidecar_config = {
        "embedding_model": sidecar_health.get("embedding_model"),
        "reranker_model": sidecar_health.get("reranker_model"),
        "embedding_dim": sidecar_health.get("embedding_dim"),
        "device": sidecar_health.get("device"),
    }

    # Warmup
    print("\nWarmup (5 creates + 5 searches)...")
    for i in range(5):
        api_request("POST", "/memories", {"text": f"warmup text for qwen2b bench {i}"})
        api_request("POST", "/memories/search", {"query": "warmup query", "limit": 3})

    # Phase 1: Create (100 ops)
    print(f"\n[1/4] Benchmark: Memory Create ({NUM_CRUD_OPS} ops)")
    create_stats, created_ids = benchmark_create(NUM_CRUD_OPS)

    # Phase 2: Search (100 ops)
    print(f"\n[2/4] Benchmark: Vector Search ({NUM_CRUD_OPS} ops)")
    search_stats = benchmark_search(NUM_CRUD_OPS)

    # Phase 3: Augment (20 ops)
    print(f"\n[3/4] Benchmark: Augment ({NUM_LLM_OPS} ops)")
    augment_stats = benchmark_augment(NUM_LLM_OPS)

    # Phase 4: Learn (20 ops)
    print(f"\n[4/4] Benchmark: Learn ({NUM_LLM_OPS} ops)")
    learn_stats = benchmark_learn(NUM_LLM_OPS)

    # Load comparisons
    print("\nLoading comparison benchmarks...")
    baseline = load_json_file(BASELINE_FILE, "no-LLM baseline")
    llm_bench = load_json_file(LLM_BENCH_FILE, "Qwen3-4B LLM benchmark")
    comparison = compare_benchmarks(
        {"create": create_stats, "search": search_stats, "augment": augment_stats, "learn": learn_stats},
        baseline,
        llm_bench,
    )

    # Summary
    print("\n" + "=" * 65)
    print("BENCHMARK SUMMARY — Qwen3-VL-Embedding-2B + Reranker-2B")
    print("=" * 65)
    for label, stats in [("create", create_stats), ("search", search_stats),
                          ("augment", augment_stats), ("learn", learn_stats)]:
        print(f"  {label:>8}: P50={stats['p50_ms']:>9.2f}ms  P95={stats['p95_ms']:>9.2f}ms  P99={stats.get('p99_ms','N/A'):>9}ms  mean={stats['mean_ms']:>9.2f}ms  (n={stats['count']})")

    if baseline:
        print("\n  vs no-LLM baseline (P50 delta):")
        for key in sorted(comparison.get("vs_no_llm_baseline", {}).keys()):
            if key.endswith("_p50_delta_ms"):
                op = key.split("_")[0]
                val = comparison["vs_no_llm_baseline"][key]
                sign = "+" if val > 0 else ""
                print(f"    {op:>8}: {sign}{val}ms")

    if llm_bench:
        print("\n  vs Qwen3-4B LLM (P50 delta):")
        for key in sorted(comparison.get("vs_qwen3_4b_llm", {}).keys()):
            if key.endswith("_p50_delta_ms"):
                op = key.split("_")[0]
                val = comparison["vs_qwen3_4b_llm"][key]
                sign = "+" if val > 0 else ""
                print(f"    {op:>8}: {sign}{val}ms")

    print("=" * 65)

    # Build results
    results = {
        "benchmark": "qwen-2b-pair",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "embedding_model": "Qwen/Qwen3-VL-Embedding-2B",
        "reranker_model": "Qwen/Qwen3-VL-Reranker-2B",
        "embedding_dim": sidecar_health.get("embedding_dim", 384),
        "device": sidecar_health.get("device", "unknown"),
        "llm_inference_active": False,
        "relation_strategy": "co_occurrence",
        "num_crud_operations": NUM_CRUD_OPS,
        "num_llm_operations": NUM_LLM_OPS,
        "server_config": server_config,
        "sidecar_config": sidecar_config,
        "create": create_stats,
        "search": search_stats,
        "augment": augment_stats,
        "learn": learn_stats,
        "comparison": comparison,
    }

    # Cleanup
    print(f"\nCleaning up {len(created_ids)} benchmark memories...")
    for mid in created_ids:
        api_request("DELETE", f"/memories/{mid}")
    print("  Cleanup done.")

    # Save
    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUTPUT_FILE}")

    all_pass = search_stats.get("p95_target_25ms") == "PASS"
    print(f"\nOverall: {'PASS' if all_pass else 'FAIL (search P95 > 25ms)'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
