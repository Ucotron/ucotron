#!/usr/bin/env python3
"""QA-011: Benchmark with Qwen3-4B-GGUF model configured.

Benchmarks /learn and /augment endpoints with the default LLM model
configured in ucotron.toml. Also benchmarks create and search for
comparison with the no-LLM baseline.
"""

import json
import time
import statistics
import requests
from datetime import datetime, timezone

BASE = "http://localhost:8420/api/v1"
NAMESPACE = "qa011-bench"
HEADERS = {"X-Ucotron-Namespace": NAMESPACE}
NUM_OPS = 20  # Per acceptance criteria: 20 operations for learn/augment
NUM_CRUD_OPS = 100  # Same as no-LLM benchmark for comparison

def percentile(data, p):
    """Calculate percentile."""
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * p / 100
    f = int(k)
    c = f + 1
    if c >= len(sorted_data):
        return sorted_data[f]
    return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])

def benchmark_create(n):
    """Benchmark memory creation."""
    latencies = []
    topics = [
        "Machine learning uses neural networks for pattern recognition",
        "Kubernetes orchestrates containerized applications across clusters",
        "Quantum computing leverages superposition for parallel computation",
        "GraphQL provides a flexible query language for APIs",
        "Rust's ownership system prevents memory safety issues at compile time",
        "Docker containers package applications with their dependencies",
        "Redis provides in-memory data structures for caching",
        "PostgreSQL supports advanced indexing and full-text search",
        "WebAssembly enables near-native performance in browsers",
        "gRPC uses Protocol Buffers for efficient RPC communication",
        "Apache Kafka handles real-time data streaming at scale",
        "Elasticsearch provides distributed full-text search capabilities",
        "TensorFlow enables deep learning model training and deployment",
        "Prometheus collects and queries time-series metrics data",
        "Nginx serves as a reverse proxy and load balancer",
        "React uses virtual DOM for efficient UI rendering",
        "MongoDB stores documents in flexible JSON-like format",
        "Terraform manages infrastructure as declarative code",
        "FastAPI builds high-performance Python web APIs automatically",
        "Envoy provides L7 proxy with observability features",
    ]
    for i in range(n):
        text = f"{topics[i % len(topics)]} (bench item {i})"
        start = time.perf_counter()
        r = requests.post(f"{BASE}/memories", json={"text": text}, headers=HEADERS)
        elapsed = (time.perf_counter() - start) * 1000
        if r.status_code in (200, 201):
            latencies.append(elapsed)
        else:
            print(f"  Create {i} failed: {r.status_code}")
    return latencies

def benchmark_search(n):
    """Benchmark vector search."""
    queries = [
        "machine learning neural networks",
        "container orchestration",
        "quantum computing",
        "API query language",
        "memory safety",
        "data caching",
        "database indexing",
        "web performance",
        "RPC communication",
        "data streaming",
    ]
    latencies = []
    for i in range(n):
        query = queries[i % len(queries)]
        start = time.perf_counter()
        r = requests.post(f"{BASE}/memories/search", json={"query": query, "limit": 5}, headers=HEADERS)
        elapsed = (time.perf_counter() - start) * 1000
        if r.status_code == 200:
            latencies.append(elapsed)
        else:
            print(f"  Search {i} failed: {r.status_code}")
    return latencies

def benchmark_augment(n):
    """Benchmark augmented retrieval."""
    contexts = [
        "Tell me about machine learning",
        "How does container orchestration work",
        "Explain quantum computing basics",
        "What are modern API patterns",
        "How does Rust handle memory safety",
        "What is data caching",
        "Explain database indexing strategies",
        "How does WebAssembly work",
        "What is gRPC used for",
        "How does event streaming work",
    ]
    latencies = []
    for i in range(n):
        ctx = contexts[i % len(contexts)]
        start = time.perf_counter()
        r = requests.post(f"{BASE}/augment", json={"context": ctx, "limit": 5}, headers=HEADERS)
        elapsed = (time.perf_counter() - start) * 1000
        if r.status_code == 200:
            latencies.append(elapsed)
        else:
            print(f"  Augment {i} failed: {r.status_code}")
    return latencies

def benchmark_learn(n):
    """Benchmark /learn endpoint (ingestion pipeline with relation extraction)."""
    conversations = [
        "User asked about deploying ML models to production. I explained that TensorFlow Serving and TorchServe are popular options. We discussed model versioning and A/B testing strategies for gradual rollouts.",
        "Discussion about microservices architecture patterns. Covered service mesh with Istio, circuit breakers with Hystrix, and event-driven communication via Kafka. User was interested in saga pattern for distributed transactions.",
        "User needed help with database optimization. We went through query plans, index strategies including partial and covering indexes, and connection pooling. Recommended pgBouncer for PostgreSQL.",
        "Conversation about CI/CD pipelines. Explained GitHub Actions workflows, Docker multi-stage builds, and ArgoCD for GitOps deployments. User wanted to implement canary releases with Flagger.",
        "Talked about observability stack. Prometheus for metrics, Grafana for dashboards, Jaeger for distributed tracing, and Loki for log aggregation. Discussed SLOs and error budgets.",
        "User asked about real-time data processing. Covered Apache Flink for stream processing, Kafka Streams for simple transformations, and ClickHouse for real-time analytics.",
        "Discussion about API security best practices. Covered OAuth2 flows, JWT token rotation, rate limiting with Redis, and API gateway patterns with Kong or Envoy.",
        "Conversation about cloud-native storage. Discussed object storage (S3), block storage (EBS), file systems (EFS), and when to use each. Covered data lifecycle policies.",
        "User needed guidance on testing strategies. Explained test pyramid, contract testing with Pact, chaos engineering with Litmus, and load testing with k6.",
        "Talked about infrastructure as code. Compared Terraform vs Pulumi vs CDK. Discussed state management, drift detection, and module composition patterns.",
        "Discussion about edge computing architectures. Covered CDN strategies, edge functions, WebSocket gateways, and geo-distributed databases like CockroachDB.",
        "User asked about data pipeline orchestration. Explained Apache Airflow DAGs, Prefect flows, Dagster assets, and dbt transformations. Discussed data quality checks.",
        "Conversation about Kubernetes networking. Covered Service types, Ingress controllers, NetworkPolicies, and service mesh data planes. Discussed mTLS and zero-trust.",
        "Talked about LLM deployment patterns. Discussed GGUF quantization, vLLM for serving, RAG pipelines, and prompt engineering. Covered cost optimization strategies.",
        "User needed help with event sourcing. Explained event stores, projections, snapshots, and CQRS. Discussed eventual consistency and compensating transactions.",
        "Discussion about WebAssembly beyond browsers. Covered WASI, Wasmtime, component model, and use cases in serverless, plugins, and edge computing.",
        "Conversation about graph databases. Compared Neo4j, DGraph, and embedded solutions. Discussed property graphs, knowledge graphs, and graph algorithms.",
        "User asked about feature flags and progressive delivery. Explained LaunchDarkly patterns, OpenFeature standard, and implementing custom flag evaluation.",
        "Talked about secrets management. Covered HashiCorp Vault, AWS Secrets Manager, SOPS, and sealed secrets. Discussed rotation strategies and least privilege.",
        "Discussion about performance profiling. Covered flamegraphs, perf events, eBPF tracing, and continuous profiling with Pyroscope or Parca.",
    ]
    latencies = []
    for i in range(n):
        conv = conversations[i % len(conversations)]
        start = time.perf_counter()
        r = requests.post(
            f"{BASE}/learn",
            json={"output": conv, "conversation_id": f"qa011-conv-{i}"},
            headers=HEADERS,
        )
        elapsed = (time.perf_counter() - start) * 1000
        if r.status_code in (200, 201):
            latencies.append(elapsed)
        else:
            print(f"  Learn {i} failed: {r.status_code} - {r.text[:200]}")
    return latencies

def compute_stats(latencies, label):
    """Compute latency statistics."""
    if not latencies:
        return {"count": 0, "error": "no successful operations"}
    return {
        "count": len(latencies),
        "min_ms": round(min(latencies), 2),
        "max_ms": round(max(latencies), 2),
        "mean_ms": round(statistics.mean(latencies), 2),
        "p50_ms": round(percentile(latencies, 50), 2),
        "p95_ms": round(percentile(latencies, 95), 2),
        "p99_ms": round(percentile(latencies, 99), 2),
    }

def load_no_llm_benchmark():
    """Load the no-LLM benchmark for comparison."""
    try:
        with open("test-results/oss-qa/benchmark-no-llm.json") as f:
            return json.load(f)
    except FileNotFoundError:
        return None

def main():
    print("=" * 60)
    print("QA-011: Benchmark with Qwen3-4B-GGUF Model")
    print("=" * 60)

    # Verify server health
    r = requests.get(f"{BASE}/health")
    health = r.json()
    print(f"\nServer status: {health['status']}")
    print(f"Embedder: {health['models']['embedder_loaded']}")
    print(f"Transcriber: {health['models']['transcriber_loaded']}")

    # Phase 1: Create memories for benchmarking
    print(f"\n--- Create benchmark ({NUM_CRUD_OPS} ops) ---")
    create_lats = benchmark_create(NUM_CRUD_OPS)
    create_stats = compute_stats(create_lats, "create")
    print(f"  P50={create_stats['p50_ms']}ms, P95={create_stats['p95_ms']}ms, P99={create_stats['p99_ms']}ms")

    # Phase 2: Search benchmark
    print(f"\n--- Search benchmark ({NUM_CRUD_OPS} ops) ---")
    search_lats = benchmark_search(NUM_CRUD_OPS)
    search_stats = compute_stats(search_lats, "search")
    search_stats["p95_target_25ms"] = "PASS" if search_stats["p95_ms"] < 25 else "FAIL"
    print(f"  P50={search_stats['p50_ms']}ms, P95={search_stats['p95_ms']}ms, P99={search_stats['p99_ms']}ms [{search_stats['p95_target_25ms']}]")

    # Phase 3: Augment benchmark
    print(f"\n--- Augment benchmark ({NUM_OPS} ops) ---")
    augment_lats = benchmark_augment(NUM_OPS)
    augment_stats = compute_stats(augment_lats, "augment")
    print(f"  P50={augment_stats['p50_ms']}ms, P95={augment_stats['p95_ms']}ms, P99={augment_stats['p99_ms']}ms")

    # Phase 4: Learn benchmark (key for this story)
    print(f"\n--- Learn benchmark ({NUM_OPS} ops) ---")
    learn_lats = benchmark_learn(NUM_OPS)
    learn_stats = compute_stats(learn_lats, "learn")
    print(f"  P50={learn_stats['p50_ms']}ms, P95={learn_stats['p95_ms']}ms, P99={learn_stats['p99_ms']}ms")

    # Load no-LLM baseline for comparison
    no_llm = load_no_llm_benchmark()
    comparison = {}
    if no_llm:
        comparison = {
            "create_overhead_ms": round(create_stats["p50_ms"] - no_llm["create"]["p50_ms"], 2),
            "search_overhead_ms": round(search_stats["p50_ms"] - no_llm["search"]["p50_ms"], 2),
            "augment_overhead_ms": round(augment_stats["p50_ms"] - no_llm["augment"]["p50_ms"], 2),
            "note": "Overhead is P50 difference between default-model and no-llm benchmarks",
        }
        print(f"\n--- Comparison vs no-LLM baseline ---")
        print(f"  Create overhead:  {comparison['create_overhead_ms']}ms (P50)")
        print(f"  Search overhead:  {comparison['search_overhead_ms']}ms (P50)")
        print(f"  Augment overhead: {comparison['augment_overhead_ms']}ms (P50)")

    # Build results
    results = {
        "benchmark": "default-model",
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model": "Qwen3-4B-GGUF (Q4_K_M, 2.3GB)",
        "num_crud_operations": NUM_CRUD_OPS,
        "num_learn_augment_operations": NUM_OPS,
        "server_config": {
            "storage_mode": "embedded",
            "vector_backend": "helix",
            "llm_model": "Qwen3-4B-GGUF",
            "llm_backend": "candle",
            "llm_inference_active": False,
            "relation_extraction": "co-occurrence (llm feature not compiled)",
        },
        "create": create_stats,
        "search": search_stats,
        "augment": augment_stats,
        "learn": learn_stats,
        "comparison_vs_no_llm": comparison,
        "findings": {
            "llm_status": "Model file present (Qwen3-4B-Q4_K_M.gguf, 2.3GB) but local LLM inference not active",
            "reason": "The 'llm' cargo feature is not compiled in. RelationStrategy::Llm falls back to co-occurrence.",
            "impact": "Learn and augment latencies are similar to no-LLM baseline since no LLM inference occurs.",
            "augment_note": "/augment endpoint is pure retrieval (vector search + graph expansion) - does not use LLM regardless of config.",
            "learn_note": "/learn runs full ingestion pipeline (chunk → embed → NER → relation extraction). Relation extraction uses co-occurrence, not LLM.",
            "recommendation": "To enable actual LLM inference: (1) compile with --features llm, (2) implement LlmRelationExtractor in relations.rs, (3) add llama-cpp-2 dependency.",
        },
    }

    # Save results
    output_path = "test-results/oss-qa/benchmark-default-model.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")
    print("\nDone!")

if __name__ == "__main__":
    main()
