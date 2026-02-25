#!/usr/bin/env bash
# =============================================================================
# Ucotron Cloud Benchmark Suite
# =============================================================================
# Runs the standard benchmark suite against a deployed Ucotron instance and
# collects latency, throughput, and resource metrics.
#
# Prerequisites:
#   - Ucotron server running and accessible at $UCOTRON_URL
#   - kubectl configured for the target cluster (for resource metrics)
#   - curl, jq installed
#
# Usage:
#   ./scripts/cloud_benchmark.sh --provider aws --url https://ucotron.example.com
#   ./scripts/cloud_benchmark.sh --provider gcp --url http://10.0.0.5:8420 --namespace ucotron
#   ./scripts/cloud_benchmark.sh --provider azure --url http://localhost:8420 --queries 1000
#
# Output:
#   results/cloud_benchmarks/<provider>_<timestamp>.json  — raw metrics
#   results/cloud_benchmarks/<provider>_<timestamp>.md    — Markdown report
# =============================================================================

set -euo pipefail

# ─── Defaults ────────────────────────────────────────────────────────────────
PROVIDER=""
UCOTRON_URL=""
NAMESPACE="default"
API_KEY=""
NUM_DOCS=1000
NUM_QUERIES=500
TOP_K=10
HOPS=2
CONCURRENCY=10
OUTPUT_DIR="results/cloud_benchmarks"
KUBECTL_NAMESPACE="ucotron"

# ─── Parse arguments ─────────────────────────────────────────────────────────
usage() {
    echo "Usage: $0 --provider <aws|gcp|azure> --url <ucotron_url> [options]"
    echo ""
    echo "Required:"
    echo "  --provider       Cloud provider (aws, gcp, azure)"
    echo "  --url            Ucotron server URL (e.g., https://ucotron.example.com)"
    echo ""
    echo "Optional:"
    echo "  --namespace      Ucotron namespace header (default: default)"
    echo "  --api-key        API key for authenticated requests"
    echo "  --docs           Number of documents to ingest (default: 1000)"
    echo "  --queries        Number of search queries (default: 500)"
    echo "  --top-k          Top-K for search (default: 10)"
    echo "  --hops           Graph traversal hops (default: 2)"
    echo "  --concurrency    Concurrent request count (default: 10)"
    echo "  --output-dir     Output directory (default: results/cloud_benchmarks)"
    echo "  --k8s-namespace  Kubernetes namespace for resource metrics (default: ucotron)"
    echo "  --help           Show this help message"
    exit 1
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --provider)     PROVIDER="$2";          shift 2 ;;
        --url)          UCOTRON_URL="$2";       shift 2 ;;
        --namespace)    NAMESPACE="$2";         shift 2 ;;
        --api-key)      API_KEY="$2";           shift 2 ;;
        --docs)         NUM_DOCS="$2";          shift 2 ;;
        --queries)      NUM_QUERIES="$2";       shift 2 ;;
        --top-k)        TOP_K="$2";             shift 2 ;;
        --hops)         HOPS="$2";              shift 2 ;;
        --concurrency)  CONCURRENCY="$2";       shift 2 ;;
        --output-dir)   OUTPUT_DIR="$2";        shift 2 ;;
        --k8s-namespace) KUBECTL_NAMESPACE="$2"; shift 2 ;;
        --help)         usage ;;
        *)              echo "Unknown option: $1"; usage ;;
    esac
done

if [[ -z "$PROVIDER" || -z "$UCOTRON_URL" ]]; then
    echo "Error: --provider and --url are required"
    usage
fi

case "$PROVIDER" in
    aws|gcp|azure) ;;
    *) echo "Error: provider must be aws, gcp, or azure"; exit 1 ;;
esac

# ─── Setup ───────────────────────────────────────────────────────────────────
TIMESTAMP=$(date -u +"%Y%m%dT%H%M%SZ")
RUN_ID="${PROVIDER}_${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"

RESULTS_JSON="$OUTPUT_DIR/${RUN_ID}.json"
RESULTS_MD="$OUTPUT_DIR/${RUN_ID}.md"

AUTH_HEADER=""
if [[ -n "$API_KEY" ]]; then
    AUTH_HEADER="-H \"Authorization: Bearer $API_KEY\""
fi

NS_HEADER="-H \"X-Ucotron-Namespace: $NAMESPACE\""

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Ucotron Cloud Benchmark Suite                              ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Provider:    $PROVIDER"
echo "║  URL:         $UCOTRON_URL"
echo "║  Namespace:   $NAMESPACE"
echo "║  Documents:   $NUM_DOCS"
echo "║  Queries:     $NUM_QUERIES"
echo "║  Concurrency: $CONCURRENCY"
echo "║  Run ID:      $RUN_ID"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# ─── Helper functions ────────────────────────────────────────────────────────

# Make an API request and return (status_code, response_time_ms, body)
api_request() {
    local method="$1"
    local path="$2"
    local body="${3:-}"

    local curl_args=(-s -w "\n%{http_code}\n%{time_total}" -X "$method")
    curl_args+=(-H "Content-Type: application/json")
    curl_args+=(-H "X-Ucotron-Namespace: $NAMESPACE")

    if [[ -n "$API_KEY" ]]; then
        curl_args+=(-H "Authorization: Bearer $API_KEY")
    fi

    if [[ -n "$body" ]]; then
        curl_args+=(-d "$body")
    fi

    curl "${curl_args[@]}" "${UCOTRON_URL}${path}"
}

# Extract latency from curl output (time_total in seconds → ms)
parse_latency() {
    local output="$1"
    local time_s
    time_s=$(echo "$output" | tail -1)
    echo "$time_s" | awk '{printf "%.2f", $1 * 1000}'
}

# Extract HTTP status code
parse_status() {
    local output="$1"
    echo "$output" | tail -2 | head -1
}

# Calculate percentile from sorted array (file with one number per line)
percentile() {
    local file="$1"
    local p="$2"
    local count
    count=$(wc -l < "$file" | tr -d ' ')
    if [[ $count -eq 0 ]]; then
        echo "0"
        return
    fi
    local idx
    idx=$(echo "$count $p" | awk '{printf "%d", ($1 * $2 / 100) + 0.5}')
    if [[ $idx -lt 1 ]]; then idx=1; fi
    if [[ $idx -gt $count ]]; then idx=$count; fi
    sed -n "${idx}p" "$file"
}

# Calculate mean from file
mean() {
    local file="$1"
    awk '{ sum += $1; n++ } END { if (n > 0) printf "%.2f", sum/n; else print "0" }' "$file"
}

# ─── Phase 1: Health Check ──────────────────────────────────────────────────
echo "▶ Phase 1: Health Check"
HEALTH_OUTPUT=$(api_request GET "/api/v1/health")
HEALTH_STATUS=$(parse_status "$HEALTH_OUTPUT")
HEALTH_LATENCY=$(parse_latency "$HEALTH_OUTPUT")

if [[ "$HEALTH_STATUS" != "200" ]]; then
    echo "  ✗ Health check failed (HTTP $HEALTH_STATUS)"
    echo "  Response: $(echo "$HEALTH_OUTPUT" | head -1)"
    exit 1
fi
echo "  ✓ Server healthy (${HEALTH_LATENCY}ms)"
echo ""

# ─── Phase 2: Ingestion Benchmark ───────────────────────────────────────────
echo "▶ Phase 2: Ingestion Benchmark ($NUM_DOCS documents)"

INGEST_LATENCIES=$(mktemp)
INGEST_ERRORS=0
INGEST_START=$(date +%s%3N 2>/dev/null || python3 -c 'import time; print(int(time.time()*1000))')

for i in $(seq 1 "$NUM_DOCS"); do
    BODY=$(cat <<ENDJSON
{"text": "Benchmark document $i for $PROVIDER cloud deployment test. This content tests ingestion throughput and latency for the Ucotron cognitive trust framework.", "metadata": {"benchmark_run": "$RUN_ID", "doc_index": $i}}
ENDJSON
)
    OUTPUT=$(api_request POST "/api/v1/memories" "$BODY")
    STATUS=$(parse_status "$OUTPUT")
    LATENCY=$(parse_latency "$OUTPUT")

    if [[ "$STATUS" == "200" || "$STATUS" == "201" ]]; then
        echo "$LATENCY" >> "$INGEST_LATENCIES"
    else
        INGEST_ERRORS=$((INGEST_ERRORS + 1))
    fi

    # Progress indicator
    if [[ $((i % 100)) -eq 0 ]]; then
        echo "  Ingested $i / $NUM_DOCS documents..."
    fi
done

INGEST_END=$(date +%s%3N 2>/dev/null || python3 -c 'import time; print(int(time.time()*1000))')
INGEST_TOTAL_MS=$((INGEST_END - INGEST_START))
INGEST_COUNT=$(wc -l < "$INGEST_LATENCIES" | tr -d ' ')

sort -n "$INGEST_LATENCIES" -o "$INGEST_LATENCIES"
INGEST_P50=$(percentile "$INGEST_LATENCIES" 50)
INGEST_P95=$(percentile "$INGEST_LATENCIES" 95)
INGEST_P99=$(percentile "$INGEST_LATENCIES" 99)
INGEST_MEAN=$(mean "$INGEST_LATENCIES")
INGEST_THROUGHPUT=$(echo "$INGEST_COUNT $INGEST_TOTAL_MS" | awk '{if ($2 > 0) printf "%.1f", $1 / ($2 / 1000); else print "0"}')

echo "  ✓ Ingested $INGEST_COUNT documents in ${INGEST_TOTAL_MS}ms"
echo "    Throughput: ${INGEST_THROUGHPUT} docs/s"
echo "    Latency — P50: ${INGEST_P50}ms, P95: ${INGEST_P95}ms, P99: ${INGEST_P99}ms"
echo "    Errors: $INGEST_ERRORS"
echo ""

# ─── Phase 3: Search Benchmark ──────────────────────────────────────────────
echo "▶ Phase 3: Search Benchmark ($NUM_QUERIES queries)"

SEARCH_LATENCIES=$(mktemp)
SEARCH_ERRORS=0

SEARCH_TERMS=("memory retrieval" "cognitive framework" "knowledge graph" "entity resolution" "vector search" "graph traversal" "semantic similarity" "temporal decay" "community detection" "contradiction detection")

for i in $(seq 1 "$NUM_QUERIES"); do
    TERM_IDX=$((i % ${#SEARCH_TERMS[@]}))
    TERM="${SEARCH_TERMS[$TERM_IDX]}"
    BODY=$(cat <<ENDJSON
{"query": "$TERM benchmark query $i", "top_k": $TOP_K}
ENDJSON
)
    OUTPUT=$(api_request POST "/api/v1/memories/search" "$BODY")
    STATUS=$(parse_status "$OUTPUT")
    LATENCY=$(parse_latency "$OUTPUT")

    if [[ "$STATUS" == "200" ]]; then
        echo "$LATENCY" >> "$SEARCH_LATENCIES"
    else
        SEARCH_ERRORS=$((SEARCH_ERRORS + 1))
    fi

    if [[ $((i % 100)) -eq 0 ]]; then
        echo "  Executed $i / $NUM_QUERIES queries..."
    fi
done

SEARCH_COUNT=$(wc -l < "$SEARCH_LATENCIES" | tr -d ' ')
sort -n "$SEARCH_LATENCIES" -o "$SEARCH_LATENCIES"
SEARCH_P50=$(percentile "$SEARCH_LATENCIES" 50)
SEARCH_P95=$(percentile "$SEARCH_LATENCIES" 95)
SEARCH_P99=$(percentile "$SEARCH_LATENCIES" 99)
SEARCH_MEAN=$(mean "$SEARCH_LATENCIES")

echo "  ✓ Executed $SEARCH_COUNT queries"
echo "    Latency — P50: ${SEARCH_P50}ms, P95: ${SEARCH_P95}ms, P99: ${SEARCH_P99}ms"
echo "    Errors: $SEARCH_ERRORS"
echo ""

# ─── Phase 4: Augmentation Benchmark ────────────────────────────────────────
echo "▶ Phase 4: Augmentation Benchmark (context retrieval, $((NUM_QUERIES / 5)) requests)"

AUGMENT_LATENCIES=$(mktemp)
AUGMENT_ERRORS=0
AUGMENT_COUNT=$((NUM_QUERIES / 5))

for i in $(seq 1 "$AUGMENT_COUNT"); do
    TERM_IDX=$((i % ${#SEARCH_TERMS[@]}))
    TERM="${SEARCH_TERMS[$TERM_IDX]}"
    BODY=$(cat <<ENDJSON
{"messages": [{"role": "user", "content": "Tell me about $TERM"}], "max_tokens": 500}
ENDJSON
)
    OUTPUT=$(api_request POST "/api/v1/augment" "$BODY")
    STATUS=$(parse_status "$OUTPUT")
    LATENCY=$(parse_latency "$OUTPUT")

    if [[ "$STATUS" == "200" ]]; then
        echo "$LATENCY" >> "$AUGMENT_LATENCIES"
    else
        AUGMENT_ERRORS=$((AUGMENT_ERRORS + 1))
    fi
done

AUGMENT_DONE=$(wc -l < "$AUGMENT_LATENCIES" | tr -d ' ')
sort -n "$AUGMENT_LATENCIES" -o "$AUGMENT_LATENCIES"
AUGMENT_P50=$(percentile "$AUGMENT_LATENCIES" 50)
AUGMENT_P95=$(percentile "$AUGMENT_LATENCIES" 95)
AUGMENT_P99=$(percentile "$AUGMENT_LATENCIES" 99)
AUGMENT_MEAN=$(mean "$AUGMENT_LATENCIES")

echo "  ✓ Executed $AUGMENT_DONE augmentation requests"
echo "    Latency — P50: ${AUGMENT_P50}ms, P95: ${AUGMENT_P95}ms, P99: ${AUGMENT_P99}ms"
echo "    Errors: $AUGMENT_ERRORS"
echo ""

# ─── Phase 5: Concurrent Load Test ──────────────────────────────────────────
echo "▶ Phase 5: Concurrent Load Test ($CONCURRENCY parallel requests)"

CONCURRENT_LATENCIES=$(mktemp)
CONCURRENT_PIDS=()
CONCURRENT_START=$(date +%s%3N 2>/dev/null || python3 -c 'import time; print(int(time.time()*1000))')

for i in $(seq 1 "$CONCURRENCY"); do
    (
        BODY="{\"query\": \"concurrent test $i\", \"top_k\": $TOP_K}"
        OUTPUT=$(api_request POST "/api/v1/memories/search" "$BODY")
        LATENCY=$(parse_latency "$OUTPUT")
        echo "$LATENCY"
    ) >> "$CONCURRENT_LATENCIES" &
    CONCURRENT_PIDS+=($!)
done

for pid in "${CONCURRENT_PIDS[@]}"; do
    wait "$pid" 2>/dev/null || true
done

CONCURRENT_END=$(date +%s%3N 2>/dev/null || python3 -c 'import time; print(int(time.time()*1000))')
CONCURRENT_TOTAL_MS=$((CONCURRENT_END - CONCURRENT_START))
CONCURRENT_COUNT=$(wc -l < "$CONCURRENT_LATENCIES" | tr -d ' ')
sort -n "$CONCURRENT_LATENCIES" -o "$CONCURRENT_LATENCIES"
CONCURRENT_P50=$(percentile "$CONCURRENT_LATENCIES" 50)
CONCURRENT_P95=$(percentile "$CONCURRENT_LATENCIES" 95)
CONCURRENT_P99=$(percentile "$CONCURRENT_LATENCIES" 99)

echo "  ✓ $CONCURRENT_COUNT concurrent requests completed in ${CONCURRENT_TOTAL_MS}ms"
echo "    Latency — P50: ${CONCURRENT_P50}ms, P95: ${CONCURRENT_P95}ms, P99: ${CONCURRENT_P99}ms"
echo ""

# ─── Phase 6: Resource Metrics (kubectl) ─────────────────────────────────────
echo "▶ Phase 6: Resource Metrics"

CPU_USAGE="N/A"
MEM_USAGE="N/A"
POD_COUNT="N/A"

if command -v kubectl &>/dev/null; then
    POD_COUNT=$(kubectl get pods -n "$KUBECTL_NAMESPACE" -l app.kubernetes.io/name=ucotron --no-headers 2>/dev/null | wc -l | tr -d ' ') || POD_COUNT="N/A"

    if [[ "$POD_COUNT" != "N/A" && "$POD_COUNT" -gt 0 ]]; then
        TOP_OUTPUT=$(kubectl top pods -n "$KUBECTL_NAMESPACE" -l app.kubernetes.io/name=ucotron --no-headers 2>/dev/null) || true
        if [[ -n "$TOP_OUTPUT" ]]; then
            CPU_USAGE=$(echo "$TOP_OUTPUT" | awk '{sum += $2} END {print sum "m"}')
            MEM_USAGE=$(echo "$TOP_OUTPUT" | awk '{sum += $3} END {print sum "Mi"}')
        fi
    fi
    echo "  Pods: $POD_COUNT"
    echo "  CPU:  $CPU_USAGE"
    echo "  RAM:  $MEM_USAGE"
else
    echo "  kubectl not available — skipping resource metrics"
fi
echo ""

# ─── Phase 7: Server Metrics ────────────────────────────────────────────────
echo "▶ Phase 7: Server Metrics"
METRICS_OUTPUT=$(api_request GET "/api/v1/metrics")
METRICS_STATUS=$(parse_status "$METRICS_OUTPUT")
if [[ "$METRICS_STATUS" == "200" ]]; then
    METRICS_BODY=$(echo "$METRICS_OUTPUT" | head -1)
    echo "  ✓ Metrics endpoint responded"
else
    METRICS_BODY="{}"
    echo "  ✗ Metrics endpoint unavailable (HTTP $METRICS_STATUS)"
fi
echo ""

# ─── Generate JSON Results ───────────────────────────────────────────────────
cat > "$RESULTS_JSON" <<ENDJSON
{
  "run_id": "$RUN_ID",
  "provider": "$PROVIDER",
  "url": "$UCOTRON_URL",
  "namespace": "$NAMESPACE",
  "timestamp": "$TIMESTAMP",
  "config": {
    "num_docs": $NUM_DOCS,
    "num_queries": $NUM_QUERIES,
    "top_k": $TOP_K,
    "hops": $HOPS,
    "concurrency": $CONCURRENCY
  },
  "health": {
    "status": $HEALTH_STATUS,
    "latency_ms": $HEALTH_LATENCY
  },
  "ingestion": {
    "total_docs": $INGEST_COUNT,
    "total_time_ms": $INGEST_TOTAL_MS,
    "throughput_docs_per_sec": $INGEST_THROUGHPUT,
    "errors": $INGEST_ERRORS,
    "latency_ms": {
      "mean": $INGEST_MEAN,
      "p50": $INGEST_P50,
      "p95": $INGEST_P95,
      "p99": $INGEST_P99
    }
  },
  "search": {
    "total_queries": $SEARCH_COUNT,
    "errors": $SEARCH_ERRORS,
    "latency_ms": {
      "mean": $SEARCH_MEAN,
      "p50": $SEARCH_P50,
      "p95": $SEARCH_P95,
      "p99": $SEARCH_P99
    }
  },
  "augmentation": {
    "total_requests": $AUGMENT_DONE,
    "errors": $AUGMENT_ERRORS,
    "latency_ms": {
      "mean": $AUGMENT_MEAN,
      "p50": $AUGMENT_P50,
      "p95": $AUGMENT_P95,
      "p99": $AUGMENT_P99
    }
  },
  "concurrent": {
    "parallelism": $CONCURRENCY,
    "total_time_ms": $CONCURRENT_TOTAL_MS,
    "requests_completed": $CONCURRENT_COUNT,
    "latency_ms": {
      "p50": $CONCURRENT_P50,
      "p95": $CONCURRENT_P95,
      "p99": $CONCURRENT_P99
    }
  },
  "resources": {
    "pod_count": "$POD_COUNT",
    "cpu_usage": "$CPU_USAGE",
    "memory_usage": "$MEM_USAGE"
  }
}
ENDJSON

echo "  ✓ JSON results: $RESULTS_JSON"

# ─── Generate Markdown Report ────────────────────────────────────────────────
PROVIDER_UPPER=$(echo "$PROVIDER" | tr '[:lower:]' '[:upper:]')

cat > "$RESULTS_MD" <<ENDMD
# Ucotron Cloud Benchmark — $PROVIDER_UPPER

**Run ID:** $RUN_ID
**Date:** $(date -u +"%Y-%m-%d %H:%M:%S UTC")
**Server:** $UCOTRON_URL
**Namespace:** $NAMESPACE

---

## Configuration

| Parameter | Value |
|-----------|-------|
| Documents ingested | $NUM_DOCS |
| Search queries | $NUM_QUERIES |
| Top-K | $TOP_K |
| Graph hops | $HOPS |
| Concurrency | $CONCURRENCY |

---

## Results Summary

### Ingestion

| Metric | Value |
|--------|-------|
| Documents ingested | $INGEST_COUNT |
| Total time | ${INGEST_TOTAL_MS}ms |
| Throughput | ${INGEST_THROUGHPUT} docs/s |
| Mean latency | ${INGEST_MEAN}ms |
| P50 latency | ${INGEST_P50}ms |
| P95 latency | ${INGEST_P95}ms |
| P99 latency | ${INGEST_P99}ms |
| Errors | $INGEST_ERRORS |

### Search

| Metric | Value |
|--------|-------|
| Queries executed | $SEARCH_COUNT |
| Mean latency | ${SEARCH_MEAN}ms |
| P50 latency | ${SEARCH_P50}ms |
| P95 latency | ${SEARCH_P95}ms |
| P99 latency | ${SEARCH_P99}ms |
| Errors | $SEARCH_ERRORS |

### Context Augmentation

| Metric | Value |
|--------|-------|
| Requests | $AUGMENT_DONE |
| Mean latency | ${AUGMENT_MEAN}ms |
| P50 latency | ${AUGMENT_P50}ms |
| P95 latency | ${AUGMENT_P95}ms |
| P99 latency | ${AUGMENT_P99}ms |
| Errors | $AUGMENT_ERRORS |

### Concurrent Load ($CONCURRENCY parallel)

| Metric | Value |
|--------|-------|
| Total wall time | ${CONCURRENT_TOTAL_MS}ms |
| Requests completed | $CONCURRENT_COUNT |
| P50 latency | ${CONCURRENT_P50}ms |
| P95 latency | ${CONCURRENT_P95}ms |
| P99 latency | ${CONCURRENT_P99}ms |

### Resource Usage

| Metric | Value |
|--------|-------|
| Pod count | $POD_COUNT |
| CPU usage | $CPU_USAGE |
| Memory usage | $MEM_USAGE |

---

*Generated by Ucotron Cloud Benchmark Suite*
ENDMD

echo "  ✓ Markdown report: $RESULTS_MD"
echo ""

# ─── Cleanup ─────────────────────────────────────────────────────────────────
rm -f "$INGEST_LATENCIES" "$SEARCH_LATENCIES" "$AUGMENT_LATENCIES" "$CONCURRENT_LATENCIES"

# ─── Summary ─────────────────────────────────────────────────────────────────
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Benchmark Complete — $PROVIDER_UPPER"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Ingestion:  ${INGEST_THROUGHPUT} docs/s (P95: ${INGEST_P95}ms)"
echo "║  Search:     P50=${SEARCH_P50}ms  P95=${SEARCH_P95}ms  P99=${SEARCH_P99}ms"
echo "║  Augment:    P50=${AUGMENT_P50}ms  P95=${AUGMENT_P95}ms  P99=${AUGMENT_P99}ms"
echo "║  Concurrent: P95=${CONCURRENT_P95}ms (${CONCURRENCY} parallel)"
echo "║  Resources:  ${POD_COUNT} pods, CPU ${CPU_USAGE}, RAM ${MEM_USAGE}"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Output: $RESULTS_JSON"
echo "║          $RESULTS_MD"
echo "╚══════════════════════════════════════════════════════════════╝"
