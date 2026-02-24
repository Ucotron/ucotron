#!/usr/bin/env bash
# Cross-language SDK integration test runner.
#
# Starts a Ucotron server, then runs integration tests for each SDK
# (Rust, TypeScript, Python, Go) against it, verifying they all produce
# consistent results.
#
# Usage:
#   ./scripts/cross_language_tests.sh [--rust-only] [--ts-only] [--python-only] [--go-only] [--java-only]
#
# Requires: cargo, node/npm, python3, go, java (JDK 11+)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PORT="${UCOTRON_TEST_PORT:-0}"
SERVER_PID=""
RESULTS_FILE="$(mktemp)"
PASS_COUNT=0
FAIL_COUNT=0
SKIP_COUNT=0

# Parse flags
RUN_RUST=true
RUN_TS=true
RUN_PYTHON=true
RUN_GO=true
RUN_JAVA=true

for arg in "$@"; do
    case "$arg" in
        --rust-only) RUN_TS=false; RUN_PYTHON=false; RUN_GO=false; RUN_JAVA=false ;;
        --ts-only) RUN_RUST=false; RUN_PYTHON=false; RUN_GO=false; RUN_JAVA=false ;;
        --python-only) RUN_RUST=false; RUN_TS=false; RUN_GO=false; RUN_JAVA=false ;;
        --go-only) RUN_RUST=false; RUN_TS=false; RUN_PYTHON=false; RUN_JAVA=false ;;
        --java-only) RUN_RUST=false; RUN_TS=false; RUN_PYTHON=false; RUN_GO=false ;;
        *) echo "Unknown flag: $arg"; exit 1 ;;
    esac
done

cleanup() {
    if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "[runner] Stopping server (PID $SERVER_PID)..."
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    rm -f "$RESULTS_FILE"
}
trap cleanup EXIT

# ─── Build server ────────────────────────────────────────────────────────────
echo "============================================"
echo "  Ucotron Cross-Language SDK Integration Tests"
echo "============================================"
echo ""
echo "[runner] Building ucotron_server (debug)..."
"$HOME/.cargo/bin/cargo" build --bin ucotron_server --manifest-path "$ROOT_DIR/Cargo.toml" 2>&1 | tail -3

SERVER_BIN="$ROOT_DIR/target/debug/ucotron_server"
if [ ! -f "$SERVER_BIN" ]; then
    echo "[runner] ERROR: Server binary not found at $SERVER_BIN"
    exit 1
fi

# ─── Start server ────────────────────────────────────────────────────────────
# Use a random available port.
# We pick a port, start the server, and poll /health until ready.
if [ "$PORT" = "0" ]; then
    # Find a free port
    PORT=$(python3 -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")
fi

echo "[runner] Starting server on port $PORT..."

# Create a temp data dir for LMDB
TEST_DATA_DIR="$(mktemp -d)"
export UCOTRON_SERVER_PORT="$PORT"
export UCOTRON_SERVER_HOST="127.0.0.1"
export UCOTRON_STORAGE_VECTOR_DATA_DIR="$TEST_DATA_DIR/vector"
export UCOTRON_STORAGE_GRAPH_DATA_DIR="$TEST_DATA_DIR/graph"

mkdir -p "$TEST_DATA_DIR/vector" "$TEST_DATA_DIR/graph"

"$SERVER_BIN" 2>/dev/null &
SERVER_PID=$!

# Wait for server to be ready (poll /health)
MAX_WAIT=30
WAITED=0
echo -n "[runner] Waiting for server..."
while [ $WAITED -lt $MAX_WAIT ]; do
    if curl -sf "http://127.0.0.1:$PORT/api/v1/health" > /dev/null 2>&1; then
        echo " ready! (${WAITED}s)"
        break
    fi
    sleep 1
    WAITED=$((WAITED + 1))
    echo -n "."
done

if [ $WAITED -ge $MAX_WAIT ]; then
    echo " TIMEOUT after ${MAX_WAIT}s"
    echo "[runner] Server failed to start. Checking if process is alive..."
    if kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "[runner] Process is alive but not responding on port $PORT"
    else
        echo "[runner] Process exited prematurely"
    fi
    exit 1
fi

SERVER_URL="http://127.0.0.1:$PORT"
echo "[runner] Server URL: $SERVER_URL"
echo ""

# ─── Helper: record result ───────────────────────────────────────────────────
record_result() {
    local lang="$1"
    local test_name="$2"
    local status="$3"  # PASS | FAIL | SKIP
    local detail="${4:-}"

    printf "%-12s %-40s %s\n" "[$lang]" "$test_name" "$status" >> "$RESULTS_FILE"

    case "$status" in
        PASS) PASS_COUNT=$((PASS_COUNT + 1)) ;;
        FAIL) FAIL_COUNT=$((FAIL_COUNT + 1)); [ -n "$detail" ] && echo "  Detail: $detail" >> "$RESULTS_FILE" ;;
        SKIP) SKIP_COUNT=$((SKIP_COUNT + 1)) ;;
    esac
}

# ─── Run SDK tests ───────────────────────────────────────────────────────────

# 1. Rust SDK integration tests
if $RUN_RUST; then
    echo "─── Rust SDK ───────────────────────────────────────────────"
    set +e
    UCOTRON_TEST_SERVER_URL="$SERVER_URL" \
        "$HOME/.cargo/bin/cargo" test \
        --manifest-path "$ROOT_DIR/ucotron_sdk/Cargo.toml" \
        --test cross_language \
        -- --test-threads=1 2>&1 | tail -20
    RC=${PIPESTATUS[0]}
    set -e
    if [ "$RC" -eq 0 ]; then
        record_result "Rust" "cross-language integration" "PASS"
    else
        record_result "Rust" "cross-language integration" "FAIL" "cargo test failed (exit $RC)"
    fi
    echo ""
fi

# 2. TypeScript SDK integration tests
if $RUN_TS; then
    echo "─── TypeScript SDK ─────────────────────────────────────────"
    if [ -f "$ROOT_DIR/sdks/typescript/node_modules/.package-lock.json" ]; then
        set +e
        UCOTRON_TEST_SERVER_URL="$SERVER_URL" \
            npx --prefix "$ROOT_DIR/sdks/typescript" \
            jest --config "$ROOT_DIR/sdks/typescript/jest.config.js" \
            --testPathPattern="cross_language" \
            --rootDir "$ROOT_DIR/sdks/typescript" 2>&1 | tail -20
        RC=${PIPESTATUS[0]}
        set -e
        if [ "$RC" -eq 0 ]; then
            record_result "TypeScript" "cross-language integration" "PASS"
        else
            record_result "TypeScript" "cross-language integration" "FAIL" "jest failed (exit $RC)"
        fi
    else
        record_result "TypeScript" "cross-language integration" "SKIP" "node_modules not installed"
    fi
    echo ""
fi

# 3. Python SDK integration tests
if $RUN_PYTHON; then
    echo "─── Python SDK ─────────────────────────────────────────────"
    set +e
    UCOTRON_TEST_SERVER_URL="$SERVER_URL" \
        python3 -m pytest \
        "$ROOT_DIR/sdks/python/tests/test_cross_language.py" \
        -v --tb=short 2>&1 | tail -30
    RC=${PIPESTATUS[0]}
    set -e
    if [ "$RC" -eq 0 ]; then
        record_result "Python" "cross-language integration" "PASS"
    else
        record_result "Python" "cross-language integration" "FAIL" "pytest failed (exit $RC)"
    fi
    echo ""
fi

# 4. Go SDK integration tests
if $RUN_GO; then
    echo "─── Go SDK ─────────────────────────────────────────────────"
    set +e
    (cd "$ROOT_DIR/sdks/go" && \
        UCOTRON_TEST_SERVER_URL="$SERVER_URL" \
        go test -v -run TestCrossLanguage \
        -count=1 \
        ./...) 2>&1 | tail -30
    RC=${PIPESTATUS[0]}
    set -e
    if [ "$RC" -eq 0 ]; then
        record_result "Go" "cross-language integration" "PASS"
    else
        record_result "Go" "cross-language integration" "FAIL" "go test failed (exit $RC)"
    fi
    echo ""
fi

# 5. Java SDK integration tests
if $RUN_JAVA; then
    echo "─── Java SDK ───────────────────────────────────────────────"
    if [ -f "$ROOT_DIR/sdks/java/gradlew" ]; then
        set +e
        UCOTRON_TEST_SERVER_URL="$SERVER_URL" \
            "$ROOT_DIR/sdks/java/gradlew" \
            -p "$ROOT_DIR/sdks/java" \
            :ucotron-sdk:test \
            --tests "com.ucotron.sdk.UcotronIntegrationTest" \
            --no-daemon 2>&1 | tail -30
        RC=${PIPESTATUS[0]}
        set -e
        if [ "$RC" -eq 0 ]; then
            record_result "Java" "cross-language integration" "PASS"
        else
            record_result "Java" "cross-language integration" "FAIL" "gradle test failed (exit $RC)"
        fi
    else
        record_result "Java" "cross-language integration" "SKIP" "gradlew not found"
    fi
    echo ""
fi

# ─── Summary ─────────────────────────────────────────────────────────────────
echo "============================================"
echo "  Test Results Summary"
echo "============================================"
echo ""
cat "$RESULTS_FILE"
echo ""
echo "────────────────────────────────────────────"
printf "PASS: %d  FAIL: %d  SKIP: %d\n" "$PASS_COUNT" "$FAIL_COUNT" "$SKIP_COUNT"
echo "────────────────────────────────────────────"

# Clean up data dir
rm -rf "$TEST_DATA_DIR"

if [ $FAIL_COUNT -gt 0 ]; then
    exit 1
fi
echo ""
echo "All cross-language tests passed!"
