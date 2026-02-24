#!/usr/bin/env bash
# =============================================================================
# Ucotron Mem0 Migration Script
# =============================================================================
#
# Migrates memories from Mem0 to Ucotron. Supports two modes:
#
# 1. FILE MODE:   Import from a JSON file exported from Mem0.
# 2. API MODE:    Fetch memories from a Mem0 API endpoint and import directly.
#
# Usage:
#   # File mode (recommended):
#   ./scripts/migrate_from_mem0.sh --file /path/to/mem0_export.json
#
#   # File mode with custom namespace:
#   ./scripts/migrate_from_mem0.sh --file export.json --namespace my_project
#
#   # API mode (requires running Ucotron server):
#   ./scripts/migrate_from_mem0.sh --api-url http://localhost:8420 --file export.json
#
# Prerequisites:
#   - Ucotron server binary built: cargo build -p ucotron-server
#   - For API mode: curl and jq installed
#
# =============================================================================

set -euo pipefail

# Defaults
FILE=""
NAMESPACE="mem0_import"
API_URL=""
CONFIG=""
LINK_SAME_USER="true"
LINK_SAME_AGENT="false"
SERVER_BIN="${UCOTRON_SERVER_BIN:-./target/debug/ucotron_server}"

usage() {
    echo "Usage: $0 [OPTIONS]"
    echo
    echo "Options:"
    echo "  --file PATH         Path to Mem0 JSON export file (required)"
    echo "  --namespace NAME    Target namespace (default: mem0_import)"
    echo "  --config PATH       Path to ucotron.toml config file"
    echo "  --api-url URL       Ucotron server URL for API mode import"
    echo "  --link-same-user    Link memories from same user (default: true)"
    echo "  --link-same-agent   Link memories from same agent (default: false)"
    echo "  --server-bin PATH   Path to ucotron_server binary"
    echo "  --help              Show this help message"
    echo
    echo "Examples:"
    echo "  # CLI mode (direct to storage):"
    echo "  $0 --file mem0_export.json"
    echo
    echo "  # API mode (via running server):"
    echo "  $0 --file mem0_export.json --api-url http://localhost:8420"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --file)
            FILE="$2"
            shift 2
            ;;
        --namespace)
            NAMESPACE="$2"
            shift 2
            ;;
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --api-url)
            API_URL="$2"
            shift 2
            ;;
        --link-same-user)
            LINK_SAME_USER="$2"
            shift 2
            ;;
        --link-same-agent)
            LINK_SAME_AGENT="$2"
            shift 2
            ;;
        --server-bin)
            SERVER_BIN="$2"
            shift 2
            ;;
        --help)
            usage
            exit 0
            ;;
        *)
            echo "Error: Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Validate required arguments
if [[ -z "$FILE" ]]; then
    echo "Error: --file is required"
    usage
    exit 1
fi

if [[ ! -f "$FILE" ]]; then
    echo "Error: File not found: $FILE"
    exit 1
fi

# Validate JSON
if ! python3 -c "import json; json.load(open('$FILE'))" 2>/dev/null && \
   ! jq empty "$FILE" 2>/dev/null; then
    echo "Error: '$FILE' is not valid JSON"
    exit 1
fi

# Count memories for progress reporting
MEM_COUNT=$(python3 -c "
import json, sys
data = json.load(open('$FILE'))
if isinstance(data, list):
    print(len(data))
elif isinstance(data, dict):
    if 'results' in data:
        print(len(data['results']))
    elif 'memories' in data:
        print(len(data['memories']))
    else:
        print(1)
else:
    print(0)
" 2>/dev/null || echo "?")

echo "=========================================="
echo " Ucotron ← Mem0 Migration"
echo "=========================================="
echo "  File:           $FILE"
echo "  Memories found: $MEM_COUNT"
echo "  Namespace:      $NAMESPACE"
echo "  Link same user: $LINK_SAME_USER"
echo "  Link same agent: $LINK_SAME_AGENT"
echo "=========================================="
echo

if [[ -n "$API_URL" ]]; then
    # API mode: send to running Ucotron server
    echo "Mode: API (sending to $API_URL)"
    echo

    DATA=$(cat "$FILE")
    BODY=$(jq -n \
        --argjson data "$DATA" \
        --argjson link_same_user "$LINK_SAME_USER" \
        --argjson link_same_agent "$LINK_SAME_AGENT" \
        '{data: $data, link_same_user: $link_same_user, link_same_agent: $link_same_agent}')

    RESPONSE=$(curl -s -X POST "$API_URL/api/v1/import/mem0" \
        -H "Content-Type: application/json" \
        -H "X-Ucotron-Namespace: $NAMESPACE" \
        -d "$BODY")

    echo "Server response:"
    echo "$RESPONSE" | jq . 2>/dev/null || echo "$RESPONSE"
else
    # CLI mode: direct to storage via ucotron_server migrate subcommand
    echo "Mode: CLI (direct to storage)"
    echo

    if [[ ! -f "$SERVER_BIN" ]]; then
        echo "Error: Server binary not found at '$SERVER_BIN'"
        echo "Build it with: cargo build -p ucotron-server"
        exit 1
    fi

    CMD="$SERVER_BIN migrate --from mem0 --file $FILE --link-same-user $LINK_SAME_USER --link-same-agent $LINK_SAME_AGENT"
    if [[ -n "$NAMESPACE" ]]; then
        CMD="$CMD --namespace $NAMESPACE"
    fi
    if [[ -n "$CONFIG" ]]; then
        CMD="$CMD --config $CONFIG"
    fi

    eval "$CMD"
fi

echo
echo "Migration complete."
