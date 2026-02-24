#!/usr/bin/env bash
# =============================================================================
# Ucotron Zep/Graphiti Migration Script
# =============================================================================
#
# Migrates data from Zep or Graphiti to Ucotron. Supports two modes:
#
# 1. FILE MODE:   Import from a JSON file exported from Zep/Graphiti.
# 2. API MODE:    Send a JSON file to a running Ucotron server via REST API.
#
# Supported formats:
#   - Graphiti temporal KG: { "entities": [...], "episodes": [...], "edges": [...] }
#   - Zep sessions: { "sessions": [...] }
#   - Zep facts: { "facts": [...] }
#   - Bare session arrays: [{ "session_id": ..., "messages": [...] }]
#
# Usage:
#   # File mode (recommended):
#   ./scripts/migrate_from_zep.sh --file /path/to/zep_export.json
#
#   # File mode with custom namespace:
#   ./scripts/migrate_from_zep.sh --file export.json --namespace my_project
#
#   # API mode (requires running Ucotron server):
#   ./scripts/migrate_from_zep.sh --file export.json --api-url http://localhost:8420
#
# Prerequisites:
#   - Ucotron server binary built: cargo build -p ucotron-server
#   - For API mode: curl and jq installed
#
# =============================================================================

set -euo pipefail

# Defaults
FILE=""
NAMESPACE="zep_import"
API_URL=""
CONFIG=""
LINK_SAME_USER="true"
PRESERVE_EXPIRED="true"
SERVER_BIN="${UCOTRON_SERVER_BIN:-./target/debug/ucotron_server}"

usage() {
    echo "Usage: $0 [OPTIONS]"
    echo
    echo "Options:"
    echo "  --file PATH           Path to Zep/Graphiti JSON export file (required)"
    echo "  --namespace NAME      Target namespace (default: zep_import)"
    echo "  --config PATH         Path to ucotron.toml config file"
    echo "  --api-url URL         Ucotron server URL for API mode import"
    echo "  --link-same-user      Link items from same user (default: true)"
    echo "  --preserve-expired    Keep expired/invalid Graphiti edges (default: true)"
    echo "  --server-bin PATH     Path to ucotron_server binary"
    echo "  --help                Show this help message"
    echo
    echo "Examples:"
    echo "  # CLI mode (direct to storage):"
    echo "  $0 --file graphiti_export.json"
    echo
    echo "  # API mode (via running server):"
    echo "  $0 --file zep_sessions.json --api-url http://localhost:8420"
    echo
    echo "  # Graphiti export with custom namespace:"
    echo "  $0 --file graphiti_kg.json --namespace production"
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
        --preserve-expired)
            PRESERVE_EXPIRED="$2"
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

# Count items for progress reporting
ITEM_COUNTS=$(python3 -c "
import json, sys
data = json.load(open('$FILE'))
if isinstance(data, list):
    print(f'sessions={len(data)}')
elif isinstance(data, dict):
    entities = len(data.get('entities', []))
    episodes = len(data.get('episodes', []))
    edges = len(data.get('edges', []))
    sessions = len(data.get('sessions', []))
    facts = len(data.get('facts', []))
    parts = []
    if entities: parts.append(f'entities={entities}')
    if episodes: parts.append(f'episodes={episodes}')
    if edges: parts.append(f'edges={edges}')
    if sessions: parts.append(f'sessions={sessions}')
    if facts: parts.append(f'facts={facts}')
    print(', '.join(parts) if parts else 'empty')
else:
    print('unknown format')
" 2>/dev/null || echo "?")

echo "=========================================="
echo " Ucotron ← Zep/Graphiti Migration"
echo "=========================================="
echo "  File:             $FILE"
echo "  Items found:      $ITEM_COUNTS"
echo "  Namespace:        $NAMESPACE"
echo "  Link same user:   $LINK_SAME_USER"
echo "  Preserve expired: $PRESERVE_EXPIRED"
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
        --argjson preserve_expired "$PRESERVE_EXPIRED" \
        '{data: $data, link_same_user: $link_same_user, link_same_group: false, preserve_expired: $preserve_expired}')

    RESPONSE=$(curl -s -X POST "$API_URL/api/v1/import/zep" \
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

    CMD="$SERVER_BIN migrate --from zep --file $FILE --link-same-user $LINK_SAME_USER"
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
