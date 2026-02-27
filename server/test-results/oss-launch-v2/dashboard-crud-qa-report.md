# Dashboard CRUD QA Report (V2-018)

**Date:** 2026-02-26
**Server:** ucotron_server v0.1.0, port 8420 (no LLM)
**Dashboard:** Next.js 15, port 3000 (dev mode, SKIP_AUTH=true)

## Test Data

Ingested 3 test memories via API:
1. Albert Einstein (born Ulm, theory of relativity) → 4 entities, 13 edges
2. Marie Curie (physicist, radioactivity) → 2 entities, 3 edges
3. Python programming language (Guido van Rossum, 1991) → 2 entities, 3 edges

Created 1 test agent via API: `qa-test-agent`

## CRUD Flow Results

### 1. Create Memory (via ingestion)

Dashboard does not have a direct "Create Memory" form — memories are created via text ingestion (`POST /memories/text`). This is by design.

- **Status:** PASS (via API, dashboard lists ingested memories correctly)

### 2. Read Memory (list + detail)

- **Memories page loads:** PASS — shows all memories with content preview, node type, ID, timestamp
- **Click memory → detail panel:** PASS — shows content, ID, type, timestamp, unix time, metadata JSON, related entities with relation counts, relations with edge types and weights
- **Pagination:** PASS — shows "Showing 1-3", prev/next buttons work correctly
- **Type filter dropdown:** PASS — All types / entity / event / fact / skill options available

### 3. Edit Memory

- **Click Edit → edit mode:** PASS — textarea replaces content, metadata becomes editable JSON
- **Modify content and Save:** PASS — content updated in both detail panel and list view
- **Tested:** Added "and won the Nobel Prize in Physics in 1921." to Einstein memory → reflected immediately

### 4. Delete Memory

- **Click Delete → confirmation:** PASS — "Are you sure?" dialog with Confirm/Cancel buttons
- **Confirm delete:** PASS — memory removed from list, detail panel closes
- **Soft-delete:** PASS — entity "1879." deleted, related Einstein memory also removed from list (soft-delete cascade)

### 5. Search (Memories page)

- **Semantic search on memories page:** PASS — typed "relativity", clicked Search
- **Results:** 10 results returned with relevance scores (0.478 to 0.352)
- **Score display:** PASS — shows "Score: 0.416" badges
- **Clear search:** "Clear search" button available

### 6. Search (Dedicated Search page)

#### Augment (AI) Search
- **Initial state:** BUG-D2 — returned 422 error: `missing field 'context'`
- **Root cause:** `augmentQuery()` in api.ts sent `{ query }` but server expects `{ context }`
- **Fix applied:** Mapped `query` → `context` and `max_memories` → `limit` in `augmentQuery()`
- **After fix:** PASS — returns Generated Context, 10 results with scores, debug info (pipeline timings), Query Explainability panel

#### Basic Search
- **Status:** PASS — returns 8 results with score breakdowns (vector sim, centrality, recency)

### 7. Agents (Create)

- **Click "Create Agent" → modal:** PASS — shows Agent Name input, Create/Cancel buttons
- **Type name and create:** PASS — "dashboard-test-agent" created, appears in list with ID
- **Agent count updates:** PASS — badge changes from 1 to 2

### 8. Agents (List + Detail)

- **List shows agents:** PASS — name, ID, owner (Admin), Share/Clone/Delete buttons
- **Agent count badge:** PASS — shows total count

### 9. Graph Visualization

- **Page loads:** PASS — force-directed graph renders with 5 nodes, 6 edges
- **Node labels:** PASS — shows entity names (Marie Curie, Guido van Rossum, etc.)
- **Color legend:** PASS — Entity (cyan), Event (teal), Fact (green), Skill (orange)
- **Filters:** PASS — Type filter, Limit dropdown, Search nodes field
- **Interactive:** PASS — nodes are clickable, graph is draggable

## Bugs Found

### BUG-D2: Augment search sends wrong field names (FIXED)

- **Severity:** P2 (feature broken)
- **File:** `dashboard/src/lib/api.ts`
- **Issue:** `augmentQuery()` sent `{ query, max_memories, max_hops }` but server expects `{ context, limit, max_hops, debug }`
- **Fix:** Added field mapping in `augmentQuery()`: `query` → `context`, `max_memories` → `limit`, added `debug: true` by default
- **Also:** Simplified `augmentQueryDebug()` to delegate to `augmentQuery()`

## Known Minor Issues (from V2-017)

- Logo alt text warning (cosmetic)
- `/api/auth/get-session` 500 without PostgreSQL (expected with SKIP_AUTH)
- Settings heading in Spanish ("Configuración")

## Summary

| Flow | Status |
|------|--------|
| Create memory | PASS (via API ingestion) |
| Edit memory | PASS |
| Delete memory | PASS |
| Search (memories page) | PASS |
| Search (augment/AI) | PASS (after BUG-D2 fix) |
| Search (basic) | PASS |
| Create agent | PASS |
| List agents | PASS |
| Graph visualization | PASS |
| **Overall** | **PASS** |
