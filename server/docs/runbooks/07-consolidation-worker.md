# Runbook: Consolidation Worker

## Overview

The consolidation worker is an async background task ("dreaming" process) that maintains knowledge graph quality. It runs automatically after a configurable number of ingestions and performs three tasks: community re-detection, entity merging, and memory decay.

---

## Configuration

```toml
[consolidation]
trigger_interval = 100          # Ingestions between consolidation runs
enable_decay = true             # Enable temporal memory decay
decay_halflife_secs = 2592000   # 30 days
```

Environment overrides:
- `UCOTRON_CONSOLIDATION_TRIGGER_INTERVAL=<number>`
- `UCOTRON_CONSOLIDATION_ENABLE_DECAY=true|false`
- `UCOTRON_CONSOLIDATION_DECAY_HALFLIFE_SECS=<seconds>`

---

## Consolidation Tasks

### 1. Leiden Community Detection

- Re-runs the Leiden algorithm on the full knowledge graph
- Detects clusters of related nodes (communities)
- Persisted in LMDB (`community_assign` and `community_members` databases)
- Used by retrieval orchestrator for community-aware search

### 2. Entity Merge

- Finds duplicate entities via name similarity (Jaccard) + embedding similarity (cosine)
- Merges duplicates: keeps the entity with more edges, redirects edges from the other
- Prevents graph bloat from repeated ingestion of the same entity

### 3. Memory Decay

- Applies exponential decay to node confidence scores based on last access time
- Formula: `confidence *= 0.5 ^ (elapsed_secs / halflife_secs)`
- Nodes below a threshold can be pruned in future consolidation cycles
- Disabled by default unless `enable_decay = true`

---

## Monitoring

The consolidation worker logs its activity:

```
INFO consolidation: Starting consolidation cycle
INFO consolidation: Community detection: 42 communities found in 1.2s
INFO consolidation: Entity merge: 15 entities merged in 0.8s
INFO consolidation: Memory decay: 230 nodes decayed in 0.3s
INFO consolidation: Consolidation cycle complete
```

Check consolidation metrics in Prometheus:
- `communities_detected` — Number of communities in last run
- `entities_merged` — Number of entity merges in last run
- `nodes_decayed` — Number of nodes with reduced confidence
- `community_detection_us` — Duration of community detection in microseconds

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| Consolidation never triggers | `trigger_interval` too high | Lower to 50 or 100 |
| Community detection slow | Graph too large (>100k nodes) | Expected; runs in background without blocking requests |
| Entities not merging | Similarity thresholds too strict | Check entity resolution config in `ucotron.toml` |
| Confidence dropping to 0 | Decay halflife too short | Increase `decay_halflife_secs` (default 30 days) |
| Worker not stopping on shutdown | Shutdown signal not received | Check for tokio runtime issues; force-stop after 30s timeout |

---

## Tuning Guidelines

- **trigger_interval = 100**: Good default for moderate ingestion rates
- **trigger_interval = 10**: For testing or low-volume deployments
- **trigger_interval = 1000**: For high-volume ingestion where consolidation overhead matters
- **decay_halflife_secs = 2592000** (30 days): Good for general knowledge
- **decay_halflife_secs = 604800** (7 days): For fast-moving information (news, chat)
- **decay_halflife_secs = 31536000** (365 days): For archival knowledge bases
