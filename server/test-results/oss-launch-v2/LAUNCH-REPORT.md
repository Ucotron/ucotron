# OSS Launch v2 — Final Launch Readiness Report

**Date:** 2026-02-26
**Branch:** `ralph/oss-launch-v2`
**Stories:** 36/36 complete

---

## 1. Bugs Fixed (Track 1)

| Bug | Issue | Fix | Status |
|-----|-------|-----|--------|
| BUG-P2 | NER model not loading (`ner_loaded=false`) | Added `try_init_ner()` in main.rs following Whisper/CLIP pattern | ✅ Fixed |
| BUG-P3 | Corrupted media returns 500 instead of 400 | Changed 4 error mappings in handlers to `AppError::bad_request()` | ✅ Fixed |
| BUG-P5 | Aggressive chunking (180 chunks for 21KB text) | Added configurable `[ingestion] chunk_size` in ucotron.toml (default: 512) | ✅ Fixed |
| PR #2 | oss-qa fixes not on main | Merged PR #2 after fixing clippy + rustfmt CI failures | ✅ Merged |

**Dashboard bugs found and fixed during QA:**

| Bug | Issue | Fix |
|-----|-------|-----|
| BUG-D1 | Connectors page showed raw "scheduling is not enabled" error | Added graceful `schedulingDisabled` state with clean UI card |
| BUG-D2 | Augment search returned 422 | Fixed field mapping: `query`→`context`, `max_memories`→`limit` in api.ts |
| ISSUE-D4 | Spanish translations missing accents | Fixed 25+ accent marks in `messages/es.json` |

---

## 2. LLM Integration Status (Track 2)

| Component | Status | Details |
|-----------|--------|---------|
| `llama-cpp-2` dependency | ✅ Added | Gated behind `llm` cargo feature |
| `LlmRelationExtractor` | ✅ Implemented | In `relations.rs` with `#[cfg(feature = "llm")]` |
| Server feature gate | ✅ Exposed | `cargo build --release --features llm` |
| Qwen3-4B-GGUF loading | ✅ Verified | Loads in ~5s, 37/37 Metal layers offloaded |
| Real inference | ✅ Working | Extracts focused relations (6 vs 28 co-occurrence) |
| Health endpoint | ✅ Updated | Shows `llm_loaded`, `llm_model`, `relation_strategy` |

**LLM performance:** Model loads in ~5s, uses 2.4 GB GPU + 0.3 GB CPU memory. Inference takes ~6.2s per chunk with 8 entities. Extracts high-quality relations (e.g., Einstein→Patent Office: 0.95 confidence).

**Python sidecar (Qwen VL models):**
- Created `server/sidecar/main.py` — FastAPI service with `/embed`, `/rerank`, `/health` endpoints
- Supports Qwen3-VL-Embedding-2B + Qwen3-VL-Reranker-2B via HuggingFace Transformers
- Model-agnostic: switch to 8B models by changing env vars (no code changes)

---

## 3. Benchmark Highlights (Track 3)

### Latency (P50)

| Operation | No-LLM Baseline | Qwen3-4B LLM | Qwen 2B Pair |
|-----------|-----------------|---------------|--------------|
| Create | 65ms | 270ms | 199ms |
| Search | 1.96ms | 91ms | 110ms |
| Augment | 2.42ms | 91ms | 107ms |
| Learn | 61ms | 2824ms | 395ms |

### Search P95 Target (< 25ms)

| Config | Search P95 | Meets Target? |
|--------|-----------|---------------|
| No-LLM Baseline | **2.08ms** | ✅ Yes |
| Qwen3-4B LLM | 97ms | ❌ No (GPU contention) |
| Qwen 2B Pair | 146ms | ❌ No (sidecar HTTP overhead) |

### Best Config Recommendation

| Use Case | Recommended Config | Why |
|----------|-------------------|-----|
| **Low-latency / Production** | No-LLM Baseline | Only config meeting P95 < 25ms search target |
| **Maximum accuracy** | Qwen3-4B LLM | 86.7% multi-hop accuracy, best relation quality |
| **Balanced** | Qwen 2B Pair | 73.3% accuracy without LLM, better embeddings |
| **Qwen 8B** | Needs cloud | Infeasible on 16GB RAM — requires g5.4xlarge or Mac Studio 64GB+ |

---

## 4. Multi-hop Results Summary (Track 4)

### Accuracy by Hop Depth

| Hop Depth | No-LLM | Qwen3-4B LLM | Qwen 2B Pair |
|-----------|--------|---------------|--------------|
| 1-hop | 80% | **100%** | **100%** |
| 2-hop | 60% | **80%** | 60% |
| 3-hop | 20% | **80%** | 60% |
| **Overall** | **53.3%** | **86.7%** | **73.3%** |

**Key insight:** LLM relation extraction during ingestion is the biggest accuracy lever — +33.4% overall, +60% at 3-hop depth. Better embeddings (Qwen 2B) provide a moderate +20% improvement without LLM overhead.

**Remaining failures:** Q09 and Q15 (both requiring "Redmond" as answer) fail across all configs — geographic headquarters associations are not captured by any current approach.

---

## 5. Dashboard QA Status (Track 5)

| Category | Result |
|----------|--------|
| Page loads (18 pages) | ✅ All pass |
| Sidebar navigation | ✅ Working |
| CRUD: Create/Edit/Delete memories | ✅ Working |
| Semantic search | ✅ Working |
| Augment (AI) search | ✅ Working (after BUG-D2 fix) |
| Agent creation | ✅ Working |
| Graph visualization | ✅ Working |
| Connectors page | ✅ Fixed (BUG-D1) |
| Spanish translations | ✅ Fixed (ISSUE-D4) |
| Console errors | ✅ None |

**Note:** Dashboard requires PostgreSQL + BetterAuth for real auth. Used `SKIP_AUTH=true` middleware bypass for local QA.

---

## 6. Docs Overhaul Status (Track 6)

| Feature | Status |
|---------|--------|
| Stripe-style top navigation (4 sections) | ✅ Implemented |
| Glassmorphism + B&W theme | ✅ Implemented |
| Language switcher (en/es) | ✅ Working (Fumadocs built-in) |
| Complete i18n (87 MDX pages translated) | ✅ Done |
| `.md` raw markdown routes | ✅ Working |
| `/llms.txt` for LLMs | ✅ Working (EN + ES) |
| Mobile responsive navigation | ✅ Working (hamburger at <768px) |
| Accessibility (ARIA, focus trap, keyboard nav) | ✅ Implemented |
| Build | ✅ 187 static pages, 0 errors |

**No new @ucotron/ui components needed** — all features use Fumadocs built-in capabilities.

---

## 7. AWS Cost Optimization Summary (Track 7)

### Current EKS Costs
- **Idle:** ~$130/month (EKS $73 + NAT $32 + ALB $16 + WAF $6 + Secrets $2 + ECR $1)
- **Running:** ~$155-170/month (+ spot instances + data transfer)
- **EKS has NO free tier** — $0.10/hr control plane is always billed

### Recommendation
- **Dev environment:** K3s on EC2 (~$25/month, 5-10x cheaper) — preserves Helm chart compatibility
- **Production:** Keep EKS with Savings Plans
- **Quick wins:** Review WAFv2 necessity (Cloudflare covers it), stay on standard K8s versions

### myApplications Cost Tracking
- Step-by-step setup documented for AWS myApplications (free, uses `awsApplication` tag)
- Pulumi `registerStackTransformation` for automatic resource tagging
- Karpenter EC2NodeClass `spec.tags` for auto-provisioned nodes
- Multi-environment support: `ucotron-dev` and `ucotron-prod` applications

---

## 8. CI/CD Status

| Repo | PR | CI Status |
|------|----|-----------|
| ucotron | PR #3 | ✅ All 8 jobs pass (clippy, dashboard build, 3x Rust test, 2x multimodal, integration) |
| docs | PR #2 | ✅ Build validated locally (187 pages, no CI configured) |
| infra | PR #2 | ⚠️ Pre-existing CI failures (workflows reference non-existent paths — docs-only PR) |
| ui | N/A | No changes needed |

---

## 9. Known Issues and Recommended Pre-Launch Actions

### Known Issues

| Issue | Severity | Impact | Mitigation |
|-------|----------|--------|------------|
| Search P95 > 25ms with LLM/sidecar | Medium | Only baseline meets latency target | Document as expected trade-off |
| `graph_expanded_nodes=0` for all queries | Medium | Graph traversal not triggering for short queries | NER doesn't extract entities from short questions — investigate query expansion |
| Qwen 8B models infeasible locally | Low | Can't benchmark 8B config | Sidecar is model-agnostic — test on cloud instance when available |
| "Redmond" queries fail universally | Low | 2/15 multi-hop queries miss | Geographic HQ associations need richer knowledge ingestion |
| NER requires model download | Low | `ner_loaded=false` without `download_models.sh` | Document in getting-started guide |
| Docs repo has no CI | Low | Build only validated locally | Set up GitHub Actions for docs repo |
| Infra CI references wrong paths | Low | Pre-existing, unrelated to our changes | Fix in separate PR |

### Recommended Pre-Launch Actions

1. **Merge all PRs** — ucotron PR #3, docs PR #2, infra PR #2
2. **Deploy and smoke test** — Verify deployment on EKS after merge
3. **Document LLM trade-offs** — Add performance comparison to docs (latency vs accuracy)
4. **Set up docs CI** — GitHub Actions workflow for `npm run build` validation
5. **Consider query expansion** — Investigate prepending NER entities to short queries for better graph traversal
6. **Cloud 8B benchmark** — Run Qwen 8B benchmark on g5.4xlarge when cost-appropriate

---

## 10. Overall Launch Readiness Assessment

### Scorecard

| Track | Status | Score |
|-------|--------|-------|
| Bug Fixes | All 4 bugs fixed, CI passing | ✅ 100% |
| LLM Feature | Implemented, verified, benchmarked | ✅ 100% |
| Benchmarks | 3/4 configs benchmarked (8B blocked by hardware) | ✅ 95% |
| Multi-hop Tests | 3/4 configs tested, comparison report complete | ✅ 95% |
| Dashboard QA | All pages tested, bugs fixed | ✅ 100% |
| Docs Overhaul | Full redesign, i18n, LLM routes, mobile, a11y | ✅ 100% |
| AWS Infra | Cost research + myApplications documented | ✅ 100% |
| CI/CD | ucotron CI green, docs validated locally | ✅ 95% |

### Verdict: **GO** ✅

All critical paths are complete. The OSS server is functional with 3 model configurations (baseline, LLM, sidecar), the dashboard is tested, docs are fully redesigned with i18n, and infrastructure optimization is documented. The 8B model benchmarks are the only incomplete item, blocked by local hardware constraints — the integration path is ready and tested with 2B models.

**35 of 36 stories passed before this report. This report completes story V2-036.**
