#!/usr/bin/env python3
"""QA-003: Test vector search and hybrid search

Acceptance criteria:
- Ingest at least 50 diverse text memories
- POST /memories/search with vector mode returns relevant results
- POST /memories/search with hybrid mode returns results
- POST /augment returns augmented context
- Search respects namespace boundaries (verify BUG-1 fix)
- Save search results to test-results/oss-qa/search-results.json
"""

import json
import time
import urllib.request
import urllib.error
import sys
import os

BASE_URL = "http://localhost:8420/api/v1"
RESULTS_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_FILE = os.path.join(RESULTS_DIR, "search-results.json")

results = []
raw_responses = {}


def api(method, path, body=None, headers=None):
    """Make API request and return parsed JSON."""
    url = f"{BASE_URL}{path}"
    data = json.dumps(body).encode() if body else None
    hdrs = {"Content-Type": "application/json"}
    if headers:
        hdrs.update(headers)
    req = urllib.request.Request(url, data=data, headers=hdrs, method=method)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode()), resp.status
    except urllib.error.HTTPError as e:
        body_text = e.read().decode() if e.fp else ""
        try:
            return json.loads(body_text), e.code
        except json.JSONDecodeError:
            return {"error": body_text}, e.code
    except Exception as e:
        return {"error": str(e)}, 0


def add_result(name, passed, detail):
    status = "PASS" if passed else "FAIL"
    results.append({"test": name, "status": status, "detail": detail})
    icon = "✅" if passed else "❌"
    print(f"  {icon} {name}: {detail[:120]}")


# ===================================================================
# Step 1: Ingest 50+ diverse text memories
# ===================================================================
print("=== QA-003: Vector Search & Hybrid Search ===\n")
print("--- Step 1: Ingesting 50+ diverse memories ---")

MEMORIES = [
    # Technology (10)
    "Rust is a systems programming language focused on safety, speed, and concurrency without a garbage collector.",
    "Docker containers package applications with all dependencies into standardized units for software development.",
    "Machine learning models can be trained using supervised, unsupervised, or reinforcement learning paradigms.",
    "GraphQL is a query language for APIs that gives clients the power to ask for exactly what they need.",
    "WebAssembly enables high-performance applications on web pages written in multiple languages.",
    "Kubernetes orchestrates containerized applications across clusters of machines for high availability.",
    "The TCP/IP protocol stack is the foundation of internet communication with four layers.",
    "Git is a distributed version control system that tracks changes in source code during software development.",
    "PostgreSQL is an advanced open-source relational database supporting both SQL and JSON querying.",
    "React is a JavaScript library for building user interfaces with a component-based architecture.",
    # Science (10)
    "Photosynthesis converts carbon dioxide and water into glucose and oxygen using sunlight energy.",
    "The human genome contains approximately 3 billion base pairs of DNA organized in 23 chromosome pairs.",
    "Quantum entanglement allows particles to be correlated regardless of the distance separating them.",
    "Black holes are regions of spacetime where gravity is so strong that nothing can escape, not even light.",
    "CRISPR-Cas9 is a revolutionary gene-editing technology that can precisely modify DNA sequences.",
    "The theory of plate tectonics explains the movement of Earth's lithospheric plates.",
    "Mitochondria are the powerhouses of the cell, generating most of the cell's supply of ATP.",
    "The Heisenberg uncertainty principle states you cannot simultaneously know position and momentum precisely.",
    "Evolution by natural selection is the process by which organisms with favorable traits survive and reproduce.",
    "Neurotransmitters are chemical messengers that transmit signals across synapses between neurons.",
    # History (10)
    "The Roman Empire at its peak controlled territory spanning from Britain to Mesopotamia.",
    "The Industrial Revolution began in Britain in the late 18th century transforming manufacturing processes.",
    "The Renaissance was a cultural movement that began in Italy in the 14th century reviving classical learning.",
    "The French Revolution of 1789 overthrew the monarchy and established principles of liberty and equality.",
    "Ancient Egypt developed one of the earliest writing systems known as hieroglyphics around 3200 BCE.",
    "The Silk Road was an ancient network of trade routes connecting China to the Mediterranean.",
    "World War II lasted from 1939 to 1945 and involved most of the world's nations.",
    "The printing press invented by Gutenberg around 1440 revolutionized the spread of knowledge.",
    "The Berlin Wall fell on November 9, 1989, symbolizing the end of the Cold War era.",
    "The ancient Greek city-states developed the earliest forms of democratic governance.",
    # Geography & Nature (10)
    "The Amazon rainforest produces approximately 20 percent of the world's oxygen supply.",
    "Mount Everest at 8,849 meters is the highest point on Earth above sea level.",
    "The Great Barrier Reef is the world's largest coral reef system visible from outer space.",
    "The Sahara Desert is the largest hot desert spanning 9.2 million square kilometers across Africa.",
    "The Mariana Trench is the deepest known point in Earth's oceans at nearly 11,000 meters.",
    "Antarctica contains about 70 percent of the world's fresh water locked in its ice sheets.",
    "The Nile River stretching over 6,650 kilometers is traditionally considered the longest river on Earth.",
    "Japan's island chain sits on the Pacific Ring of Fire making it prone to earthquakes and volcanoes.",
    "The Northern Lights or Aurora Borealis are caused by charged particles from the sun hitting the atmosphere.",
    "Coral reefs support approximately 25 percent of all marine species despite covering less than 1 percent of the ocean floor.",
    # Food & Culture (10)
    "Sushi originated in Southeast Asia as a method of preserving fish in fermented rice.",
    "Coffee was first discovered in Ethiopia when a goat herder noticed his goats becoming energetic after eating berries.",
    "The Mediterranean diet emphasizes olive oil, vegetables, whole grains, and moderate wine consumption.",
    "Kimchi is a traditional Korean fermented vegetable dish rich in probiotics and vitamins.",
    "Chocolate was first consumed as a bitter beverage by the ancient Mayans and Aztecs.",
    "Indian cuisine uses complex spice blends like garam masala combining cumin, coriander, and cardamom.",
    "French cuisine is renowned for techniques like sous vide, flambe, and the five mother sauces.",
    "Pasta is a staple of Italian cuisine with over 350 different shapes each designed for specific sauces.",
    "Tea culture in Japan involves the ceremonial preparation and presentation of matcha green tea.",
    "Sourdough bread uses naturally occurring wild yeast and lactobacillus bacteria for fermentation.",
    # Additional diverse (5)
    "The human brain contains approximately 86 billion neurons connected by trillions of synapses.",
    "Climate change is primarily driven by greenhouse gas emissions from burning fossil fuels.",
    "The International Space Station orbits Earth at approximately 28,000 kilometers per hour.",
    "Artificial intelligence aims to create machines capable of performing tasks that typically require human intelligence.",
    "The Fibonacci sequence appears frequently in nature from sunflower spirals to nautilus shells.",
]

ingested_ids = []
ingest_failures = 0

for i, mem in enumerate(MEMORIES):
    resp, status = api("POST", "/memories", {"text": mem})
    chunk_ids = resp.get("chunk_node_ids", [])
    if chunk_ids:
        ingested_ids.extend(chunk_ids)
    else:
        ingest_failures += 1
    if (i + 1) % 10 == 0:
        print(f"  Ingested {i+1}/{len(MEMORIES)} memories...")

print(f"  Total ingested: {len(ingested_ids)} ({ingest_failures} failures)")
add_result("ingest_50_memories", len(ingested_ids) >= 50,
           f"Ingested {len(ingested_ids)} diverse memories across 6 topic categories")

# Allow indexing time
time.sleep(1)

# ===================================================================
# Step 2: Vector search tests
# ===================================================================
print("\n--- Step 2: Vector search tests ---")

# 2a: Programming language query
search_prog, _ = api("POST", "/memories/search",
                      {"query": "programming language for systems development", "limit": 5})
raw_responses["vector_search_programming"] = search_prog
prog_results = search_prog.get("results", [])
prog_top = prog_results[0].get("content", "") if prog_results else ""

add_result("vector_search_returns_results", len(prog_results) > 0,
           f"Got {len(prog_results)} results for programming query")

add_result("vector_search_relevance",
           any(kw in prog_top.lower() for kw in ["rust", "programming", "language", "software", "code"]),
           f"Top result: {prog_top[:80]}...")

# 2b: Biology query
search_bio, _ = api("POST", "/memories/search",
                     {"query": "biological cells and energy production", "limit": 5})
raw_responses["vector_search_biology"] = search_bio
bio_results = search_bio.get("results", [])
bio_top = bio_results[0].get("content", "") if bio_results else ""

add_result("vector_search_biology",
           len(bio_results) > 0 and any(kw in bio_top.lower() for kw in ["cell", "mitochondria", "atp", "photosynthesis", "neuron", "energy"]),
           f"Top result: {bio_top[:80]}...")

# 2c: History query
search_hist, _ = api("POST", "/memories/search",
                      {"query": "ancient civilizations and empires", "limit": 5})
raw_responses["vector_search_history"] = search_hist
hist_results = search_hist.get("results", [])

add_result("vector_search_history", len(hist_results) > 0,
           f"History search returned {len(hist_results)} results")

# 2d: Limit parameter
search_limit, _ = api("POST", "/memories/search",
                       {"query": "science and nature", "limit": 3})
limit_results = search_limit.get("results", [])

add_result("vector_search_limit", 0 < len(limit_results) <= 3,
           f"Limit=3 returned {len(limit_results)} results")

# ===================================================================
# Step 3: Hybrid search tests (vector + graph + entities)
# ===================================================================
print("\n--- Step 3: Hybrid search tests ---")

# 3a: Entity-rich query
search_entity, _ = api("POST", "/memories/search",
                        {"query": "What happened in Berlin during the Cold War?", "limit": 5})
raw_responses["hybrid_search_berlin"] = search_entity
entity_results = search_entity.get("results", [])

add_result("hybrid_search_entities", len(entity_results) > 0,
           f"Entity-rich query returned {len(entity_results)} results")

# 3b: Mindset parameter
search_mindset, _ = api("POST", "/memories/search",
                         {"query": "how does the internet work", "limit": 5, "query_mindset": "convergent"})
mindset_results = search_mindset.get("results", [])

add_result("hybrid_search_mindset", len(mindset_results) > 0,
           f"Mindset query returned {len(mindset_results)} results")

# 3c: Verify hybrid score components in results
if prog_results:
    r = prog_results[0]
    has_vsim = "vector_sim" in r
    has_graph = "graph_centrality" in r
    has_recency = "recency" in r
    has_score = "score" in r
    detail = f"vector_sim={has_vsim}({r.get('vector_sim','N/A')}) graph_centrality={has_graph}({r.get('graph_centrality','N/A')}) recency={has_recency}({r.get('recency','N/A')}) score={has_score}({r.get('score','N/A')})"
    add_result("hybrid_score_components", has_vsim and has_graph and has_recency and has_score, detail)
else:
    add_result("hybrid_score_components", False, "No results to check score components")

# ===================================================================
# Step 4: Augment endpoint
# ===================================================================
print("\n--- Step 4: Augment endpoint tests ---")

# 4a: Basic augment
augment_resp, aug_status = api("POST", "/augment",
                                {"context": "Tell me about space exploration and the universe", "limit": 5})
raw_responses["augment_response"] = augment_resp

has_memories = "memories" in augment_resp and len(augment_resp.get("memories", [])) > 0
has_context = "context_text" in augment_resp and len(augment_resp.get("context_text", "")) > 0
has_entities = "entities" in augment_resp

add_result("augment_returns_context",
           has_memories and has_context,
           f"memories={len(augment_resp.get('memories',[]))} context_len={len(augment_resp.get('context_text',''))} entities={len(augment_resp.get('entities',[]))}")

# 4b: Augment with debug
debug_resp, _ = api("POST", "/augment",
                     {"context": "How do computers process information?", "limit": 5, "debug": True})
raw_responses["augment_debug"] = debug_resp

debug_info = debug_resp.get("debug")
has_debug = debug_info is not None and len(debug_info) > 0 if isinstance(debug_info, dict) else False
timings = debug_info.get("pipeline_timings", {}) if isinstance(debug_info, dict) else {}

add_result("augment_debug_info", has_debug,
           f"Debug present={has_debug}, timings keys={list(timings.keys())[:5] if timings else 'none'}")

# ===================================================================
# Step 5: Namespace isolation (BUG-1 verification)
# ===================================================================
print("\n--- Step 5: Namespace isolation (BUG-1 fix) ---")

# Create memories in separate namespaces
alpha_resp, _ = api("POST", "/memories",
                     {"text": "Alpha namespace secret: the password is unicorn42"},
                     headers={"X-Ucotron-Namespace": "alpha"})
alpha_ids = alpha_resp.get("chunk_node_ids", [])
alpha_id = alpha_ids[0] if alpha_ids else None

beta_resp, _ = api("POST", "/memories",
                    {"text": "Beta namespace secret: the password is dragon99"},
                    headers={"X-Ucotron-Namespace": "beta"})
beta_ids = beta_resp.get("chunk_node_ids", [])
beta_id = beta_ids[0] if beta_ids else None

time.sleep(1)

# Search in alpha - should only find alpha data
alpha_search, _ = api("POST", "/memories/search",
                       {"query": "what is the password", "limit": 10},
                       headers={"X-Ucotron-Namespace": "alpha"})
raw_responses["namespace_alpha_search"] = alpha_search

alpha_results = alpha_search.get("results", [])
alpha_contents = " ".join(r.get("content", "") for r in alpha_results)
found_alpha_in_alpha = "unicorn42" in alpha_contents
found_beta_in_alpha = "dragon99" in alpha_contents

add_result("namespace_isolation_alpha", not found_beta_in_alpha,
           f"Alpha search: found_own={found_alpha_in_alpha}, leaked_beta={found_beta_in_alpha}, count={len(alpha_results)}")

# Search in beta - should only find beta data
beta_search, _ = api("POST", "/memories/search",
                      {"query": "what is the password", "limit": 10},
                      headers={"X-Ucotron-Namespace": "beta"})
raw_responses["namespace_beta_search"] = beta_search

beta_results = beta_search.get("results", [])
beta_contents = " ".join(r.get("content", "") for r in beta_results)
found_beta_in_beta = "dragon99" in beta_contents
found_alpha_in_beta = "unicorn42" in beta_contents

add_result("namespace_isolation_beta", not found_alpha_in_beta,
           f"Beta search: found_own={found_beta_in_beta}, leaked_alpha={found_alpha_in_beta}, count={len(beta_results)}")

# Search in default - should NOT find namespaced data
default_search, _ = api("POST", "/memories/search",
                         {"query": "what is the password", "limit": 10})
raw_responses["namespace_default_search"] = default_search

default_results = default_search.get("results", [])
default_contents = " ".join(r.get("content", "") for r in default_results)
leaked_alpha = "unicorn42" in default_contents
leaked_beta = "dragon99" in default_contents

add_result("namespace_isolation_default", not leaked_alpha and not leaked_beta,
           f"Default search: leaked_alpha={leaked_alpha}, leaked_beta={leaked_beta}, count={len(default_results)}")

# Cleanup namespace data
if alpha_id:
    api("DELETE", f"/memories/{alpha_id}", headers={"X-Ucotron-Namespace": "alpha"})
if beta_id:
    api("DELETE", f"/memories/{beta_id}", headers={"X-Ucotron-Namespace": "beta"})

# ===================================================================
# Save results
# ===================================================================
print("\n--- Saving results ---")

passed = sum(1 for r in results if r["status"] == "PASS")
failed = sum(1 for r in results if r["status"] == "FAIL")

final = {
    "test_suite": "QA-003: Vector Search & Hybrid Search",
    "total_tests": len(results),
    "passed": passed,
    "failed": failed,
    "memories_ingested": len(ingested_ids),
    "results": results,
    "raw_responses": raw_responses,
}

with open(RESULTS_FILE, "w") as f:
    json.dump(final, f, indent=2)

print(f"\n=== QA-003 SUMMARY ===")
print(f"  Tests passed: {passed}")
print(f"  Tests failed: {failed}")
print(f"  Results saved to: {RESULTS_FILE}")

if failed == 0:
    print("\n  🎉 ALL TESTS PASSED")
    sys.exit(0)
else:
    print("\n  ⚠️  SOME TESTS FAILED")
    sys.exit(1)
