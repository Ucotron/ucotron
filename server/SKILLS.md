# Ucotron Skills Guide

A complete usage guide for the Ucotron cognitive trust framework.

---

## Quick Start

### Option A: MCP Server (Claude Desktop / Cursor)

Add to your MCP configuration (`claude_desktop_config.json` or `.cursor/mcp.json`):

```json
{
  "mcpServers": {
    "ucotron": {
      "command": "ucotron_mcp",
      "args": [],
      "env": {
        "UCOTRON_STORAGE_PATH": "/path/to/ucotron-data"
      }
    }
  }
}
```

Then use the tools directly in conversation:

```
> Remember that Alice works at Acme Corp and lives in Madrid.

ucotron_add_memory("Alice works at Acme Corp and lives in Madrid")
→ 3 memories created, 2 entities found (Alice, Acme Corp)

> What do you know about Alice?

ucotron_search("Alice")
→ Alice works at Acme Corp (score: 0.92)
→ Alice lives in Madrid (score: 0.88)
```

### Option B: REST API + SDK

Start the server:

```bash
# With Docker
docker compose up -d

# Or from source
cd memory_arena
cargo run --release --bin ucotron_server
```

Use the Rust SDK:

```rust
use ucotron_sdk::UcotronClient;

let client = UcotronClient::new("http://localhost:8420");

// Store a memory
let result = client.add_memory("Alice works at Acme Corp", None).await?;
println!("Created {} memories", result.chunk_node_ids.len());

// Search memories
let results = client.search("Who works at Acme?", None).await?;
for item in &results.results {
    println!("{}: {} (score: {:.2})", item.id, item.content, item.score);
}

// Augment context for an LLM prompt
let context = client.augment("Tell me about Alice", None).await?;
println!("Context: {}", context.context_text);
```

### Option C: cURL

```bash
# Store a memory
curl -X POST http://localhost:8420/api/v1/memories \
  -H "Content-Type: application/json" \
  -H "X-Ucotron-Namespace: default" \
  -d '{"text": "Alice works at Acme Corp and lives in Madrid"}'

# Search
curl -X POST http://localhost:8420/api/v1/memories/search \
  -H "Content-Type: application/json" \
  -d '{"query": "Who works at Acme?", "limit": 5}'

# Augment (get context for LLM)
curl -X POST http://localhost:8420/api/v1/augment \
  -H "Content-Type: application/json" \
  -d '{"context": "Tell me about Alice", "limit": 10}'
```

---

## Core Concepts

### Memory Types

Ucotron organizes knowledge into four node types:

| Type | Purpose | Example |
|------|---------|---------|
| **Entity** | Stable semantic knowledge about people, places, things | "Alice", "Acme Corp", "Madrid" |
| **Event** | Time-bound episodic experiences | "Alice joined Acme Corp in 2024" |
| **Fact** | Knowledge claims with confidence scores | "Alice lives in Madrid (confidence: 0.95)" |
| **Skill** | Registered tools and capabilities | "Can translate English to Spanish" |

### Relationships (Edge Types)

Memories are connected through typed edges:

| Edge Type | Meaning |
|-----------|---------|
| `RelatesTo` | General semantic relationship |
| `CausedBy` | Causal relationship between events |
| `ConflictsWith` | Contradictory information |
| `NextEpisode` | Sequential events in a timeline |
| `HasProperty` | Entity attribute |
| `Supersedes` | Newer information replaces older |
| `Actor` | Entity that performed an event |
| `Object` | Entity acted upon in an event |
| `Location` | Where an event occurred |
| `Companion` | Co-participant in an event |

### The Ingestion Pipeline

When you store text, Ucotron runs an 8-step pipeline:

1. **Chunking** — Split text into manageable segments
2. **Embedding** — Generate 384-dim vectors (all-MiniLM-L6-v2)
3. **NER** — Extract named entities (GLiNER, zero-shot)
4. **Relation Extraction** — Identify relationships between entities
5. **Entity Resolution** — Deduplicate entities (Jaccard + cosine similarity)
6. **Contradiction Detection** — Flag conflicting facts
7. **Graph Storage** — Persist nodes and edges in LMDB
8. **Community Assignment** — Leiden clustering for topic groups

### The Retrieval Pipeline (LazyGraphRAG)

Search uses a multi-stage retrieval pipeline:

1. **Vector Search** — Find semantically similar memories via HNSW
2. **Entity Extraction** — Identify entities in the query
3. **Graph Expansion** — Traverse 1-2 hops from matching nodes
4. **Community Selection** — Include community-related memories
5. **Re-ranking** — Score by vector similarity + graph centrality + recency
6. **Temporal Decay** — Downweight older memories
7. **Deduplication** — Remove duplicate content
8. **Result Assembly** — Return ranked results with metadata

### Multi-Tenancy

Isolate data per user, organization, or agent via the `X-Ucotron-Namespace` header:

```bash
# Store in "alice" namespace
curl -X POST http://localhost:8420/api/v1/memories \
  -H "X-Ucotron-Namespace: alice" \
  -d '{"text": "I prefer dark mode"}'

# Search only "alice" namespace
curl -X POST http://localhost:8420/api/v1/memories/search \
  -H "X-Ucotron-Namespace: alice" \
  -d '{"query": "UI preferences"}'
```

Data in one namespace is invisible to other namespaces.

---

## Operations

### Storing Memories

**POST /api/v1/memories** — Ingest text through the full pipeline.

```json
// Request
{
  "text": "Bob met Alice at the Berlin conference in March 2025. They discussed the new ML pipeline.",
  "metadata": {
    "source": "meeting_notes",
    "importance": "high"
  }
}

// Response
{
  "chunk_node_ids": [1, 2],
  "entity_node_ids": [3, 4, 5],
  "edges_created": 6,
  "metrics": {
    "embedding_ms": 12,
    "ner_ms": 8,
    "total_ms": 45
  }
}
```

**POST /api/v1/learn** — Extract memories from agent output (e.g., after a conversation).

```json
// Request
{
  "output": "I helped the user set up their Python environment. They prefer using Poetry over pip and work on macOS."
}

// Response
{
  "memories_created": 3,
  "entities_found": 2,
  "conflicts_found": 0
}
```

### Searching Memories

**POST /api/v1/memories/search** — Semantic search with graph re-ranking.

```json
// Request
{
  "query": "What happened at the conference?",
  "limit": 5,
  "node_type": "Event",
  "time_range": {
    "start": 1709251200,
    "end": 1711929600
  }
}

// Response
{
  "results": [
    {
      "id": 1,
      "content": "Bob met Alice at the Berlin conference in March 2025",
      "node_type": "Event",
      "score": 0.89,
      "vector_sim": 0.85,
      "graph_centrality": 0.72,
      "recency": 0.95
    }
  ],
  "total": 1,
  "query": "What happened at the conference?"
}
```

### Context Augmentation

**POST /api/v1/augment** — Get formatted context for LLM prompts.

```json
// Request
{
  "context": "The user is asking about Alice's work",
  "limit": 10
}

// Response
{
  "memories": [...],
  "entities": [...],
  "context_text": "Relevant memories:\n- Alice works at Acme Corp\n- Alice lives in Madrid\n\nRelated entities:\n- Alice (Entity): connected to Acme Corp, Madrid"
}
```

The `context_text` field is ready to inject into a system prompt.

### Entity Management

```bash
# List all entities
curl http://localhost:8420/api/v1/entities?limit=20

# Get entity with relationships
curl http://localhost:8420/api/v1/entities/42
```

### Memory CRUD

```bash
# Get a specific memory
curl http://localhost:8420/api/v1/memories/1

# Update a memory
curl -X PUT http://localhost:8420/api/v1/memories/1 \
  -H "Content-Type: application/json" \
  -d '{"content": "Updated content", "metadata": {"reviewed": "true"}}'

# Delete a memory (soft delete)
curl -X DELETE http://localhost:8420/api/v1/memories/1

# List memories with pagination
curl "http://localhost:8420/api/v1/memories?limit=10&offset=0&node_type=Entity"
```

---

## MCP Tools Reference

The MCP server exposes 6 tools for use with Claude Desktop, Cursor, and other MCP-compatible clients.

### ucotron_add_memory

Store text as structured memory. The text is chunked, embedded, and entities/relations are extracted automatically.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `text` | string | Yes | The text to ingest as a memory |

**Example:**

```
ucotron_add_memory(text: "Alice works at Acme Corp in Madrid")
→ {
    "memories_created": 2,
    "entities_found": 2
  }
```

**What happens:** The ingestion pipeline splits the text, generates embeddings, extracts entities ("Alice", "Acme Corp", "Madrid"), identifies relations ("works at", "located in"), and stores everything in the knowledge graph.

### ucotron_search

Semantic search across all memories using vector similarity and graph-based re-ranking.

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `query` | string | Yes | — | Natural language query |
| `limit` | integer | No | 10 | Maximum results to return |

**Example:**

```
ucotron_search(query: "Who works at Acme?", limit: 5)
→ [
    {
      "id": 1,
      "content": "Alice works at Acme Corp",
      "node_type": "Entity",
      "score": 0.92
    }
  ]
```

**Tips:**
- Use natural language queries for best results ("What does Alice do?" vs "Alice job")
- Lower `limit` for focused answers, higher for broader exploration
- Results are ranked by combined vector similarity, graph centrality, and recency

### ucotron_get_entity

Look up a named entity and its 1-hop neighbors in the knowledge graph.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `name` | string | Yes | The name of the entity to look up |

**Example:**

```
ucotron_get_entity(name: "Alice")
→ {
    "id": 3,
    "name": "Alice",
    "node_type": "Entity",
    "neighbors": [
      "Acme Corp",
      "Madrid",
      "Bob met Alice at the Berlin conference"
    ]
  }
```

**When to use:** When you need to explore relationships around a specific entity. The neighbors reveal how the entity is connected in the knowledge graph.

### ucotron_list_entities

List entities from the knowledge graph with optional type filter.

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `type` | string | No | "entity" | Filter: "entity", "event", "fact", "skill" |
| `limit` | integer | No | 20 | Maximum results to return |

**Example:**

```
ucotron_list_entities(type: "Entity", limit: 20)
→ [
    { "id": 3, "name": "Alice", "node_type": "Entity", "neighbors": [] },
    { "id": 4, "name": "Acme Corp", "node_type": "Entity", "neighbors": [] }
  ]
```

### ucotron_augment

Get formatted context for LLM prompt injection. Returns relevant memories, entities, and a pre-formatted `context_text` string.

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `context` | string | Yes | — | The context or user message to augment |
| `limit` | integer | No | 10 | Maximum number of memories to include |

**Example:**

```
ucotron_augment(context: "The user is asking about Alice", limit: 10)
→ {
    "memories": [
      { "id": 1, "content": "Alice works at Acme Corp", "node_type": "Entity", "score": 0.92 },
      { "id": 2, "content": "Alice lives in Madrid", "node_type": "Fact", "score": 0.88 }
    ],
    "entities": ["Alice", "Acme Corp", "Madrid"],
    "context_text": "Relevant memories:\n- Alice works at Acme Corp\n- Alice lives in Madrid\n\nRelated entities:\n- Alice (Entity): connected to Acme Corp, Madrid"
  }
```

**Best practice:** Call `ucotron_augment` once at the start of each conversation turn with the user's message. Inject the `context_text` field into your system prompt.

### ucotron_learn

Extract and store memories from agent output. Use this after generating a response to persist any new facts the agent produced.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `output` | string | Yes | Agent output or conversation text to extract memories from |

**Example:**

```
ucotron_learn(output: "I helped Alice set up Python. She prefers Poetry over pip.")
→ {
    "memories_created": 2,
    "entities_found": 1,
    "conflicts_found": 0
  }
```

**When to use:** After every agent response that contains factual information about the user, their preferences, or events. The learn endpoint runs the full ingestion pipeline including entity extraction and contradiction detection.

---

## MCP Integration Guide

### Claude Desktop Setup

Add this to your Claude Desktop config file:

**macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`
**Windows:** `%APPDATA%\Claude\claude_desktop_config.json`
**Linux:** `~/.config/Claude/claude_desktop_config.json`

#### Option 1: From Source (Development)

```json
{
  "mcpServers": {
    "ucotron": {
      "command": "/path/to/memory_arena/target/release/ucotron_mcp",
      "args": [],
      "env": {
        "UCOTRON_STORAGE_PATH": "/path/to/ucotron-data",
        "UCOTRON_MODELS_DIR": "/path/to/memory_arena/models",
        "RUST_LOG": "info"
      }
    }
  }
}
```

Build the binary first:

```bash
cd memory_arena
./scripts/download_models.sh    # Download ONNX models (first time only)
cargo build --release --bin ucotron_mcp
```

#### Option 2: With Docker

Run the Ucotron REST server via Docker, then use a bridge script:

```json
{
  "mcpServers": {
    "ucotron": {
      "command": "/path/to/memory_arena/target/release/ucotron_mcp",
      "args": [],
      "env": {
        "UCOTRON_CONFIG": "/path/to/ucotron.toml"
      }
    }
  }
}
```

#### Option 3: With Config File

Point to a custom TOML configuration:

```json
{
  "mcpServers": {
    "ucotron": {
      "command": "ucotron_mcp",
      "args": [],
      "env": {
        "UCOTRON_CONFIG": "/path/to/ucotron.toml",
        "RUST_LOG": "info"
      }
    }
  }
}
```

Where `ucotron.toml` contains:

```toml
[storage]
mode = "embedded"
path = "./ucotron-data"

[storage.hnsw]
ef_construction = 200
m = 16
ef_search = 100

[models]
models_dir = "./models"
embedding_model = "all-MiniLM-L6-v2"
ner_model = "gliner_small-v2.1"
```

### Cursor IDE Setup

Add to your project's `.cursor/mcp.json` or global Cursor MCP settings:

```json
{
  "mcpServers": {
    "ucotron": {
      "command": "/path/to/memory_arena/target/release/ucotron_mcp",
      "args": [],
      "env": {
        "UCOTRON_STORAGE_PATH": "/path/to/ucotron-data",
        "UCOTRON_MODELS_DIR": "/path/to/memory_arena/models"
      }
    }
  }
}
```

For project-specific memory, create `.cursor/mcp.json` in your project root:

```json
{
  "mcpServers": {
    "ucotron": {
      "command": "/absolute/path/to/ucotron_mcp",
      "args": [],
      "env": {
        "UCOTRON_STORAGE_PATH": "./ucotron-data",
        "UCOTRON_MODELS_DIR": "/absolute/path/to/models"
      }
    }
  }
}
```

### Claude Code Setup

Add to your project's `.mcp.json` for Claude Code:

```json
{
  "mcpServers": {
    "ucotron": {
      "command": "/path/to/memory_arena/target/release/ucotron_mcp",
      "args": [],
      "env": {
        "UCOTRON_STORAGE_PATH": "/path/to/ucotron-data",
        "UCOTRON_MODELS_DIR": "/path/to/memory_arena/models"
      }
    }
  }
}
```

### Verifying the Connection

After configuring, verify Ucotron is connected:

1. **Claude Desktop:** Look for "ucotron" in the MCP tools list (hammer icon). You should see all 6 tools.
2. **Cursor:** Open the MCP panel and verify "ucotron" appears as a connected server.
3. **Test with a simple command:**

```
> Remember that the sky is blue

Claude will call ucotron_add_memory(text: "the sky is blue")
→ { "memories_created": 1, "entities_found": 0 }
```

```
> What do you remember?

Claude will call ucotron_search(query: "what do you remember")
→ [{ "id": 1, "content": "the sky is blue", "score": 0.87 }]
```

### Troubleshooting

| Issue | Solution |
|-------|----------|
| "Server not found" | Verify the path to `ucotron_mcp` binary is absolute and correct |
| "Failed to load embedding model" | Run `./scripts/download_models.sh` to download ONNX models |
| Tools not appearing | Restart Claude Desktop / Cursor after config changes |
| "Permission denied" | Ensure `ucotron_mcp` is executable: `chmod +x target/release/ucotron_mcp` |
| Slow first response | First call initializes ONNX models (~1-2s). Subsequent calls are fast |
| Logs not visible | Set `RUST_LOG=debug` in env for verbose logging (logs go to stderr) |

### MCP Tool Usage Examples

Below are complete conversation examples showing how Claude uses Ucotron tools in practice.

#### Example 1: Building Long-Term Memory

```
User: My name is Alice and I'm a senior engineer at Acme Corp. I work on the ML platform team.

Claude calls: ucotron_add_memory(text: "Alice is a senior engineer at Acme Corp. She works on the ML platform team.")
→ { "memories_created": 2, "entities_found": 3 }

Claude: Got it! I've stored that you're a senior engineer at Acme Corp working on the ML platform team.

---

User: I just moved from San Francisco to Berlin last month.

Claude calls: ucotron_add_memory(text: "Alice moved from San Francisco to Berlin recently.")
→ { "memories_created": 1, "entities_found": 3 }

Claude: Noted — you've recently relocated from San Francisco to Berlin.

---

User: What do you know about me?

Claude calls: ucotron_augment(context: "What do you know about the user?", limit: 10)
→ {
    "memories": [
      { "content": "Alice is a senior engineer at Acme Corp", "score": 0.91 },
      { "content": "Alice works on the ML platform team", "score": 0.88 },
      { "content": "Alice moved from San Francisco to Berlin", "score": 0.85 }
    ],
    "entities": ["Alice", "Acme Corp", "San Francisco", "Berlin"],
    "context_text": "Relevant memories:\n- Alice is a senior engineer at Acme Corp\n- Alice works on the ML platform team\n- Alice moved from San Francisco to Berlin\n\nRelated entities:\n- Alice (Entity): connected to Acme Corp, Berlin, San Francisco"
  }

Claude: Here's what I know about you:
- You're Alice, a senior engineer at Acme Corp on the ML platform team
- You recently moved from San Francisco to Berlin
```

#### Example 2: Entity Exploration

```
User: Tell me about all the people I've mentioned.

Claude calls: ucotron_list_entities(type: "Entity", limit: 50)
→ [
    { "id": 3, "name": "Alice", "node_type": "Entity" },
    { "id": 7, "name": "Bob", "node_type": "Entity" },
    { "id": 12, "name": "Acme Corp", "node_type": "Entity" }
  ]

Claude calls: ucotron_get_entity(name: "Bob")
→ { "id": 7, "name": "Bob", "node_type": "Entity", "neighbors": ["Berlin conference", "Alice"] }

Claude: I have records of two people:
- **Alice** (you) — senior engineer at Acme Corp, recently moved to Berlin
- **Bob** — connected to you through the Berlin conference
```

#### Example 3: Learning from Agent Output

```
User: Can you help me write a Python script to process CSV files?

Claude: [generates Python code with pandas...]

Claude calls: ucotron_learn(output: "Helped Alice write a Python CSV processing script using pandas. She prefers pandas over the csv module. Uses Python 3.11.")
→ { "memories_created": 2, "entities_found": 1, "conflicts_found": 0 }
```

#### Example 4: Contradiction Detection

```
User: Actually, I work at TechStart now, not Acme Corp.

Claude calls: ucotron_add_memory(text: "Alice works at TechStart, not Acme Corp anymore.")
→ { "memories_created": 1, "entities_found": 2 }
// Ucotron detects the conflict with "Alice works at Acme Corp" and marks
// the older fact as Superseded, creating a CONFLICTS_WITH edge.
```

---

## Integration Patterns

### Pattern 1: Conversational Agent

Use Ucotron to give an LLM agent persistent memory across sessions.

```
┌─────────────┐     ┌──────────┐     ┌─────────┐
│  User Input  │────▶│   LLM    │────▶│ Response │
└─────────────┘     └────┬─────┘     └─────────┘
                         │
                    ┌────▼─────┐
                    │ Ucotron  │
                    │ augment  │  ← retrieve context before generation
                    │ learn    │  ← store new facts after generation
                    └──────────┘
```

**Flow:**
1. User sends message
2. Call `ucotron_augment(user_message)` to get relevant context
3. Inject context into system prompt
4. LLM generates response
5. Call `ucotron_learn(llm_response)` to extract and store new knowledge

**Example system prompt:**

```
You are a helpful assistant with persistent memory.

{context_text from ucotron_augment}

Use the above context to personalize your responses.
When you learn new facts about the user, they will be automatically stored.
```

### Pattern 2: Knowledge Base

Use Ucotron as a structured knowledge base with entity relationships.

**Flow:**
1. Ingest documents via `POST /api/v1/memories`
2. Query with semantic search: `POST /api/v1/memories/search`
3. Explore entity graph: `GET /api/v1/entities/{id}`
4. Get structured context: `POST /api/v1/augment`

**Best for:** FAQ systems, documentation search, company wikis.

### Pattern 3: Personal Assistant

Use Ucotron to remember user preferences, history, and context.

**Flow:**
1. After each interaction, call `ucotron_learn` with the conversation summary
2. Before responding, call `ucotron_augment` with the user's query
3. Use namespaces to isolate per-user data

**Example:**

```rust
// After conversation
client.learn("User prefers dark mode and uses VS Code on macOS", None).await?;

// Before next conversation
let ctx = client.augment("What editor does the user prefer?", None).await?;
// ctx.context_text contains: "User prefers dark mode and uses VS Code on macOS"
```

**Best for:** AI assistants, personalized recommendations, user preference tracking.

### Pattern 4: Multi-Agent Memory Sharing

Use namespaces and the agent API to share memory between agents.

```
┌──────────┐     ┌──────────┐     ┌──────────┐
│ Agent A   │     │ Agent B   │     │ Agent C   │
│ ns: "a"   │     │ ns: "b"   │     │ ns: "c"   │
└─────┬─────┘     └─────┬─────┘     └─────┬─────┘
      │                 │                 │
      └────────┬────────┘                 │
               ▼                          │
         ┌──────────┐                     │
         │ Shared   │◀────────────────────┘
         │ ns:"shared"│
         └──────────┘
```

Each agent writes to its own namespace and reads from a shared namespace.

---

## Anti-Patterns

### 1. Storing Raw Conversation Logs

**Wrong:** Ingesting entire chat transcripts verbatim.

```bash
# Don't do this
curl -X POST /api/v1/memories \
  -d '{"text": "User: Hi\nAssistant: Hello!\nUser: What is 2+2?\nAssistant: 4"}'
```

**Right:** Extract and store meaningful facts.

```bash
# Do this instead
curl -X POST /api/v1/learn \
  -d '{"output": "The user asked a basic math question. No preferences or facts were revealed."}'
```

**Why:** Raw logs create noise, inflate the graph, and degrade search quality. The `learn` endpoint extracts only meaningful information.

### 2. Ignoring Namespaces

**Wrong:** Storing all users' data in the default namespace.

**Right:** Use one namespace per user or per context.

```bash
curl -H "X-Ucotron-Namespace: user_123" ...
```

**Why:** Without namespaces, search returns mixed results from all users. Namespace isolation is essential for privacy and relevance.

### 3. Overly Frequent Searches

**Wrong:** Calling `ucotron_search` on every token of a streaming response.

**Right:** Call `ucotron_augment` once at the start of generation with the full user query.

**Why:** Each search involves vector computation, graph traversal, and re-ranking. Batch your context retrieval.

### 4. Storing Ephemeral Information

**Wrong:** Storing temporary states like "user is currently typing" or "session started at 3pm".

**Right:** Only store persistent facts, preferences, and knowledge.

**Why:** Ephemeral data pollutes the knowledge graph and wastes storage. Ucotron is designed for long-term memory, not session state.

### 5. Skipping Metadata

**Wrong:** Storing memories without metadata context.

```json
{"text": "Prefers Python"}
```

**Right:** Include source, timestamp, and context metadata.

```json
{
  "text": "User prefers Python for data science projects",
  "metadata": {
    "source": "onboarding_conversation",
    "confidence": "high"
  }
}
```

**Why:** Metadata enables filtering, debugging, and confidence-based conflict resolution.

### 6. Using Ucotron as a Database

**Wrong:** Storing structured records and expecting exact-match queries.

**Right:** Use Ucotron for semantic memory — fuzzy recall, context augmentation, and knowledge graphs.

**Why:** Ucotron uses embedding-based search and graph traversal. For exact lookups, use a traditional database alongside Ucotron.

---

## Best Practices: Memory Ingestion

### Chunk Size Recommendations

Ucotron uses sentence-level chunking by default — text is split on `.`, `!`, or `?` boundaries. For optimal results:

| Content Type | Recommendation | Why |
|-------------|---------------|-----|
| Conversational text | Use as-is (sentence chunking) | Natural utterance boundaries preserve meaning |
| Long documents | Pre-chunk to 200-500 tokens before calling `/memories` | Prevents oversized chunks from diluting embedding quality |
| Short facts | Send individual facts as separate calls | Each fact gets its own embedding and entity links |
| Structured data | Convert to natural language sentences first | "Alice: Engineer" → "Alice is an engineer" |

**Pre-chunking example for long documents:**

```python
# Split a long document into ~300-token paragraphs before ingestion
import re

def pre_chunk(text: str, max_tokens: int = 300) -> list[str]:
    """Split text into chunks of approximately max_tokens words."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks, current, count = [], [], 0
    for sentence in sentences:
        words = len(sentence.split())
        if count + words > max_tokens and current:
            chunks.append(' '.join(current))
            current, count = [], 0
        current.append(sentence)
        count += words
    if current:
        chunks.append(' '.join(current))
    return chunks

# Ingest each chunk separately
for chunk in pre_chunk(long_document):
    await ucotron.add_memory(chunk)
```

**What to avoid:**
- Chunks smaller than 20 tokens — too little context for meaningful embeddings
- Chunks larger than 1000 tokens — embedding quality degrades (MiniLM-L6-v2 has 256-token context window)
- Splitting mid-sentence — breaks entity co-occurrence patterns needed for relation extraction

### Metadata Best Practices

Always include metadata to enable filtering, debugging, and conflict resolution:

```json
{
  "text": "Alice joined the ML platform team in March 2025",
  "metadata": {
    "source": "meeting_notes",
    "timestamp": "2025-03-15T10:00:00Z",
    "confidence": "high",
    "session_id": "conv_abc123"
  }
}
```

| Metadata Field | Purpose | Recommended Values |
|---------------|---------|-------------------|
| `source` | Origin tracking, debugging | `"meeting_notes"`, `"email"`, `"slack"`, `"user_input"` |
| `timestamp` | Temporal ordering, conflict resolution | ISO 8601 string or Unix epoch |
| `confidence` | Weight during contradiction resolution | `"high"`, `"medium"`, `"low"` or float 0.0-1.0 |
| `session_id` | Group memories from same conversation | Unique session identifier |
| `author` | Attribution for multi-user systems | User ID or name |

**Automatic metadata added by Ucotron:**
- `chunk_index` — position within the chunked text (0-based)
- `source_type` — always `"ingestion"` for API-ingested content
- `_namespace` — namespace tag from `X-Ucotron-Namespace` header

**Tips:**
- Use `source` consistently across your application — it makes debugging retrieval issues much easier
- Include `timestamp` when ingesting historical data; Ucotron uses it for temporal conflict resolution (newer facts with >1 year gap supersede older ones)
- Don't store sensitive data in metadata (API keys, passwords) — metadata is returned in search results

### Namespace Isolation Strategies

Namespaces provide data isolation at the storage level. Choose a strategy based on your use case:

| Strategy | Namespace Pattern | Best For |
|----------|------------------|----------|
| **Per-user** | `user_{user_id}` | Personal assistants, per-user memory |
| **Per-agent** | `agent_{agent_name}` | Multi-agent systems with independent memories |
| **Per-project** | `project_{project_id}` | Knowledge bases scoped to projects |
| **Per-environment** | `dev`, `staging`, `prod` | Separate test and production data |
| **Hierarchical** | `org_{id}_team_{id}` | Enterprise multi-tenant setups |

**Configuration:**

```toml
[namespaces]
default_namespace = "default"
allowed_namespaces = ["prod", "staging"]  # Empty = allow any
max_namespaces = 0                        # 0 = unlimited
```

**Implementation patterns:**

```bash
# Per-user isolation
curl -X POST http://localhost:8420/api/v1/memories \
  -H "X-Ucotron-Namespace: user_alice" \
  -d '{"text": "I prefer dark mode"}'

# Shared knowledge base (all agents read from this)
curl -X POST http://localhost:8420/api/v1/memories \
  -H "X-Ucotron-Namespace: shared_kb" \
  -d '{"text": "Company policy: all PRs require 2 approvals"}'
```

**Key behaviors:**
- If no `X-Ucotron-Namespace` header is provided, data goes to the `"default"` namespace
- Search results are automatically filtered by namespace — no cross-namespace leakage
- Pre-existing nodes without a namespace tag are treated as belonging to `"default"`
- Use RBAC namespace scoping to restrict which API keys can access which namespaces

### Entity Resolution Tuning

Entity resolution deduplicates entities using a combined similarity score:

```
similarity = 0.6 × Jaccard(neighbor_sets) + 0.4 × cosine(embeddings)
```

The default merge threshold is **0.5**. Tune it based on your data:

| Threshold | Effect | When to Use |
|-----------|--------|-------------|
| **0.3** (loose) | Aggressively merges similar entities | High-noise data with many spelling variants |
| **0.5** (default) | Balanced dedup | General-purpose usage |
| **0.7** (strict) | Only merges near-identical entities | When false merges are costly (medical, legal) |
| **0.9** (very strict) | Effectively disabled for non-exact matches | When you want minimal automatic merging |

**Configuration:**

```toml
[extraction]
entity_resolution_threshold = 0.5  # Adjust per use case
```

```bash
# Environment variable override
UCOTRON_EXTRACTION_ENTITY_RESOLUTION_THRESHOLD=0.7
```

**Pipeline toggles for performance tuning:**

| Toggle | Default | Effect When Disabled |
|--------|---------|---------------------|
| `enable_relations` | `true` | Skips relation extraction — faster ingestion, no typed edges |
| `enable_entity_resolution` | `true` | Skips dedup — faster, but creates duplicate entities |
| `enable_contradiction_detection` | `true` | Skips conflict checks — faster, but misses contradictions |

**NER label customization:**

By default, Ucotron extracts 5 entity types: `person`, `location`, `organization`, `date`, `concept`. Narrow this for domain-specific ingestion:

```toml
[extraction]
ner_labels = ["person", "organization"]  # Only extract people and orgs
```

Fewer labels = faster NER inference and fewer false-positive entities.

### Ingestion Performance Tips

1. **Batch your ingestion** — Send multiple related facts in a single API call rather than one-at-a-time
2. **Use `/learn` for agent output** — The learn endpoint is optimized for extracting facts from conversational text
3. **Disable unused pipeline steps** — If you don't need contradiction detection, set `enable_contradiction_detection = false` for ~15% faster ingestion
4. **Monitor pipeline metrics** — The ingestion response includes per-step timing (`embedding_ms`, `ner_ms`, etc.) to identify bottlenecks
5. **Tune HNSW parameters** — Lower `ef_construction` (e.g., 100) for faster indexing at the cost of recall quality; the default 200 is a good balance
6. **Adjust consolidation frequency** — The default `trigger_interval = 100` runs Leiden community detection every 100 chunks; increase to 500+ for write-heavy workloads

---

## Best Practices: Memory Retrieval

### vector_top_k Tuning

The `vector_top_k` parameter controls the initial vector search pool size before graph expansion and re-ranking. The default is **50**.

| Scenario | Recommended vector_top_k | Rationale |
|----------|--------------------------|-----------|
| Quick lookup (chatbot) | 20-30 | Faster response, sufficient for focused queries |
| General-purpose | 50 (default) | Good balance of recall and speed |
| Comprehensive recall | 100-200 | When missing a relevant memory is costly (knowledge bases, QA) |
| Precision-focused | 20 + `min_similarity: 0.5` | Filters weak matches early; pair with low top_k |

**Key insight:** `vector_top_k` is the *candidate pool*, not the final result count. The pipeline narrows candidates through graph expansion, community selection, and re-ranking before returning `final_top_k` (default: 10) results.

```json
// Search with custom limit (maps to final_top_k)
{
  "query": "What happened at the conference?",
  "limit": 5
}
```

**Tuning tips:**
- If search misses relevant memories, increase `vector_top_k` first — the re-ranker may be filtering them out too early
- If search is slow, decrease `vector_top_k` — each candidate triggers graph expansion
- For embedding model all-MiniLM-L6-v2, similarity above 0.7 indicates strong semantic match; below 0.4 is noise
- The `min_similarity` threshold (default: 0.0) can pre-filter weak candidates: set to 0.3-0.5 for cleaner results

### Community Expansion Usage

Community expansion leverages Leiden clustering to include topically related memories that may not match the query vector directly. Controlled by `enable_community_expansion` (default: `true`) and `max_community_members` (default: `20`).

**How it works:**
1. For each seed node from vector search, Ucotron looks up its Leiden community assignment
2. All nodes in the same community are candidates for inclusion
3. Up to `max_community_members` are added per seed, with a base similarity of 0.1
4. The re-ranker scores these community members alongside direct matches

**When to enable (default):**
- Conversational agents — adds thematic context beyond literal matches
- Knowledge bases with rich entity relationships — finds related facts in the same topic cluster
- Exploratory search — surfaces serendipitous connections

**When to disable:**
- Precision QA — community members may dilute relevant results
- Simple fact lookup — direct vector match is sufficient
- High-throughput scenarios — saves community lookup latency

**Tuning `max_community_members`:**

| Setting | Effect |
|---------|--------|
| 5-10 | Minimal community context, fast |
| 20 (default) | Balanced thematic enrichment |
| 50+ | Broad context, may include irrelevant members |

**Example: Disable community expansion for precision search**

```python
# Using the Python SDK with custom retrieval
results = await m.search(
    "What is Alice's job title?",
    SearchOptions(limit=3),
)
# For precision, a smaller limit with vector-dominant results works best.
# Community expansion is controlled server-side via RetrievalConfig defaults.
```

**Consolidation frequency affects community quality.** Communities are recalculated by the consolidation worker (default: every 100 ingested chunks). If you ingest many memories quickly, communities may be stale until the next consolidation runs. Increase `trigger_interval` for write-heavy workloads or trigger manually.

### Temporal Filter Strategies

Ucotron supports two complementary time-based mechanisms: **temporal decay** (soft weighting) and **time_range filters** (hard cutoffs).

#### Temporal Decay

Temporal decay downweights older memories using exponential decay with a configurable half-life:

```
recency_score = 0.5 ^ (age_seconds / half_life_seconds)
```

The default half-life is **30 days** (2,592,000 seconds). This means:
- 0 days old → recency = 1.0
- 30 days old → recency = 0.5
- 60 days old → recency = 0.25
- 90 days old → recency = 0.125
- 1 year old → recency ≈ 0.001

**Tuning by use case:**

| Use Case | Half-Life | Why |
|----------|-----------|-----|
| Chat assistant | 7 days | Recent context matters most |
| Meeting notes | 30 days (default) | Monthly relevance window |
| Knowledge base | 365 days | Facts age slowly |
| Historical archive | 3650 days (10 years) | Near-zero decay |

**The recency score combines with other signals:**

```
final_score = vector_sim × 0.5 + graph_centrality × 0.3 + recency × 0.2
```

With the default weight of 0.2, recency influences but doesn't dominate scoring. Increase `recency_weight` to 0.4 for time-sensitive domains.

#### Time Range Filters (Hard Cutoffs)

Use `time_range` to restrict results to a specific window:

```bash
# Only memories from January 2025
curl -X POST http://localhost:8420/api/v1/memories/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "project updates",
    "time_range": {
      "start": 1735689600,
      "end": 1738368000
    }
  }'
```

**Common patterns:**

| Pattern | Implementation |
|---------|---------------|
| Last 24 hours | `start: now - 86400` |
| Last 7 days | `start: now - 604800` |
| Last 30 days | `start: now - 2592000` |
| Specific month | `start: month_start, end: month_end` |
| Before a date | `end: cutoff_timestamp` |

**Combining decay and filters:**
- Use `time_range` to exclude irrelevant time periods entirely
- Use temporal decay within the range to prefer newer memories
- Example: For a "what happened this quarter?" query, set `time_range` to the quarter boundaries and let decay rank within it

### Re-ranking Customization

The re-ranking step combines three scoring signals into a final score:

```
final_score = vector_sim × vector_sim_weight
            + graph_centrality × graph_centrality_weight
            + recency × recency_weight
```

**Default weights:** `0.5 / 0.3 / 0.2` (vector / graph / recency)

**Score components explained:**

| Component | Range | Source | What It Measures |
|-----------|-------|--------|-----------------|
| `vector_sim` | [0, 1] | HNSW/SIMD cosine similarity | Semantic closeness to query |
| `graph_centrality` | [0, 1] | degree / max_degree | How well-connected the node is |
| `recency` | [0, 1] | Exponential decay from timestamp | How recently the memory was created |

**Tuning recipes:**

| Use Case | vector_sim | graph_centrality | recency | Rationale |
|----------|------------|-----------------|---------|-----------|
| General-purpose | 0.5 | 0.3 | 0.2 | Default balanced mix |
| Semantic search | 0.7 | 0.2 | 0.1 | Prioritize meaning match |
| Entity hub finding | 0.2 | 0.6 | 0.2 | Surface well-connected entities |
| Recent context | 0.3 | 0.1 | 0.6 | Favor latest memories |
| Time-insensitive KB | 0.5 | 0.5 | 0.0 | Ignore age entirely |

**How graph centrality helps:**
- Hub entities (entities with many connections) score higher
- Useful for finding "important" facts about central entities
- If an entity is mentioned across many memories, its centrality will be high

**Graph expansion hops:**

The `graph_expansion_hops` parameter (default: 1) controls how far the traversal goes from seed nodes:

| Hops | Behavior | Best For |
|------|----------|----------|
| 0 | Vector-only, no graph traversal | Speed-critical, simple lookups |
| 1 (default) | Direct neighbors included | Balanced context enrichment |
| 2 | Friends-of-friends included | Deep exploration, when graph is sparse |
| 3+ | Very broad expansion | Rarely useful, high noise risk |

Neighbors receive decayed similarity scores: `similarity × 0.5^hops`. A 2-hop neighbor gets 25% of the original similarity score.

**Filtering by node type:**

Restrict results to specific memory types for focused retrieval:

```bash
# Only events
curl -X POST http://localhost:8420/api/v1/memories/search \
  -d '{"query": "team meetings", "node_type": "Event"}'

# Only entities
curl -X POST http://localhost:8420/api/v1/memories/search \
  -d '{"query": "people in engineering", "node_type": "Entity"}'
```

Available types: `Entity`, `Event`, `Fact`, `Skill`.

### Retrieval Performance Tips

1. **Start with defaults** — The default configuration (`vector_top_k: 50`, `graph_expansion_hops: 1`, `community_expansion: true`) works well for most use cases. Only tune when search quality or speed is inadequate.

2. **Use `limit` for fast responses** — A `limit` of 3-5 is sufficient for most chatbot augmentation. Larger limits only help when you need comprehensive recall.

3. **Filter early with `node_type` and `time_range`** — Hard filters reduce the candidate pool before expensive re-ranking. Use them when you know the expected result type.

4. **Monitor retrieval metrics** — Search responses include per-step timing. If `graph_expansion_us` dominates, reduce hops. If `community_selection_us` is high, reduce `max_community_members` or disable community expansion.

5. **Namespace isolation boosts relevance** — Search within a specific namespace to avoid cross-user noise. Always set `X-Ucotron-Namespace` in multi-tenant deployments.

6. **Augment once per turn** — Call `/augment` once at the start of each LLM generation with the full user query. Avoid multiple search calls per turn — batch your context retrieval.

7. **Interpret score breakdowns** — The `vector_sim`, `graph_centrality`, and `recency` fields in results help debug why a memory ranked high or low. If graph_centrality is inflating irrelevant hub nodes, reduce its weight.

---

## Configuration

### Server Configuration (TOML)

Generate a default config:

```bash
ucotron_server --init-config > ucotron.toml
```

Key settings:

```toml
[server]
host = "0.0.0.0"
port = 8420

[storage]
mode = "embedded"          # "embedded" or "external"
path = "./ucotron-data"    # LMDB data directory

[storage.hnsw]
ef_construction = 200      # Index build quality (higher = better, slower)
m = 16                     # Max connections per node
ef_search = 100            # Search quality (higher = better, slower)

[extraction]
embedding_model = "all-MiniLM-L6-v2"
ner_model = "gliner_small-v2.1"
chunk_size = 500           # Tokens per chunk

[auth]
enabled = false            # Enable API key auth
admin_key = ""             # Admin API key

[namespaces]
allowed_namespaces = []    # Empty = allow all
```

### Environment Variables

All settings can be overridden via `UCOTRON_` prefixed env vars:

```bash
UCOTRON_SERVER_PORT=9000
UCOTRON_STORAGE_PATH=/data/ucotron
UCOTRON_AUTH_ENABLED=true
UCOTRON_AUTH_ADMIN_KEY=your-secret-key
```

---

## Authentication & RBAC

When auth is enabled, include your API key in requests:

```bash
curl -H "Authorization: Bearer mk_your_api_key" \
     http://localhost:8420/api/v1/memories/search \
     -d '{"query": "test"}'
```

### Roles

| Role | Permissions |
|------|-------------|
| **Admin** | Full access: CRUD, auth management, audit, config |
| **Writer** | Read + write: create/update/delete memories, learn, search |
| **Reader** | Read-only: search, augment, list, get |
| **Viewer** | Minimal: health check, metrics only |

### API Key Management

```bash
# Create a key (admin only)
curl -X POST http://localhost:8420/api/v1/auth/keys \
  -H "Authorization: Bearer mk_admin_key" \
  -d '{"role": "writer", "name": "my-agent"}'

# List keys
curl http://localhost:8420/api/v1/auth/keys \
  -H "Authorization: Bearer mk_admin_key"

# Revoke a key
curl -X DELETE http://localhost:8420/api/v1/auth/keys/key_id \
  -H "Authorization: Bearer mk_admin_key"
```

---

## SDK Examples

Copy-paste examples for all 6 supported languages. Each example shows installation, configuration, and basic usage (store + search + augment).

### Rust

**Install:**

```bash
cargo add ucotron_sdk
# Or add to Cargo.toml:
# [dependencies]
# ucotron_sdk = "0.1"
```

**Basic Usage (async):**

```rust
use ucotron_sdk::{UcotronClient, AddMemoryOptions, SearchOptions, AugmentOptions};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = UcotronClient::new("http://localhost:8420");

    // Store a memory
    let result = client.add_memory(
        "Alice is a senior engineer at Acme Corp in Madrid",
        AddMemoryOptions { namespace: Some("demo".into()), ..Default::default() },
    ).await?;
    println!("Created {} memories", result.chunk_node_ids.len());

    // Learn from agent output
    let learned = client.learn(
        "User prefers dark mode and uses VS Code on macOS",
        Default::default(),
    ).await?;
    println!("Learned {} facts, {} entities", learned.memories_created, learned.entities_found);

    // Semantic search
    let results = client.search(
        "Who works at Acme?",
        SearchOptions { limit: Some(5), ..Default::default() },
    ).await?;
    for item in &results.results {
        println!("[{:.2}] {}", item.score, item.content);
    }

    // Augment context for LLM
    let ctx = client.augment(
        "Tell me about Alice",
        AugmentOptions { limit: Some(10), ..Default::default() },
    ).await?;
    println!("Context:\n{}", ctx.context_text);

    Ok(())
}
```

**Sync Usage:**

```rust
use ucotron_sdk::UcotronClient;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = UcotronClient::new("http://localhost:8420");
    let results = client.search_sync("What does Alice do?", Default::default())?;
    for item in &results.results {
        println!("{}: {}", item.id, item.content);
    }
    Ok(())
}
```

**Custom Config (retry + timeout):**

```rust
use ucotron_sdk::{UcotronClient, ClientConfig, RetryConfig};
use std::time::Duration;

let client = UcotronClient::with_config(
    "http://localhost:8420",
    ClientConfig {
        timeout: Duration::from_secs(60),
        retry: RetryConfig {
            max_retries: 5,
            base_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(5),
        },
        default_namespace: Some("my-agent".into()),
    },
);
```

**Axum Integration:**

```rust
use axum::{Router, Json, extract::State};
use ucotron_sdk::UcotronClient;
use std::sync::Arc;

#[derive(Clone)]
struct AppState {
    ucotron: Arc<UcotronClient>,
}

async fn chat(
    State(state): State<AppState>,
    Json(body): Json<serde_json::Value>,
) -> Json<serde_json::Value> {
    let query = body["message"].as_str().unwrap_or("");
    let ctx = state.ucotron.augment(query, Default::default()).await.unwrap();
    // Use ctx.context_text in your LLM prompt
    Json(serde_json::json!({ "context": ctx.context_text }))
}

let state = AppState { ucotron: Arc::new(UcotronClient::new("http://localhost:8420")) };
let app = Router::new().route("/chat", axum::routing::post(chat)).with_state(state);
```

---

### TypeScript

**Install:**

```bash
npm install ucotron-sdk
# or
yarn add ucotron-sdk
# or
pnpm add ucotron-sdk
```

**Basic Usage:**

```typescript
import { Ucotron } from "ucotron-sdk";

const ucotron = new Ucotron("http://localhost:8420", {
  defaultNamespace: "demo",
});

// Store a memory
const result = await ucotron.addMemory(
  "Alice is a senior engineer at Acme Corp in Madrid"
);
console.log(`Created ${result.chunkNodeIds.length} memories`);

// Learn from agent output
const learned = await ucotron.learn(
  "User prefers dark mode and uses VS Code on macOS"
);
console.log(`Learned ${learned.memoriesCreated} facts`);

// Semantic search
const results = await ucotron.search("Who works at Acme?", { limit: 5 });
for (const item of results.results) {
  console.log(`[${item.score.toFixed(2)}] ${item.content}`);
}

// Augment context for LLM
const ctx = await ucotron.augment("Tell me about Alice", { limit: 10 });
console.log("Context:", ctx.context_text);
```

**Next.js App Router Integration:**

```typescript
// app/api/chat/route.ts
import { Ucotron } from "ucotron-sdk";
import { NextRequest, NextResponse } from "next/server";

const ucotron = new Ucotron(process.env.UCOTRON_URL!);

export async function POST(req: NextRequest) {
  const { message, userId } = await req.json();

  // Retrieve context
  const ctx = await ucotron.augment(message, {
    namespace: userId,
    limit: 10,
  });

  // Pass ctx.context_text to your LLM...
  const llmResponse = await generateWithContext(message, ctx.context_text);

  // Learn from the response
  await ucotron.learn(llmResponse, { namespace: userId });

  return NextResponse.json({ reply: llmResponse });
}
```

**Vercel AI SDK Integration:**

```typescript
import { Ucotron } from "ucotron-sdk";
import { streamText } from "ai";
import { openai } from "@ai-sdk/openai";

const ucotron = new Ucotron("http://localhost:8420");

async function chat(userMessage: string) {
  const ctx = await ucotron.augment(userMessage);

  const result = await streamText({
    model: openai("gpt-4o"),
    system: `You have persistent memory.\n\n${ctx.context_text}`,
    messages: [{ role: "user", content: userMessage }],
  });

  return result;
}
```

---

### Python

**Install:**

```bash
pip install ucotron-sdk
# or
poetry add ucotron-sdk
```

**Async Usage:**

```python
import asyncio
from ucotron_sdk import Ucotron, AugmentOptions, SearchOptions

async def main():
    async with Ucotron("http://localhost:8420") as m:
        # Store a memory
        result = await m.add_memory(
            "Alice is a senior engineer at Acme Corp in Madrid"
        )
        print(f"Created {len(result.chunk_node_ids)} memories")

        # Learn from agent output
        learned = await m.learn(
            "User prefers dark mode and uses VS Code on macOS"
        )
        print(f"Learned {learned.memories_created} facts")

        # Semantic search
        results = await m.search(
            "Who works at Acme?",
            SearchOptions(limit=5),
        )
        for item in results.results:
            print(f"[{item.score:.2f}] {item.content}")

        # Augment context for LLM
        ctx = await m.augment(
            "Tell me about Alice",
            AugmentOptions(limit=10),
        )
        print("Context:", ctx.context_text)

asyncio.run(main())
```

**Sync Usage:**

```python
from ucotron_sdk import UcotronSync

with UcotronSync("http://localhost:8420") as m:
    results = m.search("What does Alice do?")
    for item in results.results:
        print(f"{item.id}: {item.content}")
```

**LangChain Integration:**

```python
from langchain.memory import BaseMemory
from ucotron_sdk import UcotronSync

class UcotronMemory(BaseMemory):
    """LangChain-compatible memory backed by Ucotron."""

    def __init__(self, url: str, namespace: str = "default"):
        self.client = UcotronSync(url)
        self.namespace = namespace

    @property
    def memory_variables(self) -> list[str]:
        return ["ucotron_context"]

    def load_memory_variables(self, inputs: dict) -> dict:
        query = inputs.get("input", "")
        ctx = self.client.augment(query)
        return {"ucotron_context": ctx.context_text}

    def save_context(self, inputs: dict, outputs: dict) -> None:
        self.client.learn(outputs.get("output", ""))

    def clear(self) -> None:
        pass
```

**LlamaIndex Integration:**

```python
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, TextNode, QueryBundle
from ucotron_sdk import UcotronSync

class UcotronRetriever(BaseRetriever):
    """LlamaIndex retriever backed by Ucotron semantic search."""

    def __init__(self, url: str, top_k: int = 5):
        super().__init__()
        self.client = UcotronSync(url)
        self.top_k = top_k

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        results = self.client.search(query_bundle.query_str, limit=self.top_k)
        return [
            NodeWithScore(
                node=TextNode(text=r.content, id_=str(r.id)),
                score=r.score,
            )
            for r in results.results
        ]
```

---

### Go

**Install:**

```bash
go get github.com/ucotron-ai/ucotron-go@latest
```

**Basic Usage:**

```go
package main

import (
    "context"
    "fmt"
    "log"

    ucotron "github.com/ucotron-ai/ucotron-go"
)

func main() {
    client := ucotron.NewClient("http://localhost:8420", nil)
    defer client.Close()
    ctx := context.Background()

    // Store a memory
    result, err := client.AddMemory(ctx,
        "Alice is a senior engineer at Acme Corp in Madrid",
        &ucotron.AddMemoryOptions{Namespace: "demo"},
    )
    if err != nil {
        log.Fatal(err)
    }
    fmt.Printf("Created %d memories\n", len(result.ChunkNodeIDs))

    // Learn from agent output
    learned, err := client.Learn(ctx,
        "User prefers dark mode and uses VS Code on macOS",
        nil,
    )
    if err != nil {
        log.Fatal(err)
    }
    fmt.Printf("Learned %d facts, %d entities\n",
        learned.MemoriesCreated, learned.EntitiesFound)

    // Semantic search
    limit := 5
    results, err := client.Search(ctx,
        "Who works at Acme?",
        &ucotron.SearchOptions{Limit: &limit},
    )
    if err != nil {
        log.Fatal(err)
    }
    for _, item := range results.Results {
        fmt.Printf("[%.2f] %s\n", item.Score, item.Content)
    }

    // Augment context for LLM
    augLimit := 10
    augCtx, err := client.Augment(ctx,
        "Tell me about Alice",
        &ucotron.AugmentOptions{Limit: &augLimit},
    )
    if err != nil {
        log.Fatal(err)
    }
    fmt.Println("Context:", augCtx.ContextText)
}
```

**HTTP Handler Integration:**

```go
package main

import (
    "encoding/json"
    "net/http"

    ucotron "github.com/ucotron-ai/ucotron-go"
)

func chatHandler(client *ucotron.Client) http.HandlerFunc {
    return func(w http.ResponseWriter, r *http.Request) {
        var body struct {
            Message string `json:"message"`
            UserID  string `json:"user_id"`
        }
        json.NewDecoder(r.Body).Decode(&body)

        // Retrieve context
        limit := 10
        ctx, _ := client.Augment(r.Context(), body.Message,
            &ucotron.AugmentOptions{
                Limit:     &limit,
                Namespace: body.UserID,
            },
        )

        // Use ctx.ContextText in your LLM prompt...
        json.NewEncoder(w).Encode(map[string]string{
            "context": ctx.ContextText,
        })
    }
}

func main() {
    client := ucotron.NewClient("http://localhost:8420", nil)
    defer client.Close()

    http.HandleFunc("/chat", chatHandler(client))
    http.ListenAndServe(":3000", nil)
}
```

---

### Java

**Install (Gradle):**

```groovy
dependencies {
    implementation 'com.ucotron:ucotron-sdk:0.1.0'
}
```

**Install (Maven):**

```xml
<dependency>
    <groupId>com.ucotron</groupId>
    <artifactId>ucotron-sdk</artifactId>
    <version>0.1.0</version>
</dependency>
```

**Basic Usage:**

```java
import com.ucotron.sdk.UcotronClient;
import com.ucotron.sdk.model.*;

public class UcotronExample {
    public static void main(String[] args) throws Exception {
        UcotronClient client = new UcotronClient("http://localhost:8420");

        // Store a memory
        CreateMemoryResult result = client.addMemory(
            "Alice is a senior engineer at Acme Corp in Madrid"
        );
        System.out.printf("Created %d memories%n", result.chunkNodeIds.size());

        // Learn from agent output
        LearnResult learned = client.learn(
            "User prefers dark mode and uses VS Code on macOS"
        );
        System.out.printf("Learned %d facts, %d entities%n",
            learned.memoriesCreated, learned.entitiesFound);

        // Semantic search
        SearchResult results = client.search("Who works at Acme?", 5, null, null);
        for (SearchResultItem item : results.results) {
            System.out.printf("[%.2f] %s%n", item.score, item.content);
        }

        // Augment context for LLM
        AugmentResult ctx = client.augment("Tell me about Alice", 10, null);
        System.out.println("Context: " + ctx.contextText);
    }
}
```

**Spring Boot Integration:**

```java
import com.ucotron.sdk.UcotronClient;
import com.ucotron.sdk.ClientConfig;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.bind.annotation.*;

@Configuration
class UcotronConfig {
    @Bean
    UcotronClient ucotronClient() {
        return new UcotronClient(
            "http://localhost:8420",
            ClientConfig.builder()
                .defaultNamespace("spring-app")
                .build()
        );
    }
}

@RestController
@RequestMapping("/api/chat")
class ChatController {
    private final UcotronClient ucotron;

    ChatController(UcotronClient ucotron) {
        this.ucotron = ucotron;
    }

    @PostMapping
    Map<String, String> chat(@RequestBody Map<String, String> body) throws Exception {
        String message = body.get("message");

        // Retrieve context
        AugmentResult ctx = ucotron.augment(message, 10, null);

        // Use ctx.contextText in your LLM prompt...
        return Map.of("context", ctx.contextText);
    }
}
```

---

### PHP

**Install:**

```bash
composer require ucotron/sdk
```

**Basic Usage:**

```php
<?php
require_once 'vendor/autoload.php';

use Ucotron\Sdk\UcotronClient;

$client = new UcotronClient('http://localhost:8420');

// Store a memory
$result = $client->addMemory(
    'Alice is a senior engineer at Acme Corp in Madrid'
);
echo "Created " . count($result->chunkNodeIds) . " memories\n";

// Learn from agent output
$learned = $client->learn(
    'User prefers dark mode and uses VS Code on macOS'
);
echo "Learned {$learned->memoriesCreated} facts, {$learned->entitiesFound} entities\n";

// Semantic search
$results = $client->search('Who works at Acme?', topK: 5);
foreach ($results->results as $item) {
    printf("[%.2f] %s\n", $item->score, $item->content);
}

// Augment context for LLM
$ctx = $client->augment('Tell me about Alice', topK: 10);
echo "Context: " . $ctx->contextText . "\n";
```

**Laravel Integration:**

```php
<?php
// config/ucotron.php
return [
    'url' => env('UCOTRON_URL', 'http://localhost:8420'),
    'namespace' => env('UCOTRON_NAMESPACE', 'laravel-app'),
];

// app/Providers/UcotronServiceProvider.php
namespace App\Providers;

use Illuminate\Support\ServiceProvider;
use Ucotron\Sdk\UcotronClient;
use Ucotron\Sdk\ClientConfig;

class UcotronServiceProvider extends ServiceProvider
{
    public function register(): void
    {
        $this->app->singleton(UcotronClient::class, function () {
            return new UcotronClient(
                config('ucotron.url'),
                new ClientConfig(namespace: config('ucotron.namespace'))
            );
        });
    }
}

// app/Http/Controllers/ChatController.php
namespace App\Http\Controllers;

use Illuminate\Http\Request;
use Ucotron\Sdk\UcotronClient;

class ChatController extends Controller
{
    public function chat(Request $request, UcotronClient $ucotron)
    {
        $message = $request->input('message');

        // Retrieve context
        $ctx = $ucotron->augment($message, topK: 10);

        // Use $ctx->contextText in your LLM prompt...
        return response()->json(['context' => $ctx->contextText]);
    }
}
```

**Symfony Integration:**

```php
<?php
// config/services.yaml
// services:
//   Ucotron\Sdk\UcotronClient:
//     arguments:
//       $baseUrl: '%env(UCOTRON_URL)%'

// src/Controller/ChatController.php
namespace App\Controller;

use Ucotron\Sdk\UcotronClient;
use Symfony\Bundle\FrameworkBundle\Controller\AbstractController;
use Symfony\Component\HttpFoundation\JsonResponse;
use Symfony\Component\HttpFoundation\Request;
use Symfony\Component\Routing\Attribute\Route;

class ChatController extends AbstractController
{
    #[Route('/api/chat', methods: ['POST'])]
    public function chat(Request $request, UcotronClient $ucotron): JsonResponse
    {
        $message = $request->toArray()['message'];
        $ctx = $ucotron->augment($message, topK: 10);
        return $this->json(['context' => $ctx->contextText]);
    }
}
```

---

## GDPR & Data Management

### Right to be Forgotten

```bash
curl -X POST http://localhost:8420/api/v1/gdpr/forget \
  -H "Authorization: Bearer mk_admin_key" \
  -d '{"entity_name": "Alice"}'
```

Removes all memories, entities, and edges associated with the specified entity.

### Data Export

```bash
curl http://localhost:8420/api/v1/gdpr/export \
  -H "X-Ucotron-Namespace: user_123" \
  -H "Authorization: Bearer mk_admin_key" \
  > user_data.json
```

### Retention Policies

```bash
# Check retention status
curl http://localhost:8420/api/v1/gdpr/retention/status

# Manual cleanup sweep
curl -X POST http://localhost:8420/api/v1/gdpr/retention/sweep
```

---

## Health & Monitoring

```bash
# Health check
curl http://localhost:8420/api/v1/health
# → { "status": "ok", "components": { "lmdb": "ok", "embedding": "ok" } }

# Prometheus metrics
curl http://localhost:8420/api/v1/metrics
# → { "requests_total": 1234, "uptime_seconds": 3600, ... }
```

The `/metrics` endpoint is compatible with Prometheus scraping. Use the provided Grafana dashboards in `deploy/grafana/` for visualization.

---

## Deployment

### Docker (Recommended)

```bash
# Single instance
docker compose up -d

# Multi-instance (1 writer + 2 readers)
docker compose -f docker-compose.multi.yml up -d
```

### From Source

```bash
cd memory_arena

# Download ONNX models
./scripts/download_models.sh

# Build and run
cargo run --release --bin ucotron_server -- -c ucotron.toml
```

### Kubernetes

Use the Helm charts in `deploy/helm/`:

```bash
helm install ucotron deploy/helm/ucotron \
  --set storage.size=10Gi \
  --set auth.adminKey=your-secret
```

---

## Troubleshooting

### 1. Server Fails to Start — "Failed to load embedding model"

**Symptom:** The server logs `Failed to load embedding model: ... Using stub embedder.` and all searches return zero-similarity results.

**Cause:** ONNX model files are missing from the `models/` directory. These are gitignored and must be downloaded separately.

**Fix:**

```bash
cd memory_arena
./scripts/download_models.sh
```

This downloads `all-MiniLM-L6-v2` (embeddings) and `gliner_small-v2.1` (NER) to `models/`. Verify the models exist:

```bash
ls models/all-MiniLM-L6-v2/model.onnx
ls models/gliner_small-v2.1/model.onnx
```

If the download script fails, manually download the ONNX files from HuggingFace and place them in the corresponding directories.

### 2. Server Fails to Start — Configuration Validation Error

**Symptom:** The server exits immediately with a message like `server.port must be > 0` or `storage.mode must be one of: embedded, external, shared`.

**Cause:** Invalid values in `ucotron.toml` or environment variables.

**Fix:** Check each setting matches the expected format:

| Setting | Valid Values |
|---------|-------------|
| `server.port` | Any integer > 0 (default: 8420) |
| `server.log_level` | `trace`, `debug`, `info`, `warn`, `error` |
| `server.log_format` | `text`, `json` |
| `storage.mode` | `embedded`, `external`, `shared` |
| `storage.vector.backend` | `helix`, `qdrant`, `custom` |
| `storage.graph.backend` | `helix`, `falkordb`, `custom` |

Environment variables override TOML settings with the `UCOTRON_` prefix:

```bash
# Override port via env
UCOTRON_SERVER_PORT=9000 cargo run --release --bin ucotron_server
```

### 3. "Failed to open LMDB" — Permission or Disk Errors

**Symptom:** Server fails with `Failed to open LMDB at {path}: {error}` or `Failed to create data directory`.

**Causes:**
- The data directory doesn't exist and can't be created
- Insufficient file permissions
- Disk is full
- Another process has the LMDB environment locked

**Fix:**

```bash
# Ensure directory exists with correct permissions
mkdir -p ./ucotron-data
chmod 700 ./ucotron-data

# Check disk space
df -h .

# Check if another ucotron process holds the lock
lsof ./ucotron-data/data.mdb 2>/dev/null
```

LMDB uses a lock file (`lock.mdb`) that prevents concurrent writers. If a previous process crashed, the lock may be stale. Remove it only if you're certain no other process is accessing the data:

```bash
# Only if no other process is running!
rm ./ucotron-data/lock.mdb
```

### 4. Auth Enabled But No Keys Configured

**Symptom:** Server exits with `auth.enabled is true but no authentication method is configured`.

**Cause:** You set `auth.enabled = true` but didn't provide any API keys.

**Fix:** Add at least one API key to your config:

```toml
[auth]
enabled = true

[[auth.api_keys]]
name = "admin"
key = "mk_your_secret_admin_key"
role = "admin"

[[auth.api_keys]]
name = "my-agent"
key = "mk_your_agent_key"
role = "writer"
```

Or provide a single admin key:

```toml
[auth]
enabled = true
admin_key = "mk_your_admin_key"
```

Then include the key in requests:

```bash
curl -H "Authorization: Bearer mk_your_admin_key" \
  http://localhost:8420/api/v1/memories/search \
  -d '{"query": "test"}'
```

### 5. 401 Unauthorized / 403 Forbidden on API Calls

**Symptom:** API requests return `{"code":"UNAUTHORIZED","message":"Missing or invalid Authorization header"}` or `{"code":"FORBIDDEN","message":"Insufficient permissions"}`.

**Causes:**
- Missing `Authorization: Bearer <key>` header
- Invalid or revoked API key
- Key role doesn't have permission for the endpoint

**Fix:**

```bash
# Verify your key works
curl -H "Authorization: Bearer mk_your_key" \
  http://localhost:8420/api/v1/health

# Check role permissions
# admin: full access
# writer: read + write (create/update/delete memories, learn, search)
# reader: read-only (search, augment, list, get)
# viewer: health and metrics only
```

If you need a different role, create a new key with the admin key:

```bash
curl -X POST http://localhost:8420/api/v1/auth/keys \
  -H "Authorization: Bearer mk_admin_key" \
  -H "Content-Type: application/json" \
  -d '{"role": "writer", "name": "upgraded-agent"}'
```

### 6. 403 "Read-Only Instance" on Write Operations

**Symptom:** POST/PUT/DELETE requests return `{"code":"READ_ONLY_INSTANCE","message":"This instance is configured as read-only (role=reader)"}`.

**Cause:** You're sending write requests to a reader instance in a multi-instance deployment.

**Fix:** Send write operations to the writer instance:

```bash
# Writer instance (handles writes)
curl -X POST http://writer-host:8420/api/v1/memories \
  -d '{"text": "new memory"}'

# Reader instances (read-only)
curl -X POST http://reader-host:8421/api/v1/memories/search \
  -d '{"query": "search query"}'
```

Check your `docker-compose.multi.yml` to identify which instance is the writer.

### 7. Search Returns No Results or Low-Quality Results

**Symptom:** Searches return empty results or results with very low scores (< 0.3).

**Causes:**
- Stub embedder is active (models not downloaded) — all embeddings are zero vectors
- Namespace mismatch — searching a different namespace than where memories were stored
- The query is semantically unrelated to stored content

**Fix:**

1. Check embedding model status:

```bash
curl http://localhost:8420/api/v1/health
# Look for "embedding": "ok" in components
```

2. Verify namespace consistency:

```bash
# Store in namespace "demo"
curl -X POST http://localhost:8420/api/v1/memories \
  -H "X-Ucotron-Namespace: demo" \
  -d '{"text": "Alice works at Acme"}'

# Search same namespace
curl -X POST http://localhost:8420/api/v1/memories/search \
  -H "X-Ucotron-Namespace: demo" \
  -d '{"query": "Who works at Acme?"}'
```

3. If using the default namespace, omit the header on both store and search.

4. Try increasing `vector_top_k` (server-side config) or `limit` in search requests to widen the candidate pool.

### 8. MCP Server Not Connecting — "Server not found" in Claude Desktop

**Symptom:** Claude Desktop or Cursor shows "ucotron" as disconnected or tools don't appear.

**Causes:**
- Binary path is incorrect or not absolute
- Binary not built or not executable
- Models directory not configured
- Config JSON syntax error

**Fix:**

```bash
# 1. Build the MCP binary
cd memory_arena
cargo build --release --bin ucotron_mcp

# 2. Verify it exists and is executable
ls -la target/release/ucotron_mcp

# 3. Test it directly (should print JSON-RPC to stdout)
echo '{"jsonrpc":"2.0","method":"initialize","params":{"capabilities":{}},"id":1}' | \
  UCOTRON_STORAGE_PATH=/tmp/ucotron-test \
  UCOTRON_MODELS_DIR=./models \
  ./target/release/ucotron_mcp

# 4. Use ABSOLUTE paths in your MCP config
```

Make sure your `claude_desktop_config.json` uses absolute paths, not relative:

```json
{
  "mcpServers": {
    "ucotron": {
      "command": "/absolute/path/to/memory_arena/target/release/ucotron_mcp",
      "env": {
        "UCOTRON_STORAGE_PATH": "/absolute/path/to/ucotron-data",
        "UCOTRON_MODELS_DIR": "/absolute/path/to/memory_arena/models"
      }
    }
  }
}
```

After editing the config, restart Claude Desktop completely (quit and reopen, not just close the window).

### 9. Slow First Request After Server Start

**Symptom:** The first API call takes 1-3 seconds, but subsequent calls are fast (~10-50ms).

**Cause:** ONNX model initialization is lazy — the embedding and NER models are loaded into memory on first use, not at startup.

**This is expected behavior.** Subsequent requests reuse the loaded models and are fast.

**Mitigation:** If cold-start latency is a problem, send a warm-up request after server start:

```bash
# Warm-up script (add to your startup sequence)
curl -s -X POST http://localhost:8420/api/v1/memories/search \
  -H "Content-Type: application/json" \
  -d '{"query": "warmup"}' > /dev/null
```

### 10. High Memory Usage After Large Ingestion

**Symptom:** Server RSS grows significantly after ingesting thousands of memories (> 500MB for 100k+ nodes).

**Causes:**
- LMDB memory-maps the database file — reported RSS includes memory-mapped pages even if they're shared/reclaimable
- HNSW index resides in memory
- ONNX runtime allocates model buffers

**Mitigations:**

1. **LMDB RSS is inflated** — LMDB uses `mmap`, so the OS reports mapped pages as resident memory. Actual memory pressure is much lower. Check actual usage with:

```bash
# macOS: check actual physical memory (not virtual)
vmmap $(pgrep ucotron_server) | grep "Physical footprint"

# Linux: check RssAnon (actual heap allocations)
grep RssAnon /proc/$(pgrep ucotron_server)/status
```

2. **Tune HNSW parameters** for lower memory:

```toml
[storage.hnsw]
m = 12           # Default 16; lower = less memory, slightly lower recall
ef_construction = 100  # Default 200; lower = less index build memory
```

3. **Increase consolidation frequency** to reduce stale community data:

```toml
[extraction]
consolidation_trigger_interval = 500  # Default 100; less frequent rebuilds
```

### 11. MCP Port Collision Between REST and SSE

**Symptom:** Server fails with `mcp.port (8420) must differ from server.port (8420)`.

**Cause:** When MCP transport is `sse`, the MCP server needs its own port, separate from the REST API.

**Fix:**

```toml
[server]
port = 8420

[mcp]
transport = "sse"
port = 8421        # Must differ from server.port
```

This only applies when `mcp.transport = "sse"`. The default `stdio` transport doesn't need a port.

### 12. Shared Mode — "shared_data_dir is required"

**Symptom:** Server exits with `storage.shared_data_dir is required when storage.mode is 'shared'`.

**Cause:** Multi-instance shared mode requires all instances to point to the same data directory.

**Fix:**

```toml
[storage]
mode = "shared"
shared_data_dir = "/shared/ucotron-data"  # NFS mount or shared volume

[instance]
role = "writer"  # or "reader"
```

Ensure the `shared_data_dir` is accessible from all instances (e.g., an NFS mount, EFS, or Kubernetes PVC with `ReadWriteMany`).

### 13. Import Fails — "Unknown source system"

**Symptom:** `POST /api/v1/import` returns `Unknown source system: 'xxx'. Supported: mem0, zep`.

**Cause:** The import endpoint only supports Mem0 and Zep/Graphiti export formats.

**Fix:** Use one of the supported formats:

```bash
# Import from Mem0
curl -X POST http://localhost:8420/api/v1/import \
  -F "file=@mem0_export.json" \
  -F "source=mem0"

# Import from Zep/Graphiti
curl -X POST http://localhost:8420/api/v1/import \
  -F "file=@zep_export.json" \
  -F "source=zep"
```

For other systems, convert your data to the Ucotron JSON format first and use `POST /api/v1/memories` to ingest.

---

## Performance Troubleshooting

### Diagnosing Slow Ingestion

Check the per-step timing in the ingestion response:

```json
{
  "metrics": {
    "embedding_ms": 12,
    "ner_ms": 8,
    "relation_ms": 3,
    "resolution_ms": 1,
    "storage_ms": 5,
    "total_ms": 45
  }
}
```

| Bottleneck | Typical Cause | Fix |
|-----------|--------------|-----|
| `embedding_ms` > 50ms | CPU-bound ONNX inference | Increase `intra_threads` in model config; use release build |
| `ner_ms` > 30ms | GLiNER model is large | Reduce NER labels: `ner_labels = ["person", "organization"]` |
| `relation_ms` > 20ms | Many entities per chunk | Reduce chunk size to limit entities per chunk |
| `storage_ms` > 20ms | LMDB write contention | Check disk I/O; use SSD storage |
| `total_ms` > 200ms | All steps slow | Ensure `--release` build; debug builds are 10-50x slower |

**Critical:** Always run the server with `--release`:

```bash
cargo run --release --bin ucotron_server
# NOT: cargo run --bin ucotron_server  (debug build, extremely slow)
```

### Diagnosing Slow Search

Check the retrieval step timing in search responses:

| Slow Step | Fix |
|----------|-----|
| Vector search slow | Reduce `vector_top_k` (default 50) to narrow candidate pool |
| Graph expansion slow | Reduce `graph_expansion_hops` from 2 to 1, or set to 0 for vector-only |
| Community selection slow | Reduce `max_community_members` or disable `enable_community_expansion` |
| Re-ranking slow | Reduce candidate count (lower `vector_top_k`) |

### Benchmarking Your Deployment

Use the built-in benchmark runner to establish baseline performance:

```bash
cd memory_arena

# Ingest benchmark (100k nodes, 500k edges)
cargo run --release --bin bench_runner -- ingest --count 100000 --edges 500000

# Search benchmark (1000 queries, top-10, 2-hop)
cargo run --release --bin bench_runner -- search --queries 1000 --top-k 10 --hops 2
```

Expected performance on modern hardware (M1/M2 Mac or 8-core x86):
- Ingestion: > 100k nodes/s (HelixDB)
- Vector search P95: < 5ms (100k nodes)
- 1-hop traversal P95: < 10ms
- Hybrid search P95: < 50ms

---

## FAQ

### Can I use Ucotron without ONNX models?

Yes. If models are not downloaded, the server falls back to a stub embedder that generates zero vectors. This is useful for testing API integration without the full ML pipeline, but search quality will be meaningless since all embeddings are identical.

### How much disk space does Ucotron use?

LMDB storage is compact. Approximate sizes:
- 100k nodes + 500k edges: ~200-400MB
- 1M nodes + 5M edges: ~1.5-2GB

The HNSW index adds ~10-20% overhead. ONNX models require ~100MB total (MiniLM + GLiNER).

### Can I run multiple Ucotron instances?

Yes. Use `storage.mode = "shared"` with a shared data directory (NFS, EFS, or shared PVC). Configure one instance as `role = "writer"` and others as `role = "reader"`. Readers reject write operations with a 403 error and redirect clients to the writer.

### Does Ucotron support GPU acceleration?

ONNX Runtime can use GPU backends (CUDA, CoreML, DirectML) if compiled with the appropriate feature flags. The default build uses CPU-only inference. For most use cases, CPU inference with `intra_threads = 4` is sufficient — embedding 384-dim vectors is fast even on CPU.

### How do I back up my data?

LMDB supports hot backups by copying the `data.mdb` file while the server is running (LMDB uses copy-on-write, so reads are consistent). For a complete backup:

```bash
# Hot backup (server can stay running)
cp ucotron-data/data.mdb ucotron-data/data.mdb.backup

# Or use the GDPR export for a portable JSON export
curl http://localhost:8420/api/v1/gdpr/export \
  -H "Authorization: Bearer mk_admin_key" > backup.json
```

### What happens when contradictory facts are stored?

Ucotron automatically detects contradictions during ingestion. When a new fact conflicts with an existing one (same subject + predicate, different object), the resolution rules apply:
1. **Temporal**: If timestamps differ by > 1 year, the newer fact wins (older marked `Superseded`)
2. **Confidence**: If timestamps are close but confidence differs by > 0.3, higher confidence wins
3. **Ambiguous**: If both timestamps and confidences are similar, both are marked `Contradiction` for manual review

A `CONFLICTS_WITH` edge is always created between conflicting facts. Old facts are never deleted.

### Can I use Ucotron with a remote/external vector database?

Yes. Set `storage.mode = "external"` and configure the vector backend URL:

```toml
[storage]
mode = "external"

[storage.vector]
backend = "qdrant"
url = "http://qdrant-host:6334"

[storage.graph]
backend = "helix"
```

This uses Qdrant for vectors while keeping the graph in local LMDB.

### How do I reset all data and start fresh?

Stop the server and delete the data directory:

```bash
# Stop the server first!
rm -rf ./ucotron-data

# Restart — a fresh database will be created
cargo run --release --bin ucotron_server
```

For namespace-level cleanup without deleting everything, use the GDPR forget endpoint or delete individual memories via the API.
