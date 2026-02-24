#!/usr/bin/env python3
"""
Path-derived reward dataset generator for Ucotron fine-tuning.

Queries a Ucotron server's graph API to discover entity paths,
calculates rewards based on hop decay, predicate coherence, and
node centrality, then exports the results as JSONL.

Each record contains the source/target entity pair, the full path,
edge types, and the decomposed reward score — suitable for training
retrieval-reward models or path-scoring heads.

Usage:
    from path_reward_generator import PathRewardGenerator

    gen = PathRewardGenerator("http://localhost:8420")
    records = gen.generate(count=500, max_hops=4)
    gen.export(records, "path_reward_dataset.jsonl")

    # Or via CLI:
    python path_reward_generator.py --server http://localhost:8420 --count 500 --output path_reward_dataset.jsonl

    # Offline mode with synthetic graph:
    python path_reward_generator.py --offline --nodes 200 --edges 600 --count 500 --output path_reward_dataset.jsonl

Environment:
    UCOTRON_SERVER_URL - Default server URL (optional, fallback: http://localhost:8420).
    UCOTRON_API_KEY    - API key for authenticated access (optional).
    UCOTRON_NAMESPACE  - Default namespace (optional, fallback: "default").
"""

from __future__ import annotations

import json
import logging
import math
import os
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants — matching Rust PathRewardCalculator defaults
# ---------------------------------------------------------------------------

DEFAULT_HOP_DECAY = 0.8
DEFAULT_COHERENCE_WEIGHT = 0.4
DEFAULT_CENTRALITY_WEIGHT = 0.2

# Predicate semantic groups (matching Rust PredicateCoherenceConfig::default)
EDGE_GROUPS: dict[str, int] = {
    "RelatesTo": 0,
    "HasProperty": 0,
    "CausedBy": 1,
    "Supersedes": 1,
    "NextEpisode": 1,
    "ConflictsWith": 2,
    "Actor": 3,
    "Object": 3,
    "Location": 3,
    "Companion": 3,
}

SAME_SCORE = 1.0
RELATED_SCORE = 0.7
UNRELATED_SCORE = 0.3

# Synthetic graph edge types for offline mode
ALL_EDGE_TYPES = list(EDGE_GROUPS.keys())


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class GraphNode:
    """A node from the Ucotron graph."""
    id: int
    content: str
    node_type: str
    timestamp: int
    community_id: int | None = None


@dataclass
class GraphEdge:
    """An edge from the Ucotron graph."""
    source: int
    target: int
    weight: float
    edge_type: str = "RelatesTo"


@dataclass
class PathWithEdges:
    """A graph path annotated with edge types."""
    nodes: list[int]
    edge_types: list[str]

    @property
    def hop_count(self) -> int:
        return len(self.edge_types)


@dataclass
class RewardScore:
    """Decomposed reward score for a path."""
    total: float
    hop_decay_score: float
    coherence_score: float
    centrality_score: float


@dataclass
class PathRewardRecord:
    """A single training record for path-reward fine-tuning."""
    source_id: int
    target_id: int
    source_content: str
    target_content: str
    path_node_ids: list[int]
    edge_types: list[str]
    hop_count: int
    reward_total: float
    reward_hop_decay: float
    reward_coherence: float
    reward_centrality: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "source_content": self.source_content,
            "target_content": self.target_content,
            "path_node_ids": self.path_node_ids,
            "edge_types": self.edge_types,
            "hop_count": self.hop_count,
            "reward_total": round(self.reward_total, 6),
            "reward_hop_decay": round(self.reward_hop_decay, 6),
            "reward_coherence": round(self.reward_coherence, 6),
            "reward_centrality": round(self.reward_centrality, 6),
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Reward calculator (mirrors Rust PathRewardCalculator)
# ---------------------------------------------------------------------------

def predicate_similarity(a: str, b: str) -> float:
    """Score similarity between two edge types using semantic grouping."""
    if a == b:
        return SAME_SCORE
    group_a = EDGE_GROUPS.get(a)
    group_b = EDGE_GROUPS.get(b)
    if group_a is not None and group_b is not None and group_a == group_b:
        return RELATED_SCORE
    return UNRELATED_SCORE


def compute_hop_decay(hops: int, hop_decay: float = DEFAULT_HOP_DECAY) -> float:
    """Exponential decay: hop_decay ^ hops."""
    return hop_decay ** hops


def compute_coherence(edge_types: list[str]) -> float:
    """Average predicate similarity of consecutive edge pairs."""
    if len(edge_types) <= 1:
        return 1.0
    total = sum(
        predicate_similarity(edge_types[i], edge_types[i + 1])
        for i in range(len(edge_types) - 1)
    )
    return total / (len(edge_types) - 1)


def compute_centrality(
    nodes: list[int],
    degree_map: dict[int, int],
) -> float:
    """Normalized degree centrality of intermediate nodes."""
    if len(nodes) <= 2:
        return 0.5
    intermediates = nodes[1:-1]
    degrees = [degree_map.get(n, 1) for n in intermediates]
    max_degree = max(max(degrees), 1)
    return sum(d / max_degree for d in degrees) / len(intermediates)


def calculate_reward(
    path: PathWithEdges,
    degree_map: dict[int, int],
    hop_decay: float = DEFAULT_HOP_DECAY,
    coherence_weight: float = DEFAULT_COHERENCE_WEIGHT,
    centrality_weight: float = DEFAULT_CENTRALITY_WEIGHT,
) -> RewardScore:
    """Calculate the reward for a path (mirrors Rust PathRewardCalculator)."""
    if not path.nodes:
        return RewardScore(0.0, 0.0, 0.0, 0.0)

    hd = compute_hop_decay(path.hop_count, hop_decay)
    co = compute_coherence(path.edge_types)
    ce = compute_centrality(path.nodes, degree_map)

    base_weight = 1.0 - coherence_weight - centrality_weight
    total = max(0.0, min(1.0, base_weight * hd + coherence_weight * co + centrality_weight * ce))

    return RewardScore(total, hd, co, ce)


# ---------------------------------------------------------------------------
# Graph utilities
# ---------------------------------------------------------------------------

def build_adjacency(edges: list[GraphEdge]) -> dict[int, list[tuple[int, str]]]:
    """Build bidirectional adjacency map from edges."""
    adj: dict[int, list[tuple[int, str]]] = defaultdict(list)
    for e in edges:
        adj[e.source].append((e.target, e.edge_type))
        adj[e.target].append((e.source, e.edge_type))
    return dict(adj)


def build_degree_map(edges: list[GraphEdge]) -> dict[int, int]:
    """Count total degree (in + out) for each node."""
    degree: dict[int, int] = defaultdict(int)
    for e in edges:
        degree[e.source] += 1
        degree[e.target] += 1
    return dict(degree)


def find_paths(
    adj: dict[int, list[tuple[int, str]]],
    source: int,
    target: int,
    max_hops: int = 4,
    max_paths: int = 100,
) -> list[PathWithEdges]:
    """
    Find all paths between source and target via iterative DFS.

    Mirrors the Rust path_finder::find_paths algorithm:
    - Bidirectional traversal (adj already bidirectional)
    - Cycle prevention (no repeated nodes per path)
    - Bounded by max_hops and max_paths
    """
    if source == target:
        return [PathWithEdges(nodes=[source], edge_types=[])]

    results: list[PathWithEdges] = []
    # Stack: (current_node, path_so_far, edge_types_so_far)
    stack: list[tuple[int, list[int], list[str]]] = [(source, [source], [])]

    while stack:
        if len(results) >= max_paths:
            break

        current, path, edge_types = stack.pop()
        depth = len(edge_types)
        if depth >= max_hops:
            continue

        for neighbor, et in adj.get(current, []):
            if neighbor in path:
                continue

            new_path = path + [neighbor]
            new_edge_types = edge_types + [et]

            if neighbor == target:
                results.append(PathWithEdges(nodes=new_path, edge_types=new_edge_types))
                if len(results) >= max_paths:
                    break
            else:
                stack.append((neighbor, new_path, new_edge_types))

    results.sort(key=lambda p: p.hop_count)
    return results


# ---------------------------------------------------------------------------
# Synthetic graph generation (offline mode)
# ---------------------------------------------------------------------------

def generate_synthetic_graph(
    num_nodes: int = 200,
    num_edges: int = 600,
    seed: int = 42,
) -> tuple[list[GraphNode], list[GraphEdge]]:
    """Generate a synthetic graph for offline dataset generation."""
    rng = random.Random(seed)

    node_types = ["Entity", "Event", "Fact", "Skill"]
    type_weights = [60, 25, 15, 5]  # rough distribution

    nodes = []
    for i in range(num_nodes):
        nt = rng.choices(node_types, weights=type_weights, k=1)[0]
        nodes.append(GraphNode(
            id=i + 1,
            content=f"Node {i + 1} ({nt.lower()})",
            node_type=nt,
            timestamp=1700000000 + rng.randint(0, 31536000),
        ))

    # Power-law-ish edge generation (Zipf targets)
    node_ids = [n.id for n in nodes]
    edges = []
    seen = set()
    for _ in range(num_edges):
        src = rng.choice(node_ids)
        # Bias target toward lower IDs (hub nodes)
        tgt = node_ids[int(rng.paretovariate(1.5)) % num_nodes]
        if src == tgt or (src, tgt) in seen:
            continue
        seen.add((src, tgt))
        et = rng.choice(ALL_EDGE_TYPES)
        edges.append(GraphEdge(
            source=src,
            target=tgt,
            weight=round(rng.uniform(0.1, 1.0), 2),
            edge_type=et,
        ))

    logger.info("Generated synthetic graph: %d nodes, %d edges", len(nodes), len(edges))
    return nodes, edges


# ---------------------------------------------------------------------------
# Generator class
# ---------------------------------------------------------------------------

class PathRewardGenerator:
    """
    Generates path-derived reward training data from a Ucotron knowledge graph.

    Supports two modes:
    - **Online**: queries a running Ucotron server for graph data.
    - **Offline**: generates a synthetic graph locally.

    For each entity pair, finds all paths and scores them using hop decay,
    predicate coherence, and node centrality — matching the Rust
    PathRewardCalculator implementation.
    """

    def __init__(
        self,
        server_url: str | None = None,
        api_key: str | None = None,
        namespace: str | None = None,
        hop_decay: float = DEFAULT_HOP_DECAY,
        coherence_weight: float = DEFAULT_COHERENCE_WEIGHT,
        centrality_weight: float = DEFAULT_CENTRALITY_WEIGHT,
    ):
        self.server_url = (
            server_url
            or os.environ.get("UCOTRON_SERVER_URL")
            or "http://localhost:8420"
        )
        self.api_key = api_key or os.environ.get("UCOTRON_API_KEY")
        self.namespace = namespace or os.environ.get("UCOTRON_NAMESPACE", "default")
        self.hop_decay = hop_decay
        self.coherence_weight = coherence_weight
        self.centrality_weight = centrality_weight

        self._nodes: list[GraphNode] = []
        self._edges: list[GraphEdge] = []
        self._adj: dict[int, list[tuple[int, str]]] = {}
        self._degree_map: dict[int, int] = {}
        self._node_map: dict[int, GraphNode] = {}

    def fetch_graph(self, limit: int = 500) -> tuple[list[GraphNode], list[GraphEdge]]:
        """
        Fetch graph data from a running Ucotron server.

        Args:
            limit: Maximum number of nodes to fetch.

        Returns:
            Tuple of (nodes, edges).

        Raises:
            ConnectionError: If the server is unreachable.
            RuntimeError: If the API returns an error.
        """
        try:
            import httpx
        except ImportError:
            import urllib.request
            return self._fetch_graph_urllib(limit)

        headers: dict[str, str] = {"X-Ucotron-Namespace": self.namespace}
        if self.api_key:
            headers["X-Api-Key"] = self.api_key

        try:
            with httpx.Client(timeout=30.0) as client:
                resp = client.get(
                    f"{self.server_url}/api/v1/graph",
                    params={"limit": limit},
                    headers=headers,
                )
                resp.raise_for_status()
                data = resp.json()
        except httpx.ConnectError as exc:
            raise ConnectionError(f"Cannot connect to Ucotron server at {self.server_url}") from exc
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(f"Ucotron API error: {exc.response.status_code} {exc.response.text}") from exc

        return self._parse_graph_response(data)

    def _fetch_graph_urllib(self, limit: int) -> tuple[list[GraphNode], list[GraphEdge]]:
        """Fallback graph fetching using urllib (no external deps)."""
        import urllib.request
        import urllib.error

        url = f"{self.server_url}/api/v1/graph?limit={limit}"
        req = urllib.request.Request(url)
        req.add_header("X-Ucotron-Namespace", self.namespace)
        if self.api_key:
            req.add_header("X-Api-Key", self.api_key)

        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode())
        except urllib.error.URLError as exc:
            raise ConnectionError(f"Cannot connect to Ucotron server at {self.server_url}") from exc

        return self._parse_graph_response(data)

    def _parse_graph_response(
        self, data: dict[str, Any]
    ) -> tuple[list[GraphNode], list[GraphEdge]]:
        """Parse the /api/v1/graph JSON response into typed objects."""
        nodes = [
            GraphNode(
                id=n["id"],
                content=n.get("content", ""),
                node_type=n.get("node_type", "Entity"),
                timestamp=n.get("timestamp", 0),
                community_id=n.get("community_id"),
            )
            for n in data.get("nodes", [])
        ]

        edges = [
            GraphEdge(
                source=e["source"],
                target=e["target"],
                weight=e.get("weight", 1.0),
                edge_type=e.get("edge_type", "RelatesTo"),
            )
            for e in data.get("edges", [])
        ]

        logger.info("Fetched graph: %d nodes, %d edges", len(nodes), len(edges))
        return nodes, edges

    def load_graph(
        self,
        nodes: list[GraphNode],
        edges: list[GraphEdge],
    ) -> None:
        """Load a graph into the generator (from fetch or synthetic)."""
        self._nodes = nodes
        self._edges = edges
        self._adj = build_adjacency(edges)
        self._degree_map = build_degree_map(edges)
        self._node_map = {n.id: n for n in nodes}
        logger.info(
            "Loaded graph: %d nodes, %d edges, avg degree %.1f",
            len(nodes),
            len(edges),
            sum(self._degree_map.values()) / max(len(self._degree_map), 1),
        )

    def find_entity_paths(
        self,
        source: int,
        target: int,
        max_hops: int = 4,
        max_paths: int = 20,
    ) -> list[PathWithEdges]:
        """Find all paths between two nodes."""
        return find_paths(self._adj, source, target, max_hops, max_paths)

    def score_path(self, path: PathWithEdges) -> RewardScore:
        """Calculate the reward score for a single path."""
        return calculate_reward(
            path,
            self._degree_map,
            self.hop_decay,
            self.coherence_weight,
            self.centrality_weight,
        )

    def generate(
        self,
        count: int = 500,
        max_hops: int = 4,
        max_paths_per_pair: int = 20,
        seed: int = 42,
        progress_interval: int = 100,
    ) -> list[PathRewardRecord]:
        """
        Generate path-reward training records from the loaded graph.

        Samples random entity pairs, finds paths between them,
        and scores each path. Pairs with no connecting path are skipped.

        Args:
            count: Target number of training records.
            max_hops: Maximum path length.
            max_paths_per_pair: Max paths to explore per entity pair.
            seed: RNG seed for entity pair sampling.
            progress_interval: Log progress every N attempts.

        Returns:
            List of PathRewardRecord instances.
        """
        if not self._nodes:
            raise ValueError("No graph loaded. Call load_graph() or fetch_graph() first.")

        rng = random.Random(seed)
        node_ids = [n.id for n in self._nodes]

        records: list[PathRewardRecord] = []
        attempts = 0
        max_attempts = count * 5  # Allow misses for disconnected pairs

        while len(records) < count and attempts < max_attempts:
            src_id = rng.choice(node_ids)
            tgt_id = rng.choice(node_ids)
            attempts += 1

            if src_id == tgt_id:
                continue

            paths = self.find_entity_paths(src_id, tgt_id, max_hops, max_paths_per_pair)
            if not paths:
                continue

            src_node = self._node_map.get(src_id)
            tgt_node = self._node_map.get(tgt_id)
            if not src_node or not tgt_node:
                continue

            for path in paths:
                if len(records) >= count:
                    break

                score = self.score_path(path)
                records.append(PathRewardRecord(
                    source_id=src_id,
                    target_id=tgt_id,
                    source_content=src_node.content,
                    target_content=tgt_node.content,
                    path_node_ids=path.nodes,
                    edge_types=path.edge_types,
                    hop_count=path.hop_count,
                    reward_total=score.total,
                    reward_hop_decay=score.hop_decay_score,
                    reward_coherence=score.coherence_score,
                    reward_centrality=score.centrality_score,
                    metadata={
                        "source_type": src_node.node_type,
                        "target_type": tgt_node.node_type,
                    },
                ))

            if attempts % progress_interval == 0:
                logger.info(
                    "Path reward generation progress: %d/%d records (%d attempts)",
                    len(records), count, attempts,
                )

        logger.info(
            "Path reward generation complete: %d/%d records in %d attempts (%.1f%% hit rate)",
            len(records), count, attempts,
            len(records) / max(attempts, 1) * 100,
        )
        return records

    def export(
        self,
        records: list[PathRewardRecord],
        output_path: str | Path,
    ) -> int:
        """
        Export records to JSONL file.

        Args:
            records: List of PathRewardRecords.
            output_path: Destination file path.

        Returns:
            Number of records written.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        count = 0
        with open(output_path, "w") as f:
            for record in records:
                f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
                count += 1

        logger.info("Exported %d records to %s", count, output_path)
        return count


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    """Generate path-reward dataset via CLI."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate path-derived reward dataset from Ucotron knowledge graph"
    )
    parser.add_argument(
        "--server", type=str, default=None,
        help="Ucotron server URL (default: UCOTRON_SERVER_URL or http://localhost:8420)",
    )
    parser.add_argument(
        "--api-key", type=str, default=None,
        help="Ucotron API key (default: UCOTRON_API_KEY env var)",
    )
    parser.add_argument(
        "--namespace", type=str, default=None,
        help="Ucotron namespace (default: UCOTRON_NAMESPACE or 'default')",
    )
    parser.add_argument(
        "--offline", action="store_true",
        help="Use synthetic graph instead of querying a server",
    )
    parser.add_argument(
        "--nodes", type=int, default=200,
        help="Number of nodes for synthetic graph (default: 200)",
    )
    parser.add_argument(
        "--edges", type=int, default=600,
        help="Number of edges for synthetic graph (default: 600)",
    )
    parser.add_argument(
        "--count", type=int, default=500,
        help="Number of training records to generate (default: 500)",
    )
    parser.add_argument(
        "--max-hops", type=int, default=4,
        help="Maximum path length in hops (default: 4)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--output", type=str, default="path_reward_dataset.jsonl",
        help="Output JSONL file path (default: path_reward_dataset.jsonl)",
    )
    parser.add_argument(
        "--graph-limit", type=int, default=500,
        help="Max nodes to fetch from server (default: 500)",
    )
    parser.add_argument(
        "--progress-interval", type=int, default=100,
        help="Log progress every N attempts (default: 100)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )

    gen = PathRewardGenerator(
        server_url=args.server,
        api_key=args.api_key,
        namespace=args.namespace,
    )

    if args.offline:
        nodes, edges = generate_synthetic_graph(
            num_nodes=args.nodes,
            num_edges=args.edges,
            seed=args.seed,
        )
    else:
        nodes, edges = gen.fetch_graph(limit=args.graph_limit)

    gen.load_graph(nodes, edges)

    records = gen.generate(
        count=args.count,
        max_hops=args.max_hops,
        seed=args.seed,
        progress_interval=args.progress_interval,
    )

    written = gen.export(records, args.output)
    print(f"\nGenerated {written} path reward records → {args.output}")

    # Print summary stats
    if records:
        avg_reward = sum(r.reward_total for r in records) / len(records)
        avg_hops = sum(r.hop_count for r in records) / len(records)
        print(f"Average reward: {avg_reward:.4f}")
        print(f"Average hops: {avg_hops:.1f}")
        print(f"Hop distribution: {dict(sorted(_hop_distribution(records).items()))}")


def _hop_distribution(records: list[PathRewardRecord]) -> dict[int, int]:
    """Count records by hop count."""
    dist: dict[int, int] = defaultdict(int)
    for r in records:
        dist[r.hop_count] += 1
    return dict(dist)


if __name__ == "__main__":
    main()
