#!/usr/bin/env python3
"""Unit tests for PathRewardGenerator (offline / mocked, no real server)."""

import json
import os
import tempfile
import unittest

from path_reward_generator import (
    PathRewardGenerator,
    PathRewardRecord,
    PathWithEdges,
    RewardScore,
    GraphNode,
    GraphEdge,
    build_adjacency,
    build_degree_map,
    calculate_reward,
    compute_coherence,
    compute_centrality,
    compute_hop_decay,
    find_paths,
    generate_synthetic_graph,
    predicate_similarity,
    DEFAULT_HOP_DECAY,
    DEFAULT_COHERENCE_WEIGHT,
    DEFAULT_CENTRALITY_WEIGHT,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chain_graph(length: int = 5) -> tuple[list[GraphNode], list[GraphEdge]]:
    """Create a simple chain: 1->2->3->...->length."""
    nodes = [
        GraphNode(id=i, content=f"node_{i}", node_type="Entity", timestamp=100)
        for i in range(1, length + 1)
    ]
    edges = [
        GraphEdge(source=i, target=i + 1, weight=1.0, edge_type="RelatesTo")
        for i in range(1, length)
    ]
    return nodes, edges


def _make_diamond_graph() -> tuple[list[GraphNode], list[GraphEdge]]:
    """Diamond: 1->2->4, 1->3->4 with mixed edge types."""
    nodes = [
        GraphNode(id=i, content=f"node_{i}", node_type="Entity", timestamp=100)
        for i in range(1, 5)
    ]
    edges = [
        GraphEdge(source=1, target=2, weight=1.0, edge_type="RelatesTo"),
        GraphEdge(source=1, target=3, weight=1.0, edge_type="CausedBy"),
        GraphEdge(source=2, target=4, weight=1.0, edge_type="HasProperty"),
        GraphEdge(source=3, target=4, weight=1.0, edge_type="Actor"),
    ]
    return nodes, edges


# ---------------------------------------------------------------------------
# Test: predicate similarity
# ---------------------------------------------------------------------------

class TestPredicateSimilarity(unittest.TestCase):

    def test_same_type_returns_1(self):
        self.assertEqual(predicate_similarity("Actor", "Actor"), 1.0)
        self.assertEqual(predicate_similarity("CausedBy", "CausedBy"), 1.0)

    def test_related_types_return_0_7(self):
        # Same group: Actor and Object (group 3)
        self.assertAlmostEqual(predicate_similarity("Actor", "Object"), 0.7)
        # Same group: CausedBy and Supersedes (group 1)
        self.assertAlmostEqual(predicate_similarity("CausedBy", "Supersedes"), 0.7)
        # Same group: RelatesTo and HasProperty (group 0)
        self.assertAlmostEqual(predicate_similarity("RelatesTo", "HasProperty"), 0.7)

    def test_unrelated_types_return_0_3(self):
        # Different groups: Actor (3) vs CausedBy (1)
        self.assertAlmostEqual(predicate_similarity("Actor", "CausedBy"), 0.3)
        # Different groups: RelatesTo (0) vs ConflictsWith (2)
        self.assertAlmostEqual(predicate_similarity("RelatesTo", "ConflictsWith"), 0.3)

    def test_unknown_types_return_unrelated(self):
        self.assertAlmostEqual(predicate_similarity("Actor", "UnknownType"), 0.3)


# ---------------------------------------------------------------------------
# Test: reward calculation functions
# ---------------------------------------------------------------------------

class TestComputeHopDecay(unittest.TestCase):

    def test_zero_hops(self):
        self.assertAlmostEqual(compute_hop_decay(0), 1.0)

    def test_one_hop(self):
        self.assertAlmostEqual(compute_hop_decay(1), 0.8)

    def test_four_hops(self):
        self.assertAlmostEqual(compute_hop_decay(4), 0.8**4)

    def test_custom_decay(self):
        self.assertAlmostEqual(compute_hop_decay(2, 0.5), 0.25)


class TestComputeCoherence(unittest.TestCase):

    def test_single_edge_returns_1(self):
        self.assertAlmostEqual(compute_coherence(["RelatesTo"]), 1.0)

    def test_empty_returns_1(self):
        self.assertAlmostEqual(compute_coherence([]), 1.0)

    def test_same_edges_returns_1(self):
        self.assertAlmostEqual(compute_coherence(["Actor", "Actor"]), 1.0)

    def test_related_edges(self):
        # Actor -> Object: same group (3), score = 0.7
        self.assertAlmostEqual(compute_coherence(["Actor", "Object"]), 0.7)

    def test_mixed_edges(self):
        # (Actor,Actor)=1.0, (Actor,Object)=0.7, (Object,CausedBy)=0.3
        expected = (1.0 + 0.7 + 0.3) / 3
        result = compute_coherence(["Actor", "Actor", "Object", "CausedBy"])
        self.assertAlmostEqual(result, expected, places=3)


class TestComputeCentrality(unittest.TestCase):

    def test_no_intermediates_returns_0_5(self):
        self.assertAlmostEqual(compute_centrality([1, 2], {}), 0.5)

    def test_single_intermediate(self):
        # Node 2 has degree 10, max=10, normalized=1.0
        self.assertAlmostEqual(compute_centrality([1, 2, 3], {2: 10}), 1.0)

    def test_multiple_intermediates(self):
        # Nodes 2,3 with degrees 5,10 → normalized: 0.5, 1.0 → avg = 0.75
        result = compute_centrality([1, 2, 3, 4], {2: 5, 3: 10})
        self.assertAlmostEqual(result, 0.75)


class TestCalculateReward(unittest.TestCase):

    def test_empty_path(self):
        score = calculate_reward(
            PathWithEdges(nodes=[], edge_types=[]),
            {},
        )
        self.assertEqual(score.total, 0.0)

    def test_single_hop_path(self):
        score = calculate_reward(
            PathWithEdges(nodes=[1, 2], edge_types=["RelatesTo"]),
            {1: 5, 2: 5},
        )
        self.assertGreater(score.total, 0.0)
        self.assertAlmostEqual(score.hop_decay_score, 0.8)
        self.assertAlmostEqual(score.coherence_score, 1.0)

    def test_shorter_path_scores_higher(self):
        degree_map = {i: 5 for i in range(1, 6)}
        short = calculate_reward(
            PathWithEdges(nodes=[1, 2], edge_types=["RelatesTo"]),
            degree_map,
        )
        long = calculate_reward(
            PathWithEdges(
                nodes=[1, 2, 3, 4, 5],
                edge_types=["RelatesTo", "RelatesTo", "RelatesTo", "RelatesTo"],
            ),
            degree_map,
        )
        self.assertGreater(short.total, long.total)

    def test_reward_clamped_0_to_1(self):
        score = calculate_reward(
            PathWithEdges(nodes=[1, 2], edge_types=["RelatesTo"]),
            {1: 1, 2: 1},
        )
        self.assertGreaterEqual(score.total, 0.0)
        self.assertLessEqual(score.total, 1.0)


# ---------------------------------------------------------------------------
# Test: graph utilities
# ---------------------------------------------------------------------------

class TestBuildAdjacency(unittest.TestCase):

    def test_bidirectional(self):
        edges = [GraphEdge(source=1, target=2, weight=1.0, edge_type="RelatesTo")]
        adj = build_adjacency(edges)
        self.assertIn(2, [n for n, _ in adj[1]])
        self.assertIn(1, [n for n, _ in adj[2]])

    def test_preserves_edge_type(self):
        edges = [GraphEdge(source=1, target=2, weight=1.0, edge_type="CausedBy")]
        adj = build_adjacency(edges)
        self.assertEqual(adj[1][0], (2, "CausedBy"))


class TestBuildDegreeMap(unittest.TestCase):

    def test_simple_chain(self):
        edges = [
            GraphEdge(source=1, target=2, weight=1.0),
            GraphEdge(source=2, target=3, weight=1.0),
        ]
        degrees = build_degree_map(edges)
        self.assertEqual(degrees[1], 1)
        self.assertEqual(degrees[2], 2)
        self.assertEqual(degrees[3], 1)


# ---------------------------------------------------------------------------
# Test: path finding
# ---------------------------------------------------------------------------

class TestFindPaths(unittest.TestCase):

    def test_same_source_target(self):
        adj = build_adjacency([])
        paths = find_paths(adj, 1, 1)
        self.assertEqual(len(paths), 1)
        self.assertEqual(paths[0].nodes, [1])

    def test_simple_chain(self):
        _, edges = _make_chain_graph(3)
        adj = build_adjacency(edges)
        paths = find_paths(adj, 1, 3)
        self.assertTrue(any(p.nodes == [1, 2, 3] for p in paths))

    def test_no_path(self):
        # Disconnected nodes
        edges = [GraphEdge(source=1, target=2, weight=1.0)]
        adj = build_adjacency(edges)
        paths = find_paths(adj, 1, 99)
        self.assertEqual(len(paths), 0)

    def test_hop_limit(self):
        _, edges = _make_chain_graph(5)
        adj = build_adjacency(edges)
        # 1->2->3->4->5 is 4 hops; max_hops=2 should fail
        paths = find_paths(adj, 1, 5, max_hops=2)
        self.assertEqual(len(paths), 0)

        paths = find_paths(adj, 1, 5, max_hops=4)
        self.assertGreater(len(paths), 0)

    def test_max_paths_limit(self):
        _, edges = _make_diamond_graph()
        adj = build_adjacency(edges)
        paths = find_paths(adj, 1, 4, max_paths=1)
        self.assertLessEqual(len(paths), 1)

    def test_diamond_finds_multiple_paths(self):
        _, edges = _make_diamond_graph()
        adj = build_adjacency(edges)
        paths = find_paths(adj, 1, 4, max_hops=4)
        # Should find at least 2 paths (via 2 and via 3)
        self.assertGreaterEqual(len(paths), 2)

    def test_paths_sorted_by_length(self):
        _, edges = _make_diamond_graph()
        adj = build_adjacency(edges)
        paths = find_paths(adj, 1, 4, max_hops=4)
        for i in range(len(paths) - 1):
            self.assertLessEqual(paths[i].hop_count, paths[i + 1].hop_count)

    def test_edge_types_tracked(self):
        edges = [
            GraphEdge(source=1, target=2, weight=1.0, edge_type="CausedBy"),
            GraphEdge(source=2, target=3, weight=1.0, edge_type="HasProperty"),
        ]
        adj = build_adjacency(edges)
        paths = find_paths(adj, 1, 3, max_hops=4)
        self.assertTrue(len(paths) > 0)
        p = paths[0]
        self.assertEqual(len(p.edge_types), len(p.nodes) - 1)


# ---------------------------------------------------------------------------
# Test: synthetic graph generation
# ---------------------------------------------------------------------------

class TestSyntheticGraph(unittest.TestCase):

    def test_generates_correct_count(self):
        nodes, edges = generate_synthetic_graph(num_nodes=50, num_edges=100, seed=1)
        self.assertEqual(len(nodes), 50)
        # Edges may be fewer due to dedup/self-loop filtering
        self.assertGreater(len(edges), 0)
        self.assertLessEqual(len(edges), 100)

    def test_deterministic_with_seed(self):
        a_nodes, a_edges = generate_synthetic_graph(50, 100, seed=99)
        b_nodes, b_edges = generate_synthetic_graph(50, 100, seed=99)
        self.assertEqual([n.id for n in a_nodes], [n.id for n in b_nodes])
        self.assertEqual(
            [(e.source, e.target) for e in a_edges],
            [(e.source, e.target) for e in b_edges],
        )

    def test_node_ids_sequential(self):
        nodes, _ = generate_synthetic_graph(10, 20, seed=1)
        ids = [n.id for n in nodes]
        self.assertEqual(ids, list(range(1, 11)))

    def test_edge_types_valid(self):
        _, edges = generate_synthetic_graph(30, 60, seed=1)
        from path_reward_generator import ALL_EDGE_TYPES
        for e in edges:
            self.assertIn(e.edge_type, ALL_EDGE_TYPES)


# ---------------------------------------------------------------------------
# Test: PathRewardRecord
# ---------------------------------------------------------------------------

class TestPathRewardRecord(unittest.TestCase):

    def test_to_dict_has_required_fields(self):
        rec = PathRewardRecord(
            source_id=1, target_id=2,
            source_content="A", target_content="B",
            path_node_ids=[1, 3, 2], edge_types=["RelatesTo", "CausedBy"],
            hop_count=2,
            reward_total=0.75, reward_hop_decay=0.64,
            reward_coherence=0.7, reward_centrality=0.5,
        )
        d = rec.to_dict()
        for key in [
            "source_id", "target_id", "source_content", "target_content",
            "path_node_ids", "edge_types", "hop_count",
            "reward_total", "reward_hop_decay", "reward_coherence",
            "reward_centrality", "metadata",
        ]:
            self.assertIn(key, d)

    def test_to_dict_json_serializable(self):
        rec = PathRewardRecord(
            source_id=1, target_id=2,
            source_content="A", target_content="B",
            path_node_ids=[1, 2], edge_types=["RelatesTo"],
            hop_count=1,
            reward_total=0.8, reward_hop_decay=0.8,
            reward_coherence=1.0, reward_centrality=0.5,
            metadata={"source_type": "Entity"},
        )
        json_str = json.dumps(rec.to_dict())
        self.assertIsInstance(json_str, str)
        parsed = json.loads(json_str)
        self.assertEqual(parsed["source_id"], 1)
        self.assertEqual(parsed["metadata"]["source_type"], "Entity")

    def test_reward_values_rounded(self):
        rec = PathRewardRecord(
            source_id=1, target_id=2,
            source_content="A", target_content="B",
            path_node_ids=[1, 2], edge_types=["RelatesTo"],
            hop_count=1,
            reward_total=0.123456789, reward_hop_decay=0.8,
            reward_coherence=1.0, reward_centrality=0.5,
        )
        d = rec.to_dict()
        # Should be rounded to 6 decimal places
        self.assertEqual(d["reward_total"], 0.123457)


# ---------------------------------------------------------------------------
# Test: PathRewardGenerator (end-to-end with synthetic graph)
# ---------------------------------------------------------------------------

class TestPathRewardGenerator(unittest.TestCase):

    def test_generate_with_synthetic_graph(self):
        gen = PathRewardGenerator()
        nodes, edges = generate_synthetic_graph(100, 300, seed=42)
        gen.load_graph(nodes, edges)
        records = gen.generate(count=20, seed=42)
        self.assertGreater(len(records), 0)
        self.assertLessEqual(len(records), 20)

    def test_generate_raises_without_graph(self):
        gen = PathRewardGenerator()
        with self.assertRaises(ValueError):
            gen.generate(count=10)

    def test_generate_deterministic(self):
        nodes, edges = generate_synthetic_graph(80, 200, seed=7)

        gen1 = PathRewardGenerator()
        gen1.load_graph(nodes, edges)
        records1 = gen1.generate(count=10, seed=123)

        gen2 = PathRewardGenerator()
        gen2.load_graph(nodes, edges)
        records2 = gen2.generate(count=10, seed=123)

        self.assertEqual(len(records1), len(records2))
        for r1, r2 in zip(records1, records2):
            self.assertEqual(r1.source_id, r2.source_id)
            self.assertEqual(r1.target_id, r2.target_id)
            self.assertAlmostEqual(r1.reward_total, r2.reward_total, places=5)

    def test_generate_records_have_valid_rewards(self):
        nodes, edges = generate_synthetic_graph(100, 300, seed=42)
        gen = PathRewardGenerator()
        gen.load_graph(nodes, edges)
        records = gen.generate(count=10, seed=42)

        for rec in records:
            self.assertGreaterEqual(rec.reward_total, 0.0)
            self.assertLessEqual(rec.reward_total, 1.0)
            self.assertGreater(rec.hop_count, 0)
            self.assertEqual(len(rec.edge_types), rec.hop_count)
            self.assertEqual(len(rec.path_node_ids), rec.hop_count + 1)

    def test_export_creates_valid_jsonl(self):
        nodes, edges = generate_synthetic_graph(50, 150, seed=1)
        gen = PathRewardGenerator()
        gen.load_graph(nodes, edges)
        records = gen.generate(count=5, seed=1)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            path = f.name

        try:
            written = gen.export(records, path)
            self.assertEqual(written, len(records))

            with open(path) as f:
                lines = f.readlines()
            self.assertEqual(len(lines), len(records))

            for line in lines:
                parsed = json.loads(line.strip())
                self.assertIn("source_id", parsed)
                self.assertIn("reward_total", parsed)
                self.assertIn("edge_types", parsed)
                self.assertIsInstance(parsed["path_node_ids"], list)
        finally:
            os.unlink(path)

    def test_export_creates_parent_dirs(self):
        nodes, edges = generate_synthetic_graph(30, 80, seed=1)
        gen = PathRewardGenerator()
        gen.load_graph(nodes, edges)
        records = gen.generate(count=3, seed=1)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "subdir", "nested", "output.jsonl")
            written = gen.export(records, path)
            self.assertEqual(written, len(records))
            self.assertTrue(os.path.exists(path))

    def test_load_graph_stores_metadata(self):
        nodes, edges = _make_chain_graph(5)
        gen = PathRewardGenerator()
        gen.load_graph(nodes, edges)
        self.assertEqual(len(gen._nodes), 5)
        self.assertEqual(len(gen._edges), 4)
        self.assertIn(1, gen._node_map)
        self.assertIn(1, gen._adj)


class TestPathRewardGeneratorScoring(unittest.TestCase):

    def test_score_matches_calculate_reward(self):
        nodes, edges = _make_chain_graph(4)
        gen = PathRewardGenerator()
        gen.load_graph(nodes, edges)

        path = PathWithEdges(nodes=[1, 2, 3], edge_types=["RelatesTo", "RelatesTo"])
        score_via_gen = gen.score_path(path)
        score_direct = calculate_reward(path, gen._degree_map)
        self.assertAlmostEqual(score_via_gen.total, score_direct.total, places=5)


if __name__ == "__main__":
    unittest.main()
