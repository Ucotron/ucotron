#!/usr/bin/env python3
"""Unit tests for ProjectionDatasetGenerator (mocked, no real API or model calls)."""

import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

# Set a dummy key before importing
os.environ.setdefault("FIREWORKS_API_KEY", "fw-test-key-not-real")

from llm_dataset_generator import LLMDatasetGenerator, GenerationResult
from projection_dataset_generator import (
    ProjectionDatasetGenerator,
    EmbeddingPair,
    GenerationStats,
    CATEGORY_PROMPTS,
    DESCRIPTION_SYSTEM_PROMPT,
    CLIP_EMBED_DIM,
    MINILM_EMBED_DIM,
)


# ---------------------------------------------------------------------------
# Mock encoders
# ---------------------------------------------------------------------------

class MockCLIPEncoder:
    """Mock CLIP text encoder producing deterministic 512-dim embeddings."""

    @property
    def dim(self) -> int:
        return CLIP_EMBED_DIM

    def encode(self, texts: list[str]) -> list[list[float]]:
        results = []
        for i, text in enumerate(texts):
            # Deterministic: use hash of text to seed values
            seed_val = hash(text) % 10000
            emb = [(seed_val + j) / 10000.0 for j in range(CLIP_EMBED_DIM)]
            # L2 normalize
            norm = sum(x * x for x in emb) ** 0.5
            emb = [x / norm for x in emb]
            results.append(emb)
        return results


class MockMiniLMEncoder:
    """Mock MiniLM encoder producing deterministic 384-dim embeddings."""

    @property
    def dim(self) -> int:
        return MINILM_EMBED_DIM

    def encode(self, texts: list[str]) -> list[list[float]]:
        results = []
        for text in texts:
            seed_val = hash(text) % 10000
            emb = [(seed_val + j + 100) / 10000.0 for j in range(MINILM_EMBED_DIM)]
            norm = sum(x * x for x in emb) ** 0.5
            emb = [x / norm for x in emb]
            results.append(emb)
        return results


def _make_llm_mock() -> MagicMock:
    """Create a mocked LLMDatasetGenerator."""
    mock = MagicMock(spec=LLMDatasetGenerator)
    mock.usage_summary.return_value = {
        "total_requests": 0,
        "total_prompt_tokens": 0,
        "total_completion_tokens": 0,
        "total_tokens": 0,
    }
    return mock


def _make_batch_response(count: int) -> GenerationResult:
    """Create a mock LLM response with numbered descriptions."""
    lines = []
    for i in range(1, count + 1):
        lines.append(
            f"{i}. A beautiful photograph of scene {i} with vivid colors "
            f"and interesting composition showing a detailed view."
        )
    return GenerationResult(
        text="\n".join(lines),
        prompt_tokens=100,
        completion_tokens=200,
    )


def _make_single_response(idx: int = 1) -> GenerationResult:
    """Create a mock LLM response for a single description."""
    return GenerationResult(
        text=f"A beautiful photograph of scene {idx} with vivid colors "
             f"and interesting composition showing a detailed view.",
        prompt_tokens=50,
        completion_tokens=100,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestEmbeddingPair(unittest.TestCase):
    """Test EmbeddingPair data structure."""

    def test_to_dict_roundtrip(self):
        pair = EmbeddingPair(
            description="A red car on a highway",
            clip_embedding=[0.1] * CLIP_EMBED_DIM,
            minilm_embedding=[0.2] * MINILM_EMBED_DIM,
            category="vehicle",
        )
        d = pair.to_dict()
        restored = EmbeddingPair.from_dict(d)
        self.assertEqual(restored.description, pair.description)
        self.assertEqual(len(restored.clip_embedding), CLIP_EMBED_DIM)
        self.assertEqual(len(restored.minilm_embedding), MINILM_EMBED_DIM)
        self.assertEqual(restored.category, "vehicle")

    def test_to_dict_keys(self):
        pair = EmbeddingPair(
            description="test",
            clip_embedding=[0.0] * CLIP_EMBED_DIM,
            minilm_embedding=[0.0] * MINILM_EMBED_DIM,
        )
        d = pair.to_dict()
        self.assertIn("description", d)
        self.assertIn("clip_embedding", d)
        self.assertIn("minilm_embedding", d)
        self.assertIn("category", d)

    def test_from_dict_missing_category(self):
        d = {
            "description": "test",
            "clip_embedding": [0.0] * CLIP_EMBED_DIM,
            "minilm_embedding": [0.0] * MINILM_EMBED_DIM,
        }
        pair = EmbeddingPair.from_dict(d)
        self.assertEqual(pair.category, "")


class TestGenerationStats(unittest.TestCase):
    """Test GenerationStats."""

    def test_success_rate_zero(self):
        stats = GenerationStats(total_requested=0)
        self.assertEqual(stats.success_rate, 0.0)

    def test_success_rate_partial(self):
        stats = GenerationStats(total_requested=100, pairs_encoded=75)
        self.assertAlmostEqual(stats.success_rate, 0.75)

    def test_success_rate_full(self):
        stats = GenerationStats(total_requested=50, pairs_encoded=50)
        self.assertAlmostEqual(stats.success_rate, 1.0)


class TestCategoryPrompts(unittest.TestCase):
    """Test category prompt configuration."""

    def test_at_least_20_categories(self):
        self.assertGreaterEqual(len(CATEGORY_PROMPTS), 20)

    def test_categories_are_strings(self):
        for p in CATEGORY_PROMPTS:
            self.assertIsInstance(p, str)
            self.assertGreater(len(p), 10)

    def test_system_prompt_exists(self):
        self.assertGreater(len(DESCRIPTION_SYSTEM_PROMPT), 50)


class TestNumberedListParsing(unittest.TestCase):
    """Test _parse_numbered_list method."""

    def setUp(self):
        self.gen = ProjectionDatasetGenerator(
            llm=_make_llm_mock(),
            clip_encoder=MockCLIPEncoder(),
            minilm_encoder=MockMiniLMEncoder(),
        )

    def test_parse_simple_list(self):
        text = "1. First description here.\n2. Second description.\n3. Third one."
        result = self.gen._parse_numbered_list(text)
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0], "First description here.")

    def test_parse_multiline_descriptions(self):
        text = "1. First part of description.\nContinuation of first.\n2. Second."
        result = self.gen._parse_numbered_list(text)
        self.assertEqual(len(result), 2)
        self.assertIn("Continuation", result[0])

    def test_parse_empty_string(self):
        result = self.gen._parse_numbered_list("")
        self.assertEqual(len(result), 0)

    def test_parse_with_blank_lines(self):
        text = "1. First.\n\n2. Second.\n\n3. Third."
        result = self.gen._parse_numbered_list(text)
        self.assertEqual(len(result), 3)

    def test_parse_with_parenthesis_numbering(self):
        text = "1) First.\n2) Second."
        result = self.gen._parse_numbered_list(text)
        self.assertEqual(len(result), 2)


class TestCategoryExtraction(unittest.TestCase):
    """Test _extract_category helper."""

    def setUp(self):
        self.gen = ProjectionDatasetGenerator(
            llm=_make_llm_mock(),
            clip_encoder=MockCLIPEncoder(),
            minilm_encoder=MockMiniLMEncoder(),
        )

    def test_landscape(self):
        prompt = "Describe a photograph of a natural landscape."
        self.assertEqual(self.gen._extract_category(prompt), "landscape")

    def test_urban(self):
        prompt = "Describe an urban street scene."
        self.assertEqual(self.gen._extract_category(prompt), "urban")

    def test_unknown(self):
        prompt = "Describe something completely random and unusual."
        self.assertEqual(self.gen._extract_category(prompt), "general")


class TestDescriptionGeneration(unittest.TestCase):
    """Test description generation via mocked LLM."""

    def test_batch_generation(self):
        llm = _make_llm_mock()
        llm.generate.return_value = _make_batch_response(10)

        gen = ProjectionDatasetGenerator(
            llm=llm,
            clip_encoder=MockCLIPEncoder(),
            minilm_encoder=MockMiniLMEncoder(),
            batch_size=10,
        )
        descriptions = gen._generate_descriptions(10)
        self.assertEqual(len(descriptions), 10)
        # Each is a (text, category) tuple
        for desc, cat in descriptions:
            self.assertIsInstance(desc, str)
            self.assertGreater(len(desc), 0)

    def test_single_generation_for_small_count(self):
        llm = _make_llm_mock()
        llm.generate.return_value = _make_single_response()

        gen = ProjectionDatasetGenerator(
            llm=llm,
            clip_encoder=MockCLIPEncoder(),
            minilm_encoder=MockMiniLMEncoder(),
            batch_size=10,
        )
        descriptions = gen._generate_descriptions(3)
        self.assertEqual(len(descriptions), 3)

    def test_failed_generation_tracked(self):
        llm = _make_llm_mock()
        llm.generate.side_effect = RuntimeError("API error")

        gen = ProjectionDatasetGenerator(
            llm=llm,
            clip_encoder=MockCLIPEncoder(),
            minilm_encoder=MockMiniLMEncoder(),
            batch_size=10,
        )
        descriptions = gen._generate_descriptions(5)
        # All batches failed
        self.assertEqual(len(descriptions), 0)


class TestEncodeBatch(unittest.TestCase):
    """Test embedding encoding with mock encoders."""

    def test_encode_produces_correct_dims(self):
        gen = ProjectionDatasetGenerator(
            llm=_make_llm_mock(),
            clip_encoder=MockCLIPEncoder(),
            minilm_encoder=MockMiniLMEncoder(),
        )
        descriptions = [
            ("A red car on a road", "vehicle"),
            ("A sunset over the ocean", "coastal"),
        ]
        pairs = gen._encode_batch(descriptions)
        self.assertEqual(len(pairs), 2)
        for pair in pairs:
            self.assertEqual(len(pair.clip_embedding), CLIP_EMBED_DIM)
            self.assertEqual(len(pair.minilm_embedding), MINILM_EMBED_DIM)

    def test_encode_preserves_category(self):
        gen = ProjectionDatasetGenerator(
            llm=_make_llm_mock(),
            clip_encoder=MockCLIPEncoder(),
            minilm_encoder=MockMiniLMEncoder(),
        )
        descriptions = [("A garden scene", "garden")]
        pairs = gen._encode_batch(descriptions)
        self.assertEqual(pairs[0].category, "garden")

    def test_encode_handles_encoder_error(self):
        clip = MockCLIPEncoder()
        clip.encode = MagicMock(side_effect=RuntimeError("CUDA OOM"))

        gen = ProjectionDatasetGenerator(
            llm=_make_llm_mock(),
            clip_encoder=clip,
            minilm_encoder=MockMiniLMEncoder(),
        )
        descriptions = [("A test description that is sufficiently long", "test")]
        pairs = gen._encode_batch(descriptions)
        self.assertEqual(len(pairs), 0)
        self.assertEqual(gen.stats.failed_encodings, 1)


class TestGeneratePairs(unittest.TestCase):
    """Test full pipeline generate_pairs."""

    def test_generate_small_batch(self):
        llm = _make_llm_mock()
        llm.generate.return_value = _make_batch_response(10)

        gen = ProjectionDatasetGenerator(
            llm=llm,
            clip_encoder=MockCLIPEncoder(),
            minilm_encoder=MockMiniLMEncoder(),
            batch_size=10,
        )
        pairs = gen.generate_pairs(count=5)
        self.assertEqual(len(pairs), 5)
        for pair in pairs:
            self.assertEqual(len(pair.clip_embedding), CLIP_EMBED_DIM)
            self.assertEqual(len(pair.minilm_embedding), MINILM_EMBED_DIM)

    def test_stats_updated(self):
        llm = _make_llm_mock()
        llm.generate.return_value = _make_batch_response(10)

        gen = ProjectionDatasetGenerator(
            llm=llm,
            clip_encoder=MockCLIPEncoder(),
            minilm_encoder=MockMiniLMEncoder(),
            batch_size=10,
        )
        gen.generate_pairs(count=5)
        stats = gen.stats
        self.assertEqual(stats.total_requested, 5)
        self.assertGreater(stats.descriptions_generated, 0)
        self.assertEqual(stats.pairs_encoded, 5)
        self.assertGreater(stats.elapsed_seconds, 0)


class TestExportAndLoad(unittest.TestCase):
    """Test JSONL export and load."""

    def test_export_creates_file(self):
        pairs = [
            EmbeddingPair(
                description=f"Description {i}",
                clip_embedding=[0.1] * CLIP_EMBED_DIM,
                minilm_embedding=[0.2] * MINILM_EMBED_DIM,
                category="test",
            )
            for i in range(5)
        ]

        gen = ProjectionDatasetGenerator(
            llm=_make_llm_mock(),
            clip_encoder=MockCLIPEncoder(),
            minilm_encoder=MockMiniLMEncoder(),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = gen.export(pairs, f"{tmpdir}/test.jsonl")
            self.assertTrue(path.exists())

            # Verify JSONL format
            with open(path) as f:
                lines = f.readlines()
            self.assertEqual(len(lines), 5)

            # Each line is valid JSON
            for line in lines:
                obj = json.loads(line)
                self.assertIn("description", obj)
                self.assertIn("clip_embedding", obj)
                self.assertIn("minilm_embedding", obj)
                self.assertEqual(len(obj["clip_embedding"]), CLIP_EMBED_DIM)
                self.assertEqual(len(obj["minilm_embedding"]), MINILM_EMBED_DIM)

    def test_load_roundtrip(self):
        pairs = [
            EmbeddingPair(
                description=f"Scene {i}",
                clip_embedding=[float(i)] * CLIP_EMBED_DIM,
                minilm_embedding=[float(i + 1)] * MINILM_EMBED_DIM,
                category=f"cat_{i}",
            )
            for i in range(3)
        ]

        gen = ProjectionDatasetGenerator(
            llm=_make_llm_mock(),
            clip_encoder=MockCLIPEncoder(),
            minilm_encoder=MockMiniLMEncoder(),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = gen.export(pairs, f"{tmpdir}/test.jsonl")
            loaded = ProjectionDatasetGenerator.load(path)

            self.assertEqual(len(loaded), 3)
            for orig, restored in zip(pairs, loaded):
                self.assertEqual(orig.description, restored.description)
                self.assertEqual(orig.category, restored.category)
                self.assertEqual(
                    len(restored.clip_embedding), CLIP_EMBED_DIM
                )
                self.assertEqual(
                    len(restored.minilm_embedding), MINILM_EMBED_DIM
                )

    def test_export_creates_parent_dirs(self):
        gen = ProjectionDatasetGenerator(
            llm=_make_llm_mock(),
            clip_encoder=MockCLIPEncoder(),
            minilm_encoder=MockMiniLMEncoder(),
        )
        pairs = [
            EmbeddingPair(
                description="test",
                clip_embedding=[0.0] * CLIP_EMBED_DIM,
                minilm_embedding=[0.0] * MINILM_EMBED_DIM,
            )
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = gen.export(pairs, f"{tmpdir}/nested/dir/output.jsonl")
            self.assertTrue(path.exists())


class TestMockEncoders(unittest.TestCase):
    """Verify mock encoders produce correct dimensions."""

    def test_clip_mock_dim(self):
        enc = MockCLIPEncoder()
        self.assertEqual(enc.dim, CLIP_EMBED_DIM)

    def test_clip_mock_encode(self):
        enc = MockCLIPEncoder()
        embs = enc.encode(["test text"])
        self.assertEqual(len(embs), 1)
        self.assertEqual(len(embs[0]), CLIP_EMBED_DIM)

    def test_minilm_mock_dim(self):
        enc = MockMiniLMEncoder()
        self.assertEqual(enc.dim, MINILM_EMBED_DIM)

    def test_minilm_mock_encode(self):
        enc = MockMiniLMEncoder()
        embs = enc.encode(["test text"])
        self.assertEqual(len(embs), 1)
        self.assertEqual(len(embs[0]), MINILM_EMBED_DIM)

    def test_mock_embeddings_normalized(self):
        enc = MockCLIPEncoder()
        embs = enc.encode(["test"])
        norm = sum(x * x for x in embs[0]) ** 0.5
        self.assertAlmostEqual(norm, 1.0, places=5)


class TestEndToEnd(unittest.TestCase):
    """Full pipeline test with mocked components."""

    def test_generate_and_export(self):
        llm = _make_llm_mock()
        llm.generate.return_value = _make_batch_response(10)

        gen = ProjectionDatasetGenerator(
            llm=llm,
            clip_encoder=MockCLIPEncoder(),
            minilm_encoder=MockMiniLMEncoder(),
            batch_size=10,
            seed=42,
        )

        pairs = gen.generate_pairs(count=5)
        self.assertEqual(len(pairs), 5)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = gen.export(pairs, f"{tmpdir}/output.jsonl")
            loaded = ProjectionDatasetGenerator.load(path)
            self.assertEqual(len(loaded), 5)

            # Verify dimensions are correct
            for pair in loaded:
                self.assertEqual(len(pair.clip_embedding), CLIP_EMBED_DIM)
                self.assertEqual(len(pair.minilm_embedding), MINILM_EMBED_DIM)
                self.assertGreater(len(pair.description), 0)


if __name__ == "__main__":
    unittest.main()
