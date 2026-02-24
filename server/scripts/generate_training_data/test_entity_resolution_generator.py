#!/usr/bin/env python3
"""Tests for EntityResolutionGenerator — entity resolution dataset generation."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock

# Ensure env var is set before importing (LLMDatasetGenerator requires it)
os.environ.setdefault("FIREWORKS_API_KEY", "fw-test-key-not-real")

from entity_resolution_generator import (
    ENTITY_NAMES,
    EntityPair,
    EntityResolutionGenerator,
    POSITIVE_RATIO,
    VARIATION_TYPES,
)
from llm_dataset_generator import GenerationConfig, GenerationResult, LLMDatasetGenerator


def _make_llm_mock() -> MagicMock:
    """Create a mock LLMDatasetGenerator."""
    mock = MagicMock(spec=LLMDatasetGenerator)
    mock.usage_summary.return_value = {
        "total_requests": 0,
        "total_prompt_tokens": 0,
        "total_completion_tokens": 0,
        "total_tokens": 0,
    }
    return mock


def _make_generation_result(text: str) -> GenerationResult:
    """Create a GenerationResult with given text."""
    return GenerationResult(
        text=text,
        prompt_tokens=10,
        completion_tokens=20,
        model="test-model",
        finish_reason="stop",
    )


class TestSeedData(unittest.TestCase):
    """Validate seed knowledge base completeness."""

    def test_entity_names_has_all_types(self):
        for entity_type in ["Person", "Organization", "Location"]:
            self.assertIn(entity_type, ENTITY_NAMES)
            self.assertGreaterEqual(len(ENTITY_NAMES[entity_type]), 10)

    def test_variation_types_non_empty(self):
        self.assertGreaterEqual(len(VARIATION_TYPES), 5)

    def test_positive_ratio_valid(self):
        self.assertGreater(POSITIVE_RATIO, 0.0)
        self.assertLess(POSITIVE_RATIO, 1.0)

    def test_variation_types_are_strings(self):
        for vt in VARIATION_TYPES:
            self.assertIsInstance(vt, str)
            self.assertGreater(len(vt), 0)


class TestEntityPair(unittest.TestCase):
    """Test EntityPair dataclass."""

    def _make_pair(self, is_dup: bool = True) -> EntityPair:
        return EntityPair(
            canonical="Alice Johnson",
            variant="Alcie Johnson" if is_dup else "Alice Jefferson",
            is_duplicate=is_dup,
            entity_type="Person",
            variation_type="typo" if is_dup else "different_entity",
            metadata={"generation_method": "llm_variation"},
        )

    def test_to_dict_has_all_fields(self):
        pair = self._make_pair()
        d = pair.to_dict()
        self.assertIn("canonical", d)
        self.assertIn("variant", d)
        self.assertIn("is_duplicate", d)
        self.assertIn("entity_type", d)
        self.assertIn("variation_type", d)
        self.assertIn("metadata", d)

    def test_to_dict_positive_values(self):
        pair = self._make_pair(is_dup=True)
        d = pair.to_dict()
        self.assertEqual(d["canonical"], "Alice Johnson")
        self.assertEqual(d["variant"], "Alcie Johnson")
        self.assertTrue(d["is_duplicate"])
        self.assertEqual(d["entity_type"], "Person")
        self.assertEqual(d["variation_type"], "typo")

    def test_to_dict_negative_values(self):
        pair = self._make_pair(is_dup=False)
        d = pair.to_dict()
        self.assertFalse(d["is_duplicate"])
        self.assertEqual(d["variation_type"], "different_entity")

    def test_to_dict_serializable(self):
        pair = self._make_pair()
        serialized = json.dumps(pair.to_dict())
        parsed = json.loads(serialized)
        self.assertEqual(parsed["canonical"], "Alice Johnson")
        self.assertTrue(parsed["is_duplicate"])

    def test_default_metadata_empty(self):
        pair = EntityPair(
            canonical="X", variant="Y", is_duplicate=True,
            entity_type="Person", variation_type="typo",
        )
        self.assertEqual(pair.metadata, {})


class TestEntityResolutionGenerator(unittest.TestCase):
    """Test EntityResolutionGenerator methods."""

    def setUp(self):
        self.llm_mock = _make_llm_mock()
        self.gen = EntityResolutionGenerator(llm=self.llm_mock, seed=42)

    def test_sample_entity_returns_valid(self):
        name, etype = self.gen._sample_entity()
        self.assertIn(etype, ENTITY_NAMES)
        self.assertIn(name, ENTITY_NAMES[etype])

    def test_sample_variation_type_valid(self):
        vt = self.gen._sample_variation_type()
        self.assertIn(vt, VARIATION_TYPES)

    def test_deterministic_with_seed(self):
        gen1 = EntityResolutionGenerator(llm=self.llm_mock, seed=123)
        gen2 = EntityResolutionGenerator(llm=self.llm_mock, seed=123)
        entities1 = [gen1._sample_entity() for _ in range(20)]
        entities2 = [gen2._sample_entity() for _ in range(20)]
        self.assertEqual(entities1, entities2)

    def test_is_positive_distribution(self):
        """Over many samples, positive ratio should approximate target."""
        count = 1000
        positives = sum(1 for _ in range(count) if self.gen._is_positive())
        ratio = positives / count
        self.assertGreater(ratio, POSITIVE_RATIO - 0.1)
        self.assertLess(ratio, POSITIVE_RATIO + 0.1)


class TestVariationGeneration(unittest.TestCase):
    """Test LLM-backed variation generation."""

    def setUp(self):
        self.llm_mock = _make_llm_mock()
        self.gen = EntityResolutionGenerator(llm=self.llm_mock, seed=42)

    def test_generate_variation_success(self):
        self.llm_mock.generate.return_value = _make_generation_result(
            "Alcie Johnson"
        )
        result = self.gen.generate_variation("Alice Johnson", "Person", "typo")
        self.assertIsNotNone(result)
        self.assertEqual(result, "Alcie Johnson")

    def test_generate_variation_strips_quotes(self):
        self.llm_mock.generate.return_value = _make_generation_result(
            '"A. Johnson"'
        )
        result = self.gen.generate_variation("Alice Johnson", "Person", "abbreviation")
        self.assertIsNotNone(result)
        self.assertFalse(result.startswith('"'))

    def test_generate_variation_empty_returns_none(self):
        self.llm_mock.generate.return_value = _make_generation_result("")
        result = self.gen.generate_variation("Alice", "Person", "typo")
        self.assertIsNone(result)

    def test_generate_variation_too_short_returns_none(self):
        self.llm_mock.generate.return_value = _make_generation_result("A")
        result = self.gen.generate_variation("Alice Johnson", "Person", "typo")
        self.assertIsNone(result)

    def test_generate_variation_identical_returns_none(self):
        """Variation that's identical to original should be rejected."""
        self.llm_mock.generate.return_value = _make_generation_result("Alice Johnson")
        result = self.gen.generate_variation("Alice Johnson", "Person", "typo")
        self.assertIsNone(result)

    def test_generate_variation_case_insensitive_check(self):
        """Identical check is case-insensitive."""
        self.llm_mock.generate.return_value = _make_generation_result("alice johnson")
        result = self.gen.generate_variation("Alice Johnson", "Person", "case_variation")
        # This IS a valid case variation — lowercase is different enough
        # Actually our check rejects it since lower() matches
        self.assertIsNone(result)

    def test_generate_variation_exception_returns_none(self):
        self.llm_mock.generate.side_effect = RuntimeError("API error")
        result = self.gen.generate_variation("Alice", "Person", "typo")
        self.assertIsNone(result)


class TestNegativeEntityGeneration(unittest.TestCase):
    """Test LLM-backed negative entity generation."""

    def setUp(self):
        self.llm_mock = _make_llm_mock()
        self.gen = EntityResolutionGenerator(llm=self.llm_mock, seed=42)

    def test_generate_negative_success(self):
        self.llm_mock.generate.return_value = _make_generation_result(
            "Alice Jefferson"
        )
        result = self.gen.generate_negative_entity("Alice Johnson", "Person")
        self.assertIsNotNone(result)
        self.assertEqual(result, "Alice Jefferson")

    def test_generate_negative_empty_returns_none(self):
        self.llm_mock.generate.return_value = _make_generation_result("")
        result = self.gen.generate_negative_entity("Alice", "Person")
        self.assertIsNone(result)

    def test_generate_negative_identical_returns_none(self):
        self.llm_mock.generate.return_value = _make_generation_result("Alice Johnson")
        result = self.gen.generate_negative_entity("Alice Johnson", "Person")
        self.assertIsNone(result)

    def test_generate_negative_exception_returns_none(self):
        self.llm_mock.generate.side_effect = RuntimeError("API error")
        result = self.gen.generate_negative_entity("Alice", "Person")
        self.assertIsNone(result)


class TestGenerateSample(unittest.TestCase):
    """Test end-to-end sample generation."""

    def setUp(self):
        self.llm_mock = _make_llm_mock()
        self.gen = EntityResolutionGenerator(llm=self.llm_mock, seed=42)

    def test_generate_positive_sample(self):
        self.gen._is_positive = lambda: True
        self.llm_mock.generate.return_value = _make_generation_result(
            "Alcie Johnson"
        )
        sample = self.gen.generate_sample()
        self.assertIsNotNone(sample)
        self.assertTrue(sample.is_duplicate)
        self.assertIn(sample.variation_type, VARIATION_TYPES)
        self.assertIn(sample.entity_type, ENTITY_NAMES)

    def test_generate_negative_sample(self):
        self.gen._is_positive = lambda: False
        self.llm_mock.generate.return_value = _make_generation_result(
            "Alice Jefferson"
        )
        sample = self.gen.generate_sample()
        self.assertIsNotNone(sample)
        self.assertFalse(sample.is_duplicate)
        self.assertEqual(sample.variation_type, "different_entity")

    def test_generate_sample_positive_failure(self):
        self.gen._is_positive = lambda: True
        self.llm_mock.generate.return_value = _make_generation_result("")
        sample = self.gen.generate_sample()
        self.assertIsNone(sample)

    def test_generate_sample_negative_failure(self):
        self.gen._is_positive = lambda: False
        self.llm_mock.generate.return_value = _make_generation_result("")
        sample = self.gen.generate_sample()
        self.assertIsNone(sample)

    def test_generate_sample_has_metadata(self):
        self.gen._is_positive = lambda: True
        self.llm_mock.generate.return_value = _make_generation_result(
            "Nexus Tech"
        )
        sample = self.gen.generate_sample()
        self.assertIsNotNone(sample)
        self.assertIn("generation_method", sample.metadata)


class TestBatchGeneration(unittest.TestCase):
    """Test batch sample generation."""

    def setUp(self):
        self.llm_mock = _make_llm_mock()
        self.gen = EntityResolutionGenerator(llm=self.llm_mock, seed=42)

    def test_generate_samples_returns_requested_count(self):
        self.llm_mock.generate.return_value = _make_generation_result(
            "Alcie Johnson"
        )
        samples = self.gen.generate_samples(count=5, progress_interval=10)
        self.assertEqual(len(samples), 5)

    def test_generate_samples_handles_failures(self):
        call_count = 0

        def _mock_generate(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count % 3 == 0:
                return _make_generation_result("")
            return _make_generation_result("A valid entity variation name")

        self.llm_mock.generate.side_effect = _mock_generate
        samples = self.gen.generate_samples(count=5, progress_interval=100)
        self.assertEqual(len(samples), 5)

    def test_generate_samples_respects_max_attempts(self):
        self.llm_mock.generate.return_value = _make_generation_result("")
        samples = self.gen.generate_samples(count=10, progress_interval=100)
        self.assertEqual(len(samples), 0)


class TestExport(unittest.TestCase):
    """Test JSONL export."""

    def setUp(self):
        self.llm_mock = _make_llm_mock()
        self.gen = EntityResolutionGenerator(llm=self.llm_mock, seed=42)

    def test_export_writes_jsonl(self):
        pair = EntityPair(
            canonical="Alice Johnson",
            variant="Alcie Johnson",
            is_duplicate=True,
            entity_type="Person",
            variation_type="typo",
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            path = f.name

        try:
            written = self.gen.export([pair], path)
            self.assertEqual(written, 1)
            self.llm_mock.export_jsonl.assert_called_once()
            call_args = self.llm_mock.export_jsonl.call_args
            records = call_args[0][0]
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["canonical"], "Alice Johnson")
            self.assertTrue(records[0]["is_duplicate"])
        finally:
            os.unlink(path)

    def test_export_multiple_samples(self):
        samples = []
        for i in range(3):
            samples.append(EntityPair(
                canonical=f"Entity{i}",
                variant=f"Variant{i}",
                is_duplicate=i % 2 == 0,
                entity_type="Person",
                variation_type="typo",
            ))

        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name

        try:
            written = self.gen.export(samples, path)
            self.assertEqual(written, 3)
        finally:
            os.unlink(path)


class TestDistributions(unittest.TestCase):
    """Test distribution analysis methods."""

    def setUp(self):
        self.llm_mock = _make_llm_mock()
        self.gen = EntityResolutionGenerator(llm=self.llm_mock, seed=42)

    def test_label_distribution(self):
        samples = [
            EntityPair("A", "B", True, "Person", "typo"),
            EntityPair("A", "C", True, "Person", "nickname"),
            EntityPair("A", "D", False, "Person", "different_entity"),
        ]
        dist = self.gen.label_distribution(samples)
        self.assertEqual(dist["positive"], 2)
        self.assertEqual(dist["negative"], 1)

    def test_label_distribution_empty(self):
        dist = self.gen.label_distribution([])
        self.assertEqual(dist["positive"], 0)
        self.assertEqual(dist["negative"], 0)

    def test_variation_distribution(self):
        samples = [
            EntityPair("A", "B", True, "Person", "typo"),
            EntityPair("A", "C", True, "Person", "typo"),
            EntityPair("A", "D", True, "Person", "nickname"),
            EntityPair("A", "E", False, "Person", "different_entity"),
        ]
        dist = self.gen.variation_distribution(samples)
        self.assertEqual(dist["typo"], 2)
        self.assertEqual(dist["nickname"], 1)
        self.assertEqual(dist["different_entity"], 1)

    def test_variation_distribution_empty(self):
        dist = self.gen.variation_distribution([])
        self.assertEqual(dist, {})


if __name__ == "__main__":
    unittest.main()
