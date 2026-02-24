#!/usr/bin/env python3
"""Tests for ContradictionGenerator — contradiction detection dataset generation."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

# Ensure env var is set before importing (LLMDatasetGenerator requires it)
os.environ.setdefault("FIREWORKS_API_KEY", "fw-test-key-not-real")

from contradiction_generator import (
    BASE_TIMESTAMP,
    ContradictionGenerator,
    ContradictionSample,
    ENTITY_NAMES,
    FactStatement,
    LABEL_AGREES,
    LABEL_AMBIGUOUS,
    LABEL_CONTRADICTS,
    LABEL_SUPERSEDES,
    LABELS,
    LABEL_WEIGHTS,
    ONE_YEAR_SECS,
    PREDICATES,
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

    def test_predicates_non_empty(self):
        self.assertGreaterEqual(len(PREDICATES), 10)

    def test_predicates_have_required_fields(self):
        for pred in PREDICATES:
            self.assertIn("predicate", pred)
            self.assertIn("subject_type", pred)
            self.assertIn("description", pred)
            self.assertIn("value_type", pred)

    def test_predicates_subject_types_in_entity_names(self):
        for pred in PREDICATES:
            self.assertIn(
                pred["subject_type"],
                ENTITY_NAMES,
                f"Predicate '{pred['predicate']}' uses subject_type '{pred['subject_type']}' "
                f"not found in ENTITY_NAMES",
            )

    def test_labels_match_weights(self):
        self.assertEqual(len(LABELS), len(LABEL_WEIGHTS))
        self.assertAlmostEqual(sum(LABEL_WEIGHTS), 1.0, places=5)

    def test_label_constants(self):
        self.assertEqual(LABEL_SUPERSEDES, "supersedes")
        self.assertEqual(LABEL_CONTRADICTS, "contradicts")
        self.assertEqual(LABEL_AMBIGUOUS, "ambiguous")
        self.assertEqual(LABEL_AGREES, "agrees")


class TestFactStatement(unittest.TestCase):
    """Test FactStatement dataclass."""

    def test_construction(self):
        fact = FactStatement(
            subject="Alice",
            predicate="lives_in",
            object_value="Madrid",
            text="Alice lives in Madrid.",
            confidence=0.9,
            timestamp=1_000_000,
        )
        self.assertEqual(fact.subject, "Alice")
        self.assertEqual(fact.predicate, "lives_in")
        self.assertEqual(fact.object_value, "Madrid")
        self.assertEqual(fact.confidence, 0.9)


class TestContradictionSample(unittest.TestCase):
    """Test ContradictionSample dataclass."""

    def _make_sample(self) -> ContradictionSample:
        fact_a = FactStatement("Alice", "lives_in", "Madrid",
                               "Alice lives in Madrid.", 0.9, 1000)
        fact_b = FactStatement("Alice", "lives_in", "Berlin",
                               "Alice lives in Berlin.", 0.8, 2_000_000_000)
        return ContradictionSample(
            fact_a=fact_a, fact_b=fact_b,
            label=LABEL_SUPERSEDES,
            resolution_strategy="temporal",
            metadata={"subject_type": "Person"},
        )

    def test_to_dict_has_all_fields(self):
        sample = self._make_sample()
        d = sample.to_dict()
        self.assertIn("fact_a", d)
        self.assertIn("fact_b", d)
        self.assertIn("label", d)
        self.assertIn("resolution_strategy", d)
        self.assertIn("metadata", d)

    def test_to_dict_fact_fields(self):
        sample = self._make_sample()
        d = sample.to_dict()
        for fact_key in ("fact_a", "fact_b"):
            fact = d[fact_key]
            self.assertIn("subject", fact)
            self.assertIn("predicate", fact)
            self.assertIn("object", fact)
            self.assertIn("text", fact)
            self.assertIn("confidence", fact)
            self.assertIn("timestamp", fact)

    def test_to_dict_values(self):
        sample = self._make_sample()
        d = sample.to_dict()
        self.assertEqual(d["fact_a"]["subject"], "Alice")
        self.assertEqual(d["fact_a"]["object"], "Madrid")
        self.assertEqual(d["fact_b"]["object"], "Berlin")
        self.assertEqual(d["label"], "supersedes")
        self.assertEqual(d["resolution_strategy"], "temporal")

    def test_to_dict_serializable(self):
        sample = self._make_sample()
        serialized = json.dumps(sample.to_dict())
        parsed = json.loads(serialized)
        self.assertEqual(parsed["label"], "supersedes")

    def test_default_metadata_empty(self):
        fact_a = FactStatement("X", "p", "v", "X p v.", 0.5, 100)
        fact_b = FactStatement("X", "p", "w", "X p w.", 0.5, 200)
        sample = ContradictionSample(
            fact_a=fact_a, fact_b=fact_b,
            label=LABEL_CONTRADICTS,
            resolution_strategy="confidence",
        )
        self.assertEqual(sample.metadata, {})


class TestContradictionGenerator(unittest.TestCase):
    """Test ContradictionGenerator methods."""

    def setUp(self):
        self.llm_mock = _make_llm_mock()
        self.gen = ContradictionGenerator(llm=self.llm_mock, seed=42)

    def test_sample_predicate(self):
        pred = self.gen._sample_predicate()
        self.assertIn("predicate", pred)
        self.assertIn("subject_type", pred)

    def test_sample_subject_person(self):
        subject = self.gen._sample_subject("Person")
        self.assertIn(subject, ENTITY_NAMES["Person"])

    def test_sample_subject_unknown_type_falls_back(self):
        subject = self.gen._sample_subject("Unknown")
        self.assertIn(subject, ENTITY_NAMES["Person"])

    def test_sample_label_in_valid_labels(self):
        for _ in range(100):
            label = self.gen._sample_label()
            self.assertIn(label, LABELS)

    def test_deterministic_with_seed(self):
        gen1 = ContradictionGenerator(llm=self.llm_mock, seed=123)
        gen2 = ContradictionGenerator(llm=self.llm_mock, seed=123)
        labels1 = [gen1._sample_label() for _ in range(20)]
        labels2 = [gen2._sample_label() for _ in range(20)]
        self.assertEqual(labels1, labels2)

    def test_resolution_strategy_mapping(self):
        self.assertEqual(self.gen._resolution_strategy(LABEL_SUPERSEDES), "temporal")
        self.assertEqual(self.gen._resolution_strategy(LABEL_CONTRADICTS), "confidence")
        self.assertEqual(self.gen._resolution_strategy(LABEL_AMBIGUOUS), "ambiguous")
        self.assertEqual(self.gen._resolution_strategy(LABEL_AGREES), "none")


class TestTimestampAndConfidenceGeneration(unittest.TestCase):
    """Test timestamp and confidence pair generation for each label."""

    def setUp(self):
        self.llm_mock = _make_llm_mock()
        self.gen = ContradictionGenerator(llm=self.llm_mock, seed=42)

    def test_supersedes_has_large_time_gap(self):
        for _ in range(50):
            ts_a, ts_b, conf_a, conf_b = self.gen._generate_timestamps_and_confidence(
                LABEL_SUPERSEDES
            )
            self.assertGreater(abs(ts_b - ts_a), ONE_YEAR_SECS)

    def test_contradicts_has_close_timestamps_high_conf_gap(self):
        for _ in range(50):
            ts_a, ts_b, conf_a, conf_b = self.gen._generate_timestamps_and_confidence(
                LABEL_CONTRADICTS
            )
            # Timestamps within half a year
            self.assertLessEqual(abs(ts_b - ts_a), ONE_YEAR_SECS // 2)
            # Confidence gap > 0.3
            self.assertGreater(abs(conf_a - conf_b), 0.3)

    def test_ambiguous_has_close_timestamps_and_confidence(self):
        for _ in range(50):
            ts_a, ts_b, conf_a, conf_b = self.gen._generate_timestamps_and_confidence(
                LABEL_AMBIGUOUS
            )
            # Timestamps within quarter year
            self.assertLessEqual(abs(ts_b - ts_a), ONE_YEAR_SECS // 4)
            # Confidence gap <= 0.3 (ambiguous = close)
            self.assertLessEqual(abs(conf_a - conf_b), 0.3 + 0.01)  # small epsilon

    def test_confidence_always_in_valid_range(self):
        for label in LABELS:
            for _ in range(20):
                _, _, conf_a, conf_b = self.gen._generate_timestamps_and_confidence(label)
                self.assertGreaterEqual(conf_a, 0.0)
                self.assertLessEqual(conf_a, 1.0)
                self.assertGreaterEqual(conf_b, 0.0)
                self.assertLessEqual(conf_b, 1.0)


class TestFactGeneration(unittest.TestCase):
    """Test LLM-backed fact generation."""

    def setUp(self):
        self.llm_mock = _make_llm_mock()
        self.gen = ContradictionGenerator(llm=self.llm_mock, seed=42)

    def test_generate_original_fact_success(self):
        self.llm_mock.generate.return_value = _make_generation_result(
            "Alice Johnson works at Nexus Technologies as a senior engineer."
        )
        pred_info = PREDICATES[0]  # lives_in
        result = self.gen.generate_original_fact("Alice Johnson", pred_info)
        self.assertIsNotNone(result)
        self.llm_mock.generate.assert_called_once()

    def test_generate_original_fact_empty_returns_none(self):
        self.llm_mock.generate.return_value = _make_generation_result("")
        result = self.gen.generate_original_fact("Alice", PREDICATES[0])
        self.assertIsNone(result)

    def test_generate_original_fact_short_returns_none(self):
        self.llm_mock.generate.return_value = _make_generation_result("Hi")
        result = self.gen.generate_original_fact("Alice", PREDICATES[0])
        self.assertIsNone(result)

    def test_generate_original_fact_exception_returns_none(self):
        self.llm_mock.generate.side_effect = RuntimeError("API error")
        result = self.gen.generate_original_fact("Alice", PREDICATES[0])
        self.assertIsNone(result)

    def test_generate_contradiction_success(self):
        self.llm_mock.generate.return_value = _make_generation_result(
            "Alice Johnson now lives in Berlin, Germany."
        )
        result = self.gen.generate_contradiction(
            "Alice Johnson lives in Madrid.", "Alice Johnson", PREDICATES[0]
        )
        self.assertIsNotNone(result)
        self.assertIn("Berlin", result)

    def test_generate_contradiction_strips_quotes(self):
        self.llm_mock.generate.return_value = _make_generation_result(
            '"Alice now resides in Tokyo."'
        )
        result = self.gen.generate_contradiction(
            "Alice lives in Madrid.", "Alice", PREDICATES[0]
        )
        self.assertIsNotNone(result)
        self.assertFalse(result.startswith('"'))

    def test_generate_contradiction_exception_returns_none(self):
        self.llm_mock.generate.side_effect = RuntimeError("API error")
        result = self.gen.generate_contradiction(
            "Alice lives in Madrid.", "Alice", PREDICATES[0]
        )
        self.assertIsNone(result)

    def test_generate_agreement_success(self):
        self.llm_mock.generate.return_value = _make_generation_result(
            "Alice Johnson resides in Madrid, Spain."
        )
        result = self.gen.generate_agreement(
            "Alice Johnson lives in Madrid.", "Alice Johnson", PREDICATES[0]
        )
        self.assertIsNotNone(result)

    def test_generate_agreement_exception_returns_none(self):
        self.llm_mock.generate.side_effect = RuntimeError("API error")
        result = self.gen.generate_agreement(
            "Alice lives in Madrid.", "Alice", PREDICATES[0]
        )
        self.assertIsNone(result)


class TestObjectExtraction(unittest.TestCase):
    """Test object value extraction from generated text."""

    def setUp(self):
        self.llm_mock = _make_llm_mock()
        self.gen = ContradictionGenerator(llm=self.llm_mock, seed=42)

    def test_extracts_known_location(self):
        pred = {"predicate": "lives_in", "value_type": "Location", "description": "city"}
        obj = self.gen._extract_object_from_text(
            "Alice lives in Madrid, Spain.", "Alice", pred
        )
        self.assertEqual(obj, "Madrid")

    def test_extracts_known_organization(self):
        pred = {"predicate": "works_at", "value_type": "Organization", "description": "employer"}
        obj = self.gen._extract_object_from_text(
            "Alice works at Nexus Technologies as an engineer.", "Alice", pred
        )
        self.assertEqual(obj, "Nexus Technologies")

    def test_extracts_known_person(self):
        pred = {"predicate": "ceo", "value_type": "Person", "description": "CEO"}
        obj = self.gen._extract_object_from_text(
            "Nexus Technologies is led by James Chen as CEO.", "Nexus Technologies", pred
        )
        self.assertEqual(obj, "James Chen")

    def test_person_extraction_excludes_subject(self):
        pred = {"predicate": "ceo", "value_type": "Person", "description": "CEO"}
        # Subject is also a known person name — should find a different person
        obj = self.gen._extract_object_from_text(
            "Alice Johnson appointed Raj Patel as new CEO.", "Alice Johnson", pred
        )
        self.assertEqual(obj, "Raj Patel")

    def test_fallback_to_text_for_free_text(self):
        pred = {"predicate": "job_title", "value_type": "free_text", "description": "role"}
        obj = self.gen._extract_object_from_text(
            "Alice is a senior software engineer.", "Alice", pred
        )
        # Falls back to full text (trimmed)
        self.assertIn("Alice", obj)

    def test_fallback_truncates_long_text(self):
        pred = {"predicate": "job_title", "value_type": "free_text", "description": "role"}
        long_text = "A" * 200
        obj = self.gen._extract_object_from_text(long_text, "Alice", pred)
        self.assertLessEqual(len(obj), 100)


class TestGenerateSample(unittest.TestCase):
    """Test end-to-end sample generation."""

    def setUp(self):
        self.llm_mock = _make_llm_mock()
        self.gen = ContradictionGenerator(llm=self.llm_mock, seed=42)

    def test_generate_sample_success(self):
        self.llm_mock.generate.side_effect = [
            _make_generation_result("Alice Johnson lives in Madrid."),
            _make_generation_result("Alice Johnson now resides in Berlin."),
        ]
        sample = self.gen.generate_sample()
        self.assertIsNotNone(sample)
        self.assertIn(sample.label, LABELS)
        self.assertEqual(sample.fact_a.subject, sample.fact_b.subject)
        self.assertEqual(sample.fact_a.predicate, sample.fact_b.predicate)

    def test_generate_sample_first_fact_fails(self):
        self.llm_mock.generate.return_value = _make_generation_result("")
        sample = self.gen.generate_sample()
        self.assertIsNone(sample)

    def test_generate_sample_second_fact_fails(self):
        self.llm_mock.generate.side_effect = [
            _make_generation_result("Alice lives in Madrid."),
            _make_generation_result(""),  # second fact fails
        ]
        sample = self.gen.generate_sample()
        self.assertIsNone(sample)

    def test_generate_sample_metadata_populated(self):
        self.llm_mock.generate.side_effect = [
            _make_generation_result("Carlos García works at Nexus Technologies."),
            _make_generation_result("Carlos García works at Pacific Dynamics."),
        ]
        sample = self.gen.generate_sample()
        self.assertIsNotNone(sample)
        self.assertIn("subject_type", sample.metadata)
        self.assertIn("predicate", sample.metadata)
        self.assertIn("value_type", sample.metadata)

    def test_agreement_label_calls_generate_agreement(self):
        """When label is 'agrees', should call generate_agreement not generate_contradiction."""
        # Force the label to be 'agrees' by mocking _sample_label
        self.gen._sample_label = lambda: LABEL_AGREES
        self.llm_mock.generate.side_effect = [
            _make_generation_result("Alice Johnson lives in Madrid."),
            _make_generation_result("Alice Johnson resides in Madrid, Spain."),
        ]
        sample = self.gen.generate_sample()
        self.assertIsNotNone(sample)
        self.assertEqual(sample.label, LABEL_AGREES)
        self.assertEqual(sample.resolution_strategy, "none")


class TestBatchGeneration(unittest.TestCase):
    """Test batch sample generation."""

    def setUp(self):
        self.llm_mock = _make_llm_mock()
        self.gen = ContradictionGenerator(llm=self.llm_mock, seed=42)

    def test_generate_samples_returns_requested_count(self):
        self.llm_mock.generate.return_value = _make_generation_result(
            "Alice Johnson lives in Madrid, Spain."
        )
        samples = self.gen.generate_samples(count=5, progress_interval=10)
        self.assertEqual(len(samples), 5)

    def test_generate_samples_handles_failures(self):
        call_count = 0

        def _mock_generate(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            # Fail every 3rd call
            if call_count % 3 == 0:
                return _make_generation_result("")
            return _make_generation_result("A valid fact about someone.")

        self.llm_mock.generate.side_effect = _mock_generate
        samples = self.gen.generate_samples(count=5, progress_interval=100)
        # Should still get 5 despite some failures
        self.assertEqual(len(samples), 5)

    def test_generate_samples_respects_max_attempts(self):
        # All calls fail
        self.llm_mock.generate.return_value = _make_generation_result("")
        samples = self.gen.generate_samples(count=10, progress_interval=100)
        # Should give up after 2x attempts (20) and return 0
        self.assertEqual(len(samples), 0)


class TestExport(unittest.TestCase):
    """Test JSONL export."""

    def setUp(self):
        self.llm_mock = _make_llm_mock()
        self.gen = ContradictionGenerator(llm=self.llm_mock, seed=42)

    def test_export_writes_jsonl(self):
        fact_a = FactStatement("Alice", "lives_in", "Madrid",
                               "Alice lives in Madrid.", 0.9, 1000)
        fact_b = FactStatement("Alice", "lives_in", "Berlin",
                               "Alice lives in Berlin.", 0.8, 2_000_000_000)
        sample = ContradictionSample(
            fact_a=fact_a, fact_b=fact_b,
            label=LABEL_SUPERSEDES,
            resolution_strategy="temporal",
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            path = f.name

        try:
            written = self.gen.export([sample], path)
            self.assertEqual(written, 1)
            self.llm_mock.export_jsonl.assert_called_once()
            # Verify the data passed to export_jsonl
            call_args = self.llm_mock.export_jsonl.call_args
            records = call_args[0][0]
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["label"], "supersedes")
        finally:
            os.unlink(path)

    def test_export_multiple_samples(self):
        samples = []
        for i in range(3):
            fact_a = FactStatement("X", "p", f"v{i}", f"X p v{i}.", 0.5, 100 + i)
            fact_b = FactStatement("X", "p", f"w{i}", f"X p w{i}.", 0.5, 200 + i)
            samples.append(ContradictionSample(
                fact_a=fact_a, fact_b=fact_b,
                label=LABEL_CONTRADICTS,
                resolution_strategy="confidence",
            ))

        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name

        try:
            written = self.gen.export(samples, path)
            self.assertEqual(written, 3)
        finally:
            os.unlink(path)


class TestLabelDistribution(unittest.TestCase):
    """Test label distribution analysis."""

    def setUp(self):
        self.llm_mock = _make_llm_mock()
        self.gen = ContradictionGenerator(llm=self.llm_mock, seed=42)

    def test_label_distribution_counts(self):
        fact = FactStatement("X", "p", "v", "text", 0.5, 100)
        samples = [
            ContradictionSample(fact, fact, LABEL_SUPERSEDES, "temporal"),
            ContradictionSample(fact, fact, LABEL_SUPERSEDES, "temporal"),
            ContradictionSample(fact, fact, LABEL_CONTRADICTS, "confidence"),
            ContradictionSample(fact, fact, LABEL_AMBIGUOUS, "ambiguous"),
        ]
        dist = self.gen.label_distribution(samples)
        self.assertEqual(dist[LABEL_SUPERSEDES], 2)
        self.assertEqual(dist[LABEL_CONTRADICTS], 1)
        self.assertEqual(dist[LABEL_AMBIGUOUS], 1)
        self.assertEqual(dist[LABEL_AGREES], 0)

    def test_label_distribution_empty(self):
        dist = self.gen.label_distribution([])
        for label in LABELS:
            self.assertEqual(dist[label], 0)


if __name__ == "__main__":
    unittest.main()
