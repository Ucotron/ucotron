#!/usr/bin/env python3
"""Unit tests for REDatasetGenerator (mocked, no real API calls)."""

import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch, call

# Set a dummy key before importing
os.environ.setdefault("FIREWORKS_API_KEY", "fw-test-key-not-real")

from llm_dataset_generator import LLMDatasetGenerator, GenerationConfig, GenerationResult
from re_dataset_generator import (
    REDatasetGenerator,
    RESample,
    EntitySpan,
    Relation,
    SEED_ENTITIES,
    RELATION_TEMPLATES,
    RELATION_TYPES,
    ENTITY_TYPES,
    GENERATION_SYSTEM_PROMPT,
    EXTRACTION_SYSTEM_PROMPT,
)


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


class TestSeedData(unittest.TestCase):
    """Test seed entity data completeness."""

    def test_entity_types_match_seed_keys(self):
        for etype in ENTITY_TYPES:
            self.assertIn(etype, SEED_ENTITIES, f"Missing seed entities for {etype}")

    def test_seed_entities_non_empty(self):
        for etype, entities in SEED_ENTITIES.items():
            self.assertGreater(len(entities), 0, f"Empty seed list for {etype}")

    def test_relation_templates_valid(self):
        for tmpl in RELATION_TEMPLATES:
            self.assertIn("subject_type", tmpl)
            self.assertIn("relation", tmpl)
            self.assertIn("object_type", tmpl)
            self.assertIn(tmpl["subject_type"], SEED_ENTITIES)
            self.assertIn(tmpl["object_type"], SEED_ENTITIES)
            self.assertIn(tmpl["relation"], RELATION_TYPES)

    def test_relation_types_coverage(self):
        # At least 10 relation types
        self.assertGreaterEqual(len(RELATION_TYPES), 10)

    def test_seed_entities_sufficient_for_10k(self):
        # With 30 people, 20 orgs, 20 locations, 22 templates → enough combos for 10k
        total_combos = 0
        for tmpl in RELATION_TEMPLATES:
            s_count = len(SEED_ENTITIES[tmpl["subject_type"]])
            o_count = len(SEED_ENTITIES[tmpl["object_type"]])
            total_combos += s_count * o_count
        self.assertGreater(total_combos, 1000, "Need enough entity combos for diversity")


class TestEntityPairSampling(unittest.TestCase):
    """Test entity pair sampling logic."""

    def test_sample_returns_required_keys(self):
        gen = REDatasetGenerator(llm=_make_llm_mock(), seed=42)
        pair = gen.sample_entity_pair()
        self.assertIn("subject", pair)
        self.assertIn("subject_type", pair)
        self.assertIn("object", pair)
        self.assertIn("object_type", pair)
        self.assertIn("relation", pair)

    def test_sample_uses_valid_entities(self):
        gen = REDatasetGenerator(llm=_make_llm_mock(), seed=42)
        for _ in range(50):
            pair = gen.sample_entity_pair()
            self.assertIn(pair["subject"], SEED_ENTITIES[pair["subject_type"]])
            self.assertIn(pair["object"], SEED_ENTITIES[pair["object_type"]])

    def test_sample_avoids_self_reference(self):
        gen = REDatasetGenerator(llm=_make_llm_mock(), seed=42)
        for _ in range(100):
            pair = gen.sample_entity_pair()
            if pair["subject_type"] == pair["object_type"]:
                self.assertNotEqual(pair["subject"], pair["object"])

    def test_deterministic_with_seed(self):
        gen1 = REDatasetGenerator(llm=_make_llm_mock(), seed=123)
        gen2 = REDatasetGenerator(llm=_make_llm_mock(), seed=123)
        pairs1 = [gen1.sample_entity_pair() for _ in range(10)]
        pairs2 = [gen2.sample_entity_pair() for _ in range(10)]
        self.assertEqual(pairs1, pairs2)

    def test_different_seeds_produce_different_pairs(self):
        gen1 = REDatasetGenerator(llm=_make_llm_mock(), seed=1)
        gen2 = REDatasetGenerator(llm=_make_llm_mock(), seed=999)
        pairs1 = [gen1.sample_entity_pair() for _ in range(10)]
        pairs2 = [gen2.sample_entity_pair() for _ in range(10)]
        self.assertNotEqual(pairs1, pairs2)


class TestTextGeneration(unittest.TestCase):
    """Test text generation step."""

    def test_generate_text_calls_llm(self):
        mock_llm = _make_llm_mock()
        mock_llm.generate.return_value = GenerationResult(
            text="Alice Chen works at Nexus AI Labs in Berlin."
        )
        gen = REDatasetGenerator(llm=mock_llm, seed=42)
        pair = {
            "subject": "Alice Chen",
            "subject_type": "Person",
            "object": "Nexus AI Labs",
            "object_type": "Organization",
            "relation": "works_at",
        }
        text = gen.generate_text(pair)
        self.assertEqual(text, "Alice Chen works at Nexus AI Labs in Berlin.")
        mock_llm.generate.assert_called_once()

    def test_generate_text_returns_none_on_empty(self):
        mock_llm = _make_llm_mock()
        mock_llm.generate.return_value = GenerationResult(text="")
        gen = REDatasetGenerator(llm=mock_llm, seed=42)
        pair = gen.sample_entity_pair()
        self.assertIsNone(gen.generate_text(pair))

    def test_generate_text_returns_none_on_failure(self):
        mock_llm = _make_llm_mock()
        mock_llm.generate.side_effect = RuntimeError("API error")
        gen = REDatasetGenerator(llm=mock_llm, seed=42)
        pair = gen.sample_entity_pair()
        self.assertIsNone(gen.generate_text(pair))


class TestRelationExtraction(unittest.TestCase):
    """Test relation extraction step."""

    def test_extract_relations_success(self):
        mock_llm = _make_llm_mock()
        mock_llm.generate_json.return_value = {
            "entities": [
                {"text": "Alice Chen", "type": "Person"},
                {"text": "Nexus AI Labs", "type": "Organization"},
            ],
            "relations": [
                {
                    "subject": "Alice Chen",
                    "subject_type": "Person",
                    "relation": "works_at",
                    "object": "Nexus AI Labs",
                    "object_type": "Organization",
                },
            ],
        }
        gen = REDatasetGenerator(llm=mock_llm, seed=42)
        result = gen.extract_relations("Alice Chen works at Nexus AI Labs.")
        self.assertIsNotNone(result)
        self.assertEqual(len(result["entities"]), 2)
        self.assertEqual(len(result["relations"]), 1)

    def test_extract_relations_missing_fields(self):
        mock_llm = _make_llm_mock()
        mock_llm.generate_json.return_value = {"entities": []}  # missing 'relations'
        gen = REDatasetGenerator(llm=mock_llm, seed=42)
        result = gen.extract_relations("some text")
        self.assertIsNone(result)

    def test_extract_relations_failure(self):
        mock_llm = _make_llm_mock()
        mock_llm.generate_json.side_effect = ValueError("JSON parse error")
        gen = REDatasetGenerator(llm=mock_llm, seed=42)
        result = gen.extract_relations("some text")
        self.assertIsNone(result)


class TestBuildSample(unittest.TestCase):
    """Test sample building and validation."""

    def test_build_sample_success(self):
        mock_llm = _make_llm_mock()
        gen = REDatasetGenerator(llm=mock_llm, seed=42)

        text = "Alice Chen works at Nexus AI Labs in Berlin."
        extraction = {
            "entities": [
                {"text": "Alice Chen", "type": "Person"},
                {"text": "Nexus AI Labs", "type": "Organization"},
                {"text": "Berlin", "type": "Location"},
            ],
            "relations": [
                {
                    "subject": "Alice Chen",
                    "subject_type": "Person",
                    "relation": "works_at",
                    "object": "Nexus AI Labs",
                    "object_type": "Organization",
                },
            ],
        }
        pair = {
            "subject": "Alice Chen",
            "subject_type": "Person",
            "object": "Nexus AI Labs",
            "object_type": "Organization",
            "relation": "works_at",
        }

        sample = gen._build_sample(text, extraction, pair)
        self.assertIsNotNone(sample)
        self.assertEqual(sample.text, text)
        self.assertEqual(len(sample.entities), 3)
        self.assertEqual(len(sample.relations), 1)
        self.assertEqual(sample.relations[0].relation, "works_at")

    def test_build_sample_entity_offsets(self):
        mock_llm = _make_llm_mock()
        gen = REDatasetGenerator(llm=mock_llm, seed=42)

        text = "Alice Chen works at Nexus AI Labs."
        extraction = {
            "entities": [
                {"text": "Alice Chen", "type": "Person"},
                {"text": "Nexus AI Labs", "type": "Organization"},
            ],
            "relations": [
                {
                    "subject": "Alice Chen",
                    "subject_type": "Person",
                    "relation": "works_at",
                    "object": "Nexus AI Labs",
                    "object_type": "Organization",
                },
            ],
        }
        pair = {"subject": "Alice Chen", "subject_type": "Person",
                "object": "Nexus AI Labs", "object_type": "Organization",
                "relation": "works_at"}

        sample = gen._build_sample(text, extraction, pair)
        # Verify offsets
        alice = sample.entities[0]
        self.assertEqual(alice.start, 0)
        self.assertEqual(alice.end, 10)
        self.assertEqual(text[alice.start:alice.end], "Alice Chen")

        nexus = sample.entities[1]
        self.assertEqual(text[nexus.start:nexus.end], "Nexus AI Labs")

    def test_build_sample_case_insensitive_match(self):
        mock_llm = _make_llm_mock()
        gen = REDatasetGenerator(llm=mock_llm, seed=42)

        text = "alice chen is a researcher."
        extraction = {
            "entities": [{"text": "Alice Chen", "type": "Person"}],
            "relations": [
                {
                    "subject": "Alice Chen",
                    "subject_type": "Person",
                    "relation": "works_at",
                    "object": "Alice Chen",  # self-ref for testing
                    "object_type": "Person",
                },
            ],
        }
        pair = {"subject": "Alice Chen", "subject_type": "Person",
                "object": "Alice Chen", "object_type": "Person",
                "relation": "works_at"}

        sample = gen._build_sample(text, extraction, pair)
        self.assertIsNotNone(sample)
        self.assertEqual(len(sample.entities), 1)

    def test_build_sample_returns_none_no_entities(self):
        mock_llm = _make_llm_mock()
        gen = REDatasetGenerator(llm=mock_llm, seed=42)

        text = "A random sentence with no matching entities."
        extraction = {
            "entities": [{"text": "NonExistent", "type": "Person"}],
            "relations": [],
        }
        pair = {"subject": "X", "subject_type": "Person",
                "object": "Y", "object_type": "Organization",
                "relation": "works_at"}

        sample = gen._build_sample(text, extraction, pair)
        self.assertIsNone(sample)

    def test_build_sample_filters_orphan_relations(self):
        mock_llm = _make_llm_mock()
        gen = REDatasetGenerator(llm=mock_llm, seed=42)

        text = "Alice Chen lives in Berlin."
        extraction = {
            "entities": [
                {"text": "Alice Chen", "type": "Person"},
                {"text": "Berlin", "type": "Location"},
            ],
            "relations": [
                {
                    "subject": "Alice Chen",
                    "subject_type": "Person",
                    "relation": "lives_in",
                    "object": "Berlin",
                    "object_type": "Location",
                },
                {
                    # This relation references an entity not in text
                    "subject": "Alice Chen",
                    "subject_type": "Person",
                    "relation": "works_at",
                    "object": "Nexus AI Labs",
                    "object_type": "Organization",
                },
            ],
        }
        pair = {"subject": "Alice Chen", "subject_type": "Person",
                "object": "Berlin", "object_type": "Location",
                "relation": "lives_in"}

        sample = gen._build_sample(text, extraction, pair)
        self.assertIsNotNone(sample)
        self.assertEqual(len(sample.relations), 1)
        self.assertEqual(sample.relations[0].relation, "lives_in")


class TestGenerateSample(unittest.TestCase):
    """Test end-to-end sample generation."""

    def test_generate_sample_full_pipeline(self):
        mock_llm = _make_llm_mock()
        mock_llm.generate.return_value = GenerationResult(
            text="Alice Chen works at Nexus AI Labs in Berlin."
        )
        mock_llm.generate_json.return_value = {
            "entities": [
                {"text": "Alice Chen", "type": "Person"},
                {"text": "Nexus AI Labs", "type": "Organization"},
            ],
            "relations": [
                {
                    "subject": "Alice Chen",
                    "subject_type": "Person",
                    "relation": "works_at",
                    "object": "Nexus AI Labs",
                    "object_type": "Organization",
                },
            ],
        }

        gen = REDatasetGenerator(llm=mock_llm, seed=42)
        sample = gen.generate_sample()
        self.assertIsNotNone(sample)
        self.assertIsInstance(sample, RESample)

    def test_generate_sample_none_on_text_failure(self):
        mock_llm = _make_llm_mock()
        mock_llm.generate.return_value = GenerationResult(text="")
        gen = REDatasetGenerator(llm=mock_llm, seed=42)
        sample = gen.generate_sample()
        self.assertIsNone(sample)

    def test_generate_sample_none_on_extract_failure(self):
        mock_llm = _make_llm_mock()
        mock_llm.generate.return_value = GenerationResult(text="Some text here.")
        mock_llm.generate_json.side_effect = ValueError("Parse error")
        gen = REDatasetGenerator(llm=mock_llm, seed=42)
        sample = gen.generate_sample()
        self.assertIsNone(sample)


class TestGenerateSamples(unittest.TestCase):
    """Test batch sample generation."""

    def test_generate_samples_reaches_target(self):
        mock_llm = _make_llm_mock()
        mock_llm.generate.return_value = GenerationResult(
            text="Bob Martinez lives in Madrid."
        )
        mock_llm.generate_json.return_value = {
            "entities": [
                {"text": "Bob Martinez", "type": "Person"},
                {"text": "Madrid", "type": "Location"},
            ],
            "relations": [
                {
                    "subject": "Bob Martinez",
                    "subject_type": "Person",
                    "relation": "lives_in",
                    "object": "Madrid",
                    "object_type": "Location",
                },
            ],
        }

        gen = REDatasetGenerator(llm=mock_llm, seed=42)
        samples = gen.generate_samples(count=5)
        self.assertEqual(len(samples), 5)

    def test_generate_samples_handles_failures(self):
        mock_llm = _make_llm_mock()
        call_count = [0]

        def generate_side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] % 3 == 0:
                return GenerationResult(text="")  # Fails every 3rd call
            return GenerationResult(text="Bob Martinez lives in Madrid.")

        mock_llm.generate.side_effect = generate_side_effect
        mock_llm.generate_json.return_value = {
            "entities": [
                {"text": "Bob Martinez", "type": "Person"},
                {"text": "Madrid", "type": "Location"},
            ],
            "relations": [
                {
                    "subject": "Bob Martinez",
                    "subject_type": "Person",
                    "relation": "lives_in",
                    "object": "Madrid",
                    "object_type": "Location",
                },
            ],
        }

        gen = REDatasetGenerator(llm=mock_llm, seed=42)
        samples = gen.generate_samples(count=3)
        self.assertEqual(len(samples), 3)

    def test_generate_samples_respects_max_attempts(self):
        mock_llm = _make_llm_mock()
        mock_llm.generate.return_value = GenerationResult(text="")  # Always fails

        gen = REDatasetGenerator(llm=mock_llm, seed=42)
        samples = gen.generate_samples(count=5)
        self.assertEqual(len(samples), 0)
        # Should have stopped at 2x count = 10 attempts
        self.assertEqual(mock_llm.generate.call_count, 10)


class TestRESampleSerialization(unittest.TestCase):
    """Test RESample serialization."""

    def test_to_dict_structure(self):
        sample = RESample(
            text="Alice Chen works at Nexus AI Labs.",
            entities=[
                EntitySpan("Alice Chen", "Person", 0, 10),
                EntitySpan("Nexus AI Labs", "Organization", 20, 33),
            ],
            relations=[
                Relation("Alice Chen", "Person", "works_at", "Nexus AI Labs", "Organization"),
            ],
            metadata={"seed_relation": "works_at"},
        )
        d = sample.to_dict()
        self.assertEqual(d["text"], "Alice Chen works at Nexus AI Labs.")
        self.assertEqual(len(d["entities"]), 2)
        self.assertEqual(len(d["relations"]), 1)
        self.assertEqual(d["metadata"]["seed_relation"], "works_at")

        # Check entity structure
        ent = d["entities"][0]
        self.assertEqual(ent["text"], "Alice Chen")
        self.assertEqual(ent["type"], "Person")
        self.assertEqual(ent["start"], 0)
        self.assertEqual(ent["end"], 10)

        # Check relation structure
        rel = d["relations"][0]
        self.assertEqual(rel["subject"], "Alice Chen")
        self.assertEqual(rel["relation"], "works_at")
        self.assertEqual(rel["object"], "Nexus AI Labs")

    def test_to_dict_json_serializable(self):
        sample = RESample(
            text="Test text.",
            entities=[EntitySpan("Test", "Concept", 0, 4)],
            relations=[Relation("Test", "Concept", "relates_to", "Test", "Concept")],
        )
        # Should not raise
        json_str = json.dumps(sample.to_dict())
        self.assertIsInstance(json_str, str)


class TestExport(unittest.TestCase):
    """Test JSONL export."""

    def test_export_writes_jsonl(self):
        mock_llm = _make_llm_mock()
        mock_llm.export_jsonl.return_value = 2

        gen = REDatasetGenerator(llm=mock_llm, seed=42)
        samples = [
            RESample(
                text="Sample 1",
                entities=[EntitySpan("A", "Person", 0, 1)],
                relations=[Relation("A", "Person", "works_at", "B", "Organization")],
            ),
            RESample(
                text="Sample 2",
                entities=[EntitySpan("C", "Location", 0, 1)],
                relations=[Relation("C", "Location", "part_of", "D", "Location")],
            ),
        ]

        count = gen.export(samples, "/tmp/test_re.jsonl")
        self.assertEqual(count, 2)
        mock_llm.export_jsonl.assert_called_once()
        records = mock_llm.export_jsonl.call_args[0][0]
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["text"], "Sample 1")
        self.assertEqual(records[1]["text"], "Sample 2")


if __name__ == "__main__":
    unittest.main()
