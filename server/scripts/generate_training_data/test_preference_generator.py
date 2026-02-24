#!/usr/bin/env python3
"""Unit tests for PreferenceGenerator (mocked, no real API calls)."""

import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

# Set a dummy key before importing
os.environ.setdefault("FIREWORKS_API_KEY", "fw-test-key-not-real")

from llm_dataset_generator import LLMDatasetGenerator, GenerationConfig, GenerationResult
from preference_generator import (
    PreferenceGenerator,
    PreferencePair,
    CHOSEN_SYSTEM_PROMPT,
    CHOSEN_USER_TEMPLATE,
    REJECTED_SYSTEM_PROMPT,
    REJECTED_USER_TEMPLATE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def _write_re_dataset(path: str, samples: list[dict] | None = None) -> str:
    """Write a minimal RE dataset JSONL file and return the path."""
    if samples is None:
        samples = [
            {
                "text": "Alice Chen works at Nexus AI Labs in Berlin.",
                "entities": [
                    {"text": "Alice Chen", "type": "Person", "start": 0, "end": 10},
                    {"text": "Nexus AI Labs", "type": "Organization", "start": 20, "end": 33},
                    {"text": "Berlin", "type": "Location", "start": 37, "end": 43},
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
                "metadata": {"seed_relation": "works_at"},
            },
            {
                "text": "Bob Martinez lives in Madrid and studies at Atlas Research.",
                "entities": [
                    {"text": "Bob Martinez", "type": "Person", "start": 0, "end": 12},
                    {"text": "Madrid", "type": "Location", "start": 22, "end": 28},
                    {"text": "Atlas Research", "type": "Organization", "start": 44, "end": 58},
                ],
                "relations": [
                    {
                        "subject": "Bob Martinez",
                        "subject_type": "Person",
                        "relation": "lives_in",
                        "object": "Madrid",
                        "object_type": "Location",
                    },
                    {
                        "subject": "Bob Martinez",
                        "subject_type": "Person",
                        "relation": "studied_at",
                        "object": "Atlas Research",
                        "object_type": "Organization",
                    },
                ],
                "metadata": {"seed_relation": "lives_in"},
            },
        ]
    with open(path, "w") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    return path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPreferencePairDataclass(unittest.TestCase):
    """Test PreferencePair data structure."""

    def test_to_dict_has_required_fields(self):
        pair = PreferencePair(
            prompt="Extract entities from: text",
            chosen='{"entities": [], "relations": []}',
            rejected='{"entities": []}',
            ground_truth='{"entities": [], "relations": []}',
        )
        d = pair.to_dict()
        self.assertIn("prompt", d)
        self.assertIn("chosen", d)
        self.assertIn("rejected", d)
        self.assertIn("ground_truth", d)
        self.assertIn("metadata", d)

    def test_to_dict_json_serializable(self):
        pair = PreferencePair(
            prompt="test prompt",
            chosen="chosen text",
            rejected="rejected text",
            ground_truth="{}",
            metadata={"key": "value"},
        )
        json_str = json.dumps(pair.to_dict())
        self.assertIsInstance(json_str, str)
        parsed = json.loads(json_str)
        self.assertEqual(parsed["prompt"], "test prompt")
        self.assertEqual(parsed["metadata"]["key"], "value")

    def test_default_metadata_is_empty(self):
        pair = PreferencePair(
            prompt="p", chosen="c", rejected="r", ground_truth="{}"
        )
        self.assertEqual(pair.metadata, {})


class TestPromptTemplates(unittest.TestCase):
    """Test prompt template formatting."""

    def test_chosen_template_formats(self):
        result = CHOSEN_USER_TEMPLATE.format(text="Hello world.")
        self.assertIn("Hello world.", result)
        self.assertIn("thorough", result.lower())

    def test_rejected_template_formats(self):
        result = REJECTED_USER_TEMPLATE.format(text="Hello world.")
        self.assertIn("Hello world.", result)
        self.assertIn("quickly", result.lower())

    def test_system_prompts_are_nonempty(self):
        self.assertTrue(len(CHOSEN_SYSTEM_PROMPT) > 10)
        self.assertTrue(len(REJECTED_SYSTEM_PROMPT) > 10)


class TestLoadREDataset(unittest.TestCase):
    """Test RE dataset loading."""

    def test_load_valid_dataset(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            _write_re_dataset(f.name)
            path = f.name

        try:
            gen = PreferenceGenerator(llm=_make_llm_mock())
            samples = gen.load_re_dataset(path)
            self.assertEqual(len(samples), 2)
            self.assertIn("text", samples[0])
            self.assertIn("entities", samples[0])
        finally:
            os.unlink(path)

    def test_load_missing_file_raises(self):
        gen = PreferenceGenerator(llm=_make_llm_mock())
        with self.assertRaises(FileNotFoundError):
            gen.load_re_dataset("/nonexistent/path.jsonl")

    def test_load_empty_file_raises(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write("")
            path = f.name

        try:
            gen = PreferenceGenerator(llm=_make_llm_mock())
            with self.assertRaises(ValueError):
                gen.load_re_dataset(path)
        finally:
            os.unlink(path)

    def test_load_skips_invalid_lines(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write("not json\n")
            f.write(json.dumps({"text": "valid", "entities": []}) + "\n")
            f.write("{bad json\n")
            f.write(json.dumps({"text": "also valid", "entities": []}) + "\n")
            path = f.name

        try:
            gen = PreferenceGenerator(llm=_make_llm_mock())
            samples = gen.load_re_dataset(path)
            self.assertEqual(len(samples), 2)
        finally:
            os.unlink(path)

    def test_load_skips_records_without_text(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps({"entities": []}) + "\n")  # no text field
            f.write(json.dumps({"text": "has text", "entities": []}) + "\n")
            path = f.name

        try:
            gen = PreferenceGenerator(llm=_make_llm_mock())
            samples = gen.load_re_dataset(path)
            self.assertEqual(len(samples), 1)
        finally:
            os.unlink(path)


class TestGenerateChosen(unittest.TestCase):
    """Test chosen (thorough) generation."""

    def test_generate_chosen_success(self):
        mock_llm = _make_llm_mock()
        mock_llm.generate.return_value = GenerationResult(
            text='{"entities": [{"text": "Alice", "type": "Person"}], "relations": []}'
        )
        gen = PreferenceGenerator(llm=mock_llm)
        result = gen.generate_chosen("Alice works here.")
        self.assertIsNotNone(result)
        self.assertIn("Alice", result)

    def test_generate_chosen_returns_none_on_empty(self):
        mock_llm = _make_llm_mock()
        mock_llm.generate.return_value = GenerationResult(text="")
        gen = PreferenceGenerator(llm=mock_llm)
        result = gen.generate_chosen("Some text.")
        self.assertIsNone(result)

    def test_generate_chosen_returns_none_on_error(self):
        mock_llm = _make_llm_mock()
        mock_llm.generate.side_effect = RuntimeError("API error")
        gen = PreferenceGenerator(llm=mock_llm)
        result = gen.generate_chosen("Some text.")
        self.assertIsNone(result)

    def test_generate_chosen_uses_correct_config(self):
        mock_llm = _make_llm_mock()
        mock_llm.generate.return_value = GenerationResult(text="response")
        gen = PreferenceGenerator(llm=mock_llm)
        gen.generate_chosen("test")

        call_kwargs = mock_llm.generate.call_args
        config = call_kwargs.kwargs.get("config") or call_kwargs[1].get("config")
        self.assertEqual(config.temperature, 0.0)
        self.assertEqual(config.max_tokens, 300)


class TestGenerateRejected(unittest.TestCase):
    """Test rejected (quick/incomplete) generation."""

    def test_generate_rejected_success(self):
        mock_llm = _make_llm_mock()
        mock_llm.generate.return_value = GenerationResult(
            text='{"entities": [{"text": "Alice", "type": "Person"}], "relations": []}'
        )
        gen = PreferenceGenerator(llm=mock_llm)
        result = gen.generate_rejected("Alice works here.")
        self.assertIsNotNone(result)

    def test_generate_rejected_returns_none_on_empty(self):
        mock_llm = _make_llm_mock()
        mock_llm.generate.return_value = GenerationResult(text="")
        gen = PreferenceGenerator(llm=mock_llm)
        result = gen.generate_rejected("Some text.")
        self.assertIsNone(result)

    def test_generate_rejected_returns_none_on_error(self):
        mock_llm = _make_llm_mock()
        mock_llm.generate.side_effect = RuntimeError("API error")
        gen = PreferenceGenerator(llm=mock_llm)
        result = gen.generate_rejected("Some text.")
        self.assertIsNone(result)

    def test_generate_rejected_uses_correct_config(self):
        mock_llm = _make_llm_mock()
        mock_llm.generate.return_value = GenerationResult(text="response")
        gen = PreferenceGenerator(llm=mock_llm)
        gen.generate_rejected("test")

        call_kwargs = mock_llm.generate.call_args
        config = call_kwargs.kwargs.get("config") or call_kwargs[1].get("config")
        self.assertEqual(config.temperature, 0.7)
        self.assertEqual(config.max_tokens, 150)


class TestGeneratePair(unittest.TestCase):
    """Test single pair generation."""

    def test_generate_pair_success(self):
        mock_llm = _make_llm_mock()
        chosen_json = '{"entities": [{"text": "Alice Chen", "type": "Person"}], "relations": [{"subject": "Alice Chen", "relation": "works_at", "object": "Nexus"}]}'
        rejected_json = '{"entities": [{"text": "Alice Chen", "type": "Person"}], "relations": []}'

        mock_llm.generate.side_effect = [
            GenerationResult(text=chosen_json),
            GenerationResult(text=rejected_json),
        ]

        gen = PreferenceGenerator(llm=mock_llm)
        sample = {
            "text": "Alice Chen works at Nexus AI Labs.",
            "entities": [{"text": "Alice Chen", "type": "Person"}],
            "relations": [{"subject": "Alice Chen", "relation": "works_at", "object": "Nexus AI Labs"}],
        }

        pair = gen.generate_pair(sample)
        self.assertIsNotNone(pair)
        self.assertIn("Alice Chen works at Nexus AI Labs.", pair.prompt)
        self.assertEqual(pair.chosen, chosen_json)
        self.assertEqual(pair.rejected, rejected_json)
        self.assertIn("entities", pair.ground_truth)

    def test_generate_pair_returns_none_on_empty_text(self):
        mock_llm = _make_llm_mock()
        gen = PreferenceGenerator(llm=mock_llm)
        pair = gen.generate_pair({"text": "", "entities": [], "relations": []})
        self.assertIsNone(pair)

    def test_generate_pair_returns_none_on_chosen_failure(self):
        mock_llm = _make_llm_mock()
        mock_llm.generate.return_value = GenerationResult(text="")
        gen = PreferenceGenerator(llm=mock_llm)
        sample = {"text": "Some text.", "entities": [], "relations": []}
        pair = gen.generate_pair(sample)
        self.assertIsNone(pair)

    def test_generate_pair_returns_none_on_rejected_failure(self):
        mock_llm = _make_llm_mock()
        mock_llm.generate.side_effect = [
            GenerationResult(text="chosen response"),
            GenerationResult(text=""),  # rejected fails
        ]
        gen = PreferenceGenerator(llm=mock_llm)
        sample = {"text": "Some text.", "entities": [], "relations": []}
        pair = gen.generate_pair(sample)
        self.assertIsNone(pair)

    def test_generate_pair_ground_truth_from_sample(self):
        mock_llm = _make_llm_mock()
        mock_llm.generate.side_effect = [
            GenerationResult(text="chosen"),
            GenerationResult(text="rejected"),
        ]
        gen = PreferenceGenerator(llm=mock_llm)
        sample = {
            "text": "Test.",
            "entities": [{"text": "Test", "type": "Concept"}],
            "relations": [{"subject": "A", "relation": "relates_to", "object": "B"}],
        }
        pair = gen.generate_pair(sample)
        self.assertIsNotNone(pair)
        gt = json.loads(pair.ground_truth)
        self.assertEqual(len(gt["entities"]), 1)
        self.assertEqual(len(gt["relations"]), 1)

    def test_generate_pair_metadata(self):
        mock_llm = _make_llm_mock()
        mock_llm.generate.side_effect = [
            GenerationResult(text="chosen"),
            GenerationResult(text="rejected"),
        ]
        gen = PreferenceGenerator(llm=mock_llm)
        sample = {"text": "Hello world.", "entities": [], "relations": []}
        pair = gen.generate_pair(sample)
        self.assertIsNotNone(pair)
        self.assertEqual(pair.metadata["source"], "re_dataset")
        self.assertEqual(pair.metadata["text_length"], len("Hello world."))


class TestGeneratePairs(unittest.TestCase):
    """Test batch pair generation."""

    def test_generate_pairs_reaches_target(self):
        mock_llm = _make_llm_mock()
        mock_llm.generate.return_value = GenerationResult(text="mock response")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            _write_re_dataset(f.name)
            path = f.name

        try:
            gen = PreferenceGenerator(llm=mock_llm)
            pairs = gen.generate_pairs(path, count=4)
            self.assertEqual(len(pairs), 4)
        finally:
            os.unlink(path)

    def test_generate_pairs_cycles_through_samples(self):
        mock_llm = _make_llm_mock()
        mock_llm.generate.return_value = GenerationResult(text="response")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            # Only 2 RE samples, requesting 4 pairs → must cycle
            _write_re_dataset(f.name)
            path = f.name

        try:
            gen = PreferenceGenerator(llm=mock_llm)
            pairs = gen.generate_pairs(path, count=4)
            self.assertEqual(len(pairs), 4)
            # LLM called 2x per pair (chosen + rejected) * 4 pairs = 8 calls
            self.assertEqual(mock_llm.generate.call_count, 8)
        finally:
            os.unlink(path)

    def test_generate_pairs_handles_failures(self):
        mock_llm = _make_llm_mock()
        call_count = [0]

        def generate_side_effect(*args, **kwargs):
            call_count[0] += 1
            # Fail every 4th call (makes every 2nd pair fail on rejected)
            if call_count[0] % 4 == 0:
                return GenerationResult(text="")
            return GenerationResult(text="mock response")

        mock_llm.generate.side_effect = generate_side_effect

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            _write_re_dataset(f.name)
            path = f.name

        try:
            gen = PreferenceGenerator(llm=mock_llm)
            pairs = gen.generate_pairs(path, count=3)
            self.assertEqual(len(pairs), 3)
        finally:
            os.unlink(path)

    def test_generate_pairs_respects_max_attempts(self):
        mock_llm = _make_llm_mock()
        mock_llm.generate.return_value = GenerationResult(text="")  # always fails

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            _write_re_dataset(f.name)
            path = f.name

        try:
            gen = PreferenceGenerator(llm=mock_llm)
            pairs = gen.generate_pairs(path, count=5)
            self.assertEqual(len(pairs), 0)
            # 2x count = 10 attempts, each calls generate once for chosen (which fails)
            self.assertEqual(mock_llm.generate.call_count, 10)
        finally:
            os.unlink(path)


class TestExport(unittest.TestCase):
    """Test JSONL export."""

    def test_export_delegates_to_llm(self):
        mock_llm = _make_llm_mock()
        mock_llm.export_jsonl.return_value = 2

        gen = PreferenceGenerator(llm=mock_llm)
        pairs = [
            PreferencePair("p1", "c1", "r1", "{}"),
            PreferencePair("p2", "c2", "r2", "{}"),
        ]

        count = gen.export(pairs, "/tmp/test_pref.jsonl")
        self.assertEqual(count, 2)
        mock_llm.export_jsonl.assert_called_once()
        records = mock_llm.export_jsonl.call_args[0][0]
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["prompt"], "p1")
        self.assertEqual(records[0]["chosen"], "c1")
        self.assertEqual(records[0]["rejected"], "r1")

    def test_export_preserves_all_fields(self):
        mock_llm = _make_llm_mock()
        mock_llm.export_jsonl.return_value = 1

        gen = PreferenceGenerator(llm=mock_llm)
        pairs = [
            PreferencePair(
                prompt="Extract entities from: test",
                chosen='{"entities": []}',
                rejected='{}',
                ground_truth='{"entities": [], "relations": []}',
                metadata={"source": "re_dataset", "text_length": 4},
            ),
        ]

        gen.export(pairs, "/tmp/test_export.jsonl")
        record = mock_llm.export_jsonl.call_args[0][0][0]
        self.assertEqual(record["ground_truth"], '{"entities": [], "relations": []}')
        self.assertEqual(record["metadata"]["source"], "re_dataset")


class TestCustomConfig(unittest.TestCase):
    """Test custom generation configs."""

    def test_custom_chosen_config(self):
        mock_llm = _make_llm_mock()
        mock_llm.generate.return_value = GenerationResult(text="response")
        custom_config = GenerationConfig(max_tokens=500, temperature=0.1)

        gen = PreferenceGenerator(llm=mock_llm, chosen_config=custom_config)
        gen.generate_chosen("test")

        call_kwargs = mock_llm.generate.call_args
        config = call_kwargs.kwargs.get("config") or call_kwargs[1].get("config")
        self.assertEqual(config.max_tokens, 500)
        self.assertEqual(config.temperature, 0.1)

    def test_custom_rejected_config(self):
        mock_llm = _make_llm_mock()
        mock_llm.generate.return_value = GenerationResult(text="response")
        custom_config = GenerationConfig(max_tokens=200, temperature=0.9)

        gen = PreferenceGenerator(llm=mock_llm, rejected_config=custom_config)
        gen.generate_rejected("test")

        call_kwargs = mock_llm.generate.call_args
        config = call_kwargs.kwargs.get("config") or call_kwargs[1].get("config")
        self.assertEqual(config.max_tokens, 200)
        self.assertEqual(config.temperature, 0.9)


if __name__ == "__main__":
    unittest.main()
