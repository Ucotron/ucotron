#!/usr/bin/env python3
"""Unit tests for LLMDatasetGenerator (mocked, no real API calls)."""

import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

# Set a dummy key before importing (prevents ValueError on import-time usage)
os.environ.setdefault("FIREWORKS_API_KEY", "fw-test-key-not-real")

from llm_dataset_generator import (
    LLMDatasetGenerator,
    GenerationConfig,
    GenerationResult,
    BatchStats,
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
)


def _mock_response(text="Hello world", prompt_tokens=10, completion_tokens=20,
                    model="test-model", finish_reason="stop"):
    """Build a mock OpenAI ChatCompletion response."""
    choice = MagicMock()
    choice.message.content = text
    choice.finish_reason = finish_reason

    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens

    response = MagicMock()
    response.choices = [choice]
    response.usage = usage
    response.model = model
    return response


class TestLLMDatasetGeneratorInit(unittest.TestCase):
    """Test initialization and configuration."""

    def test_init_with_env_var(self):
        with patch.dict(os.environ, {"FIREWORKS_API_KEY": "fw-from-env"}):
            gen = LLMDatasetGenerator()
            self.assertEqual(gen.model, DEFAULT_MODEL)

    def test_init_with_explicit_key(self):
        gen = LLMDatasetGenerator(api_key="fw-explicit")
        self.assertEqual(gen.model, DEFAULT_MODEL)

    def test_init_missing_key_raises(self):
        with patch.dict(os.environ, {}, clear=True):
            # Remove FIREWORKS_API_KEY if present
            env = dict(os.environ)
            env.pop("FIREWORKS_API_KEY", None)
            with patch.dict(os.environ, env, clear=True):
                with self.assertRaises(ValueError) as ctx:
                    LLMDatasetGenerator()
                self.assertIn("FIREWORKS_API_KEY", str(ctx.exception))

    def test_init_custom_model(self):
        gen = LLMDatasetGenerator(
            api_key="fw-test",
            model="accounts/fireworks/models/custom-model",
        )
        self.assertEqual(gen.model, "accounts/fireworks/models/custom-model")

    def test_initial_usage_is_zero(self):
        gen = LLMDatasetGenerator(api_key="fw-test")
        self.assertEqual(gen.total_tokens_used, 0)
        self.assertEqual(gen.total_requests, 0)
        summary = gen.usage_summary()
        self.assertEqual(summary["total_tokens"], 0)


class TestGenerate(unittest.TestCase):
    """Test single-turn generation."""

    def setUp(self):
        self.gen = LLMDatasetGenerator(api_key="fw-test")
        self.mock_create = MagicMock(return_value=_mock_response("Test output"))
        self.gen.client.chat.completions.create = self.mock_create

    def test_generate_returns_result(self):
        result = self.gen.generate("Hello")
        self.assertIsInstance(result, GenerationResult)
        self.assertEqual(result.text, "Test output")
        self.assertEqual(result.prompt_tokens, 10)
        self.assertEqual(result.completion_tokens, 20)
        self.assertEqual(result.total_tokens, 30)

    def test_generate_passes_messages(self):
        self.gen.generate("What is 2+2?", system_prompt="Be concise.")
        call_kwargs = self.mock_create.call_args[1]
        messages = call_kwargs["messages"]
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], "Be concise.")
        self.assertEqual(messages[1]["role"], "user")
        self.assertEqual(messages[1]["content"], "What is 2+2?")

    def test_generate_with_config(self):
        cfg = GenerationConfig(max_tokens=100, temperature=0.3, top_p=0.8)
        self.gen.generate("test", config=cfg)
        call_kwargs = self.mock_create.call_args[1]
        self.assertEqual(call_kwargs["max_tokens"], 100)
        self.assertEqual(call_kwargs["temperature"], 0.3)
        self.assertEqual(call_kwargs["top_p"], 0.8)

    def test_generate_tracks_usage(self):
        self.gen.generate("first")
        self.gen.generate("second")
        self.assertEqual(self.gen.total_requests, 2)
        self.assertEqual(self.gen._total_prompt_tokens, 20)
        self.assertEqual(self.gen._total_completion_tokens, 40)


class TestGenerateChat(unittest.TestCase):
    """Test multi-turn chat generation."""

    def setUp(self):
        self.gen = LLMDatasetGenerator(api_key="fw-test")
        self.mock_create = MagicMock(return_value=_mock_response("Chat reply"))
        self.gen.client.chat.completions.create = self.mock_create

    def test_chat_passes_all_messages(self):
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
            {"role": "user", "content": "How are you?"},
        ]
        result = self.gen.generate_chat(messages)
        self.assertEqual(result.text, "Chat reply")
        call_kwargs = self.mock_create.call_args[1]
        self.assertEqual(len(call_kwargs["messages"]), 4)


class TestGenerateJson(unittest.TestCase):
    """Test JSON-mode generation."""

    def setUp(self):
        self.gen = LLMDatasetGenerator(api_key="fw-test")

    def test_json_mode_parses_response(self):
        json_text = json.dumps({"name": "Alice", "age": 30})
        self.gen.client.chat.completions.create = MagicMock(
            return_value=_mock_response(json_text)
        )
        result = self.gen.generate_json("Give me a person")
        self.assertEqual(result["name"], "Alice")
        self.assertEqual(result["age"], 30)

    def test_json_mode_strips_code_fences(self):
        fenced = '```json\n{"key": "value"}\n```'
        self.gen.client.chat.completions.create = MagicMock(
            return_value=_mock_response(fenced)
        )
        result = self.gen.generate_json("Give JSON")
        self.assertEqual(result["key"], "value")

    def test_json_mode_extracts_embedded_json(self):
        text = 'Here is the result: {"x": 1} and some trailing text.'
        self.gen.client.chat.completions.create = MagicMock(
            return_value=_mock_response(text)
        )
        result = self.gen.generate_json("Give JSON")
        self.assertEqual(result["x"], 1)

    def test_json_mode_raises_on_invalid(self):
        self.gen.client.chat.completions.create = MagicMock(
            return_value=_mock_response("This is not JSON at all.")
        )
        with self.assertRaises(ValueError):
            self.gen.generate_json("Give JSON")


class TestRetryLogic(unittest.TestCase):
    """Test exponential backoff retry logic."""

    def setUp(self):
        self.gen = LLMDatasetGenerator(
            api_key="fw-test", max_retries=3, retry_delay=0.01
        )

    @patch("llm_dataset_generator.time.sleep")
    def test_retries_on_connection_error(self, mock_sleep):
        from openai import APIConnectionError

        self.gen.client.chat.completions.create = MagicMock(
            side_effect=APIConnectionError(request=MagicMock())
        )
        with self.assertRaises(RuntimeError) as ctx:
            self.gen.generate("test")
        self.assertIn("3 retries", str(ctx.exception))
        self.assertEqual(self.gen.client.chat.completions.create.call_count, 3)

    @patch("llm_dataset_generator.time.sleep")
    def test_retries_on_rate_limit(self, mock_sleep):
        from openai import RateLimitError

        mock_resp = MagicMock()
        mock_resp.status_code = 429
        mock_resp.headers = {}
        self.gen.client.chat.completions.create = MagicMock(
            side_effect=RateLimitError(
                message="rate limited",
                response=mock_resp,
                body=None,
            )
        )
        with self.assertRaises(RuntimeError):
            self.gen.generate("test")
        # Should have retried 3 times
        self.assertEqual(self.gen.client.chat.completions.create.call_count, 3)

    @patch("llm_dataset_generator.time.sleep")
    def test_succeeds_after_transient_failure(self, mock_sleep):
        from openai import APITimeoutError

        self.gen.client.chat.completions.create = MagicMock(
            side_effect=[
                APITimeoutError(request=MagicMock()),
                _mock_response("recovered"),
            ]
        )
        result = self.gen.generate("test")
        self.assertEqual(result.text, "recovered")
        self.assertEqual(self.gen.client.chat.completions.create.call_count, 2)


class TestBatchGeneration(unittest.TestCase):
    """Test batch generation with statistics."""

    def setUp(self):
        self.gen = LLMDatasetGenerator(
            api_key="fw-test", max_retries=1, retry_delay=0.01
        )

    def test_batch_all_succeed(self):
        self.gen.client.chat.completions.create = MagicMock(
            return_value=_mock_response("ok")
        )
        results, stats = self.gen.generate_batch(
            ["p1", "p2", "p3"], progress_interval=100
        )
        self.assertEqual(len(results), 3)
        self.assertEqual(stats.successful, 3)
        self.assertEqual(stats.failed, 0)
        self.assertAlmostEqual(stats.success_rate, 1.0)

    @patch("llm_dataset_generator.time.sleep")
    def test_batch_with_failures(self, mock_sleep):
        from openai import APIConnectionError

        self.gen.client.chat.completions.create = MagicMock(
            side_effect=[
                _mock_response("ok"),
                APIConnectionError(request=MagicMock()),
                _mock_response("ok"),
            ]
        )
        results, stats = self.gen.generate_batch(
            ["p1", "p2", "p3"], progress_interval=100
        )
        self.assertEqual(stats.successful, 2)
        self.assertEqual(stats.failed, 1)
        self.assertEqual(results[1].text, "")  # failed prompt

    def test_batch_stats_token_tracking(self):
        self.gen.client.chat.completions.create = MagicMock(
            return_value=_mock_response("ok", prompt_tokens=5, completion_tokens=15)
        )
        _, stats = self.gen.generate_batch(["p1", "p2"], progress_interval=100)
        self.assertEqual(stats.total_prompt_tokens, 10)
        self.assertEqual(stats.total_completion_tokens, 30)
        self.assertEqual(stats.total_tokens, 40)


class TestExportJsonl(unittest.TestCase):
    """Test JSONL file export."""

    def test_export_writes_jsonl(self):
        gen = LLMDatasetGenerator(api_key="fw-test")
        records = [
            {"text": "Alice met Bob", "entities": ["Alice", "Bob"]},
            {"text": "Carol likes Dave", "entities": ["Carol", "Dave"]},
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "subdir", "output.jsonl")
            count = gen.export_jsonl(records, path)
            self.assertEqual(count, 2)

            with open(path) as f:
                lines = f.readlines()
            self.assertEqual(len(lines), 2)
            self.assertEqual(json.loads(lines[0])["text"], "Alice met Bob")

    def test_export_handles_unicode(self):
        gen = LLMDatasetGenerator(api_key="fw-test")
        records = [{"text": "Berlín está en Alemania"}]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "unicode.jsonl")
            gen.export_jsonl(records, path)
            with open(path, encoding="utf-8") as f:
                data = json.loads(f.readline())
            self.assertIn("Berlín", data["text"])


class TestParseJson(unittest.TestCase):
    """Test static JSON parsing helper."""

    def test_direct_json(self):
        result = LLMDatasetGenerator._parse_json('{"a": 1}')
        self.assertEqual(result, {"a": 1})

    def test_code_fence_json(self):
        text = '```json\n{"a": 1}\n```'
        result = LLMDatasetGenerator._parse_json(text)
        self.assertEqual(result, {"a": 1})

    def test_embedded_json(self):
        text = 'The answer is {"a": 1} as shown.'
        result = LLMDatasetGenerator._parse_json(text)
        self.assertEqual(result, {"a": 1})

    def test_invalid_raises(self):
        with self.assertRaises(ValueError):
            LLMDatasetGenerator._parse_json("no json here")


if __name__ == "__main__":
    unittest.main()
