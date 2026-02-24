#!/usr/bin/env python3
"""
LLM-powered dataset generator for Ucotron fine-tuning pipelines.

Uses GLM-5 via Fireworks.ai (OpenAI-compatible API) to synthesize training
datasets for relation extraction, DPO preference alignment, contradiction
detection, and entity resolution.

Usage:
    from llm_dataset_generator import LLMDatasetGenerator

    gen = LLMDatasetGenerator()
    # or with explicit config:
    gen = LLMDatasetGenerator(
        api_key="fw-...",
        model="accounts/fireworks/models/glm-5",
        base_url="https://api.fireworks.ai/inference/v1",
    )

    response = gen.generate("Write a sentence about two people meeting.")
    samples = gen.generate_batch(prompts, batch_size=10)

Environment:
    FIREWORKS_API_KEY  - Required. Fireworks.ai API key (never in config files).

Requirements:
    pip install openai>=1.0
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from openai import OpenAI, APIConnectionError, APITimeoutError, RateLimitError

logger = logging.getLogger(__name__)

# Default Fireworks configuration
DEFAULT_BASE_URL = "https://api.fireworks.ai/inference/v1"
DEFAULT_MODEL = "accounts/fireworks/models/glm-4-plus"
DEFAULT_MAX_TOKENS = 2048
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY = 2.0  # seconds, doubles on each retry


@dataclass
class GenerationConfig:
    """Parameters for a single LLM generation call."""

    max_tokens: int = DEFAULT_MAX_TOKENS
    temperature: float = DEFAULT_TEMPERATURE
    top_p: float = 0.9
    stop: list[str] | None = None
    response_format: dict[str, str] | None = None


@dataclass
class GenerationResult:
    """Result of a single LLM generation call."""

    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model: str = ""
    finish_reason: str = ""

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class BatchStats:
    """Aggregate statistics for a batch generation run."""

    total_prompts: int = 0
    successful: int = 0
    failed: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    elapsed_seconds: float = 0.0

    @property
    def success_rate(self) -> float:
        return self.successful / max(self.total_prompts, 1)

    @property
    def total_tokens(self) -> int:
        return self.total_prompt_tokens + self.total_completion_tokens


class LLMDatasetGenerator:
    """
    Generates training datasets using an LLM via Fireworks.ai.

    Uses the OpenAI-compatible API endpoint. The API key must be provided
    via the FIREWORKS_API_KEY environment variable or the api_key parameter.

    Supports:
    - Single-turn generation (system + user prompt)
    - Multi-turn chat generation
    - Batch generation with retry and progress logging
    - JSON mode for structured output
    - JSONL export for training pipelines
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_delay: float = DEFAULT_RETRY_DELAY,
        timeout: float = 120.0,
    ):
        resolved_key = api_key or os.environ.get("FIREWORKS_API_KEY")
        if not resolved_key:
            raise ValueError(
                "Fireworks API key required. Set FIREWORKS_API_KEY env var "
                "or pass api_key parameter."
            )

        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        self.client = OpenAI(
            api_key=resolved_key,
            base_url=base_url,
            timeout=timeout,
        )

        self._total_prompt_tokens = 0
        self._total_completion_tokens = 0
        self._total_requests = 0

    # ------------------------------------------------------------------
    # Core generation
    # ------------------------------------------------------------------

    def generate(
        self,
        user_prompt: str,
        system_prompt: str = "You are a helpful assistant.",
        config: GenerationConfig | None = None,
    ) -> GenerationResult:
        """
        Generate a single completion from the LLM.

        Args:
            user_prompt: The user message content.
            system_prompt: System message to set model behavior.
            config: Optional generation parameters.

        Returns:
            GenerationResult with the generated text and token usage.

        Raises:
            RuntimeError: After exhausting all retries.
        """
        cfg = config or GenerationConfig()
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return self._chat_completion(messages, cfg)

    def generate_chat(
        self,
        messages: list[dict[str, str]],
        config: GenerationConfig | None = None,
    ) -> GenerationResult:
        """
        Generate a completion from a multi-turn conversation.

        Args:
            messages: List of {"role": ..., "content": ...} dicts.
            config: Optional generation parameters.

        Returns:
            GenerationResult with the generated text and token usage.
        """
        cfg = config or GenerationConfig()
        return self._chat_completion(messages, cfg)

    def generate_json(
        self,
        user_prompt: str,
        system_prompt: str = "You are a helpful assistant that responds in JSON.",
        config: GenerationConfig | None = None,
    ) -> dict[str, Any]:
        """
        Generate a JSON-structured response.

        Parses the LLM output as JSON. Falls back to extracting the first
        JSON object from the text if the full output isn't valid JSON.

        Args:
            user_prompt: The user message content.
            system_prompt: System message (should mention JSON output).
            config: Optional generation parameters.

        Returns:
            Parsed JSON as a Python dict.

        Raises:
            ValueError: If the output cannot be parsed as JSON.
        """
        cfg = config or GenerationConfig()
        if cfg.response_format is None:
            cfg.response_format = {"type": "json_object"}

        result = self.generate(user_prompt, system_prompt, cfg)
        return self._parse_json(result.text)

    # ------------------------------------------------------------------
    # Batch generation
    # ------------------------------------------------------------------

    def generate_batch(
        self,
        prompts: list[str],
        system_prompt: str = "You are a helpful assistant.",
        config: GenerationConfig | None = None,
        batch_size: int = 10,
        progress_interval: int = 50,
    ) -> tuple[list[GenerationResult], BatchStats]:
        """
        Generate completions for a list of prompts with progress logging.

        Args:
            prompts: List of user prompts.
            system_prompt: Shared system prompt for all calls.
            config: Optional generation parameters.
            batch_size: Not used for parallelism (sequential), but groups
                        progress logging.
            progress_interval: Log progress every N prompts.

        Returns:
            Tuple of (results list, batch statistics).
            Failed prompts have an empty GenerationResult in the list.
        """
        cfg = config or GenerationConfig()
        stats = BatchStats(total_prompts=len(prompts))
        results: list[GenerationResult] = []
        start = time.monotonic()

        for i, prompt in enumerate(prompts):
            try:
                result = self.generate(prompt, system_prompt, cfg)
                results.append(result)
                stats.successful += 1
                stats.total_prompt_tokens += result.prompt_tokens
                stats.total_completion_tokens += result.completion_tokens
            except RuntimeError:
                logger.warning("Prompt %d failed after retries, skipping.", i)
                results.append(GenerationResult(text=""))
                stats.failed += 1

            if (i + 1) % progress_interval == 0:
                elapsed = time.monotonic() - start
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                logger.info(
                    "Progress: %d/%d (%.1f/s) | tokens: %d",
                    i + 1,
                    len(prompts),
                    rate,
                    stats.total_tokens,
                )

        stats.elapsed_seconds = time.monotonic() - start
        logger.info(
            "Batch complete: %d/%d succeeded (%.1f%%) in %.1fs",
            stats.successful,
            stats.total_prompts,
            stats.success_rate * 100,
            stats.elapsed_seconds,
        )
        return results, stats

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_jsonl(
        self,
        records: list[dict[str, Any]],
        output_path: str | Path,
    ) -> int:
        """
        Write records to a JSONL file.

        Args:
            records: List of dicts to serialize.
            output_path: Destination file path.

        Returns:
            Number of records written.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        count = 0
        with open(output_path, "w") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1

        logger.info("Exported %d records to %s", count, output_path)
        return count

    # ------------------------------------------------------------------
    # Usage tracking
    # ------------------------------------------------------------------

    @property
    def total_tokens_used(self) -> int:
        """Cumulative tokens used across all requests."""
        return self._total_prompt_tokens + self._total_completion_tokens

    @property
    def total_requests(self) -> int:
        """Cumulative number of API requests made."""
        return self._total_requests

    def usage_summary(self) -> dict[str, int]:
        """Return a summary of cumulative API usage."""
        return {
            "total_requests": self._total_requests,
            "total_prompt_tokens": self._total_prompt_tokens,
            "total_completion_tokens": self._total_completion_tokens,
            "total_tokens": self.total_tokens_used,
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _chat_completion(
        self,
        messages: list[dict[str, str]],
        config: GenerationConfig,
    ) -> GenerationResult:
        """Execute a chat completion with retry logic."""
        last_error: Exception | None = None
        delay = self.retry_delay

        for attempt in range(1, self.max_retries + 1):
            try:
                kwargs: dict[str, Any] = {
                    "model": self.model,
                    "messages": messages,
                    "max_tokens": config.max_tokens,
                    "temperature": config.temperature,
                    "top_p": config.top_p,
                }
                if config.stop:
                    kwargs["stop"] = config.stop
                if config.response_format:
                    kwargs["response_format"] = config.response_format

                response = self.client.chat.completions.create(**kwargs)
                choice = response.choices[0]
                usage = response.usage

                self._total_requests += 1
                if usage:
                    self._total_prompt_tokens += usage.prompt_tokens
                    self._total_completion_tokens += usage.completion_tokens

                return GenerationResult(
                    text=choice.message.content or "",
                    prompt_tokens=usage.prompt_tokens if usage else 0,
                    completion_tokens=usage.completion_tokens if usage else 0,
                    model=response.model or self.model,
                    finish_reason=choice.finish_reason or "",
                )

            except (APIConnectionError, APITimeoutError) as e:
                last_error = e
                logger.warning(
                    "Attempt %d/%d failed (connection/timeout): %s",
                    attempt,
                    self.max_retries,
                    e,
                )
            except RateLimitError as e:
                last_error = e
                logger.warning(
                    "Attempt %d/%d rate-limited, waiting %.1fs",
                    attempt,
                    self.max_retries,
                    delay,
                )

            if attempt < self.max_retries:
                time.sleep(delay)
                delay *= 2  # exponential backoff

        raise RuntimeError(
            f"LLM generation failed after {self.max_retries} retries: {last_error}"
        )

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any]:
        """Parse JSON from LLM output, handling markdown code fences."""
        text = text.strip()

        # Try direct parse first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Strip markdown code fences
        if text.startswith("```"):
            lines = text.split("\n")
            # Remove first line (```json) and last line (```)
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines).strip()
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                pass

        # Try to find first { ... } block
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                pass

        raise ValueError(f"Could not parse JSON from LLM output: {text[:200]}...")


# ------------------------------------------------------------------
# CLI entry point for quick testing
# ------------------------------------------------------------------

def main():
    """Quick smoke test: generate a single response from GLM-5."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Test LLMDatasetGenerator connection to Fireworks"
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Model ID (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--prompt",
        default="List 3 famous scientists and one discovery each. Respond in JSON.",
        help="Test prompt to send",
    )
    parser.add_argument(
        "--json-mode",
        action="store_true",
        help="Request JSON-formatted response",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    gen = LLMDatasetGenerator(model=args.model)
    logger.info("Connected to Fireworks API with model: %s", gen.model)

    if args.json_mode:
        result = gen.generate_json(args.prompt)
        print(json.dumps(result, indent=2))
    else:
        result = gen.generate(args.prompt)
        print(result.text)

    print(f"\n--- Usage: {gen.usage_summary()} ---")


if __name__ == "__main__":
    main()
