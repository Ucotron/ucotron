#!/usr/bin/env python3
"""
Preference (DPO) dataset generator for Ucotron fine-tuning.

Generates chosen/rejected extraction pairs from RE training samples.
For each sample text, calls the LLM twice:
  - "chosen": thorough extraction (temp=0.0, max_tokens=300)
  - "rejected": quick/incomplete extraction (temp=0.7, max_tokens=150)

The output is suitable for Direct Preference Optimization (DPO) training.

Usage:
    from preference_generator import PreferenceGenerator

    gen = PreferenceGenerator()
    pairs = gen.generate_pairs("re_dataset.jsonl", count=100)
    gen.export(pairs, "preference_dataset.jsonl")

    # Or via CLI:
    python preference_generator.py --input re_dataset.jsonl --count 100 --output preference_dataset.jsonl

Environment:
    FIREWORKS_API_KEY  - Required. Fireworks.ai API key.

Requirements:
    pip install openai>=1.0
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from llm_dataset_generator import (
    LLMDatasetGenerator,
    GenerationConfig,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PreferencePair:
    """A single DPO preference pair for training."""
    prompt: str
    chosen: str
    rejected: str
    ground_truth: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable dict for JSONL export."""
        return {
            "prompt": self.prompt,
            "chosen": self.chosen,
            "rejected": self.rejected,
            "ground_truth": self.ground_truth,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

CHOSEN_SYSTEM_PROMPT = """\
You are a precise entity and relation extraction model. \
Given a text, identify ALL entities and the relations between them. \
Be thorough, accurate, and exhaustive. Never miss an entity or relation."""

CHOSEN_USER_TEMPLATE = """\
Extract all entities and relations from this text. Be thorough and accurate. \
Include every entity and relationship you can identify.

Text: {text}

Return a JSON object with two arrays:
- "entities": each with "text" (exact span), "type" (Person, Organization, \
Location, Event, Concept, or Skill)
- "relations": each with "subject", "subject_type", "relation" (one of: \
works_at, lives_in, born_in, founded, located_in, part_of, caused_by, \
relates_to, has_property, studied_at, created, manages, collaborates_with, \
married_to, child_of, friend_of, competitor_of, acquired, invested_in, speaks), \
"object", "object_type"

Respond with ONLY the JSON object."""

REJECTED_SYSTEM_PROMPT = """\
You are a quick entity extraction helper. \
Extract the most obvious entities and relations from the given text."""

REJECTED_USER_TEMPLATE = """\
Quickly extract some entities and relations from this text. \
Focus only on the most obvious ones.

Text: {text}

Return a JSON object with "entities" and "relations" arrays.

Respond with ONLY the JSON object."""


# ---------------------------------------------------------------------------
# Generator class
# ---------------------------------------------------------------------------

class PreferenceGenerator:
    """
    Generates DPO preference pairs from RE training data.

    For each RE sample:
    1. Constructs a prompt asking for entity/relation extraction
    2. Generates a "chosen" response (thorough, low-temperature)
    3. Generates a "rejected" response (quick, high-temperature, shorter)
    4. Packages as a preference pair with the ground truth

    The chosen response is expected to be more complete and accurate,
    while the rejected response is expected to miss entities/relations
    due to the "quick" framing and higher temperature.
    """

    def __init__(
        self,
        llm: LLMDatasetGenerator | None = None,
        chosen_config: GenerationConfig | None = None,
        rejected_config: GenerationConfig | None = None,
    ):
        """
        Initialize the preference generator.

        Args:
            llm: Pre-configured LLMDatasetGenerator (creates one if None).
            chosen_config: Config for thorough extraction (default: temp=0.0, 300 tokens).
            rejected_config: Config for quick extraction (default: temp=0.7, 150 tokens).
        """
        self.llm = llm or LLMDatasetGenerator()
        self.chosen_config = chosen_config or GenerationConfig(
            max_tokens=300,
            temperature=0.0,
        )
        self.rejected_config = rejected_config or GenerationConfig(
            max_tokens=150,
            temperature=0.7,
        )

    def load_re_dataset(self, path: str | Path) -> list[dict[str, Any]]:
        """
        Load RE samples from a JSONL file.

        Args:
            path: Path to JSONL file (one JSON record per line).

        Returns:
            List of RE sample dicts.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file is empty or has no valid records.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"RE dataset not found: {path}")

        samples: list[dict[str, Any]] = []
        with open(path) as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    if "text" in record:
                        samples.append(record)
                except json.JSONDecodeError:
                    logger.warning("Skipping invalid JSON at line %d", line_num)

        if not samples:
            raise ValueError(f"No valid RE samples found in {path}")

        logger.info("Loaded %d RE samples from %s", len(samples), path)
        return samples

    def generate_chosen(self, text: str) -> str | None:
        """
        Generate a thorough (chosen) extraction for the given text.

        Args:
            text: Input text to extract from.

        Returns:
            JSON string of the extraction, or None on failure.
        """
        prompt = CHOSEN_USER_TEMPLATE.format(text=text)
        try:
            result = self.llm.generate(
                user_prompt=prompt,
                system_prompt=CHOSEN_SYSTEM_PROMPT,
                config=self.chosen_config,
            )
            response = result.text.strip()
            if not response:
                return None
            return response
        except RuntimeError:
            logger.warning("Chosen generation failed for text: %.80s...", text)
            return None

    def generate_rejected(self, text: str) -> str | None:
        """
        Generate a quick/incomplete (rejected) extraction for the given text.

        Args:
            text: Input text to extract from.

        Returns:
            JSON string of the extraction, or None on failure.
        """
        prompt = REJECTED_USER_TEMPLATE.format(text=text)
        try:
            result = self.llm.generate(
                user_prompt=prompt,
                system_prompt=REJECTED_SYSTEM_PROMPT,
                config=self.rejected_config,
            )
            response = result.text.strip()
            if not response:
                return None
            return response
        except RuntimeError:
            logger.warning("Rejected generation failed for text: %.80s...", text)
            return None

    def generate_pair(self, sample: dict[str, Any]) -> PreferencePair | None:
        """
        Generate a single preference pair from an RE sample.

        Args:
            sample: An RE sample dict with 'text', 'entities', 'relations'.

        Returns:
            PreferencePair or None if either generation fails.
        """
        text = sample.get("text", "")
        if not text:
            return None

        chosen = self.generate_chosen(text)
        if chosen is None:
            return None

        rejected = self.generate_rejected(text)
        if rejected is None:
            return None

        # Build the ground truth from the original RE sample
        ground_truth = json.dumps({
            "entities": sample.get("entities", []),
            "relations": sample.get("relations", []),
        }, ensure_ascii=False)

        prompt_text = f"Extract entities and relations from: {text}"

        return PreferencePair(
            prompt=prompt_text,
            chosen=chosen,
            rejected=rejected,
            ground_truth=ground_truth,
            metadata={
                "source": "re_dataset",
                "text_length": len(text),
            },
        )

    def generate_pairs(
        self,
        re_dataset_path: str | Path,
        count: int = 5000,
        progress_interval: int = 100,
    ) -> list[PreferencePair]:
        """
        Generate preference pairs from an RE dataset.

        Iterates through RE samples (cycling if needed) to produce
        the target number of pairs.

        Args:
            re_dataset_path: Path to RE dataset JSONL file.
            count: Target number of preference pairs.
            progress_interval: Log progress every N attempts.

        Returns:
            List of successfully generated PreferencePairs.
        """
        samples = self.load_re_dataset(re_dataset_path)
        pairs: list[PreferencePair] = []
        attempts = 0
        max_attempts = count * 2  # allow 2x attempts for failures

        while len(pairs) < count and attempts < max_attempts:
            # Cycle through samples
            sample = samples[attempts % len(samples)]
            pair = self.generate_pair(sample)
            attempts += 1

            if pair is not None:
                pairs.append(pair)

            if attempts % progress_interval == 0:
                logger.info(
                    "Preference generation progress: %d/%d pairs (%d attempts)",
                    len(pairs), count, attempts,
                )

        logger.info(
            "Preference generation complete: %d/%d pairs in %d attempts (%.1f%% success rate)",
            len(pairs), count, attempts,
            len(pairs) / max(attempts, 1) * 100,
        )
        return pairs

    def export(
        self,
        pairs: list[PreferencePair],
        output_path: str | Path,
    ) -> int:
        """
        Export preference pairs to JSONL file.

        Args:
            pairs: List of PreferencePairs to export.
            output_path: Destination file path.

        Returns:
            Number of records written.
        """
        records = [p.to_dict() for p in pairs]
        return self.llm.export_jsonl(records, output_path)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    """Generate preference dataset via CLI."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate DPO preference dataset from RE samples"
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="Path to RE dataset JSONL file (from re_dataset_generator)",
    )
    parser.add_argument(
        "--count", type=int, default=5000,
        help="Number of preference pairs to generate (default: 5000)",
    )
    parser.add_argument(
        "--output", type=str, default="preference_dataset.jsonl",
        help="Output JSONL file path (default: preference_dataset.jsonl)",
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

    gen = PreferenceGenerator()
    pairs = gen.generate_pairs(
        re_dataset_path=args.input,
        count=args.count,
        progress_interval=args.progress_interval,
    )
    written = gen.export(pairs, args.output)
    print(f"\nGenerated {written} preference pairs → {args.output}")
    print(f"API usage: {gen.llm.usage_summary()}")


if __name__ == "__main__":
    main()
