#!/usr/bin/env python3
"""
Entity resolution dataset generator for Ucotron fine-tuning.

Generates training samples for entity resolution by:
1. Selecting canonical entity names from a seed knowledge base
2. Generating name variations (typos, abbreviations, nicknames) via GLM-5
3. Creating positive pairs (is_duplicate=true) and negative pairs (is_duplicate=false)
4. Exporting as JSONL with canonical, variant, is_duplicate, metadata fields

The dataset trains models to recognize when two entity mentions refer to the
same real-world entity, using Ucotron's structural+semantic similarity approach
(0.6 Jaccard + 0.4 cosine).

Usage:
    from entity_resolution_generator import EntityResolutionGenerator

    gen = EntityResolutionGenerator()
    samples = gen.generate_samples(count=2000)
    gen.export(samples, "entity_resolution_dataset.jsonl")

    # Or via CLI:
    python entity_resolution_generator.py --count 2000 --output entity_resolution_dataset.jsonl

Environment:
    FIREWORKS_API_KEY  - Required. Fireworks.ai API key.

Requirements:
    pip install openai>=1.0
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from llm_dataset_generator import (
    LLMDatasetGenerator,
    GenerationConfig,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Seed knowledge base for entity generation
# ---------------------------------------------------------------------------

ENTITY_NAMES: dict[str, list[str]] = {
    "Person": [
        "Alice Johnson", "Carlos García", "Maria Ivanova", "Yuki Tanaka",
        "Ahmed Hassan", "Priya Sharma", "Lukas Müller", "Sofia Rossi",
        "James Chen", "Fatima Al-Rashid", "Olga Petrova", "Diego López",
        "Emma Wilson", "Raj Patel", "Hannah Schmidt", "Kenji Nakamura",
        "Nina Kowalski", "Samuel Okafor", "Lena Eriksson", "David Kim",
    ],
    "Organization": [
        "Nexus Technologies", "Global Health Initiative", "Quantum Research Lab",
        "Stellar Media Group", "Pacific Dynamics", "Atlas Financial",
        "Verde Sustainability", "Apex Robotics", "Meridian Consulting",
        "Horizon Education Foundation", "Nordic Energy Solutions", "Orion Aerospace",
    ],
    "Location": [
        "Madrid", "Tokyo", "Berlin", "New York", "São Paulo", "Mumbai",
        "Sydney", "London", "Seoul", "Toronto", "Dubai", "Stockholm",
        "Singapore", "Paris", "Mexico City", "Buenos Aires",
    ],
}

# Variation types for entity names
VARIATION_TYPES: list[str] = [
    "typo",           # Common misspellings
    "abbreviation",   # Shortened forms
    "nickname",       # Informal names
    "case_variation", # Different capitalization
    "partial",        # First name only, last name only
    "formal",         # Full formal version
    "transliteration",# Alternate romanization for non-Latin names
]

# Distribution: 60% positive (duplicates), 40% negative (different entities)
POSITIVE_RATIO = 0.60

# ---------------------------------------------------------------------------
# LLM prompts
# ---------------------------------------------------------------------------

VARIATION_GENERATION_PROMPT = """You are generating name variations for entity resolution training data.

Entity name: {entity_name}
Entity type: {entity_type}
Variation type: {variation_type}

Generate ONE realistic variation of this entity name using the specified variation type:
- typo: Common misspelling (swap letters, missing letter, double letter)
- abbreviation: Shortened form (initials, acronyms, common abbreviations)
- nickname: Informal name (common nickname, shortened given name)
- case_variation: Different capitalization (all caps, lowercase, mixed)
- partial: Part of the name only (first name, last name, or key word)
- formal: More formal version (add title, full middle name, suffix)
- transliteration: Alternate spelling for non-English names

Rules:
- The variation must still plausibly refer to the SAME entity
- Be realistic — real people/organizations use these variations
- Respond with ONLY the variation, no explanation

Example: "Alice Johnson" with typo → "Alcie Johnson"
Example: "Nexus Technologies" with abbreviation → "Nexus Tech"
Example: "Carlos García" with nickname → "Charlie García"

Respond with ONLY the name variation:"""

NEGATIVE_ENTITY_PROMPT = """You are generating training data for entity resolution. Given an entity name, create a DIFFERENT entity that has a superficially similar name but refers to a completely different real-world entity.

Original entity: {entity_name}
Entity type: {entity_type}

Create ONE entity name that:
- Is DIFFERENT from the original (different person/organization/place)
- Has some surface-level similarity (shared word, similar spelling)
- Would be a realistic confusing pair for entity resolution
- Is 1-5 words long

Example: "Alice Johnson" → "Alice Jefferson" (different person)
Example: "Nexus Technologies" → "Nexus Therapeutics" (different company)
Example: "New York" → "New Orleans" (different city)

Respond with ONLY the entity name:"""


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class EntityPair:
    """A pair of entity mentions for entity resolution training."""
    canonical: str
    variant: str
    is_duplicate: bool
    entity_type: str
    variation_type: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "canonical": self.canonical,
            "variant": self.variant,
            "is_duplicate": self.is_duplicate,
            "entity_type": self.entity_type,
            "variation_type": self.variation_type,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class EntityResolutionGenerator:
    """Generates entity resolution training samples via LLM."""

    def __init__(
        self,
        llm: LLMDatasetGenerator | None = None,
        seed: int = 42,
        gen_config: GenerationConfig | None = None,
    ):
        self.llm = llm or LLMDatasetGenerator()
        self.rng = random.Random(seed)
        self.gen_config = gen_config or GenerationConfig(
            max_tokens=64,
            temperature=0.8,
            top_p=0.9,
        )

    def _sample_entity(self) -> tuple[str, str]:
        """Sample a random entity name and its type.

        Returns (entity_name, entity_type).
        """
        entity_type = self.rng.choice(list(ENTITY_NAMES.keys()))
        entity_name = self.rng.choice(ENTITY_NAMES[entity_type])
        return entity_name, entity_type

    def _sample_variation_type(self) -> str:
        """Sample a random variation type."""
        return self.rng.choice(VARIATION_TYPES)

    def _is_positive(self) -> bool:
        """Decide if the next sample should be a positive (duplicate) pair."""
        return self.rng.random() < POSITIVE_RATIO

    def generate_variation(
        self, entity_name: str, entity_type: str, variation_type: str
    ) -> str | None:
        """Generate a name variation for a given entity via LLM."""
        prompt = VARIATION_GENERATION_PROMPT.format(
            entity_name=entity_name,
            entity_type=entity_type,
            variation_type=variation_type,
        )
        try:
            result = self.llm.generate(prompt, config=self.gen_config)
            text = result.text.strip().strip('"').strip("'")
            if not text or len(text) < 2:
                return None
            # Sanity check: variation shouldn't be identical to original
            if text.lower() == entity_name.lower():
                return None
            return text
        except Exception:
            logger.warning(
                "Failed to generate variation for %s (%s)",
                entity_name, variation_type,
            )
            return None

    def generate_negative_entity(
        self, entity_name: str, entity_type: str
    ) -> str | None:
        """Generate a confusingly similar but different entity via LLM."""
        prompt = NEGATIVE_ENTITY_PROMPT.format(
            entity_name=entity_name,
            entity_type=entity_type,
        )
        try:
            result = self.llm.generate(prompt, config=self.gen_config)
            text = result.text.strip().strip('"').strip("'")
            if not text or len(text) < 2:
                return None
            # Sanity check: negative should be different from original
            if text.lower() == entity_name.lower():
                return None
            return text
        except Exception:
            logger.warning(
                "Failed to generate negative entity for %s", entity_name,
            )
            return None

    def generate_sample(self) -> EntityPair | None:
        """Generate a single entity resolution training sample.

        Returns None if generation fails at any step.
        """
        entity_name, entity_type = self._sample_entity()
        is_positive = self._is_positive()

        if is_positive:
            # Positive pair: same entity, different name variation
            variation_type = self._sample_variation_type()
            variant = self.generate_variation(entity_name, entity_type, variation_type)
            if variant is None:
                return None
            return EntityPair(
                canonical=entity_name,
                variant=variant,
                is_duplicate=True,
                entity_type=entity_type,
                variation_type=variation_type,
                metadata={
                    "generation_method": "llm_variation",
                },
            )
        else:
            # Negative pair: different entity, superficially similar name
            negative = self.generate_negative_entity(entity_name, entity_type)
            if negative is None:
                return None
            return EntityPair(
                canonical=entity_name,
                variant=negative,
                is_duplicate=False,
                entity_type=entity_type,
                variation_type="different_entity",
                metadata={
                    "generation_method": "llm_negative",
                },
            )

    def generate_samples(
        self, count: int = 2000, progress_interval: int = 100
    ) -> list[EntityPair]:
        """Generate multiple entity resolution training samples.

        Allows up to 2x attempts to reach the target count (some may fail).
        """
        samples: list[EntityPair] = []
        attempts = 0
        max_attempts = count * 2

        while len(samples) < count and attempts < max_attempts:
            sample = self.generate_sample()
            attempts += 1
            if sample is not None:
                samples.append(sample)
            if attempts % progress_interval == 0:
                logger.info(
                    "Progress: %d/%d samples (%d attempts)",
                    len(samples),
                    count,
                    attempts,
                )

        success_rate = (len(samples) / attempts * 100) if attempts > 0 else 0.0
        logger.info(
            "Complete: %d/%d samples in %d attempts (%.1f%% success rate)",
            len(samples),
            count,
            attempts,
            success_rate,
        )
        return samples

    def export(
        self, samples: list[EntityPair], output_path: str | Path
    ) -> int:
        """Export samples to JSONL file. Returns number of records written."""
        records = [s.to_dict() for s in samples]
        self.llm.export_jsonl(records, str(output_path))
        return len(records)

    def label_distribution(
        self, samples: list[EntityPair]
    ) -> dict[str, int]:
        """Return count per label (positive/negative) for the given samples."""
        dist = {"positive": 0, "negative": 0}
        for s in samples:
            if s.is_duplicate:
                dist["positive"] += 1
            else:
                dist["negative"] += 1
        return dist

    def variation_distribution(
        self, samples: list[EntityPair]
    ) -> dict[str, int]:
        """Return count per variation type for the given samples."""
        dist: dict[str, int] = {}
        for s in samples:
            dist[s.variation_type] = dist.get(s.variation_type, 0) + 1
        return dist


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate entity resolution training dataset"
    )
    parser.add_argument(
        "--count", type=int, default=2000, help="Number of samples to generate"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="entity_resolution_dataset.jsonl",
        help="Output JSONL file path",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=100,
        help="Log progress every N attempts",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    gen = EntityResolutionGenerator(seed=args.seed)
    samples = gen.generate_samples(
        count=args.count, progress_interval=args.progress_interval
    )
    written = gen.export(samples, args.output)

    label_dist = gen.label_distribution(samples)
    var_dist = gen.variation_distribution(samples)
    print(f"\nGenerated {written} entity resolution samples → {args.output}")
    print(f"Label distribution: {label_dist}")
    print(f"Variation types: {var_dist}")
    print(f"API usage: {gen.llm.usage_summary()}")


if __name__ == "__main__":
    main()
