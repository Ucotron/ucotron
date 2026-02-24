#!/usr/bin/env python3
"""
Contradiction dataset generator for Ucotron fine-tuning.

Generates training samples for contradiction detection by:
1. Generating factual statements about entities via GLM-5
2. Generating contradicting statements for each fact via GLM-5
3. Classifying contradiction type (contradicts, supersedes, ambiguous)
4. Exporting as JSONL with fact_a, fact_b, label, metadata fields

The dataset follows Ucotron's three-strategy conflict resolution:
- Temporal: newer fact supersedes older (>1 year gap)
- Confidence: higher confidence overrides lower (>0.3 gap)
- Ambiguous: close timestamps + close confidence → contradiction

Usage:
    from contradiction_generator import ContradictionGenerator

    gen = ContradictionGenerator()
    samples = gen.generate_samples(count=3000)
    gen.export(samples, "contradiction_dataset.jsonl")

    # Or via CLI:
    python contradiction_generator.py --count 3000 --output contradiction_dataset.jsonl

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
# Seed knowledge base for fact generation
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

# Predicates that can have contradicting values
PREDICATES: list[dict[str, Any]] = [
    {
        "predicate": "lives_in",
        "subject_type": "Person",
        "description": "city or country of residence",
        "value_type": "Location",
    },
    {
        "predicate": "works_at",
        "subject_type": "Person",
        "description": "employer organization",
        "value_type": "Organization",
    },
    {
        "predicate": "job_title",
        "subject_type": "Person",
        "description": "professional title or role",
        "value_type": "free_text",
    },
    {
        "predicate": "founded_in",
        "subject_type": "Organization",
        "description": "year the organization was founded",
        "value_type": "free_text",
    },
    {
        "predicate": "headquartered_in",
        "subject_type": "Organization",
        "description": "city where headquarters are located",
        "value_type": "Location",
    },
    {
        "predicate": "population",
        "subject_type": "Location",
        "description": "population count or estimate",
        "value_type": "free_text",
    },
    {
        "predicate": "speaks",
        "subject_type": "Person",
        "description": "primary language spoken",
        "value_type": "free_text",
    },
    {
        "predicate": "studied_at",
        "subject_type": "Person",
        "description": "university or educational institution",
        "value_type": "free_text",
    },
    {
        "predicate": "ceo",
        "subject_type": "Organization",
        "description": "current CEO or director",
        "value_type": "Person",
    },
    {
        "predicate": "capital_of",
        "subject_type": "Location",
        "description": "country that this city is capital of",
        "value_type": "free_text",
    },
    {
        "predicate": "nationality",
        "subject_type": "Person",
        "description": "country of citizenship",
        "value_type": "free_text",
    },
    {
        "predicate": "industry",
        "subject_type": "Organization",
        "description": "primary industry or sector",
        "value_type": "free_text",
    },
]

# Contradiction labels matching Ucotron's resolution strategies
LABEL_SUPERSEDES = "supersedes"
LABEL_CONTRADICTS = "contradicts"
LABEL_AMBIGUOUS = "ambiguous"
LABEL_AGREES = "agrees"  # negative examples (no contradiction)

LABELS = [LABEL_SUPERSEDES, LABEL_CONTRADICTS, LABEL_AMBIGUOUS, LABEL_AGREES]

# Distribution: 35% supersedes, 30% contradicts, 20% ambiguous, 15% agrees
LABEL_WEIGHTS = [0.35, 0.30, 0.20, 0.15]

# ---------------------------------------------------------------------------
# LLM prompts
# ---------------------------------------------------------------------------

FACT_GENERATION_PROMPT = """You are a knowledge base assistant. Generate a factual statement as a single natural sentence.

Subject: {subject}
Predicate: {predicate} ({description})

Write ONE factual sentence stating a specific {description} for {subject}. The sentence must:
- Be a clear, unambiguous factual claim
- Contain both the subject and a specific value for the predicate
- Be 10-30 words long
- Sound natural and realistic

Respond with ONLY the sentence, no explanation."""

CONTRADICTION_GENERATION_PROMPT = """You are generating training data for contradiction detection. Given a factual statement, create a contradicting statement.

Original fact: {original_fact}
Subject: {subject}
Predicate: {predicate} ({description})

Create ONE sentence that CONTRADICTS the original by stating a DIFFERENT {description} for {subject}. The contradiction must:
- Be about the same subject and predicate
- State a clearly different, incompatible value
- Be 10-30 words long
- Sound natural and realistic

Respond with ONLY the contradicting sentence, no explanation."""

AGREEMENT_GENERATION_PROMPT = """You are generating training data. Given a factual statement, rephrase it as an agreeing statement.

Original fact: {original_fact}
Subject: {subject}
Predicate: {predicate} ({description})

Create ONE sentence that AGREES with the original, expressing the SAME information in different words. The sentence must:
- State the same {description} for {subject} (same value, different phrasing)
- Be 10-30 words long
- Sound natural and realistic

Respond with ONLY the sentence, no explanation."""


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class FactStatement:
    """A single factual statement (subject-predicate-object triple)."""
    subject: str
    predicate: str
    object_value: str
    text: str
    confidence: float
    timestamp: int


@dataclass
class ContradictionSample:
    """A pair of facts with a contradiction label."""
    fact_a: FactStatement
    fact_b: FactStatement
    label: str  # supersedes, contradicts, ambiguous, agrees
    resolution_strategy: str  # temporal, confidence, ambiguous, none
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "fact_a": {
                "subject": self.fact_a.subject,
                "predicate": self.fact_a.predicate,
                "object": self.fact_a.object_value,
                "text": self.fact_a.text,
                "confidence": self.fact_a.confidence,
                "timestamp": self.fact_a.timestamp,
            },
            "fact_b": {
                "subject": self.fact_b.subject,
                "predicate": self.fact_b.predicate,
                "object": self.fact_b.object_value,
                "text": self.fact_b.text,
                "confidence": self.fact_b.confidence,
                "timestamp": self.fact_b.timestamp,
            },
            "label": self.label,
            "resolution_strategy": self.resolution_strategy,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

# One year in seconds (Ucotron's temporal threshold)
ONE_YEAR_SECS = 31_536_000
# Base timestamp: 2024-01-01 00:00:00 UTC
BASE_TIMESTAMP = 1_704_067_200


class ContradictionGenerator:
    """Generates contradiction detection training samples via LLM."""

    def __init__(
        self,
        llm: LLMDatasetGenerator | None = None,
        seed: int = 42,
        gen_config: GenerationConfig | None = None,
    ):
        self.llm = llm or LLMDatasetGenerator()
        self.rng = random.Random(seed)
        self.gen_config = gen_config or GenerationConfig(
            max_tokens=128,
            temperature=0.8,
            top_p=0.9,
        )

    def _sample_predicate(self) -> dict[str, Any]:
        """Sample a random predicate definition."""
        return self.rng.choice(PREDICATES)

    def _sample_subject(self, subject_type: str) -> str:
        """Sample a random entity name of the given type."""
        entities = ENTITY_NAMES.get(subject_type, ENTITY_NAMES["Person"])
        return self.rng.choice(entities)

    def _sample_label(self) -> str:
        """Sample a contradiction label according to target distribution."""
        return self.rng.choices(LABELS, weights=LABEL_WEIGHTS, k=1)[0]

    def _generate_timestamps_and_confidence(
        self, label: str
    ) -> tuple[int, int, float, float]:
        """Generate timestamp and confidence pairs appropriate for the label.

        Returns (timestamp_a, timestamp_b, confidence_a, confidence_b).
        """
        base_ts = BASE_TIMESTAMP + self.rng.randint(0, ONE_YEAR_SECS)

        if label == LABEL_SUPERSEDES:
            # Temporal: >1 year gap, newer supersedes older
            gap = ONE_YEAR_SECS + self.rng.randint(1, ONE_YEAR_SECS)
            ts_a = base_ts
            ts_b = base_ts + gap
            conf_a = round(self.rng.uniform(0.5, 0.95), 2)
            conf_b = round(self.rng.uniform(0.5, 0.95), 2)
            return ts_a, ts_b, conf_a, conf_b

        if label == LABEL_CONTRADICTS:
            # Confidence: close timestamps, >0.3 confidence gap
            ts_a = base_ts
            ts_b = base_ts + self.rng.randint(0, ONE_YEAR_SECS // 2)
            high_conf = round(self.rng.uniform(0.75, 0.98), 2)
            low_conf = round(high_conf - self.rng.uniform(0.31, 0.5), 2)
            low_conf = max(0.05, low_conf)
            if self.rng.random() < 0.5:
                return ts_a, ts_b, high_conf, low_conf
            return ts_a, ts_b, low_conf, high_conf

        if label == LABEL_AMBIGUOUS:
            # Close timestamps AND close confidence
            ts_a = base_ts
            ts_b = base_ts + self.rng.randint(0, ONE_YEAR_SECS // 4)
            conf_a = round(self.rng.uniform(0.5, 0.85), 2)
            conf_b = round(conf_a + self.rng.uniform(-0.2, 0.2), 2)
            conf_b = max(0.1, min(0.99, conf_b))
            return ts_a, ts_b, conf_a, conf_b

        # AGREES: timestamps and confidence don't matter for resolution
        ts_a = base_ts
        ts_b = base_ts + self.rng.randint(0, ONE_YEAR_SECS)
        conf_a = round(self.rng.uniform(0.5, 0.95), 2)
        conf_b = round(self.rng.uniform(0.5, 0.95), 2)
        return ts_a, ts_b, conf_a, conf_b

    def _resolution_strategy(self, label: str) -> str:
        """Return the resolution strategy string for a label."""
        if label == LABEL_SUPERSEDES:
            return "temporal"
        if label == LABEL_CONTRADICTS:
            return "confidence"
        if label == LABEL_AMBIGUOUS:
            return "ambiguous"
        return "none"

    def generate_original_fact(
        self, subject: str, predicate_info: dict[str, Any]
    ) -> str | None:
        """Generate a factual statement about a subject via LLM."""
        prompt = FACT_GENERATION_PROMPT.format(
            subject=subject,
            predicate=predicate_info["predicate"],
            description=predicate_info["description"],
        )
        try:
            result = self.llm.generate(prompt, config=self.gen_config)
            text = result.text.strip().strip('"').strip("'")
            if not text or len(text) < 5:
                return None
            return text
        except Exception:
            logger.warning("Failed to generate fact for %s/%s", subject, predicate_info["predicate"])
            return None

    def generate_contradiction(
        self, original_fact: str, subject: str, predicate_info: dict[str, Any]
    ) -> str | None:
        """Generate a contradicting statement for an original fact via LLM."""
        prompt = CONTRADICTION_GENERATION_PROMPT.format(
            original_fact=original_fact,
            subject=subject,
            predicate=predicate_info["predicate"],
            description=predicate_info["description"],
        )
        try:
            result = self.llm.generate(prompt, config=self.gen_config)
            text = result.text.strip().strip('"').strip("'")
            if not text or len(text) < 5:
                return None
            return text
        except Exception:
            logger.warning("Failed to generate contradiction for: %s", original_fact[:50])
            return None

    def generate_agreement(
        self, original_fact: str, subject: str, predicate_info: dict[str, Any]
    ) -> str | None:
        """Generate an agreeing restatement of an original fact via LLM."""
        prompt = AGREEMENT_GENERATION_PROMPT.format(
            original_fact=original_fact,
            subject=subject,
            predicate=predicate_info["predicate"],
            description=predicate_info["description"],
        )
        try:
            result = self.llm.generate(prompt, config=self.gen_config)
            text = result.text.strip().strip('"').strip("'")
            if not text or len(text) < 5:
                return None
            return text
        except Exception:
            logger.warning("Failed to generate agreement for: %s", original_fact[:50])
            return None

    def _extract_object_from_text(
        self, text: str, subject: str, predicate_info: dict[str, Any]
    ) -> str:
        """Extract the object value from a generated sentence.

        Uses a simple heuristic: the value is whatever comes after the subject
        in the context of the predicate. Falls back to the full text if parsing fails.
        """
        # For value_type == Location, try matching known locations
        if predicate_info.get("value_type") == "Location":
            for loc in ENTITY_NAMES.get("Location", []):
                if loc.lower() in text.lower():
                    return loc
        # For value_type == Organization, try matching known orgs
        if predicate_info.get("value_type") == "Organization":
            for org in ENTITY_NAMES.get("Organization", []):
                if org.lower() in text.lower():
                    return org
        # For value_type == Person, try matching known people
        if predicate_info.get("value_type") == "Person":
            for person in ENTITY_NAMES.get("Person", []):
                if person.lower() in text.lower() and person.lower() != subject.lower():
                    return person

        # Fallback: use the text itself as the object (trimmed)
        return text.strip()[:100]

    def generate_sample(self) -> ContradictionSample | None:
        """Generate a single contradiction training sample.

        Returns None if generation fails at any step.
        """
        label = self._sample_label()
        pred_info = self._sample_predicate()
        subject = self._sample_subject(pred_info["subject_type"])

        # Step 1: generate original fact
        fact_text_a = self.generate_original_fact(subject, pred_info)
        if fact_text_a is None:
            return None

        # Step 2: generate second fact (contradiction or agreement)
        if label == LABEL_AGREES:
            fact_text_b = self.generate_agreement(fact_text_a, subject, pred_info)
        else:
            fact_text_b = self.generate_contradiction(fact_text_a, subject, pred_info)

        if fact_text_b is None:
            return None

        # Step 3: extract object values
        object_a = self._extract_object_from_text(fact_text_a, subject, pred_info)
        object_b = self._extract_object_from_text(fact_text_b, subject, pred_info)

        # Step 4: generate timestamps and confidence
        ts_a, ts_b, conf_a, conf_b = self._generate_timestamps_and_confidence(label)

        fact_a = FactStatement(
            subject=subject,
            predicate=pred_info["predicate"],
            object_value=object_a,
            text=fact_text_a,
            confidence=conf_a,
            timestamp=ts_a,
        )
        fact_b = FactStatement(
            subject=subject,
            predicate=pred_info["predicate"],
            object_value=object_b,
            text=fact_text_b,
            confidence=conf_b,
            timestamp=ts_b,
        )

        return ContradictionSample(
            fact_a=fact_a,
            fact_b=fact_b,
            label=label,
            resolution_strategy=self._resolution_strategy(label),
            metadata={
                "subject_type": pred_info["subject_type"],
                "predicate": pred_info["predicate"],
                "value_type": pred_info.get("value_type", "free_text"),
            },
        )

    def generate_samples(
        self, count: int = 3000, progress_interval: int = 100
    ) -> list[ContradictionSample]:
        """Generate multiple contradiction training samples.

        Allows up to 2x attempts to reach the target count (some may fail).
        """
        samples: list[ContradictionSample] = []
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
        self, samples: list[ContradictionSample], output_path: str | Path
    ) -> int:
        """Export samples to JSONL file. Returns number of records written."""
        records = [s.to_dict() for s in samples]
        self.llm.export_jsonl(records, str(output_path))
        return len(records)

    def label_distribution(
        self, samples: list[ContradictionSample]
    ) -> dict[str, int]:
        """Return count per label for the given samples."""
        dist: dict[str, int] = {label: 0 for label in LABELS}
        for s in samples:
            dist[s.label] = dist.get(s.label, 0) + 1
        return dist


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate contradiction detection training dataset"
    )
    parser.add_argument(
        "--count", type=int, default=3000, help="Number of samples to generate"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="contradiction_dataset.jsonl",
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

    gen = ContradictionGenerator(seed=args.seed)
    samples = gen.generate_samples(
        count=args.count, progress_interval=args.progress_interval
    )
    written = gen.export(samples, args.output)

    dist = gen.label_distribution(samples)
    print(f"\nGenerated {written} contradiction samples → {args.output}")
    print(f"Label distribution: {dist}")
    print(f"API usage: {gen.llm.usage_summary()}")


if __name__ == "__main__":
    main()
