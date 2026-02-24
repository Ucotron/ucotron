#!/usr/bin/env python3
"""
Relation Extraction (RE) dataset generator for Ucotron fine-tuning.

Generates training samples for relation extraction by:
1. Sampling seed entity pairs from a predefined knowledge base
2. Generating natural text containing those entities via GLM-5
3. Extracting entities and relations from the generated text via a second GLM-5 call
4. Exporting as JSONL with text, entities, relations fields

Usage:
    from re_dataset_generator import REDatasetGenerator

    gen = REDatasetGenerator()
    samples = gen.generate_samples(count=100)
    gen.export(samples, "re_dataset.jsonl")

    # Or via CLI:
    python re_dataset_generator.py --count 100 --output re_dataset.jsonl

Environment:
    FIREWORKS_API_KEY  - Required. Fireworks.ai API key.

Requirements:
    pip install openai>=1.0
"""

from __future__ import annotations

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
# Seed knowledge base
# ---------------------------------------------------------------------------

# Entity types that map to Ucotron NodeType variants
ENTITY_TYPES = ["Person", "Organization", "Location", "Event", "Concept", "Skill"]

# Relation types that map to Ucotron EdgeType variants
RELATION_TYPES = [
    "works_at",
    "lives_in",
    "born_in",
    "founded",
    "located_in",
    "part_of",
    "caused_by",
    "relates_to",
    "has_property",
    "studied_at",
    "created",
    "manages",
    "collaborates_with",
    "married_to",
    "child_of",
    "friend_of",
    "competitor_of",
    "acquired",
    "invested_in",
    "speaks",
]

# Seed entities organized by type for realistic text generation
SEED_ENTITIES: dict[str, list[str]] = {
    "Person": [
        "Alice Chen", "Bob Martinez", "Clara Schmidt", "David Okafor",
        "Elena Volkov", "Fatima Al-Rashid", "Gabriel Santos", "Hiroshi Tanaka",
        "Irene Dubois", "James O'Brien", "Keiko Nakamura", "Luis Herrera",
        "Maria Gonzalez", "Nikolai Petrov", "Olga Ivanova", "Pedro Silva",
        "Rachel Kim", "Samuel Nkomo", "Tomoko Yamada", "Umar Hassan",
        "Victoria Park", "Wei Zhang", "Xander Reeves", "Yuki Sato",
        "Zara Patel", "Andre Moreau", "Beatriz Costa", "Carlos Mendoza",
        "Diana Kuznetsova", "Erik Lindqvist",
    ],
    "Organization": [
        "Nexus AI Labs", "TerraFlow Inc", "CogniTech Solutions", "Vertex Dynamics",
        "Atlas Research", "Quantum Horizons", "SynapseWorks", "MindBridge Corp",
        "EcoVentures", "DataSphere", "OpenMind Foundation", "NeuralPath",
        "BioGenesis Labs", "CloudForge", "GreenLeaf Energy", "CyberGuard Security",
        "Stellar Robotics", "InnovateTech", "GlobalEdge Partners", "PulseMedia",
    ],
    "Location": [
        "Berlin", "Tokyo", "San Francisco", "Madrid", "São Paulo",
        "Mumbai", "London", "Seoul", "Mexico City", "Sydney",
        "Lagos", "Moscow", "Cairo", "Buenos Aires", "Singapore",
        "Stockholm", "Istanbul", "Dubai", "Toronto", "Zurich",
    ],
    "Event": [
        "AI Summit 2024", "TechCon Europe", "NeurIPS 2024", "Global Startup Week",
        "Data Science Bootcamp", "Innovation Hackathon", "Climate Action Forum",
        "Cybersecurity Conference", "Robotics Expo", "Blockchain Summit",
    ],
    "Concept": [
        "machine learning", "natural language processing", "graph databases",
        "knowledge graphs", "neural networks", "computer vision",
        "reinforcement learning", "distributed systems", "data pipelines",
        "semantic search", "entity resolution", "memory consolidation",
        "cognitive architecture", "embeddings", "transformer models",
    ],
    "Skill": [
        "Python programming", "Rust development", "data analysis",
        "project management", "public speaking", "system design",
        "database administration", "cloud architecture", "API design",
        "technical writing",
    ],
}

# Predefined entity-relation templates for diversity
RELATION_TEMPLATES: list[dict[str, str]] = [
    {"subject_type": "Person", "relation": "works_at", "object_type": "Organization"},
    {"subject_type": "Person", "relation": "lives_in", "object_type": "Location"},
    {"subject_type": "Person", "relation": "born_in", "object_type": "Location"},
    {"subject_type": "Person", "relation": "studied_at", "object_type": "Organization"},
    {"subject_type": "Person", "relation": "collaborates_with", "object_type": "Person"},
    {"subject_type": "Person", "relation": "married_to", "object_type": "Person"},
    {"subject_type": "Person", "relation": "child_of", "object_type": "Person"},
    {"subject_type": "Person", "relation": "friend_of", "object_type": "Person"},
    {"subject_type": "Person", "relation": "manages", "object_type": "Organization"},
    {"subject_type": "Person", "relation": "created", "object_type": "Concept"},
    {"subject_type": "Person", "relation": "speaks", "object_type": "Concept"},
    {"subject_type": "Organization", "relation": "located_in", "object_type": "Location"},
    {"subject_type": "Organization", "relation": "part_of", "object_type": "Organization"},
    {"subject_type": "Organization", "relation": "competitor_of", "object_type": "Organization"},
    {"subject_type": "Organization", "relation": "acquired", "object_type": "Organization"},
    {"subject_type": "Organization", "relation": "invested_in", "object_type": "Organization"},
    {"subject_type": "Organization", "relation": "founded", "object_type": "Person"},
    {"subject_type": "Location", "relation": "part_of", "object_type": "Location"},
    {"subject_type": "Event", "relation": "located_in", "object_type": "Location"},
    {"subject_type": "Concept", "relation": "relates_to", "object_type": "Concept"},
    {"subject_type": "Concept", "relation": "caused_by", "object_type": "Concept"},
    {"subject_type": "Concept", "relation": "has_property", "object_type": "Concept"},
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class EntitySpan:
    """An entity mention in text with character offsets."""
    text: str
    entity_type: str
    start: int
    end: int


@dataclass
class Relation:
    """A relation between two entity spans."""
    subject: str
    subject_type: str
    relation: str
    object: str
    object_type: str


@dataclass
class RESample:
    """A single relation extraction training sample."""
    text: str
    entities: list[EntitySpan]
    relations: list[Relation]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable dict for JSONL export."""
        return {
            "text": self.text,
            "entities": [
                {
                    "text": e.text,
                    "type": e.entity_type,
                    "start": e.start,
                    "end": e.end,
                }
                for e in self.entities
            ],
            "relations": [
                {
                    "subject": r.subject,
                    "subject_type": r.subject_type,
                    "relation": r.relation,
                    "object": r.object,
                    "object_type": r.object_type,
                }
                for r in self.relations
            ],
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

GENERATION_SYSTEM_PROMPT = """\
You are a training data generator for a relation extraction model. \
Your task is to write natural, diverse sentences that contain the given \
entities and express the specified relationship between them. \
Write realistic text as it might appear in news, conversations, or documents. \
Vary sentence structure, length, and style. Include additional context \
to make sentences feel natural."""

GENERATION_USER_TEMPLATE = """\
Write a natural sentence (1-3 sentences) that mentions these entities \
and expresses the relationship between them:

Subject: {subject} (type: {subject_type})
Object: {object} (type: {object_type})
Relationship: {relation}

Requirements:
- Use the exact entity names as given
- The relationship should be clearly expressed in the text
- Add surrounding context for naturalness
- Vary your writing style

Respond with ONLY the generated text, no explanations."""

EXTRACTION_SYSTEM_PROMPT = """\
You are a precise relation extraction model. Given a text, identify all entities \
and the relations between them. Be exhaustive and accurate."""

EXTRACTION_USER_TEMPLATE = """\
Extract all entities and relations from the following text. Return the result \
as a JSON object with two arrays: "entities" and "relations".

Text: {text}

Each entity should have: "text" (exact span), "type" (Person, Organization, \
Location, Event, Concept, or Skill).

Each relation should have: "subject" (entity text), "subject_type", "relation" \
(one of: works_at, lives_in, born_in, founded, located_in, part_of, caused_by, \
relates_to, has_property, studied_at, created, manages, collaborates_with, \
married_to, child_of, friend_of, competitor_of, acquired, invested_in, speaks), \
"object" (entity text), "object_type".

Respond with ONLY the JSON object."""


# ---------------------------------------------------------------------------
# Generator class
# ---------------------------------------------------------------------------

class REDatasetGenerator:
    """
    Generates relation extraction training data using LLM synthesis.

    Two-step pipeline per sample:
    1. Generate natural text from seed entity pair + relation
    2. Extract entities and relations from the generated text

    This produces high-quality labeled data where the LLM acts as both
    the data generator and the annotation oracle.
    """

    def __init__(
        self,
        llm: LLMDatasetGenerator | None = None,
        seed: int = 42,
        gen_config: GenerationConfig | None = None,
        extract_config: GenerationConfig | None = None,
    ):
        """
        Initialize the RE dataset generator.

        Args:
            llm: Pre-configured LLMDatasetGenerator (creates one if None).
            seed: Random seed for entity pair sampling.
            gen_config: Config for text generation step.
            extract_config: Config for extraction step.
        """
        self.llm = llm or LLMDatasetGenerator()
        self.rng = random.Random(seed)
        self.gen_config = gen_config or GenerationConfig(
            max_tokens=512,
            temperature=0.8,
        )
        self.extract_config = extract_config or GenerationConfig(
            max_tokens=1024,
            temperature=0.1,  # low temperature for precise extraction
            response_format={"type": "json_object"},
        )

    def sample_entity_pair(self) -> dict[str, str]:
        """
        Sample a random entity pair with a relation from seed data.

        Returns:
            Dict with subject, subject_type, object, object_type, relation.
        """
        template = self.rng.choice(RELATION_TEMPLATES)
        subject_type = template["subject_type"]
        object_type = template["object_type"]
        relation = template["relation"]

        subject = self.rng.choice(SEED_ENTITIES[subject_type])
        obj = self.rng.choice(SEED_ENTITIES[object_type])

        # Avoid self-references
        if subject_type == object_type:
            attempts = 0
            while obj == subject and attempts < 10:
                obj = self.rng.choice(SEED_ENTITIES[object_type])
                attempts += 1

        return {
            "subject": subject,
            "subject_type": subject_type,
            "object": obj,
            "object_type": object_type,
            "relation": relation,
        }

    def generate_text(self, pair: dict[str, str]) -> str | None:
        """
        Generate natural text containing the given entity pair.

        Args:
            pair: Entity pair dict from sample_entity_pair().

        Returns:
            Generated text or None if generation fails.
        """
        prompt = GENERATION_USER_TEMPLATE.format(**pair)
        try:
            result = self.llm.generate(
                user_prompt=prompt,
                system_prompt=GENERATION_SYSTEM_PROMPT,
                config=self.gen_config,
            )
            text = result.text.strip()
            if not text:
                return None
            return text
        except RuntimeError:
            logger.warning("Text generation failed for pair: %s", pair)
            return None

    def extract_relations(self, text: str) -> dict[str, Any] | None:
        """
        Extract entities and relations from text via LLM.

        Args:
            text: Natural language text to analyze.

        Returns:
            Dict with 'entities' and 'relations' lists, or None on failure.
        """
        prompt = EXTRACTION_USER_TEMPLATE.format(text=text)
        try:
            result = self.llm.generate_json(
                user_prompt=prompt,
                system_prompt=EXTRACTION_SYSTEM_PROMPT,
                config=self.extract_config,
            )
            # Validate structure
            if "entities" not in result or "relations" not in result:
                logger.warning("Extraction missing required fields: %s", list(result.keys()))
                return None
            return result
        except (RuntimeError, ValueError) as e:
            logger.warning("Extraction failed: %s", e)
            return None

    def _build_sample(
        self,
        text: str,
        extraction: dict[str, Any],
        pair: dict[str, str],
    ) -> RESample | None:
        """
        Build an RESample from generated text and extraction results.

        Validates entity spans exist in text and builds character offsets.
        """
        entities: list[EntitySpan] = []
        for ent in extraction.get("entities", []):
            ent_text = ent.get("text", "")
            ent_type = ent.get("type", "")
            if not ent_text or not ent_type:
                continue

            # Find exact span in text
            start = text.find(ent_text)
            if start == -1:
                # Try case-insensitive search
                lower_text = text.lower()
                start = lower_text.find(ent_text.lower())
            if start == -1:
                continue  # entity not found in text, skip

            entities.append(EntitySpan(
                text=ent_text,
                entity_type=ent_type,
                start=start,
                end=start + len(ent_text),
            ))

        relations: list[Relation] = []
        entity_texts = {e.text.lower() for e in entities}
        for rel in extraction.get("relations", []):
            subj = rel.get("subject", "")
            obj = rel.get("object", "")
            # Only include relations whose entities we found
            if subj.lower() in entity_texts and obj.lower() in entity_texts:
                relations.append(Relation(
                    subject=subj,
                    subject_type=rel.get("subject_type", ""),
                    relation=rel.get("relation", ""),
                    object=obj,
                    object_type=rel.get("object_type", ""),
                ))

        if not entities or not relations:
            return None

        return RESample(
            text=text,
            entities=entities,
            relations=relations,
            metadata={
                "seed_subject": pair["subject"],
                "seed_object": pair["object"],
                "seed_relation": pair["relation"],
            },
        )

    def generate_sample(self) -> RESample | None:
        """
        Generate a single RE training sample.

        Returns:
            RESample or None if any step fails.
        """
        pair = self.sample_entity_pair()
        text = self.generate_text(pair)
        if text is None:
            return None

        extraction = self.extract_relations(text)
        if extraction is None:
            return None

        return self._build_sample(text, extraction, pair)

    def generate_samples(
        self,
        count: int = 100,
        progress_interval: int = 50,
    ) -> list[RESample]:
        """
        Generate multiple RE training samples.

        Retries on failure to reach the target count. Will attempt up to
        2x the requested count before giving up.

        Args:
            count: Target number of successful samples.
            progress_interval: Log progress every N attempts.

        Returns:
            List of successfully generated RESamples.
        """
        samples: list[RESample] = []
        attempts = 0
        max_attempts = count * 2  # allow 2x attempts for failures

        while len(samples) < count and attempts < max_attempts:
            sample = self.generate_sample()
            attempts += 1

            if sample is not None:
                samples.append(sample)

            if attempts % progress_interval == 0:
                logger.info(
                    "RE generation progress: %d/%d samples (%d attempts)",
                    len(samples), count, attempts,
                )

        logger.info(
            "RE generation complete: %d/%d samples in %d attempts (%.1f%% success rate)",
            len(samples), count, attempts,
            len(samples) / max(attempts, 1) * 100,
        )
        return samples

    def export(
        self,
        samples: list[RESample],
        output_path: str | Path,
    ) -> int:
        """
        Export RE samples to JSONL file.

        Args:
            samples: List of RESamples to export.
            output_path: Destination file path.

        Returns:
            Number of records written.
        """
        records = [s.to_dict() for s in samples]
        return self.llm.export_jsonl(records, output_path)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    """Generate RE dataset via CLI."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate relation extraction training dataset"
    )
    parser.add_argument(
        "--count", type=int, default=100,
        help="Number of samples to generate (default: 100)",
    )
    parser.add_argument(
        "--output", type=str, default="re_dataset.jsonl",
        help="Output JSONL file path (default: re_dataset.jsonl)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--progress-interval", type=int, default=50,
        help="Log progress every N attempts (default: 50)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )

    gen = REDatasetGenerator(seed=args.seed)
    samples = gen.generate_samples(
        count=args.count,
        progress_interval=args.progress_interval,
    )
    written = gen.export(samples, args.output)
    print(f"\nGenerated {written} RE samples → {args.output}")
    print(f"API usage: {gen.llm.usage_summary()}")


if __name__ == "__main__":
    main()
