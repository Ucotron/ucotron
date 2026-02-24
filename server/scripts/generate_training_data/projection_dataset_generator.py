#!/usr/bin/env python3
"""
Projection layer training dataset generator for Ucotron cross-modal bridge.

Generates paired (CLIP_embedding, MiniLM_embedding) training data for the
projection MLP that maps 512-dim CLIP space → 384-dim MiniLM space.

Pipeline:
1. Generate diverse image descriptions via GLM-5 (Fireworks API)
2. Encode each description with CLIP text encoder (512-dim)
3. Encode each description with MiniLM (384-dim)
4. Export paired embeddings as JSONL

The resulting dataset trains the ProjectionLayerPipeline (see
ucotron_extraction/src/cross_modal.rs) to bridge image and text search spaces.

Usage:
    from projection_dataset_generator import ProjectionDatasetGenerator

    gen = ProjectionDatasetGenerator()
    pairs = gen.generate_pairs(count=1000)
    gen.export(pairs, "projection_dataset.jsonl")

    # Or via CLI:
    python projection_dataset_generator.py --count 50000 --output projection_dataset.jsonl

Environment:
    FIREWORKS_API_KEY  - Required. Fireworks.ai API key.

Requirements:
    pip install openai>=1.0 transformers torch sentence-transformers
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from llm_dataset_generator import (
    LLMDatasetGenerator,
    GenerationConfig,
    GenerationResult,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CLIP_EMBED_DIM = 512
MINILM_EMBED_DIM = 384

# System prompt for generating diverse image descriptions
DESCRIPTION_SYSTEM_PROMPT = """\
You are a visual scene descriptor. Generate vivid, diverse image descriptions \
suitable for training a vision-language model. Each description should be a \
single paragraph (2-4 sentences) describing a specific scene, object, or \
situation as if you were describing a photograph or painting.

Requirements:
- Be specific about visual details (colors, shapes, positions, lighting)
- Cover diverse categories: nature, urban, people, animals, food, technology, \
  art, architecture, sports, science
- Vary complexity from simple object descriptions to complex multi-element scenes
- Use natural, descriptive English
- Do NOT use lists or bullet points
- Do NOT reference camera settings or photography techniques"""

# Prompt templates for diverse category coverage
CATEGORY_PROMPTS = [
    "Describe a photograph of a natural landscape with distinctive features.",
    "Describe an urban street scene with people and buildings.",
    "Describe a close-up photograph of an animal in its habitat.",
    "Describe a photograph of food arranged on a table or plate.",
    "Describe a photograph showing modern technology or gadgets.",
    "Describe a painting or artwork with distinctive style and subjects.",
    "Describe a photograph of an architectural landmark or building.",
    "Describe a sports action shot capturing a key moment.",
    "Describe a scientific or medical image showing a process or structure.",
    "Describe a photograph of people interacting in a social setting.",
    "Describe a photograph of a vehicle or mode of transportation.",
    "Describe a photograph of a workspace or office environment.",
    "Describe a photograph of a garden or botanical setting with flowers.",
    "Describe a photograph of weather phenomena or atmospheric conditions.",
    "Describe a photograph of a marketplace or shopping area.",
    "Describe a nighttime photograph of a city skyline.",
    "Describe a photograph of children playing outdoors.",
    "Describe a photograph of a musical instrument or performance.",
    "Describe a photograph of a historical monument or ruins.",
    "Describe a macro photograph of an insect or small object.",
    "Describe a photograph of a coastal or ocean scene.",
    "Describe a photograph of mountains or highlands.",
    "Describe a photograph of a library, museum, or cultural institution.",
    "Describe a photograph of industrial machinery or a factory.",
    "Describe a photograph of a festive celebration or holiday scene.",
]

# Batch prompt for generating multiple descriptions at once
BATCH_PROMPT_TEMPLATE = """\
Generate {count} distinct image descriptions, one per line. \
Each description should be 2-4 sentences. Cover different categories: \
{categories}. Number each description (1., 2., etc.).

Make each description unique and visually specific."""


# ---------------------------------------------------------------------------
# Embedding encoder protocol
# ---------------------------------------------------------------------------

class EmbeddingEncoder(Protocol):
    """Protocol for text-to-embedding encoders."""

    def encode(self, texts: list[str]) -> list[list[float]]:
        """Encode texts into embedding vectors."""
        ...

    @property
    def dim(self) -> int:
        """Embedding dimensionality."""
        ...


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class EmbeddingPair:
    """A paired (CLIP, MiniLM) embedding for projection layer training."""

    description: str
    clip_embedding: list[float]
    minilm_embedding: list[float]
    category: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "description": self.description,
            "clip_embedding": self.clip_embedding,
            "minilm_embedding": self.minilm_embedding,
            "category": self.category,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EmbeddingPair:
        return cls(
            description=d["description"],
            clip_embedding=d["clip_embedding"],
            minilm_embedding=d["minilm_embedding"],
            category=d.get("category", ""),
        )


@dataclass
class GenerationStats:
    """Statistics for a dataset generation run."""

    total_requested: int = 0
    descriptions_generated: int = 0
    pairs_encoded: int = 0
    failed_generations: int = 0
    failed_encodings: int = 0
    elapsed_seconds: float = 0.0

    @property
    def success_rate(self) -> float:
        return self.pairs_encoded / max(self.total_requested, 1)


# ---------------------------------------------------------------------------
# CLIP encoder wrapper
# ---------------------------------------------------------------------------

class CLIPTextEncoder:
    """Encodes text using CLIP text encoder (produces 512-dim embeddings).

    Uses transformers library to load CLIP model. Falls back to
    sentence-transformers if available.
    """

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        try:
            from transformers import CLIPTokenizer, CLIPTextModel
            import torch
        except ImportError:
            raise ImportError(
                "CLIP encoder requires: pip install transformers torch"
            )

        self._tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self._model = CLIPTextModel.from_pretrained(model_name)
        self._model.eval()
        self._torch = torch

    def encode(self, texts: list[str]) -> list[list[float]]:
        with self._torch.no_grad():
            inputs = self._tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=77,
                return_tensors="pt",
            )
            outputs = self._model(**inputs)
            # Use pooler_output (CLS token projected) for 512-dim
            embeddings = outputs.pooler_output
            # L2 normalize
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
            return embeddings.cpu().tolist()

    @property
    def dim(self) -> int:
        return CLIP_EMBED_DIM


class MiniLMEncoder:
    """Encodes text using all-MiniLM-L6-v2 (produces 384-dim embeddings).

    Uses sentence-transformers for encoding.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "MiniLM encoder requires: pip install sentence-transformers"
            )

        self._model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> list[list[float]]:
        embeddings = self._model.encode(texts, normalize_embeddings=True)
        return embeddings.tolist()

    @property
    def dim(self) -> int:
        return MINILM_EMBED_DIM


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

class ProjectionDatasetGenerator:
    """Generates paired embedding datasets for projection layer training.

    Workflow:
    1. Use LLM (GLM-5 via Fireworks) to generate image descriptions
    2. Encode each description with both CLIP text encoder and MiniLM
    3. Export (CLIP_embedding, MiniLM_embedding) pairs as JSONL

    The resulting dataset trains a projection MLP (512→384) that bridges
    CLIP image embeddings to MiniLM text embedding space.
    """

    def __init__(
        self,
        llm: LLMDatasetGenerator | None = None,
        clip_encoder: EmbeddingEncoder | None = None,
        minilm_encoder: EmbeddingEncoder | None = None,
        seed: int = 42,
        batch_size: int = 10,
        encoding_batch_size: int = 64,
    ):
        self.llm = llm or LLMDatasetGenerator()
        self.clip_encoder = clip_encoder
        self.minilm_encoder = minilm_encoder
        self.seed = seed
        self.batch_size = batch_size
        self.encoding_batch_size = encoding_batch_size
        self._rng = random.Random(seed)
        self._stats = GenerationStats()

    def _init_encoders(self) -> None:
        """Lazily initialize encoders if not provided."""
        if self.clip_encoder is None:
            self.clip_encoder = CLIPTextEncoder()
        if self.minilm_encoder is None:
            self.minilm_encoder = MiniLMEncoder()

    def _pick_prompt(self) -> str:
        """Select a random category prompt."""
        return self._rng.choice(CATEGORY_PROMPTS)

    def _generate_descriptions(self, count: int) -> list[tuple[str, str]]:
        """Generate image descriptions via LLM.

        Returns list of (description, category) tuples.
        """
        descriptions: list[tuple[str, str]] = []
        remaining = count

        while remaining > 0:
            batch = min(remaining, self.batch_size)

            if batch >= 5:
                # Use batch prompt for efficiency
                categories = ", ".join(
                    self._rng.sample(
                        CATEGORY_PROMPTS,
                        min(5, len(CATEGORY_PROMPTS)),
                    )
                )
                prompt = BATCH_PROMPT_TEMPLATE.format(
                    count=batch,
                    categories=categories,
                )
                try:
                    result = self.llm.generate(
                        user_prompt=prompt,
                        system_prompt=DESCRIPTION_SYSTEM_PROMPT,
                        config=GenerationConfig(
                            max_tokens=4096,
                            temperature=0.9,
                            top_p=0.95,
                        ),
                    )
                    parsed = self._parse_numbered_list(result.text)
                    for desc in parsed:
                        if len(desc.strip()) >= 20:
                            descriptions.append((desc.strip(), "mixed"))
                            remaining -= 1
                            if remaining <= 0:
                                break
                except Exception as e:
                    logger.warning("Batch generation failed: %s", e)
                    self._stats.failed_generations += batch
                    remaining -= batch
            else:
                # Generate one at a time for small remaining counts
                for _ in range(batch):
                    prompt = self._pick_prompt()
                    category = self._extract_category(prompt)
                    try:
                        result = self.llm.generate(
                            user_prompt=prompt,
                            system_prompt=DESCRIPTION_SYSTEM_PROMPT,
                            config=GenerationConfig(
                                max_tokens=512,
                                temperature=0.9,
                                top_p=0.95,
                            ),
                        )
                        desc = result.text.strip()
                        if len(desc) >= 20:
                            descriptions.append((desc, category))
                    except Exception as e:
                        logger.warning("Single generation failed: %s", e)
                        self._stats.failed_generations += 1
                    remaining -= 1
                    if remaining <= 0:
                        break

            if descriptions:
                logger.info(
                    "Generated %d/%d descriptions", len(descriptions), count
                )

        return descriptions

    def _parse_numbered_list(self, text: str) -> list[str]:
        """Parse numbered list from LLM output (1. ..., 2. ..., etc.)."""
        lines = text.strip().split("\n")
        descriptions: list[str] = []
        current: list[str] = []

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue

            # Check if line starts with a number followed by period
            is_numbered = False
            for i, ch in enumerate(stripped):
                if ch.isdigit():
                    continue
                if ch in ".)" and i > 0:
                    is_numbered = True
                break

            if is_numbered:
                # Save previous
                if current:
                    descriptions.append(" ".join(current))
                # Strip the number prefix
                after = stripped.lstrip("0123456789").lstrip(".)").strip()
                current = [after] if after else []
            else:
                # Continuation of current description
                current.append(stripped)

        if current:
            descriptions.append(" ".join(current))

        return descriptions

    def _extract_category(self, prompt: str) -> str:
        """Extract category keyword from a category prompt."""
        keywords = [
            "landscape", "urban", "animal", "food", "technology",
            "artwork", "architecture", "sports", "scientific", "social",
            "vehicle", "workspace", "garden", "weather", "marketplace",
            "nighttime", "children", "musical", "historical", "macro",
            "coastal", "mountains", "library", "industrial", "festive",
        ]
        prompt_lower = prompt.lower()
        for kw in keywords:
            if kw in prompt_lower:
                return kw
        return "general"

    def _encode_batch(
        self,
        descriptions: list[tuple[str, str]],
    ) -> list[EmbeddingPair]:
        """Encode a batch of descriptions with both encoders.

        Returns list of EmbeddingPair with both CLIP and MiniLM embeddings.
        """
        texts = [d[0] for d in descriptions]
        categories = [d[1] for d in descriptions]
        pairs: list[EmbeddingPair] = []

        # Encode in sub-batches to manage memory
        for i in range(0, len(texts), self.encoding_batch_size):
            batch_texts = texts[i : i + self.encoding_batch_size]
            batch_cats = categories[i : i + self.encoding_batch_size]

            try:
                clip_embs = self.clip_encoder.encode(batch_texts)
                minilm_embs = self.minilm_encoder.encode(batch_texts)

                for text, clip_emb, minilm_emb, cat in zip(
                    batch_texts, clip_embs, minilm_embs, batch_cats
                ):
                    if (
                        len(clip_emb) == CLIP_EMBED_DIM
                        and len(minilm_emb) == MINILM_EMBED_DIM
                    ):
                        pairs.append(
                            EmbeddingPair(
                                description=text,
                                clip_embedding=clip_emb,
                                minilm_embedding=minilm_emb,
                                category=cat,
                            )
                        )
                        self._stats.pairs_encoded += 1
                    else:
                        logger.warning(
                            "Dimension mismatch: CLIP=%d, MiniLM=%d",
                            len(clip_emb),
                            len(minilm_emb),
                        )
                        self._stats.failed_encodings += 1
            except Exception as e:
                logger.warning("Encoding batch failed: %s", e)
                self._stats.failed_encodings += len(batch_texts)

        return pairs

    def generate_pairs(
        self,
        count: int = 50000,
        progress_interval: int = 1000,
    ) -> list[EmbeddingPair]:
        """Generate paired embeddings for projection layer training.

        Args:
            count: Number of embedding pairs to generate.
            progress_interval: Log progress every N pairs.

        Returns:
            List of EmbeddingPair with CLIP and MiniLM embeddings.
        """
        self._init_encoders()
        self._stats = GenerationStats(total_requested=count)
        start = time.monotonic()

        logger.info("Generating %d projection training pairs...", count)

        # Phase 1: Generate descriptions via LLM
        logger.info("Phase 1: Generating image descriptions via LLM...")
        # Request extra to account for failures
        target = int(count * 1.1)
        descriptions = self._generate_descriptions(target)
        descriptions = descriptions[:count]
        self._stats.descriptions_generated = len(descriptions)
        logger.info("Generated %d descriptions", len(descriptions))

        # Phase 2: Encode with both models
        logger.info("Phase 2: Encoding with CLIP and MiniLM...")
        pairs = self._encode_batch(descriptions)

        self._stats.elapsed_seconds = time.monotonic() - start
        logger.info(
            "Done: %d pairs in %.1fs (%.0f pairs/s)",
            len(pairs),
            self._stats.elapsed_seconds,
            len(pairs) / max(self._stats.elapsed_seconds, 0.001),
        )

        return pairs

    def export(
        self,
        pairs: list[EmbeddingPair],
        output_path: str | Path,
    ) -> Path:
        """Export embedding pairs to JSONL file.

        Each line is a JSON object with:
        - description: str
        - clip_embedding: list[float] (512-dim)
        - minilm_embedding: list[float] (384-dim)
        - category: str

        Args:
            pairs: List of EmbeddingPair to export.
            output_path: Path to output JSONL file.

        Returns:
            Path to the written file.
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            for pair in pairs:
                f.write(json.dumps(pair.to_dict()) + "\n")

        logger.info("Exported %d pairs to %s", len(pairs), path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> list[EmbeddingPair]:
        """Load embedding pairs from JSONL file.

        Args:
            path: Path to JSONL file.

        Returns:
            List of EmbeddingPair loaded from file.
        """
        pairs: list[EmbeddingPair] = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    pairs.append(EmbeddingPair.from_dict(json.loads(line)))
        return pairs

    @property
    def stats(self) -> GenerationStats:
        return self._stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate projection layer training dataset (CLIP→MiniLM pairs)"
    )
    parser.add_argument(
        "--count",
        type=int,
        default=50000,
        help="Number of embedding pairs to generate (default: 50000)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="projection_dataset.jsonl",
        help="Output JSONL file path (default: projection_dataset.jsonl)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="LLM generation batch size (default: 10)",
    )
    parser.add_argument(
        "--encoding-batch-size",
        type=int,
        default=64,
        help="Embedding encoding batch size (default: 64)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    gen = ProjectionDatasetGenerator(
        seed=args.seed,
        batch_size=args.batch_size,
        encoding_batch_size=args.encoding_batch_size,
    )

    pairs = gen.generate_pairs(count=args.count)
    gen.export(pairs, args.output)

    stats = gen.stats
    print(f"\n--- Generation Stats ---")
    print(f"Requested:   {stats.total_requested}")
    print(f"Descriptions: {stats.descriptions_generated}")
    print(f"Pairs:       {stats.pairs_encoded}")
    print(f"Failed gen:  {stats.failed_generations}")
    print(f"Failed enc:  {stats.failed_encodings}")
    print(f"Success:     {stats.success_rate:.1%}")
    print(f"Time:        {stats.elapsed_seconds:.1f}s")
    print(f"Output:      {args.output}")


if __name__ == "__main__":
    main()
