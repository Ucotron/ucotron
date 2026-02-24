#!/usr/bin/env python3
"""
Entry-point script for projection layer dataset generation, training, and ONNX export.

Orchestrates the full pipeline:
1. Generate 50k text-image description pairs via GLM-5 (Fireworks)
2. Encode with CLIP text encoder (512-dim) and MiniLM (384-dim)
3. Export as JSONL for projection MLP training
4. Train projection MLP (512→384) with cosine similarity loss
5. Export trained model to ONNX with dynamic batch axis and verification

Usage:
    # Generate dataset (requires FIREWORKS_API_KEY)
    python scripts/train_projection_layer.py generate --count 50000

    # Validate existing dataset
    python scripts/train_projection_layer.py validate --input projection_dataset.jsonl

    # Show dataset stats
    python scripts/train_projection_layer.py stats --input projection_dataset.jsonl

    # Train projection MLP (requires torch)
    python scripts/train_projection_layer.py train --input projection_dataset.jsonl

    # Export to ONNX (requires torch, onnxruntime)
    python scripts/train_projection_layer.py export --checkpoint models/projection_layer.pt
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path

# Add generate_training_data and fine_tune to path for imports
sys.path.insert(0, str(Path(__file__).parent / "generate_training_data"))
sys.path.insert(0, str(Path(__file__).parent))

from projection_dataset_generator import (
    ProjectionDatasetGenerator,
    EmbeddingPair,
    CLIP_EMBED_DIM,
    MINILM_EMBED_DIM,
)

logger = logging.getLogger(__name__)


def cmd_generate(args: argparse.Namespace) -> None:
    """Generate projection training dataset."""
    gen = ProjectionDatasetGenerator(
        seed=args.seed,
        batch_size=args.batch_size,
        encoding_batch_size=args.encoding_batch_size,
    )

    pairs = gen.generate_pairs(count=args.count)
    gen.export(pairs, args.output)

    stats = gen.stats
    print(f"\n--- Generation Complete ---")
    print(f"Pairs generated: {stats.pairs_encoded}/{stats.total_requested}")
    print(f"Success rate:    {stats.success_rate:.1%}")
    print(f"Time:            {stats.elapsed_seconds:.1f}s")
    print(f"Output:          {args.output}")


def cmd_validate(args: argparse.Namespace) -> None:
    """Validate an existing projection dataset."""
    path = Path(args.input)
    if not path.exists():
        print(f"Error: {path} not found")
        sys.exit(1)

    pairs = ProjectionDatasetGenerator.load(path)
    errors = []

    for i, pair in enumerate(pairs):
        if len(pair.clip_embedding) != CLIP_EMBED_DIM:
            errors.append(f"Line {i+1}: CLIP dim={len(pair.clip_embedding)}, expected {CLIP_EMBED_DIM}")
        if len(pair.minilm_embedding) != MINILM_EMBED_DIM:
            errors.append(f"Line {i+1}: MiniLM dim={len(pair.minilm_embedding)}, expected {MINILM_EMBED_DIM}")
        if len(pair.description.strip()) < 10:
            errors.append(f"Line {i+1}: description too short ({len(pair.description)} chars)")

        # Check L2 norm of embeddings (should be ~1.0)
        clip_norm = math.sqrt(sum(x * x for x in pair.clip_embedding))
        if abs(clip_norm - 1.0) > 0.01:
            errors.append(f"Line {i+1}: CLIP norm={clip_norm:.4f}, expected ~1.0")

        minilm_norm = math.sqrt(sum(x * x for x in pair.minilm_embedding))
        if abs(minilm_norm - 1.0) > 0.01:
            errors.append(f"Line {i+1}: MiniLM norm={minilm_norm:.4f}, expected ~1.0")

    if errors:
        print(f"VALIDATION FAILED: {len(errors)} errors")
        for err in errors[:20]:
            print(f"  - {err}")
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more")
        sys.exit(1)
    else:
        print(f"VALIDATION PASSED: {len(pairs)} pairs, all dimensions correct")


def cmd_stats(args: argparse.Namespace) -> None:
    """Show statistics for an existing dataset."""
    path = Path(args.input)
    if not path.exists():
        print(f"Error: {path} not found")
        sys.exit(1)

    pairs = ProjectionDatasetGenerator.load(path)

    categories: dict[str, int] = {}
    desc_lengths: list[int] = []

    for pair in pairs:
        cat = pair.category or "unknown"
        categories[cat] = categories.get(cat, 0) + 1
        desc_lengths.append(len(pair.description))

    print(f"--- Dataset Statistics ---")
    print(f"Total pairs:     {len(pairs)}")
    print(f"CLIP dim:        {CLIP_EMBED_DIM}")
    print(f"MiniLM dim:      {MINILM_EMBED_DIM}")
    print(f"File size:       {path.stat().st_size / (1024*1024):.1f} MB")
    print(f"\nDescription lengths:")
    if desc_lengths:
        print(f"  Min:    {min(desc_lengths)}")
        print(f"  Max:    {max(desc_lengths)}")
        print(f"  Mean:   {sum(desc_lengths)/len(desc_lengths):.0f}")
    print(f"\nCategories ({len(categories)}):")
    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count}")


def cmd_train(args: argparse.Namespace) -> None:
    """Train projection MLP on existing dataset."""
    from fine_tune.train_projection_mlp import ProjectionMLPTrainer, TrainConfig

    config = TrainConfig.from_args(args)
    trainer = ProjectionMLPTrainer(config)
    result = trainer.train(args.input, args.output)

    if not result.converged:
        print(f"\nWARNING: best val loss {result.best_val_loss:.4f} > 0.15 target")
        sys.exit(1)


def cmd_export(args: argparse.Namespace) -> None:
    """Export trained projection MLP to ONNX format."""
    from fine_tune.train_projection_mlp import ProjectionMLPTrainer, TrainConfig

    config = TrainConfig.from_args(args)
    trainer = ProjectionMLPTrainer(config)
    result = trainer.export_onnx(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        opset=args.opset,
        verify=not args.skip_verify,
    )

    if not result.verified:
        print(f"\nWARNING: ONNX verification failed (max diff: {result.max_abs_diff:.6f})")
        sys.exit(1)
    print(f"\nONNX model ready: {result.onnx_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Projection layer dataset generation, management, and training"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Generate
    gen_parser = subparsers.add_parser("generate", help="Generate training dataset")
    gen_parser.add_argument("--count", type=int, default=50000)
    gen_parser.add_argument("--output", type=str, default="projection_dataset.jsonl")
    gen_parser.add_argument("--batch-size", type=int, default=10)
    gen_parser.add_argument("--encoding-batch-size", type=int, default=64)
    gen_parser.add_argument("--seed", type=int, default=42)
    gen_parser.add_argument("--verbose", action="store_true")

    # Validate
    val_parser = subparsers.add_parser("validate", help="Validate dataset")
    val_parser.add_argument("--input", type=str, required=True)

    # Stats
    stats_parser = subparsers.add_parser("stats", help="Show dataset stats")
    stats_parser.add_argument("--input", type=str, required=True)

    # Train
    train_parser = subparsers.add_parser("train", help="Train projection MLP (requires torch)")
    train_parser.add_argument("--input", type=str, required=True, help="JSONL dataset path")
    train_parser.add_argument("--output", type=str, default="models/projection_layer.pt")
    train_parser.add_argument("--epochs", type=int, default=10)
    train_parser.add_argument("--batch-size", type=int, default=256)
    train_parser.add_argument("--learning-rate", type=float, default=1e-4)
    train_parser.add_argument("--dropout", type=float, default=0.1)
    train_parser.add_argument("--hidden-dim", type=int, default=1024)
    train_parser.add_argument("--val-split", type=float, default=0.1)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--device", type=str, default="cpu")
    train_parser.add_argument("--patience", type=int, default=3)
    train_parser.add_argument("--verbose", action="store_true")

    # Export to ONNX
    export_parser = subparsers.add_parser("export", help="Export trained checkpoint to ONNX (requires torch, onnxruntime)")
    export_parser.add_argument("--checkpoint", type=str, default="models/projection_layer.pt", help="PyTorch checkpoint path")
    export_parser.add_argument("--output", type=str, default="models/projection_layer.onnx", help="ONNX output path")
    export_parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    export_parser.add_argument("--skip-verify", action="store_true", help="Skip ONNX verification")
    export_parser.add_argument("--hidden-dim", type=int, default=1024)
    export_parser.add_argument("--dropout", type=float, default=0.1)
    export_parser.add_argument("--device", type=str, default="cpu")
    export_parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if getattr(args, "verbose", False) else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.command == "generate":
        cmd_generate(args)
    elif args.command == "validate":
        cmd_validate(args)
    elif args.command == "stats":
        cmd_stats(args)
    elif args.command == "train":
        cmd_train(args)
    elif args.command == "export":
        cmd_export(args)


if __name__ == "__main__":
    main()
