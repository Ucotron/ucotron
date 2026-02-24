#!/usr/bin/env python3
"""
Projection MLP trainer: CLIP (512-dim) → MiniLM (384-dim) bridge.

Trains a 3-layer MLP with LayerNorm + GELU activations to project
CLIP visual embeddings into MiniLM text embedding space.

Architecture:
    512 → 1024 (LayerNorm, GELU, Dropout)
       → 512  (LayerNorm, GELU, Dropout)
       → 384  (Tanh for bounded output)

Loss: Cosine similarity loss = 1 - mean(cosine_sim(pred, target))
Optimizer: AdamW with lr=1e-4
Target: final validation loss < 0.15

Usage:
    python -m fine_tune.train_projection_mlp train \\
        --input projection_dataset.jsonl \\
        --output models/projection_layer.pt \\
        --epochs 10

    python -m fine_tune.train_projection_mlp evaluate \\
        --input projection_dataset.jsonl \\
        --checkpoint models/projection_layer.pt
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

logger = logging.getLogger(__name__)

# Embedding dimensions
CLIP_EMBED_DIM = 512
MINILM_EMBED_DIM = 384


# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    """Training hyperparameters."""
    epochs: int = 10
    batch_size: int = 256
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    dropout: float = 0.1
    hidden_dim: int = 1024
    val_split: float = 0.1
    seed: int = 42
    device: str = "cpu"
    save_best: bool = True
    patience: int = 3

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> TrainConfig:
        return cls(
            epochs=getattr(args, "epochs", 10),
            batch_size=getattr(args, "batch_size", 256),
            learning_rate=getattr(args, "learning_rate", 1e-4),
            dropout=getattr(args, "dropout", 0.1),
            hidden_dim=getattr(args, "hidden_dim", 1024),
            val_split=getattr(args, "val_split", 0.1),
            seed=getattr(args, "seed", 42),
            device=getattr(args, "device", "cpu"),
            patience=getattr(args, "patience", 3),
        )


@dataclass
class TrainResult:
    """Training run results."""
    final_train_loss: float
    final_val_loss: float
    best_val_loss: float
    best_epoch: int
    total_epochs: int
    total_seconds: float
    epoch_losses: list[tuple[float, float]] = field(default_factory=list)
    checkpoint_path: str | None = None

    @property
    def converged(self) -> bool:
        return self.best_val_loss < 0.15

    def summary(self) -> str:
        lines = [
            "--- Training Summary ---",
            f"Epochs:          {self.total_epochs}",
            f"Best epoch:      {self.best_epoch + 1}",
            f"Best val loss:   {self.best_val_loss:.4f}",
            f"Final train:     {self.final_train_loss:.4f}",
            f"Final val:       {self.final_val_loss:.4f}",
            f"Converged:       {'YES' if self.converged else 'NO'} (target < 0.15)",
            f"Time:            {self.total_seconds:.1f}s",
        ]
        if self.checkpoint_path:
            lines.append(f"Checkpoint:      {self.checkpoint_path}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Dataset loading (no torch dependency)
# ---------------------------------------------------------------------------

@dataclass
class EmbeddingPair:
    """A paired CLIP + MiniLM embedding from the training dataset."""
    description: str
    clip_embedding: list[float]
    minilm_embedding: list[float]
    category: str | None = None

    @classmethod
    def from_dict(cls, d: dict) -> EmbeddingPair:
        return cls(
            description=d["description"],
            clip_embedding=d["clip_embedding"],
            minilm_embedding=d["minilm_embedding"],
            category=d.get("category"),
        )


def load_jsonl(path: str | Path) -> list[EmbeddingPair]:
    """Load embedding pairs from JSONL file."""
    pairs = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            pairs.append(EmbeddingPair.from_dict(data))
    return pairs


# ---------------------------------------------------------------------------
# Model builder (abstracts torch dependency for testability)
# ---------------------------------------------------------------------------

@dataclass
class ExportResult:
    """ONNX export results."""
    onnx_path: str
    input_dim: int
    output_dim: int
    opset_version: int
    dynamic_batch: bool
    verified: bool
    max_abs_diff: float
    file_size_bytes: int

    def summary(self) -> str:
        lines = [
            "--- ONNX Export Summary ---",
            f"Output:          {self.onnx_path}",
            f"Input dim:       {self.input_dim}",
            f"Output dim:      {self.output_dim}",
            f"Opset version:   {self.opset_version}",
            f"Dynamic batch:   {self.dynamic_batch}",
            f"Verified:        {'PASS' if self.verified else 'FAIL'}",
            f"Max abs diff:    {self.max_abs_diff:.6f}",
            f"File size:       {self.file_size_bytes / 1024:.1f} KB",
        ]
        return "\n".join(lines)


class TorchBackend(Protocol):
    """Protocol for torch operations — enables mocking in tests."""

    def build_model(self, config: TrainConfig) -> Any: ...
    def create_optimizer(self, model: Any, config: TrainConfig) -> Any: ...
    def create_dataloader(
        self, clip_embs: list[list[float]], minilm_embs: list[list[float]],
        batch_size: int, shuffle: bool
    ) -> Any: ...
    def train_epoch(self, model: Any, dataloader: Any, optimizer: Any, loss_fn: Any) -> float: ...
    def eval_epoch(self, model: Any, dataloader: Any, loss_fn: Any) -> float: ...
    def save_checkpoint(self, model: Any, path: str) -> None: ...
    def load_checkpoint(self, path: str, config: TrainConfig) -> Any: ...
    def set_seed(self, seed: int) -> None: ...
    def cosine_loss_fn(self) -> Any: ...
    def export_onnx(self, model: Any, output_path: str, opset: int) -> None: ...
    def verify_onnx(self, model: Any, onnx_path: str, input_dim: int) -> float: ...


def _get_real_torch_backend() -> TorchBackend:
    """Import and return real PyTorch backend."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    class RealTorchBackend:
        def create_optimizer(self, model: nn.Module, config: TrainConfig) -> Any:
            return torch.optim.AdamW(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
            )

        def build_model(self, config: TrainConfig) -> nn.Module:
            model = nn.Sequential(
                nn.Linear(CLIP_EMBED_DIM, config.hidden_dim),
                nn.LayerNorm(config.hidden_dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dim, CLIP_EMBED_DIM),
                nn.LayerNorm(CLIP_EMBED_DIM),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(CLIP_EMBED_DIM, MINILM_EMBED_DIM),
                nn.Tanh(),
            )
            return model.to(config.device)

        def create_dataloader(
            self, clip_embs: list[list[float]], minilm_embs: list[list[float]],
            batch_size: int, shuffle: bool
        ) -> DataLoader:
            clip_t = torch.tensor(clip_embs, dtype=torch.float32)
            minilm_t = torch.tensor(minilm_embs, dtype=torch.float32)
            ds = TensorDataset(clip_t, minilm_t)
            return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

        def train_epoch(self, model: nn.Module, dataloader: DataLoader,
                        optimizer: Any, loss_fn: Any) -> float:
            model.train()
            total_loss = 0.0
            n_batches = 0
            for clip_batch, minilm_batch in dataloader:
                optimizer.zero_grad()
                pred = model(clip_batch)
                loss = loss_fn(pred, minilm_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                n_batches += 1
            return total_loss / max(n_batches, 1)

        def eval_epoch(self, model: nn.Module, dataloader: DataLoader,
                       loss_fn: Any) -> float:
            model.eval()
            total_loss = 0.0
            n_batches = 0
            with torch.no_grad():
                for clip_batch, minilm_batch in dataloader:
                    pred = model(clip_batch)
                    loss = loss_fn(pred, minilm_batch)
                    total_loss += loss.item()
                    n_batches += 1
            return total_loss / max(n_batches, 1)

        def save_checkpoint(self, model: nn.Module, path: str) -> None:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            torch.save(model.state_dict(), path)

        def load_checkpoint(self, path: str, config: TrainConfig) -> nn.Module:
            model = self.build_model(config)
            state_dict = torch.load(path, map_location=config.device, weights_only=True)
            model.load_state_dict(state_dict)
            return model

        def set_seed(self, seed: int) -> None:
            torch.manual_seed(seed)

        def cosine_loss_fn(self):
            def cosine_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
                pred_norm = nn.functional.normalize(pred, dim=1)
                target_norm = nn.functional.normalize(target, dim=1)
                return 1.0 - (pred_norm * target_norm).sum(dim=1).mean()
            return cosine_loss

        def export_onnx(self, model: nn.Module, output_path: str, opset: int) -> None:
            model.eval()
            dummy_input = torch.randn(1, CLIP_EMBED_DIM)
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            torch.onnx.export(
                model,
                dummy_input,
                output_path,
                opset_version=opset,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={
                    "input": {0: "batch"},
                    "output": {0: "batch"},
                },
            )

        def verify_onnx(self, model: nn.Module, onnx_path: str, input_dim: int) -> float:
            import onnxruntime as ort
            import numpy as np

            model.eval()
            session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

            max_diff = 0.0
            # Test with batch sizes 1, 4, 16 to verify dynamic batch axis
            for batch_size in [1, 4, 16]:
                test_input = torch.randn(batch_size, input_dim)

                # PyTorch output
                with torch.no_grad():
                    pt_output = model(test_input).numpy()

                # ONNX output
                onnx_output = session.run(None, {"input": test_input.numpy()})[0]

                diff = float(np.max(np.abs(pt_output - onnx_output)))
                max_diff = max(max_diff, diff)

            return max_diff

    return RealTorchBackend()


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class ProjectionMLPTrainer:
    """Train a 3-layer MLP to project CLIP → MiniLM embedding space."""

    def __init__(self, config: TrainConfig, backend: TorchBackend | None = None):
        self.config = config
        self.backend = backend or _get_real_torch_backend()

    def load_dataset(self, path: str | Path) -> tuple[list[EmbeddingPair], list[EmbeddingPair]]:
        """Load JSONL dataset and split into train/val."""
        pairs = load_jsonl(path)
        if len(pairs) == 0:
            raise ValueError(f"Empty dataset at {path}")

        # Deterministic shuffle for reproducible split
        import random
        rng = random.Random(self.config.seed)
        indices = list(range(len(pairs)))
        rng.shuffle(indices)

        val_count = max(1, int(len(pairs) * self.config.val_split))
        val_indices = set(indices[:val_count])
        train_pairs = [pairs[i] for i in range(len(pairs)) if i not in val_indices]
        val_pairs = [pairs[i] for i in range(len(pairs)) if i in val_indices]

        logger.info(f"Dataset: {len(train_pairs)} train, {len(val_pairs)} val")
        return train_pairs, val_pairs

    def _pairs_to_arrays(self, pairs: list[EmbeddingPair]) -> tuple[list[list[float]], list[list[float]]]:
        """Extract clip and minilm embeddings from pairs."""
        clip_embs = [p.clip_embedding for p in pairs]
        minilm_embs = [p.minilm_embedding for p in pairs]
        return clip_embs, minilm_embs

    def train(
        self,
        input_path: str | Path,
        output_path: str | Path = "models/projection_layer.pt",
    ) -> TrainResult:
        """Run the full training pipeline."""
        start_time = time.time()
        self.backend.set_seed(self.config.seed)

        # Load and split
        train_pairs, val_pairs = self.load_dataset(input_path)
        train_clip, train_minilm = self._pairs_to_arrays(train_pairs)
        val_clip, val_minilm = self._pairs_to_arrays(val_pairs)

        # Create data loaders
        train_dl = self.backend.create_dataloader(
            train_clip, train_minilm, self.config.batch_size, shuffle=True
        )
        val_dl = self.backend.create_dataloader(
            val_clip, val_minilm, self.config.batch_size, shuffle=False
        )

        # Build model and optimizer
        model = self.backend.build_model(self.config)
        loss_fn = self.backend.cosine_loss_fn()
        optimizer = self.backend.create_optimizer(model, self.config)

        # Training loop
        best_val_loss = float("inf")
        best_epoch = 0
        epochs_no_improve = 0
        epoch_losses: list[tuple[float, float]] = []

        for epoch in range(self.config.epochs):
            train_loss = self.backend.train_epoch(model, train_dl, optimizer, loss_fn)
            val_loss = self.backend.eval_epoch(model, val_dl, loss_fn)
            epoch_losses.append((train_loss, val_loss))

            improved = val_loss < best_val_loss
            marker = " *" if improved else ""
            logger.info(
                f"Epoch {epoch+1}/{self.config.epochs} | "
                f"train={train_loss:.4f} val={val_loss:.4f}{marker}"
            )
            print(
                f"Epoch {epoch+1}/{self.config.epochs} | "
                f"train={train_loss:.4f} val={val_loss:.4f}{marker}"
            )

            if improved:
                best_val_loss = val_loss
                best_epoch = epoch
                epochs_no_improve = 0
                if self.config.save_best:
                    self.backend.save_checkpoint(model, str(output_path))
            else:
                epochs_no_improve += 1
                if self.config.patience > 0 and epochs_no_improve >= self.config.patience:
                    logger.info(f"Early stopping at epoch {epoch+1} (no improvement for {self.config.patience} epochs)")
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        elapsed = time.time() - start_time
        final_train, final_val = epoch_losses[-1]

        result = TrainResult(
            final_train_loss=final_train,
            final_val_loss=final_val,
            best_val_loss=best_val_loss,
            best_epoch=best_epoch,
            total_epochs=len(epoch_losses),
            total_seconds=elapsed,
            epoch_losses=epoch_losses,
            checkpoint_path=str(output_path) if self.config.save_best else None,
        )

        print(f"\n{result.summary()}")
        return result

    def export_onnx(
        self,
        checkpoint_path: str | Path,
        output_path: str | Path = "models/projection_layer.onnx",
        opset: int = 17,
        verify: bool = True,
    ) -> ExportResult:
        """Export a trained checkpoint to ONNX format.

        Steps:
        1. Load PyTorch checkpoint
        2. Export to ONNX with dynamic batch axis
        3. Verify ONNX output matches PyTorch output
        """
        model = self.backend.load_checkpoint(str(checkpoint_path), self.config)

        # Export to ONNX
        onnx_path = str(output_path)
        self.backend.export_onnx(model, onnx_path, opset)
        logger.info(f"Exported ONNX model to {onnx_path}")

        # Verify
        max_diff = 0.0
        verified = True
        if verify:
            max_diff = self.backend.verify_onnx(model, onnx_path, CLIP_EMBED_DIM)
            # Allow small numerical differences (float32 precision)
            verified = max_diff < 1e-4
            if verified:
                logger.info(f"Verification PASSED (max diff: {max_diff:.6f})")
            else:
                logger.warning(f"Verification FAILED (max diff: {max_diff:.6f}, threshold: 1e-4)")

        file_size = os.path.getsize(onnx_path)

        result = ExportResult(
            onnx_path=onnx_path,
            input_dim=CLIP_EMBED_DIM,
            output_dim=MINILM_EMBED_DIM,
            opset_version=opset,
            dynamic_batch=True,
            verified=verified,
            max_abs_diff=max_diff,
            file_size_bytes=file_size,
        )
        print(f"\n{result.summary()}")
        return result

    def evaluate(
        self,
        input_path: str | Path,
        checkpoint_path: str | Path,
    ) -> float:
        """Evaluate a trained checkpoint on the validation split."""
        _, val_pairs = self.load_dataset(input_path)
        val_clip, val_minilm = self._pairs_to_arrays(val_pairs)

        val_dl = self.backend.create_dataloader(
            val_clip, val_minilm, self.config.batch_size, shuffle=False
        )

        model = self.backend.load_checkpoint(str(checkpoint_path), self.config)
        loss_fn = self.backend.cosine_loss_fn()
        val_loss = self.backend.eval_epoch(model, val_dl, loss_fn)

        print(f"Validation loss: {val_loss:.4f} ({'PASS' if val_loss < 0.15 else 'FAIL'} target < 0.15)")
        return val_loss


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def cmd_train(args: argparse.Namespace) -> None:
    """CLI handler for train subcommand."""
    config = TrainConfig.from_args(args)
    trainer = ProjectionMLPTrainer(config)
    result = trainer.train(args.input, args.output)
    sys.exit(0 if result.converged else 1)


def cmd_export(args: argparse.Namespace) -> None:
    """CLI handler for export subcommand."""
    config = TrainConfig.from_args(args)
    trainer = ProjectionMLPTrainer(config)
    result = trainer.export_onnx(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        opset=args.opset,
        verify=not args.skip_verify,
    )
    sys.exit(0 if result.verified else 1)


def cmd_evaluate(args: argparse.Namespace) -> None:
    """CLI handler for evaluate subcommand."""
    config = TrainConfig.from_args(args)
    trainer = ProjectionMLPTrainer(config)
    val_loss = trainer.evaluate(args.input, args.checkpoint)
    sys.exit(0 if val_loss < 0.15 else 1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train projection MLP: CLIP (512) → MiniLM (384)"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Train
    train_parser = subparsers.add_parser("train", help="Train projection MLP")
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
    export_parser = subparsers.add_parser("export", help="Export trained checkpoint to ONNX")
    export_parser.add_argument("--checkpoint", type=str, required=True, help="PyTorch checkpoint path (.pt)")
    export_parser.add_argument("--output", type=str, default="models/projection_layer.onnx", help="ONNX output path")
    export_parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    export_parser.add_argument("--skip-verify", action="store_true", help="Skip ONNX verification")
    export_parser.add_argument("--hidden-dim", type=int, default=1024)
    export_parser.add_argument("--dropout", type=float, default=0.1)
    export_parser.add_argument("--device", type=str, default="cpu")
    export_parser.add_argument("--verbose", action="store_true")

    # Evaluate
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate trained checkpoint")
    eval_parser.add_argument("--input", type=str, required=True, help="JSONL dataset path")
    eval_parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint path")
    eval_parser.add_argument("--batch-size", type=int, default=256)
    eval_parser.add_argument("--val-split", type=float, default=0.1)
    eval_parser.add_argument("--seed", type=int, default=42)
    eval_parser.add_argument("--device", type=str, default="cpu")
    eval_parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if getattr(args, "verbose", False) else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.command == "train":
        cmd_train(args)
    elif args.command == "export":
        cmd_export(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)


if __name__ == "__main__":
    main()
