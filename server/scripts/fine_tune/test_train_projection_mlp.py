#!/usr/bin/env python3
"""
Tests for ProjectionMLPTrainer — fully mocked, no real torch dependency required.

Run with: python -m pytest scripts/fine_tune/test_train_projection_mlp.py -v
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from fine_tune.train_projection_mlp import (
    CLIP_EMBED_DIM,
    MINILM_EMBED_DIM,
    EmbeddingPair,
    ExportResult,
    ProjectionMLPTrainer,
    TrainConfig,
    TrainResult,
    load_jsonl,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_embedding(dim: int, seed: int = 0) -> list[float]:
    """Create a deterministic L2-normalized embedding."""
    raw = [(seed + i + 1) / (dim * 10) for i in range(dim)]
    norm = math.sqrt(sum(x * x for x in raw))
    return [x / norm for x in raw]


def _make_pair(idx: int = 0, category: str = "test") -> dict:
    """Create a single embedding pair dict for JSONL."""
    return {
        "description": f"Test description number {idx}",
        "clip_embedding": _make_embedding(CLIP_EMBED_DIM, seed=idx),
        "minilm_embedding": _make_embedding(MINILM_EMBED_DIM, seed=idx + 1000),
        "category": category,
    }


def _write_jsonl(pairs: list[dict], tmpdir: str, name: str = "dataset.jsonl") -> Path:
    """Write pairs to a JSONL file."""
    path = Path(tmpdir) / name
    with open(path, "w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")
    return path


class MockTorchBackend:
    """Mock backend that simulates training without torch."""

    def __init__(self, loss_sequence: list[float] | None = None):
        self.loss_sequence = loss_sequence or [0.5, 0.3, 0.2, 0.15, 0.12, 0.10, 0.09, 0.08, 0.07, 0.06]
        self._epoch = 0
        self.model = MagicMock()
        self.model.parameters = MagicMock(return_value=[MagicMock()])
        self.saved_checkpoints: list[str] = []
        self.seeds_set: list[int] = []
        self._train_dl = MagicMock()
        self._val_dl = MagicMock()

    def build_model(self, config):
        return self.model

    def create_optimizer(self, model, config):
        return MagicMock()

    def create_dataloader(self, clip_embs, minilm_embs, batch_size, shuffle):
        return self._train_dl if shuffle else self._val_dl

    def train_epoch(self, model, dataloader, optimizer, loss_fn):
        idx = min(self._epoch, len(self.loss_sequence) - 1)
        loss = self.loss_sequence[idx]
        return loss

    def eval_epoch(self, model, dataloader, loss_fn):
        idx = min(self._epoch, len(self.loss_sequence) - 1)
        loss = self.loss_sequence[idx] + 0.01  # val slightly higher
        self._epoch += 1
        return loss

    def save_checkpoint(self, model, path):
        self.saved_checkpoints.append(path)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        Path(path).write_text("mock_checkpoint")

    def load_checkpoint(self, path, config):
        return self.model

    def set_seed(self, seed):
        self.seeds_set.append(seed)

    def cosine_loss_fn(self):
        return MagicMock()

    def export_onnx(self, model, output_path, opset):
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        Path(output_path).write_text("mock_onnx_model")
        self.exported_paths = getattr(self, "exported_paths", [])
        self.exported_paths.append(output_path)
        self.export_opset = opset

    def verify_onnx(self, model, onnx_path, input_dim):
        return self._verify_max_diff if hasattr(self, "_verify_max_diff") else 1e-6


# ---------------------------------------------------------------------------
# EmbeddingPair tests
# ---------------------------------------------------------------------------

class TestEmbeddingPair:
    def test_from_dict(self):
        d = _make_pair(0)
        pair = EmbeddingPair.from_dict(d)
        assert pair.description == "Test description number 0"
        assert len(pair.clip_embedding) == CLIP_EMBED_DIM
        assert len(pair.minilm_embedding) == MINILM_EMBED_DIM
        assert pair.category == "test"

    def test_from_dict_no_category(self):
        d = _make_pair(0)
        del d["category"]
        pair = EmbeddingPair.from_dict(d)
        assert pair.category is None

    def test_embedding_dimensions(self):
        pair = EmbeddingPair.from_dict(_make_pair(42))
        assert len(pair.clip_embedding) == 512
        assert len(pair.minilm_embedding) == 384


# ---------------------------------------------------------------------------
# load_jsonl tests
# ---------------------------------------------------------------------------

class TestLoadJsonl:
    def test_load_basic(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_jsonl([_make_pair(i) for i in range(10)], tmpdir)
            pairs = load_jsonl(path)
            assert len(pairs) == 10
            assert pairs[0].description == "Test description number 0"

    def test_load_empty_lines_skipped(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "data.jsonl"
            with open(path, "w") as f:
                f.write(json.dumps(_make_pair(0)) + "\n")
                f.write("\n")
                f.write(json.dumps(_make_pair(1)) + "\n")
                f.write("\n\n")
            pairs = load_jsonl(path)
            assert len(pairs) == 2

    def test_load_preserves_order(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data = [_make_pair(i) for i in range(5)]
            path = _write_jsonl(data, tmpdir)
            pairs = load_jsonl(path)
            for i, pair in enumerate(pairs):
                assert pair.description == f"Test description number {i}"


# ---------------------------------------------------------------------------
# TrainConfig tests
# ---------------------------------------------------------------------------

class TestTrainConfig:
    def test_defaults(self):
        cfg = TrainConfig()
        assert cfg.epochs == 10
        assert cfg.batch_size == 256
        assert cfg.learning_rate == 1e-4
        assert cfg.dropout == 0.1
        assert cfg.hidden_dim == 1024
        assert cfg.val_split == 0.1
        assert cfg.seed == 42
        assert cfg.device == "cpu"
        assert cfg.patience == 3

    def test_from_args(self):
        args = MagicMock()
        args.epochs = 5
        args.batch_size = 128
        args.learning_rate = 2e-4
        args.dropout = 0.2
        args.hidden_dim = 512
        args.val_split = 0.2
        args.seed = 123
        args.device = "cuda"
        args.patience = 5
        cfg = TrainConfig.from_args(args)
        assert cfg.epochs == 5
        assert cfg.batch_size == 128
        assert cfg.learning_rate == 2e-4
        assert cfg.hidden_dim == 512
        assert cfg.seed == 123


# ---------------------------------------------------------------------------
# TrainResult tests
# ---------------------------------------------------------------------------

class TestTrainResult:
    def test_converged_true(self):
        result = TrainResult(
            final_train_loss=0.10,
            final_val_loss=0.12,
            best_val_loss=0.12,
            best_epoch=5,
            total_epochs=10,
            total_seconds=30.0,
        )
        assert result.converged is True

    def test_converged_false(self):
        result = TrainResult(
            final_train_loss=0.20,
            final_val_loss=0.25,
            best_val_loss=0.20,
            best_epoch=8,
            total_epochs=10,
            total_seconds=30.0,
        )
        assert result.converged is False

    def test_converged_boundary(self):
        result = TrainResult(
            final_train_loss=0.15,
            final_val_loss=0.15,
            best_val_loss=0.15,
            best_epoch=9,
            total_epochs=10,
            total_seconds=30.0,
        )
        # 0.15 is NOT < 0.15
        assert result.converged is False

    def test_summary_contains_metrics(self):
        result = TrainResult(
            final_train_loss=0.10,
            final_val_loss=0.12,
            best_val_loss=0.08,
            best_epoch=7,
            total_epochs=10,
            total_seconds=45.2,
            checkpoint_path="models/projection_layer.pt",
        )
        s = result.summary()
        assert "Best val loss:   0.0800" in s
        assert "Best epoch:      8" in s
        assert "Converged:       YES" in s
        assert "models/projection_layer.pt" in s


# ---------------------------------------------------------------------------
# ProjectionMLPTrainer tests
# ---------------------------------------------------------------------------

class TestProjectionMLPTrainer:
    def _make_dataset(self, tmpdir: str, count: int = 100) -> Path:
        """Create a JSONL dataset file with `count` pairs."""
        return _write_jsonl([_make_pair(i) for i in range(count)], tmpdir)

    def test_load_dataset_splits(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._make_dataset(tmpdir, count=100)
            backend = MockTorchBackend()
            trainer = ProjectionMLPTrainer(TrainConfig(val_split=0.2), backend=backend)
            train_pairs, val_pairs = trainer.load_dataset(path)
            assert len(train_pairs) == 80
            assert len(val_pairs) == 20

    def test_load_dataset_deterministic(self):
        """Same seed should produce same split."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._make_dataset(tmpdir, count=50)
            backend = MockTorchBackend()
            trainer = ProjectionMLPTrainer(TrainConfig(seed=42), backend=backend)
            t1, v1 = trainer.load_dataset(path)
            t2, v2 = trainer.load_dataset(path)
            assert [p.description for p in t1] == [p.description for p in t2]
            assert [p.description for p in v1] == [p.description for p in v2]

    def test_load_dataset_empty_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "empty.jsonl"
            path.write_text("")
            backend = MockTorchBackend()
            trainer = ProjectionMLPTrainer(TrainConfig(), backend=backend)
            with pytest.raises(ValueError, match="Empty dataset"):
                trainer.load_dataset(path)

    def test_train_basic(self):
        """Train completes and returns result."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ds_path = self._make_dataset(tmpdir, count=50)
            out_path = os.path.join(tmpdir, "model.pt")
            backend = MockTorchBackend(loss_sequence=[0.5, 0.3, 0.2, 0.12, 0.10])
            config = TrainConfig(epochs=5, patience=0)  # patience=0 disables early stopping
            trainer = ProjectionMLPTrainer(config, backend=backend)
            result = trainer.train(ds_path, out_path)
            assert result.total_epochs == 5
            assert result.best_val_loss < 0.15
            assert result.converged

    def test_train_sets_seed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ds_path = self._make_dataset(tmpdir)
            out_path = os.path.join(tmpdir, "model.pt")
            backend = MockTorchBackend()
            trainer = ProjectionMLPTrainer(TrainConfig(seed=99, patience=0), backend=backend)
            trainer.train(ds_path, out_path)
            assert 99 in backend.seeds_set

    def test_train_saves_best_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ds_path = self._make_dataset(tmpdir)
            out_path = os.path.join(tmpdir, "model.pt")
            # Losses go down: each epoch should save
            backend = MockTorchBackend(loss_sequence=[0.5, 0.4, 0.3])
            config = TrainConfig(epochs=3, save_best=True, patience=0)
            trainer = ProjectionMLPTrainer(config, backend=backend)
            result = trainer.train(ds_path, out_path)
            assert len(backend.saved_checkpoints) == 3  # improved each epoch
            assert result.checkpoint_path == out_path

    def test_train_no_save_when_disabled(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ds_path = self._make_dataset(tmpdir)
            out_path = os.path.join(tmpdir, "model.pt")
            backend = MockTorchBackend()
            config = TrainConfig(epochs=2, save_best=False, patience=0)
            trainer = ProjectionMLPTrainer(config, backend=backend)
            result = trainer.train(ds_path, out_path)
            assert len(backend.saved_checkpoints) == 0
            assert result.checkpoint_path is None

    def test_train_early_stopping(self):
        """Training stops when val loss doesn't improve for `patience` epochs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ds_path = self._make_dataset(tmpdir)
            out_path = os.path.join(tmpdir, "model.pt")
            # Loss goes down then plateaus
            backend = MockTorchBackend(loss_sequence=[0.5, 0.3, 0.3, 0.3, 0.3, 0.3])
            config = TrainConfig(epochs=6, patience=2)
            trainer = ProjectionMLPTrainer(config, backend=backend)
            result = trainer.train(ds_path, out_path)
            # Epoch 1: 0.5→0.3 (improve), epoch 2: 0.3→0.3 (no), epoch 3: same (no), epoch 4: stop
            assert result.total_epochs == 4  # stopped early

    def test_train_epoch_losses_tracked(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ds_path = self._make_dataset(tmpdir)
            out_path = os.path.join(tmpdir, "model.pt")
            backend = MockTorchBackend(loss_sequence=[0.5, 0.4, 0.3])
            config = TrainConfig(epochs=3, patience=0)
            trainer = ProjectionMLPTrainer(config, backend=backend)
            result = trainer.train(ds_path, out_path)
            assert len(result.epoch_losses) == 3
            # Each tuple is (train_loss, val_loss)
            for train_l, val_l in result.epoch_losses:
                assert isinstance(train_l, float)
                assert isinstance(val_l, float)

    def test_train_not_converged(self):
        """High losses mean converged=False."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ds_path = self._make_dataset(tmpdir)
            out_path = os.path.join(tmpdir, "model.pt")
            backend = MockTorchBackend(loss_sequence=[0.8, 0.7, 0.6])
            config = TrainConfig(epochs=3, patience=0)
            trainer = ProjectionMLPTrainer(config, backend=backend)
            result = trainer.train(ds_path, out_path)
            assert not result.converged

    def test_evaluate(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ds_path = self._make_dataset(tmpdir)
            ckpt_path = os.path.join(tmpdir, "model.pt")
            Path(ckpt_path).write_text("mock")
            backend = MockTorchBackend(loss_sequence=[0.10])
            trainer = ProjectionMLPTrainer(TrainConfig(), backend=backend)
            val_loss = trainer.evaluate(ds_path, ckpt_path)
            assert isinstance(val_loss, float)

    def test_pairs_to_arrays(self):
        pairs = [EmbeddingPair.from_dict(_make_pair(i)) for i in range(3)]
        backend = MockTorchBackend()
        trainer = ProjectionMLPTrainer(TrainConfig(), backend=backend)
        clip_embs, minilm_embs = trainer._pairs_to_arrays(pairs)
        assert len(clip_embs) == 3
        assert len(minilm_embs) == 3
        assert len(clip_embs[0]) == CLIP_EMBED_DIM
        assert len(minilm_embs[0]) == MINILM_EMBED_DIM

    def test_val_split_minimum_one(self):
        """Even with tiny dataset, val split should have at least 1 sample."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._make_dataset(tmpdir, count=5)
            backend = MockTorchBackend()
            trainer = ProjectionMLPTrainer(TrainConfig(val_split=0.01), backend=backend)
            train_pairs, val_pairs = trainer.load_dataset(path)
            assert len(val_pairs) >= 1
            assert len(train_pairs) + len(val_pairs) == 5


# ---------------------------------------------------------------------------
# ExportResult tests
# ---------------------------------------------------------------------------

class TestExportResult:
    def test_summary_contains_fields(self):
        result = ExportResult(
            onnx_path="models/projection_layer.onnx",
            input_dim=512,
            output_dim=384,
            opset_version=17,
            dynamic_batch=True,
            verified=True,
            max_abs_diff=1e-6,
            file_size_bytes=2048,
        )
        s = result.summary()
        assert "projection_layer.onnx" in s
        assert "512" in s
        assert "384" in s
        assert "17" in s
        assert "PASS" in s

    def test_summary_fail(self):
        result = ExportResult(
            onnx_path="out.onnx",
            input_dim=512,
            output_dim=384,
            opset_version=17,
            dynamic_batch=True,
            verified=False,
            max_abs_diff=0.5,
            file_size_bytes=1024,
        )
        s = result.summary()
        assert "FAIL" in s


# ---------------------------------------------------------------------------
# ONNX export tests
# ---------------------------------------------------------------------------

class TestOnnxExport:
    def _make_dataset(self, tmpdir: str, count: int = 50) -> Path:
        return _write_jsonl([_make_pair(i) for i in range(count)], tmpdir)

    def test_export_basic(self):
        """Export produces ONNX file and returns result."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = os.path.join(tmpdir, "model.pt")
            Path(ckpt_path).write_text("mock")
            onnx_path = os.path.join(tmpdir, "model.onnx")
            backend = MockTorchBackend()
            trainer = ProjectionMLPTrainer(TrainConfig(), backend=backend)
            result = trainer.export_onnx(ckpt_path, onnx_path, opset=17)
            assert result.onnx_path == onnx_path
            assert result.input_dim == 512
            assert result.output_dim == 384
            assert result.opset_version == 17
            assert result.dynamic_batch is True
            assert result.verified is True
            assert result.file_size_bytes > 0
            assert Path(onnx_path).exists()

    def test_export_custom_opset(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = os.path.join(tmpdir, "model.pt")
            Path(ckpt_path).write_text("mock")
            onnx_path = os.path.join(tmpdir, "model.onnx")
            backend = MockTorchBackend()
            trainer = ProjectionMLPTrainer(TrainConfig(), backend=backend)
            result = trainer.export_onnx(ckpt_path, onnx_path, opset=14)
            assert result.opset_version == 14
            assert backend.export_opset == 14

    def test_export_verification_pass(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = os.path.join(tmpdir, "model.pt")
            Path(ckpt_path).write_text("mock")
            onnx_path = os.path.join(tmpdir, "model.onnx")
            backend = MockTorchBackend()
            backend._verify_max_diff = 1e-7  # Very small diff
            trainer = ProjectionMLPTrainer(TrainConfig(), backend=backend)
            result = trainer.export_onnx(ckpt_path, onnx_path)
            assert result.verified is True
            assert result.max_abs_diff < 1e-4

    def test_export_verification_fail(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = os.path.join(tmpdir, "model.pt")
            Path(ckpt_path).write_text("mock")
            onnx_path = os.path.join(tmpdir, "model.onnx")
            backend = MockTorchBackend()
            backend._verify_max_diff = 0.5  # Large diff
            trainer = ProjectionMLPTrainer(TrainConfig(), backend=backend)
            result = trainer.export_onnx(ckpt_path, onnx_path)
            assert result.verified is False
            assert result.max_abs_diff == 0.5

    def test_export_skip_verify(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = os.path.join(tmpdir, "model.pt")
            Path(ckpt_path).write_text("mock")
            onnx_path = os.path.join(tmpdir, "model.onnx")
            backend = MockTorchBackend()
            trainer = ProjectionMLPTrainer(TrainConfig(), backend=backend)
            result = trainer.export_onnx(ckpt_path, onnx_path, verify=False)
            assert result.verified is True  # default to True when skipped
            assert result.max_abs_diff == 0.0

    def test_export_creates_output_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = os.path.join(tmpdir, "model.pt")
            Path(ckpt_path).write_text("mock")
            nested_path = os.path.join(tmpdir, "subdir", "nested", "model.onnx")
            backend = MockTorchBackend()
            trainer = ProjectionMLPTrainer(TrainConfig(), backend=backend)
            result = trainer.export_onnx(ckpt_path, nested_path)
            assert Path(nested_path).exists()


# ---------------------------------------------------------------------------
# CLI argument tests
# ---------------------------------------------------------------------------

class TestCLI:
    def test_train_config_defaults_from_args(self):
        """Verify TrainConfig.from_args handles missing attributes gracefully."""
        args = MagicMock(spec=[])  # no attributes
        cfg = TrainConfig.from_args(args)
        assert cfg.epochs == 10
        assert cfg.batch_size == 256

    def test_main_requires_command(self):
        """main() should fail without subcommand."""
        from fine_tune.train_projection_mlp import main
        with pytest.raises(SystemExit):
            with patch("sys.argv", ["train_projection_mlp"]):
                main()
