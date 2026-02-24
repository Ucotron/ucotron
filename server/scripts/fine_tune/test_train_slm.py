#!/usr/bin/env python3
"""
Tests for FireworksFineTuner — remote fine-tuning client.

All tests use mocked HTTP responses (no real API calls).
Run with: python -m pytest scripts/fine_tune/test_train_slm.py -v
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure FIREWORKS_API_KEY is set for tests (never calls real API)
os.environ.setdefault("FIREWORKS_API_KEY", "fw-test-key-not-real")
os.environ.setdefault("FIREWORKS_ACCOUNT_ID", "test-account")

from fine_tune.train_slm import (
    MODELS,
    DEFAULT_RE_TEST_PROMPTS,
    RE_SYSTEM_PROMPT,
    DpoJobConfig,
    FireworksFineTuner,
    FireworksAuthError,
    FireworksError,
    FireworksNotFoundError,
    FireworksRateLimitError,
    JobStatus,
    SftJobConfig,
    JOB_STATE_COMPLETED,
    JOB_STATE_CREATING,
    JOB_STATE_FAILED,
    JOB_STATE_PENDING,
    JOB_STATE_RUNNING,
    TERMINAL_STATES,
    ACTIVE_STATES,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tuner(**kwargs) -> FireworksFineTuner:
    """Create a FireworksFineTuner with test defaults."""
    defaults = {
        "account_id": "test-account",
        "api_key": "fw-test-key",
        "max_retries": 1,
        "retry_delay": 0.01,
    }
    defaults.update(kwargs)
    return FireworksFineTuner(**defaults)


def _mock_response(status_code: int = 200, json_data: dict | None = None, text: str = "") -> MagicMock:
    """Create a mock requests.Response."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = text or json.dumps(json_data or {})
    resp.content = resp.text.encode() if resp.text else b""
    resp.json.return_value = json_data or {}
    return resp


def _write_jsonl(tmpdir: str, lines: list[dict], name: str = "train.jsonl") -> Path:
    """Write a JSONL file to tmpdir."""
    path = Path(tmpdir) / name
    with open(path, "w") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")
    return path


# ---------------------------------------------------------------------------
# MODELS registry tests
# ---------------------------------------------------------------------------

class TestModels:
    def test_three_model_sizes(self):
        assert "slm" in MODELS
        assert "small" in MODELS
        assert "medium" in MODELS

    def test_slm_is_0_5b(self):
        assert MODELS["slm"]["params"] == "0.5B"
        assert "0.5b" in MODELS["slm"]["model_id"]

    def test_small_is_1_5b(self):
        assert MODELS["small"]["params"] == "1.5B"
        assert "1.5b" in MODELS["small"]["model_id"]

    def test_medium_is_7b(self):
        assert MODELS["medium"]["params"] == "7B"
        assert "7b" in MODELS["medium"]["model_id"]

    def test_all_models_have_required_fields(self):
        required = {"model_id", "display_name", "params", "use_case",
                     "default_epochs", "default_lora_rank", "default_learning_rate",
                     "default_max_context"}
        for key, model in MODELS.items():
            missing = required - set(model.keys())
            assert not missing, f"Model '{key}' missing fields: {missing}"


# ---------------------------------------------------------------------------
# SftJobConfig tests
# ---------------------------------------------------------------------------

class TestSftJobConfig:
    def test_defaults(self):
        cfg = SftJobConfig()
        assert cfg.epochs == 3
        assert cfg.learning_rate == 2e-4
        assert cfg.lora_rank == 16
        assert cfg.max_context_length == 2048

    def test_to_api_params_basic(self):
        cfg = SftJobConfig(epochs=5, learning_rate=1e-5, lora_rank=8)
        params = cfg.to_api_params()
        assert params["epochs"] == 5
        assert params["learningRate"] == 1e-5
        assert params["loraRank"] == 8
        assert "batchSize" not in params

    def test_to_api_params_with_batch_size(self):
        cfg = SftJobConfig(batch_size=32)
        params = cfg.to_api_params()
        assert params["batchSize"] == 32

    def test_to_api_params_with_eval_dataset(self):
        cfg = SftJobConfig(evaluation_dataset="eval-ds-1")
        params = cfg.to_api_params()
        assert params["evaluationDataset"] == "eval-ds-1"

    def test_to_api_params_with_wandb(self):
        cfg = SftJobConfig(wandb_project="ucotron-ft", wandb_api_key="wb-key-123")
        params = cfg.to_api_params()
        assert params["wandbConfig"]["project"] == "ucotron-ft"
        assert params["wandbConfig"]["apiKey"] == "wb-key-123"

    def test_to_api_params_wandb_requires_both(self):
        cfg = SftJobConfig(wandb_project="ucotron-ft")  # No api_key
        params = cfg.to_api_params()
        assert "wandbConfig" not in params


# ---------------------------------------------------------------------------
# JobStatus tests
# ---------------------------------------------------------------------------

class TestJobStatus:
    def test_terminal_states(self):
        for state in TERMINAL_STATES:
            s = JobStatus(name="j1", state=state)
            assert s.is_terminal
            assert not s.is_active

    def test_active_states(self):
        for state in ACTIVE_STATES:
            s = JobStatus(name="j1", state=state)
            assert s.is_active
            assert not s.is_terminal

    def test_is_success(self):
        s = JobStatus(name="j1", state=JOB_STATE_COMPLETED)
        assert s.is_success

    def test_failed_not_success(self):
        s = JobStatus(name="j1", state=JOB_STATE_FAILED)
        assert not s.is_success
        assert s.is_terminal


# ---------------------------------------------------------------------------
# FireworksFineTuner construction tests
# ---------------------------------------------------------------------------

class TestFireworksFineTunerInit:
    def test_requires_api_key(self):
        with patch.dict(os.environ, {}, clear=True):
            # Remove env vars to force error
            env = {k: v for k, v in os.environ.items()
                   if k not in ("FIREWORKS_API_KEY",)}
            with patch.dict(os.environ, env, clear=True):
                with pytest.raises(ValueError, match="API key"):
                    FireworksFineTuner(account_id="acc")

    def test_requires_account_id(self):
        with patch.dict(os.environ, {}, clear=True):
            env = {k: v for k, v in os.environ.items()
                   if k not in ("FIREWORKS_ACCOUNT_ID",)}
            with patch.dict(os.environ, env, clear=True):
                with pytest.raises(ValueError, match="account ID"):
                    FireworksFineTuner(api_key="fw-key")

    def test_constructs_with_explicit_args(self):
        tuner = _make_tuner()
        assert tuner.account_id == "test-account"
        assert tuner.api_key == "fw-test-key"

    def test_constructs_from_env(self):
        with patch.dict(os.environ, {
            "FIREWORKS_API_KEY": "fw-env-key",
            "FIREWORKS_ACCOUNT_ID": "env-account",
        }):
            tuner = FireworksFineTuner()
            assert tuner.api_key == "fw-env-key"
            assert tuner.account_id == "env-account"


# ---------------------------------------------------------------------------
# Dataset operations tests
# ---------------------------------------------------------------------------

class TestDatasetOperations:
    def test_upload_file_not_found(self):
        tuner = _make_tuner()
        with pytest.raises(FileNotFoundError):
            tuner.upload_file("/nonexistent/train.jsonl")

    @patch("fine_tune.train_slm.requests.request")
    def test_upload_file_success(self, mock_req):
        mock_req.return_value = _mock_response(200, {"name": "datasets/train"})

        tuner = _make_tuner()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_jsonl(tmpdir, [
                {"messages": [{"role": "user", "content": "hi"}]},
                {"messages": [{"role": "user", "content": "hello"}]},
            ])
            result = tuner.upload_file(path)
            assert result == "datasets/train"

    def test_upload_invalid_jsonl(self):
        tuner = _make_tuner()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bad.jsonl"
            path.write_text("not json\n")
            with pytest.raises(ValueError, match="Invalid JSON"):
                tuner.upload_file(path)

    def test_upload_empty_jsonl(self):
        tuner = _make_tuner()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "empty.jsonl"
            path.write_text("\n\n")
            with pytest.raises(ValueError, match="Empty JSONL"):
                tuner.upload_file(path)

    @patch.object(FireworksFineTuner, "_request")
    def test_list_datasets(self, mock_req):
        mock_req.return_value = {"datasets": [{"name": "ds1"}, {"name": "ds2"}]}
        tuner = _make_tuner()
        datasets = tuner.list_datasets()
        assert len(datasets) == 2


# ---------------------------------------------------------------------------
# SFT job creation tests
# ---------------------------------------------------------------------------

class TestCreateSftJob:
    @patch.object(FireworksFineTuner, "_request")
    def test_create_basic(self, mock_req):
        mock_req.return_value = {
            "name": "accounts/test/sftj/job-123",
            "state": JOB_STATE_CREATING,
        }
        tuner = _make_tuner()
        job = tuner.create_sft_job(
            dataset="train-ds",
            base_model=MODELS["slm"]["model_id"],
        )
        assert "job-123" in job["name"]

        # Verify request body
        call_args = mock_req.call_args
        body = call_args.kwargs.get("json", call_args[1].get("json", {}))
        assert body["dataset"] == "train-ds"
        assert body["baseModel"] == MODELS["slm"]["model_id"]

    @patch.object(FireworksFineTuner, "_request")
    def test_create_with_output_model(self, mock_req):
        mock_req.return_value = {"name": "job-1", "state": JOB_STATE_CREATING}
        tuner = _make_tuner()
        tuner.create_sft_job(
            dataset="ds",
            base_model=MODELS["medium"]["model_id"],
            output_model="my-custom-model",
        )
        body = mock_req.call_args.kwargs.get("json", mock_req.call_args[1].get("json", {}))
        assert body["outputModel"] == "my-custom-model"

    @patch.object(FireworksFineTuner, "_request")
    def test_create_with_custom_config(self, mock_req):
        mock_req.return_value = {"name": "job-2", "state": JOB_STATE_CREATING}
        cfg = SftJobConfig(epochs=5, learning_rate=1e-5, lora_rank=32)
        tuner = _make_tuner()
        tuner.create_sft_job(dataset="ds", base_model="model-id", config=cfg)
        body = mock_req.call_args.kwargs.get("json", mock_req.call_args[1].get("json", {}))
        assert body["epochs"] == 5
        assert body["learningRate"] == 1e-5
        assert body["loraRank"] == 32

    @patch.object(FireworksFineTuner, "_request")
    def test_create_qwen_0_5b_with_model_defaults(self, mock_req):
        """End-to-end: create SFT job for Qwen2.5-0.5B using model defaults."""
        mock_req.return_value = {
            "name": "accounts/test/sftj/qwen-sft-001",
            "state": JOB_STATE_CREATING,
            "baseModel": MODELS["slm"]["model_id"],
        }
        tuner = _make_tuner()
        cfg = tuner.create_sft_config_for_model("slm")
        job = tuner.create_sft_job(
            dataset="ucotron-re-train",
            base_model=MODELS["slm"]["model_id"],
            output_model="ucotron-re-qwen-0.5b",
            config=cfg,
            display_name="Ucotron RE Qwen 0.5B SFT",
        )
        assert job["name"] == "accounts/test/sftj/qwen-sft-001"

        body = mock_req.call_args.kwargs.get("json", mock_req.call_args[1].get("json", {}))
        assert body["baseModel"] == "accounts/fireworks/models/qwen2p5-0.5b-instruct"
        assert body["outputModel"] == "ucotron-re-qwen-0.5b"
        assert body["displayName"] == "Ucotron RE Qwen 0.5B SFT"
        assert body["epochs"] == 3
        assert body["learningRate"] == 2e-4
        assert body["loraRank"] == 8
        assert body["maxContextLength"] == 2048

    @patch.object(FireworksFineTuner, "_request")
    def test_create_with_batch_size(self, mock_req):
        """Verify batch_size is included in API params when set."""
        mock_req.return_value = {"name": "job-bs", "state": JOB_STATE_CREATING}
        cfg = SftJobConfig(batch_size=4)
        tuner = _make_tuner()
        tuner.create_sft_job(dataset="ds", base_model="model-id", config=cfg)
        body = mock_req.call_args.kwargs.get("json", mock_req.call_args[1].get("json", {}))
        assert body["batchSize"] == 4

    @patch.object(FireworksFineTuner, "_request")
    def test_create_without_batch_size_omits_key(self, mock_req):
        """Verify batch_size is omitted from API params when None."""
        mock_req.return_value = {"name": "job-no-bs", "state": JOB_STATE_CREATING}
        cfg = SftJobConfig()
        tuner = _make_tuner()
        tuner.create_sft_job(dataset="ds", base_model="model-id", config=cfg)
        body = mock_req.call_args.kwargs.get("json", mock_req.call_args[1].get("json", {}))
        assert "batchSize" not in body


# ---------------------------------------------------------------------------
# DpoJobConfig tests
# ---------------------------------------------------------------------------

class TestDpoJobConfig:
    def test_defaults(self):
        cfg = DpoJobConfig()
        assert cfg.epochs == 2
        assert cfg.learning_rate == 5e-5
        assert cfg.lora_rank == 8
        assert cfg.max_context_length == 2048
        assert cfg.beta == 0.1
        assert cfg.loss_method == "DPO"

    def test_to_api_params_structure(self):
        """DPO params use trainingConfig + lossConfig wrappers."""
        cfg = DpoJobConfig(epochs=3, learning_rate=1e-5, lora_rank=16, beta=0.2)
        params = cfg.to_api_params()
        assert "trainingConfig" in params
        assert "lossConfig" in params
        assert params["trainingConfig"]["epochs"] == 3
        assert params["trainingConfig"]["learningRate"] == 1e-5
        assert params["trainingConfig"]["loraRank"] == 16
        assert params["lossConfig"]["method"] == "DPO"
        assert params["lossConfig"]["klBeta"] == 0.2

    def test_to_api_params_without_batch_size(self):
        cfg = DpoJobConfig()
        params = cfg.to_api_params()
        assert "batchSize" not in params["trainingConfig"]

    def test_to_api_params_with_batch_size(self):
        cfg = DpoJobConfig(batch_size=4)
        params = cfg.to_api_params()
        assert params["trainingConfig"]["batchSize"] == 4

    def test_to_api_params_with_wandb(self):
        cfg = DpoJobConfig(wandb_project="ucotron-dpo", wandb_api_key="wb-key")
        params = cfg.to_api_params()
        assert params["wandbConfig"]["project"] == "ucotron-dpo"

    def test_to_api_params_wandb_requires_both(self):
        cfg = DpoJobConfig(wandb_project="ucotron-dpo")
        params = cfg.to_api_params()
        assert "wandbConfig" not in params

    def test_custom_loss_method(self):
        cfg = DpoJobConfig(loss_method="GRPO", beta=0.5)
        params = cfg.to_api_params()
        assert params["lossConfig"]["method"] == "GRPO"
        assert params["lossConfig"]["klBeta"] == 0.5


# ---------------------------------------------------------------------------
# DPO job creation tests
# ---------------------------------------------------------------------------

class TestCreateDpoJob:
    @patch.object(FireworksFineTuner, "_request")
    def test_create_basic(self, mock_req):
        mock_req.return_value = {
            "name": "accounts/test/dpoj/dpo-123",
            "state": JOB_STATE_CREATING,
        }
        tuner = _make_tuner()
        job = tuner.create_dpo_job(
            dataset="pref-ds",
            base_model="accounts/test/models/sft-model",
        )
        assert "dpo-123" in job["name"]

        # Verify request body
        call_args = mock_req.call_args
        body = call_args.kwargs.get("json", call_args[1].get("json", {}))
        assert body["dataset"] == "pref-ds"
        assert body["trainingConfig"]["baseModel"] == "accounts/test/models/sft-model"
        assert body["lossConfig"]["method"] == "DPO"
        assert body["lossConfig"]["klBeta"] == 0.1

    @patch.object(FireworksFineTuner, "_request")
    def test_create_with_output_model(self, mock_req):
        mock_req.return_value = {"name": "dpoj-1", "state": JOB_STATE_CREATING}
        tuner = _make_tuner()
        tuner.create_dpo_job(
            dataset="ds",
            base_model="sft-model",
            output_model="my-dpo-model",
        )
        body = mock_req.call_args.kwargs.get("json", mock_req.call_args[1].get("json", {}))
        assert body["trainingConfig"]["outputModel"] == "my-dpo-model"

    @patch.object(FireworksFineTuner, "_request")
    def test_create_with_display_name(self, mock_req):
        mock_req.return_value = {"name": "dpoj-2", "state": JOB_STATE_CREATING}
        tuner = _make_tuner()
        tuner.create_dpo_job(
            dataset="ds",
            base_model="sft-model",
            display_name="My DPO Run",
        )
        body = mock_req.call_args.kwargs.get("json", mock_req.call_args[1].get("json", {}))
        assert body["displayName"] == "My DPO Run"

    @patch.object(FireworksFineTuner, "_request")
    def test_create_with_custom_config(self, mock_req):
        mock_req.return_value = {"name": "dpoj-3", "state": JOB_STATE_CREATING}
        cfg = DpoJobConfig(epochs=3, learning_rate=1e-5, lora_rank=32, beta=0.3)
        tuner = _make_tuner()
        tuner.create_dpo_job(dataset="ds", base_model="model-id", config=cfg)
        body = mock_req.call_args.kwargs.get("json", mock_req.call_args[1].get("json", {}))
        assert body["trainingConfig"]["epochs"] == 3
        assert body["trainingConfig"]["learningRate"] == 1e-5
        assert body["trainingConfig"]["loraRank"] == 32
        assert body["lossConfig"]["klBeta"] == 0.3

    @patch.object(FireworksFineTuner, "_request")
    def test_create_uses_dpo_endpoint(self, mock_req):
        """Verify DPO uses /dpoJobs, not /supervisedFineTuningJobs."""
        mock_req.return_value = {"name": "dpoj-4", "state": JOB_STATE_CREATING}
        tuner = _make_tuner()
        tuner.create_dpo_job(dataset="ds", base_model="model")
        call_args = mock_req.call_args
        url = call_args[0][1]  # Second positional arg is URL
        assert "/dpoJobs" in url
        assert "/supervisedFineTuningJobs" not in url

    @patch.object(FireworksFineTuner, "_request")
    def test_create_qwen_0_5b_dpo_with_model_defaults(self, mock_req):
        """End-to-end: create DPO job for Qwen 0.5B using SFT model as base."""
        mock_req.return_value = {
            "name": "accounts/test/dpoj/qwen-dpo-001",
            "state": JOB_STATE_CREATING,
        }
        tuner = _make_tuner()
        cfg = tuner.create_dpo_config_for_model("slm", beta=0.15)
        job = tuner.create_dpo_job(
            dataset="ucotron-pref-train",
            base_model="accounts/test/models/ucotron-re-qwen-0.5b",
            output_model="ucotron-dpo-qwen-0.5b",
            config=cfg,
            display_name="Ucotron DPO Qwen 0.5B",
        )
        assert job["name"] == "accounts/test/dpoj/qwen-dpo-001"

        body = mock_req.call_args.kwargs.get("json", mock_req.call_args[1].get("json", {}))
        assert body["trainingConfig"]["baseModel"] == "accounts/test/models/ucotron-re-qwen-0.5b"
        assert body["trainingConfig"]["outputModel"] == "ucotron-dpo-qwen-0.5b"
        assert body["trainingConfig"]["epochs"] == 2
        assert body["trainingConfig"]["learningRate"] == 5e-5
        assert body["trainingConfig"]["loraRank"] == 8
        assert body["lossConfig"]["klBeta"] == 0.15
        assert body["displayName"] == "Ucotron DPO Qwen 0.5B"


# ---------------------------------------------------------------------------
# Job status tests
# ---------------------------------------------------------------------------

class TestJobStatus2:
    @patch.object(FireworksFineTuner, "_request")
    def test_get_job(self, mock_req):
        mock_req.return_value = {
            "name": "job-1",
            "state": JOB_STATE_RUNNING,
            "jobProgress": {"percent": 45.0, "epoch": 1.5, "tokensProcessed": 100000},
            "estimatedCost": "$1.23",
            "outputModel": "my-model",
        }
        tuner = _make_tuner()
        status = tuner.get_job("job-1")
        assert status.name == "job-1"
        assert status.state == JOB_STATE_RUNNING
        assert status.progress_percent == 45.0
        assert status.progress_epoch == 1.5
        assert status.tokens_processed == 100000
        assert status.estimated_cost == "$1.23"
        assert status.is_active

    @patch.object(FireworksFineTuner, "_request")
    def test_list_jobs(self, mock_req):
        mock_req.return_value = {
            "supervisedFineTuningJobs": [
                {"name": "j1", "state": "COMPLETED"},
                {"name": "j2", "state": "RUNNING"},
            ]
        }
        tuner = _make_tuner()
        jobs = tuner.list_jobs()
        assert len(jobs) == 2


# ---------------------------------------------------------------------------
# wait_for_completion tests
# ---------------------------------------------------------------------------

class TestWaitForCompletion:
    @patch.object(FireworksFineTuner, "get_job")
    def test_already_completed(self, mock_get):
        mock_get.return_value = JobStatus(name="j1", state=JOB_STATE_COMPLETED)
        tuner = _make_tuner()
        result = tuner.wait_for_completion("j1", poll_interval=0.01)
        assert result.is_success
        mock_get.assert_called_once()

    @patch.object(FireworksFineTuner, "get_job")
    def test_polls_until_complete(self, mock_get):
        mock_get.side_effect = [
            JobStatus(name="j1", state=JOB_STATE_RUNNING, progress_percent=50.0),
            JobStatus(name="j1", state=JOB_STATE_RUNNING, progress_percent=90.0),
            JobStatus(name="j1", state=JOB_STATE_COMPLETED, progress_percent=100.0),
        ]
        tuner = _make_tuner()
        result = tuner.wait_for_completion("j1", poll_interval=0.01)
        assert result.is_success
        assert mock_get.call_count == 3

    @patch.object(FireworksFineTuner, "get_job")
    def test_raises_on_failure(self, mock_get):
        mock_get.return_value = JobStatus(
            name="j1", state=JOB_STATE_FAILED, error_message="OOM"
        )
        tuner = _make_tuner()
        with pytest.raises(FireworksError, match="failed"):
            tuner.wait_for_completion("j1", poll_interval=0.01)

    @patch.object(FireworksFineTuner, "get_job")
    def test_timeout(self, mock_get):
        mock_get.return_value = JobStatus(name="j1", state=JOB_STATE_RUNNING)
        tuner = _make_tuner()
        with pytest.raises(TimeoutError):
            tuner.wait_for_completion("j1", poll_interval=0.01, timeout=0.03)

    @patch.object(FireworksFineTuner, "get_job")
    def test_progress_callback(self, mock_get):
        mock_get.side_effect = [
            JobStatus(name="j1", state=JOB_STATE_RUNNING),
            JobStatus(name="j1", state=JOB_STATE_COMPLETED),
        ]
        callback = MagicMock()
        tuner = _make_tuner()
        tuner.wait_for_completion("j1", poll_interval=0.01, progress_callback=callback)
        assert callback.call_count == 2


# ---------------------------------------------------------------------------
# Cancel job tests
# ---------------------------------------------------------------------------

class TestCancelJob:
    @patch.object(FireworksFineTuner, "_request")
    def test_cancel(self, mock_req):
        mock_req.return_value = {"name": "j1", "state": "JOB_STATE_CANCELLED"}
        tuner = _make_tuner()
        result = tuner.cancel_job("j1")
        assert result["state"] == "JOB_STATE_CANCELLED"


# ---------------------------------------------------------------------------
# Model testing
# ---------------------------------------------------------------------------

class TestModelTesting:
    @patch.object(FireworksFineTuner, "_request")
    def test_test_model_success(self, mock_req):
        mock_req.return_value = {
            "choices": [{
                "message": {"content": "Paris is the capital."},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 20, "completion_tokens": 10},
        }
        tuner = _make_tuner()
        results = tuner.test_model("my-model", ["What is the capital of France?"])
        assert len(results) == 1
        assert results[0]["response"] == "Paris is the capital."
        assert results[0]["prompt_tokens"] == 20

    @patch.object(FireworksFineTuner, "_request")
    def test_test_model_error_handled(self, mock_req):
        mock_req.side_effect = FireworksError("Server error", status_code=500)
        tuner = _make_tuner()
        results = tuner.test_model("my-model", ["test prompt"])
        assert len(results) == 1
        assert results[0]["response"] == ""
        assert "error" in results[0]


# ---------------------------------------------------------------------------
# Error handling tests
# ---------------------------------------------------------------------------

class TestErrorHandling:
    def test_auth_error(self):
        tuner = _make_tuner()
        with patch.object(tuner._session, "request", return_value=_mock_response(401)):
            with pytest.raises(FireworksAuthError):
                tuner._request("GET", "https://api.fireworks.ai/test")

    def test_not_found_error(self):
        tuner = _make_tuner()
        with patch.object(tuner._session, "request", return_value=_mock_response(404)):
            with pytest.raises(FireworksNotFoundError):
                tuner._request("GET", "https://api.fireworks.ai/test")

    def test_rate_limit_retried(self):
        tuner = _make_tuner(max_retries=2, retry_delay=0.01)
        resp_429 = _mock_response(429)
        resp_200 = _mock_response(200, {"ok": True})
        with patch.object(tuner._session, "request", side_effect=[resp_429, resp_200]):
            result = tuner._request("GET", "https://api.fireworks.ai/test")
            assert result == {"ok": True}

    def test_server_error_retried(self):
        tuner = _make_tuner(max_retries=2, retry_delay=0.01)
        resp_500 = _mock_response(500, {"error": {"message": "Internal"}})
        resp_200 = _mock_response(200, {"ok": True})
        with patch.object(tuner._session, "request", side_effect=[resp_500, resp_200]):
            result = tuner._request("GET", "https://api.fireworks.ai/test")
            assert result == {"ok": True}

    def test_client_error_not_retried(self):
        tuner = _make_tuner(max_retries=3, retry_delay=0.01)
        resp_400 = _mock_response(400, {"error": {"message": "Bad request"}})
        with patch.object(tuner._session, "request", return_value=resp_400):
            with pytest.raises(FireworksError, match="Bad request"):
                tuner._request("GET", "https://api.fireworks.ai/test")

    def test_retries_exhausted(self):
        tuner = _make_tuner(max_retries=2, retry_delay=0.01)
        resp_500 = _mock_response(500, {"error": {"message": "Down"}})
        with patch.object(tuner._session, "request", return_value=resp_500):
            with pytest.raises(FireworksError, match="retries"):
                tuner._request("GET", "https://api.fireworks.ai/test")


# ---------------------------------------------------------------------------
# Helper method tests
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_get_model_config(self):
        tuner = _make_tuner()
        cfg = tuner.get_model_config("slm")
        assert cfg["params"] == "0.5B"
        # Returns copy
        cfg["params"] = "changed"
        assert MODELS["slm"]["params"] == "0.5B"

    def test_get_model_config_unknown(self):
        tuner = _make_tuner()
        with pytest.raises(KeyError, match="unknown"):
            tuner.get_model_config("unknown")

    def test_create_sft_config_for_model(self):
        tuner = _make_tuner()
        cfg = tuner.create_sft_config_for_model("medium")
        assert cfg.epochs == MODELS["medium"]["default_epochs"]
        assert cfg.lora_rank == MODELS["medium"]["default_lora_rank"]
        assert cfg.learning_rate == MODELS["medium"]["default_learning_rate"]
        assert cfg.max_context_length == MODELS["medium"]["default_max_context"]

    def test_create_dpo_config_for_model_slm(self):
        tuner = _make_tuner()
        cfg = tuner.create_dpo_config_for_model("slm")
        assert cfg.epochs == MODELS["slm"]["dpo_epochs"]
        assert cfg.learning_rate == MODELS["slm"]["dpo_learning_rate"]
        assert cfg.lora_rank == MODELS["slm"]["dpo_lora_rank"]
        assert cfg.max_context_length == MODELS["slm"]["dpo_max_context"]
        assert cfg.beta == 0.1  # default

    def test_create_dpo_config_for_model_medium(self):
        tuner = _make_tuner()
        cfg = tuner.create_dpo_config_for_model("medium", beta=0.2)
        assert cfg.epochs == MODELS["medium"]["dpo_epochs"]
        assert cfg.learning_rate == MODELS["medium"]["dpo_learning_rate"]
        assert cfg.lora_rank == MODELS["medium"]["dpo_lora_rank"]
        assert cfg.beta == 0.2

    def test_create_dpo_config_unknown_model(self):
        tuner = _make_tuner()
        with pytest.raises(KeyError, match="unknown"):
            tuner.create_dpo_config_for_model("unknown")

    def test_all_models_have_dpo_fields(self):
        """Every model tier must have DPO default fields."""
        dpo_fields = {"dpo_epochs", "dpo_learning_rate", "dpo_lora_rank", "dpo_max_context"}
        for key, model in MODELS.items():
            missing = dpo_fields - set(model.keys())
            assert not missing, f"Model '{key}' missing DPO fields: {missing}"

    def test_validate_jsonl_valid(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_jsonl(tmpdir, [{"a": 1}, {"b": 2}])
            count = FireworksFineTuner._validate_jsonl(path)
            assert count == 2

    def test_validate_jsonl_invalid(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bad.jsonl"
            path.write_text('{"ok": true}\nnot json\n')
            with pytest.raises(ValueError, match="Invalid JSON at line 2"):
                FireworksFineTuner._validate_jsonl(path)

    def test_validate_jsonl_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "empty.jsonl"
            path.write_text("")
            with pytest.raises(ValueError, match="Empty JSONL"):
                FireworksFineTuner._validate_jsonl(path)

    def test_parse_job_status(self):
        data = {
            "name": "accounts/x/sftj/y",
            "state": "JOB_STATE_RUNNING",
            "createTime": "2025-01-01T00:00:00Z",
            "jobProgress": {
                "percent": 55.5,
                "epoch": 1.7,
                "tokensProcessed": 50000,
            },
            "estimatedCost": "$2.50",
            "outputModel": "my-output",
            "status": {"message": "Training in progress"},
        }
        status = FireworksFineTuner._parse_job_status(data)
        assert status.name == "accounts/x/sftj/y"
        assert status.state == "JOB_STATE_RUNNING"
        assert status.progress_percent == 55.5
        assert status.progress_epoch == 1.7
        assert status.tokens_processed == 50000
        assert status.output_model == "my-output"

    def test_parse_estimated_finish_time(self):
        data = {
            "name": "j1",
            "state": "JOB_STATE_RUNNING",
            "estimatedCompletionTime": "2025-06-15T14:30:00Z",
        }
        status = FireworksFineTuner._parse_job_status(data)
        assert status.estimated_finish_time == "2025-06-15T14:30:00Z"

    def test_parse_no_estimated_finish_time(self):
        data = {"name": "j1", "state": "JOB_STATE_RUNNING"}
        status = FireworksFineTuner._parse_job_status(data)
        assert status.estimated_finish_time == ""


# ---------------------------------------------------------------------------
# wait_for_job tests
# ---------------------------------------------------------------------------

class TestWaitForJob:
    """Tests for wait_for_job (enhanced polling with progress display)."""

    @patch.object(FireworksFineTuner, "get_job")
    def test_already_completed(self, mock_get):
        mock_get.return_value = JobStatus(name="j1", state=JOB_STATE_COMPLETED)
        tuner = _make_tuner()
        result = tuner.wait_for_job("j1", poll_interval=0.01)
        assert result.is_success
        mock_get.assert_called_once()

    @patch.object(FireworksFineTuner, "get_job")
    def test_polls_until_complete(self, mock_get):
        mock_get.side_effect = [
            JobStatus(name="j1", state=JOB_STATE_PENDING, progress_percent=0.0),
            JobStatus(name="j1", state=JOB_STATE_RUNNING, progress_percent=50.0),
            JobStatus(name="j1", state=JOB_STATE_COMPLETED, progress_percent=100.0),
        ]
        tuner = _make_tuner()
        result = tuner.wait_for_job("j1", poll_interval=0.01)
        assert result.is_success
        assert mock_get.call_count == 3

    @patch.object(FireworksFineTuner, "get_job")
    def test_raises_on_failure(self, mock_get):
        mock_get.return_value = JobStatus(
            name="j1", state=JOB_STATE_FAILED, error_message="OOM"
        )
        tuner = _make_tuner()
        with pytest.raises(FireworksError, match="failed"):
            tuner.wait_for_job("j1", poll_interval=0.01)

    @patch.object(FireworksFineTuner, "get_job")
    def test_timeout(self, mock_get):
        mock_get.return_value = JobStatus(name="j1", state=JOB_STATE_RUNNING)
        tuner = _make_tuner()
        with pytest.raises(TimeoutError):
            tuner.wait_for_job("j1", poll_interval=0.01, timeout=0.03)

    @patch.object(FireworksFineTuner, "get_job")
    def test_progress_callback(self, mock_get):
        mock_get.side_effect = [
            JobStatus(name="j1", state=JOB_STATE_RUNNING),
            JobStatus(name="j1", state=JOB_STATE_COMPLETED),
        ]
        callback = MagicMock()
        tuner = _make_tuner()
        tuner.wait_for_job("j1", poll_interval=0.01, progress_callback=callback)
        assert callback.call_count == 2

    @patch.object(FireworksFineTuner, "get_job")
    def test_shows_api_estimated_finish_time(self, mock_get):
        """When API provides estimated finish time, it should be used."""
        mock_get.side_effect = [
            JobStatus(
                name="j1",
                state=JOB_STATE_RUNNING,
                progress_percent=50.0,
                estimated_finish_time="2025-06-15T14:30:00Z",
            ),
            JobStatus(name="j1", state=JOB_STATE_COMPLETED, progress_percent=100.0),
        ]
        tuner = _make_tuner()
        result = tuner.wait_for_job("j1", poll_interval=0.01)
        assert result.is_success

    @patch.object(FireworksFineTuner, "get_job")
    def test_backward_compat_wait_for_completion(self, mock_get):
        """wait_for_completion still works as alias."""
        mock_get.return_value = JobStatus(name="j1", state=JOB_STATE_COMPLETED)
        tuner = _make_tuner()
        result = tuner.wait_for_completion("j1", poll_interval=0.01)
        assert result.is_success


# ---------------------------------------------------------------------------
# display_state and ETA estimation tests
# ---------------------------------------------------------------------------

class TestJobStatusDisplay:
    def test_display_state_mapping(self):
        assert JobStatus(name="j", state=JOB_STATE_CREATING).display_state == "creating"
        assert JobStatus(name="j", state=JOB_STATE_PENDING).display_state == "queued"
        assert JobStatus(name="j", state=JOB_STATE_RUNNING).display_state == "running"
        assert JobStatus(name="j", state=JOB_STATE_COMPLETED).display_state == "succeeded"
        assert JobStatus(name="j", state=JOB_STATE_FAILED).display_state == "failed"

    def test_display_state_unknown(self):
        assert JobStatus(name="j", state="WEIRD_STATE").display_state == "weird_state"

    def test_estimate_remaining_at_50_percent(self):
        # 60s elapsed, 50% done → 60s remaining
        result = FireworksFineTuner._estimate_remaining(60.0, 50.0)
        assert result == "1m 0s"

    def test_estimate_remaining_at_75_percent(self):
        # 90s elapsed, 75% done → 30s remaining
        result = FireworksFineTuner._estimate_remaining(90.0, 75.0)
        assert result == "30s"

    def test_estimate_remaining_zero_progress(self):
        result = FireworksFineTuner._estimate_remaining(60.0, 0.0)
        assert result is None

    def test_estimate_remaining_complete(self):
        result = FireworksFineTuner._estimate_remaining(60.0, 100.0)
        assert result is None

    def test_estimate_remaining_hours(self):
        # 3600s elapsed (1h), 25% done → 3h remaining
        result = FireworksFineTuner._estimate_remaining(3600.0, 25.0)
        assert result == "3h 0m 0s"

    def test_format_elapsed_seconds(self):
        assert FireworksFineTuner._format_elapsed(45.0) == "45s"

    def test_format_elapsed_minutes(self):
        assert FireworksFineTuner._format_elapsed(125.0) == "2m 5s"

    def test_format_elapsed_hours(self):
        assert FireworksFineTuner._format_elapsed(3725.0) == "1h 2m 5s"


# ---------------------------------------------------------------------------
# Model testing and comparison tests
# ---------------------------------------------------------------------------

class TestModelTesting:
    """Tests for test_model / compare_models / format_comparison."""

    def test_default_re_prompts_exist(self):
        """DEFAULT_RE_TEST_PROMPTS has at least 3 prompts with expected structure."""
        assert len(DEFAULT_RE_TEST_PROMPTS) >= 3
        for p in DEFAULT_RE_TEST_PROMPTS:
            assert "prompt" in p
            assert "expected_keywords" in p
            assert isinstance(p["expected_keywords"], list)
            assert len(p["expected_keywords"]) > 0

    def test_re_system_prompt_defined(self):
        assert "relation extraction" in RE_SYSTEM_PROMPT.lower()

    @patch.object(FireworksFineTuner, "_request")
    def test_compare_models_all_keywords_hit(self, mock_req):
        """Fine-tuned model returns all keywords, base returns none."""
        call_count = 0

        def side_effect(method, url, json=None, **kw):
            nonlocal call_count
            call_count += 1
            model = json.get("model", "")
            if "finetuned" in model:
                # Fine-tuned responds with all keywords
                return {
                    "choices": [{
                        "message": {"content": '{"entities": [{"name": "Alice"}, {"name": "Google"}], "relations": [{"subject": "Alice", "predicate": "works_at", "object": "Google"}]}'},
                        "finish_reason": "stop",
                    }],
                    "usage": {"prompt_tokens": 50, "completion_tokens": 30},
                }
            else:
                # Base responds with nothing useful
                return {
                    "choices": [{
                        "message": {"content": "I cannot extract relations."},
                        "finish_reason": "stop",
                    }],
                    "usage": {"prompt_tokens": 50, "completion_tokens": 10},
                }

        mock_req.side_effect = side_effect
        tuner = _make_tuner()

        prompts = [{
            "prompt": "Extract relations from: \"Alice works at Google in Mountain View.\"",
            "expected_keywords": ["Alice", "Google", "works_at"],
        }]
        result = tuner.compare_models(
            finetuned_model="accounts/test/models/finetuned",
            base_model="accounts/fireworks/models/qwen2p5-0.5b-instruct",
            prompts=prompts,
        )

        assert result["num_prompts"] == 1
        assert result["summary"]["finetuned_avg_score"] == 1.0
        assert result["summary"]["base_avg_score"] == 0.0
        assert result["summary"]["finetuned_wins"] == 1
        assert result["summary"]["base_wins"] == 0
        assert result["summary"]["improvement"] == 1.0
        assert result["comparisons"][0]["winner"] == "finetuned"

    @patch.object(FireworksFineTuner, "_request")
    def test_compare_models_tie(self, mock_req):
        """Both models return same keywords — tie."""
        mock_req.return_value = {
            "choices": [{
                "message": {"content": "Alice works_at Google in Mountain View."},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 50, "completion_tokens": 20},
        }
        tuner = _make_tuner()
        prompts = [{
            "prompt": "Test prompt",
            "expected_keywords": ["Alice", "Google"],
        }]
        result = tuner.compare_models(
            finetuned_model="ft-model",
            base_model="base-model",
            prompts=prompts,
        )
        assert result["comparisons"][0]["winner"] == "tie"
        assert result["summary"]["ties"] == 1
        assert result["summary"]["improvement"] == 0.0

    @patch.object(FireworksFineTuner, "_request")
    def test_compare_models_uses_default_prompts(self, mock_req):
        """When no prompts passed, uses DEFAULT_RE_TEST_PROMPTS."""
        mock_req.return_value = {
            "choices": [{"message": {"content": "response"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
        tuner = _make_tuner()
        result = tuner.compare_models(
            finetuned_model="ft-model",
            base_model="base-model",
        )
        assert result["num_prompts"] == len(DEFAULT_RE_TEST_PROMPTS)

    @patch.object(FireworksFineTuner, "_request")
    def test_compare_models_handles_errors(self, mock_req):
        """If one model errors, comparison still works."""
        call_count = 0

        def side_effect(method, url, json=None, **kw):
            nonlocal call_count
            call_count += 1
            model = json.get("model", "")
            if "ft" in model:
                raise FireworksError("Server error", status_code=500)
            return {
                "choices": [{"message": {"content": "Alice Google"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            }

        mock_req.side_effect = side_effect
        tuner = _make_tuner()
        prompts = [{
            "prompt": "Test",
            "expected_keywords": ["Alice", "Google"],
        }]
        result = tuner.compare_models(
            finetuned_model="ft-model",
            base_model="base-model",
            prompts=prompts,
        )
        # Fine-tuned errored → score 0, base got hits
        assert result["summary"]["finetuned_avg_score"] == 0.0
        assert result["summary"]["base_avg_score"] == 1.0
        assert result["comparisons"][0]["finetuned"]["error"] is not None

    def test_format_comparison_output(self):
        """format_comparison produces readable output."""
        result = {
            "finetuned_model": "ft-model",
            "base_model": "base-model",
            "num_prompts": 1,
            "comparisons": [{
                "prompt": "Extract relations from test text",
                "expected_keywords": ["Alice", "Google"],
                "finetuned": {
                    "response": "Alice works_at Google",
                    "keyword_hits": 2,
                    "keyword_score": 1.0,
                    "tokens": 15,
                    "error": None,
                },
                "base": {
                    "response": "unknown",
                    "keyword_hits": 0,
                    "keyword_score": 0.0,
                    "tokens": 5,
                    "error": None,
                },
                "winner": "finetuned",
            }],
            "summary": {
                "finetuned_avg_score": 1.0,
                "base_avg_score": 0.0,
                "improvement": 1.0,
                "finetuned_wins": 1,
                "base_wins": 0,
                "ties": 0,
            },
        }
        output = FireworksFineTuner.format_comparison(result)
        assert "Model Comparison Report" in output
        assert "ft-model" in output
        assert "base-model" in output
        assert "FINETUNED" in output  # winner
        assert "100.0%" in output
        assert "0.0%" in output
        assert "+100.0%" in output

    def test_format_comparison_with_error(self):
        """format_comparison shows errors."""
        result = {
            "finetuned_model": "ft",
            "base_model": "base",
            "num_prompts": 1,
            "comparisons": [{
                "prompt": "test",
                "expected_keywords": ["X"],
                "finetuned": {
                    "response": "",
                    "keyword_hits": 0,
                    "keyword_score": 0.0,
                    "tokens": 0,
                    "error": "Server error",
                },
                "base": {
                    "response": "X is here",
                    "keyword_hits": 1,
                    "keyword_score": 1.0,
                    "tokens": 5,
                    "error": None,
                },
                "winner": "base",
            }],
            "summary": {
                "finetuned_avg_score": 0.0,
                "base_avg_score": 1.0,
                "improvement": -1.0,
                "finetuned_wins": 0,
                "base_wins": 1,
                "ties": 0,
            },
        }
        output = FireworksFineTuner.format_comparison(result)
        assert "ERROR: Server error" in output
        assert "BASE" in output
