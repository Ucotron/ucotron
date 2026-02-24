#!/usr/bin/env python3
"""
Remote fine-tuning client for Fireworks.ai supervised fine-tuning API.

Manages the complete lifecycle of fine-tuning jobs on Fireworks:
  - Dataset upload (JSONL files)
  - SFT/DPO job creation with configurable hyperparameters
  - Job status polling with progress reporting
  - Model evaluation against test prompts

Usage:
    from fine_tune.train_slm import FireworksFineTuner, MODELS

    tuner = FireworksFineTuner(account_id="my-account")
    file_id = tuner.upload_file("data/train.jsonl")
    job = tuner.create_sft_job(
        dataset=file_id,
        base_model=MODELS["slm"]["model_id"],
        output_model="my-finetuned-model",
    )
    tuner.wait_for_completion(job["name"])

Environment:
    FIREWORKS_API_KEY  - Required. Fireworks.ai API key (never in config files).
    FIREWORKS_ACCOUNT_ID - Optional. Defaults to constructor parameter.

Requirements:
    pip install requests>=2.28
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Fireworks API constants
# ---------------------------------------------------------------------------
FIREWORKS_API_BASE = "https://api.fireworks.ai/v1"

# Model registry: slm (0.5B edge), small (1.5B), medium (7B cloud)
MODELS: dict[str, dict[str, Any]] = {
    "slm": {
        "model_id": "accounts/fireworks/models/qwen2p5-0.5b-instruct",
        "display_name": "Qwen 2.5 0.5B (Edge SLM)",
        "params": "0.5B",
        "use_case": "Edge deployment, on-device inference",
        "default_epochs": 3,
        "default_lora_rank": 8,
        "default_learning_rate": 2e-4,
        "default_max_context": 2048,
        "dpo_epochs": 2,
        "dpo_learning_rate": 5e-5,
        "dpo_lora_rank": 8,
        "dpo_max_context": 2048,
    },
    "small": {
        "model_id": "accounts/fireworks/models/qwen2p5-1.5b-instruct",
        "display_name": "Qwen 2.5 1.5B",
        "params": "1.5B",
        "use_case": "Balanced quality and latency",
        "default_epochs": 3,
        "default_lora_rank": 16,
        "default_learning_rate": 2e-4,
        "default_max_context": 2048,
        "dpo_epochs": 2,
        "dpo_learning_rate": 5e-5,
        "dpo_lora_rank": 16,
        "dpo_max_context": 2048,
    },
    "medium": {
        "model_id": "accounts/fireworks/models/qwen2p5-7b-instruct",
        "display_name": "Qwen 2.5 7B (Cloud LLM)",
        "params": "7B",
        "use_case": "Cloud deployment, high quality",
        "default_epochs": 2,
        "default_lora_rank": 16,
        "default_learning_rate": 1e-4,
        "default_max_context": 4096,
        "dpo_epochs": 1,
        "dpo_learning_rate": 3e-5,
        "dpo_lora_rank": 16,
        "dpo_max_context": 4096,
    },
}

# Job states
JOB_STATE_CREATING = "JOB_STATE_CREATING"
JOB_STATE_PENDING = "JOB_STATE_PENDING"
JOB_STATE_RUNNING = "JOB_STATE_RUNNING"
JOB_STATE_COMPLETED = "JOB_STATE_COMPLETED"
JOB_STATE_FAILED = "JOB_STATE_FAILED"
JOB_STATE_CANCELLED = "JOB_STATE_CANCELLED"
JOB_STATE_EARLY_STOPPED = "JOB_STATE_EARLY_STOPPED"

TERMINAL_STATES = {
    JOB_STATE_COMPLETED,
    JOB_STATE_FAILED,
    JOB_STATE_CANCELLED,
    JOB_STATE_EARLY_STOPPED,
}

ACTIVE_STATES = {
    JOB_STATE_CREATING,
    JOB_STATE_PENDING,
    JOB_STATE_RUNNING,
}

# Default test prompts for Ucotron relation extraction evaluation
DEFAULT_RE_TEST_PROMPTS: list[dict[str, str]] = [
    {
        "prompt": (
            "Extract relations from: "
            "\"Alice works at Google in Mountain View.\""
        ),
        "expected_keywords": ["Alice", "Google", "Mountain View", "works_at"],
    },
    {
        "prompt": (
            "Extract relations from: "
            "\"Dr. Smith diagnosed the patient with diabetes in 2023.\""
        ),
        "expected_keywords": ["Smith", "patient", "diabetes", "diagnosed"],
    },
    {
        "prompt": (
            "Extract relations from: "
            "\"The Eiffel Tower was designed by Gustave Eiffel "
            "and is located in Paris, France.\""
        ),
        "expected_keywords": ["Eiffel Tower", "Gustave Eiffel", "Paris", "designed_by"],
    },
    {
        "prompt": (
            "Extract relations from: "
            "\"Maria lives in Berlin and studies computer science at TU Berlin.\""
        ),
        "expected_keywords": ["Maria", "Berlin", "TU Berlin", "lives_in", "studies_at"],
    },
    {
        "prompt": (
            "Extract relations from: "
            "\"The meeting was attended by John, Sarah, and Mike at the "
            "New York office on January 15th.\""
        ),
        "expected_keywords": ["John", "Sarah", "Mike", "New York", "attended"],
    },
]

RE_SYSTEM_PROMPT = (
    "You are a relation extraction assistant for a knowledge graph. "
    "Given a sentence, extract entities and their relationships. "
    "Output JSON with 'entities' (list of {name, type}) and "
    "'relations' (list of {subject, predicate, object})."
)


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------
@dataclass
class SftJobConfig:
    """Configuration for a supervised fine-tuning job."""

    epochs: int = 3
    learning_rate: float = 2e-4
    lora_rank: int = 16
    batch_size: int | None = None
    max_context_length: int = 2048
    evaluation_dataset: str | None = None
    wandb_project: str | None = None
    wandb_api_key: str | None = None

    def to_api_params(self) -> dict[str, Any]:
        """Convert to Fireworks API request parameters."""
        params: dict[str, Any] = {
            "epochs": self.epochs,
            "learningRate": self.learning_rate,
            "loraRank": self.lora_rank,
            "maxContextLength": self.max_context_length,
        }
        if self.batch_size is not None:
            params["batchSize"] = self.batch_size
        if self.evaluation_dataset:
            params["evaluationDataset"] = self.evaluation_dataset
        if self.wandb_project and self.wandb_api_key:
            params["wandbConfig"] = {
                "project": self.wandb_project,
                "apiKey": self.wandb_api_key,
            }
        return params


@dataclass
class DpoJobConfig:
    """Configuration for a Direct Preference Optimization (DPO) job.

    DPO aligns a model using preference pairs (chosen/rejected responses).
    Typically run after SFT to refine the model's output quality.

    The beta parameter controls the strength of the KL divergence penalty
    against the reference model — higher beta = more conservative updates.
    """

    epochs: int = 2
    learning_rate: float = 5e-5
    lora_rank: int = 8
    batch_size: int | None = None
    max_context_length: int = 2048
    beta: float = 0.1
    loss_method: str = "DPO"
    wandb_project: str | None = None
    wandb_api_key: str | None = None

    def to_api_params(self) -> dict[str, Any]:
        """Convert to Fireworks DPO API request parameters.

        DPO jobs use a different body structure than SFT:
        - Training params go under ``trainingConfig``
        - Loss params go under ``lossConfig``
        """
        training_config: dict[str, Any] = {
            "learningRate": self.learning_rate,
            "epochs": self.epochs,
            "loraRank": self.lora_rank,
        }
        if self.batch_size is not None:
            training_config["batchSize"] = self.batch_size

        loss_config: dict[str, Any] = {
            "method": self.loss_method,
            "klBeta": self.beta,
        }

        params: dict[str, Any] = {
            "trainingConfig": training_config,
            "lossConfig": loss_config,
        }

        if self.wandb_project and self.wandb_api_key:
            params["wandbConfig"] = {
                "project": self.wandb_project,
                "apiKey": self.wandb_api_key,
            }
        return params


@dataclass
class JobStatus:
    """Status of a fine-tuning job."""

    name: str
    state: str
    create_time: str = ""
    update_time: str = ""
    completed_time: str = ""
    progress_percent: float = 0.0
    progress_epoch: float = 0.0
    tokens_processed: int = 0
    estimated_cost: str = ""
    output_model: str = ""
    error_message: str = ""
    estimated_finish_time: str = ""

    @property
    def is_terminal(self) -> bool:
        return self.state in TERMINAL_STATES

    @property
    def is_active(self) -> bool:
        return self.state in ACTIVE_STATES

    @property
    def is_success(self) -> bool:
        return self.state == JOB_STATE_COMPLETED

    @property
    def display_state(self) -> str:
        """Human-readable state label."""
        return {
            JOB_STATE_CREATING: "creating",
            JOB_STATE_PENDING: "queued",
            JOB_STATE_RUNNING: "running",
            JOB_STATE_COMPLETED: "succeeded",
            JOB_STATE_FAILED: "failed",
            JOB_STATE_CANCELLED: "cancelled",
            JOB_STATE_EARLY_STOPPED: "early-stopped",
        }.get(self.state, self.state.lower())


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------
class FireworksError(Exception):
    """Base error for Fireworks API operations."""

    def __init__(self, message: str, status_code: int | None = None, response: dict | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.response = response


class FireworksAuthError(FireworksError):
    """Authentication/authorization error (401/403)."""
    pass


class FireworksNotFoundError(FireworksError):
    """Resource not found (404)."""
    pass


class FireworksRateLimitError(FireworksError):
    """Rate limit exceeded (429)."""
    pass


# ---------------------------------------------------------------------------
# FireworksFineTuner
# ---------------------------------------------------------------------------
class FireworksFineTuner:
    """
    Client for Fireworks.ai supervised fine-tuning API.

    Manages the full fine-tuning lifecycle: dataset upload, job creation,
    status polling, and model evaluation.

    Args:
        account_id: Fireworks account identifier.
        api_key: API key (or set FIREWORKS_API_KEY env var).
        base_url: API base URL (default: https://api.fireworks.ai/v1).
        timeout: Request timeout in seconds (default: 120).
        max_retries: Max retries for transient errors (default: 3).
        retry_delay: Initial retry delay in seconds (default: 2.0).
    """

    def __init__(
        self,
        account_id: str | None = None,
        api_key: str | None = None,
        base_url: str = FIREWORKS_API_BASE,
        timeout: float = 120.0,
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ):
        self.api_key = api_key or os.environ.get("FIREWORKS_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Fireworks API key required. Set FIREWORKS_API_KEY env var "
                "or pass api_key parameter."
            )

        self.account_id = account_id or os.environ.get("FIREWORKS_ACCOUNT_ID")
        if not self.account_id:
            raise ValueError(
                "Fireworks account ID required. Set FIREWORKS_ACCOUNT_ID env var "
                "or pass account_id parameter."
            )

        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        })

    # ------------------------------------------------------------------
    # Dataset operations
    # ------------------------------------------------------------------

    def upload_file(self, file_path: str | Path) -> str:
        """
        Upload a JSONL dataset file to Fireworks.

        The file is uploaded via a two-step process:
        1. Create a dataset entry
        2. Upload the file content

        Args:
            file_path: Path to the JSONL file.

        Returns:
            Dataset ID for use in fine-tuning jobs.

        Raises:
            FileNotFoundError: If the file doesn't exist.
            FireworksError: On API errors.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")
        if not path.suffix == ".jsonl":
            logger.warning("Expected .jsonl file, got: %s", path.suffix)

        # Validate JSONL format
        line_count = self._validate_jsonl(path)
        logger.info("Validated %d lines in %s", line_count, path.name)

        # Create dataset entry
        dataset_id = path.stem
        url = f"{self.base_url}/accounts/{self.account_id}/datasets"

        # Upload with multipart form
        with open(path, "rb") as f:
            response = self._request(
                "POST",
                url,
                files={"file": (path.name, f, "application/jsonl")},
                params={"datasetId": dataset_id},
                content_type=None,  # Let requests set multipart boundary
            )

        created_id = response.get("name", dataset_id)
        logger.info("Dataset uploaded: %s (%d samples)", created_id, line_count)
        return created_id

    def list_datasets(self, page_size: int = 50) -> list[dict[str, Any]]:
        """List all datasets in the account."""
        url = f"{self.base_url}/accounts/{self.account_id}/datasets"
        response = self._request("GET", url, params={"pageSize": page_size})
        return response.get("datasets", [])

    # ------------------------------------------------------------------
    # Fine-tuning job operations
    # ------------------------------------------------------------------

    def create_sft_job(
        self,
        dataset: str,
        base_model: str,
        output_model: str | None = None,
        config: SftJobConfig | None = None,
        display_name: str | None = None,
    ) -> dict[str, Any]:
        """
        Create a supervised fine-tuning job.

        Args:
            dataset: Dataset ID (from upload_file).
            base_model: Base model ID (e.g., from MODELS dict).
            output_model: Output model name (defaults to auto-generated).
            config: Training configuration.
            display_name: Human-readable job name.

        Returns:
            Job response dict with name, state, etc.

        Raises:
            FireworksError: On API errors.
        """
        cfg = config or SftJobConfig()
        url = f"{self.base_url}/accounts/{self.account_id}/supervisedFineTuningJobs"

        body: dict[str, Any] = {
            "dataset": dataset,
            "baseModel": base_model,
            **cfg.to_api_params(),
        }

        if output_model:
            body["outputModel"] = output_model
        if display_name:
            body["displayName"] = display_name

        response = self._request("POST", url, json=body)
        job_name = response.get("name", "unknown")
        logger.info(
            "Created SFT job: %s (base=%s, epochs=%d, lr=%.1e)",
            job_name,
            base_model,
            cfg.epochs,
            cfg.learning_rate,
        )
        return response

    def create_dpo_job(
        self,
        dataset: str,
        base_model: str,
        output_model: str | None = None,
        config: DpoJobConfig | None = None,
        display_name: str | None = None,
    ) -> dict[str, Any]:
        """
        Create a Direct Preference Optimization (DPO) job.

        DPO trains on preference pairs (chosen/rejected) to align the model.
        Typically uses an SFT-trained model as the base.

        Args:
            dataset: Dataset ID containing preference pairs (from upload_file).
            base_model: Base model ID — usually an SFT-trained model.
            output_model: Output model name (defaults to auto-generated).
            config: DPO training configuration (beta, epochs, etc.).
            display_name: Human-readable job name.

        Returns:
            Job response dict with name, state, etc.

        Raises:
            FireworksError: On API errors.
        """
        cfg = config or DpoJobConfig()
        url = f"{self.base_url}/accounts/{self.account_id}/dpoJobs"

        # DPO uses trainingConfig wrapper with baseModel inside
        api_params = cfg.to_api_params()
        api_params["trainingConfig"]["baseModel"] = base_model

        body: dict[str, Any] = {
            "dataset": dataset,
            **api_params,
        }

        if output_model:
            body["trainingConfig"]["outputModel"] = output_model
        if display_name:
            body["displayName"] = display_name

        response = self._request("POST", url, json=body)
        job_name = response.get("name", "unknown")
        logger.info(
            "Created DPO job: %s (base=%s, epochs=%d, lr=%.1e, beta=%.2f)",
            job_name,
            base_model,
            cfg.epochs,
            cfg.learning_rate,
            cfg.beta,
        )
        return response

    def get_job(self, job_name: str) -> JobStatus:
        """
        Get the current status of a fine-tuning job.

        Args:
            job_name: Job identifier (from create_sft_job response).

        Returns:
            JobStatus with current state and progress.
        """
        url = f"{self.base_url}/{job_name}"
        response = self._request("GET", url)
        return self._parse_job_status(response)

    def list_jobs(
        self,
        page_size: int = 50,
        filter_str: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        List fine-tuning jobs with optional filtering.

        Args:
            page_size: Max results per page (default: 50, max: 200).
            filter_str: AIP-160 filter string.

        Returns:
            List of job dicts.
        """
        url = f"{self.base_url}/accounts/{self.account_id}/supervisedFineTuningJobs"
        params: dict[str, Any] = {"pageSize": min(page_size, 200)}
        if filter_str:
            params["filter"] = filter_str

        response = self._request("GET", url, params=params)
        return response.get("supervisedFineTuningJobs", [])

    def cancel_job(self, job_name: str) -> dict[str, Any]:
        """
        Cancel a running fine-tuning job.

        Args:
            job_name: Job identifier.

        Returns:
            Updated job response.
        """
        url = f"{self.base_url}/{job_name}:cancel"
        response = self._request("POST", url)
        logger.info("Cancelled job: %s", job_name)
        return response

    @staticmethod
    def _estimate_remaining(
        elapsed_secs: float,
        progress_percent: float,
    ) -> str | None:
        """Estimate time remaining based on elapsed time and progress.

        Returns a human-readable string like '3m 20s' or None if not enough
        data to estimate.
        """
        if progress_percent <= 0 or progress_percent >= 100:
            return None
        total_estimated = elapsed_secs / (progress_percent / 100.0)
        remaining = total_estimated - elapsed_secs
        if remaining < 0:
            remaining = 0
        mins, secs = divmod(int(remaining), 60)
        hours, mins = divmod(mins, 60)
        if hours > 0:
            return f"{hours}h {mins}m {secs}s"
        if mins > 0:
            return f"{mins}m {secs}s"
        return f"{secs}s"

    @staticmethod
    def _format_elapsed(secs: float) -> str:
        """Format seconds as human-readable elapsed time."""
        mins, s = divmod(int(secs), 60)
        hours, mins = divmod(mins, 60)
        if hours > 0:
            return f"{hours}h {mins}m {s}s"
        if mins > 0:
            return f"{mins}m {s}s"
        return f"{s}s"

    def wait_for_job(
        self,
        job_name: str,
        poll_interval: float = 30.0,
        timeout: float | None = None,
        progress_callback: Any = None,
    ) -> JobStatus:
        """
        Poll a job until it reaches a terminal state with progress reporting.

        Displays status updates including elapsed time and estimated finish
        time. The estimated finish time comes from the API when available,
        or is calculated from progress rate.

        Args:
            job_name: Job identifier (e.g. 'accounts/acme/supervisedFineTuningJobs/abc123').
            poll_interval: Seconds between status checks (default: 30).
            timeout: Max wait time in seconds (None = unlimited).
            progress_callback: Optional callable(JobStatus) invoked on each poll.

        Returns:
            Final JobStatus.

        Raises:
            TimeoutError: If timeout exceeded.
            FireworksError: If job fails.
        """
        start = time.monotonic()
        last_state = ""

        while True:
            status = self.get_job(job_name)
            elapsed = time.monotonic() - start
            elapsed_str = self._format_elapsed(elapsed)

            # Build ETA string: prefer API estimate, fall back to calculation
            eta_str = ""
            if status.estimated_finish_time:
                eta_str = f", ETA: {status.estimated_finish_time}"
            else:
                remaining = self._estimate_remaining(elapsed, status.progress_percent)
                if remaining:
                    eta_str = f", ~{remaining} remaining"

            # Log on state change or progress milestone
            if status.state != last_state or status.state == JOB_STATE_RUNNING:
                logger.info(
                    "Job %s: %s (%.1f%%, epoch %.1f, elapsed %s%s)",
                    job_name,
                    status.display_state,
                    status.progress_percent,
                    status.progress_epoch,
                    elapsed_str,
                    eta_str,
                )
                last_state = status.state

            if progress_callback:
                progress_callback(status)

            if status.is_terminal:
                logger.info(
                    "Job %s finished: %s (total time: %s)",
                    job_name,
                    status.display_state,
                    elapsed_str,
                )
                if status.state == JOB_STATE_FAILED:
                    raise FireworksError(
                        f"Job {job_name} failed: {status.error_message}",
                    )
                return status

            if timeout is not None:
                if elapsed >= timeout:
                    raise TimeoutError(
                        f"Job {job_name} did not complete within {timeout}s "
                        f"(last state: {status.state})"
                    )

            time.sleep(poll_interval)

    def wait_for_completion(
        self,
        job_name: str,
        poll_interval: float = 30.0,
        timeout: float | None = None,
        progress_callback: Any = None,
    ) -> JobStatus:
        """Alias for :meth:`wait_for_job` (backwards compatible)."""
        return self.wait_for_job(
            job_name,
            poll_interval=poll_interval,
            timeout=timeout,
            progress_callback=progress_callback,
        )

    # ------------------------------------------------------------------
    # Model evaluation
    # ------------------------------------------------------------------

    def test_model(
        self,
        model_id: str,
        prompts: list[str],
        system_prompt: str = "You are a helpful assistant.",
        max_tokens: int = 512,
        temperature: float = 0.1,
    ) -> list[dict[str, Any]]:
        """
        Test a fine-tuned model with sample prompts via chat completions.

        Args:
            model_id: Model to test (e.g., output_model from fine-tuning).
            prompts: List of test prompts.
            system_prompt: System message.
            max_tokens: Max tokens per response.
            temperature: Sampling temperature.

        Returns:
            List of dicts with prompt, response, tokens, and finish_reason.
        """
        results = []
        url = f"{self.base_url}/chat/completions"

        for prompt in prompts:
            body = {
                "model": model_id,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": max_tokens,
                "temperature": temperature,
            }

            try:
                response = self._request("POST", url, json=body)
                choice = response.get("choices", [{}])[0]
                usage = response.get("usage", {})

                results.append({
                    "prompt": prompt,
                    "response": choice.get("message", {}).get("content", ""),
                    "finish_reason": choice.get("finish_reason", ""),
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                })
            except FireworksError as e:
                logger.warning("Test prompt failed: %s", e)
                results.append({
                    "prompt": prompt,
                    "response": "",
                    "error": str(e),
                })

        return results

    def compare_models(
        self,
        finetuned_model: str,
        base_model: str,
        prompts: list[dict[str, str]] | None = None,
        system_prompt: str = RE_SYSTEM_PROMPT,
        max_tokens: int = 512,
        temperature: float = 0.1,
    ) -> dict[str, Any]:
        """
        Compare a fine-tuned model against its base model on test prompts.

        Runs the same set of prompts through both models and computes quality
        metrics based on expected keyword coverage.

        Args:
            finetuned_model: Fine-tuned model ID.
            base_model: Base model ID for comparison.
            prompts: List of dicts with "prompt" and optional "expected_keywords".
                     Defaults to DEFAULT_RE_TEST_PROMPTS.
            system_prompt: System message for both models.
            max_tokens: Max tokens per response.
            temperature: Sampling temperature.

        Returns:
            Dict with per-prompt comparisons and aggregate scores.
        """
        if prompts is None:
            prompts = DEFAULT_RE_TEST_PROMPTS

        prompt_texts = [p["prompt"] for p in prompts]

        logger.info(
            "Testing %d prompts: finetuned=%s base=%s",
            len(prompt_texts),
            finetuned_model,
            base_model,
        )

        finetuned_results = self.test_model(
            finetuned_model,
            prompt_texts,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        base_results = self.test_model(
            base_model,
            prompt_texts,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        comparisons = []
        ft_total_score = 0.0
        base_total_score = 0.0

        for i, prompt_info in enumerate(prompts):
            expected = prompt_info.get("expected_keywords", [])
            ft_resp = finetuned_results[i] if i < len(finetuned_results) else {}
            base_resp = base_results[i] if i < len(base_results) else {}

            ft_text = ft_resp.get("response", "").lower()
            base_text = base_resp.get("response", "").lower()

            # Compute keyword hit rate
            ft_hits = sum(
                1 for kw in expected if kw.lower() in ft_text
            ) if expected else 0
            base_hits = sum(
                1 for kw in expected if kw.lower() in base_text
            ) if expected else 0

            ft_score = ft_hits / len(expected) if expected else 0.0
            base_score = base_hits / len(expected) if expected else 0.0
            ft_total_score += ft_score
            base_total_score += base_score

            comparisons.append({
                "prompt": prompt_info["prompt"],
                "expected_keywords": expected,
                "finetuned": {
                    "response": ft_resp.get("response", ""),
                    "keyword_hits": ft_hits,
                    "keyword_score": ft_score,
                    "tokens": ft_resp.get("completion_tokens", 0),
                    "error": ft_resp.get("error"),
                },
                "base": {
                    "response": base_resp.get("response", ""),
                    "keyword_hits": base_hits,
                    "keyword_score": base_score,
                    "tokens": base_resp.get("completion_tokens", 0),
                    "error": base_resp.get("error"),
                },
                "winner": (
                    "finetuned" if ft_score > base_score
                    else "base" if base_score > ft_score
                    else "tie"
                ),
            })

        n = len(prompts) or 1
        ft_avg = ft_total_score / n
        base_avg = base_total_score / n

        return {
            "finetuned_model": finetuned_model,
            "base_model": base_model,
            "num_prompts": len(prompts),
            "comparisons": comparisons,
            "summary": {
                "finetuned_avg_score": round(ft_avg, 3),
                "base_avg_score": round(base_avg, 3),
                "improvement": round(ft_avg - base_avg, 3),
                "finetuned_wins": sum(
                    1 for c in comparisons if c["winner"] == "finetuned"
                ),
                "base_wins": sum(
                    1 for c in comparisons if c["winner"] == "base"
                ),
                "ties": sum(
                    1 for c in comparisons if c["winner"] == "tie"
                ),
            },
        }

    @staticmethod
    def format_comparison(result: dict[str, Any]) -> str:
        """Format compare_models() output as a human-readable report."""
        lines = [
            "",
            "=" * 72,
            "  Model Comparison Report",
            "=" * 72,
            f"  Fine-tuned : {result['finetuned_model']}",
            f"  Base       : {result['base_model']}",
            f"  Prompts    : {result['num_prompts']}",
            "-" * 72,
        ]

        for i, comp in enumerate(result["comparisons"], 1):
            prompt = comp["prompt"]
            if len(prompt) > 60:
                prompt = prompt[:57] + "..."
            ft = comp["finetuned"]
            base = comp["base"]
            winner = comp["winner"].upper()

            lines.append(f"\n  [{i}] {prompt}")
            lines.append(f"      Keywords: {', '.join(comp['expected_keywords'])}")
            lines.append(
                f"      Fine-tuned: {ft['keyword_hits']}/{len(comp['expected_keywords'])} "
                f"hits ({ft['keyword_score']:.0%}) | {ft['tokens']} tokens"
            )
            if ft["error"]:
                lines.append(f"        ERROR: {ft['error']}")
            lines.append(
                f"      Base:       {base['keyword_hits']}/{len(comp['expected_keywords'])} "
                f"hits ({base['keyword_score']:.0%}) | {base['tokens']} tokens"
            )
            if base["error"]:
                lines.append(f"        ERROR: {base['error']}")
            lines.append(f"      Winner: {winner}")

        summary = result["summary"]
        lines.extend([
            "",
            "-" * 72,
            "  SUMMARY",
            "-" * 72,
            f"  Fine-tuned avg score : {summary['finetuned_avg_score']:.1%}",
            f"  Base avg score       : {summary['base_avg_score']:.1%}",
            f"  Improvement          : {summary['improvement']:+.1%}",
            f"  Wins: Fine-tuned={summary['finetuned_wins']}  "
            f"Base={summary['base_wins']}  Tie={summary['ties']}",
            "=" * 72,
            "",
        ])

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def get_model_config(self, model_key: str) -> dict[str, Any]:
        """
        Get default configuration for a model size.

        Args:
            model_key: One of "slm", "small", "medium".

        Returns:
            Model config dict with model_id, defaults, etc.

        Raises:
            KeyError: If model_key not in MODELS.
        """
        if model_key not in MODELS:
            raise KeyError(
                f"Unknown model key '{model_key}'. "
                f"Available: {list(MODELS.keys())}"
            )
        return MODELS[model_key].copy()

    def create_sft_config_for_model(self, model_key: str) -> SftJobConfig:
        """
        Create an SftJobConfig with defaults for a model size.

        Args:
            model_key: One of "slm", "small", "medium".

        Returns:
            SftJobConfig with model-appropriate defaults.
        """
        m = self.get_model_config(model_key)
        return SftJobConfig(
            epochs=m["default_epochs"],
            learning_rate=m["default_learning_rate"],
            lora_rank=m["default_lora_rank"],
            max_context_length=m["default_max_context"],
        )

    def create_dpo_config_for_model(
        self,
        model_key: str,
        beta: float = 0.1,
    ) -> DpoJobConfig:
        """
        Create a DpoJobConfig with defaults for a model size.

        Args:
            model_key: One of "slm", "small", "medium".
            beta: KL divergence penalty coefficient (default: 0.1).

        Returns:
            DpoJobConfig with model-appropriate defaults.
        """
        m = self.get_model_config(model_key)
        return DpoJobConfig(
            epochs=m["dpo_epochs"],
            learning_rate=m["dpo_learning_rate"],
            lora_rank=m["dpo_lora_rank"],
            max_context_length=m["dpo_max_context"],
            beta=beta,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _request(
        self,
        method: str,
        url: str,
        json: dict | None = None,
        params: dict | None = None,
        files: dict | None = None,
        content_type: str | None = "application/json",
    ) -> dict[str, Any]:
        """Execute an API request with retry logic."""
        last_error: Exception | None = None
        delay = self.retry_delay

        for attempt in range(1, self.max_retries + 1):
            try:
                headers = {}
                if content_type is None:
                    # Remove Content-Type for multipart uploads
                    headers = {k: v for k, v in self._session.headers.items()
                               if k.lower() != "content-type"}
                    resp = requests.request(
                        method,
                        url,
                        headers=headers,
                        json=json,
                        params=params,
                        files=files,
                        timeout=self.timeout,
                    )
                else:
                    resp = self._session.request(
                        method,
                        url,
                        json=json,
                        params=params,
                        timeout=self.timeout,
                    )

                if resp.status_code == 401 or resp.status_code == 403:
                    raise FireworksAuthError(
                        f"Authentication failed: {resp.status_code}",
                        status_code=resp.status_code,
                        response=self._safe_json(resp),
                    )

                if resp.status_code == 404:
                    raise FireworksNotFoundError(
                        f"Resource not found: {url}",
                        status_code=404,
                        response=self._safe_json(resp),
                    )

                if resp.status_code == 429:
                    raise FireworksRateLimitError(
                        "Rate limit exceeded",
                        status_code=429,
                        response=self._safe_json(resp),
                    )

                if resp.status_code >= 400:
                    body = self._safe_json(resp)
                    msg = body.get("error", {}).get("message", resp.text[:200]) if isinstance(body, dict) else resp.text[:200]
                    raise FireworksError(
                        f"API error {resp.status_code}: {msg}",
                        status_code=resp.status_code,
                        response=body,
                    )

                # Success — return JSON or empty dict for 204
                if resp.status_code == 204 or not resp.content:
                    return {}
                return resp.json()

            except (requests.ConnectionError, requests.Timeout) as e:
                last_error = e
                logger.warning(
                    "Attempt %d/%d failed (connection): %s",
                    attempt,
                    self.max_retries,
                    e,
                )
            except FireworksRateLimitError as e:
                last_error = e
                logger.warning(
                    "Attempt %d/%d rate-limited, waiting %.1fs",
                    attempt,
                    self.max_retries,
                    delay,
                )
            except (FireworksAuthError, FireworksNotFoundError):
                # Don't retry auth or 404 errors
                raise
            except FireworksError as e:
                if e.status_code and e.status_code < 500:
                    # Don't retry client errors
                    raise
                last_error = e
                logger.warning(
                    "Attempt %d/%d server error: %s",
                    attempt,
                    self.max_retries,
                    e,
                )

            if attempt < self.max_retries:
                time.sleep(delay)
                delay *= 2

        raise FireworksError(
            f"Request failed after {self.max_retries} retries: {last_error}"
        )

    @staticmethod
    def _validate_jsonl(path: Path) -> int:
        """Validate a JSONL file and return line count."""
        count = 0
        with open(path) as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    json.loads(line)
                    count += 1
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON at line {i}: {e}")
        if count == 0:
            raise ValueError(f"Empty JSONL file: {path}")
        return count

    @staticmethod
    def _parse_job_status(data: dict[str, Any]) -> JobStatus:
        """Parse API response into JobStatus."""
        progress = data.get("jobProgress", {})
        status_info = data.get("status", {})

        return JobStatus(
            name=data.get("name", ""),
            state=data.get("state", "UNKNOWN"),
            create_time=data.get("createTime", ""),
            update_time=data.get("updateTime", ""),
            completed_time=data.get("completedTime", ""),
            progress_percent=progress.get("percent", 0.0),
            progress_epoch=progress.get("epoch", 0.0),
            tokens_processed=progress.get("tokensProcessed", 0),
            estimated_cost=data.get("estimatedCost", ""),
            output_model=data.get("outputModel", ""),
            error_message=status_info.get("message", ""),
            estimated_finish_time=data.get("estimatedCompletionTime", ""),
        )

    @staticmethod
    def _safe_json(resp: requests.Response) -> dict | str:
        """Safely parse response JSON, returning text on failure."""
        try:
            return resp.json()
        except (json.JSONDecodeError, ValueError):
            return resp.text


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main():
    """CLI for quick fine-tuning operations."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Fireworks.ai fine-tuning client for Ucotron"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # -- list-models --
    list_models = sub.add_parser("list-models", help="Show available model configs")

    # -- upload --
    upload = sub.add_parser("upload", help="Upload a JSONL dataset")
    upload.add_argument("file", help="Path to JSONL file")
    upload.add_argument("--account-id", required=True, help="Fireworks account ID")

    # -- create-job (SFT) --
    create = sub.add_parser("create-job", help="Create an SFT fine-tuning job")
    create.add_argument("--account-id", required=True, help="Fireworks account ID")
    create.add_argument("--dataset", required=True, help="Dataset ID")
    create.add_argument(
        "--model-size",
        choices=["slm", "small", "medium"],
        default="slm",
        help="Model size (default: slm)",
    )
    create.add_argument("--output-model", help="Output model name")
    create.add_argument("--epochs", type=int, help="Override default epochs")
    create.add_argument("--learning-rate", type=float, help="Override default LR")

    # -- create-dpo-job --
    create_dpo = sub.add_parser("create-dpo-job", help="Create a DPO alignment job")
    create_dpo.add_argument("--account-id", required=True, help="Fireworks account ID")
    create_dpo.add_argument("--dataset", required=True, help="Preference dataset ID")
    create_dpo.add_argument("--base-model", required=True, help="SFT-trained model ID")
    create_dpo.add_argument(
        "--model-size",
        choices=["slm", "small", "medium"],
        default="slm",
        help="Model size for default hyperparams (default: slm)",
    )
    create_dpo.add_argument("--output-model", help="Output model name")
    create_dpo.add_argument("--beta", type=float, default=0.1, help="KL beta (default: 0.1)")
    create_dpo.add_argument("--epochs", type=int, help="Override default epochs")
    create_dpo.add_argument("--learning-rate", type=float, help="Override default LR")

    # -- status --
    status = sub.add_parser("status", help="Check job status")
    status.add_argument("job_name", help="Job identifier")
    status.add_argument("--account-id", required=True, help="Fireworks account ID")

    # -- wait --
    wait = sub.add_parser("wait", help="Wait for job completion with progress reporting")
    wait.add_argument("job_name", help="Job identifier")
    wait.add_argument("--account-id", required=True, help="Fireworks account ID")
    wait.add_argument(
        "--poll-interval", type=float, default=30.0,
        help="Seconds between status checks (default: 30)",
    )
    wait.add_argument(
        "--timeout", type=float, default=None,
        help="Max wait time in seconds (default: unlimited)",
    )

    # -- list-jobs --
    list_jobs = sub.add_parser("list-jobs", help="List fine-tuning jobs")
    list_jobs.add_argument("--account-id", required=True, help="Fireworks account ID")
    list_jobs.add_argument("--limit", type=int, default=10, help="Max results")

    # -- test-model --
    test_cmd = sub.add_parser(
        "test-model",
        help="Test a fine-tuned model and optionally compare with base model",
    )
    test_cmd.add_argument("model_id", help="Fine-tuned model ID to test")
    test_cmd.add_argument("--account-id", required=True, help="Fireworks account ID")
    test_cmd.add_argument(
        "--compare-base",
        metavar="BASE_MODEL",
        help="Base model ID for side-by-side comparison",
    )
    test_cmd.add_argument(
        "--model-size",
        choices=["slm", "small", "medium"],
        help="Auto-select base model for comparison from model registry",
    )
    test_cmd.add_argument(
        "--prompts",
        nargs="+",
        help="Custom test prompts (default: built-in RE prompts)",
    )
    test_cmd.add_argument(
        "--system-prompt",
        default=RE_SYSTEM_PROMPT,
        help="System prompt for the model",
    )
    test_cmd.add_argument(
        "--max-tokens", type=int, default=512, help="Max tokens per response",
    )
    test_cmd.add_argument(
        "--temperature", type=float, default=0.1, help="Sampling temperature",
    )
    test_cmd.add_argument(
        "--json", action="store_true", dest="output_json",
        help="Output results as JSON instead of formatted report",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if args.command == "list-models":
        print("\nAvailable models for fine-tuning:\n")
        for key, model in MODELS.items():
            print(f"  {key:8s}  {model['display_name']}")
            print(f"           ID: {model['model_id']}")
            print(f"           Params: {model['params']}")
            print(f"           Use: {model['use_case']}")
            print(f"           Defaults: epochs={model['default_epochs']}, "
                  f"lora_rank={model['default_lora_rank']}, "
                  f"lr={model['default_learning_rate']}")
            print()
        return

    tuner = FireworksFineTuner(account_id=args.account_id)

    if args.command == "upload":
        dataset_id = tuner.upload_file(args.file)
        print(f"Dataset uploaded: {dataset_id}")

    elif args.command == "create-job":
        model_cfg = tuner.get_model_config(args.model_size)
        config = tuner.create_sft_config_for_model(args.model_size)

        if args.epochs:
            config.epochs = args.epochs
        if args.learning_rate:
            config.learning_rate = args.learning_rate

        job = tuner.create_sft_job(
            dataset=args.dataset,
            base_model=model_cfg["model_id"],
            output_model=args.output_model,
            config=config,
        )
        print(f"Job created: {job.get('name', 'unknown')}")
        print(f"State: {job.get('state', 'unknown')}")

    elif args.command == "create-dpo-job":
        dpo_config = tuner.create_dpo_config_for_model(args.model_size, beta=args.beta)

        if args.epochs:
            dpo_config.epochs = args.epochs
        if args.learning_rate:
            dpo_config.learning_rate = args.learning_rate

        job = tuner.create_dpo_job(
            dataset=args.dataset,
            base_model=args.base_model,
            output_model=args.output_model,
            config=dpo_config,
        )
        print(f"DPO job created: {job.get('name', 'unknown')}")
        print(f"State: {job.get('state', 'unknown')}")
        print(f"Beta: {dpo_config.beta}")

    elif args.command == "status":
        status = tuner.get_job(args.job_name)
        print(f"Job: {status.name}")
        print(f"State: {status.display_state}")
        print(f"Progress: {status.progress_percent:.1f}%")
        print(f"Epoch: {status.progress_epoch:.1f}")
        if status.tokens_processed:
            print(f"Tokens processed: {status.tokens_processed:,}")
        if status.estimated_finish_time:
            print(f"Est. finish: {status.estimated_finish_time}")
        if status.estimated_cost:
            print(f"Est. cost: {status.estimated_cost}")
        if status.output_model:
            print(f"Output model: {status.output_model}")
        if status.error_message:
            print(f"Error: {status.error_message}")

    elif args.command == "wait":
        print(f"Waiting for job: {args.job_name}")
        print(f"Poll interval: {args.poll_interval}s")
        if args.timeout:
            print(f"Timeout: {args.timeout}s")
        try:
            final = tuner.wait_for_job(
                args.job_name,
                poll_interval=args.poll_interval,
                timeout=args.timeout,
            )
            print(f"\nJob completed: {final.display_state}")
            if final.output_model:
                print(f"Output model: {final.output_model}")
            if final.estimated_cost:
                print(f"Est. cost: {final.estimated_cost}")
        except TimeoutError as e:
            print(f"\nTimeout: {e}")
            raise SystemExit(1)
        except FireworksError as e:
            print(f"\nJob failed: {e}")
            raise SystemExit(1)

    elif args.command == "list-jobs":
        jobs = tuner.list_jobs(page_size=args.limit)
        if not jobs:
            print("No fine-tuning jobs found.")
            return
        for job in jobs:
            name = job.get("name", "?")
            state = job.get("state", "?")
            base = job.get("baseModel", "?")
            print(f"  {name}  state={state}  base={base}")

    elif args.command == "test-model":
        # Determine base model for comparison
        base_model = args.compare_base
        if not base_model and args.model_size:
            base_model = MODELS[args.model_size]["model_id"]

        # Build prompts list
        if args.prompts:
            prompt_dicts = [{"prompt": p, "expected_keywords": []} for p in args.prompts]
        else:
            prompt_dicts = None  # Uses DEFAULT_RE_TEST_PROMPTS

        if base_model:
            # Full comparison mode
            result = tuner.compare_models(
                finetuned_model=args.model_id,
                base_model=base_model,
                prompts=prompt_dicts,
                system_prompt=args.system_prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
            )
            if args.output_json:
                print(json.dumps(result, indent=2))
            else:
                print(FireworksFineTuner.format_comparison(result))
        else:
            # Simple test mode (no comparison)
            test_prompts = (
                args.prompts
                if args.prompts
                else [p["prompt"] for p in DEFAULT_RE_TEST_PROMPTS]
            )
            results = tuner.test_model(
                model_id=args.model_id,
                prompts=test_prompts,
                system_prompt=args.system_prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
            )
            if args.output_json:
                print(json.dumps(results, indent=2))
            else:
                print(f"\nModel: {args.model_id}")
                print(f"Prompts: {len(results)}")
                print("-" * 60)
                for i, r in enumerate(results, 1):
                    prompt = r["prompt"]
                    if len(prompt) > 50:
                        prompt = prompt[:47] + "..."
                    print(f"\n  [{i}] {prompt}")
                    if r.get("error"):
                        print(f"      ERROR: {r['error']}")
                    else:
                        resp = r["response"]
                        if len(resp) > 200:
                            resp = resp[:197] + "..."
                        print(f"      Response: {resp}")
                        print(
                            f"      Tokens: prompt={r.get('prompt_tokens', 0)} "
                            f"completion={r.get('completion_tokens', 0)}"
                        )
                print()


if __name__ == "__main__":
    main()
