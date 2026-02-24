"""
Configuration loader for Ucotron fine-tuning pipeline.

Loads and validates fireworks_config.yaml with environment variable overrides
for secrets (API keys are never stored in config files).

Usage:
    from scripts.config import load_fireworks_config
    config = load_fireworks_config()  # loads default path
    config = load_fireworks_config("path/to/fireworks_config.yaml")
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def _default_config_path() -> Path:
    """Return the default config file path relative to the scripts/ directory."""
    return Path(__file__).parent / "fireworks_config.yaml"


def load_fireworks_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load and validate Fireworks configuration from YAML file.

    Args:
        path: Path to fireworks_config.yaml. Defaults to scripts/config/fireworks_config.yaml.

    Returns:
        Validated configuration dictionary.

    Raises:
        FileNotFoundError: If config file does not exist.
        ValueError: If required fields are missing or invalid.
        ImportError: If PyYAML is not installed.
    """
    try:
        import yaml
    except ImportError:
        raise ImportError(
            "PyYAML is required to load config. Install with: pip install pyyaml"
        )

    config_path = Path(path) if path else _default_config_path()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError(f"Config file must contain a YAML mapping, got {type(config).__name__}")

    validate_fireworks_config(config)
    return config


def validate_fireworks_config(config: dict[str, Any]) -> None:
    """Validate required fields and value constraints in Fireworks config.

    Args:
        config: Parsed YAML configuration dictionary.

    Raises:
        ValueError: If validation fails.
    """
    errors: list[str] = []

    # --- API section ---
    api = config.get("api")
    if not isinstance(api, dict):
        errors.append("Missing or invalid 'api' section")
    else:
        if not api.get("fine_tuning_base_url"):
            errors.append("api.fine_tuning_base_url is required")
        if not api.get("inference_base_url"):
            errors.append("api.inference_base_url is required")

        timeout = api.get("timeout", 0)
        if not isinstance(timeout, (int, float)) or timeout <= 0:
            errors.append(f"api.timeout must be a positive number, got {timeout}")

        max_retries = api.get("max_retries", -1)
        if not isinstance(max_retries, int) or max_retries < 0:
            errors.append(f"api.max_retries must be a non-negative integer, got {max_retries}")

        retry_delay = api.get("retry_delay", 0)
        if not isinstance(retry_delay, (int, float)) or retry_delay <= 0:
            errors.append(f"api.retry_delay must be a positive number, got {retry_delay}")

    # --- Models section ---
    models = config.get("models")
    if not isinstance(models, dict):
        errors.append("Missing or invalid 'models' section")
    else:
        gen_model = models.get("generation")
        if not isinstance(gen_model, dict) or not gen_model.get("model_id"):
            errors.append("models.generation.model_id is required")

        ft_models = models.get("fine_tuning")
        if not isinstance(ft_models, dict):
            errors.append("models.fine_tuning section is required")
        else:
            for tier in ("slm", "small", "medium"):
                tier_cfg = ft_models.get(tier)
                if not isinstance(tier_cfg, dict) or not tier_cfg.get("model_id"):
                    errors.append(f"models.fine_tuning.{tier}.model_id is required")

    # --- Training section ---
    training = config.get("training")
    if not isinstance(training, dict):
        errors.append("Missing or invalid 'training' section")
    else:
        for job_type in ("sft", "dpo"):
            job_cfg = training.get(job_type)
            if not isinstance(job_cfg, dict):
                errors.append(f"training.{job_type} section is required")
                continue
            for tier in ("slm", "small", "medium"):
                tier_cfg = job_cfg.get(tier)
                if not isinstance(tier_cfg, dict):
                    errors.append(f"training.{job_type}.{tier} section is required")
                    continue
                epochs = tier_cfg.get("epochs", 0)
                if not isinstance(epochs, int) or epochs < 1:
                    errors.append(f"training.{job_type}.{tier}.epochs must be >= 1, got {epochs}")
                lr = tier_cfg.get("learning_rate", 0)
                if not isinstance(lr, (int, float)) or lr <= 0:
                    errors.append(f"training.{job_type}.{tier}.learning_rate must be > 0, got {lr}")
                lora = tier_cfg.get("lora_rank", 0)
                if not isinstance(lora, int) or lora < 1:
                    errors.append(f"training.{job_type}.{tier}.lora_rank must be >= 1, got {lora}")
                ctx = tier_cfg.get("max_context_length", 0)
                if not isinstance(ctx, int) or ctx < 128:
                    errors.append(f"training.{job_type}.{tier}.max_context_length must be >= 128, got {ctx}")

    # --- Generation section ---
    generation = config.get("generation")
    if not isinstance(generation, dict):
        errors.append("Missing or invalid 'generation' section")
    else:
        defaults = generation.get("defaults")
        if isinstance(defaults, dict):
            temp = defaults.get("temperature", -1)
            if not isinstance(temp, (int, float)) or temp < 0 or temp > 2:
                errors.append(f"generation.defaults.temperature must be in [0, 2], got {temp}")
            max_tok = defaults.get("max_tokens", 0)
            if not isinstance(max_tok, int) or max_tok < 1:
                errors.append(f"generation.defaults.max_tokens must be >= 1, got {max_tok}")

    if errors:
        raise ValueError(
            "Fireworks config validation failed:\n  - " + "\n  - ".join(errors)
        )


def get_api_key() -> str:
    """Get Fireworks API key from environment.

    Returns:
        API key string.

    Raises:
        EnvironmentError: If FIREWORKS_API_KEY is not set.
    """
    key = os.environ.get("FIREWORKS_API_KEY", "")
    if not key:
        raise EnvironmentError(
            "FIREWORKS_API_KEY environment variable is required. "
            "Get your key at https://fireworks.ai/account/api-keys"
        )
    return key


def get_account_id() -> str:
    """Get Fireworks account ID from environment.

    Returns:
        Account ID string.

    Raises:
        EnvironmentError: If FIREWORKS_ACCOUNT_ID is not set.
    """
    account_id = os.environ.get("FIREWORKS_ACCOUNT_ID", "")
    if not account_id:
        raise EnvironmentError(
            "FIREWORKS_ACCOUNT_ID environment variable is required for fine-tuning jobs. "
            "Find your account ID at https://fireworks.ai/account"
        )
    return account_id


def get_training_config(
    config: dict[str, Any],
    job_type: str,
    model_tier: str,
) -> dict[str, Any]:
    """Extract training hyperparameters for a specific job type and model tier.

    Args:
        config: Full Fireworks config dictionary.
        job_type: "sft" or "dpo".
        model_tier: "slm", "small", or "medium".

    Returns:
        Dict with epochs, learning_rate, lora_rank, max_context_length, batch_size.

    Raises:
        KeyError: If job_type or model_tier not found in config.
    """
    return config["training"][job_type][model_tier]


def get_generation_config(
    config: dict[str, Any],
    task: str | None = None,
) -> dict[str, Any]:
    """Get generation parameters, optionally merged with task-specific overrides.

    Args:
        config: Full Fireworks config dictionary.
        task: Optional task name (e.g., "relation_extraction", "preference", "contradiction").

    Returns:
        Dict with max_tokens, temperature, top_p, and any task-specific fields.
    """
    defaults = dict(config.get("generation", {}).get("defaults", {}))
    if task:
        task_cfg = config.get("generation", {}).get("tasks", {}).get(task, {})
        defaults.update(task_cfg)
    return defaults
