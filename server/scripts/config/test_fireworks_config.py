#!/usr/bin/env python3
"""Tests for Fireworks configuration loading and validation."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest
import yaml

from scripts.config import (
    load_fireworks_config,
    validate_fireworks_config,
    get_api_key,
    get_account_id,
    get_training_config,
    get_generation_config,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def config_path() -> Path:
    """Path to the real fireworks_config.yaml."""
    return Path(__file__).parent / "fireworks_config.yaml"


@pytest.fixture
def config(config_path: Path) -> dict:
    """Load the real config file."""
    return load_fireworks_config(config_path)


def _write_yaml(data: dict, path: Path) -> None:
    with open(path, "w") as f:
        yaml.dump(data, f)


def _minimal_valid_config() -> dict:
    """Return a minimal valid config for testing."""
    return {
        "api": {
            "fine_tuning_base_url": "https://api.fireworks.ai/v1",
            "inference_base_url": "https://api.fireworks.ai/inference/v1",
            "timeout": 60.0,
            "max_retries": 3,
            "retry_delay": 1.0,
        },
        "models": {
            "generation": {"model_id": "test-model"},
            "fine_tuning": {
                "slm": {"model_id": "slm-model"},
                "small": {"model_id": "small-model"},
                "medium": {"model_id": "medium-model"},
            },
        },
        "training": {
            "sft": {
                "slm": {"epochs": 3, "learning_rate": 2e-4, "lora_rank": 8, "max_context_length": 2048},
                "small": {"epochs": 3, "learning_rate": 2e-4, "lora_rank": 16, "max_context_length": 2048},
                "medium": {"epochs": 2, "learning_rate": 1e-4, "lora_rank": 16, "max_context_length": 4096},
            },
            "dpo": {
                "slm": {"epochs": 2, "learning_rate": 5e-5, "lora_rank": 8, "max_context_length": 2048},
                "small": {"epochs": 2, "learning_rate": 5e-5, "lora_rank": 16, "max_context_length": 2048},
                "medium": {"epochs": 1, "learning_rate": 3e-5, "lora_rank": 16, "max_context_length": 4096},
            },
        },
        "generation": {
            "defaults": {"max_tokens": 2048, "temperature": 0.7, "top_p": 0.9},
        },
    }


# ---------------------------------------------------------------------------
# Test: Config file loads successfully
# ---------------------------------------------------------------------------

class TestLoadConfig:
    def test_loads_real_config_file(self, config_path: Path):
        config = load_fireworks_config(config_path)
        assert isinstance(config, dict)
        assert "api" in config
        assert "models" in config
        assert "training" in config
        assert "generation" in config

    def test_loads_default_path(self):
        config = load_fireworks_config()
        assert isinstance(config, dict)

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_fireworks_config("/nonexistent/path.yaml")

    def test_loads_from_string_path(self, config_path: Path):
        config = load_fireworks_config(str(config_path))
        assert isinstance(config, dict)


# ---------------------------------------------------------------------------
# Test: API section
# ---------------------------------------------------------------------------

class TestApiSection:
    def test_api_urls(self, config: dict):
        assert config["api"]["fine_tuning_base_url"] == "https://api.fireworks.ai/v1"
        assert config["api"]["inference_base_url"] == "https://api.fireworks.ai/inference/v1"

    def test_api_timeout(self, config: dict):
        assert config["api"]["timeout"] == 120.0

    def test_api_retries(self, config: dict):
        assert config["api"]["max_retries"] == 3
        assert config["api"]["retry_delay"] == 2.0


# ---------------------------------------------------------------------------
# Test: Models section
# ---------------------------------------------------------------------------

class TestModelsSection:
    def test_generation_model(self, config: dict):
        gen = config["models"]["generation"]
        assert gen["model_id"] == "accounts/fireworks/models/glm-4-plus"
        assert "display_name" in gen

    def test_fine_tuning_tiers(self, config: dict):
        ft = config["models"]["fine_tuning"]
        assert "slm" in ft
        assert "small" in ft
        assert "medium" in ft

    def test_slm_model(self, config: dict):
        slm = config["models"]["fine_tuning"]["slm"]
        assert slm["model_id"] == "accounts/fireworks/models/qwen2p5-0.5b-instruct"
        assert slm["params"] == "0.5B"

    def test_medium_model(self, config: dict):
        med = config["models"]["fine_tuning"]["medium"]
        assert med["model_id"] == "accounts/fireworks/models/qwen2p5-7b-instruct"
        assert med["params"] == "7B"


# ---------------------------------------------------------------------------
# Test: Training hyperparameters
# ---------------------------------------------------------------------------

class TestTrainingSection:
    def test_sft_slm_defaults(self, config: dict):
        sft = config["training"]["sft"]["slm"]
        assert sft["epochs"] == 3
        assert sft["learning_rate"] == 2.0e-4
        assert sft["lora_rank"] == 8
        assert sft["max_context_length"] == 2048

    def test_sft_medium_lower_lr(self, config: dict):
        sft = config["training"]["sft"]["medium"]
        assert sft["learning_rate"] == 1.0e-4
        assert sft["epochs"] == 2
        assert sft["max_context_length"] == 4096

    def test_dpo_exists_for_all_tiers(self, config: dict):
        dpo = config["training"]["dpo"]
        for tier in ("slm", "small", "medium"):
            assert tier in dpo
            assert "epochs" in dpo[tier]
            assert "learning_rate" in dpo[tier]

    def test_dpo_lower_lr_than_sft(self, config: dict):
        for tier in ("slm", "small", "medium"):
            sft_lr = config["training"]["sft"][tier]["learning_rate"]
            dpo_lr = config["training"]["dpo"][tier]["learning_rate"]
            assert dpo_lr < sft_lr, f"DPO LR should be lower than SFT LR for {tier}"


# ---------------------------------------------------------------------------
# Test: Generation settings
# ---------------------------------------------------------------------------

class TestGenerationSection:
    def test_defaults(self, config: dict):
        defaults = config["generation"]["defaults"]
        assert defaults["max_tokens"] == 2048
        assert defaults["temperature"] == 0.7
        assert defaults["top_p"] == 0.9

    def test_task_target_samples(self, config: dict):
        tasks = config["generation"]["tasks"]
        assert tasks["relation_extraction"]["target_samples"] == 10000
        assert tasks["preference"]["target_samples"] == 5000
        assert tasks["contradiction"]["target_samples"] == 3000
        assert tasks["entity_resolution"]["target_samples"] == 2000

    def test_preference_chosen_rejected(self, config: dict):
        pref = config["generation"]["tasks"]["preference"]
        assert pref["chosen"]["temperature"] == 0.0
        assert pref["rejected"]["temperature"] == 0.7


# ---------------------------------------------------------------------------
# Test: Validation
# ---------------------------------------------------------------------------

class TestValidation:
    def test_valid_config_passes(self):
        validate_fireworks_config(_minimal_valid_config())

    def test_missing_api_section(self):
        cfg = _minimal_valid_config()
        del cfg["api"]
        with pytest.raises(ValueError, match="api"):
            validate_fireworks_config(cfg)

    def test_missing_models_section(self):
        cfg = _minimal_valid_config()
        del cfg["models"]
        with pytest.raises(ValueError, match="models"):
            validate_fireworks_config(cfg)

    def test_invalid_timeout(self):
        cfg = _minimal_valid_config()
        cfg["api"]["timeout"] = -1
        with pytest.raises(ValueError, match="timeout"):
            validate_fireworks_config(cfg)

    def test_invalid_max_retries(self):
        cfg = _minimal_valid_config()
        cfg["api"]["max_retries"] = -1
        with pytest.raises(ValueError, match="max_retries"):
            validate_fireworks_config(cfg)

    def test_missing_generation_model_id(self):
        cfg = _minimal_valid_config()
        cfg["models"]["generation"] = {"display_name": "no id"}
        with pytest.raises(ValueError, match="generation.model_id"):
            validate_fireworks_config(cfg)

    def test_missing_fine_tuning_tier(self):
        cfg = _minimal_valid_config()
        del cfg["models"]["fine_tuning"]["slm"]
        with pytest.raises(ValueError, match="slm.model_id"):
            validate_fireworks_config(cfg)

    def test_invalid_epochs(self):
        cfg = _minimal_valid_config()
        cfg["training"]["sft"]["slm"]["epochs"] = 0
        with pytest.raises(ValueError, match="epochs"):
            validate_fireworks_config(cfg)

    def test_invalid_learning_rate(self):
        cfg = _minimal_valid_config()
        cfg["training"]["sft"]["slm"]["learning_rate"] = -0.01
        with pytest.raises(ValueError, match="learning_rate"):
            validate_fireworks_config(cfg)

    def test_invalid_lora_rank(self):
        cfg = _minimal_valid_config()
        cfg["training"]["dpo"]["medium"]["lora_rank"] = 0
        with pytest.raises(ValueError, match="lora_rank"):
            validate_fireworks_config(cfg)

    def test_invalid_context_length(self):
        cfg = _minimal_valid_config()
        cfg["training"]["sft"]["small"]["max_context_length"] = 64
        with pytest.raises(ValueError, match="max_context_length"):
            validate_fireworks_config(cfg)

    def test_invalid_temperature(self):
        cfg = _minimal_valid_config()
        cfg["generation"]["defaults"]["temperature"] = 3.0
        with pytest.raises(ValueError, match="temperature"):
            validate_fireworks_config(cfg)

    def test_invalid_max_tokens(self):
        cfg = _minimal_valid_config()
        cfg["generation"]["defaults"]["max_tokens"] = 0
        with pytest.raises(ValueError, match="max_tokens"):
            validate_fireworks_config(cfg)


# ---------------------------------------------------------------------------
# Test: Helper functions
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_get_training_config(self, config: dict):
        sft_slm = get_training_config(config, "sft", "slm")
        assert sft_slm["epochs"] == 3
        assert sft_slm["learning_rate"] == 2.0e-4

    def test_get_training_config_dpo(self, config: dict):
        dpo_med = get_training_config(config, "dpo", "medium")
        assert dpo_med["epochs"] == 1
        assert dpo_med["lora_rank"] == 16

    def test_get_generation_config_defaults(self, config: dict):
        gen = get_generation_config(config)
        assert gen["max_tokens"] == 2048
        assert gen["temperature"] == 0.7

    def test_get_generation_config_with_task(self, config: dict):
        gen = get_generation_config(config, "relation_extraction")
        assert gen["target_samples"] == 10000
        assert gen["temperature"] == 0.7  # from task override
        assert gen["top_p"] == 0.9  # from defaults

    def test_get_generation_config_unknown_task(self, config: dict):
        gen = get_generation_config(config, "nonexistent_task")
        # Falls back to defaults only
        assert gen["max_tokens"] == 2048
        assert "target_samples" not in gen

    def test_get_api_key_missing(self, monkeypatch):
        monkeypatch.delenv("FIREWORKS_API_KEY", raising=False)
        with pytest.raises(EnvironmentError, match="FIREWORKS_API_KEY"):
            get_api_key()

    def test_get_api_key_present(self, monkeypatch):
        monkeypatch.setenv("FIREWORKS_API_KEY", "test-key-123")
        assert get_api_key() == "test-key-123"

    def test_get_account_id_missing(self, monkeypatch):
        monkeypatch.delenv("FIREWORKS_ACCOUNT_ID", raising=False)
        with pytest.raises(EnvironmentError, match="FIREWORKS_ACCOUNT_ID"):
            get_account_id()

    def test_get_account_id_present(self, monkeypatch):
        monkeypatch.setenv("FIREWORKS_ACCOUNT_ID", "my-account")
        assert get_account_id() == "my-account"


# ---------------------------------------------------------------------------
# Test: Polling and output sections
# ---------------------------------------------------------------------------

class TestOptionalSections:
    def test_polling_settings(self, config: dict):
        polling = config.get("polling", {})
        assert polling["interval"] == 30
        assert polling["verbose"] is True

    def test_tracking_settings(self, config: dict):
        tracking = config.get("tracking", {})
        assert tracking["enabled"] is False
        assert tracking["wandb_project"] == "ucotron-finetune"

    def test_output_settings(self, config: dict):
        output = config.get("output", {})
        assert output["data_dir"] == "data/training"
        assert output["model_dir"] == "models/fine_tuned"
        assert "{task}" in output["model_name_template"]
        assert "{model_tier}" in output["model_name_template"]
