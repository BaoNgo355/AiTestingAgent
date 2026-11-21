"""Unit tests cho aitest.config (Giai đoạn 5 - AC-01, AC-10, AC-18, NFR-03)."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from aitest.config import AitestConfig, RubricWeights, load_config
from aitest.utils import compute_config_hash


def test_default_config_values_and_rubric_sum() -> None:
    """Kiểm tra cấu hình mặc định khớp với SRS §10.1."""
    cfg = AitestConfig()
    assert cfg.provider == "ollama"
    assert cfg.base_url == "http://localhost:11434"
    assert cfg.model == "qwen2.5-coder:7b"
    assert cfg.mutation.enabled is True
    assert cfg.mutation.tool == "auto"
    assert cfg.eval.threshold_pass == 8.0
    assert cfg.eval.correctness_cap_offset == 2.0
    assert abs(sum(cfg.eval.rubric.to_dict().values()) - 1.0) <= 1e-6


def test_config_priority_cli_over_env_over_yaml_over_default(tmp_path: Path) -> None:
    """Kiểm tra thứ tự ưu tiên cấu hình: CLI flags > AITEST_* env > aitest.yaml > defaults."""
    yaml_path = tmp_path / "aitest.yaml"
    yaml_path.write_text(
        "model: llama3:8b\n"
        "eval:\n"
        "  threshold_pass: 7.5\n"
        "llm:\n"
        "  temperature: 0.4\n",
        encoding="utf-8",
    )
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text(
        "OLLAMA_BASE_URL=http://10.0.0.5:11434/\n"
        "OLLAMA_API_KEY=secret-token-999\n",
        encoding="utf-8",
    )

    env = {
        "OLLAMA_BASE_URL": "http://10.0.0.5:11434/",
        "OLLAMA_API_KEY": "secret-token-999",
        "AITEST_LLM__TEMPERATURE": "0.15",
        "AITEST_AGENT__MAX_ITERS": "5",
    }
    cli_overrides = {
        "model": "qwen2.5-coder:7b",
    }

    cfg = load_config(
        config_path=yaml_path,
        cli_overrides=cli_overrides,
        env=env,
        dotenv_path=dotenv_path,
    )
    assert cfg.model == "qwen2.5-coder:7b"
    assert cfg.eval.threshold_pass == 7.5
    assert cfg.llm.temperature == 0.15
    assert cfg.agent.max_iters == 5
    assert cfg.base_url == "http://10.0.0.5:11434"
    assert cfg.ollama_api_key is not None
    assert cfg.ollama_api_key.get_secret_value() == "secret-token-999"

    # Đảm bảo khóa bí mật không bao giờ xuất hiện trong safe_dump hoặc config_hash (NFR-03)
    dumped = cfg.safe_dump()
    assert "ollama_api_key" not in dumped
    assert "secret-token-999" not in str(dumped)
    assert compute_config_hash(cfg).startswith("sha256:")


def test_invalid_rubric_weights_rejected() -> None:
    """Kiểm tra từ chối bộ trọng số rubric có tổng khác 1.0."""
    with pytest.raises(ValidationError):
        RubricWeights(correctness=0.90)
