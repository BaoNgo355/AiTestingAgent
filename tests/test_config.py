"""Unit tests cho aitest.config (Giai đoạn 5 - AC-01, AC-10, AC-18, NFR-03)."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from aitest.config import SDK_DEFAULT_MODEL, AitestConfig, RubricWeights, load_config
from aitest.utils import compute_config_hash


def test_default_config_values_and_rubric_sum() -> None:
    """Kiểm tra cấu hình mặc định khớp với SRS v4.0.0 §10.1."""
    cfg = AitestConfig()
    assert cfg.provider == "antigravity"
    assert cfg.model == "gemini-4-hydrogen"
    assert cfg.effective_model_label == SDK_DEFAULT_MODEL
    assert cfg.timeout_sec == 120
    assert cfg.eval.judge_model is None
    assert cfg.llm.api_key is None
    assert cfg.llm.vertex is False
    assert cfg.agent.max_total_tokens == 400000
    assert cfg.mutation.enabled is True
    assert cfg.mutation.tool == "auto"
    assert cfg.eval.threshold_pass == 8.0
    assert cfg.eval.correctness_cap_offset == 2.0
    assert abs(sum(cfg.eval.rubric.to_dict().values()) - 1.0) <= 1e-6


def test_config_priority_cli_over_env_over_yaml_over_default(tmp_path: Path) -> None:
    """Kiểm tra thứ tự ưu tiên cấu hình: CLI flags > AITEST_* / GEMINI_API_KEY > aitest.yaml > defaults."""
    yaml_path = tmp_path / "aitest.yaml"
    yaml_path.write_text(
        "model: gemini-3.8-pro\n"
        "eval:\n"
        "  threshold_pass: 7.5\n"
        "  judge_model: gemini-3.8-pro\n"
        "llm:\n"
        "  vertex: false\n",
        encoding="utf-8",
    )
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text(
        "GEMINI_API_KEY=from-dotenv-secret\n",
        encoding="utf-8",
    )

    env = {
        "GEMINI_API_KEY": "from-env-secret-key",
        "AITEST_LLM__VERTEX": "true",
        "GOOGLE_CLOUD_PROJECT": "demo-gcp-proj",
        "GOOGLE_CLOUD_LOCATION": "us-central1",
        "AITEST_AGENT__MAX_ITERS": "5",
        "AITEST_EVAL__JUDGE_MODEL": "null",
    }
    cli_overrides = {
        "model": "gemini-4-hydrogen",
    }

    cfg = load_config(
        config_path=yaml_path,
        cli_overrides=cli_overrides,
        env=env,
        dotenv_path=dotenv_path,
    )
    assert cfg.model == "gemini-4-hydrogen"
    assert cfg.effective_model_label == "gemini-4-hydrogen"
    assert cfg.eval.threshold_pass == 7.5
    assert cfg.eval.judge_model is None
    assert cfg.llm.api_key == "from-env-secret-key"
    assert cfg.llm.vertex is True
    assert cfg.llm.project == "demo-gcp-proj"
    assert cfg.llm.location == "us-central1"
    assert cfg.agent.max_iters == 5

    # Bảo mật NFR-03: safe_dump() & config_hash không bao giờ chứa api_key
    dumped = cfg.safe_dump()
    assert dumped["provider"] == "antigravity"
    assert "api_key" not in dumped["llm"]
    assert "from-env-secret-key" not in str(dumped)
    assert compute_config_hash(cfg).startswith("sha256:")


def test_invalid_rubric_weights_rejected() -> None:
    """Kiểm tra từ chối bộ trọng số rubric có tổng khác 1.0."""
    with pytest.raises(ValidationError):
        RubricWeights(correctness=0.90)
