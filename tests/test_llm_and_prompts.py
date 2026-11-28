"""Unit tests cho aitest.llm.client (Google Antigravity SDK), aitest.llm.prompts và aitest.context (AC-14, AC-16, AC-18)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aitest.config import SDK_DEFAULT_MODEL, AitestConfig
from aitest.context import (
    build_eval_prompts,
    build_fix_prompts,
    build_gen_prompts,
    build_improve_prompts,
)
from aitest.llm.client import (
    AntigravityAuthError,
    AntigravityClient,
    AntigravityModelNotFoundError,
    AntigravitySchemaValidationError,
    AntigravityServerError,
    SDKCallSpec,
    SDKTurnResult,
    build_sdk_agent_config,
    validate_sdk_credentials,
)
from aitest.schemas import ErrorCategory, LLMAssessment, MutationResult, RunResult, SurvivedMutant


def test_build_sdk_agent_config_isolation_and_credential_validation(tmp_path: Path) -> None:
    """Kiểm tra khởi tạo LocalAgentConfig thực của SDK (khóa tool, app_data_dir, model=None mặc định) và validate key."""
    # Thiếu GEMINI_API_KEY ở chế độ mặc định -> AntigravityAuthError có link Google AI Studio
    cfg_no_key = AitestConfig()
    with pytest.raises(AntigravityAuthError, match="aistudio.google.com"):
        validate_sdk_credentials(cfg_no_key)

    # Vertex AI thiếu cả api_key lẫn project+location -> AntigravityAuthError
    cfg_bad_vertex = AitestConfig()
    cfg_bad_vertex.llm.vertex = True
    with pytest.raises(AntigravityAuthError, match="Vertex AI"):
        validate_sdk_credentials(cfg_bad_vertex)

    # Hợp lệ với GEMINI_API_KEY
    cfg_ok = AitestConfig()
    cfg_ok.llm.api_key = "test-api-key"
    validate_sdk_credentials(cfg_ok)

    spec = SDKCallSpec(
        system_prompt="You are a QA engineer.",
        full_prompt="Generate pytest tests.",
        model=SDK_DEFAULT_MODEL,
        api_key=cfg_ok.llm.api_key,
        vertex=False,
        project=None,
        location=None,
        timeout_sec=120.0,
        response_schema_model=LLMAssessment,
    )
    sdk_cfg = build_sdk_agent_config(spec, app_data_dir=str(tmp_path / "sdk_data"))
    assert sdk_cfg.model == "gemini-4-hydrogen"
    assert sdk_cfg.capabilities.enable_subagents is False
    assert sdk_cfg.capabilities.enabled_tools == []
    assert sdk_cfg.response_schema is not None
    assert Path(sdk_cfg.app_data_dir).is_absolute()


def test_antigravity_sdk_client_retry_and_tokens_ac14_ac18() -> None:
    """Kiểm tra retry lỗi server tạm thời, đọc usage tokens (gồm thinking_tokens) và truy vết model mặc định."""
    attempts = {"chat": 0}
    seen_specs: list[SDKCallSpec] = []

    def runner(spec: SDKCallSpec) -> SDKTurnResult:
        seen_specs.append(spec)
        attempts["chat"] += 1
        if attempts["chat"] == 1:
            raise RuntimeError("503 Temporary backend unavailable")
        return SDKTurnResult(
            text="def test_ok():\n    assert 1 == 1\n",
            model_actual=SDK_DEFAULT_MODEL,
            conversation_id="conv-sdk-001",
            prompt_tokens=120,
            completion_tokens=30,
            thinking_tokens=15,
            cache_read_tokens=20,
            total_tokens=165,
        )

    cfg = AitestConfig()
    cfg.llm.api_key = "test-api-key"

    client = AntigravityClient(cfg, sdk_runner=runner, backoff_base_sec=0.0)
    resp = client.chat(system_prompt="sys", user_prompt="usr")
    assert attempts["chat"] == 2
    assert seen_specs[-1].model == SDK_DEFAULT_MODEL
    assert seen_specs[-1].api_key == "test-api-key"
    assert resp.model_requested == SDK_DEFAULT_MODEL
    assert resp.model_actual == SDK_DEFAULT_MODEL
    assert resp.conversation_id == "conv-sdk-001"
    assert resp.thinking_tokens == 15
    assert resp.total_tokens == 165
    assert client.total_calls == 1
    assert client.total_thinking_tokens == 15
    assert client.total_tokens == 165


def test_antigravity_sdk_client_auth_and_unknown_model_no_retry_ac14() -> None:
    """Kiểm tra AntigravityModelNotFoundError và AntigravityAuthError dừng ngay không retry."""
    calls = {"n": 0}

    def unknown_model_runner(spec: SDKCallSpec) -> SDKTurnResult:
        calls["n"] += 1
        raise RuntimeError('404 models/missing-model is not found for API version')

    cfg = AitestConfig(model="missing-model")
    cfg.llm.api_key = "test-api-key"
    client = AntigravityClient(cfg, sdk_runner=unknown_model_runner, backoff_base_sec=0.0)
    with pytest.raises(AntigravityModelNotFoundError):
        client.chat(system_prompt="sys", user_prompt="usr")
    assert calls["n"] == 1

    calls["n"] = 0

    def auth_err_runner(spec: SDKCallSpec) -> SDKTurnResult:
        calls["n"] += 1
        raise RuntimeError("401 API_KEY_INVALID: API key not valid")

    client_auth = AntigravityClient(cfg, sdk_runner=auth_err_runner, backoff_base_sec=0.0)
    with pytest.raises(AntigravityAuthError, match="GEMINI_API_KEY"):
        client_auth.chat(system_prompt="sys", user_prompt="usr")
    assert calls["n"] == 1


def test_antigravity_sdk_client_structured_assessment_judge_model_retry_and_log_ac16(tmp_path: Path) -> None:
    """Kiểm tra AC-16: response_schema=LLMAssessment, judge_model routing, retry 1 lần và lưu raw log."""
    call_idx = {"n": 0}
    seen_specs: list[SDKCallSpec] = []
    valid_obj = {
        "correctness": 8.5,
        "readability": 9.0,
        "maintainability": 8.0,
        "isolation": 9.0,
        "robustness": 7.5,
        "determinism": 9.5,
        "issues": [
            {
                "title": "Thiếu biên âm",
                "description": "Chưa test số âm",
                "suggestion": "Thêm test với -1",
            }
        ],
        "summary": "Bộ test tốt.",
    }

    def recover_runner(spec: SDKCallSpec) -> SDKTurnResult:
        call_idx["n"] += 1
        seen_specs.append(spec)
        assert spec.response_schema_model is LLMAssessment
        if call_idx["n"] == 1:
            return SDKTurnResult(
                text="{invalid json",
                structured_output=None,
                model_actual=spec.model or SDK_DEFAULT_MODEL,
                conversation_id="conv-eval-1",
            )
        return SDKTurnResult(
            text=json.dumps(valid_obj),
            structured_output=valid_obj,
            model_actual=spec.model or SDK_DEFAULT_MODEL,
            conversation_id="conv-eval-1",
        )

    cfg = AitestConfig()
    cfg.llm.api_key = "test-api-key"
    cfg.eval.judge_model = "gemini-3.8-pro"
    log_dir = tmp_path / ".aitest_logs"

    client = AntigravityClient(cfg, sdk_runner=recover_runner, backoff_base_sec=0.0)
    assessment, resp = client.generate_structured_assessment(
        system_prompt="sys",
        user_prompt="usr",
        run_id="run-ac16",
        log_dir=log_dir,
    )
    assert assessment.correctness == 8.5
    assert [s.model for s in seen_specs] == ["gemini-3.8-pro", "gemini-3.8-pro"]
    assert resp.model_actual == "gemini-3.8-pro"
    assert call_idx["n"] == 2
    assert any(log_dir.glob("*eval_schema_invalid_attempt1.log"))

    def always_bad_runner(spec: SDKCallSpec) -> SDKTurnResult:
        return SDKTurnResult(
            text='{"correctness": 999}',
            structured_output={"correctness": 999},
            model_actual=spec.model or SDK_DEFAULT_MODEL,
        )

    bad_client = AntigravityClient(cfg, sdk_runner=always_bad_runner, backoff_base_sec=0.0)
    with pytest.raises(AntigravitySchemaValidationError):
        bad_client.generate_structured_assessment(
            system_prompt="sys",
            user_prompt="usr",
            run_id="run-fail",
            log_dir=log_dir,
        )
    assert any(log_dir.glob("*eval_schema_invalid_attempt2.log"))


def test_context_and_jinja2_prompts_render_cleanly() -> None:
    src = "def add(a: int, b: int) -> int:\n    return a + b\n"
    sys_g, usr_g = build_gen_prompts(
        target_file="examples/calc.py",
        source_code=src,
        existing_test_names=["test_add_basic"],
    )
    assert "raw Python test code" in sys_g
    assert "test_add_basic" in usr_g

    sys_f, usr_f = build_fix_prompts(
        target_file="examples/calc.py",
        test_file="tests/test_calc.py",
        source_code=src,
        current_test_code="def test_add():\n    assert add(1, 2) == 4\n",
        pytest_output="AssertionError: assert 3 == 4",
        error_category=ErrorCategory.TEST_FAILURE,
    )
    assert "Do NOT weaken assertions" in sys_f
    assert "AssertionError" in usr_f

    sys_i, usr_i = build_improve_prompts(
        target_file="examples/calc.py",
        source_code=src,
        existing_test_names=["test_add"],
        survived_mutants=[SurvivedMutant(file="examples/calc.py", line=2, mutation="+ -> -")],
        missing_lines=[2],
    )
    assert "surviving mutants" in sys_i
    assert "+ -> -" in usr_i
