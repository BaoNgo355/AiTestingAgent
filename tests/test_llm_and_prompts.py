"""Unit tests cho aitest.llm.client, aitest.llm.prompts và aitest.context (Giai đoạn 2 - AC-14, AC-16, AC-18)."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from aitest.config import AitestConfig
from aitest.context import (
    build_eval_prompts,
    build_fix_prompts,
    build_gen_prompts,
    build_improve_prompts,
)
from aitest.llm.client import (
    OllamaClient,
    OllamaModelNotFoundError,
    OllamaSchemaValidationError,
    OllamaServerError,
)
from aitest.schemas import ErrorCategory, MutationResult, RunResult, SurvivedMutant


def test_ollama_client_tags_chat_and_retry_ac14_ac18() -> None:
    """Kiểm tra lấy model_digest, thống kê token/latency, retry 5xx và báo lỗi 404 model not found."""
    attempts = {"chat": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(
                200,
                json={
                    "models": [
                        {
                            "name": "qwen2.5-coder:7b",
                            "digest": "2bada8a7450677000f678be90653b85d364de7db25eb5ea54136ada5f3933730",
                        }
                    ]
                },
            )
        if request.url.path == "/api/chat":
            attempts["chat"] += 1
            if attempts["chat"] == 1:
                return httpx.Response(500, text="Temporary internal error")
            return httpx.Response(
                200,
                json={
                    "model": "qwen2.5-coder:7b",
                    "message": {"role": "assistant", "content": "def test_ok():\n    assert 1 == 1\n"},
                    "prompt_eval_count": 120,
                    "eval_count": 30,
                },
            )
        return httpx.Response(404)

    cfg = AitestConfig()
    client = OllamaClient(cfg, transport=httpx.MockTransport(handler), backoff_base_sec=0.0)
    resp = client.chat(system_prompt="sys", user_prompt="usr")

    assert attempts["chat"] == 2  # Lần 1 lỗi 500, lần 2 thành công
    assert "def test_ok():" in resp.content
    assert resp.model_actual == "qwen2.5-coder:7b"
    assert resp.model_digest == "sha256:2bada8a7450677000f678be90653b85d364de7db25eb5ea54136ada5f3933730"
    assert resp.total_tokens == 150
    assert client.total_calls == 1
    assert client.total_tokens == 150


def test_ollama_client_404_model_not_found_no_retry_ac14() -> None:
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        return httpx.Response(404, json={"error": "model 'missing:7b' not found"})

    cfg = AitestConfig(model="missing:7b")
    client = OllamaClient(cfg, transport=httpx.MockTransport(handler), backoff_base_sec=0.0)
    with pytest.raises(OllamaModelNotFoundError, match="ollama pull missing:7b"):
        client.chat(system_prompt="sys", user_prompt="usr")
    assert calls["count"] == 1  # Không retry khi 404


def test_ollama_client_structured_assessment_retry_and_log_ac16(tmp_path: Path) -> None:
    """Kiểm tra AC-16: Lưu raw response vào .aitest_logs/, retry 1 lần khi JSON sai schema, và báo lỗi nếu vẫn sai."""
    call_idx = {"n": 0}
    valid_json = json.dumps(
        {
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
    )

    def recover_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": []})
        call_idx["n"] += 1
        content = "{invalid json" if call_idx["n"] == 1 else valid_json
        return httpx.Response(
            200,
            json={
                "model": "qwen2.5-coder:7b",
                "message": {"role": "assistant", "content": content},
                "prompt_eval_count": 50,
                "eval_count": 50,
            },
        )

    cfg = AitestConfig()
    log_dir = tmp_path / ".aitest_logs"
    client = OllamaClient(cfg, transport=httpx.MockTransport(recover_handler), backoff_base_sec=0.0)
    assessment, _ = client.generate_structured_assessment(
        system_prompt="sys",
        user_prompt="usr",
        run_id="run-ac16",
        log_dir=log_dir,
    )
    assert assessment.correctness == 8.5
    assert call_idx["n"] == 2
    assert any(log_dir.glob("*eval_schema_invalid_attempt1.log"))

    # Trường hợp cả 2 lần đều trả JSON sai -> ném OllamaSchemaValidationError, không bịa điểm
    def always_bad_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": "qwen2.5-coder:7b",
                "message": {"role": "assistant", "content": '{"correctness": 999}'},
            },
        )

    bad_client = OllamaClient(cfg, transport=httpx.MockTransport(always_bad_handler), backoff_base_sec=0.0)
    with pytest.raises(OllamaSchemaValidationError):
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

    run_res = RunResult(
        target_file="examples/calc.py",
        test_file="tests/test_calc.py",
        exit_code=0,
        all_tests_passed=True,
        tests_passed=1,
        pass_rate=1.0,
        line_coverage_pct=100.0,
        test_count=1,
        mutation=MutationResult(
            mutation_score_pct=80.0,
            mutation_killed=4,
            mutation_survived=1,
            mutation_excluded=0,
            mutation_status="available",
            mutation_available=True,
            survived_mutants=[SurvivedMutant(file="examples/calc.py", line=2, mutation="+ -> -")],
        ),
    )
    sys_e, usr_e = build_eval_prompts(
        target_file="examples/calc.py",
        test_file="tests/test_calc.py",
        source_code=src,
        test_code="def test_add():\n    assert add(1, 2) == 3\n",
        run_result=run_res,
        objective_metrics=run_res.to_objective_metrics(),
    )
    assert "Do NOT compute weighted_score" in sys_e
    assert "mutation_score_pct: 80.0" in usr_e
