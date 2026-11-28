"""Unit tests cho aitest.fixer, aitest.agent và aitest.cli (Giai đoạn 4 - AC-01, AC-02, AC-06, AC-07, AC-08, AC-13, AC-24)."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from typer.testing import CliRunner

from aitest.agent import run_agent_loop
from aitest.cli import app
from aitest.config import SDK_DEFAULT_MODEL, AitestConfig
from aitest.fixer import fix_test_file
from aitest.llm.client import AntigravityClient, SDKCallSpec, SDKTurnResult
from aitest.schemas import ErrorCategory, RunResult


def _ensure_fake_credentials(cfg: AitestConfig, tmp_path: Path) -> None:
    cfg.llm.api_key = "test-api-key"


def _sdk_turn_ok(
    model: str | None,
    response: str,
    *,
    structured_output: dict | None = None,
    prompt_tokens: int = 50,
    output_tokens: int = 30,
) -> SDKTurnResult:
    return SDKTurnResult(
        text=response,
        structured_output=structured_output,
        model_actual=model or SDK_DEFAULT_MODEL,
        conversation_id="conv-fix-1",
        prompt_tokens=prompt_tokens,
        completion_tokens=output_tokens,
        thinking_tokens=0,
        total_tokens=prompt_tokens + output_tokens,
    )


def test_fixer_stops_immediately_on_unfixable_error_without_calling_llm_ac06(
    tmp_path: Path,
) -> None:
    """Kiểm tra AC-06: Khi gặp ENVIRONMENT_ERROR hoặc SOURCE_ERROR -> dừng ngay, 0 lần gọi LLM."""
    src_file = tmp_path / "app.py"
    src_file.write_text("def f():\n    return 1\n", encoding="utf-8")
    test_file = tmp_path / "test_app.py"
    test_file.write_text("import missing_dep_xyz\n\ndef test_f():\n    assert True\n", encoding="utf-8")

    llm_called = {"count": 0}

    def runner(spec: SDKCallSpec) -> SDKTurnResult:
        llm_called["count"] += 1
        return _sdk_turn_ok(spec.model, "")

    cfg = AitestConfig()
    _ensure_fake_credentials(cfg, tmp_path)
    client = AntigravityClient(cfg, sdk_runner=runner)
    initial_run = RunResult(
        target_file=str(src_file),
        test_file=str(test_file),
        exit_code=2,
        all_tests_passed=False,
        error_category=ErrorCategory.ENVIRONMENT_ERROR,
        error_detail="ModuleNotFoundError: No module named 'missing_dep_xyz'",
    )

    res = fix_test_file(
        target_file=src_file,
        config=cfg,
        test_file=test_file,
        initial_run_result=initial_run,
        llm_client=client,
        project_root=tmp_path,
    )
    assert res.fixed is False
    assert res.stop_reason == "unfixable_error:ENVIRONMENT_ERROR"
    assert llm_called["count"] == 0


def test_fixer_rejects_weakened_assertion_and_accepts_valid_fix_ac07_ac08(
    tmp_path: Path,
) -> None:
    """Kiểm tra AC-07 & AC-08:
    - Vòng 1 LLM cố tình làm yếu assertion thành `assert True` -> bị Quality Guard reject & rollback.
    - Vòng 2 LLM sửa đúng hằng số kỳ vọng `assert add(2, 3) == 5` -> được chấp nhận và pass pytest.
    """
    src_file = tmp_path / "calc_fix.py"
    src_file.write_text("def add(a: int, b: int) -> int:\n    return a + b\n", encoding="utf-8")
    test_file = tmp_path / "test_calc_fix.py"
    test_file.write_text(
        "from calc_fix import add\n\ndef test_add():\n    assert add(2, 3) == 6\n",
        encoding="utf-8",
    )

    chat_turn = {"n": 0}

    def runner(spec: SDKCallSpec) -> SDKTurnResult:
        chat_turn["n"] += 1
        if chat_turn["n"] == 1:
            code = "from calc_fix import add\n\ndef test_add():\n    assert True\n"
        else:
            code = "from calc_fix import add\n\ndef test_add():\n    assert add(2, 3) == 5\n"
        return _sdk_turn_ok(spec.model, code)

    cfg = AitestConfig()
    cfg.mutation.enabled = False
    _ensure_fake_credentials(cfg, tmp_path)
    client = AntigravityClient(cfg, sdk_runner=runner, backoff_base_sec=0.0)

    res = fix_test_file(
        target_file=src_file,
        config=cfg,
        test_file=test_file,
        max_iters=3,
        llm_client=client,
        project_root=tmp_path,
    )
    assert res.fixed is True
    assert res.stop_reason == "all_tests_passed"
    assert res.iterations_run == 2
    assert res.records[0].accepted is False
    assert "Quality Guard" in res.records[0].reason
    assert res.records[1].accepted is True
    assert any(d.classification == "allowed_value_update" for d in res.assertion_diffs)


def test_agent_baseline_vs_feedback_driven_improve_loop_ac13_ac24(
    tmp_path: Path,
) -> None:
    """Kiểm tra AC-13 & AC-24:
    - Chế độ `--baseline` (Baseline A) dừng ngay sau lần đánh giá đầu tiên (không chạy improve).
    - Chế độ Agent B dùng phản hồi `survived_mutants` ở pha `improve` để viết thêm test biên,
      nâng mutation score lên 100% và dừng theo điều kiện thành công.
    """
    src_file = tmp_path / "gate.py"
    src_file.write_text(
        "def can_enter(age: int) -> bool:\n"
        "    if age >= 18:\n"
        "        return True\n"
        "    return False\n",
        encoding="utf-8",
    )

    initial_weak_test = (
        "from gate import can_enter\n\n"
        "def test_can_enter_adult():\n"
        "    assert can_enter(25) is True\n"
    )
    improve_boundary_test = (
        "def test_can_enter_boundary_18_and_17():\n"
        "    assert can_enter(18) is True\n"
        "    assert can_enter(17) is False\n"
    )
    eval_obj = {
        "correctness": 9.0,
        "readability": 9.0,
        "maintainability": 9.0,
        "isolation": 9.0,
        "robustness": 9.0,
        "determinism": 9.0,
        "issues": [],
        "summary": "Đánh giá tốt.",
    }

    def make_runner():
        def runner(spec: SDKCallSpec) -> SDKTurnResult:
            if spec.response_schema_model is not None:
                return _sdk_turn_ok(spec.model, json.dumps(eval_obj), structured_output=eval_obj, prompt_tokens=100, output_tokens=40)
            content = improve_boundary_test if "mutation-testing engineer" in spec.system_prompt else initial_weak_test
            return _sdk_turn_ok(spec.model, content, prompt_tokens=100, output_tokens=40)

        return runner

    cfg = AitestConfig()
    cfg.eval.threshold_pass = 8.8
    _ensure_fake_credentials(cfg, tmp_path)

    # 1. Chạy nhánh Baseline A (--baseline)
    baseline_test_file = tmp_path / "tests_baseline" / "test_gate.py"
    client_a = AntigravityClient(cfg, sdk_runner=make_runner(), backoff_base_sec=0.0)
    res_baseline = run_agent_loop(
        target_file=src_file,
        config=cfg,
        test_file=baseline_test_file,
        baseline=True,
        llm_client=client_a,
        project_root=tmp_path,
    )
    assert res_baseline.stop_reason == "baseline_completed"
    assert res_baseline.iterations_completed == 1
    assert not any(h.action == "improve" for h in res_baseline.history)
    baseline_mut = res_baseline.final_report.objective_metrics.mutation_score_pct or 0.0

    # 2. Chạy nhánh Agent B (Feedback-driven)
    agent_test_file = tmp_path / "tests_agent" / "test_gate.py"
    client_b = AntigravityClient(cfg, sdk_runner=make_runner(), backoff_base_sec=0.0)
    res_agent = run_agent_loop(
        target_file=src_file,
        config=cfg,
        test_file=agent_test_file,
        baseline=False,
        llm_client=client_b,
        project_root=tmp_path,
    )
    assert any(h.action == "improve" and h.accepted for h in res_agent.history)
    agent_mut = res_agent.final_report.objective_metrics.mutation_score_pct or 0.0
    assert agent_mut == 100.0
    assert agent_mut > baseline_mut
    assert res_agent.stop_reason in ("threshold_reached", "max_objective_metrics_reached")
    assert res_agent.history_file is not None and Path(res_agent.history_file).is_file()


def test_cli_init_and_single_file_constraint_ac01_ac02(tmp_path: Path, monkeypatch) -> None:
    """Kiểm tra AC-01 (`aitest init` không ghi đè khi thiếu `--force`) và AC-02 (`aitest gen` từ chối thư mục)."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GEMINI_API_KEY", "test-cli-api-key")
    runner = CliRunner()

    # AC-01: Lần 1 chạy aitest init -> tạo aitest.yaml, .env.example, .gitignore
    r1 = runner.invoke(app, ["init"])
    assert r1.exit_code == 0
    yaml_file = tmp_path / "aitest.yaml"
    assert yaml_file.is_file()
    assert (tmp_path / ".env.example").is_file()
    assert (tmp_path / ".gitignore").is_file()
    assert "Google Antigravity SDK" in r1.output

    # Sửa nội dung aitest.yaml rồi chạy lại `aitest init` không có --force -> không được ghi đè
    yaml_file.write_text("# custom config\nmodel: custom-model\n", encoding="utf-8")
    r2 = runner.invoke(app, ["init"])
    assert r2.exit_code == 0
    assert "custom-model" in yaml_file.read_text(encoding="utf-8")

    # Chạy `aitest init --force` -> ghi đè về mặc định v4.0.0
    r3 = runner.invoke(app, ["init", "--force"])
    assert r3.exit_code == 0
    assert "gemini-4-hydrogen" in yaml_file.read_text(encoding="utf-8")

    # AC-02: Truyền thư mục vào `aitest gen` -> báo lỗi rõ ràng, exit_code != 0
    r_dir = runner.invoke(app, ["gen", str(tmp_path)])
    assert r_dir.exit_code != 0


def test_targeted_function_fix_preserves_passing_tests_and_selective_improve(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Kiểm tra Targeted Fix, Verified Run Reuse và Selective Improve với AntigravityClient."""
    src_file = tmp_path / "arith.py"
    src_file.write_text(
        "def add(a: int, b: int) -> int:\n"
        "    return a + b\n\n"
        "def is_even(n: int) -> bool:\n"
        "    if n % 2 == 0:\n"
        "        return True\n"
        "    return False\n",
        encoding="utf-8",
    )
    test_file = tmp_path / "tests" / "test_arith.py"
    test_file.parent.mkdir(parents=True, exist_ok=True)
    test_file.write_text(
        "from arith import add, is_even\n\n"
        "def test_add():\n"
        "    # comment kept untouched\n"
        "    assert add(2, 3) == 5\n\n"
        "def test_is_even():\n"
        "    assert is_even(4) == False\n",
        encoding="utf-8",
    )

    seen_prompts: list[str] = []

    def fix_runner(spec: SDKCallSpec) -> SDKTurnResult:
        seen_prompts.append(spec.full_prompt)
        return _sdk_turn_ok(
            spec.model,
            "def test_is_even():\n    assert is_even(4) is True\n",
        )

    cfg = AitestConfig()
    cfg.mutation.enabled = False
    _ensure_fake_credentials(cfg, tmp_path)
    fix_res = fix_test_file(
        target_file=src_file,
        config=cfg,
        test_file=test_file,
        llm_client=AntigravityClient(cfg, sdk_runner=fix_runner, backoff_base_sec=0.0),
        project_root=tmp_path,
    )
    assert fix_res.fixed is True
    assert len(seen_prompts) == 1
    assert "Failing Test Function(s)" in seen_prompts[0]
    final_txt = test_file.read_text(encoding="utf-8")
    assert "# comment kept untouched" in final_txt
    assert "assert is_even(4) is True" in final_txt

    src2 = tmp_path / "check_num.py"
    src2.write_text(
        "def above_ten(x: int) -> bool:\n"
        "    if x >= 10:\n"
        "        return True\n"
        "    return False\n",
        encoding="utf-8",
    )
    gen_code = "from check_num import above_ten\n\ndef test_above():\n    assert above_ten(20) is True\n"
    improve_mixed = (
        "def test_boundary_10_good():\n"
        "    assert above_ten(10) is True\n"
        "    assert above_ten(9) is False\n\n"
        "def test_wrong_calc_bad():\n"
        "    assert above_ten(10) is False\n"
    )
    eval_obj = {
        "correctness": 9.0,
        "readability": 9.0,
        "maintainability": 9.0,
        "isolation": 9.0,
        "robustness": 9.0,
        "determinism": 9.0,
        "issues": [],
        "summary": "Tốt.",
    }

    def agent_runner(spec: SDKCallSpec) -> SDKTurnResult:
        if spec.response_schema_model is not None:
            return _sdk_turn_ok(spec.model, json.dumps(eval_obj), structured_output=eval_obj)
        c = improve_mixed if "mutation-testing engineer" in spec.system_prompt else gen_code
        return _sdk_turn_ok(spec.model, c)

    import aitest.agent as agent_mod

    run_calls = {"count": 0}
    orig_run_tests = agent_mod.run_tests

    def counted_run_tests(*args, **kwargs):
        run_calls["count"] += 1
        return orig_run_tests(*args, **kwargs)

    monkeypatch.setattr(agent_mod, "run_tests", counted_run_tests)

    cfg2 = AitestConfig()
    cfg2.eval.threshold_pass = 8.9
    _ensure_fake_credentials(cfg2, tmp_path)
    agent_test = tmp_path / "tests" / "test_check_num.py"
    agent_res = run_agent_loop(
        target_file=src2,
        config=cfg2,
        test_file=agent_test,
        max_iters=2,
        llm_client=AntigravityClient(cfg2, sdk_runner=agent_runner, backoff_base_sec=0.0),
        project_root=tmp_path,
    )
    assert agent_res.final_report.execution.all_tests_passed is True
    assert agent_res.final_report.objective_metrics.mutation_score_pct == 100.0
    assert any("Selective Accept" in h.reason for h in agent_res.history if h.action == "improve")
    assert run_calls["count"] == 3
