"""Unit tests cho aitest.fixer, aitest.agent và aitest.cli (Giai đoạn 4 - AC-01, AC-02, AC-06, AC-07, AC-08, AC-13, AC-24)."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
from typer.testing import CliRunner

from aitest.agent import run_agent_loop
from aitest.cli import app
from aitest.config import AitestConfig
from aitest.fixer import fix_test_file
from aitest.llm.client import OllamaClient
from aitest.schemas import ErrorCategory, RunResult


def test_fixer_stops_immediately_on_unfixable_error_without_calling_llm_ac06(
    tmp_path: Path,
) -> None:
    """Kiểm tra AC-06: Khi gặp ENVIRONMENT_ERROR hoặc SOURCE_ERROR -> dừng ngay, 0 lần gọi LLM."""
    src_file = tmp_path / "app.py"
    src_file.write_text("def f():\n    return 1\n", encoding="utf-8")
    test_file = tmp_path / "test_app.py"
    test_file.write_text("import missing_dep_xyz\n\ndef test_f():\n    assert True\n", encoding="utf-8")

    llm_called = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        llm_called["count"] += 1
        return httpx.Response(200, json={})

    cfg = AitestConfig()
    client = OllamaClient(cfg, transport=httpx.MockTransport(handler))
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

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": []})
        chat_turn["n"] += 1
        if chat_turn["n"] == 1:
            # Lần 1: LLM gian lận bằng tautology assert True
            code = "from calc_fix import add\n\ndef test_add():\n    assert True\n"
        else:
            # Lần 2: LLM sửa đúng giá trị hằng số kỳ vọng từ 6 thành 5
            code = "from calc_fix import add\n\ndef test_add():\n    assert add(2, 3) == 5\n"
        return httpx.Response(
            200,
            json={
                "model": "qwen2.5-coder:7b",
                "message": {"role": "assistant", "content": code},
                "prompt_eval_count": 50,
                "eval_count": 30,
            },
        )

    cfg = AitestConfig()
    cfg.mutation.enabled = False
    client = OllamaClient(cfg, transport=httpx.MockTransport(handler), backoff_base_sec=0.0)

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
    eval_json = json.dumps(
        {
            "correctness": 9.0,
            "readability": 9.0,
            "maintainability": 9.0,
            "isolation": 9.0,
            "robustness": 9.0,
            "determinism": 9.0,
            "issues": [],
            "summary": "Đánh giá tốt.",
        }
    )

    def make_transport() -> httpx.MockTransport:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/api/tags":
                return httpx.Response(
                    200,
                    json={"models": [{"name": "qwen2.5-coder:7b", "digest": "sha256:abcd"}]},
                )
            body = json.loads(request.content.decode("utf-8"))
            if "format" in body:
                # Bước eval
                content = eval_json
            else:
                sys_msg = body["messages"][0]["content"]
                if "mutation-testing engineer" in sys_msg:
                    content = improve_boundary_test
                else:
                    content = initial_weak_test
            return httpx.Response(
                200,
                json={
                    "model": "qwen2.5-coder:7b",
                    "message": {"role": "assistant", "content": content},
                    "prompt_eval_count": 100,
                    "eval_count": 40,
                },
            )

        return httpx.MockTransport(handler)

    cfg = AitestConfig()
    # Đặt threshold_pass = 8.8 để vòng 1 (mutation < 100%) chưa đạt ngưỡng, buộc Agent B chạy pha improve
    cfg.eval.threshold_pass = 8.8

    # 1. Chạy nhánh Baseline A (--baseline)
    baseline_test_file = tmp_path / "tests_baseline" / "test_gate.py"
    client_a = OllamaClient(cfg, transport=make_transport(), backoff_base_sec=0.0)
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
    client_b = OllamaClient(cfg, transport=make_transport(), backoff_base_sec=0.0)
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
    runner = CliRunner()

    # AC-01: Lần 1 chạy aitest init -> tạo aitest.yaml, .env.example, .gitignore
    r1 = runner.invoke(app, ["init"])
    assert r1.exit_code == 0
    yaml_file = tmp_path / "aitest.yaml"
    assert yaml_file.is_file()
    assert (tmp_path / ".env.example").is_file()
    assert (tmp_path / ".gitignore").is_file()

    # Sửa nội dung aitest.yaml rồi chạy lại `aitest init` không có --force -> không được ghi đè
    yaml_file.write_text("# custom config\nmodel: custom:7b\n", encoding="utf-8")
    r2 = runner.invoke(app, ["init"])
    assert r2.exit_code == 0
    assert "custom:7b" in yaml_file.read_text(encoding="utf-8")

    # Chạy `aitest init --force` -> ghi đè về mặc định
    r3 = runner.invoke(app, ["init", "--force"])
    assert r3.exit_code == 0
    assert "qwen2.5-coder:7b" in yaml_file.read_text(encoding="utf-8")

    # AC-02: Truyền thư mục vào `aitest gen` -> báo lỗi rõ ràng, exit_code != 0
    r_dir = runner.invoke(app, ["gen", str(tmp_path)])
    assert r_dir.exit_code != 0
