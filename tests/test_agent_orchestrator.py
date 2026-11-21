"""Unit tests cho aitest.agent và toàn bộ 5 điều kiện dừng theo thứ tự ưu tiên SRS §6.3 (Giai đoạn 5 - AC-13, AC-24)."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from aitest.agent import run_agent_loop
from aitest.analyzer import analyze_source_file
from aitest.config import AitestConfig
from aitest.llm.client import OllamaClient
from aitest.mutation import collect_ast_mutants


def _make_agent_client(
    cfg: AitestConfig,
    *,
    gen_code: str,
    improve_code: str = "def test_extra():\n    assert 1 == 1\n",
    eval_scores: float = 7.0,
    tokens_per_call: int = 100,
) -> OllamaClient:
    eval_payload = json.dumps(
        {
            "correctness": eval_scores,
            "readability": eval_scores,
            "maintainability": eval_scores,
            "isolation": eval_scores,
            "robustness": eval_scores,
            "determinism": eval_scores,
            "issues": [],
            "summary": "Đánh giá tự động.",
        }
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(
                200,
                json={"models": [{"name": cfg.model, "digest": "sha256:deadbeef"}]},
            )
        body = json.loads(request.content.decode("utf-8"))
        if "format" in body:
            content = eval_payload
        else:
            sys_msg = body["messages"][0]["content"]
            if "mutation-testing engineer" in sys_msg:
                content = improve_code
            else:
                content = gen_code
        return httpx.Response(
            200,
            json={
                "model": cfg.model,
                "message": {"role": "assistant", "content": content},
                "prompt_eval_count": tokens_per_call // 2,
                "eval_count": tokens_per_call // 2,
            },
        )

    return OllamaClient(cfg, transport=httpx.MockTransport(handler), backoff_base_sec=0.0)


def test_agent_stopping_priority_1_unfixable_error_ac13(tmp_path: Path) -> None:
    """Điều kiện dừng (1): Gặp lỗi không thể sửa bằng test (ENVIRONMENT_ERROR) -> dừng ngay."""
    src_file = tmp_path / "mod_a.py"
    src_file.write_text("def calc(x: int) -> int:\n    return x + 1\n", encoding="utf-8")

    # Test sinh ra import một thư viện không tồn tại -> ENVIRONMENT_ERROR
    bad_env_test = "import non_existent_library_999\n\ndef test_calc():\n    assert True\n"
    cfg = AitestConfig()
    client = _make_agent_client(cfg, gen_code=bad_env_test)

    res = run_agent_loop(
        target_file=src_file,
        config=cfg,
        test_file=tmp_path / "tests" / "test_mod_a.py",
        llm_client=client,
        project_root=tmp_path,
    )
    assert res.stop_reason == "unfixable_error:ENVIRONMENT_ERROR"
    assert res.iterations_completed == 1
    assert res.final_report.execution.all_tests_passed is False


def test_agent_stopping_priority_2_budget_exceeded_ac13(tmp_path: Path) -> None:
    """Điều kiện dừng (2): Vượt ngân sách tài nguyên (max_llm_calls hoặc max_total_tokens) -> dừng."""
    src_file = tmp_path / "mod_b.py"
    src_file.write_text(
        "def check(x: int) -> bool:\n"
        "    if x >= 10:\n"
        "        return True\n"
        "    return False\n",
        encoding="utf-8",
    )
    weak_test = "from mod_b import check\n\ndef test_check():\n    assert check(20) is True\n"
    cfg = AitestConfig()
    cfg.eval.threshold_pass = 9.9
    # Gen tốn 1 call + Eval vòng 1 tốn 1 call = 2 calls -> chạm ngưỡng max_llm_calls=2
    client = _make_agent_client(cfg, gen_code=weak_test, eval_scores=7.0)

    res = run_agent_loop(
        target_file=src_file,
        config=cfg,
        test_file=tmp_path / "tests" / "test_mod_b.py",
        max_llm_calls=2,
        llm_client=client,
        project_root=tmp_path,
    )
    assert res.stop_reason.startswith("budget_exceeded:max_llm_calls")
    assert res.total_llm_calls == 2


def test_agent_stopping_priority_4_and_5_max_iters_and_early_stop_ac13(tmp_path: Path) -> None:
    """Điều kiện dừng (4) max_iters và (5) early_stop khi điểm không cải thiện qua patience vòng."""
    src_file = tmp_path / "mod_c.py"
    src_file.write_text(
        "def classify(x: int) -> int:\n"
        "    if x >= 10:\n"
        "        return 1\n"
        "    return 0\n",
        encoding="utf-8",
    )
    weak_test = "from mod_c import classify\n\ndef test_classify():\n    assert classify(20) == 1\n"
    # Improve sinh ra hàm test trùng lặp không giết thêm mutant nào
    no_gain_improve = "def test_classify_again():\n    assert classify(30) == 1\n"

    cfg = AitestConfig()
    cfg.eval.threshold_pass = 9.8
    cfg.agent.early_stop_patience = 1

    client = _make_agent_client(
        cfg,
        gen_code=weak_test,
        improve_code=no_gain_improve,
        eval_scores=7.0,
    )
    res = run_agent_loop(
        target_file=src_file,
        config=cfg,
        test_file=tmp_path / "tests" / "test_mod_c.py",
        max_iters=3,
        llm_client=client,
        project_root=tmp_path,
    )
    assert res.stop_reason.startswith("early_stop_no_improvement")
    assert res.iterations_completed == 2


@pytest.mark.parametrize(
    "example_filename",
    [
        "calculator.py",
        "string_utils.py",
        "date_scheduler.py",
        "inventory.py",
        "bank_account.py",
    ],
)
def test_examples_dataset_ast_and_mutants_valid(example_filename: str) -> None:
    """Kiểm tra cả 5 file trong bộ dữ liệu thực nghiệm `examples/` đều hợp lệ AST và sinh được tập mutant chuẩn."""
    project_root = Path(__file__).resolve().parent.parent
    example_path = project_root / "examples" / example_filename
    assert example_path.is_file()

    analysis = analyze_source_file(example_path, project_root=project_root)
    total_callables = len(analysis.functions) + sum(len(c.methods) for c in analysis.classes)
    assert total_callables >= 3
    assert 50 <= analysis.total_lines <= 300

    mutants = collect_ast_mutants(example_path.read_text(encoding="utf-8"))
    assert len(mutants) >= 10


def test_experiment_summary_and_markdown_report(tmp_path: Path) -> None:
    """Kiểm tra hàm tổng hợp Mean ± SD và bảng báo cáo Markdown của scripts/run_experiment.py."""
    import importlib.util

    project_root = Path(__file__).resolve().parent.parent
    exp_script = project_root / "scripts" / "run_experiment.py"
    spec = importlib.util.spec_from_file_location("run_experiment", exp_script)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    src_file = tmp_path / "simple.py"
    src_file.write_text("def inc(x: int) -> int:\n    return x + 1\n", encoding="utf-8")
    good_test = "from simple import inc\n\ndef test_inc():\n    assert inc(1) == 2\n"
    cfg = AitestConfig()
    client = _make_agent_client(cfg, gen_code=good_test, eval_scores=9.0)

    r1 = run_agent_loop(
        target_file=src_file,
        config=cfg,
        test_file=tmp_path / "tests" / "test_simple.py",
        baseline=True,
        llm_client=client,
        project_root=tmp_path,
    )
    branch_summary = mod.summarize_branch_runs([r1])
    assert branch_summary["total_runs"] == 1
    assert branch_summary["comparable_runs"] == 1
    assert branch_summary["all_tests_passed_rate_pct"] == 100.0

    md = mod.render_markdown_comparison_report(
        {
            "meta": {
                "generated_at": "2026-11-21T12:00:00Z",
                "model": "qwen2.5-coder:7b",
                "temperature": 0.2,
                "num_ctx": 8192,
                "runs_per_file": 1,
                "max_iters": 3,
                "threshold_pass": 8.0,
                "platform": "Windows",
                "python_version": "3.12",
            },
            "baseline_a_summary": branch_summary,
            "agent_b_summary": branch_summary,
        }
    )
    assert "Mean ± SD" in md
    assert "Mutation Score (%)" in md
