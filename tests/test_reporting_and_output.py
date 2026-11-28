"""Unit tests cho hệ thống xuất báo cáo đa định dạng, ngữ cảnh mã nguồn, Rich Dashboard, Smart TTY và bố cục test PEP 8 (v3.1.0)."""

from __future__ import annotations

import ast
import csv
import io
import subprocess
from pathlib import Path

import pytest
from pydantic import ValidationError
from rich.console import Console
from typer.testing import CliRunner

from aitest.agent import AgentRunResult
from aitest.cli import app
from aitest.config import AitestConfig, OutputConfig, normalize_output_format
from aitest.fixer import FixResult
from aitest.generator import format_python_test_layout, safe_merge_and_append
from aitest.output import (
    deliver_report,
    infer_report_format,
    print_agent_summary,
    print_eval_dashboard,
    print_fix_result,
    print_run_result,
)
from aitest.reporting import (
    build_missing_line_rows,
    build_mutant_context_rows,
    compress_line_ranges,
    load_source_lines,
    render_eval_html,
    render_eval_markdown,
    render_experiment_csv,
    render_experiment_html,
)
from aitest.schemas import (
    AgentIterationLog,
    AssessmentIssue,
    ErrorCategory,
    EvalReport,
    Execution,
    LLMAssessment,
    Meta,
    MutationResult,
    ObjectiveMetrics,
    RunResult,
    SurvivedMutant,
    build_eval_report,
)


def _sample_report(target_file: Path, *, mutation_score_pct: float = 70.0) -> EvalReport:
    meta = Meta(
        version="4.0.0",
        generated_at="2026-11-28T10:00:00Z",
        target_file=str(target_file),
        test_file="tests/test_sample.py",
        model_requested="gemini-3.8-flash",
        model_actual="gemini-3.8-flash",
        judge_model_requested="gemini-3.8-pro",
        judge_model_actual="gemini-3.8-pro",
        vertex=False,
        conversation_id="conv-report-1",
        config_hash="sha256:abc123",
        run_id="run-report-1234",
    )
    execution = Execution(
        exit_code=0,
        all_tests_passed=True,
        tests_passed=6,
        tests_failed=0,
        tests_skipped=0,
        pass_rate=1.0,
        duration_ms=420,
    )
    objective = ObjectiveMetrics(
        line_coverage_pct=85.0,
        mutation_score_pct=mutation_score_pct,
        mutation_killed=7,
        mutation_survived=3,
        mutation_excluded=0,
        mutation_status="available",
        mutation_available=True,
        test_count=6,
        missing_lines=[5, 6],
        survived_mutants=[
            SurvivedMutant(file=str(target_file), line=2, mutation=">= -> >"),
            SurvivedMutant(file=str(target_file), line=4, mutation="* -> /"),
        ],
    )
    llm = LLMAssessment(
        correctness=9.5,
        readability=9.0,
        maintainability=8.5,
        isolation=9.5,
        robustness=8.0,
        determinism=10.0,
        issues=[
            AssessmentIssue(
                title="Thiếu kiểm tra mốc biên >= và ký tự đặc biệt <tag>",
                description="Hàm kiểm tra điều kiện tại dòng 2 chưa có test đúng tại ngưỡng biên.",
                suggestion="Thêm test kiểm tra giá trị bằng đúng ngưỡng biên và trường hợp items[0].",
            )
        ],
        summary="Bộ test rõ ràng nhưng cần bổ sung biên >= để diệt mutant dòng 2.",
    )
    return build_eval_report(
        meta=meta,
        execution=execution,
        objective_metrics=objective,
        llm_assessment=llm,
        weights=AitestConfig().eval.rubric.to_dict(),
        threshold_pass=8.0,
        correctness_cap_offset=2.0,
    )


def test_output_format_normalization_and_inference() -> None:
    assert normalize_output_format("json") == "json"
    assert normalize_output_format("markdown") == "md"
    assert normalize_output_format("HTM") == "html"
    assert OutputConfig(format="md").format == "md"

    with pytest.raises(ValidationError):
        OutputConfig(format="xml")  # type: ignore[arg-type]

    assert infer_report_format(None, Path("out/report.html")) == "html"
    assert infer_report_format(None, Path("out/report.markdown")) == "md"
    assert infer_report_format("json", Path("out/report.html")) == "json"
    assert infer_report_format(None, None, "md") == "md"


def test_source_context_utilities(tmp_path: Path) -> None:
    assert compress_line_ranges([12, 10, 11, 18, 23, 22]) == "10-12, 18, 22-23"
    assert compress_line_ranges([]) == "(Không có)"

    src = tmp_path / "logic.py"
    src.write_text(
        "def check(items: list[int]) -> int:\n"
        "    if items[0] >= 10:\n"
        "        return items[0]\n"
        "    return 0\n",
        encoding="utf-8",
    )
    lines = load_source_lines(src)
    mut_rows = build_mutant_context_rows(
        [SurvivedMutant(file=str(src), line=2, mutation=">= -> >")],
        lines,
    )
    assert mut_rows == [{"line": 2, "mutation": ">= -> >", "code": "if items[0] >= 10:"}]
    miss_rows = build_missing_line_rows([4, 3, 4], lines)
    assert miss_rows == [
        {"line": 3, "code": "return items[0]"},
        {"line": 4, "code": "return 0"},
    ]
    assert load_source_lines(tmp_path / "missing_file.py") == []


def test_render_eval_markdown_and_html_with_code_context_and_agent_history(tmp_path: Path) -> None:
    src = tmp_path / "target_ops.py"
    src.write_text(
        "def evaluate_qty(qty: int, price: int, items: list[int]) -> int:\n"
        "    if qty >= 10:\n"
        "        return qty * price\n"
        "    bonus = items[0] * 2\n"
        "    if bonus < 0:\n"
        "        raise ValueError('<invalid>')\n"
        "    return bonus\n",
        encoding="utf-8",
    )
    report = _sample_report(src)
    agent_res = AgentRunResult(
        run_id="run-report-1234",
        target_file=str(src),
        test_file="tests/test_sample.py",
        baseline_mode=False,
        stop_reason="threshold_reached",
        iterations_completed=2,
        total_llm_calls=4,
        total_tokens=2400,
        runtime_sec=4.5,
        quality_guard_rejections=0,
        history=[
            AgentIterationLog(
                iteration=1,
                action="eval",
                metrics_before={"test_count": 4, "line_coverage_pct": 85.0, "mutation_score_pct": 50.0},
                metrics_after={"test_count": 4, "line_coverage_pct": 85.0, "mutation_score_pct": 50.0, "weighted_score": 7.80},
                llm_calls=1,
                tokens_used=600,
                accepted=True,
                reason="Đánh giá vòng 1",
            ),
            AgentIterationLog(
                iteration=2,
                action="improve",
                metrics_before={"test_count": 4, "line_coverage_pct": 85.0, "mutation_score_pct": 50.0},
                metrics_after={"test_count": 6, "line_coverage_pct": 85.0, "mutation_score_pct": 70.0},
                llm_calls=1,
                tokens_used=900,
                accepted=True,
                reason="Bổ sung 2 hàm test",
            ),
        ],
        final_report=report,
    )

    md = render_eval_markdown(report, agent_result=agent_res, project_root=tmp_path)
    assert "### Survived Mutants (2)" in md
    assert "if qty >= 10:" in md
    assert "5-6" in md
    assert "(+20.00%)" in md
    assert "Bị áp trần từ 9.50 -> 9.00 theo mutation" in md

    html_doc = render_eval_html(report, agent_result=agent_res, project_root=tmp_path)
    assert html_doc.startswith("<!DOCTYPE html>")
    assert "<svg" in html_doc
    assert "Capped" in html_doc
    assert "UNCOVERED" in html_doc
    assert "⚠ Mutant: &gt;= -&gt; &gt;" in html_doc
    assert "raise ValueError(&#x27;&lt;invalid&gt;&#x27;)" in html_doc
    assert "<invalid>" not in html_doc  # đảm bảo đã escape thẻ XML/HTML trong code
    assert "Tiến trình Vòng lặp Agent" in html_doc


def test_render_experiment_html_and_csv() -> None:
    summary_data = {
        "meta": {
            "generated_at": "2026-11-28T10:00:00Z",
            "provider": "antigravity",
            "model": "gemini-3.8-flash",
            "judge_model": "gemini-3.8-pro",
            "vertex": False,
            "runs_per_file": 3,
            "threshold_pass": 8.0,
        },
        "baseline_a_summary": {
            "line_coverage_pct": {"mean": 92.0, "sd": 1.5, "formatted": "92.00 ± 1.50%"},
            "mutation_score_pct": {"mean": 75.0, "sd": 3.0, "formatted": "75.00 ± 3.00%"},
            "mutation_killed": {"mean": 30.0, "sd": 1.0, "formatted": "30.00 ± 1.00"},
            "mutation_survived": {"mean": 10.0, "sd": 1.0, "formatted": "10.00 ± 1.00"},
            "test_count": {"mean": 12.0, "sd": 0.5, "formatted": "12.00 ± 0.50"},
            "weighted_score": {"mean": 8.10, "sd": 0.2, "formatted": "8.10 ± 0.20"},
            "llm_calls": {"mean": 2.0, "sd": 0.0, "formatted": "2.00 ± 0.00"},
            "total_tokens": {"mean": 1200.0, "sd": 50.0, "formatted": "1200.00 ± 50.00"},
            "runtime_sec": {"mean": 4.0, "sd": 0.5, "formatted": "4.00 ± 0.50s"},
            "quality_guard_rejections": {"total": 0},
            "all_tests_passed_rate_pct": 100.0,
            "report_comparable_rate_pct": 100.0,
        },
        "agent_b_summary": {
            "line_coverage_pct": {"mean": 98.0, "sd": 1.0, "formatted": "98.00 ± 1.00%"},
            "mutation_score_pct": {"mean": 94.0, "sd": 2.0, "formatted": "94.00 ± 2.00%"},
            "mutation_killed": {"mean": 38.0, "sd": 1.0, "formatted": "38.00 ± 1.00"},
            "mutation_survived": {"mean": 2.0, "sd": 1.0, "formatted": "2.00 ± 1.00"},
            "test_count": {"mean": 18.0, "sd": 1.0, "formatted": "18.00 ± 1.00"},
            "weighted_score": {"mean": 9.25, "sd": 0.15, "formatted": "9.25 ± 0.15"},
            "llm_calls": {"mean": 4.0, "sd": 0.5, "formatted": "4.00 ± 0.50"},
            "total_tokens": {"mean": 2800.0, "sd": 100.0, "formatted": "2800.00 ± 100.00"},
            "runtime_sec": {"mean": 9.5, "sd": 1.0, "formatted": "9.50 ± 1.00s"},
            "quality_guard_rejections": {"total": 1},
            "all_tests_passed_rate_pct": 100.0,
            "report_comparable_rate_pct": 100.0,
        },
        "delta_analysis": {
            "win_tie_loss": {"win": 1, "tie": 0, "loss": 0},
            "per_file": [
                {
                    "target_file": "examples/calculator.py",
                    "baseline_a_mean_mut_pct": 75.0,
                    "agent_b_mean_mut_pct": 94.0,
                    "delta_mut_pct": 19.0,
                    "outcome": "Win",
                }
            ],
        },
    }

    exp_html = render_experiment_html(summary_data)
    assert "<!DOCTYPE html>" in exp_html
    assert "<svg" in exp_html
    assert "examples/calculator.py" in exp_html
    assert "1 / 0 / 0" in exp_html

    exp_csv = render_experiment_csv(summary_data)
    parsed_rows = [r for r in csv.reader(io.StringIO(exp_csv)) if r]
    assert parsed_rows[0][0] == "Section"
    assert any(r[0] == "Summary" and r[1] == "Mutation Score (%)" for r in parsed_rows)
    assert any(r[0] == "PerFileDelta" and r[1] == "examples/calculator.py" for r in parsed_rows)


def test_rich_dashboards_and_smart_tty_routing(tmp_path: Path, capsys) -> None:
    src = tmp_path / "bracket_code.py"
    src.write_text(
        "def pick(items: list[int]) -> int:\n"
        "    if len(items) >= 1:\n"
        "        return items[0]\n"
        "    return -1\n",
        encoding="utf-8",
    )
    report = _sample_report(src)

    # Kiểm tra Rich không bị lỗi MarkupError khi source code chứa cặp ngoặc vuông items[0]
    buf = io.StringIO()
    cons = Console(file=buf, force_terminal=False, width=120, color_system=None)

    run_before = RunResult(
        target_file=str(src),
        test_file="tests/test_sample.py",
        exit_code=1,
        all_tests_passed=False,
        tests_passed=3,
        tests_failed=2,
        pass_rate=0.6,
        line_coverage_pct=60.0,
        missing_lines=[2, 3, 4],
        error_category=ErrorCategory.TEST_FAILURE,
    )
    run_after = RunResult(
        target_file=str(src),
        test_file="tests/test_sample.py",
        exit_code=0,
        all_tests_passed=True,
        tests_passed=5,
        tests_failed=0,
        pass_rate=1.0,
        line_coverage_pct=100.0,
        mutation=MutationResult(
            mutation_score_pct=75.0,
            mutation_killed=3,
            mutation_survived=1,
            mutation_excluded=0,
            mutation_status="available",
            mutation_available=True,
            survived_mutants=[SurvivedMutant(file=str(src), line=3, mutation="items[0] -> items[1]")],
        ),
    )
    fix_res = FixResult(
        target_file=str(src),
        test_file="tests/test_sample.py",
        fixed=True,
        iterations_run=1,
        stop_reason="all_tests_passed",
        initial_error_category=ErrorCategory.TEST_FAILURE,
        initial_run_result=run_before,
        final_run_result=run_after,
    )
    agent_res = AgentRunResult(
        run_id="run-rich-1",
        target_file=str(src),
        test_file="tests/test_sample.py",
        stop_reason="threshold_reached",
        iterations_completed=1,
        total_llm_calls=2,
        total_tokens=500,
        runtime_sec=1.2,
        history=[
            AgentIterationLog(
                iteration=1,
                action="eval",
                metrics_before={"mutation_score_pct": 50.0},
                metrics_after={"test_count": 5, "line_coverage_pct": 100.0, "mutation_score_pct": 75.0, "weighted_score": 8.6},
                llm_calls=1,
                tokens_used=250,
                accepted=True,
                reason="Đạt ngưỡng",
            )
        ],
        final_report=report,
    )

    print_run_result(cons, run_after, project_root=tmp_path)
    print_fix_result(cons, fix_res)
    print_agent_summary(cons, agent_res, project_root=tmp_path, include_eval_dashboard=False)
    print_eval_dashboard(cons, report, project_root=tmp_path)
    rendered = buf.getvalue()
    assert "Passed: 3 -> 5" in rendered
    assert "items[0]" in rendered
    assert "Chi tiết 8 Tiêu chí Rubric" in rendered
    assert "(+25.00%)" in rendered

    # Smart TTY = True, không truyền --format -> KHÔNG xả JSON thô ra stdout, chỉ hiện Dashboard trên ui_console
    ui_buf = io.StringIO()
    ui_cons = Console(file=ui_buf, force_terminal=False, width=120, color_system=None)
    capsys.readouterr()
    deliver_report(
        report,
        ui_console=ui_cons,
        quiet=False,
        is_stdout_tty=True,
        project_root=tmp_path,
    )
    out_tty = capsys.readouterr()
    assert out_tty.out == ""
    assert "Chi tiết 8 Tiêu chí Rubric" in ui_buf.getvalue()
    assert "-o report.html" in ui_buf.getvalue()

    # Smart TTY = False (pipe/CI) -> xuất JSON chuẩn ra stdout
    html_out_path = tmp_path / "out_dash.html"
    deliver_report(
        report,
        output_file=html_out_path,
        ui_console=ui_cons,
        quiet=False,
        is_stdout_tty=False,
        project_root=tmp_path,
    )
    out_pipe = capsys.readouterr()
    assert EvalReport.model_validate_json(out_pipe.out).weighted_score == report.weighted_score
    assert html_out_path.is_file()
    assert "<!DOCTYPE html>" in html_out_path.read_text(encoding="utf-8")


def test_cli_eval_smart_tty_and_format_options(tmp_path: Path, monkeypatch) -> None:
    src = tmp_path / "calc_mini.py"
    src.write_text("def inc(x: int) -> int:\n    return x + 1\n", encoding="utf-8")
    sample_rep = _sample_report(src)

    monkeypatch.setattr("aitest.cli.evaluate_test_suite", lambda *a, **kw: sample_rep)
    runner = CliRunner()

    # Non-TTY với -q mặc định trả EvalReport JSON hợp lệ trên stdout
    r_json = runner.invoke(app, ["eval", str(src), "-q"])
    assert r_json.exit_code == 0
    assert EvalReport.model_validate_json(r_json.stdout).weighted_score == sample_rep.weighted_score

    # --format md xuất Markdown ra stdout
    r_md = runner.invoke(app, ["eval", str(src), "--format", "md", "-q"])
    assert r_md.exit_code == 0
    assert "# Báo cáo Đánh giá Unit Test" in r_md.stdout

    # -o report.html tự nhận diện đuôi .html và tạo file Standalone HTML
    html_file = tmp_path / "report.html"
    r_html = runner.invoke(app, ["eval", str(src), "-o", str(html_file), "-q"])
    assert r_html.exit_code == 0
    assert html_file.is_file()
    assert "<!DOCTYPE html>" in html_file.read_text(encoding="utf-8")

    # --format không hợp lệ -> exit code != 0
    r_bad = runner.invoke(app, ["eval", str(src), "--format", "xml"])
    assert r_bad.exit_code != 0


def test_format_python_test_layout_pep8_and_header(tmp_path: Path) -> None:
    messy_code = (
        "import math\n"
        "import pytest\n"
        "import math\n"
        "def test_one():\n"
        "    assert math.sqrt(4) == 2\n"
        "def test_two():\n"
        "    assert 1 + 1 == 2\n"
        "class TestGroup:\n"
        "    def test_three(self):\n"
        "        assert True\n"
    )
    formatted = format_python_test_layout(
        messy_code,
        header_comment="# Generated by aitest v4.0.0 | target: examples/calc.py | model: gemini-3.8-flash",
    )
    assert formatted.startswith("# Generated by aitest v4.0.0 | target: examples/calc.py")
    assert formatted.count("import math") == 1
    assert "\n\n\ndef test_two():" in formatted
    assert "\n\n\nclass TestGroup:" in formatted
    ast.parse(formatted)

    dest = tmp_path / "tests" / "test_clean.py"
    res = safe_merge_and_append(
        test_file=dest,
        new_code=messy_code,
        mode="overwrite",
        backup_dir=tmp_path / ".aitest_backup",
        log_dir=tmp_path / ".aitest_logs",
        header_comment="# Generated by aitest v4.0.0 | target: mod.py",
    )
    assert res.mode == "created"
    written = dest.read_text(encoding="utf-8")
    assert written.startswith("# Generated by aitest v4.0.0")
    assert "\n\n\ndef test_two():" in written
    ast.parse(written)
