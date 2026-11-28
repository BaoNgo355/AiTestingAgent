"""Chuẩn bị ngữ cảnh (Context Builder) cho 4 tác vụ LLM: gen, fix, improve, eval (SRS v2.3.0 §9.1)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from aitest.analyzer import SourceAnalysis, analyze_source_code, analyze_test_code
from aitest.llm.prompts import (
    render_eval_prompt,
    render_fix_prompt,
    render_gen_prompt,
    render_improve_prompt,
    render_targeted_fix_prompt,
)
from aitest.schemas import ErrorCategory, ObjectiveMetrics, RunResult, SurvivedMutant


def format_numbered_source(source_code: str) -> str:
    """Đánh số dòng cho mã nguồn (bắt đầu từ 1) để LLM đối chiếu chính xác dòng survived mutant."""
    lines = source_code.splitlines()
    width = max(3, len(str(len(lines))))
    return "\n".join(f"{idx:>{width}} | {line}" for idx, line in enumerate(lines, start=1))


def build_gen_prompts(
    *,
    target_file: Path | str,
    source_code: str,
    source_analysis: SourceAnalysis | None = None,
    existing_test_names: list[str] | None = None,
    project_root: Path | str | None = None,
) -> tuple[str, str]:
    """Tạo cặp `(system_prompt, user_prompt)` cho tác vụ sinh test (`gen`)."""
    analysis = source_analysis or analyze_source_code(
        source_code, file_path=target_file, project_root=project_root
    )
    return render_gen_prompt(
        target_file=str(target_file),
        module_name=analysis.module_name,
        source_code=source_code,
        ast_summary_lines=analysis.target_summary_lines(),
        existing_test_names=existing_test_names or [],
    )


def build_fix_prompts(
    *,
    target_file: Path | str,
    test_file: Path | str,
    source_code: str,
    current_test_code: str,
    pytest_output: str,
    error_category: ErrorCategory = ErrorCategory.TEST_FAILURE,
    project_root: Path | str | None = None,
) -> tuple[str, str]:
    """Tạo cặp `(system_prompt, user_prompt)` cho tác vụ sửa lỗi test (`fix`)."""
    src_analysis = analyze_source_code(
        source_code, file_path=target_file, project_root=project_root
    )
    existing_names: list[str] = []
    try:
        test_analysis = analyze_test_code(current_test_code, file_path=test_file)
        existing_names = test_analysis.test_names
    except SyntaxError:
        pass

    # Giới hạn độ dài traceback nếu quá dài để không tràn context window
    trimmed_output = pytest_output.strip()
    if len(trimmed_output) > 6000:
        trimmed_output = trimmed_output[:3000] + "\n...[truncated]...\n" + trimmed_output[-3000:]

    return render_fix_prompt(
        target_file=str(target_file),
        test_file=str(test_file),
        module_name=src_analysis.module_name,
        error_category=error_category.value,
        source_code=source_code,
        ast_summary_lines=src_analysis.target_summary_lines(),
        current_test_code=current_test_code,
        pytest_output=trimmed_output,
        existing_test_names=existing_names,
    )


def build_targeted_fix_prompts(
    *,
    target_file: Path | str,
    test_file: Path | str,
    source_code: str,
    header_context: str,
    failing_tests_code: str,
    failing_test_names: list[str],
    pytest_output: str,
    error_category: ErrorCategory = ErrorCategory.TEST_FAILURE,
    project_root: Path | str | None = None,
) -> tuple[str, str]:
    """Tạo cặp `(system_prompt, user_prompt)` cho tác vụ sửa khoanh vùng riêng các hàm test bị fail."""
    src_analysis = analyze_source_code(
        source_code, file_path=target_file, project_root=project_root
    )
    trimmed_output = pytest_output.strip()
    if len(trimmed_output) > 6000:
        trimmed_output = trimmed_output[:3000] + "\n...[truncated]...\n" + trimmed_output[-3000:]

    return render_targeted_fix_prompt(
        target_file=str(target_file),
        test_file=str(test_file),
        module_name=src_analysis.module_name,
        error_category=error_category.value,
        source_code=source_code,
        ast_summary_lines=src_analysis.target_summary_lines(),
        header_context=header_context.strip(),
        failing_tests_code=failing_tests_code.strip(),
        failing_test_names=failing_test_names,
        pytest_output=trimmed_output,
    )


def build_improve_prompts(
    *,
    target_file: Path | str,
    source_code: str,
    existing_test_names: list[str],
    survived_mutants: list[SurvivedMutant],
    missing_lines: list[int] | None = None,
    existing_fixtures: list[str] | None = None,
    existing_helpers: list[str] | None = None,
    project_root: Path | str | None = None,
) -> tuple[str, str]:
    """Tạo cặp `(system_prompt, user_prompt)` cho pha `improve` trong Agent Loop."""
    src_analysis = analyze_source_code(
        source_code, file_path=target_file, project_root=project_root
    )
    src_lines = source_code.splitlines()

    mutant_details: list[dict[str, Any]] = []
    for m in survived_mutants:
        line_idx = m.line - 1
        orig_line = src_lines[line_idx].strip() if 0 <= line_idx < len(src_lines) else ""
        mutant_details.append(
            {
                "line": m.line,
                "mutation": m.mutation,
                "source_line": orig_line,
            }
        )

    return render_improve_prompt(
        target_file=str(target_file),
        module_name=src_analysis.module_name,
        numbered_source_code=format_numbered_source(source_code),
        existing_test_names=existing_test_names,
        survived_mutants_details=mutant_details,
        missing_lines=sorted(set(missing_lines or [])),
        existing_fixtures=existing_fixtures or [],
        existing_helpers=existing_helpers or [],
    )


def build_eval_prompts(
    *,
    target_file: Path | str,
    test_file: Path | str,
    source_code: str,
    test_code: str,
    run_result: RunResult,
    objective_metrics: ObjectiveMetrics,
) -> tuple[str, str]:
    """Tạo cặp `(system_prompt, user_prompt)` cho bước chấm điểm chủ quan (`eval`)."""
    return render_eval_prompt(
        target_file=str(target_file),
        test_file=str(test_file),
        source_code=source_code,
        test_code=test_code,
        all_tests_passed=run_result.all_tests_passed,
        tests_passed=run_result.tests_passed,
        tests_failed=run_result.tests_failed,
        tests_skipped=run_result.tests_skipped,
        test_count=objective_metrics.test_count,
        line_coverage_pct=objective_metrics.line_coverage_pct,
        mutation_status=objective_metrics.mutation_status,
        mutation_score_pct=objective_metrics.mutation_score_pct,
        survived_mutants_count=len(objective_metrics.survived_mutants),
        survived_mutants_preview=objective_metrics.survived_mutants[:15],
    )
