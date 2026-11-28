"""Unit tests cho aitest.schemas (Giai đoạn 5 - AC-09, AC-10, AC-11, AC-19, AC-20, AC-21, AC-22, AC-23)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from aitest.config import AitestConfig
from aitest.schemas import (
    EvalReport,
    Execution,
    LLMAssessment,
    Meta,
    ObjectiveMetrics,
    build_eval_report,
)


def _sample_meta() -> Meta:
    return Meta(
        version="4.0.0",
        generated_at="2026-11-28T10:00:00Z",
        target_file="examples/calculator.py",
        test_file="tests/test_calculator.py",
        model_requested="gemini-4-hydrogen",
        model_actual="gemini-4-hydrogen",
        judge_model_requested="gemini-4-hydrogen",
        judge_model_actual="gemini-4-hydrogen",
        vertex=False,
        conversation_id="conv-schema-test",
        config_hash="sha256:0123456789abcdef",
        run_id="run-schema-test",
    )


def test_eval_report_invariants_and_unrounded_weighted_score_ac09_ac19() -> None:
    """Kiểm tra AC-09, AC-10, AC-19: Tính weighted_score từ điểm gốc chưa làm tròn (8.4175 -> 8.42)."""
    execution = Execution(
        exit_code=0,
        all_tests_passed=True,
        tests_passed=10,
        tests_failed=0,
        tests_skipped=0,
        pass_rate=1.0,
        duration_ms=250,
    )
    objective = ObjectiveMetrics(
        line_coverage_pct=92.5,
        mutation_score_pct=78.0,
        mutation_killed=39,
        mutation_survived=11,
        mutation_excluded=0,
        mutation_status="available",
        mutation_available=True,
        test_count=10,
    )
    llm = LLMAssessment(
        correctness=8.0,
        readability=9.0,
        maintainability=8.5,
        isolation=8.5,
        robustness=8.0,
        determinism=9.0,
        summary="Bộ test đạt yêu cầu.",
    )
    weights = AitestConfig().eval.rubric.to_dict()
    report = build_eval_report(
        meta=_sample_meta(),
        execution=execution,
        objective_metrics=objective,
        llm_assessment=llm,
        weights=weights,
        threshold_pass=8.0,
    )
    assert report.weighted_score == 8.42
    assert report.passed_eval is True
    assert report.report_comparable is True
    assert report.non_comparable_reason is None
    assert EvalReport.model_validate_json(report.model_dump_json()).weighted_score == 8.42


def test_schema_rejects_inconsistent_mutation_or_pass_rate_ac20_ac21_ac22() -> None:
    """Kiểm tra các @model_validator chặn dữ liệu mâu thuẫn trên Execution, ObjectiveMetrics và EvalReport."""
    # AC-20: 0 test chạy nhưng pass_rate != None -> lỗi
    with pytest.raises(ValidationError):
        Execution(
            exit_code=1,
            all_tests_passed=False,
            tests_passed=0,
            tests_failed=0,
            tests_skipped=0,
            pass_rate=0.0,
            duration_ms=10,
        )

    # AC-21: mutation_available=False nhưng có mutation_score_pct != None -> lỗi
    with pytest.raises(ValidationError):
        ObjectiveMetrics(
            line_coverage_pct=80.0,
            mutation_score_pct=50.0,
            mutation_status="disabled",
            mutation_available=False,
            test_count=2,
        )

    # AC-22: mutation_status="no_valid_mutants" nhưng killed + survived > 0 -> lỗi
    with pytest.raises(ValidationError):
        ObjectiveMetrics(
            line_coverage_pct=80.0,
            mutation_score_pct=None,
            mutation_killed=1,
            mutation_survived=0,
            mutation_status="no_valid_mutants",
            mutation_available=False,
            test_count=2,
        )
