"""Unit tests cho aitest.config, aitest.schemas và aitest.utils (Giai đoạn 1)."""

from __future__ import annotations

from pathlib import Path
import re

import pytest
from pydantic import ValidationError

from aitest.config import AitestConfig, RubricWeights, load_config
from aitest.schemas import (
    AssessmentIssue,
    ErrorCategory,
    EvalReport,
    Execution,
    LLMAssessment,
    Meta,
    ObjectiveMetrics,
    SurvivedMutant,
    build_eval_report,
)
from aitest.utils import (
    SINGLE_PY_FILE_ERROR_MSG,
    atomic_write_text,
    backup_file,
    compute_config_hash,
    restore_backup,
    sanitize_code_block,
    validate_single_python_file,
)


def test_config_priority_and_env_override(tmp_path: Path) -> None:
    """Kiểm tra thứ tự ưu tiên: CLI flags > AITEST_* / GEMINI_API_KEY > aitest.yaml > defaults."""
    yaml_file = tmp_path / "aitest.yaml"
    yaml_file.write_text(
        "model: gemini-3.8-pro\nllm:\n  vertex: false\n",
        encoding="utf-8",
    )

    env = {
        "GEMINI_API_KEY": "secret-key-abc",
        "AITEST_LLM__VERTEX": "true",
        "GOOGLE_CLOUD_PROJECT": "proj-env",
        "GOOGLE_CLOUD_LOCATION": "us-central1",
        "AITEST_EVAL__THRESHOLD_PASS": "8.5",
    }
    cli_overrides = {
        "model": "gemini-3.8-flash",
    }

    cfg = load_config(config_path=yaml_file, cli_overrides=cli_overrides, env=env)
    assert cfg.provider == "antigravity"
    assert cfg.model == "gemini-3.8-flash"  # từ CLI
    assert cfg.llm.vertex is True  # từ AITEST_LLM__VERTEX
    assert cfg.llm.api_key == "secret-key-abc"  # từ GEMINI_API_KEY
    assert cfg.llm.project == "proj-env"
    assert cfg.llm.location == "us-central1"
    assert cfg.eval.threshold_pass == 8.5  # từ AITEST_EVAL__THRESHOLD_PASS

    dumped = cfg.safe_dump()
    assert dumped["provider"] == "antigravity"
    assert "api_key" not in dumped["llm"]
    cfg_hash = compute_config_hash(cfg)
    assert cfg_hash.startswith("sha256:")


def test_rubric_weights_sum_validation() -> None:
    """Kiểm tra tổng trọng số rubric bằng 1.0 (sai số 1e-6)."""
    with pytest.raises(ValidationError):
        RubricWeights(correctness=0.50)  # tổng != 1.0


def test_eval_report_reference_example_ac19() -> None:
    """Kiểm tra AC-09, AC-10, AC-11, AC-19 khớp ví dụ chuẩn trong SRS §8.2 (8.4175 -> 8.42)."""
    meta = Meta(
        version="4.0.0",
        generated_at="2026-11-28T14:00:00Z",
        target_file="examples/calculator.py",
        test_file="tests/test_calculator.py",
        model_requested="gemini-3.8-flash",
        model_actual="gemini-3.8-flash",
        judge_model_requested="gemini-3.8-flash",
        judge_model_actual="gemini-3.8-flash",
        conversation_id="conv-ac19",
        config_hash="sha256:8f9d2a1b3c4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a",
        run_id="550e8400-e29b-41d4-a716-446655440000",
    )
    execution = Execution(
        exit_code=0,
        all_tests_passed=True,
        tests_passed=5,
        tests_failed=0,
        tests_skipped=0,
        pass_rate=1.0,
        duration_ms=1234,
    )
    objective = ObjectiveMetrics(
        line_coverage_pct=92.5,
        branch_coverage_pct=None,
        mutation_score_pct=78.0,
        mutation_killed=39,
        mutation_survived=11,
        mutation_excluded=0,
        mutation_status="available",
        mutation_available=True,
        mutation_unavailable_reason=None,
        test_count=5,
        survived_mutants=[
            SurvivedMutant(file="examples/calculator.py", line=12, mutation=">= -> >")
        ],
    )
    llm_assessment = LLMAssessment(
        correctness=8.0,
        readability=9.0,
        maintainability=8.5,
        isolation=8.5,
        robustness=8.0,
        determinism=9.0,
        issues=[
            AssessmentIssue(
                title="Thiếu kiểm tra giá trị biên",
                description="Hàm điều kiện tại dòng 12 dùng toán tử >= nhưng chưa có test tại đúng mốc biên.",
                suggestion="Thêm test case kiểm tra chính xác giá trị tại điểm biên.",
            )
        ],
        summary="Bộ test bao phủ tốt các luồng chính nhưng còn thiếu kiểm tra điều kiện biên tại dòng 12.",
    )
    weights = AitestConfig().eval.rubric.to_dict()

    report = build_eval_report(
        meta=meta,
        execution=execution,
        objective_metrics=objective,
        llm_assessment=llm_assessment,
        weights=weights,
        threshold_pass=8.0,
        correctness_cap_offset=2.0,
    )
    assert report.weighted_score == 8.42
    assert report.passed_eval is True
    assert report.report_comparable is True
    assert report.non_comparable_reason is None
    assert report.correctness_cap_applied is True
    assert report.scores["coverage"] == 9.25
    assert report.scores["mutation"] == 7.8

    # Kiểm tra validate JSON round-trip
    json_str = report.model_dump_json()
    reloaded = EvalReport.model_validate_json(json_str)
    assert reloaded.weighted_score == 8.42


def test_correctness_cap_and_no_valid_mutants_ac11_ac20_ac21_ac22_ac23() -> None:
    """Kiểm tra AC-11 (cap correctness), AC-20 (pass_rate null), AC-21/22/23 (no_valid_mutants & renormalize)."""
    # AC-20: 0 test chạy -> pass_rate = None, all_tests_passed = False
    exec_zero = Execution(
        exit_code=2,
        all_tests_passed=False,
        tests_passed=0,
        tests_failed=0,
        tests_skipped=0,
        pass_rate=None,
        duration_ms=100,
    )
    assert exec_zero.pass_rate is None
    with pytest.raises(ValidationError):
        Execution(
            exit_code=0,
            all_tests_passed=True,
            tests_passed=0,
            tests_failed=0,
            tests_skipped=0,
            pass_rate=1.0,
            duration_ms=100,
        )

    # AC-22 & AC-23: mutation_status = no_valid_mutants
    meta = Meta(
        version="4.0.0",
        generated_at="2026-11-28T14:00:00Z",
        target_file="examples/empty.py",
        test_file="tests/test_empty.py",
        model_requested="gemini-3.8-flash",
        config_hash="sha256:abc",
        run_id="run-123",
    )
    execution = Execution(
        exit_code=0,
        all_tests_passed=True,
        tests_passed=2,
        tests_failed=0,
        tests_skipped=0,
        pass_rate=1.0,
        duration_ms=50,
    )
    objective_no_mut = ObjectiveMetrics(
        line_coverage_pct=100.0,
        mutation_score_pct=None,
        mutation_killed=0,
        mutation_survived=0,
        mutation_excluded=0,
        mutation_status="no_valid_mutants",
        mutation_available=False,
        mutation_unavailable_reason="Không sinh được mutant hợp lệ nào",
        test_count=2,
    )
    llm = LLMAssessment(
        correctness=9.5,
        readability=9.0,
        maintainability=9.0,
        isolation=9.0,
        robustness=9.0,
        determinism=9.0,
        summary="Tốt",
    )
    weights = AitestConfig().eval.rubric.to_dict()
    report = build_eval_report(
        meta=meta,
        execution=execution,
        objective_metrics=objective_no_mut,
        llm_assessment=llm,
        weights=weights,
    )
    assert report.scores["mutation"] is None
    assert report.weights_effective["mutation"] == 0.0
    assert abs(sum(report.weights_effective.values()) - 1.0) <= 1e-6
    assert report.correctness_cap_applied is False
    assert report.scores["correctness"] == 9.5
    assert report.report_comparable is False
    assert report.non_comparable_reason is not None


def test_utils_sanitize_backup_atomic_and_single_py_ac02_ac04(tmp_path: Path) -> None:
    """Kiểm tra AC-02 (chỉ nhận 1 file .py), AC-04 (sanitize markdown fence), backup & atomic write."""
    raw_llm = "Dưới đây là code test:\n```python\nimport pytest\n\ndef test_add():\n    assert 1 + 1 == 2\n```\nChúc thành công!"
    cleaned = sanitize_code_block(raw_llm)
    assert "```" not in cleaned
    assert "def test_add():" in cleaned

    # Atomic write với kiểm tra AST
    test_file = tmp_path / "tests" / "test_sample.py"
    atomic_write_text(test_file, cleaned, verify_python_ast=True)
    assert test_file.is_file()

    # Backup & restore
    bak = backup_file(test_file, backup_root=tmp_path / ".aitest_backup", run_id="run1")
    assert bak is not None and bak.is_file()

    atomic_write_text(test_file, "def test_modified():\n    assert True\n", verify_python_ast=True)
    restore_backup(bak, test_file)
    assert "def test_add():" in test_file.read_text(encoding="utf-8")

    # AC-02: Từ chối thư mục hoặc file không phải .py
    with pytest.raises(ValueError, match=re.escape(SINGLE_PY_FILE_ERROR_MSG)):
        validate_single_python_file(tmp_path)

    txt_file = tmp_path / "note.txt"
    txt_file.write_text("hello", encoding="utf-8")
    with pytest.raises(ValueError, match=re.escape(SINGLE_PY_FILE_ERROR_MSG)):
        validate_single_python_file(txt_file)

    assert ErrorCategory.TEST_FAILURE.is_fixable_by_llm is True
    assert ErrorCategory.ENVIRONMENT_ERROR.is_fixable_by_llm is False
