"""Unit tests cho aitest.evaluator (Giai đoạn 3 - AC-09, AC-10, AC-11, AC-21, AC-23)."""

from __future__ import annotations

import json
from pathlib import Path

import httpx

from aitest.config import AitestConfig
from aitest.evaluator import evaluate_test_suite
from aitest.llm.client import OllamaClient
from aitest.schemas import MutationResult, RunResult, SurvivedMutant


def _make_mock_ollama_client(cfg: AitestConfig, correctness_score: float = 9.5) -> OllamaClient:
    assessment_payload = json.dumps(
        {
            "correctness": correctness_score,
            "readability": 9.0,
            "maintainability": 8.5,
            "isolation": 9.0,
            "robustness": 8.0,
            "determinism": 9.0,
            "issues": [],
            "summary": "Đánh giá thử nghiệm.",
        }
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(
                200,
                json={
                    "models": [
                        {
                            "name": cfg.model,
                            "digest": "sha256:1111222233334444555566667777888899990000aaaabbbbccccddddeeeeffff",
                        }
                    ]
                },
            )
        return httpx.Response(
            200,
            json={
                "model": cfg.model,
                "message": {"role": "assistant", "content": assessment_payload},
                "prompt_eval_count": 200,
                "eval_count": 80,
            },
        )

    return OllamaClient(cfg, transport=httpx.MockTransport(handler), backoff_base_sec=0.0)


def test_evaluator_applies_correctness_cap_when_mutation_available_ac11(tmp_path: Path) -> None:
    """Kiểm tra AC-11: correctness bị chặn trên bởi mutation_score_10 + 2.0 khi mutation_available == True."""
    src_file = tmp_path / "calc.py"
    src_file.write_text("def add(a: int, b: int) -> int:\n    return a + b\n", encoding="utf-8")
    test_file = tmp_path / "test_calc.py"
    test_file.write_text("from calc import add\n\ndef test_add():\n    assert add(1, 2) == 3\n", encoding="utf-8")

    # Giả lập mutation_score_pct = 50.0% (tức mutation_score_10 = 5.0), LLM chấm correctness = 9.5
    # Kỳ vọng: correctness_final = min(9.5, 5.0 + 2.0) = 7.0
    run_res = RunResult(
        target_file=str(src_file),
        test_file=str(test_file),
        exit_code=0,
        all_tests_passed=True,
        tests_passed=1,
        tests_failed=0,
        tests_skipped=0,
        pass_rate=1.0,
        duration_ms=100,
        line_coverage_pct=90.0,
        test_count=1,
        mutation=MutationResult(
            mutation_score_pct=50.0,
            mutation_killed=2,
            mutation_survived=2,
            mutation_excluded=0,
            mutation_status="available",
            mutation_available=True,
            survived_mutants=[SurvivedMutant(file="calc.py", line=2, mutation="+ -> -")],
        ),
    )

    cfg = AitestConfig()
    client = _make_mock_ollama_client(cfg, correctness_score=9.5)
    report = evaluate_test_suite(
        target_file=src_file,
        config=cfg,
        test_file=test_file,
        run_result=run_res,
        llm_client=client,
        project_root=tmp_path,
    )

    assert report.correctness_cap_applied is True
    assert report.llm_assessment.correctness == 9.5
    assert report.scores["mutation"] == 5.0
    assert report.scores["correctness"] == 7.0  # Đã bị chặn trên bởi 5.0 + 2.0
    assert report.report_comparable is True
    assert report.non_comparable_reason is None


def test_evaluator_when_mutation_disabled_marks_non_comparable_and_renormalizes_ac21_ac23(
    tmp_path: Path,
) -> None:
    """Kiểm tra AC-21 & AC-23: Khi thiếu mutation -> mutation_score_pct = None, chuẩn hóa weights_effective, report_comparable = False."""
    src_file = tmp_path / "calc.py"
    src_file.write_text("def add(a: int, b: int) -> int:\n    return a + b\n", encoding="utf-8")
    test_file = tmp_path / "test_calc.py"
    test_file.write_text("from calc import add\n\ndef test_add():\n    assert add(1, 2) == 3\n", encoding="utf-8")

    run_res = RunResult(
        target_file=str(src_file),
        test_file=str(test_file),
        exit_code=0,
        all_tests_passed=True,
        tests_passed=1,
        pass_rate=1.0,
        duration_ms=80,
        line_coverage_pct=100.0,
        test_count=1,
        mutation=MutationResult(
            mutation_score_pct=None,
            mutation_status="disabled",
            mutation_available=False,
            mutation_unavailable_reason="Tắt bằng cờ --no-mutation",
        ),
    )

    cfg = AitestConfig()
    client = _make_mock_ollama_client(cfg, correctness_score=9.5)
    report = evaluate_test_suite(
        target_file=src_file,
        config=cfg,
        test_file=test_file,
        run_result=run_res,
        llm_client=client,
        project_root=tmp_path,
    )

    assert report.objective_metrics.mutation_score_pct is None
    assert report.scores["mutation"] is None
    assert report.correctness_cap_applied is False
    assert report.scores["correctness"] == 9.5
    assert report.weights_effective["mutation"] == 0.0
    assert abs(sum(report.weights_effective.values()) - 1.0) <= 1e-6
    assert report.report_comparable is False
    assert report.non_comparable_reason is not None
