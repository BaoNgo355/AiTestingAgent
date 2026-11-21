"""Đánh giá chất lượng Unit Test theo mô hình 2 tầng: Chỉ số khách quan + LLM Rubric (SRS v2.3.0 §7 FR-05, §8, AC-09..AC-11, AC-18..AC-23)."""

from __future__ import annotations

from pathlib import Path

from aitest import __version__
from aitest.config import AitestConfig
from aitest.context import build_eval_prompts
from aitest.llm.client import OllamaClient
from aitest.progress import NullProgressReporter, ProgressReporter
from aitest.runner import run_tests
from aitest.schemas import EvalReport, Meta, RunResult, build_eval_report
from aitest.utils import (
    compute_config_hash,
    discover_project_root,
    generate_run_id,
    resolve_test_file_path,
    utc_now_iso,
    validate_single_python_file,
)


def _display_rel_path(path: Path | str, root: Path) -> str:
    p = Path(path)
    try:
        return p.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return p.as_posix()


def evaluate_test_suite(
    target_file: Path | str,
    config: AitestConfig,
    *,
    test_file: Path | str | None = None,
    run_result: RunResult | None = None,
    llm_client: OllamaClient | None = None,
    run_id: str | None = None,
    enable_mutation: bool | None = None,
    project_root: Path | str | None = None,
    reporter: ProgressReporter | None = None,
) -> EvalReport:
    """Thực thi đánh giá 2 tầng và trả về `EvalReport` chuẩn hóa 100% qua Pydantic (FR-05).

    1. Nếu `run_result` chưa được truyền vào -> gọi `run_tests` để đo chỉ số khách quan
       (pytest pass/fail, line coverage, mutation score & survived mutants).
    2. Gọi LLM qua `generate_structured_assessment` để lấy 6 điểm chủ quan (`LLMAssessment`).
    3. Áp dụng trần `correctness <= mutation_score_10 + correctness_cap_offset` khi `mutation_available == True`.
    4. Chuẩn hóa `weights_effective`, tính `weighted_score` từ điểm gốc chưa làm tròn và xác định `report_comparable`.
    """
    rep = reporter or NullProgressReporter()
    validated_target = validate_single_python_file(target_file)
    root_dir = (
        Path(project_root).resolve()
        if project_root
        else discover_project_root(validated_target)
    )
    resolved_test = (
        Path(test_file)
        if test_file is not None
        else resolve_test_file_path(validated_target, config.paths.tests_dir)
    )
    if not resolved_test.is_absolute():
        resolved_test = root_dir / resolved_test

    if not resolved_test.is_file():
        raise FileNotFoundError(
            f"Không tìm thấy file test '{resolved_test}'. Hãy chạy `aitest gen {target_file}` trước."
        )

    effective_run_id = run_id or generate_run_id()

    if run_result is None:
        run_result = run_tests(
            target_file=validated_target,
            config=config,
            test_file=resolved_test,
            project_root=root_dir,
            enable_mutation=enable_mutation,
            reporter=rep,
        )

    source_code = validated_target.read_text(encoding="utf-8")
    test_code = resolved_test.read_text(encoding="utf-8")

    execution = run_result.to_execution()
    objective_metrics = run_result.to_objective_metrics()

    sys_prompt, usr_prompt = build_eval_prompts(
        target_file=_display_rel_path(validated_target, root_dir),
        test_file=_display_rel_path(resolved_test, root_dir),
        source_code=source_code,
        test_code=test_code,
        run_result=run_result,
        objective_metrics=objective_metrics,
    )

    owns_client = llm_client is None
    client = llm_client or OllamaClient(config)
    try:
        with rep.step(
            f"Đang gọi Ollama ({config.model}) chấm điểm Rubric 2 tầng cho '{resolved_test.name}'",
            done_message=f"Ollama ({config.model}) đã hoàn tất chấm điểm Rubric",
        ):
            llm_assessment, llm_resp = client.generate_structured_assessment(
                system_prompt=sys_prompt,
                user_prompt=usr_prompt,
                run_id=effective_run_id,
                log_dir=root_dir / config.output.log_dir,
            )
        rep.on_llm_call_end(
            task_name="eval",
            model=llm_resp.model_actual or config.model,
            prompt_tokens=llm_resp.prompt_tokens,
            completion_tokens=llm_resp.completion_tokens,
            latency_ms=llm_resp.latency_ms,
        )
    finally:
        if owns_client:
            client.close()

    meta = Meta(
        tool="aitest",
        version=__version__,
        generated_at=utc_now_iso(),
        target_file=_display_rel_path(validated_target, root_dir),
        test_file=_display_rel_path(resolved_test, root_dir),
        provider="ollama",
        base_url=config.base_url,
        model_requested=config.model,
        model_actual=llm_resp.model_actual,
        model_digest=llm_resp.model_digest,
        config_hash=compute_config_hash(config),
        run_id=effective_run_id,
    )

    return build_eval_report(
        meta=meta,
        execution=execution,
        objective_metrics=objective_metrics,
        llm_assessment=llm_assessment,
        weights=config.eval.rubric.to_dict(),
        threshold_pass=config.eval.threshold_pass,
        correctness_cap_offset=config.eval.correctness_cap_offset,
        require_mutation_for_comparable_report=config.eval.require_mutation_for_comparable_report,
    )
