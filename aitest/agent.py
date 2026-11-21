"""Agent Orchestrator: Điều phối vòng lặp tự động gen -> run -> fix -> mutation -> eval -> improve và chế độ --baseline (SRS v2.3.0 §6.2, §6.3, §7 FR-06, §12.5, AC-13, AC-24)."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from aitest.analyzer import analyze_test_code
from aitest.config import AitestConfig
from aitest.context import build_improve_prompts
from aitest.evaluator import evaluate_test_suite
from aitest.fixer import fix_test_file
from aitest.generator import SafeAppendError, generate_tests, safe_merge_and_append
from aitest.llm.client import OllamaClient
from aitest.runner import run_tests
from aitest.schemas import AgentIterationLog, EvalReport, RunResult
from aitest.utils import (
    atomic_write_text,
    backup_file,
    generate_run_id,
    resolve_test_file_path,
    restore_backup,
    validate_single_python_file,
)


class AgentRunResult(BaseModel):
    """Kết quả tổng hợp của một phiên chạy `aitest agent` (hoặc `aitest agent --baseline`)."""

    run_id: str
    target_file: str
    test_file: str
    baseline_mode: bool = False
    stop_reason: str
    iterations_completed: int = 0
    total_llm_calls: int = 0
    total_tokens: int = 0
    runtime_sec: float = 0.0
    quality_guard_rejections: int = 0
    history_file: str | None = None
    history: list[AgentIterationLog] = Field(default_factory=list)
    final_report: EvalReport


def _snapshot_metrics(run_res: RunResult | None, eval_rep: EvalReport | None = None) -> dict[str, Any]:
    if run_res is None:
        return {}
    mut_pct = (
        run_res.mutation.mutation_score_pct
        if (run_res.mutation and run_res.mutation.mutation_available)
        else None
    )
    snap: dict[str, Any] = {
        "all_tests_passed": run_res.all_tests_passed,
        "tests_passed": run_res.tests_passed,
        "tests_failed": run_res.tests_failed,
        "test_count": run_res.test_count,
        "line_coverage_pct": run_res.line_coverage_pct,
        "mutation_score_pct": mut_pct,
    }
    if eval_rep is not None:
        snap["weighted_score"] = eval_rep.weighted_score
        snap["passed_eval"] = eval_rep.passed_eval
    return snap


def run_agent_loop(
    target_file: Path | str,
    config: AitestConfig,
    *,
    test_file: Path | str | None = None,
    max_iters: int | None = None,
    threshold_pass: float | None = None,
    max_llm_calls: int | None = None,
    max_total_tokens: int | None = None,
    max_runtime_sec: int | None = None,
    baseline: bool = False,
    llm_client: OllamaClient | None = None,
    run_id: str | None = None,
    project_root: Path | str | None = None,
) -> AgentRunResult:
    """Thực thi vòng lặp Agent tự động theo đúng quy trình FR-06 & thứ tự ưu tiên điều kiện dừng §6.3.

    Thứ tự ưu tiên điều kiện dừng (§6.3, AC-13):
    1. Lỗi không thể sửa bằng test (`ENVIRONMENT_ERROR`, `SOURCE_ERROR`, `TIMEOUT`, `UNKNOWN_ERROR`, `COLLECTION_ERROR`).
    2. Vượt ngân sách tài nguyên (`max_llm_calls`, `max_total_tokens`, `max_runtime_sec`).
    3. Đạt điều kiện thành công (`all_tests_passed == True` và `weighted_score >= threshold_pass` và mutation score không giảm).
    4. Đạt số vòng tối đa (`max_iters`).
    5. Không còn cải thiện sau `early_stop_patience` vòng liên tiếp (early stop) hoặc đạt 100% mutation & coverage.
    """
    validated_target = validate_single_python_file(target_file)
    root_dir = Path(project_root).resolve() if project_root else Path.cwd().resolve()
    resolved_test = (
        Path(test_file)
        if test_file is not None
        else resolve_test_file_path(validated_target, config.paths.tests_dir)
    )
    if not resolved_test.is_absolute():
        resolved_test = root_dir / resolved_test

    effective_run_id = run_id or generate_run_id()
    eff_max_iters = max_iters if max_iters is not None else config.agent.max_iters
    eff_threshold = threshold_pass if threshold_pass is not None else config.eval.threshold_pass
    eff_max_calls = max_llm_calls if max_llm_calls is not None else config.agent.max_llm_calls
    eff_max_tokens = (
        max_total_tokens if max_total_tokens is not None else config.agent.max_total_tokens
    )
    eff_max_runtime = (
        max_runtime_sec if max_runtime_sec is not None else config.agent.max_runtime_sec
    )
    patience = config.agent.early_stop_patience

    # Đồng bộ threshold_pass vào bản sao config cho evaluator
    work_config = config.model_copy(deep=True)
    work_config.eval.threshold_pass = eff_threshold

    start_ns = time.perf_counter_ns()
    owns_client = llm_client is None
    client = llm_client or OllamaClient(work_config)

    history: list[AgentIterationLog] = []
    stop_reason = "max_iters_reached"
    guard_rejections = 0
    last_eval_report: EvalReport | None = None
    last_run_result: RunResult | None = None

    def _elapsed_sec() -> float:
        return (time.perf_counter_ns() - start_ns) / 1_000_000_000

    def _budget_exceeded_reason() -> str | None:
        if client.total_calls >= eff_max_calls:
            return f"budget_exceeded:max_llm_calls({client.total_calls}/{eff_max_calls})"
        if client.total_tokens >= eff_max_tokens:
            return f"budget_exceeded:max_total_tokens({client.total_tokens}/{eff_max_tokens})"
        if _elapsed_sec() >= float(eff_max_runtime):
            return f"budget_exceeded:max_runtime_sec({_elapsed_sec():.1f}s/{eff_max_runtime}s)"
        return None

    try:
        # Bước 2: Nếu chưa có file test -> gọi bước gen (FR-02) để sinh bộ test ban đầu
        if not resolved_test.is_file():
            calls_before = client.total_calls
            tokens_before = client.total_tokens
            gen_res = generate_tests(
                target_file=validated_target,
                config=work_config,
                mode="overwrite",
                test_file=resolved_test,
                llm_client=client,
                run_id=effective_run_id,
                project_root=root_dir,
            )
            history.append(
                AgentIterationLog(
                    iteration=0,
                    action="gen",
                    metrics_before={"test_count": 0},
                    metrics_after={"test_count": len(gen_res.added_tests)},
                    llm_calls=client.total_calls - calls_before,
                    tokens_used=client.total_tokens - tokens_before,
                    accepted=True,
                    reason=f"Sinh bộ test khởi đầu ({len(gen_res.added_tests)} hàm test)",
                )
            )

        best_weighted_score: float = -1.0
        best_mutation_score: float = -1.0
        prev_mutation_score: float | None = None
        no_improve_streak: int = 0
        iterations_completed = 0

        # Bước 3: Vòng lặp Agent (iteration = 1 .. eff_max_iters)
        for iteration in range(1, eff_max_iters + 1):
            iterations_completed = iteration

            # 3a. Chạy run + coverage (+ mutation nếu test pass)
            curr_run = run_tests(
                target_file=validated_target,
                config=work_config,
                test_file=resolved_test,
                project_root=root_dir,
                enable_coverage=work_config.coverage.enabled,
                enable_mutation=work_config.mutation.enabled,
            )
            last_run_result = curr_run

            # 3b. Nếu all_tests_passed == False -> Phân loại lỗi & gọi Fix có kiểm soát
            if not curr_run.all_tests_passed:
                # Ưu tiên dừng (1): Lỗi không thể sửa bằng test
                if not curr_run.error_category.is_fixable_by_llm:
                    stop_reason = f"unfixable_error:{curr_run.error_category.value}"
                    history.append(
                        AgentIterationLog(
                            iteration=iteration,
                            action="fix",
                            metrics_before=_snapshot_metrics(curr_run),
                            metrics_after=_snapshot_metrics(curr_run),
                            llm_calls=0,
                            tokens_used=0,
                            accepted=False,
                            reason=f"Dừng ngay do gặp lỗi không được phép sửa bằng test: {stop_reason}",
                        )
                    )
                    break

                # Ưu tiên dừng (2): Kiểm tra ngân sách trước khi gọi fix
                b_reason = _budget_exceeded_reason()
                if b_reason:
                    stop_reason = b_reason
                    history.append(
                        AgentIterationLog(
                            iteration=iteration,
                            action="fix",
                            metrics_before=_snapshot_metrics(curr_run),
                            metrics_after=_snapshot_metrics(curr_run),
                            llm_calls=0,
                            tokens_used=0,
                            accepted=False,
                            reason=f"Dừng trước khi fix do vượt ngân sách: {stop_reason}",
                        )
                    )
                    break

                calls_before = client.total_calls
                tokens_before = client.total_tokens
                metrics_before_fix = _snapshot_metrics(curr_run)

                remaining_call_budget = max(1, eff_max_calls - client.total_calls)
                fix_iters = min(work_config.fix.max_iters, remaining_call_budget)

                fix_res = fix_test_file(
                    target_file=validated_target,
                    config=work_config,
                    test_file=resolved_test,
                    max_iters=fix_iters,
                    initial_run_result=curr_run,
                    baseline_mutation_score=prev_mutation_score,
                    llm_client=client,
                    run_id=effective_run_id,
                    project_root=root_dir,
                )
                guard_rejections += sum(1 for r in fix_res.records if not r.accepted)
                curr_run = fix_res.final_run_result
                last_run_result = curr_run

                history.append(
                    AgentIterationLog(
                        iteration=iteration,
                        action="fix",
                        metrics_before=metrics_before_fix,
                        metrics_after=_snapshot_metrics(curr_run),
                        llm_calls=client.total_calls - calls_before,
                        tokens_used=client.total_tokens - tokens_before,
                        accepted=fix_res.fixed,
                        reason=f"Fix kết thúc với trạng thái: {fix_res.stop_reason}",
                    )
                )

                if not curr_run.all_tests_passed:
                    if not curr_run.error_category.is_fixable_by_llm:
                        stop_reason = f"unfixable_error:{curr_run.error_category.value}"
                    else:
                        stop_reason = f"fix_failed:{fix_res.stop_reason}"
                    break

            # 3c. Khi all_tests_passed == True -> Đánh giá 2 tầng (eval)
            calls_before_eval = client.total_calls
            tokens_before_eval = client.total_tokens
            eval_report = evaluate_test_suite(
                target_file=validated_target,
                config=work_config,
                test_file=resolved_test,
                run_result=curr_run,
                llm_client=client,
                run_id=effective_run_id,
                project_root=root_dir,
            )
            last_eval_report = eval_report

            curr_mut_pct = eval_report.objective_metrics.mutation_score_pct
            curr_cov_pct = eval_report.objective_metrics.line_coverage_pct
            curr_weighted = eval_report.weighted_score

            # Kiểm tra mức cải thiện để tính early_stop_patience
            improved = False
            if curr_weighted > best_weighted_score + 1e-6:
                best_weighted_score = curr_weighted
                improved = True
            if curr_mut_pct is not None and curr_mut_pct > best_mutation_score + 1e-6:
                best_mutation_score = curr_mut_pct
                improved = True

            if iteration == 1 or improved:
                no_improve_streak = 0
            else:
                no_improve_streak += 1

            mutation_did_not_drop = (
                prev_mutation_score is None
                or curr_mut_pct is None
                or curr_mut_pct >= prev_mutation_score - 1e-6
            )
            if curr_mut_pct is not None:
                prev_mutation_score = max(prev_mutation_score or 0.0, curr_mut_pct)

            history.append(
                AgentIterationLog(
                    iteration=iteration,
                    action="eval",
                    metrics_before=_snapshot_metrics(curr_run),
                    metrics_after=_snapshot_metrics(curr_run, eval_report),
                    llm_calls=client.total_calls - calls_before_eval,
                    tokens_used=client.total_tokens - tokens_before_eval,
                    accepted=True,
                    reason=(
                        f"weighted_score={curr_weighted:.2f} (threshold={eff_threshold:.2f}), "
                        f"coverage={curr_cov_pct}%, mutation={curr_mut_pct}%"
                    ),
                )
            )

            # Nếu chạy chế độ --baseline (Baseline A, §6.2 & §16.2, AC-24) -> Dừng ngay sau lần eval đầu tiên
            if baseline:
                stop_reason = "baseline_completed"
                break

            # Kiểm tra điều kiện dừng theo ĐÚNG thứ tự ưu tiên ở SRS §6.3 (AC-13):
            # (1) Lỗi không sửa được (đã chặn ở 3b)
            # (2) Vượt ngân sách tài nguyên
            b_reason = _budget_exceeded_reason()
            if b_reason:
                stop_reason = b_reason
                break

            # (3) Đạt điều kiện thành công: all_tests_passed và weighted_score >= threshold_pass và mutation không giảm
            if (
                eval_report.execution.all_tests_passed
                and curr_weighted >= eff_threshold
                and mutation_did_not_drop
            ):
                stop_reason = "threshold_reached"
                break

            if curr_mut_pct == 100.0 and (curr_cov_pct is None or curr_cov_pct == 100.0):
                stop_reason = "max_objective_metrics_reached"
                break

            # (4) Đạt số vòng tối đa (max_iters)
            if iteration >= eff_max_iters:
                stop_reason = "max_iters_reached"
                break

            # (5) Không còn cải thiện (Early stop) hoặc không còn survived mutant / missing line nào để cải thiện
            survived_list = eval_report.objective_metrics.survived_mutants
            if no_improve_streak >= patience:
                stop_reason = f"early_stop_no_improvement(streak={no_improve_streak})"
                break

            if not survived_list and not curr_run.missing_lines:
                stop_reason = "early_stop_no_remaining_targets"
                break

            # Pha Improve: Bổ sung hàm test_* mới nhắm trực tiếp vào survived_mutants & missing_lines
            metrics_before_improve = _snapshot_metrics(curr_run, eval_report)
            bak_improve = backup_file(
                resolved_test,
                backup_root=root_dir / work_config.output.backup_dir,
                run_id=effective_run_id,
                subdir=f"improve_iter_{iteration}",
            )

            source_code = validated_target.read_text(encoding="utf-8")
            current_test_code = resolved_test.read_text(encoding="utf-8")
            try:
                existing_names = analyze_test_code(current_test_code, file_path=resolved_test).test_names
            except SyntaxError:
                existing_names = []

            sys_imp, usr_imp = build_improve_prompts(
                target_file=validated_target,
                source_code=source_code,
                existing_test_names=existing_names,
                survived_mutants=survived_list,
                missing_lines=curr_run.missing_lines,
                project_root=root_dir,
            )

            calls_before_imp = client.total_calls
            tokens_before_imp = client.total_tokens
            imp_resp = client.chat(system_prompt=sys_imp, user_prompt=usr_imp)

            try:
                append_res = safe_merge_and_append(
                    test_file=resolved_test,
                    new_code=imp_resp.content,
                    mode="append",
                    backup_dir=root_dir / work_config.output.backup_dir,
                    log_dir=root_dir / work_config.output.log_dir,
                    run_id=effective_run_id,
                )
            except SafeAppendError as exc:
                guard_rejections += 1
                if bak_improve:
                    restore_backup(bak_improve, resolved_test)
                history.append(
                    AgentIterationLog(
                        iteration=iteration,
                        action="improve",
                        metrics_before=metrics_before_improve,
                        metrics_after=metrics_before_improve,
                        llm_calls=client.total_calls - calls_before_imp,
                        tokens_used=client.total_tokens - tokens_before_imp,
                        accepted=False,
                        reason=f"Safe Append từ chối đoạn test improve: {exc}",
                    )
                )
                continue

            # Chạy thử file test sau khi append để đảm bảo không làm hỏng bộ test hoặc làm giảm mutation score
            verify_run = run_tests(
                target_file=validated_target,
                config=work_config,
                test_file=resolved_test,
                project_root=root_dir,
                enable_coverage=work_config.coverage.enabled,
                enable_mutation=work_config.mutation.enabled,
            )
            verify_mut_pct = (
                verify_run.mutation.mutation_score_pct
                if (verify_run.mutation and verify_run.mutation.mutation_available)
                else None
            )

            mut_dropped = (
                curr_mut_pct is not None
                and verify_mut_pct is not None
                and verify_mut_pct < curr_mut_pct - 1e-6
            )

            if not verify_run.all_tests_passed or mut_dropped:
                guard_rejections += 1
                if bak_improve:
                    restore_backup(bak_improve, resolved_test)
                rb_reason = (
                    "Bộ test bị fail sau khi append hàm test mới"
                    if not verify_run.all_tests_passed
                    else f"Mutation score bị giảm ({verify_mut_pct}% < {curr_mut_pct}%)"
                )
                history.append(
                    AgentIterationLog(
                        iteration=iteration,
                        action="improve",
                        metrics_before=metrics_before_improve,
                        metrics_after=metrics_before_improve,
                        llm_calls=client.total_calls - calls_before_imp,
                        tokens_used=client.total_tokens - tokens_before_imp,
                        accepted=False,
                        reason=f"Đã rollback pha improve: {rb_reason}",
                    )
                )
            else:
                last_run_result = verify_run
                history.append(
                    AgentIterationLog(
                        iteration=iteration,
                        action="improve",
                        metrics_before=metrics_before_improve,
                        metrics_after=_snapshot_metrics(verify_run),
                        llm_calls=client.total_calls - calls_before_imp,
                        tokens_used=client.total_tokens - tokens_before_imp,
                        accepted=True,
                        reason=f"Đã bổ sung {len(append_res.added_tests)} hàm test mới qua Safe Append",
                    )
                )

        # Nếu dừng sớm trước khi có last_eval_report (ví dụ dừng ở 3b do lỗi không sửa được),
        # vẫn sinh EvalReport trạng thái cuối để trả về báo cáo chuẩn
        if last_eval_report is None:
            final_run = last_run_result or run_tests(
                target_file=validated_target,
                config=work_config,
                test_file=resolved_test,
                project_root=root_dir,
                enable_mutation=False,
            )
            last_eval_report = evaluate_test_suite(
                target_file=validated_target,
                config=work_config,
                test_file=resolved_test,
                run_result=final_run,
                llm_client=client,
                run_id=effective_run_id,
                project_root=root_dir,
            )

        # Ghi lý do dừng cuối cùng vào phần tử cuối của history[] (§6.3)
        if history:
            history[-1].reason = f"{history[-1].reason} | [stop_reason={stop_reason}]"

        runtime_sec = round(_elapsed_sec(), 3)
        log_dir_path = root_dir / work_config.output.log_dir
        log_dir_path.mkdir(parents=True, exist_ok=True)
        history_file_path = log_dir_path / f"agent_history_{effective_run_id}.json"

        history_payload = {
            "run_id": effective_run_id,
            "target_file": str(target_file),
            "test_file": str(resolved_test),
            "baseline_mode": baseline,
            "stop_reason": stop_reason,
            "iterations_completed": iterations_completed,
            "total_llm_calls": client.total_calls,
            "total_tokens": client.total_tokens,
            "runtime_sec": runtime_sec,
            "quality_guard_rejections": guard_rejections,
            "history": [item.model_dump() for item in history],
        }
        atomic_write_text(
            history_file_path,
            json.dumps(history_payload, indent=2, ensure_ascii=False) + "\n",
        )

        return AgentRunResult(
            run_id=effective_run_id,
            target_file=str(target_file),
            test_file=str(resolved_test),
            baseline_mode=baseline,
            stop_reason=stop_reason,
            iterations_completed=iterations_completed,
            total_llm_calls=client.total_calls,
            total_tokens=client.total_tokens,
            runtime_sec=runtime_sec,
            quality_guard_rejections=guard_rejections,
            history_file=str(history_file_path),
            history=history,
            final_report=last_eval_report,
        )
    finally:
        if owns_client:
            client.close()
