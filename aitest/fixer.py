"""Sửa test tự động có kiểm soát (Fixer) kèm Quality Guard và Rollback (SRS v2.3.0 §7 FR-04, §12.3, AC-06..AC-08)."""

from __future__ import annotations

import ast
import difflib
from pathlib import Path

from pydantic import BaseModel, Field

from aitest.config import AitestConfig
from aitest.context import build_fix_prompts
from aitest.llm.client import OllamaClient
from aitest.progress import NullProgressReporter, ProgressReporter
from aitest.quality_guard import check_quality_guard
from aitest.runner import run_tests
from aitest.schemas import AssertionDiffEntry, ErrorCategory, GuardResult, RunResult
from aitest.utils import (
    atomic_write_text,
    backup_file,
    discover_project_root,
    ensure_pytest_imported,
    generate_run_id,
    resolve_test_file_path,
    restore_backup,
    sanitize_code_block,
    save_raw_log,
    validate_single_python_file,
)


class FixIterationRecord(BaseModel):
    """Bản ghi chi tiết một vòng lặp sửa lỗi test."""

    iteration: int
    error_category_before: ErrorCategory
    guard_result: GuardResult | None = None
    accepted: bool = False
    all_tests_passed_after: bool = False
    mutation_score_after: float | None = None
    backup_path: str | None = None
    reason: str = ""
    diff_preview: str = ""


class FixResult(BaseModel):
    """Kết quả tổng hợp của lệnh `aitest fix` (hoặc pha `fix` trong Agent Loop)."""

    target_file: str
    test_file: str
    fixed: bool
    dry_run: bool = False
    iterations_run: int = 0
    stop_reason: str
    initial_error_category: ErrorCategory = ErrorCategory.NONE
    final_run_result: RunResult
    last_guard_result: GuardResult | None = None
    assertion_diffs: list[AssertionDiffEntry] = Field(default_factory=list)
    diff_preview: str = ""
    records: list[FixIterationRecord] = Field(default_factory=list)


def _make_unified_diff(old_text: str, new_text: str, filename: str) -> str:
    diff_lines = difflib.unified_diff(
        old_text.splitlines(keepends=True),
        new_text.splitlines(keepends=True),
        fromfile=f"a/{filename}",
        tofile=f"b/{filename}",
    )
    return "".join(diff_lines)


def _is_same_ast(code_a: str, code_b: str) -> bool:
    if code_a.strip() == code_b.strip():
        return True
    try:
        tree_a = ast.parse(code_a)
        tree_b = ast.parse(code_b)
        return ast.dump(tree_a, include_attributes=False) == ast.dump(tree_b, include_attributes=False)
    except SyntaxError:
        return False


def fix_test_file(
    target_file: Path | str,
    config: AitestConfig,
    *,
    test_file: Path | str | None = None,
    max_iters: int | None = None,
    dry_run: bool = False,
    initial_run_result: RunResult | None = None,
    baseline_mutation_score: float | None = None,
    llm_client: OllamaClient | None = None,
    run_id: str | None = None,
    project_root: Path | str | None = None,
    reporter: ProgressReporter | None = None,
) -> FixResult:
    """Thực thi quy trình sửa lỗi test có kiểm soát (FR-04 & §12.3).

    Nguyên tắc bất biến:
    - Chỉ sửa file test, tuyệt đối không sửa file mã nguồn.
    - Bắt buộc phân loại lỗi trước khi gọi LLM: chỉ `TEST_FAILURE` và `TEST_SYNTAX` mới được phép sửa.
    - Luôn backup trước mỗi lần ghi.
    - Kiểm tra Quality Guard (AST Assertion Diff) và mutation score; vi phạm -> rollback ngay từ backup.
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
    effective_max_iters = max_iters if max_iters is not None else config.fix.max_iters

    curr_run = initial_run_result or run_tests(
        target_file=validated_target,
        config=config,
        test_file=resolved_test,
        project_root=root_dir,
        enable_mutation=False,
        reporter=rep,
    )

    if curr_run.all_tests_passed:
        return FixResult(
            target_file=str(target_file),
            test_file=str(resolved_test),
            fixed=True,
            dry_run=dry_run,
            iterations_run=0,
            stop_reason="already_passed",
            initial_error_category=ErrorCategory.NONE,
            final_run_result=curr_run,
        )

    initial_cat = curr_run.error_category
    if not initial_cat.is_fixable_by_llm:
        return FixResult(
            target_file=str(target_file),
            test_file=str(resolved_test),
            fixed=False,
            dry_run=dry_run,
            iterations_run=0,
            stop_reason=f"unfixable_error:{initial_cat.value}",
            initial_error_category=initial_cat,
            final_run_result=curr_run,
        )

    source_code = validated_target.read_text(encoding="utf-8")
    owns_client = llm_client is None
    client = llm_client or OllamaClient(config)

    records: list[FixIterationRecord] = []
    collected_diffs: list[AssertionDiffEntry] = []
    last_guard: GuardResult | None = None
    last_diff_preview = ""
    stop_reason = "max_iters_reached"
    extra_feedback = ""

    try:
        for iter_idx in range(1, effective_max_iters + 1):
            if not curr_run.error_category.is_fixable_by_llm:
                stop_reason = f"unfixable_error:{curr_run.error_category.value}"
                break

            current_test_code = resolved_test.read_text(encoding="utf-8")
            bak_path = backup_file(
                resolved_test,
                backup_root=root_dir / config.output.backup_dir,
                run_id=effective_run_id,
                subdir=f"iter_{iter_idx}",
            )

            pytest_output = f"{curr_run.stdout}\n{curr_run.stderr}".strip()
            if extra_feedback:
                pytest_output = f"{pytest_output}\n\n[Previous Fix Attempt Rejected by Quality Guard]:\n{extra_feedback}"

            sys_prompt, usr_prompt = build_fix_prompts(
                target_file=validated_target,
                test_file=resolved_test,
                source_code=source_code,
                current_test_code=current_test_code,
                pytest_output=pytest_output,
                error_category=curr_run.error_category,
                project_root=root_dir,
            )

            with rep.step(
                f"[Fix {iter_idx}/{effective_max_iters}] Đang gọi Ollama ({config.model}) sửa lỗi {curr_run.error_category.value}",
                done_message=f"[Fix {iter_idx}/{effective_max_iters}] Ollama đã trả về bản sửa cho '{resolved_test.name}'",
            ):
                llm_resp = client.chat(system_prompt=sys_prompt, user_prompt=usr_prompt)
            rep.on_llm_call_end(
                task_name=f"fix#{iter_idx}",
                model=llm_resp.model_actual or config.model,
                prompt_tokens=llm_resp.prompt_tokens,
                completion_tokens=llm_resp.completion_tokens,
                latency_ms=llm_resp.latency_ms,
            )
            candidate_code = ensure_pytest_imported(sanitize_code_block(llm_resp.content))

            if not dry_run and _is_same_ast(current_test_code, candidate_code):
                no_change_msg = (
                    "LLM returned the exact same test file without making any changes! "
                    "You MUST modify the failing assertions or exception arguments shown in the Pytest Traceback."
                )
                rep.warn(
                    f"  [yellow]⚠ \\[Fix {iter_idx}/{effective_max_iters}] LLM trả về file test chưa thay đổi, yêu cầu sửa lại...[/yellow]"
                )
                records.append(
                    FixIterationRecord(
                        iteration=iter_idx,
                        error_category_before=curr_run.error_category,
                        guard_result=None,
                        accepted=False,
                        all_tests_passed_after=False,
                        backup_path=str(bak_path) if bak_path else None,
                        reason="LLM trả về mã test giống hệt bản đang lỗi (không có thay đổi)",
                        diff_preview="",
                    )
                )
                extra_feedback = no_change_msg
                stop_reason = "max_iters_reached"
                continue

            guard_res = check_quality_guard(
                current_test_code,
                candidate_code,
                guard_config=config.fix.quality_guard,
                allow_skip=config.fix.allow_skip,
            )
            last_guard = guard_res
            diff_str = _make_unified_diff(current_test_code, candidate_code, resolved_test.name)
            last_diff_preview = diff_str

            if dry_run:
                records.append(
                    FixIterationRecord(
                        iteration=iter_idx,
                        error_category_before=curr_run.error_category,
                        guard_result=guard_res,
                        accepted=guard_res.passed,
                        all_tests_passed_after=False,
                        backup_path=str(bak_path) if bak_path else None,
                        reason="Chế độ --dry-run (chỉ kiểm tra Quality Guard và tạo diff, không ghi đè file test)",
                        diff_preview=diff_str,
                    )
                )
                collected_diffs.extend(guard_res.assertion_diffs)
                stop_reason = "dry_run_completed"
                break

            if not guard_res.passed:
                reason_msg = "; ".join(guard_res.reasons)
                rep.warn(
                    f"  [red]✗ [Fix {iter_idx}/{effective_max_iters}] Bị chặn bởi Quality Guard:[/red] {reason_msg}"
                )
                save_raw_log(
                    root_dir / config.output.log_dir,
                    effective_run_id,
                    f"fix_guard_rejected_iter_{iter_idx}",
                    f"REASONS:\n{reason_msg}\n\nCANDIDATE CODE:\n{candidate_code}",
                )
                if bak_path:
                    restore_backup(bak_path, resolved_test)
                records.append(
                    FixIterationRecord(
                        iteration=iter_idx,
                        error_category_before=curr_run.error_category,
                        guard_result=guard_res,
                        accepted=False,
                        all_tests_passed_after=False,
                        backup_path=str(bak_path) if bak_path else None,
                        reason=f"Bị chặn bởi Quality Guard: {reason_msg}",
                        diff_preview=diff_str,
                    )
                )
                extra_feedback = reason_msg
                stop_reason = "quality_guard_rejected"
                continue

            # Quality Guard passed -> Ghi file bằng Atomic Write và chạy lại pytest (+ mutation nếu pass)
            rep.info(
                f"  [green]✓[/green] [Fix {iter_idx}/{effective_max_iters}] Vượt qua Quality Guard, đang chạy lại kiểm thử..."
            )
            atomic_write_text(resolved_test, candidate_code, verify_python_ast=True)
            new_run = run_tests(
                target_file=validated_target,
                config=config,
                test_file=resolved_test,
                project_root=root_dir,
                enable_mutation=config.mutation.enabled,
                reporter=rep,
            )

            # Kiểm tra hậu nghiệm mutation score (FR-04 bước 8): nếu giảm so với baseline -> rollback ngay
            new_mut_score = (
                new_run.mutation.mutation_score_pct
                if (new_run.mutation and new_run.mutation.mutation_available)
                else None
            )
            if (
                new_run.all_tests_passed
                and baseline_mutation_score is not None
                and new_mut_score is not None
                and config.fix.quality_guard.reject_on_mutation_drop
                and new_mut_score < baseline_mutation_score
            ):
                if bak_path:
                    restore_backup(bak_path, resolved_test)
                drop_msg = (
                    f"Mutation score bị giảm sau khi sửa ({new_mut_score:.2f}% < baseline {baseline_mutation_score:.2f}%). "
                    "Đã rollback về bản backup."
                )
                records.append(
                    FixIterationRecord(
                        iteration=iter_idx,
                        error_category_before=curr_run.error_category,
                        guard_result=guard_res,
                        accepted=False,
                        all_tests_passed_after=False,
                        mutation_score_after=new_mut_score,
                        backup_path=str(bak_path) if bak_path else None,
                        reason=drop_msg,
                        diff_preview=diff_str,
                    )
                )
                extra_feedback = drop_msg
                stop_reason = "mutation_score_dropped"
                continue

            curr_run = new_run
            collected_diffs.extend(guard_res.assertion_diffs)
            if curr_run.all_tests_passed:
                records.append(
                    FixIterationRecord(
                        iteration=iter_idx,
                        error_category_before=initial_cat,
                        guard_result=guard_res,
                        accepted=True,
                        all_tests_passed_after=True,
                        mutation_score_after=new_mut_score,
                        backup_path=str(bak_path) if bak_path else None,
                        reason="Sửa test thành công, vượt qua Quality Guard và toàn bộ test đã pass",
                        diff_preview=diff_str,
                    )
                )
                stop_reason = "all_tests_passed"
                break

            records.append(
                FixIterationRecord(
                    iteration=iter_idx,
                    error_category_before=curr_run.error_category,
                    guard_result=guard_res,
                    accepted=True,
                    all_tests_passed_after=False,
                    mutation_score_after=new_mut_score,
                    backup_path=str(bak_path) if bak_path else None,
                    reason=f"Đạt Quality Guard nhưng test vẫn chưa pass hoàn toàn ({curr_run.tests_failed} failed)",
                    diff_preview=diff_str,
                )
            )
            stop_reason = "max_iters_reached"
    finally:
        if owns_client:
            client.close()

    return FixResult(
        target_file=str(target_file),
        test_file=str(resolved_test),
        fixed=curr_run.all_tests_passed and not dry_run,
        dry_run=dry_run,
        iterations_run=len(records),
        stop_reason=stop_reason,
        initial_error_category=initial_cat,
        final_run_result=curr_run,
        last_guard_result=last_guard,
        assertion_diffs=collected_diffs,
        diff_preview=last_diff_preview,
        records=records,
    )
