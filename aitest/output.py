"""Hiển thị kết quả CLI bằng Rich (Table, Panel), cảnh báo an toàn thực thi (NFR-11) và xuất JSON."""

from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from aitest.agent import AgentRunResult
from aitest.fixer import FixResult
from aitest.generator import SafeAppendResult
from aitest.schemas import ErrorCategory, EvalReport, RunResult
from aitest.utils import atomic_write_text

EXECUTION_SAFETY_WARNING = (
    "[bold yellow]Lưu ý an toàn thực thi (NFR-11):[/bold yellow] "
    "Mã test do LLM sinh được thực thi cục bộ qua pytest với timeout cứng. "
    "Khuyến nghị chạy trong môi trường ảo/container cô lập đối với mã nguồn chưa tin cậy."
)

ERROR_CATEGORY_GUIDANCE: dict[ErrorCategory, str] = {
    ErrorCategory.ENVIRONMENT_ERROR: (
        "Lỗi môi trường / thiếu thư viện (ImportError / ModuleNotFoundError). "
        "Hệ thống KHÔNG gọi LLM sửa test. Vui lòng kiểm tra PYTHONPATH hoặc cài đặt dependency còn thiếu."
    ),
    ErrorCategory.SOURCE_ERROR: (
        "Phát hiện lỗi runtime nội tại phát sinh từ file mã nguồn gốc (SOURCE_ERROR). "
        "Hệ thống KHÔNG sửa test để che giấu bug của mã nguồn. Vui lòng kiểm tra lại logic file nguồn."
    ),
    ErrorCategory.TIMEOUT: (
        "Quá thời gian thực thi (TIMEOUT). Tiến trình con đã bị hủy sạch. "
        "Vui lòng kiểm tra vòng lặp vô hạn hoặc tăng `--timeout`."
    ),
    ErrorCategory.COLLECTION_ERROR: (
        "Pytest không thể thu thập test (COLLECTION_ERROR). Vui lòng kiểm tra cấu trúc file test."
    ),
    ErrorCategory.UNKNOWN_ERROR: (
        "Lỗi không xác định (UNKNOWN_ERROR). Hệ thống dừng để đảm bảo an toàn."
    ),
}


def print_safety_notice(console: Console) -> None:
    """Hiển thị thông báo lưu ý an toàn thực thi trước khi chạy pytest (NFR-11)."""
    console.print(EXECUTION_SAFETY_WARNING)


def print_gen_result(console: Console, result: SafeAppendResult, target_file: str) -> None:
    """Hiển thị bảng tổng kết sau khi chạy `aitest gen`."""
    table = Table(title=f"Kết quả Sinh Unit Test ({result.mode})", show_header=True)
    table.add_column("Chỉ mục", style="cyan", no_wrap=True)
    table.add_column("Giá trị", style="white")

    table.add_row("File nguồn", target_file)
    table.add_row("File test", result.test_file)
    table.add_row("Chế độ ghi", result.mode)
    table.add_row("File backup", result.backup_path or "(Không có)")
    table.add_row(
        f"Hàm test thêm mới ({len(result.added_tests)})",
        ", ".join(result.added_tests) if result.added_tests else "(Không có)",
    )
    table.add_row(
        f"Hàm test bỏ qua do trùng ({len(result.skipped_tests)})",
        ", ".join(result.skipped_tests) if result.skipped_tests else "(Không có)",
    )
    renamed_str = (
        ", ".join(f"{old} -> {new}" for old, new in result.renamed_tests)
        if result.renamed_tests
        else "(Không có)"
    )
    table.add_row(f"Hàm test đổi tên ({len(result.renamed_tests)})", renamed_str)

    console.print(table)
    console.print(f"[dim]Bước tiếp theo gợi ý:[/dim] [bold green]aitest run {target_file}[/bold green]")


def print_run_result(console: Console, result: RunResult) -> None:
    """Hiển thị bảng tổng kết sau khi chạy `aitest run`."""
    status_text = (
        "[bold green]PASSED (100%)[/bold green]"
        if result.all_tests_passed
        else f"[bold red]FAILED ({result.error_category.value})[/bold red]"
    )
    table = Table(title="Kết quả Thực thi Unit Test & Chỉ số Khách quan", show_header=True)
    table.add_column("Chỉ số", style="cyan", no_wrap=True)
    table.add_column("Kết quả", style="white")

    table.add_row("Trạng thái", status_text)
    table.add_row("Exit Code", str(result.exit_code))
    table.add_row(
        "Số lượng Test",
        f"Tổng AST: {result.test_count} | Passed: {result.tests_passed} | "
        f"Failed: {result.tests_failed} | Skipped: {result.tests_skipped}",
    )
    pass_rate_str = f"{result.pass_rate * 100:.2f}%" if result.pass_rate is not None else "null"
    table.add_row("Pass Rate", pass_rate_str)
    cov_str = (
        f"{result.line_coverage_pct:.2f}%"
        if result.line_coverage_pct is not None
        else "null"
    )
    table.add_row("Line Coverage (File nguồn)", cov_str)
    if result.missing_lines:
        table.add_row("Dòng chưa phủ (Missing lines)", ", ".join(str(x) for x in result.missing_lines))

    if result.mutation:
        mut = result.mutation
        mut_score_str = (
            f"{mut.mutation_score_pct:.2f}%"
            if mut.mutation_score_pct is not None
            else f"null ({mut.mutation_status})"
        )
        table.add_row(
            "Mutation Score",
            f"{mut_score_str} [Killed: {mut.mutation_killed}, Survived: {mut.mutation_survived}, Excluded: {mut.mutation_excluded}]",
        )
        if mut.survived_mutants:
            preview = "; ".join(
                f"dòng {m.line} ({m.mutation})" for m in mut.survived_mutants[:8]
            )
            table.add_row(f"Survived Mutants ({len(mut.survived_mutants)})", preview)

    table.add_row("Thời gian chạy", f"{result.duration_ms} ms")
    console.print(table)

    if not result.all_tests_passed:
        guidance = ERROR_CATEGORY_GUIDANCE.get(result.error_category)
        if guidance:
            console.print(Panel(guidance, title=f"Cảnh báo phân loại lỗi: {result.error_category.value}", border_style="red"))
        elif result.error_detail:
            console.print(f"[yellow]Chi tiết lỗi:[/yellow] {result.error_detail}")


def print_fix_result(console: Console, result: FixResult) -> None:
    """Hiển thị kết quả sau khi chạy `aitest fix`."""
    title = "Kết quả Sửa Test (Dry-Run)" if result.dry_run else "Kết quả Sửa Test Tự động (aitest fix)"
    status_str = (
        "[bold green]THÀNH CÔNG[/bold green]"
        if result.fixed
        else f"[bold yellow]DỪNG ({result.stop_reason})[/bold yellow]"
    )

    lines = [
        f"Trạng thái: {status_str}",
        f"File test: {result.test_file}",
        f"Phân loại lỗi ban đầu: {result.initial_error_category.value}",
        f"Số vòng lặp đã chạy: {result.iterations_run}",
        f"Lý do dừng: {result.stop_reason}",
    ]

    if result.initial_error_category in ERROR_CATEGORY_GUIDANCE:
        lines.append(f"\n[bold red]Hướng dẫn:[/bold red] {ERROR_CATEGORY_GUIDANCE[result.initial_error_category]}")

    if result.assertion_diffs:
        lines.append("\n[bold cyan]Nhật ký AST Assertion Diff:[/bold cyan]")
        for d in result.assertion_diffs:
            lines.append(
                f"  - [{d.classification}] `{d.test_name}`: `{d.old_assertion}` -> `{d.new_assertion}` ({d.reason})"
            )

    console.print(Panel("\n".join(lines), title=title, border_style="cyan"))

    if result.dry_run and result.diff_preview:
        console.print(Syntax(result.diff_preview, "diff", theme="monokai", line_numbers=False))


def print_agent_summary(console: Console, result: AgentRunResult) -> None:
    """Hiển thị bảng lịch sử vòng lặp Agent ra stderr/console."""
    mode_label = "Baseline A (--baseline)" if result.baseline_mode else "Agent B (Feedback-driven)"
    table = Table(title=f"Lịch sử Vòng lặp Agent [{mode_label}] - run_id={result.run_id[:8]}", show_header=True)
    table.add_column("Vòng", style="cyan", justify="right")
    table.add_column("Hành động", style="magenta")
    table.add_column("Chấp nhận", justify="center")
    table.add_column("LLM Calls", justify="right")
    table.add_column("Tokens", justify="right")
    table.add_column("Chi tiết / Lý do", style="white")

    for item in result.history:
        acc_str = "[green]Yes[/green]" if item.accepted else "[red]No[/red]"
        table.add_row(
            str(item.iteration),
            item.action,
            acc_str,
            str(item.llm_calls),
            str(item.tokens_used),
            item.reason,
        )

    console.print(table)
    rep = result.final_report
    summary_line = (
        f"[bold]Stop reason:[/bold] {result.stop_reason} | "
        f"[bold]Weighted Score:[/bold] {rep.weighted_score:.2f}/{rep.threshold_pass:.2f} | "
        f"[bold]Passed Eval:[/bold] {rep.passed_eval} | "
        f"[bold]Comparable:[/bold] {rep.report_comparable} | "
        f"[bold]Total LLM Calls:[/bold] {result.total_llm_calls} | "
        f"[bold]Total Tokens:[/bold] {result.total_tokens} | "
        f"[bold]Runtime:[/bold] {result.runtime_sec:.2f}s"
    )
    console.print(Panel(summary_line, title="Tổng kết Agent Loop", border_style="green" if rep.passed_eval else "yellow"))


def emit_eval_report_json(
    report: EvalReport,
    *,
    output_file: Path | str | None = None,
    print_to_stdout: bool = True,
) -> str:
    """Serialize `EvalReport` thành JSON chuẩn (FR-05). Ghi ra file nếu có `--output` và in ra stdout."""
    json_text = report.model_dump_json(indent=2)
    if output_file is not None:
        atomic_write_text(Path(output_file), json_text + "\n")
    if print_to_stdout:
        print(json_text)
    return json_text
