"""Hiển thị kết quả CLI bằng Rich (Dashboard, Table, Panel), định tuyến Smart TTY và xuất báo cáo đa định dạng (SRS v3.1.0)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Literal

from rich.console import Console
from rich.markup import escape as escape_markup
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from aitest.agent import AgentRunResult
from aitest.config import OutputFormat, normalize_output_format
from aitest.fixer import FixResult
from aitest.generator import SafeAppendResult
from aitest.reporting import (
    CRITERION_SOURCE_LABELS,
    RUBRIC_LABELS,
    _format_agent_history_row,
    _rubric_note,
    build_missing_line_rows,
    build_mutant_context_rows,
    compress_line_ranges,
    load_source_lines,
    render_eval_html,
    render_eval_markdown,
)
from aitest.schemas import RUBRIC_KEYS, ErrorCategory, EvalReport, RunResult
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

_EXT_TO_FORMAT: dict[str, OutputFormat] = {
    ".json": "json",
    ".md": "md",
    ".markdown": "md",
    ".html": "html",
    ".htm": "html",
}


def print_safety_notice(console: Console) -> None:
    """Hiển thị thông báo lưu ý an toàn thực thi trước khi chạy pytest (NFR-11)."""
    console.print(EXECUTION_SAFETY_WARNING)


def _score_bar(score: float | None, width: int = 10) -> str:
    """Tạo thanh tiến trình Unicode có màu cho thang điểm 0-10."""
    if score is None:
        return "[dim]" + ("░" * width) + "[/dim]"
    clamped = max(0.0, min(10.0, score))
    filled = int(round((clamped / 10.0) * width))
    color = "green" if clamped >= 8.0 else ("yellow" if clamped >= 6.0 else "red")
    return f"[{color}]{'█' * filled}[/{color}][dim]{'░' * (width - filled)}[/dim]"


def print_gen_result(console: Console, result: SafeAppendResult, target_file: str) -> None:
    """Hiển thị bảng tổng kết sau khi chạy `aitest gen`."""
    table = Table(title=f"Kết quả Sinh Unit Test ({result.mode})", show_header=True)
    table.add_column("Chỉ mục", style="cyan", no_wrap=True)
    table.add_column("Giá trị", style="white")

    table.add_row("File nguồn", escape_markup(target_file))
    table.add_row("File test", escape_markup(result.test_file))
    table.add_row("Chế độ ghi", escape_markup(result.mode))
    table.add_row("File backup", escape_markup(result.backup_path or "(Không có)"))
    table.add_row(
        f"Hàm test thêm mới ({len(result.added_tests)})",
        escape_markup(", ".join(result.added_tests)) if result.added_tests else "(Không có)",
    )
    table.add_row(
        f"Hàm test bỏ qua do trùng ({len(result.skipped_tests)})",
        escape_markup(", ".join(result.skipped_tests)) if result.skipped_tests else "(Không có)",
    )
    renamed_str = (
        ", ".join(f"{old} -> {new}" for old, new in result.renamed_tests)
        if result.renamed_tests
        else "(Không có)"
    )
    table.add_row(f"Hàm test đổi tên ({len(result.renamed_tests)})", escape_markup(renamed_str))

    console.print(table)
    console.print(
        f"[dim]Bước tiếp theo gợi ý:[/dim] [bold green]aitest run {escape_markup(target_file)}[/bold green] "
        f"[dim]| hoặc đánh giá trực tiếp:[/dim] [bold cyan]aitest eval {escape_markup(target_file)} -o report.html[/bold cyan]"
    )


def print_run_result(
    console: Console,
    result: RunResult,
    *,
    project_root: Path | None = None,
) -> None:
    """Hiển thị bảng tổng kết sau khi chạy `aitest run` kèm ngữ cảnh dòng code nguồn."""
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
    cov_str = f"{result.line_coverage_pct:.2f}%" if result.line_coverage_pct is not None else "null"
    table.add_row("Line Coverage (File nguồn)", cov_str)
    if result.missing_lines:
        table.add_row("Dòng chưa phủ (Missing lines)", escape_markup(compress_line_ranges(result.missing_lines)))

    source_lines = load_source_lines(result.target_file, project_root=project_root)

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

    table.add_row("Thời gian chạy", f"{result.duration_ms} ms")
    console.print(table)

    if result.mutation and result.mutation.survived_mutants:
        mutant_rows = build_mutant_context_rows(result.mutation.survived_mutants, source_lines)
        mut_table = Table(
            title=f"Survived Mutants tiêu biểu ({len(mutant_rows)})",
            show_header=True,
        )
        mut_table.add_column("Dòng", style="yellow", justify="right", no_wrap=True)
        mut_table.add_column("Phép đột biến", style="magenta", no_wrap=True)
        mut_table.add_column("Mã nguồn tại dòng lỗi", style="white")
        for row in mutant_rows[:10]:
            mut_table.add_row(
                f"L{row['line']}",
                escape_markup(row["mutation"]),
                escape_markup(row["code"] or "(Không có trích đoạn)"),
            )
        if len(mutant_rows) > 10:
            mut_table.add_row("...", f"+{len(mutant_rows) - 10}", "Xem toàn bộ trong báo cáo JSON/HTML")
        console.print(mut_table)

    if not result.all_tests_passed:
        guidance = ERROR_CATEGORY_GUIDANCE.get(result.error_category)
        if guidance:
            console.print(Panel(guidance, title=f"Cảnh báo phân loại lỗi: {result.error_category.value}", border_style="red"))
        elif result.error_detail:
            console.print(f"[yellow]Chi tiết lỗi:[/yellow] {escape_markup(result.error_detail)}")


def print_fix_result(console: Console, result: FixResult) -> None:
    """Hiển thị kết quả sau khi chạy `aitest fix` với so sánh Trước -> Sau."""
    title = "Kết quả Sửa Test (Dry-Run)" if result.dry_run else "Kết quả Sửa Test Tự động (aitest fix)"
    status_str = (
        "[bold green]THÀNH CÔNG[/bold green]"
        if result.fixed
        else f"[bold yellow]DỪNG ({escape_markup(result.stop_reason)})[/bold yellow]"
    )

    lines = [
        f"Trạng thái: {status_str}",
        f"File test: {escape_markup(result.test_file)}",
        f"Phân loại lỗi ban đầu: {result.initial_error_category.value}",
        f"Số vòng lặp đã chạy: {result.iterations_run}",
        f"Lý do dừng: {escape_markup(result.stop_reason)}",
    ]

    if result.initial_run_result is not None:
        ini = result.initial_run_result
        fin = result.final_run_result
        lines.append(
            "So sánh Trước -> Sau: "
            f"[bold]Passed:[/bold] {ini.tests_passed} -> {fin.tests_passed} | "
            f"[bold]Failed:[/bold] {ini.tests_failed} -> {fin.tests_failed} | "
            f"[bold]Coverage:[/bold] "
            f"{f'{ini.line_coverage_pct:.1f}%' if ini.line_coverage_pct is not None else 'null'} -> "
            f"{f'{fin.line_coverage_pct:.1f}%' if fin.line_coverage_pct is not None else 'null'}"
        )

    if result.initial_error_category in ERROR_CATEGORY_GUIDANCE:
        lines.append(f"\n[bold red]Hướng dẫn:[/bold red] {ERROR_CATEGORY_GUIDANCE[result.initial_error_category]}")

    if result.assertion_diffs:
        lines.append("\n[bold cyan]Nhật ký AST Assertion Diff:[/bold cyan]")
        for d in result.assertion_diffs:
            lines.append(
                f"  - [{escape_markup(d.classification)}] `{escape_markup(d.test_name)}`: "
                f"`{escape_markup(d.old_assertion)}` -> `{escape_markup(d.new_assertion)}` ({escape_markup(d.reason)})"
            )

    console.print(Panel("\n".join(lines), title=title, border_style="cyan"))

    if result.dry_run and result.diff_preview:
        console.print(Syntax(result.diff_preview, "diff", theme="monokai", line_numbers=False))


def print_eval_dashboard(
    console: Console,
    report: EvalReport,
    *,
    project_root: Path | None = None,
) -> None:
    """Hiển thị Rich Dashboard đánh giá toàn diện 8 tiêu chí Rubric, ngữ cảnh code và góp ý LLM."""
    status_badge = (
        "[bold green]PASSED EVAL[/bold green]"
        if report.passed_eval
        else (
            "[bold yellow]BELOW THRESHOLD[/bold yellow]"
            if report.execution.all_tests_passed
            else "[bold red]TESTS FAILED[/bold red]"
        )
    )
    pass_rate_str = (
        f"{report.execution.pass_rate * 100:.2f}%"
        if report.execution.pass_rate is not None
        else "null"
    )
    cov_str = (
        f"{report.objective_metrics.line_coverage_pct:.2f}%"
        if report.objective_metrics.line_coverage_pct is not None
        else "null"
    )
    mut_str = (
        f"{report.objective_metrics.mutation_score_pct:.2f}% "
        f"({report.objective_metrics.mutation_killed or 0}/{(report.objective_metrics.mutation_killed or 0) + (report.objective_metrics.mutation_survived or 0)} killed)"
        if report.objective_metrics.mutation_score_pct is not None
        else f"null ({report.objective_metrics.mutation_status})"
    )
    judge_str = report.meta.judge_model_actual or report.meta.judge_model_requested or report.meta.model_requested

    kpi_lines = [
        f"[bold]Trạng thái:[/bold] {status_badge} | "
        f"[bold]Weighted Score:[/bold] [bold cyan]{report.weighted_score:.2f}[/bold cyan] / {report.threshold_pass:.2f} | "
        f"[bold]Comparable:[/bold] {report.report_comparable}",
        f"[bold]Pass Rate:[/bold] {pass_rate_str} ({report.execution.tests_passed} pass, {report.execution.tests_failed} fail, {report.execution.tests_skipped} skip) | "
        f"[bold]Line Coverage:[/bold] {cov_str} | "
        f"[bold]Mutation:[/bold] {escape_markup(mut_str)}",
        f"[dim]Target: {escape_markup(report.meta.target_file)} | Model: {escape_markup(report.meta.model_actual or report.meta.model_requested)} | Judge: {escape_markup(judge_str)} | run_id={escape_markup(report.meta.run_id[:8])}[/dim]",
    ]
    if not report.report_comparable and report.non_comparable_reason:
        kpi_lines.append(f"[yellow]⚠ Lưu ý so sánh:[/yellow] {escape_markup(report.non_comparable_reason)}")

    console.print(
        Panel(
            "\n".join(kpi_lines),
            title="Bảng Đánh giá Chất lượng Unit Test (aitest eval)",
            border_style="green" if report.passed_eval else "yellow",
        )
    )

    rubric_table = Table(title="Chi tiết 8 Tiêu chí Rubric", show_header=True)
    rubric_table.add_column("Tiêu chí", style="cyan", no_wrap=True)
    rubric_table.add_column("Mức điểm", no_wrap=True)
    rubric_table.add_column("Điểm", justify="right", no_wrap=True)
    rubric_table.add_column("Trọng số", justify="right", no_wrap=True)
    rubric_table.add_column("Hiệu dụng", justify="right", no_wrap=True)
    rubric_table.add_column("Nguồn & Ghi chú", style="white")

    for key in RUBRIC_KEYS:
        val = report.scores.get(key)
        score_txt = f"{val:.2f}" if val is not None else "null"
        src = CRITERION_SOURCE_LABELS.get(key, "-")
        note = _rubric_note(report, key)
        note_col = f"{src}" if note == "-" else f"{src} — {note}"
        rubric_table.add_row(
            f"{RUBRIC_LABELS.get(key, key)} ({key})",
            _score_bar(val),
            score_txt,
            f"{report.weights.get(key, 0.0) * 100:.0f}%",
            f"{report.weights_effective.get(key, 0.0) * 100:.1f}%",
            escape_markup(note_col),
        )
    console.print(rubric_table)

    source_lines = load_source_lines(report.meta.target_file, project_root=project_root)
    mutant_rows = build_mutant_context_rows(report.objective_metrics.survived_mutants, source_lines)
    missing_rows = build_missing_line_rows(report.objective_metrics.missing_lines, source_lines)

    if mutant_rows or missing_rows:
        gap_table = Table(
            title=f"Lỗ hổng Kiểm thử (Survived Mutants: {len(mutant_rows)}, Missing Lines: {len(missing_rows)})",
            show_header=True,
        )
        gap_table.add_column("Loại", style="yellow", no_wrap=True)
        gap_table.add_column("Dòng", justify="right", no_wrap=True)
        gap_table.add_column("Chi tiết", style="magenta")
        gap_table.add_column("Mã nguồn tại dòng lỗi", style="white")

        for row in mutant_rows[:10]:
            gap_table.add_row(
                "Mutant",
                f"L{row['line']}",
                escape_markup(row["mutation"]),
                escape_markup(row["code"] or "(Không đọc được dòng nguồn)"),
            )
        if len(mutant_rows) > 10:
            gap_table.add_row("Mutant", "...", f"+{len(mutant_rows) - 10} khác", "Xem đầy đủ trong báo cáo HTML/MD")

        for row in missing_rows[:8]:
            gap_table.add_row(
                "[red]Uncovered[/red]",
                f"L{row['line']}",
                "Dòng chưa thực thi",
                escape_markup(row["code"] or "(Không đọc được dòng nguồn)"),
            )
        if len(missing_rows) > 8:
            gap_table.add_row(
                "[red]Uncovered[/red]",
                "...",
                escape_markup(compress_line_ranges(report.objective_metrics.missing_lines)),
                "Xem đầy đủ trong báo cáo HTML/MD",
            )
        console.print(gap_table)

    llm_lines = [f"[bold]Tóm tắt:[/bold] {escape_markup(report.llm_assessment.summary)}"]
    if report.llm_assessment.issues:
        llm_lines.append("")
        for idx, issue in enumerate(report.llm_assessment.issues, start=1):
            llm_lines.append(f"[bold cyan]{idx}. {escape_markup(issue.title)}[/bold cyan]")
            llm_lines.append(f"   [white]Mô tả:[/white] {escape_markup(issue.description)}")
            llm_lines.append(f"   [green]Gợi ý:[/green] {escape_markup(issue.suggestion)}")
    console.print(Panel("\n".join(llm_lines), title="Nhận xét & Gợi ý từ LLM Judge", border_style="blue"))


def print_agent_summary(
    console: Console,
    result: AgentRunResult,
    *,
    project_root: Path | None = None,
    include_eval_dashboard: bool = True,
) -> None:
    """Hiển thị bảng lịch sử biến thiên chỉ số vòng lặp Agent và Dashboard đánh giá cuối cùng."""
    mode_label = "Baseline A (--baseline)" if result.baseline_mode else "Agent B (Feedback-driven)"
    table = Table(title=f"Lịch sử Vòng lặp Agent [{mode_label}] - run_id={result.run_id[:8]}", show_header=True)
    table.add_column("Vòng", style="cyan", justify="right", no_wrap=True)
    table.add_column("Hành động", style="magenta", no_wrap=True)
    table.add_column("KQ", justify="center", no_wrap=True)
    table.add_column("Tests", justify="right", no_wrap=True)
    table.add_column("Cov %", justify="right", no_wrap=True)
    table.add_column("Mut % (Δ)", no_wrap=True)
    table.add_column("Score", justify="right", no_wrap=True)
    table.add_column("Calls", justify="right", no_wrap=True)
    table.add_column("Tokens", justify="right", no_wrap=True)
    table.add_column("Chi tiết / Lý do", style="white")

    for item in result.history:
        row = _format_agent_history_row(item)
        acc_str = "[green]Yes[/green]" if row["accepted"] else "[red]No[/red]"
        mut_cell = escape_markup(row["mutation"])
        if "(+" in mut_cell:
            mut_cell = f"[green]{mut_cell}[/green]"
        table.add_row(
            row["iteration"],
            escape_markup(row["action"]),
            acc_str,
            escape_markup(row["tests"]),
            escape_markup(row["coverage"]),
            mut_cell,
            escape_markup(row["score"]),
            row["llm_calls"],
            row["tokens"],
            escape_markup(row["reason"]),
        )

    console.print(table)
    rep = result.final_report
    summary_line = (
        f"[bold]Stop reason:[/bold] {escape_markup(result.stop_reason)} | "
        f"[bold]Weighted Score:[/bold] {rep.weighted_score:.2f}/{rep.threshold_pass:.2f} | "
        f"[bold]Passed Eval:[/bold] {rep.passed_eval} | "
        f"[bold]Comparable:[/bold] {rep.report_comparable} | "
        f"[bold]Total LLM Calls:[/bold] {result.total_llm_calls} | "
        f"[bold]Total Tokens:[/bold] {result.total_tokens} | "
        f"[bold]Runtime:[/bold] {result.runtime_sec:.2f}s"
    )
    console.print(Panel(summary_line, title="Tổng kết Agent Loop", border_style="green" if rep.passed_eval else "yellow"))

    if include_eval_dashboard:
        print_eval_dashboard(console, rep, project_root=project_root)


def infer_report_format(
    explicit_format: str | None,
    output_file: Path | str | None = None,
    default_format: str = "json",
) -> OutputFormat:
    """Xác định định dạng xuất báo cáo theo thứ tự ưu tiên: `--format` > đuôi file `-o` > cấu hình mặc định."""
    if explicit_format is not None and str(explicit_format).strip():
        return normalize_output_format(explicit_format)
    if output_file is not None:
        ext = Path(output_file).suffix.strip().lower()
        if ext in _EXT_TO_FORMAT:
            return _EXT_TO_FORMAT[ext]
    return normalize_output_format(default_format)


def _render_report_content(
    report: EvalReport,
    fmt: OutputFormat,
    *,
    agent_result: AgentRunResult | None = None,
    project_root: Path | None = None,
) -> str:
    if fmt == "json":
        return report.model_dump_json(indent=2)
    if fmt == "md":
        return render_eval_markdown(report, agent_result=agent_result, project_root=project_root)
    return render_eval_html(report, agent_result=agent_result, project_root=project_root)


def deliver_report(
    report: EvalReport,
    *,
    agent_result: AgentRunResult | None = None,
    explicit_format: str | None = None,
    config_format: str = "json",
    output_file: Path | str | None = None,
    ui_console: Console | None = None,
    quiet: bool = False,
    is_stdout_tty: bool | None = None,
    project_root: Path | None = None,
) -> str:
    """Điều phối xuất báo cáo Smart TTY và đa định dạng (json | md | html) cho `eval` và `agent`."""
    explicit_norm: OutputFormat | None = (
        normalize_output_format(explicit_format)
        if explicit_format is not None and str(explicit_format).strip()
        else None
    )
    tty = sys.stdout.isatty() if is_stdout_tty is None else is_stdout_tty

    file_fmt: OutputFormat | None = None
    file_content: str | None = None
    if output_file is not None:
        file_fmt = infer_report_format(explicit_norm, output_file, config_format)
        file_content = _render_report_content(
            report,
            file_fmt,
            agent_result=agent_result,
            project_root=project_root,
        )
        atomic_write_text(Path(output_file), file_content + "\n")
        if not quiet and ui_console is not None:
            ui_console.print(
                f"[green]✓[/green] Đã lưu báo cáo {file_fmt.upper()}: [bold]{escape_markup(str(output_file))}[/bold]"
            )

    if not quiet and ui_console is not None:
        show_rich = (
            (explicit_norm is None)
            or (output_file is not None and tty and explicit_norm in ("html", "md"))
            or (not tty)
        )
        if show_rich:
            if agent_result is not None:
                print_agent_summary(
                    ui_console,
                    agent_result,
                    project_root=project_root,
                    include_eval_dashboard=True,
                )
            else:
                print_eval_dashboard(ui_console, report, project_root=project_root)
            if tty and explicit_norm is None and output_file is None:
                ui_console.print(
                    "[dim]Mẹo: thêm -o report.html để xuất HTML Dashboard hoặc --format json để lấy JSON thô.[/dim]"
                )

    emitted_text: str | None = None
    if explicit_norm is not None:
        emit_stdout = not (output_file is not None and tty and not quiet and explicit_norm in ("html", "md"))
        if emit_stdout:
            emitted_text = (
                file_content
                if (file_content is not None and file_fmt == explicit_norm)
                else _render_report_content(
                    report,
                    explicit_norm,
                    agent_result=agent_result,
                    project_root=project_root,
                )
            )
            print(emitted_text)
    elif not tty or quiet:
        default_fmt = normalize_output_format(config_format)
        emitted_text = (
            file_content
            if (file_content is not None and file_fmt == default_fmt)
            else _render_report_content(
                report,
                default_fmt,
                agent_result=agent_result,
                project_root=project_root,
            )
        )
        print(emitted_text)

    if emitted_text is not None:
        return emitted_text
    if file_content is not None:
        return file_content
    return report.model_dump_json(indent=2)


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
