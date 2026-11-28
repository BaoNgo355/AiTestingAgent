"""Giao diện dòng lệnh (CLI) đầy đủ 6 lệnh cho aitest sử dụng Typer và Rich (SRS v3.0.0 §6.2, §7 FR-01..FR-08, §9.1)."""

from __future__ import annotations

import importlib.util
import os
import shutil
from pathlib import Path
from typing import Any, Literal, Optional

import typer
from rich.console import Console
from rich.panel import Panel

from aitest import __version__
from aitest.agent import run_agent_loop
from aitest.config import DEFAULT_AITEST_YAML_TEMPLATE, load_config, normalize_output_format
from aitest.evaluator import evaluate_test_suite
from aitest.fixer import fix_test_file
from aitest.generator import SafeAppendError, generate_tests
from aitest.llm.client import (
    AntigravityClient,
    AntigravityError,
    validate_sdk_credentials,
)
from aitest.output import (
    deliver_report,
    print_fix_result,
    print_gen_result,
    print_run_result,
    print_safety_notice,
)
from aitest.progress import RichProgressReporter
from aitest.runner import run_tests
from aitest.utils import atomic_write_text

app = typer.Typer(
    name="aitest",
    help="CLI AI Agent hỗ trợ tự động sinh, cải thiện và đánh giá Unit Test Python bằng Google Antigravity SDK.",
    add_completion=False,
    no_args_is_help=True,
)
console = Console()
err_console = Console(stderr=True)


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"aitest version {__version__}")
        raise typer.Exit()


def _fail_with_error(message: str, code: int = 1) -> None:
    err_console.print(f"[bold red]Lỗi:[/bold red] {message}")
    raise typer.Exit(code=code)


def _build_cli_overrides(
    *,
    model: str | None = None,
    judge_model: str | None = None,
) -> dict[str, Any] | None:
    overrides: dict[str, Any] = {}
    if model is not None and model.strip():
        overrides["model"] = model.strip()
    if judge_model is not None and judge_model.strip():
        overrides.setdefault("eval", {})["judge_model"] = judge_model.strip()
    return overrides or None


@app.callback()
def main(
    version: Optional[bool] = typer.Option(
        None,
        "--version",
        "-v",
        help="Hiển thị phiên bản của aitest và thoát.",
        callback=_version_callback,
        is_eager=True,
    ),
) -> None:
    """aitest CLI root callback."""


@app.command("init")
def init_cmd(
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Ghi đè aitest.yaml và .env.example nếu đã tồn tại.",
    ),
) -> None:
    """Khởi tạo file cấu hình aitest.yaml, .env.example, cập nhật .gitignore và kiểm tra môi trường (FR-01)."""
    created_files: list[str] = []
    skipped_files: list[str] = []

    yaml_path = Path("aitest.yaml")
    if yaml_path.exists() and not force:
        skipped_files.append("aitest.yaml (đã tồn tại, dùng --force để ghi đè)")
    else:
        atomic_write_text(yaml_path, DEFAULT_AITEST_YAML_TEMPLATE)
        created_files.append("aitest.yaml")

    env_example_path = Path(".env.example")
    env_example_content = (
        "# Khóa API cho Google Antigravity SDK (tạo tại https://aistudio.google.com/app/api-keys)\n"
        "GEMINI_API_KEY=your_gemini_api_key_here\n"
        "# AITEST_MODEL=gemini-3.8-flash\n"
    )
    if env_example_path.exists() and not force:
        skipped_files.append(".env.example (đã tồn tại)")
    else:
        atomic_write_text(env_example_path, env_example_content)
        created_files.append(".env.example")

    gitignore_path = Path(".gitignore")
    required_ignores = [".env", ".aitest_backup/", ".aitest_logs/"]
    if gitignore_path.exists():
        existing_text = gitignore_path.read_text(encoding="utf-8")
        existing_lines = {line.strip() for line in existing_text.splitlines()}
        missing = [item for item in required_ignores if item not in existing_lines]
        if missing:
            suffix = ("\n" if not existing_text.endswith("\n") else "") + "\n".join(missing) + "\n"
            atomic_write_text(gitignore_path, existing_text + suffix)
            created_files.append(f".gitignore (bổ sung: {', '.join(missing)})")
        else:
            skipped_files.append(".gitignore (đã có đủ mục cần thiết)")
    else:
        atomic_write_text(gitignore_path, "\n".join(required_ignores) + "\n")
        created_files.append(".gitignore")

    missing_pkgs: list[str] = []
    for pkg, mod in [("pytest", "pytest"), ("pytest-cov", "pytest_cov")]:
        if importlib.util.find_spec(mod) is None:
            missing_pkgs.append(pkg)

    cfg = load_config()
    sdk_installed = importlib.util.find_spec("google.antigravity") is not None
    antigravity_status = ""
    if not sdk_installed:
        antigravity_status = (
            "[bold red]Chưa cài gói `google-antigravity`.[/bold red] "
            "Cài đặt bằng: `pip install \"google-antigravity>=0.1.24\"`."
        )
    else:
        try:
            validate_sdk_credentials(cfg)
            mode_label = "Vertex AI" if cfg.llm.vertex else "Gemini API (GEMINI_API_KEY)"
            judge_note = (
                f" | Judge model: '{cfg.eval.judge_model}'."
                if cfg.eval.judge_model
                else ""
            )
            antigravity_status = (
                f"[green]Google Antigravity SDK đã sẵn sàng - Chế độ: {mode_label}, "
                f"model hiệu dụng: '{cfg.effective_model_label}'.[/green]{judge_note}"
            )
        except AntigravityError as exc:
            antigravity_status = f"[yellow]{exc}[/yellow]"
        except Exception as exc:
            antigravity_status = f"[yellow]Chưa kiểm tra được trạng thái Google Antigravity SDK: {exc}[/yellow]"

    lines: list[str] = []
    if created_files:
        lines.append("[bold green]Đã tạo / cập nhật:[/bold green]")
        lines.extend(f"  - {item}" for item in created_files)
    if skipped_files:
        lines.append("[bold yellow]Đã bỏ qua:[/bold yellow]")
        lines.extend(f"  - {item}" for item in skipped_files)
    if missing_pkgs:
        lines.append(
            f"[bold red]Thiếu thư viện:[/bold red] {', '.join(missing_pkgs)} "
            f"(cài đặt bằng: `pip install {' '.join(missing_pkgs)}`)"
        )
    else:
        lines.append("[green]Môi trường kiểm thử:[/green] pytest, pytest-cov và AST Mutation Engine đã sẵn sàng.")

    if os.name != "nt" and cfg.mutation.tool == "mutmut" and shutil.which("mutmut") is None:
        lines.append(
            "[yellow]Lưu ý mutation backend:[/yellow] `mutation.tool=mutmut` nhưng chưa thấy lệnh `mutmut`. "
            "Cài đặt bằng: `pip install \"mutmut>=3,<4\"` hoặc chuyển sang `tool: auto`."
        )

    lines.append(f"Trạng thái Antigravity: {antigravity_status}")

    console.print(Panel("\n".join(lines), title="aitest init", border_style="cyan"))


@app.command("gen")
def gen_cmd(
    file: str = typer.Argument(..., help="Đường dẫn tới một file Python (.py) đơn lẻ cần sinh test."),
    mode: str = typer.Option(
        "append",
        "--mode",
        "-m",
        help="Chế độ ghi file test: 'append' (mặc định - Safe Append) hoặc 'overwrite'.",
    ),
    no_llm_dedup: bool = typer.Option(
        False,
        "--no-llm-dedup",
        help="Không gửi danh sách tên hàm test hiện có vào prompt LLM (Safe Append vẫn dedup bằng AST).",
    ),
    test_file: Optional[Path] = typer.Option(
        None,
        "--test-file",
        help="Đường dẫn file test đích tùy chọn.",
    ),
    model: Optional[str] = typer.Option(
        None,
        "--model",
        help="Ghi đè model ID của Google Antigravity SDK (mặc định: gemini-3.8-flash).",
    ),
    quiet: bool = typer.Option(
        False,
        "--quiet",
        "-q",
        help="Tắt toàn bộ hiển thị tiến độ trực quan trung gian.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        help="Hiển thị thêm chi tiết token và latency của từng lần gọi LLM.",
    ),
) -> None:
    """Sinh unit test cho một file Python (.py) đơn lẻ với cơ chế Safe Append (FR-02)."""
    norm_mode = mode.strip().lower()
    if norm_mode not in ("append", "overwrite"):
        _fail_with_error("Giá trị --mode chỉ nhận 'append' hoặc 'overwrite'.")
    lit_mode: Literal["append", "overwrite"] = "overwrite" if norm_mode == "overwrite" else "append"

    try:
        overrides = _build_cli_overrides(model=model)
        cfg = load_config(cli_overrides=overrides)
        reporter = RichProgressReporter(console=err_console, quiet=quiet, verbose=verbose)
        res = generate_tests(
            target_file=file,
            config=cfg,
            mode=lit_mode,
            no_llm_dedup=no_llm_dedup,
            test_file=test_file,
            reporter=reporter,
        )
        print_gen_result(console, res, target_file=file)
    except (ValueError, FileNotFoundError, SyntaxError, SafeAppendError, AntigravityError) as exc:
        _fail_with_error(str(exc), code=1)


@app.command("run")
def run_cmd(
    file: str = typer.Argument(..., help="Đường dẫn tới file Python (.py) mục tiêu."),
    cov: bool = typer.Option(
        True,
        "--cov/--no-cov",
        help="Bật/tắt đo line coverage của riêng file nguồn mục tiêu.",
    ),
    mutation: bool = typer.Option(
        True,
        "--mutation/--no-mutation",
        help="Bật/tắt Mutation Testing (chỉ chạy khi bộ test pass 100%).",
    ),
    timeout: Optional[int] = typer.Option(
        None,
        "--timeout",
        "-t",
        help="Timeout thực thi pytest (giây, mặc định 120s).",
    ),
    test_file: Optional[Path] = typer.Option(
        None,
        "--test-file",
        help="Đường dẫn file test đích tùy chọn.",
    ),
    quiet: bool = typer.Option(
        False,
        "--quiet",
        "-q",
        help="Tắt toàn bộ hiển thị tiến độ trực quan trung gian.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        help="Hiển thị chi tiết trạng thái của từng mutant trong lúc chạy.",
    ),
) -> None:
    """Chạy unit test, thu thập line coverage và mutation score khách quan (FR-03)."""
    try:
        cfg = load_config()
        if not quiet:
            print_safety_notice(console)
        reporter = RichProgressReporter(console=err_console, quiet=quiet, verbose=verbose)
        res = run_tests(
            target_file=file,
            config=cfg,
            test_file=test_file,
            enable_coverage=cov,
            enable_mutation=mutation,
            timeout_sec=timeout,
            reporter=reporter,
        )
        print_run_result(console, res)
        if not res.all_tests_passed:
            raise typer.Exit(code=1)
    except (ValueError, FileNotFoundError, SyntaxError) as exc:
        _fail_with_error(str(exc), code=1)


@app.command("fix")
def fix_cmd(
    file: str = typer.Argument(..., help="Đường dẫn tới file Python (.py) mục tiêu."),
    max_iters: Optional[int] = typer.Option(
        None,
        "--max-iters",
        help="Số vòng lặp sửa lỗi tối đa (mặc định từ aitest.yaml = 3).",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Chỉ kiểm tra Quality Guard và hiển thị unified diff, không ghi đè file test.",
    ),
    test_file: Optional[Path] = typer.Option(
        None,
        "--test-file",
        help="Đường dẫn file test đích tùy chọn.",
    ),
    model: Optional[str] = typer.Option(
        None,
        "--model",
        help="Ghi đè model ID của Google Antigravity SDK (mặc định: gemini-3.8-flash).",
    ),
    quiet: bool = typer.Option(
        False,
        "--quiet",
        "-q",
        help="Tắt toàn bộ hiển thị tiến độ trực quan trung gian.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        help="Hiển thị thêm chi tiết token, latency và từng mutant.",
    ),
) -> None:
    """Sửa lỗi file test tự động có kiểm soát với Quality Guard và AST Assertion Diff (FR-04)."""
    try:
        overrides = _build_cli_overrides(model=model)
        cfg = load_config(cli_overrides=overrides)
        if not quiet:
            print_safety_notice(console)
        reporter = RichProgressReporter(console=err_console, quiet=quiet, verbose=verbose)
        res = fix_test_file(
            target_file=file,
            config=cfg,
            test_file=test_file,
            max_iters=max_iters,
            dry_run=dry_run,
            reporter=reporter,
        )
        print_fix_result(console, res)
        if not res.fixed and not res.dry_run:
            raise typer.Exit(code=1)
    except (ValueError, FileNotFoundError, SyntaxError, AntigravityError) as exc:
        _fail_with_error(str(exc), code=1)


@app.command("eval")
def eval_cmd(
    file: str = typer.Argument(..., help="Đường dẫn tới file Python (.py) mục tiêu."),
    format: Optional[str] = typer.Option(
        None,
        "--format",
        "-f",
        help="Định dạng xuất báo cáo: json, md, html (mặc định tự động theo Smart TTY hoặc đuôi file -o).",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Đường dẫn file để ghi báo cáo EvalReport (hỗ trợ .json, .md, .html).",
    ),
    no_mutation: bool = typer.Option(
        False,
        "--no-mutation",
        help="Tắt Mutation Testing (báo cáo sẽ được đánh dấu report_comparable = false).",
    ),
    test_file: Optional[Path] = typer.Option(
        None,
        "--test-file",
        help="Đường dẫn file test đích tùy chọn.",
    ),
    model: Optional[str] = typer.Option(
        None,
        "--model",
        help="Ghi đè model ID của Google Antigravity SDK (mặc định: gemini-3.8-flash).",
    ),
    judge_model: Optional[str] = typer.Option(
        None,
        "--judge-model",
        help="Chỉ định model riêng cho bước chấm điểm rubric.",
    ),
    quiet: bool = typer.Option(
        False,
        "--quiet",
        "-q",
        help="Tắt toàn bộ hiển thị tiến độ trên stderr (chỉ xuất báo cáo ra stdout).",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        help="Hiển thị thêm chi tiết token, latency và từng mutant trên stderr.",
    ),
) -> None:
    """Đánh giá chất lượng unit test theo mô hình 2 tầng và xuất báo cáo chuẩn (json/md/html) (FR-05)."""
    try:
        if format is not None:
            normalize_output_format(format)
        overrides = _build_cli_overrides(model=model, judge_model=judge_model)
        cfg = load_config(cli_overrides=overrides)
        if not quiet:
            print_safety_notice(err_console)
        reporter = RichProgressReporter(console=err_console, quiet=quiet, verbose=verbose)
        report = evaluate_test_suite(
            target_file=file,
            config=cfg,
            test_file=test_file,
            enable_mutation=(False if no_mutation else cfg.mutation.enabled),
            reporter=reporter,
        )
        deliver_report(
            report,
            explicit_format=format,
            config_format=cfg.output.format,
            output_file=output,
            ui_console=err_console,
            quiet=quiet,
        )
    except (ValueError, FileNotFoundError, SyntaxError, AntigravityError) as exc:
        _fail_with_error(str(exc), code=1)


@app.command("agent")
def agent_cmd(
    file: str = typer.Argument(..., help="Đường dẫn tới file Python (.py) mục tiêu."),
    max_iters: Optional[int] = typer.Option(
        None,
        "--max-iters",
        help="Số vòng lặp Agent tối đa (mặc định 3).",
    ),
    threshold: Optional[float] = typer.Option(
        None,
        "--threshold",
        help="Ngưỡng điểm weighted_score để đạt yêu cầu (mặc định 8.0).",
    ),
    max_llm_calls: Optional[int] = typer.Option(
        None,
        "--max-llm-calls",
        help="Ngân sách tổng số lần gọi LLM tối đa (mặc định 12).",
    ),
    max_total_tokens: Optional[int] = typer.Option(
        None,
        "--max-total-tokens",
        help="Ngân sách tổng số token tối đa (mặc định 400000).",
    ),
    max_runtime_sec: Optional[int] = typer.Option(
        None,
        "--max-runtime-sec",
        help="Ngân sách tổng thời gian chạy tối đa tính bằng giây (mặc định 900).",
    ),
    baseline: bool = typer.Option(
        False,
        "--baseline",
        help="Chạy nhánh đối sánh Baseline A (sinh test + sửa lỗi nếu có + đánh giá 1 lần, bỏ qua vòng lặp improve).",
    ),
    format: Optional[str] = typer.Option(
        None,
        "--format",
        "-f",
        help="Định dạng xuất báo cáo: json, md, html (mặc định tự động theo Smart TTY hoặc đuôi file -o).",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Ghi báo cáo EvalReport cuối cùng ra file (hỗ trợ .json, .md, .html).",
    ),
    test_file: Optional[Path] = typer.Option(
        None,
        "--test-file",
        help="Đường dẫn file test đích tùy chọn.",
    ),
    model: Optional[str] = typer.Option(
        None,
        "--model",
        help="Ghi đè model ID của Google Antigravity SDK (mặc định: gemini-3.8-flash).",
    ),
    judge_model: Optional[str] = typer.Option(
        None,
        "--judge-model",
        help="Chỉ định model riêng cho bước chấm điểm rubric.",
    ),
    quiet: bool = typer.Option(
        False,
        "--quiet",
        "-q",
        help="Tắt toàn bộ hiển thị tiến độ trên stderr (chỉ xuất báo cáo ra stdout).",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        help="Hiển thị thêm chi tiết token, latency và từng mutant trên stderr.",
    ),
) -> None:
    """Thực thi vòng lặp Agent tự động: gen -> run -> fix -> mutation -> eval -> improve (FR-06)."""
    try:
        if format is not None:
            normalize_output_format(format)
        overrides = _build_cli_overrides(model=model, judge_model=judge_model)
        cfg = load_config(cli_overrides=overrides)
        if not quiet:
            print_safety_notice(err_console)
        reporter = RichProgressReporter(console=err_console, quiet=quiet, verbose=verbose)
        agent_res = run_agent_loop(
            target_file=file,
            config=cfg,
            test_file=test_file,
            max_iters=max_iters,
            threshold_pass=threshold,
            max_llm_calls=max_llm_calls,
            max_total_tokens=max_total_tokens,
            max_runtime_sec=max_runtime_sec,
            baseline=baseline,
            reporter=reporter,
        )
        deliver_report(
            agent_res.final_report,
            agent_result=agent_res,
            explicit_format=format,
            config_format=cfg.output.format,
            output_file=output,
            ui_console=err_console,
            quiet=quiet,
        )
    except (ValueError, FileNotFoundError, SyntaxError, SafeAppendError, AntigravityError) as exc:
        _fail_with_error(str(exc), code=1)


if __name__ == "__main__":
    app()
