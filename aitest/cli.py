"""Giao diện dòng lệnh (CLI) đầy đủ 6 lệnh cho aitest sử dụng Typer và Rich (SRS v2.3.0 §6.2, §7 FR-01..FR-07, §9.1)."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Literal, Optional

import httpx
import typer
from rich.console import Console
from rich.panel import Panel

from aitest import __version__
from aitest.agent import run_agent_loop
from aitest.config import DEFAULT_AITEST_YAML_TEMPLATE, load_config
from aitest.evaluator import evaluate_test_suite
from aitest.fixer import fix_test_file
from aitest.generator import SafeAppendError, generate_tests
from aitest.llm.client import OllamaError
from aitest.output import (
    emit_eval_report_json,
    print_agent_summary,
    print_fix_result,
    print_gen_result,
    print_run_result,
    print_safety_notice,
)
from aitest.runner import run_tests
from aitest.utils import atomic_write_text

app = typer.Typer(
    name="aitest",
    help="CLI AI Agent hỗ trợ tự động sinh, cải thiện và đánh giá Unit Test Python bằng LLM qua Ollama.",
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
        "OLLAMA_BASE_URL=http://localhost:11434\n"
        "# OLLAMA_API_KEY=...   # chỉ cần nếu kết nối tới Ollama remote có bật xác thực\n"
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
    ollama_status = ""
    try:
        resp = httpx.get(f"{cfg.base_url}/api/tags", timeout=5.0)
        resp.raise_for_status()
        tags_data = resp.json()
        models = [m.get("name", "") for m in tags_data.get("models", [])]
        if any(m == cfg.model or m.startswith(f"{cfg.model}:") for m in models):
            ollama_status = f"[green]Đã kết nối Ollama ({cfg.base_url}) - Model '{cfg.model}' đã sẵn sàng.[/green]"
        else:
            ollama_status = (
                f"[yellow]Đã kết nối Ollama ({cfg.base_url}) nhưng chưa thấy model '{cfg.model}'.\n"
                f"Gợi ý: chạy `ollama pull {cfg.model}`[/yellow]"
            )
    except Exception:
        ollama_status = (
            f"[yellow]Không kết nối được Ollama tại {cfg.base_url}.\n"
            f"Gợi ý: chạy `ollama serve` và `ollama pull {cfg.model}`[/yellow]"
        )

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
    lines.append(f"Trạng thái Ollama: {ollama_status}")

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
    model: Optional[str] = typer.Option(None, "--model", help="Ghi đè tên model Ollama."),
) -> None:
    """Sinh unit test cho một file Python (.py) đơn lẻ với cơ chế Safe Append (FR-02)."""
    norm_mode = mode.strip().lower()
    if norm_mode not in ("append", "overwrite"):
        _fail_with_error("Giá trị --mode chỉ nhận 'append' hoặc 'overwrite'.")
    lit_mode: Literal["append", "overwrite"] = "overwrite" if norm_mode == "overwrite" else "append"

    try:
        cfg = load_config(cli_overrides={"model": model} if model else None)
        res = generate_tests(
            target_file=file,
            config=cfg,
            mode=lit_mode,
            no_llm_dedup=no_llm_dedup,
        )
        print_gen_result(console, res, target_file=file)
    except (ValueError, FileNotFoundError, SyntaxError, SafeAppendError, OllamaError) as exc:
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
) -> None:
    """Chạy unit test, thu thập line coverage và mutation score khách quan (FR-03)."""
    try:
        cfg = load_config()
        print_safety_notice(console)
        res = run_tests(
            target_file=file,
            config=cfg,
            enable_coverage=cov,
            enable_mutation=mutation,
            timeout_sec=timeout,
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
    model: Optional[str] = typer.Option(None, "--model", help="Ghi đè tên model Ollama."),
) -> None:
    """Sửa lỗi file test tự động có kiểm soát với Quality Guard và AST Assertion Diff (FR-04)."""
    try:
        overrides = {"model": model} if model else None
        cfg = load_config(cli_overrides=overrides)
        print_safety_notice(console)
        res = fix_test_file(
            target_file=file,
            config=cfg,
            max_iters=max_iters,
            dry_run=dry_run,
        )
        print_fix_result(console, res)
        if not res.fixed and not res.dry_run:
            raise typer.Exit(code=1)
    except (ValueError, FileNotFoundError, SyntaxError, OllamaError) as exc:
        _fail_with_error(str(exc), code=1)


@app.command("eval")
def eval_cmd(
    file: str = typer.Argument(..., help="Đường dẫn tới file Python (.py) mục tiêu."),
    format: str = typer.Option(
        "json",
        "--format",
        help="Định dạng xuất báo cáo (MVP hỗ trợ 'json').",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Đường dẫn file để ghi báo cáo EvalReport JSON.",
    ),
    no_mutation: bool = typer.Option(
        False,
        "--no-mutation",
        help="Tắt Mutation Testing (báo cáo sẽ được đánh dấu report_comparable = false).",
    ),
    model: Optional[str] = typer.Option(None, "--model", help="Ghi đè tên model Ollama."),
) -> None:
    """Đánh giá chất lượng unit test theo mô hình 2 tầng và xuất báo cáo JSON chuẩn Pydantic (FR-05)."""
    if format.strip().lower() != "json":
        _fail_with_error("MVP chỉ hỗ trợ định dạng xuất '--format json'.")
    try:
        cfg = load_config(cli_overrides={"model": model} if model else None)
        print_safety_notice(err_console)
        report = evaluate_test_suite(
            target_file=file,
            config=cfg,
            enable_mutation=(False if no_mutation else cfg.mutation.enabled),
        )
        emit_eval_report_json(report, output_file=output, print_to_stdout=True)
    except (ValueError, FileNotFoundError, SyntaxError, OllamaError) as exc:
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
        help="Ngân sách tổng số token tối đa (mặc định 200000).",
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
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Ghi báo cáo EvalReport JSON cuối cùng ra file.",
    ),
    model: Optional[str] = typer.Option(None, "--model", help="Ghi đè tên model Ollama."),
) -> None:
    """Thực thi vòng lặp Agent tự động: gen -> run -> fix -> mutation -> eval -> improve (FR-06)."""
    try:
        cfg = load_config(cli_overrides={"model": model} if model else None)
        print_safety_notice(err_console)
        agent_res = run_agent_loop(
            target_file=file,
            config=cfg,
            max_iters=max_iters,
            threshold_pass=threshold,
            max_llm_calls=max_llm_calls,
            max_total_tokens=max_total_tokens,
            max_runtime_sec=max_runtime_sec,
            baseline=baseline,
        )
        print_agent_summary(err_console, agent_res)
        emit_eval_report_json(agent_res.final_report, output_file=output, print_to_stdout=True)
    except (ValueError, FileNotFoundError, SyntaxError, SafeAppendError, OllamaError) as exc:
        _fail_with_error(str(exc), code=1)
