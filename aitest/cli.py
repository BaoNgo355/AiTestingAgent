"""Giao diện dòng lệnh (CLI) cho aitest sử dụng Typer và Rich."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Optional

import httpx
import typer
from rich.console import Console
from rich.panel import Panel

from aitest import __version__
from aitest.config import DEFAULT_AITEST_YAML_TEMPLATE, load_config
from aitest.utils import atomic_write_text

app = typer.Typer(
    name="aitest",
    help="CLI AI Agent hỗ trợ tự động sinh, cải thiện và đánh giá Unit Test Python bằng LLM qua Ollama.",
    add_completion=False,
    no_args_is_help=True,
)
console = Console()


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"aitest version {__version__}")
        raise typer.Exit()


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

    # Kiểm tra công cụ pytest, pytest-cov
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
