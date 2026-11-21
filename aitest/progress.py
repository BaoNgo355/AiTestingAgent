"""Hệ thống hiển thị tiến độ trực quan (Live Progress Reporter) cho aitest CLI.

Thiết kế:
- Tách biệt hoàn toàn giao diện tiến độ khỏi core logic thông qua `ProgressReporter`.
- Hỗ trợ 3 mức độ:
  * Mặc định (Normal): Spinner kèm đồng hồ đếm giây khi gọi Ollama/pytest, thanh Progress Bar
    (`i/N`, `%`, `Killed/Survived`) khi chạy Mutation Testing, in cảnh báo khi có mutant SURVIVED,
    và in dòng tóm tắt chỉ số sau mỗi vòng Agent Loop.
  * `--verbose`: Hiển thị thêm thống kê token/latency sau mỗi lần gọi LLM và trạng thái của từng mutant.
  * `--quiet / -q`: Tắt toàn bộ hiển thị tiến độ trung gian.
- Tự động phát hiện môi trường non-TTY (`not console.is_terminal`) để chuyển sang in log tĩnh ngắn gọn,
  không sinh mã điều khiển ANSI làm bẩn file log/CI.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator, Literal

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)


class ProgressReporter:
    """Giao diện cơ sở (No-op) cho Progress Reporter.

    Mặc định khi các hàm core (`run_tests`, `generate_tests`, `run_agent_loop`, ...)
    không truyền `reporter`, hệ thống dùng lớp này để không ảnh hưởng tới unit tests.
    """

    quiet: bool = True
    verbose: bool = False

    def info(self, message: str) -> None:
        """In thông báo mốc tiến độ thông thường."""

    def verbose_info(self, message: str) -> None:
        """In thông báo chi tiết (chỉ hiển thị khi bật `--verbose`)."""

    def warn(self, message: str) -> None:
        """In cảnh báo trong quá trình thực thi."""

    @contextmanager
    def step(self, description: str, done_message: str | None = None) -> Iterator[None]:
        """Ngữ cảnh thực thi một pha (hiển thị spinner + thời gian chờ trên TTY)."""
        yield

    def on_llm_call_end(
        self,
        *,
        task_name: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: int,
    ) -> None:
        """Sự kiện sau mỗi lần gọi Ollama `/api/chat` thành công."""

    def on_mutation_start(self, total_mutants: int, target_file: str) -> None:
        """Sự kiện bắt đầu chạy danh sách mutants."""

    def on_mutant_tested(
        self,
        *,
        index: int,
        total: int,
        lineno: int,
        mutation: str,
        status: Literal["killed", "survived", "timeout", "excluded"],
        killed_so_far: int,
        survived_so_far: int,
    ) -> None:
        """Sự kiện sau khi kiểm thử xong 1 mutant."""

    def on_mutation_end(
        self,
        *,
        killed: int,
        survived: int,
        excluded: int,
        score_pct: float | None,
        duration_ms: int,
    ) -> None:
        """Sự kiện kết thúc quá trình Mutation Testing."""

    def on_agent_iteration_start(
        self,
        *,
        iteration: int,
        max_iters: int,
        baseline: bool,
    ) -> None:
        """Sự kiện bắt đầu một vòng lặp trong `aitest agent`."""

    def on_agent_iteration_summary(
        self,
        *,
        iteration: int,
        max_iters: int,
        cov_pct: float | None,
        mut_pct: float | None,
        weighted_score: float,
        threshold: float,
        total_tokens: int,
        elapsed_sec: float,
    ) -> None:
        """Sự kiện tóm tắt chỉ số cuối mỗi vòng lặp trong `aitest agent`."""


class NullProgressReporter(ProgressReporter):
    """Reporter rỗng (im lặng 100%) dùng làm mặc định trong gọi hàm thư viện/unit tests."""


class RichProgressReporter(ProgressReporter):
    """Reporter hiển thị tiến độ trực quan qua Rich Console (hỗ trợ TTY động & non-TTY fallback)."""

    def __init__(
        self,
        console: Console | None = None,
        *,
        quiet: bool = False,
        verbose: bool = False,
        force_non_tty: bool | None = None,
    ) -> None:
        self.console = console or Console(stderr=True)
        self.quiet = quiet
        self.verbose = verbose and not quiet
        self.is_tty = (
            (not force_non_tty)
            if force_non_tty is not None
            else bool(self.console.is_terminal)
        )

        self._active_progress: Progress | None = None
        self._mutation_progress: Progress | None = None
        self._mutation_task_id: TaskID | None = None

    def _stop_active_spinner(self) -> None:
        if self._active_progress is not None:
            try:
                self._active_progress.stop()
            except Exception:
                pass
            self._active_progress = None

    def _print(self, message: str) -> None:
        if self.quiet:
            return
        if self._mutation_progress is not None:
            self._mutation_progress.console.print(message)
        elif self._active_progress is not None:
            self._active_progress.console.print(message)
        else:
            self.console.print(message)

    def info(self, message: str) -> None:
        if not self.quiet:
            self._print(message)

    def verbose_info(self, message: str) -> None:
        if self.verbose and not self.quiet:
            self._print(message)

    def warn(self, message: str) -> None:
        if not self.quiet:
            self._print(message)

    @contextmanager
    def step(self, description: str, done_message: str | None = None) -> Iterator[None]:
        """Hiển thị Spinner kèm đồng hồ đếm giây trên TTY, hoặc in dòng tĩnh trên non-TTY."""
        if self.quiet:
            yield
            return

        start_ns = time.perf_counter_ns()
        if self.is_tty and self._mutation_progress is None:
            self._stop_active_spinner()
            prog = Progress(
                SpinnerColumn(),
                TextColumn("[cyan]{task.description}"),
                TimeElapsedColumn(),
                console=self.console,
                transient=True,
            )
            self._active_progress = prog
            prog.start()
            prog.add_task(description, total=None)
            ok = True
            try:
                yield
            except Exception:
                ok = False
                raise
            finally:
                self._stop_active_spinner()
                elapsed = (time.perf_counter_ns() - start_ns) / 1_000_000_000
                final_label = done_message or description
                if ok:
                    self.console.print(
                        f"[green]✓[/green] {final_label} [dim]({elapsed:.1f}s)[/dim]"
                    )
                else:
                    self.console.print(
                        f"[red]✗[/red] {final_label} thất bại [dim]({elapsed:.1f}s)[/dim]"
                    )
        else:
            # Non-TTY fallback: chỉ in dòng log tĩnh gọn gàng
            self.console.print(f"[dim]• {description}...[/dim]")
            ok = True
            try:
                yield
            except Exception:
                ok = False
                raise
            finally:
                elapsed = (time.perf_counter_ns() - start_ns) / 1_000_000_000
                final_label = done_message or description
                marker = "✓" if ok else "✗"
                self.console.print(f"{marker} {final_label} ({elapsed:.1f}s)")

    def on_llm_call_end(
        self,
        *,
        task_name: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: int,
    ) -> None:
        if not self.verbose or self.quiet:
            return
        total_tok = prompt_tokens + completion_tokens
        self._print(
            f"  [dim]↳ LLM [{task_name}] ({model}): "
            f"{prompt_tokens} prompt + {completion_tokens} completion = {total_tok} tokens "
            f"({latency_ms / 1000.0:.2f}s)[/dim]"
        )

    def on_mutation_start(self, total_mutants: int, target_file: str) -> None:
        if self.quiet:
            return
        self._stop_active_spinner()
        if self.is_tty:
            prog = Progress(
                SpinnerColumn(),
                TextColumn("[bold cyan]{task.description}"),
                BarColumn(bar_width=28),
                MofNCompleteColumn(),
                TaskProgressColumn(),
                TextColumn(
                    "[green]Killed: {task.fields[killed]}[/green] | "
                    "[yellow]Survived: {task.fields[survived]}[/yellow]"
                ),
                TimeElapsedColumn(),
                console=self.console,
                transient=True,
            )
            self._mutation_progress = prog
            prog.start()
            self._mutation_task_id = prog.add_task(
                f"Mutation Testing ({target_file})",
                total=total_mutants,
                killed=0,
                survived=0,
            )
        else:
            self.console.print(
                f"• Bắt đầu Mutation Testing trên {target_file} ({total_mutants} mutants)..."
            )

    def on_mutant_tested(
        self,
        *,
        index: int,
        total: int,
        lineno: int,
        mutation: str,
        status: Literal["killed", "survived", "timeout", "excluded"],
        killed_so_far: int,
        survived_so_far: int,
    ) -> None:
        if self.quiet:
            return

        if self._mutation_progress is not None and self._mutation_task_id is not None:
            self._mutation_progress.update(
                self._mutation_task_id,
                advance=1,
                killed=killed_so_far,
                survived=survived_so_far,
            )

        if status == "survived":
            self._print(
                f"  [yellow]⚠ Mutant survived[/yellow] [{index}/{total}] dòng {lineno}: "
                f"[bold]{mutation}[/bold]"
            )
        elif self.verbose:
            self._print(
                f"  [dim]✓ Mutant {status} [{index}/{total}] dòng {lineno}: {mutation}[/dim]"
            )

    def on_mutation_end(
        self,
        *,
        killed: int,
        survived: int,
        excluded: int,
        score_pct: float | None,
        duration_ms: int,
    ) -> None:
        if self.quiet:
            return
        if self._mutation_progress is not None:
            try:
                self._mutation_progress.stop()
            except Exception:
                pass
            self._mutation_progress = None
            self._mutation_task_id = None

        sec = duration_ms / 1000.0
        if score_pct is not None:
            self.console.print(
                f"[green]✓[/green] Mutation Testing hoàn tất: [bold]{score_pct:.2f}%[/bold] "
                f"(Killed: [green]{killed}[/green], Survived: [yellow]{survived}[/yellow], "
                f"Excluded: {excluded}) [dim]({sec:.1f}s)[/dim]"
            )
        else:
            self.console.print(
                f"[yellow]⚠[/yellow] Mutation Testing: không có mutant hợp lệ "
                f"(Killed: {killed}, Survived: {survived}, Excluded: {excluded}) [dim]({sec:.1f}s)[/dim]"
            )

    def on_agent_iteration_start(
        self,
        *,
        iteration: int,
        max_iters: int,
        baseline: bool,
    ) -> None:
        if self.quiet:
            return
        mode_tag = "Baseline A" if baseline else "Agent B"
        self._print(
            f"\n[bold magenta]━━━ Vòng lặp {iteration}/{max_iters} [{mode_tag}] ━━━[/bold magenta]"
        )

    def on_agent_iteration_summary(
        self,
        *,
        iteration: int,
        max_iters: int,
        cov_pct: float | None,
        mut_pct: float | None,
        weighted_score: float,
        threshold: float,
        total_tokens: int,
        elapsed_sec: float,
    ) -> None:
        if self.quiet:
            return
        cov_str = f"{cov_pct:.2f}%" if cov_pct is not None else "null"
        mut_str = f"{mut_pct:.2f}%" if mut_pct is not None else "null"
        self._print(
            f"  [bold cyan]↳ Tóm tắt vòng {iteration}/{max_iters}:[/bold cyan] "
            f"Cov=[bold]{cov_str}[/bold] | Mut=[bold]{mut_str}[/bold] | "
            f"Score=[bold]{weighted_score:.2f}/{threshold:.2f}[/bold] | "
            f"Tokens=[bold]{total_tokens}[/bold] | Elapsed=[bold]{elapsed_sec:.1f}s[/bold]"
        )
