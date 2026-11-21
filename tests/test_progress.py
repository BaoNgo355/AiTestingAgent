"""Unit tests cho aitest.progress (Live Progress Reporter: TTY, Non-TTY, --quiet, --verbose)."""

from __future__ import annotations

import io
import re

from rich.console import Console

from aitest.progress import NullProgressReporter, RichProgressReporter


def test_rich_progress_reporter_tty_and_non_tty_modes() -> None:
    """Kiểm tra RichProgressReporter hoạt động mượt ở cả chế độ TTY và Non-TTY fallback."""
    for force_non_tty in (False, True):
        buf = io.StringIO()
        con = Console(file=buf, force_terminal=(not force_non_tty), width=120)
        rep = RichProgressReporter(
            console=con,
            quiet=False,
            verbose=False,
            force_non_tty=force_non_tty,
        )

        rep.on_agent_iteration_start(iteration=1, max_iters=3, baseline=False)
        with rep.step("Đang gọi Ollama sinh test", done_message="Đã sinh xong test"):
            pass

        rep.on_mutation_start(total_mutants=2, target_file="examples/calc.py")
        rep.on_mutant_tested(
            index=1,
            total=2,
            lineno=10,
            mutation="+ -> -",
            status="killed",
            killed_so_far=1,
            survived_so_far=0,
        )
        rep.on_mutant_tested(
            index=2,
            total=2,
            lineno=12,
            mutation=">= -> >",
            status="survived",
            killed_so_far=1,
            survived_so_far=1,
        )
        rep.on_mutation_end(
            killed=1,
            survived=1,
            excluded=0,
            score_pct=50.0,
            duration_ms=1200,
        )
        rep.on_agent_iteration_summary(
            iteration=1,
            max_iters=3,
            cov_pct=92.5,
            mut_pct=50.0,
            weighted_score=7.85,
            threshold=8.0,
            total_tokens=340,
            elapsed_sec=2.5,
        )

        out = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", buf.getvalue())
        assert "Vòng lặp 1/3" in out
        assert "Đã sinh xong test" in out
        assert "Mutant survived" in out
        assert ">= -> >" in out
        assert "50.00%" in out
        assert "Tóm tắt vòng 1/3" in out


def test_rich_progress_reporter_quiet_and_verbose_modes() -> None:
    """Kiểm tra cờ --quiet (im lặng 100%) và --verbose (hiển thị chi tiết token/latency & mutant killed)."""
    # 1. Quiet mode
    q_buf = io.StringIO()
    q_con = Console(file=q_buf, force_terminal=False)
    q_rep = RichProgressReporter(console=q_con, quiet=True, verbose=True)
    with q_rep.step("step"):
        pass
    q_rep.info("info")
    q_rep.warn("warn")
    q_rep.on_llm_call_end(
        task_name="gen",
        model="qwen2.5-coder:7b",
        prompt_tokens=100,
        completion_tokens=50,
        latency_ms=500,
    )
    assert q_buf.getvalue() == ""

    # 2. Verbose mode
    v_buf = io.StringIO()
    v_con = Console(file=v_buf, force_terminal=False, width=120)
    v_rep = RichProgressReporter(
        console=v_con,
        quiet=False,
        verbose=True,
        force_non_tty=True,
    )
    v_rep.on_llm_call_end(
        task_name="gen",
        model="qwen2.5-coder:7b",
        prompt_tokens=100,
        completion_tokens=50,
        latency_ms=500,
    )
    v_rep.on_mutant_tested(
        index=1,
        total=1,
        lineno=5,
        mutation="== -> !=",
        status="killed",
        killed_so_far=1,
        survived_so_far=0,
    )
    v_out = v_buf.getvalue()
    assert "150 tokens" in v_out
    assert "Mutant killed" in v_out

    # 3. NullProgressReporter no-op
    null_rep = NullProgressReporter()
    with null_rep.step("noop"):
        pass
