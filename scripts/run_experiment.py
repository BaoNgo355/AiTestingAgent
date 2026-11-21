"""Kịch bản thực nghiệm tự động đối sánh Baseline A (`--baseline`) vs Agent B (`aitest agent`) theo SRS v2.3.0 §16.

Cách dùng:
    python scripts/run_experiment.py [--runs 3] [--max-iters 3] [--threshold 8.0] [--files examples/calculator.py ...]
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import shutil
import statistics
import tempfile
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from aitest.agent import AgentRunResult, run_agent_loop
from aitest.config import AitestConfig, load_config
from aitest.progress import RichProgressReporter
from aitest.utils import atomic_write_text, utc_now_iso

DEFAULT_EXAMPLE_FILES: tuple[str, ...] = (
    "examples/calculator.py",
    "examples/string_utils.py",
    "examples/date_scheduler.py",
    "examples/inventory.py",
    "examples/bank_account.py",
)

console = Console()


def mean_std(values: list[float]) -> tuple[float, float]:
    """Tính giá trị Trung bình (Mean) và Độ lệch chuẩn mẫu (SD)."""
    if not values:
        return 0.0, 0.0
    m = statistics.mean(values)
    sd = statistics.stdev(values) if len(values) >= 2 else 0.0
    return round(m, 2), round(sd, 2)


def fmt_mean_sd(values: list[float], unit: str = "") -> str:
    if not values:
        return "N/A"
    m, sd = mean_std(values)
    return f"{m:.2f} ± {sd:.2f}{unit}"


def summarize_branch_runs(runs: list[AgentRunResult]) -> dict[str, Any]:
    """Tổng hợp thống kê Mean ± SD cho một nhánh thực nghiệm (Baseline A hoặc Agent B) theo SRS §16.4 & §16.5."""
    total_runs = len(runs)
    if total_runs == 0:
        return {"total_runs": 0, "comparable_runs": 0}

    passed_runs = [r for r in runs if r.final_report.execution.all_tests_passed]
    # Chỉ đưa các lần chạy có report_comparable == True vào bảng so sánh chính (§16.5)
    comparable_runs = [r for r in runs if r.final_report.report_comparable]
    non_comparable_runs = [r for r in runs if not r.final_report.report_comparable]

    target_pool = comparable_runs if comparable_runs else runs

    cov_vals = [
        r.final_report.objective_metrics.line_coverage_pct
        for r in target_pool
        if r.final_report.objective_metrics.line_coverage_pct is not None
    ]
    mut_vals = [
        r.final_report.objective_metrics.mutation_score_pct
        for r in target_pool
        if r.final_report.objective_metrics.mutation_score_pct is not None
    ]
    killed_vals = [
        float(r.final_report.objective_metrics.mutation_killed or 0) for r in target_pool
    ]
    survived_vals = [
        float(r.final_report.objective_metrics.mutation_survived or 0) for r in target_pool
    ]
    excluded_vals = [
        float(r.final_report.objective_metrics.mutation_excluded or 0) for r in target_pool
    ]
    test_count_vals = [
        float(r.final_report.objective_metrics.test_count) for r in target_pool
    ]
    weighted_vals = [r.final_report.weighted_score for r in target_pool]
    llm_call_vals = [float(r.total_llm_calls) for r in target_pool]
    token_vals = [float(r.total_tokens) for r in target_pool]
    runtime_vals = [r.runtime_sec for r in target_pool]
    guard_reject_vals = [float(r.quality_guard_rejections) for r in runs]

    return {
        "total_runs": total_runs,
        "comparable_runs": len(comparable_runs),
        "non_comparable_runs": len(non_comparable_runs),
        "all_tests_passed_rate_pct": round((len(passed_runs) / total_runs) * 100.0, 2),
        "report_comparable_rate_pct": round((len(comparable_runs) / total_runs) * 100.0, 2),
        "line_coverage_pct": {
            "mean": mean_std(cov_vals)[0],
            "sd": mean_std(cov_vals)[1],
            "formatted": fmt_mean_sd(cov_vals, "%"),
        },
        "mutation_score_pct": {
            "mean": mean_std(mut_vals)[0],
            "sd": mean_std(mut_vals)[1],
            "formatted": fmt_mean_sd(mut_vals, "%"),
        },
        "mutation_killed": {
            "mean": mean_std(killed_vals)[0],
            "sd": mean_std(killed_vals)[1],
            "formatted": fmt_mean_sd(killed_vals),
        },
        "mutation_survived": {
            "mean": mean_std(survived_vals)[0],
            "sd": mean_std(survived_vals)[1],
            "formatted": fmt_mean_sd(survived_vals),
        },
        "mutation_excluded": {
            "mean": mean_std(excluded_vals)[0],
            "sd": mean_std(excluded_vals)[1],
            "formatted": fmt_mean_sd(excluded_vals),
        },
        "test_count": {
            "mean": mean_std(test_count_vals)[0],
            "sd": mean_std(test_count_vals)[1],
            "formatted": fmt_mean_sd(test_count_vals),
        },
        "quality_guard_rejections": {
            "total": int(sum(guard_reject_vals)),
            "mean": mean_std(guard_reject_vals)[0],
            "sd": mean_std(guard_reject_vals)[1],
            "formatted": fmt_mean_sd(guard_reject_vals),
        },
        "weighted_score": {
            "mean": mean_std(weighted_vals)[0],
            "sd": mean_std(weighted_vals)[1],
            "formatted": fmt_mean_sd(weighted_vals),
        },
        "llm_calls": {
            "mean": mean_std(llm_call_vals)[0],
            "sd": mean_std(llm_call_vals)[1],
            "formatted": fmt_mean_sd(llm_call_vals),
        },
        "total_tokens": {
            "mean": mean_std(token_vals)[0],
            "sd": mean_std(token_vals)[1],
            "formatted": fmt_mean_sd(token_vals),
        },
        "runtime_sec": {
            "mean": mean_std(runtime_vals)[0],
            "sd": mean_std(runtime_vals)[1],
            "formatted": fmt_mean_sd(runtime_vals, "s"),
        },
    }


def render_markdown_comparison_report(summary_data: dict[str, Any]) -> str:
    """Sinh báo cáo đối sánh dạng Markdown bảng Mean ± SD giữa Baseline A và Agent B."""
    b_a = summary_data["baseline_a_summary"]
    b_b = summary_data["agent_b_summary"]
    meta = summary_data["meta"]

    lines: list[str] = [
        "# Báo cáo Thực nghiệm Đối sánh: Baseline A vs Agent B (`aitest`)",
        "",
        f"- **Thời gian thực hiện:** `{meta['generated_at']}`",
        f"- **Model:** `{meta['model']}` (temperature=`{meta['temperature']}`, num_ctx=`{meta['num_ctx']}`)",
        f"- **Số lần chạy lặp mỗi file (`runs`):** `{meta['runs_per_file']}`",
        f"- **Số vòng lặp tối đa (`max_iters`):** `{meta['max_iters']}` | **Ngưỡng đạt (`threshold_pass`):** `{meta['threshold_pass']}`",
        f"- **Cấu hình hệ thống:** `{meta['platform']}` (Python `{meta['python_version']}`)",
        "",
        "## 1. Bảng Đối sánh Tổng hợp (Mean ± SD trên các lần chạy `report_comparable == true`)",
        "",
        "| Nhóm | Chỉ số | Nhánh Baseline A (`--baseline`) | Nhánh Agent B (`aitest agent`) |",
        "|---|---|---|---|",
        f"| **Hiệu quả kiểm thử** | Line Coverage (%) | {b_a['line_coverage_pct']['formatted']} | {b_b['line_coverage_pct']['formatted']} |",
        f"| | **Mutation Score (%)** | **{b_a['mutation_score_pct']['formatted']}** | **{b_b['mutation_score_pct']['formatted']}** |",
        f"| | Số mutant Killed | {b_a['mutation_killed']['formatted']} | {b_b['mutation_killed']['formatted']} |",
        f"| | Số mutant Survived | {b_a['mutation_survived']['formatted']} | {b_b['mutation_survived']['formatted']} |",
        f"| | Số lượng hàm test (`test_count`) | {b_a['test_count']['formatted']} | {b_b['test_count']['formatted']} |",
        f"| **Độ tin cậy** | Tỷ lệ `all_tests_passed == true` | {b_a['all_tests_passed_rate_pct']:.2f}% | {b_b['all_tests_passed_rate_pct']:.2f}% |",
        f"| | Tỷ lệ `report_comparable == true` | {b_a['report_comparable_rate_pct']:.2f}% ({b_a['comparable_runs']}/{b_a['total_runs']}) | {b_b['report_comparable_rate_pct']:.2f}% ({b_b['comparable_runs']}/{b_b['total_runs']}) |",
        f"| | Số lần Quality Guard reject | {b_a['quality_guard_rejections']['total']} ({b_a['quality_guard_rejections']['formatted']}) | {b_b['quality_guard_rejections']['total']} ({b_b['quality_guard_rejections']['formatted']}) |",
        f"| **Đánh giá tổng hợp** | **Điểm tổng hợp (`weighted_score`)** | **{b_a['weighted_score']['formatted']}** | **{b_b['weighted_score']['formatted']}** |",
        f"| **Chi phí tài nguyên** | Số lần gọi LLM (`llm_calls`) | {b_a['llm_calls']['formatted']} | {b_b['llm_calls']['formatted']} |",
        f"| | Tổng số token (`total_tokens`) | {b_a['total_tokens']['formatted']} | {b_b['total_tokens']['formatted']} |",
        f"| | Thời gian thực thi (`runtime_sec`) | {b_a['runtime_sec']['formatted']} | {b_b['runtime_sec']['formatted']} |",
        "",
    ]
    return "\n".join(lines)


def run_experiment(
    *,
    target_files: list[str],
    runs_per_file: int = 3,
    max_iters: int = 3,
    threshold_pass: float = 8.0,
    model_override: str | None = None,
    output_json: Path | str = ".aitest_logs/experiment_report.json",
    output_md: Path | str = ".aitest_logs/experiment_report.md",
    project_root: Path | str | None = None,
    quiet: bool = False,
    verbose: bool = False,
) -> dict[str, Any]:
    """Thực thi kịch bản đối sánh thực nghiệm Baseline A và Agent B trên tập file mẫu."""
    root_dir = Path(project_root).resolve() if project_root else Path.cwd().resolve()
    cli_overrides = {"model": model_override} if model_override else None
    cfg: AitestConfig = load_config(cli_overrides=cli_overrides)
    cfg.agent.max_iters = max_iters
    cfg.eval.threshold_pass = threshold_pass
    reporter = RichProgressReporter(console=console, quiet=quiet, verbose=verbose)

    baseline_results: list[AgentRunResult] = []
    agent_results: list[AgentRunResult] = []
    per_file_records: list[dict[str, Any]] = []

    work_sandbox = Path(tempfile.mkdtemp(prefix="aitest_exp_tests_"))
    try:
        for file_rel in target_files:
            target_path = root_dir / file_rel
            stem = target_path.stem
            for run_idx in range(1, runs_per_file + 1):
                if not quiet:
                    console.print(
                        f"\n[bold cyan]>>> File: {file_rel} | Run {run_idx}/{runs_per_file}[/bold cyan]"
                    )

                # 1. Nhánh Baseline A (--baseline)
                test_file_a = work_sandbox / f"baseline_run_{run_idx}" / f"test_{stem}.py"
                res_a = run_agent_loop(
                    target_file=target_path,
                    config=cfg,
                    test_file=test_file_a,
                    max_iters=max_iters,
                    threshold_pass=threshold_pass,
                    baseline=True,
                    project_root=root_dir,
                    reporter=reporter,
                )
                baseline_results.append(res_a)

                # 2. Nhánh Agent B (Feedback-driven Agent Loop)
                test_file_b = work_sandbox / f"agent_run_{run_idx}" / f"test_{stem}.py"
                res_b = run_agent_loop(
                    target_file=target_path,
                    config=cfg,
                    test_file=test_file_b,
                    max_iters=max_iters,
                    threshold_pass=threshold_pass,
                    baseline=False,
                    project_root=root_dir,
                    reporter=reporter,
                )
                agent_results.append(res_b)

                per_file_records.append(
                    {
                        "target_file": file_rel,
                        "run_index": run_idx,
                        "baseline_a": {
                            "run_id": res_a.run_id,
                            "stop_reason": res_a.stop_reason,
                            "all_tests_passed": res_a.final_report.execution.all_tests_passed,
                            "report_comparable": res_a.final_report.report_comparable,
                            "line_coverage_pct": res_a.final_report.objective_metrics.line_coverage_pct,
                            "mutation_score_pct": res_a.final_report.objective_metrics.mutation_score_pct,
                            "mutation_killed": res_a.final_report.objective_metrics.mutation_killed,
                            "mutation_survived": res_a.final_report.objective_metrics.mutation_survived,
                            "test_count": res_a.final_report.objective_metrics.test_count,
                            "weighted_score": res_a.final_report.weighted_score,
                            "llm_calls": res_a.total_llm_calls,
                            "total_tokens": res_a.total_tokens,
                            "runtime_sec": res_a.runtime_sec,
                            "quality_guard_rejections": res_a.quality_guard_rejections,
                        },
                        "agent_b": {
                            "run_id": res_b.run_id,
                            "stop_reason": res_b.stop_reason,
                            "iterations_completed": res_b.iterations_completed,
                            "all_tests_passed": res_b.final_report.execution.all_tests_passed,
                            "report_comparable": res_b.final_report.report_comparable,
                            "line_coverage_pct": res_b.final_report.objective_metrics.line_coverage_pct,
                            "mutation_score_pct": res_b.final_report.objective_metrics.mutation_score_pct,
                            "mutation_killed": res_b.final_report.objective_metrics.mutation_killed,
                            "mutation_survived": res_b.final_report.objective_metrics.mutation_survived,
                            "test_count": res_b.final_report.objective_metrics.test_count,
                            "weighted_score": res_b.final_report.weighted_score,
                            "llm_calls": res_b.total_llm_calls,
                            "total_tokens": res_b.total_tokens,
                            "runtime_sec": res_b.runtime_sec,
                            "quality_guard_rejections": res_b.quality_guard_rejections,
                        },
                    }
                )
    finally:
        shutil.rmtree(str(work_sandbox), ignore_errors=True)

    summary_a = summarize_branch_runs(baseline_results)
    summary_b = summarize_branch_runs(agent_results)

    payload: dict[str, Any] = {
        "meta": {
            "generated_at": utc_now_iso(),
            "model": cfg.model,
            "temperature": cfg.llm.temperature,
            "num_ctx": cfg.llm.num_ctx,
            "max_tokens": cfg.llm.max_tokens,
            "runs_per_file": runs_per_file,
            "max_iters": max_iters,
            "threshold_pass": threshold_pass,
            "target_files": target_files,
            "platform": platform.platform(),
            "python_version": platform.python_version(),
        },
        "baseline_a_summary": summary_a,
        "agent_b_summary": summary_b,
        "runs": per_file_records,
    }

    out_json_path = root_dir / output_json
    out_md_path = root_dir / output_md
    atomic_write_text(out_json_path, json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    md_content = render_markdown_comparison_report(payload)
    atomic_write_text(out_md_path, md_content)

    # Hiển thị bảng đối sánh ra terminal bằng Rich
    table = Table(
        title="Kết quả Thực nghiệm Đối sánh: Baseline A vs Agent B (Mean ± SD)",
        show_header=True,
    )
    table.add_column("Chỉ số", style="cyan")
    table.add_column("Baseline A (--baseline)", style="yellow")
    table.add_column("Agent B (Feedback-driven)", style="green")

    table.add_row("Line Coverage (%)", summary_a["line_coverage_pct"]["formatted"], summary_b["line_coverage_pct"]["formatted"])
    table.add_row("Mutation Score (%)", summary_a["mutation_score_pct"]["formatted"], summary_b["mutation_score_pct"]["formatted"])
    table.add_row("Mutants Killed", summary_a["mutation_killed"]["formatted"], summary_b["mutation_killed"]["formatted"])
    table.add_row("Mutants Survived", summary_a["mutation_survived"]["formatted"], summary_b["mutation_survived"]["formatted"])
    table.add_row("Số lượng hàm test", summary_a["test_count"]["formatted"], summary_b["test_count"]["formatted"])
    table.add_row("Tỷ lệ Pass 100%", f"{summary_a['all_tests_passed_rate_pct']:.2f}%", f"{summary_b['all_tests_passed_rate_pct']:.2f}%")
    table.add_row("Điểm tổng hợp (weighted_score)", summary_a["weighted_score"]["formatted"], summary_b["weighted_score"]["formatted"])
    table.add_row("Số lần gọi LLM", summary_a["llm_calls"]["formatted"], summary_b["llm_calls"]["formatted"])
    table.add_row("Tổng số Tokens", summary_a["total_tokens"]["formatted"], summary_b["total_tokens"]["formatted"])
    table.add_row("Thời gian chạy (s)", summary_a["runtime_sec"]["formatted"], summary_b["runtime_sec"]["formatted"])

    console.print(table)
    console.print(
        Panel(
            f"Đã lưu báo cáo JSON: [bold]{out_json_path}[/bold]\n"
            f"Đã lưu báo cáo Markdown: [bold]{out_md_path}[/bold]",
            title="Hoàn tất Thực nghiệm",
            border_style="green",
        )
    )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chạy kịch bản thực nghiệm đối sánh Baseline A vs Agent B cho aitest."
    )
    parser.add_argument(
        "--files",
        nargs="+",
        default=list(DEFAULT_EXAMPLE_FILES),
        help="Danh sách các file Python mục tiêu trong examples/.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Số lần chạy lặp lại trên mỗi file (mặc định: 3).",
    )
    parser.add_argument(
        "--max-iters",
        type=int,
        default=3,
        help="Số vòng lặp tối đa của Agent B (mặc định: 3).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=8.0,
        help="Ngưỡng điểm weighted_score dừng sớm (mặc định: 8.0).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Ghi đè tên model Ollama (mặc định lấy từ aitest.yaml).",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=".aitest_logs/experiment_report.json",
        help="Đường dẫn file xuất kết quả JSON.",
    )
    parser.add_argument(
        "--output-md",
        type=str,
        default=".aitest_logs/experiment_report.md",
        help="Đường dẫn file xuất bảng đối sánh Markdown.",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Tắt hiển thị tiến độ trực quan trung gian.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Hiển thị thêm chi tiết token, latency và từng mutant.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_experiment(
        target_files=args.files,
        runs_per_file=args.runs,
        max_iters=args.max_iters,
        threshold_pass=args.threshold,
        model_override=args.model,
        output_json=args.output_json,
        output_md=args.output_md,
        quiet=args.quiet,
        verbose=args.verbose,
    )
