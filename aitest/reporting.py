"""Bộ kết xuất báo cáo đa định dạng (Markdown, Standalone HTML + SVG, CSV) và ngữ cảnh mã nguồn (SRS v3.1.0)."""

from __future__ import annotations

import csv
import html
import io
import math
from pathlib import Path
from typing import Any

from aitest.schemas import RUBRIC_KEYS, EvalReport, SurvivedMutant

RUBRIC_LABELS: dict[str, str] = {
    "correctness": "Correctness",
    "coverage": "Line Coverage",
    "mutation": "Mutation Score",
    "readability": "Readability",
    "maintainability": "Maintainability",
    "isolation": "Isolation",
    "robustness": "Robustness",
    "determinism": "Determinism",
}

CRITERION_SOURCE_LABELS: dict[str, str] = {
    "correctness": "LLM + Mutation Cap",
    "coverage": "Tool (pytest-cov)",
    "mutation": "Tool (Mutation Engine)",
    "readability": "LLM Judge",
    "maintainability": "LLM Judge",
    "isolation": "LLM Judge",
    "robustness": "LLM Judge",
    "determinism": "LLM Judge",
}


def load_source_lines(
    target_file: str | Path,
    project_root: Path | None = None,
) -> list[str]:
    """Đọc an toàn từng dòng mã nguồn mục tiêu để gắn ngữ cảnh vào báo cáo; trả về [] nếu không tìm thấy file."""
    raw_path = Path(target_file)
    candidates: list[Path] = [raw_path]
    if project_root is not None:
        candidates.append(project_root / raw_path)
    candidates.append(Path.cwd() / raw_path)

    for cand in candidates:
        try:
            if cand.is_file():
                return cand.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
    return []


def compress_line_ranges(lines: list[int]) -> str:
    """Rút gọn danh sách số dòng thành các khoảng liên tiếp (ví dụ [10, 11, 12, 18] -> '10-12, 18')."""
    unique_sorted = sorted({int(ln) for ln in lines if isinstance(ln, int) and ln >= 1})
    if not unique_sorted:
        return "(Không có)"

    ranges: list[str] = []
    start = unique_sorted[0]
    end = unique_sorted[0]
    for ln in unique_sorted[1:]:
        if ln == end + 1:
            end = ln
        else:
            ranges.append(f"{start}-{end}" if end > start else str(start))
            start = ln
            end = ln
    ranges.append(f"{start}-{end}" if end > start else str(start))
    return ", ".join(ranges)


def build_mutant_context_rows(
    survived_mutants: list[SurvivedMutant],
    source_lines: list[str],
) -> list[dict[str, Any]]:
    """Ghép thông tin từng survived mutant với nội dung dòng mã nguồn tương ứng."""
    rows: list[dict[str, Any]] = []
    total_lines = len(source_lines)
    for m in survived_mutants:
        snippet = (
            source_lines[m.line - 1].strip()
            if 1 <= m.line <= total_lines
            else ""
        )
        rows.append(
            {
                "line": m.line,
                "mutation": m.mutation,
                "code": snippet,
            }
        )
    return rows


def build_missing_line_rows(
    missing_lines: list[int],
    source_lines: list[str],
) -> list[dict[str, Any]]:
    """Ghép các dòng chưa được bao phủ với nội dung dòng mã nguồn tương ứng."""
    rows: list[dict[str, Any]] = []
    total_lines = len(source_lines)
    for ln in sorted({int(x) for x in missing_lines if isinstance(x, int) and x >= 1}):
        snippet = source_lines[ln - 1].strip() if 1 <= ln <= total_lines else ""
        rows.append({"line": ln, "code": snippet})
    return rows


def _fmt_pct(val: float | None) -> str:
    return f"{val:.2f}%" if val is not None else "null"


def _fmt_score(val: float | None) -> str:
    return f"{val:.2f}" if val is not None else "null"


def _format_agent_history_row(item: Any) -> dict[str, str]:
    """Chuẩn hóa dữ liệu một bước AgentIterationLog để hiển thị đồng nhất trên Rich/Markdown/HTML."""
    before = getattr(item, "metrics_before", None) or {}
    after = getattr(item, "metrics_after", None) or {}

    test_count = after.get("test_count", before.get("test_count"))
    tests_str = str(test_count) if test_count is not None else "-"

    cov_val = after.get("line_coverage_pct", before.get("line_coverage_pct"))
    cov_str = f"{float(cov_val):.1f}%" if cov_val is not None else "-"

    mut_before = before.get("mutation_score_pct")
    mut_after = after.get("mutation_score_pct")
    if mut_after is not None:
        mut_str = f"{float(mut_after):.2f}%"
        if mut_before is not None and abs(float(mut_after) - float(mut_before)) > 1e-6:
            delta = float(mut_after) - float(mut_before)
            sign = "+" if delta > 0 else ""
            mut_str = f"{mut_str} ({sign}{delta:.2f}%)"
    elif mut_before is not None:
        mut_str = f"{float(mut_before):.2f}%"
    else:
        mut_str = "-"

    score_val = after.get("weighted_score")
    score_str = f"{float(score_val):.2f}" if score_val is not None else "-"

    return {
        "iteration": str(getattr(item, "iteration", 0)),
        "action": str(getattr(item, "action", "")),
        "accepted": bool(getattr(item, "accepted", True)),
        "tests": tests_str,
        "coverage": cov_str,
        "mutation": mut_str,
        "score": score_str,
        "llm_calls": str(getattr(item, "llm_calls", 0)),
        "tokens": str(getattr(item, "tokens_used", 0)),
        "reason": str(getattr(item, "reason", "")),
    }


def _rubric_note(report: EvalReport, key: str) -> str:
    if report.scores.get(key) is None:
        return "Không khả dụng (renormalize)"
    if key == "correctness" and report.correctness_cap_applied:
        llm_raw = report.llm_assessment.correctness
        final_c = report.scores.get("correctness")
        if final_c is not None and final_c < llm_raw - 1e-6:
            return f"Bị áp trần từ {llm_raw:.2f} -> {final_c:.2f} theo mutation"
        return "Có áp dụng mutation cap"
    return "-"


def render_eval_markdown(
    report: EvalReport,
    *,
    agent_result: Any | None = None,
    project_root: Path | None = None,
) -> str:
    """Kết xuất `EvalReport` (và lịch sử Agent nếu có) thành báo cáo Markdown chuẩn GFM."""
    source_lines = load_source_lines(report.meta.target_file, project_root=project_root)
    mutant_rows = build_mutant_context_rows(report.objective_metrics.survived_mutants, source_lines)
    missing_rows = build_missing_line_rows(report.objective_metrics.missing_lines, source_lines)

    status_badge = "PASSED" if report.passed_eval else "FAILED"
    pass_rate_str = (
        f"{report.execution.pass_rate * 100:.2f}%"
        if report.execution.pass_rate is not None
        else "null"
    )
    judge_str = report.meta.judge_model_actual or report.meta.judge_model_requested or report.meta.model_requested

    lines: list[str] = [
        f"# Báo cáo Đánh giá Unit Test (`aitest v{report.meta.version}`)",
        "",
        f"- **Trạng thái đánh giá:** **{status_badge}** (`weighted_score = {report.weighted_score:.2f} / {report.threshold_pass:.2f}`)",
        f"- **File nguồn mục tiêu:** `{report.meta.target_file}` | **File test:** `{report.meta.test_file}`",
        f"- **Provider / Model:** `{report.meta.provider}` / `{report.meta.model_actual or report.meta.model_requested}` | **Judge:** `{judge_str}` | **Endpoint:** `{'Vertex AI' if report.meta.vertex else 'Gemini API'}`",
        f"- **Run ID:** `{report.meta.run_id}` | **Conversation ID:** `{report.meta.conversation_id or 'N/A'}` | **Thời điểm:** `{report.meta.generated_at}`",
        "",
        "## 1. Tổng quan Chỉ số (KPI)",
        "",
        "| Chỉ số | Giá trị | Ghi chú |",
        "|---|---|---|",
        f"| **Weighted Score** | **{report.weighted_score:.2f} / 10.00** | Ngưỡng đạt: `{report.threshold_pass:.2f}` |",
        f"| **Thực thi Test** | `{pass_rate_str}` ({report.execution.tests_passed} passed, {report.execution.tests_failed} failed, {report.execution.tests_skipped} skipped) | `test_count={report.objective_metrics.test_count}`, `duration={report.execution.duration_ms} ms` |",
        f"| **Line Coverage** | `{_fmt_pct(report.objective_metrics.line_coverage_pct)}` | Dòng chưa phủ: `{compress_line_ranges(report.objective_metrics.missing_lines)}` |",
        f"| **Mutation Score** | `{_fmt_pct(report.objective_metrics.mutation_score_pct)}` | Status: `{report.objective_metrics.mutation_status}` (Killed: {report.objective_metrics.mutation_killed or 0}, Survived: {report.objective_metrics.mutation_survived or 0}, Excluded: {report.objective_metrics.mutation_excluded or 0}) |",
        f"| **Khả năng so sánh (`report_comparable`)** | `{report.report_comparable}` | {report.non_comparable_reason or 'Đủ mọi chỉ số khách quan'} |",
        "",
        "## 2. Chi tiết 8 Tiêu chí Rubric",
        "",
        "| Tiêu chí | Điểm (0–10) | Trọng số gốc | Trọng số hiệu dụng | Nguồn chấm | Ghi chú |",
        "|---|---:|---:|---:|---|---|",
    ]

    for key in RUBRIC_KEYS:
        label = RUBRIC_LABELS.get(key, key)
        src_lbl = CRITERION_SOURCE_LABELS.get(key, "-")
        score_s = _fmt_score(report.scores.get(key))
        w_orig = f"{report.weights.get(key, 0.0) * 100:.0f}%"
        w_eff = f"{report.weights_effective.get(key, 0.0) * 100:.1f}%"
        note = _rubric_note(report, key)
        lines.append(f"| **{label}** (`{key}`) | {score_s} | {w_orig} | {w_eff} | {src_lbl} | {note} |")

    lines.extend(["", "## 3. Lỗ hổng Kiểm thử & Ngữ cảnh Mã nguồn", ""])
    if mutant_rows:
        lines.extend(
            [
                f"### Survived Mutants ({len(mutant_rows)})",
                "",
                "| Dòng | Phép đột biến | Mã nguồn tại dòng lỗi |",
                "|---:|---|---|",
            ]
        )
        for row in mutant_rows:
            code_esc = row["code"].replace("|", "\\|").replace("`", "'") or "(Không đọc được dòng nguồn)"
            lines.append(f"| {row['line']} | `{row['mutation']}` | `{code_esc}` |")
        lines.append("")
    else:
        lines.append("- **Survived Mutants:** Không có mutant sống sót.\n")

    if missing_rows:
        lines.extend(
            [
                f"### Missing Lines (`{compress_line_ranges(report.objective_metrics.missing_lines)}`)",
                "",
                "| Dòng | Mã nguồn chưa được thực thi |",
                "|---:|---|",
            ]
        )
        for row in missing_rows:
            code_esc = row["code"].replace("|", "\\|").replace("`", "'") or "(Không đọc được dòng nguồn)"
            lines.append(f"| {row['line']} | `{code_esc}` |")
        lines.append("")

    lines.extend(
        [
            "## 4. Nhận xét & Gợi ý từ LLM Judge",
            "",
            f"> {report.llm_assessment.summary}",
            "",
        ]
    )
    if report.llm_assessment.issues:
        for idx, issue in enumerate(report.llm_assessment.issues, start=1):
            lines.append(f"{idx}. **{issue.title}**")
            lines.append(f"   - *Mô tả:* {issue.description}")
            lines.append(f"   - *Gợi ý:* {issue.suggestion}")
    else:
        lines.append("- Không ghi nhận vấn đề bổ sung từ LLM Judge.")

    if agent_result is not None:
        mode_lbl = "Baseline A (--baseline)" if getattr(agent_result, "baseline_mode", False) else "Agent B (Feedback-driven)"
        lines.extend(
            [
                "",
                "## 5. Tiến trình Vòng lặp Agent",
                "",
                f"- **Chế độ:** `{mode_lbl}` | **Stop reason:** `{getattr(agent_result, 'stop_reason', '')}`",
                f"- **Tổng LLM Calls:** `{getattr(agent_result, 'total_llm_calls', 0)}` | **Tổng Tokens:** `{getattr(agent_result, 'total_tokens', 0)}` | **Runtime:** `{getattr(agent_result, 'runtime_sec', 0.0):.2f}s` | **Guard Rejections:** `{getattr(agent_result, 'quality_guard_rejections', 0)}`",
                "",
                "| Vòng | Hành động | Chấp nhận | Tests | Coverage | Mutation (Δ) | Weighted Score | LLM Calls | Tokens | Ghi chú |",
                "|---:|---|:---:|---:|---:|---|---:|---:|---:|---|",
            ]
        )
        for item in getattr(agent_result, "history", []):
            row = _format_agent_history_row(item)
            acc = "Yes" if row["accepted"] else "No"
            reason_esc = row["reason"].replace("|", "\\|")
            lines.append(
                f"| {row['iteration']} | `{row['action']}` | {acc} | {row['tests']} | "
                f"{row['coverage']} | {row['mutation']} | {row['score']} | "
                f"{row['llm_calls']} | {row['tokens']} | {reason_esc} |"
            )

    lines.append("")
    return "\n".join(lines)


def _build_rubric_radar_svg(report: EvalReport) -> str:
    """Vẽ biểu đồ SVG Radar 8 trục cho 8 tiêu chí Rubric (không dùng thư viện ngoài)."""
    cx, cy, radius = 175.0, 165.0, 108.0
    n = len(RUBRIC_KEYS)

    def _pt(idx: int, ratio: float) -> tuple[float, float]:
        angle = (-math.pi / 2.0) + (2.0 * math.pi * idx / n)
        return cx + radius * ratio * math.cos(angle), cy + radius * ratio * math.sin(angle)

    grid_rings: list[str] = []
    for ring_pct in (0.25, 0.5, 0.75, 1.0):
        pts = " ".join(f"{_pt(i, ring_pct)[0]:.1f},{_pt(i, ring_pct)[1]:.1f}" for i in range(n))
        grid_rings.append(
            f'<polygon points="{pts}" fill="none" stroke="#e2e8f0" stroke-width="1" />'
        )

    axis_lines: list[str] = []
    label_nodes: list[str] = []
    poly_pts: list[str] = []
    dot_nodes: list[str] = []

    for i, key in enumerate(RUBRIC_KEYS):
        x_outer, y_outer = _pt(i, 1.0)
        axis_lines.append(
            f'<line x1="{cx:.1f}" y1="{cy:.1f}" x2="{x_outer:.1f}" y2="{y_outer:.1f}" stroke="#cbd5e1" stroke-width="1" />'
        )
        raw_s = report.scores.get(key)
        ratio = max(0.0, min(1.0, (raw_s or 0.0) / 10.0))
        vx, vy = _pt(i, ratio)
        poly_pts.append(f"{vx:.1f},{vy:.1f}")
        dot_color = "#94a3b8" if raw_s is None else "#2563eb"
        dot_nodes.append(
            f'<circle cx="{vx:.1f}" cy="{vy:.1f}" r="4" fill="{dot_color}" stroke="#ffffff" stroke-width="1.5" />'
        )

        lx, ly = _pt(i, 1.24)
        anchor = "middle"
        if lx < cx - 18:
            anchor = "end"
        elif lx > cx + 18:
            anchor = "start"
        lbl = html.escape(RUBRIC_LABELS.get(key, key))
        val_txt = f"{raw_s:.1f}" if raw_s is not None else "N/A"
        label_nodes.append(
            f'<text x="{lx:.1f}" y="{ly:.1f}" text-anchor="{anchor}" '
            f'font-size="11" font-weight="600" fill="#334155">{lbl} ({val_txt})</text>'
        )

    data_polygon = (
        f'<polygon points="{" ".join(poly_pts)}" fill="rgba(37, 99, 235, 0.18)" '
        f'stroke="#2563eb" stroke-width="2.2" />'
    )
    return (
        '<svg viewBox="0 0 350 330" width="100%" style="max-width:350px;height:auto;" '
        'role="img" aria-label="Biểu đồ Radar 8 tiêu chí Rubric">'
        + "".join(grid_rings)
        + "".join(axis_lines)
        + data_polygon
        + "".join(dot_nodes)
        + "".join(label_nodes)
        + "</svg>"
    )


def _build_agent_trajectory_svg(agent_result: Any) -> str:
    """Vẽ biểu đồ đường SVG thể hiện tiến trình Weighted Score & Mutation Score qua các vòng eval."""
    eval_points: list[tuple[int, float, float | None]] = []
    for item in getattr(agent_result, "history", []):
        if getattr(item, "action", "") == "eval":
            after = getattr(item, "metrics_after", None) or {}
            w = after.get("weighted_score")
            m = after.get("mutation_score_pct")
            if w is not None:
                eval_points.append(
                    (
                        int(getattr(item, "iteration", len(eval_points) + 1)),
                        float(w),
                        float(m) / 10.0 if m is not None else None,
                    )
                )
    if not eval_points:
        return ""

    w_svg, h_svg = 520.0, 170.0
    pad_l, pad_r, pad_t, pad_b = 42.0, 24.0, 20.0, 32.0
    plot_w = w_svg - pad_l - pad_r
    plot_h = h_svg - pad_t - pad_b

    def _x(idx: int) -> float:
        if len(eval_points) == 1:
            return pad_l + plot_w / 2.0
        return pad_l + (plot_w * idx / (len(eval_points) - 1))

    def _y(val_0_10: float) -> float:
        clamped = max(0.0, min(10.0, val_0_10))
        return pad_t + plot_h * (1.0 - clamped / 10.0)

    parts: list[str] = []
    for tick in (0, 5, 8, 10):
        yy = _y(float(tick))
        stroke = "#94a3b8" if tick == 8 else "#e2e8f0"
        dash = ' stroke-dasharray="4 3"' if tick == 8 else ""
        parts.append(f'<line x1="{pad_l}" y1="{yy:.1f}" x2="{w_svg - pad_r}" y2="{yy:.1f}" stroke="{stroke}" stroke-width="1"{dash} />')
        parts.append(f'<text x="{pad_l - 8}" y="{yy + 4:.1f}" text-anchor="end" font-size="10" fill="#64748b">{tick}</text>')

    w_coords: list[str] = []
    m_coords: list[str] = []
    for i, (iter_no, w_val, m_val) in enumerate(eval_points):
        xx = _x(i)
        yw = _y(w_val)
        w_coords.append(f"{xx:.1f},{yw:.1f}")
        parts.append(f'<text x="{xx:.1f}" y="{h_svg - 8}" text-anchor="middle" font-size="10" fill="#475569">Vòng {iter_no}</text>')
        parts.append(f'<circle cx="{xx:.1f}" cy="{yw:.1f}" r="4.5" fill="#2563eb" stroke="#fff" stroke-width="1.5" />')
        parts.append(f'<text x="{xx:.1f}" y="{yw - 8:.1f}" text-anchor="middle" font-size="10" font-weight="600" fill="#1d4ed8">{w_val:.2f}</text>')
        if m_val is not None:
            ym = _y(m_val)
            m_coords.append(f"{xx:.1f},{ym:.1f}")
            parts.append(f'<circle cx="{xx:.1f}" cy="{ym:.1f}" r="4" fill="#059669" stroke="#fff" stroke-width="1.5" />')

    lines_svg = ""
    if len(w_coords) >= 2:
        lines_svg += f'<polyline points="{" ".join(w_coords)}" fill="none" stroke="#2563eb" stroke-width="2.5" />'
    if len(m_coords) >= 2:
        lines_svg += f'<polyline points="{" ".join(m_coords)}" fill="none" stroke="#059669" stroke-width="2" stroke-dasharray="5 3" />'

    return (
        f'<svg viewBox="0 0 {int(w_svg)} {int(h_svg)}" width="100%" style="max-width:560px;height:auto;" '
        'role="img" aria-label="Tiến trình điểm qua các vòng Agent">'
        + lines_svg
        + "".join(parts)
        + "</svg>"
    )


def _bar_color_hex(score_0_10: float | None) -> str:
    if score_0_10 is None:
        return "#94a3b8"
    if score_0_10 >= 8.0:
        return "#16a34a"
    if score_0_10 >= 6.0:
        return "#d97706"
    return "#dc2626"


def render_eval_html(
    report: EvalReport,
    *,
    agent_result: Any | None = None,
    project_root: Path | None = None,
) -> str:
    """Kết xuất `EvalReport` (và tiến trình Agent nếu có) thành báo cáo HTML độc lập (Single-file Standalone HTML)."""
    source_lines = load_source_lines(report.meta.target_file, project_root=project_root)
    mutant_rows = build_mutant_context_rows(report.objective_metrics.survived_mutants, source_lines)
    missing_rows = build_missing_line_rows(report.objective_metrics.missing_lines, source_lines)

    mutants_by_line: dict[int, list[str]] = {}
    for m in report.objective_metrics.survived_mutants:
        mutants_by_line.setdefault(m.line, []).append(m.mutation)
    missing_set = set(report.objective_metrics.missing_lines)

    if report.passed_eval:
        status_cls, status_text = "badge-pass", "PASSED EVAL"
    elif report.execution.all_tests_passed:
        status_cls, status_text = "badge-warn", "BELOW THRESHOLD"
    else:
        status_cls, status_text = "badge-fail", "TESTS FAILED"

    pass_rate_pct = (
        report.execution.pass_rate * 100.0
        if report.execution.pass_rate is not None
        else None
    )
    judge_str = report.meta.judge_model_actual or report.meta.judge_model_requested or report.meta.model_requested

    rubric_rows_html: list[str] = []
    for key in RUBRIC_KEYS:
        val = report.scores.get(key)
        w_orig = report.weights.get(key, 0.0) * 100.0
        w_eff = report.weights_effective.get(key, 0.0) * 100.0
        pct_width = 0.0 if val is None else max(0.0, min(100.0, val * 10.0))
        color = _bar_color_hex(val)
        note = _rubric_note(report, key)
        cap_pill = (
            ' <span class="pill pill-amber">Capped</span>'
            if key == "correctness" and report.correctness_cap_applied
            else ""
        )
        rubric_rows_html.append(
            "<tr>"
            f"<td><strong>{html.escape(RUBRIC_LABELS.get(key, key))}</strong> <span class=\"muted\">({html.escape(key)})</span>{cap_pill}</td>"
            f"<td><div class=\"bar-track\"><div class=\"bar-fill\" style=\"width:{pct_width:.1f}%;background:{color};\"></div></div></td>"
            f"<td class=\"num\"><strong>{html.escape(_fmt_score(val))}</strong></td>"
            f"<td class=\"num\">{w_orig:.0f}%</td>"
            f"<td class=\"num\">{w_eff:.1f}%</td>"
            f"<td>{html.escape(CRITERION_SOURCE_LABELS.get(key, '-'))}</td>"
            f"<td class=\"muted\">{html.escape(note)}</td>"
            "</tr>"
        )

    # Source Code Viewer hoặc fallback table
    if source_lines:
        code_tr_list: list[str] = []
        for idx, line_text in enumerate(source_lines, start=1):
            is_missing = idx in missing_set
            line_mutants = mutants_by_line.get(idx, [])
            row_cls = ""
            badges: list[str] = []
            if is_missing:
                row_cls = "line-uncovered"
                badges.append('<span class="pill pill-red">UNCOVERED</span>')
            if line_mutants:
                row_cls = row_cls or "line-mutant"
                for mut_op in line_mutants:
                    badges.append(
                        f'<span class="pill pill-amber">⚠ Mutant: {html.escape(mut_op, quote=True)}</span>'
                    )
            badge_html = f'<span class="line-badges">{"".join(badges)}</span>' if badges else ""
            code_tr_list.append(
                f'<tr class="{row_cls}">'
                f'<td class="ln">{idx}</td>'
                f'<td class="code-cell"><pre>{html.escape(line_text, quote=True) or " "}</pre>{badge_html}</td>'
                "</tr>"
            )
        code_viewer_html = (
            '<div class="code-viewer-wrap"><table class="code-viewer">'
            + "".join(code_tr_list)
            + "</table></div>"
        )
    else:
        fallback_parts: list[str] = [
            '<p class="muted">Không tìm thấy file nguồn trên đĩa để dựng khung xem mã nguồn đầy đủ. Dưới đây là danh sách vị trí ghi nhận trong báo cáo:</p>'
        ]
        if mutant_rows:
            trs = "".join(
                f"<tr><td class=\"num\">{r['line']}</td><td><code>{html.escape(r['mutation'], quote=True)}</code></td></tr>"
                for r in mutant_rows
            )
            fallback_parts.append(
                f"<h3>Survived Mutants ({len(mutant_rows)})</h3>"
                f"<table><thead><tr><th>Dòng</th><th>Phép đột biến</th></tr></thead><tbody>{trs}</tbody></table>"
            )
        if missing_rows:
            fallback_parts.append(
                f"<p><strong>Missing Lines:</strong> <code>{html.escape(compress_line_ranges(report.objective_metrics.missing_lines))}</code></p>"
            )
        if not mutant_rows and not missing_rows:
            fallback_parts.append('<p class="pill pill-green">Không có survived mutant hoặc dòng chưa phủ nào.</p>')
        code_viewer_html = "".join(fallback_parts)

    # LLM issues
    if report.llm_assessment.issues:
        issues_html = "".join(
            '<div class="issue-card">'
            f'<div class="issue-title">{idx}. {html.escape(issue.title, quote=True)}</div>'
            f'<p class="issue-desc"><strong>Mô tả:</strong> {html.escape(issue.description, quote=True)}</p>'
            f'<p class="issue-sug"><strong>Gợi ý:</strong> {html.escape(issue.suggestion, quote=True)}</p>'
            "</div>"
            for idx, issue in enumerate(report.llm_assessment.issues, start=1)
        )
    else:
        issues_html = '<p class="muted">LLM Judge không ghi nhận vấn đề bổ sung nào.</p>'

    # Optional Agent section
    agent_section_html = ""
    if agent_result is not None:
        mode_lbl = "Baseline A (--baseline)" if getattr(agent_result, "baseline_mode", False) else "Agent B (Feedback-driven)"
        traj_svg = _build_agent_trajectory_svg(agent_result)
        hist_rows_html: list[str] = []
        for item in getattr(agent_result, "history", []):
            r = _format_agent_history_row(item)
            acc_badge = '<span class="pill pill-green">Yes</span>' if r["accepted"] else '<span class="pill pill-red">No</span>'
            hist_rows_html.append(
                "<tr>"
                f"<td class=\"num\">{html.escape(r['iteration'])}</td>"
                f"<td><code>{html.escape(r['action'])}</code></td>"
                f"<td>{acc_badge}</td>"
                f"<td class=\"num\">{html.escape(r['tests'])}</td>"
                f"<td class=\"num\">{html.escape(r['coverage'])}</td>"
                f"<td>{html.escape(r['mutation'])}</td>"
                f"<td class=\"num\"><strong>{html.escape(r['score'])}</strong></td>"
                f"<td class=\"num\">{html.escape(r['llm_calls'])}</td>"
                f"<td class=\"num\">{html.escape(r['tokens'])}</td>"
                f"<td class=\"muted\">{html.escape(r['reason'])}</td>"
                "</tr>"
            )
        agent_section_html = (
            '<section class="card">'
            "<h2>5. Tiến trình Vòng lặp Agent</h2>"
            '<div class="chips">'
            f'<span class="chip">Chế độ: <strong>{html.escape(mode_lbl)}</strong></span>'
            f'<span class="chip">Stop reason: <code>{html.escape(str(getattr(agent_result, "stop_reason", "")))}</code></span>'
            f'<span class="chip">LLM Calls: <strong>{getattr(agent_result, "total_llm_calls", 0)}</strong></span>'
            f'<span class="chip">Tokens: <strong>{getattr(agent_result, "total_tokens", 0)}</strong></span>'
            f'<span class="chip">Runtime: <strong>{getattr(agent_result, "runtime_sec", 0.0):.2f}s</strong></span>'
            "</div>"
            + (f'<div class="svg-center">{traj_svg}</div>' if traj_svg else "")
            + '<table><thead><tr><th>Vòng</th><th>Hành động</th><th>KQ</th><th>Tests</th><th>Cov</th><th>Mutation (Δ)</th><th>Score</th><th>Calls</th><th>Tokens</th><th>Ghi chú</th></tr></thead>'
            + f'<tbody>{"".join(hist_rows_html)}</tbody></table>'
            "</section>"
        )

    non_comp_banner = ""
    if not report.report_comparable:
        non_comp_banner = (
            '<div class="alert-warn">'
            f'<strong>Lưu ý so sánh (`report_comparable = false`):</strong> '
            f'{html.escape(report.non_comparable_reason or "Thiếu chỉ số khách quan")}'
            "</div>"
        )

    radar_svg = _build_rubric_radar_svg(report)
    title_str = f"aitest Report — {report.meta.target_file}"

    return f"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>{html.escape(title_str, quote=True)}</title>
<style>
  :root {{ --bg:#f8fafc; --card:#ffffff; --border:#e2e8f0; --text:#0f172a; --muted:#64748b; --blue:#2563eb; --green:#16a34a; --amber:#d97706; --red:#dc2626; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; padding:24px; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; background:var(--bg); color:var(--text); line-height:1.5; }}
  .container {{ max-width:1160px; margin:0 auto; display:flex; flex-direction:column; gap:18px; }}
  .card {{ background:var(--card); border:1px solid var(--border); border-radius:12px; padding:20px; box-shadow:0 1px 2px rgba(15,23,42,0.04); }}
  .header-row {{ display:flex; flex-wrap:wrap; justify-content:space-between; align-items:flex-start; gap:12px; }}
  h1 {{ margin:0 0 6px; font-size:22px; }}
  h2 {{ margin:0 0 14px; font-size:17px; border-bottom:1px solid var(--border); padding-bottom:8px; }}
  h3 {{ margin:14px 0 8px; font-size:15px; }}
  .muted {{ color:var(--muted); font-size:13px; }}
  .badge {{ padding:6px 14px; border-radius:999px; font-weight:700; font-size:13px; letter-spacing:0.03em; }}
  .badge-pass {{ background:#dcfce7; color:#166534; border:1px solid #86efac; }}
  .badge-warn {{ background:#fef3c7; color:#92400e; border:1px solid #fcd34d; }}
  .badge-fail {{ background:#fee2e2; color:#991b1b; border:1px solid #fca5a5; }}
  .chips {{ display:flex; flex-wrap:wrap; gap:8px; margin-top:8px; margin-bottom:10px; }}
  .chip {{ background:#f1f5f9; border:1px solid var(--border); border-radius:6px; padding:4px 10px; font-size:12.5px; }}
  .kpi-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(210px,1fr)); gap:14px; }}
  .kpi {{ background:var(--card); border:1px solid var(--border); border-radius:12px; padding:16px; }}
  .kpi-label {{ font-size:12.5px; color:var(--muted); font-weight:600; text-transform:uppercase; }}
  .kpi-val {{ font-size:28px; font-weight:800; margin:6px 0 4px; }}
  .kpi-sub {{ font-size:12.5px; color:var(--muted); }}
  .rubric-layout {{ display:grid; grid-template-columns:minmax(280px,350px) 1fr; gap:20px; align-items:center; }}
  @media (max-width:860px) {{ .rubric-layout {{ grid-template-columns:1fr; }} }}
  table {{ width:100%; border-collapse:collapse; font-size:13.5px; }}
  th, td {{ border-bottom:1px solid var(--border); padding:8px 10px; text-align:left; vertical-align:middle; }}
  th {{ background:#f8fafc; font-weight:600; color:#334155; }}
  td.num, th.num {{ text-align:right; font-variant-numeric:tabular-nums; }}
  .bar-track {{ width:120px; height:9px; background:#e2e8f0; border-radius:999px; overflow:hidden; }}
  .bar-fill {{ height:100%; border-radius:999px; }}
  .pill {{ display:inline-block; padding:2px 8px; border-radius:999px; font-size:11.5px; font-weight:600; margin-left:4px; }}
  .pill-green {{ background:#dcfce7; color:#166534; }}
  .pill-amber {{ background:#fef3c7; color:#92400e; border:1px solid #f59e0b; }}
  .pill-red {{ background:#fee2e2; color:#991b1b; border:1px solid #ef4444; }}
  .alert-warn {{ background:#fffbeb; border:1px solid #fcd34d; color:#92400e; padding:10px 14px; border-radius:8px; font-size:13.5px; }}
  .summary-box {{ background:#eff6ff; border-left:4px solid var(--blue); padding:12px 16px; border-radius:6px; margin-bottom:14px; }}
  .issue-card {{ border:1px solid var(--border); border-radius:8px; padding:12px 14px; margin-bottom:10px; background:#fcfcfd; }}
  .issue-title {{ font-weight:700; margin-bottom:6px; color:#1e293b; }}
  .issue-desc, .issue-sug {{ margin:4px 0; font-size:13.5px; }}
  .code-viewer-wrap {{ max-height:480px; overflow:auto; border:1px solid var(--border); border-radius:8px; }}
  .code-viewer {{ font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; font-size:12.5px; }}
  .code-viewer td {{ padding:2px 8px; border-bottom:1px solid #f1f5f9; }}
  .code-viewer td.ln {{ width:46px; text-align:right; color:#94a3b8; user-select:none; background:#f8fafc; border-right:1px solid var(--border); }}
  .code-viewer td.code-cell {{ display:flex; justify-content:space-between; align-items:center; gap:12px; }}
  .code-viewer pre {{ margin:0; white-space:pre-wrap; word-break:break-word; font-family:inherit; }}
  .line-mutant {{ background:#fffbeb; }}
  .line-uncovered {{ background:#fff1f2; }}
  .svg-center {{ display:flex; justify-content:center; padding:8px 0; }}
  code {{ font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; font-size:12.5px; background:#f1f5f9; padding:1px 5px; border-radius:4px; }}
  @media print {{ body {{ background:#fff; padding:0; }} .card,.kpi {{ box-shadow:none; break-inside:avoid; }} .code-viewer-wrap {{ max-height:none; }} }}
</style>
</head>
<body>
<div class="container">
  <header class="card">
    <div class="header-row">
      <div>
        <h1>Báo cáo Đánh giá Unit Test — <code>{html.escape(report.meta.target_file, quote=True)}</code></h1>
        <div class="muted">File test: <code>{html.escape(report.meta.test_file, quote=True)}</code> &bull; Tạo lúc: {html.escape(report.meta.generated_at)}</div>
      </div>
      <span class="badge {status_cls}">{status_text}</span>
    </div>
    <div class="chips">
      <span class="chip">Tool: <strong>aitest v{html.escape(report.meta.version)}</strong></span>
      <span class="chip">Model: <code>{html.escape(report.meta.model_actual or report.meta.model_requested)}</code></span>
      <span class="chip">Judge: <code>{html.escape(judge_str)}</code></span>
      <span class="chip">Endpoint: <code>{'Vertex AI' if report.meta.vertex else 'Gemini API'}</code></span>
      <span class="chip">Run ID: <code>{html.escape(report.meta.run_id)}</code></span>
      <span class="chip">Conv ID: <code>{html.escape(report.meta.conversation_id or 'N/A')}</code></span>
    </div>
    {non_comp_banner}
  </header>

  <section class="kpi-grid">
    <div class="kpi">
      <div class="kpi-label">Weighted Score</div>
      <div class="kpi-val" style="color:{_bar_color_hex(report.weighted_score)};">{report.weighted_score:.2f} / 10</div>
      <div class="kpi-sub">Ngưỡng đạt: {report.threshold_pass:.2f}</div>
    </div>
    <div class="kpi">
      <div class="kpi-label">Pass Rate</div>
      <div class="kpi-val">{html.escape(_fmt_pct(pass_rate_pct))}</div>
      <div class="kpi-sub">{report.execution.tests_passed} passed &bull; {report.execution.tests_failed} failed &bull; {report.execution.tests_skipped} skipped ({report.execution.duration_ms} ms)</div>
    </div>
    <div class="kpi">
      <div class="kpi-label">Line Coverage</div>
      <div class="kpi-val">{html.escape(_fmt_pct(report.objective_metrics.line_coverage_pct))}</div>
      <div class="kpi-sub">Missing lines: {html.escape(compress_line_ranges(report.objective_metrics.missing_lines))}</div>
    </div>
    <div class="kpi">
      <div class="kpi-label">Mutation Score</div>
      <div class="kpi-val">{html.escape(_fmt_pct(report.objective_metrics.mutation_score_pct))}</div>
      <div class="kpi-sub">Killed: {report.objective_metrics.mutation_killed or 0} &bull; Survived: {report.objective_metrics.mutation_survived or 0} &bull; Excluded: {report.objective_metrics.mutation_excluded or 0}</div>
    </div>
  </section>

  <section class="card">
    <h2>2. Phân tích 8 Tiêu chí Rubric</h2>
    <div class="rubric-layout">
      <div class="svg-center">{radar_svg}</div>
      <table>
        <thead><tr><th>Tiêu chí</th><th>Mức điểm</th><th class="num">Điểm</th><th class="num">Trọng số</th><th class="num">Hiệu dụng</th><th>Nguồn chấm</th><th>Ghi chú</th></tr></thead>
        <tbody>{"".join(rubric_rows_html)}</tbody>
      </table>
    </div>
  </section>

  <section class="card">
    <h2>3. Ngữ cảnh Mã nguồn &amp; Lỗ hổng Kiểm thử</h2>
    <p class="muted">Dòng nền vàng là vị trí có <strong>Survived Mutant</strong>; dòng nền đỏ là <strong>UNCOVERED</strong> chưa được bộ test đi qua.</p>
    {code_viewer_html}
  </section>

  <section class="card">
    <h2>4. Nhận xét &amp; Gợi ý Cải thiện từ LLM Judge</h2>
    <div class="summary-box">{html.escape(report.llm_assessment.summary, quote=True)}</div>
    {issues_html}
  </section>

  {agent_section_html}
</div>
</body>
</html>
"""


def _build_experiment_bar_svg(per_file: list[dict[str, Any]]) -> str:
    """Vẽ biểu đồ cột đôi SVG so sánh Mutation Score (%) Baseline A vs Agent B theo từng file."""
    if not per_file:
        return ""

    w_svg = max(560.0, 120.0 + len(per_file) * 110.0)
    h_svg = 250.0
    pad_l, pad_r, pad_t, pad_b = 48.0, 24.0, 28.0, 56.0
    plot_w = w_svg - pad_l - pad_r
    plot_h = h_svg - pad_t - pad_b

    parts: list[str] = []
    for tick in (0, 25, 50, 75, 100):
        yy = pad_t + plot_h * (1.0 - tick / 100.0)
        parts.append(f'<line x1="{pad_l}" y1="{yy:.1f}" x2="{w_svg - pad_r}" y2="{yy:.1f}" stroke="#e2e8f0" stroke-width="1" />')
        parts.append(f'<text x="{pad_l - 8}" y="{yy + 4:.1f}" text-anchor="end" font-size="10.5" fill="#64748b">{tick}%</text>')

    group_w = plot_w / len(per_file)
    bar_w = min(28.0, group_w * 0.28)
    for i, row in enumerate(per_file):
        center = pad_l + group_w * (i + 0.5)
        val_a = max(0.0, min(100.0, float(row.get("baseline_a_mean_mut_pct", 0.0))))
        val_b = max(0.0, min(100.0, float(row.get("agent_b_mean_mut_pct", 0.0))))
        ha = plot_h * (val_a / 100.0)
        hb = plot_h * (val_b / 100.0)
        ya = pad_t + plot_h - ha
        yb = pad_t + plot_h - hb
        xa = center - bar_w - 3.0
        xb = center + 3.0

        parts.append(f'<rect x="{xa:.1f}" y="{ya:.1f}" width="{bar_w:.1f}" height="{ha:.1f}" rx="4" fill="#f59e0b" />')
        parts.append(f'<text x="{xa + bar_w / 2:.1f}" y="{ya - 5:.1f}" text-anchor="middle" font-size="10" fill="#92400e">{val_a:.1f}</text>')
        parts.append(f'<rect x="{xb:.1f}" y="{yb:.1f}" width="{bar_w:.1f}" height="{hb:.1f}" rx="4" fill="#16a34a" />')
        parts.append(f'<text x="{xb + bar_w / 2:.1f}" y="{yb - 5:.1f}" text-anchor="middle" font-size="10" font-weight="700" fill="#166534">{val_b:.1f}</text>')

        fname = Path(str(row.get("target_file", ""))).name
        parts.append(f'<text x="{center:.1f}" y="{h_svg - 28}" text-anchor="middle" font-size="11" font-weight="600" fill="#334155">{html.escape(fname)}</text>')
        delta = float(row.get("delta_mut_pct", 0.0))
        sign = "+" if delta > 0 else ""
        d_color = "#166534" if delta > 0 else ("#991b1b" if delta < 0 else "#475569")
        parts.append(f'<text x="{center:.1f}" y="{h_svg - 12}" text-anchor="middle" font-size="10" fill="{d_color}">Δ {sign}{delta:.2f}%</text>')

    return (
        f'<svg viewBox="0 0 {int(w_svg)} {int(h_svg)}" width="100%" style="max-width:860px;height:auto;" '
        'role="img" aria-label="Biểu đồ đối sánh Mutation Score Baseline A và Agent B">'
        + "".join(parts)
        + "</svg>"
    )


def render_experiment_html(summary_data: dict[str, Any]) -> str:
    """Kết xuất báo cáo thực nghiệm đối sánh Baseline A vs Agent B thành Standalone HTML."""
    meta = summary_data.get("meta", {})
    b_a = summary_data.get("baseline_a_summary", {})
    b_b = summary_data.get("agent_b_summary", {})
    delta = summary_data.get("delta_analysis", {}) or {}
    wtl = delta.get("win_tie_loss", {"win": 0, "tie": 0, "loss": 0})
    per_file = delta.get("per_file", [])

    def _m(summary: dict[str, Any], key: str) -> str:
        val = summary.get(key)
        if isinstance(val, dict):
            return str(val.get("formatted", "N/A"))
        return str(val if val is not None else "N/A")

    comp_rows = [
        ("Line Coverage (%)", _m(b_a, "line_coverage_pct"), _m(b_b, "line_coverage_pct")),
        ("Mutation Score (%)", _m(b_a, "mutation_score_pct"), _m(b_b, "mutation_score_pct")),
        ("Mutants Killed", _m(b_a, "mutation_killed"), _m(b_b, "mutation_killed")),
        ("Mutants Survived", _m(b_a, "mutation_survived"), _m(b_b, "mutation_survived")),
        ("Số hàm test (test_count)", _m(b_a, "test_count"), _m(b_b, "test_count")),
        ("Tỷ lệ all_tests_passed", f"{b_a.get('all_tests_passed_rate_pct', 0.0):.2f}%", f"{b_b.get('all_tests_passed_rate_pct', 0.0):.2f}%"),
        ("Tỷ lệ report_comparable", f"{b_a.get('report_comparable_rate_pct', 0.0):.2f}%", f"{b_b.get('report_comparable_rate_pct', 0.0):.2f}%"),
        ("Weighted Score", _m(b_a, "weighted_score"), _m(b_b, "weighted_score")),
        ("Quality Guard Rejections", str((b_a.get("quality_guard_rejections") or {}).get("total", 0)), str((b_b.get("quality_guard_rejections") or {}).get("total", 0))),
        ("Số lần gọi LLM", _m(b_a, "llm_calls"), _m(b_b, "llm_calls")),
        ("Tổng số Tokens", _m(b_a, "total_tokens"), _m(b_b, "total_tokens")),
        ("Thời gian thực thi (s)", _m(b_a, "runtime_sec"), _m(b_b, "runtime_sec")),
    ]
    comp_tr = "".join(
        f"<tr><td><strong>{html.escape(k)}</strong></td><td>{html.escape(va)}</td><td>{html.escape(vb)}</td></tr>"
        for k, va, vb in comp_rows
    )

    file_tr = "".join(
        "<tr>"
        f"<td><code>{html.escape(str(r.get('target_file', '')))}</code></td>"
        f"<td class=\"num\">{float(r.get('baseline_a_mean_mut_pct', 0.0)):.2f}%</td>"
        f"<td class=\"num\">{float(r.get('agent_b_mean_mut_pct', 0.0)):.2f}%</td>"
        f"<td class=\"num\">{'+' if float(r.get('delta_mut_pct', 0.0)) > 0 else ''}{float(r.get('delta_mut_pct', 0.0)):.2f}%</td>"
        f"<td><strong>{html.escape(str(r.get('outcome', '')))}</strong></td>"
        "</tr>"
        for r in per_file
    )

    chart_svg = _build_experiment_bar_svg(per_file)

    return f"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Báo cáo Thực nghiệm Đối sánh Baseline A vs Agent B (aitest)</title>
<style>
  body {{ margin:0; padding:24px; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; background:#f8fafc; color:#0f172a; line-height:1.5; }}
  .container {{ max-width:1100px; margin:0 auto; display:flex; flex-direction:column; gap:18px; }}
  .card {{ background:#fff; border:1px solid #e2e8f0; border-radius:12px; padding:20px; }}
  .kpi-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(200px,1fr)); gap:14px; }}
  .kpi {{ background:#fff; border:1px solid #e2e8f0; border-radius:12px; padding:16px; }}
  .kpi-label {{ font-size:12px; font-weight:600; color:#64748b; text-transform:uppercase; }}
  .kpi-val {{ font-size:24px; font-weight:800; margin-top:6px; }}
  .chips {{ display:flex; flex-wrap:wrap; gap:8px; margin-top:8px; }}
  .chip {{ background:#f1f5f9; border:1px solid #e2e8f0; border-radius:6px; padding:4px 10px; font-size:12.5px; }}
  table {{ width:100%; border-collapse:collapse; font-size:13.5px; }}
  th, td {{ border-bottom:1px solid #e2e8f0; padding:8px 10px; text-align:left; }}
  th {{ background:#f8fafc; font-weight:600; }}
  td.num, th.num {{ text-align:right; font-variant-numeric:tabular-nums; }}
  .legend {{ display:flex; gap:16px; font-size:13px; margin-bottom:8px; }}
  .dot {{ display:inline-block; width:11px; height:11px; border-radius:3px; margin-right:5px; vertical-align:middle; }}
  code {{ font-family:ui-monospace,Consolas,monospace; background:#f1f5f9; padding:1px 5px; border-radius:4px; font-size:12.5px; }}
</style>
</head>
<body>
<div class="container">
  <header class="card">
    <h1 style="margin:0 0 6px;font-size:22px;">Báo cáo Thực nghiệm Đối sánh: Baseline A vs Agent B (<code>aitest</code>)</h1>
    <div class="chips">
      <span class="chip">Thời gian: <strong>{html.escape(str(meta.get('generated_at', '')))}</strong></span>
      <span class="chip">Provider: <code>{html.escape(str(meta.get('provider', 'antigravity')))}</code></span>
      <span class="chip">Model: <code>{html.escape(str(meta.get('model', '')))}</code></span>
      <span class="chip">Judge: <code>{html.escape(str(meta.get('judge_model') or meta.get('model', '')))}</code></span>
      <span class="chip">Endpoint: <code>{'Vertex AI' if meta.get('vertex') else 'Gemini API'}</code></span>
      <span class="chip">Runs/file: <strong>{meta.get('runs_per_file', '')}</strong></span>
      <span class="chip">Threshold: <strong>{meta.get('threshold_pass', '')}</strong></span>
    </div>
  </header>

  <section class="kpi-grid">
    <div class="kpi"><div class="kpi-label">Mutation Score (Baseline A)</div><div class="kpi-val" style="color:#d97706;">{html.escape(_m(b_a, 'mutation_score_pct'))}</div></div>
    <div class="kpi"><div class="kpi-label">Mutation Score (Agent B)</div><div class="kpi-val" style="color:#16a34a;">{html.escape(_m(b_b, 'mutation_score_pct'))}</div></div>
    <div class="kpi"><div class="kpi-label">Weighted Score (A vs B)</div><div class="kpi-val">{html.escape(_m(b_a, 'weighted_score'))} → {html.escape(_m(b_b, 'weighted_score'))}</div></div>
    <div class="kpi"><div class="kpi-label">Win / Tie / Loss (theo File)</div><div class="kpi-val">{wtl.get('win', 0)} / {wtl.get('tie', 0)} / {wtl.get('loss', 0)}</div></div>
  </section>

  {f'<section class="card"><h2 style="margin:0 0 10px;font-size:17px;">Biểu đồ Mutation Score (%) theo từng File</h2><div class="legend"><span><span class="dot" style="background:#f59e0b;"></span>Baseline A</span><span><span class="dot" style="background:#16a34a;"></span>Agent B</span></div>{chart_svg}</section>' if chart_svg else ''}

  <section class="card">
    <h2 style="margin:0 0 10px;font-size:17px;">Bảng Đối sánh Tổng hợp (Mean ± SD)</h2>
    <table><thead><tr><th>Chỉ số</th><th>Baseline A (--baseline)</th><th>Agent B (Feedback-driven)</th></tr></thead><tbody>{comp_tr}</tbody></table>
  </section>

  {f'<section class="card"><h2 style="margin:0 0 10px;font-size:17px;">Chi tiết Mức cải thiện theo File</h2><table><thead><tr><th>File nguồn</th><th class="num">Baseline A (%)</th><th class="num">Agent B (%)</th><th class="num">Δ Mutation</th><th>Kết quả</th></tr></thead><tbody>{file_tr}</tbody></table></section>' if file_tr else ''}
</div>
</body>
</html>
"""


def render_experiment_csv(summary_data: dict[str, Any]) -> str:
    """Kết xuất số liệu thực nghiệm Baseline A vs Agent B thành chuỗi CSV chuẩn cho Excel."""
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")

    b_a = summary_data.get("baseline_a_summary", {})
    b_b = summary_data.get("agent_b_summary", {})
    delta = summary_data.get("delta_analysis", {}) or {}
    per_file = delta.get("per_file", [])

    def _m_sd(summary: dict[str, Any], key: str) -> tuple[Any, Any, str]:
        val = summary.get(key)
        if isinstance(val, dict):
            return val.get("mean", ""), val.get("sd", ""), str(val.get("formatted", ""))
        return val if val is not None else "", "", str(val if val is not None else "")

    writer.writerow(["Section", "Metric", "Baseline_A_Mean", "Baseline_A_SD", "Baseline_A_Formatted", "Agent_B_Mean", "Agent_B_SD", "Agent_B_Formatted"])
    for label, key in [
        ("Line Coverage (%)", "line_coverage_pct"),
        ("Mutation Score (%)", "mutation_score_pct"),
        ("Mutants Killed", "mutation_killed"),
        ("Mutants Survived", "mutation_survived"),
        ("Test Count", "test_count"),
        ("Weighted Score", "weighted_score"),
        ("LLM Calls", "llm_calls"),
        ("Total Tokens", "total_tokens"),
        ("Runtime (s)", "runtime_sec"),
    ]:
        am, asd, af = _m_sd(b_a, key)
        bm, bsd, bf = _m_sd(b_b, key)
        writer.writerow(["Summary", label, am, asd, af, bm, bsd, bf])

    writer.writerow(["Summary", "All Tests Passed Rate (%)", b_a.get("all_tests_passed_rate_pct", ""), "", f"{b_a.get('all_tests_passed_rate_pct', 0.0):.2f}%", b_b.get("all_tests_passed_rate_pct", ""), "", f"{b_b.get('all_tests_passed_rate_pct', 0.0):.2f}%"])
    writer.writerow(["Summary", "Report Comparable Rate (%)", b_a.get("report_comparable_rate_pct", ""), "", f"{b_a.get('report_comparable_rate_pct', 0.0):.2f}%", b_b.get("report_comparable_rate_pct", ""), "", f"{b_b.get('report_comparable_rate_pct', 0.0):.2f}%"])

    if per_file:
        writer.writerow([])
        writer.writerow(["Section", "Target_File", "Baseline_A_Mean_Mutation_Pct", "Agent_B_Mean_Mutation_Pct", "Delta_Mutation_Pct", "Outcome"])
        for row in per_file:
            writer.writerow(
                [
                    "PerFileDelta",
                    row.get("target_file", ""),
                    row.get("baseline_a_mean_mut_pct", ""),
                    row.get("agent_b_mean_mut_pct", ""),
                    row.get("delta_mut_pct", ""),
                    row.get("outcome", ""),
                ]
            )

    return buf.getvalue()
