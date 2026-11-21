"""Thực thi pytest + pytest-cov, trích xuất coverage riêng của file nguồn mục tiêu và phân loại lỗi 7 nhóm (SRS v2.3.0 §7 FR-03, FR-04, AC-05, AC-06, AC-15, AC-20)."""

from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from aitest.analyzer import analyze_test_file
from aitest.config import AitestConfig
from aitest.mutation import run_mutation_testing
from aitest.schemas import ErrorCategory, MutationResult, RunResult
from aitest.utils import resolve_test_file_path, validate_single_python_file

_SUMMARY_COUNT_RE = re.compile(
    r"(\d+)\s+(passed|failed|skipped|error|errors|xfailed|xpassed)",
    re.IGNORECASE,
)


def _kill_process_tree(pid: int) -> None:
    """Hủy toàn bộ cây tiến trình con khi pytest vượt quá timeout (AC-15)."""
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(pid)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    else:
        try:
            os.kill(pid, 9)
        except OSError:
            pass


def parse_pytest_summary(output: str, exit_code: int) -> tuple[int, int, int, int]:
    """Trích xuất `(tests_passed, tests_failed, tests_skipped, errors)` từ đầu ra của pytest."""
    passed = 0
    failed = 0
    skipped = 0
    errors = 0

    # Ưu tiên quét từ các dòng cuối cùng của pytest summary
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    summary_Matches: list[tuple[str, str]] = []
    for line in reversed(lines[-15:]):
        found = _SUMMARY_COUNT_RE.findall(line)
        if found:
            summary_Matches = found
            break

    if not summary_Matches:
        summary_Matches = _SUMMARY_COUNT_RE.findall(output)

    for count_str, label in summary_Matches:
        cnt = int(count_str)
        low = label.lower()
        if low == "passed":
            passed = cnt
        elif low == "failed":
            failed = cnt
        elif low in ("skipped", "xfailed", "xpassed"):
            skipped += cnt
        elif low in ("error", "errors"):
            errors += cnt

    # Fallback: đếm từ ký tự rút gọn `.FEs` của pytest -q nếu không có dòng chữ summary
    if passed == 0 and failed == 0 and skipped == 0 and errors == 0 and exit_code in (0, 1):
        for line in lines:
            m = re.match(r"^([.FEskxX]+)\s+\[\s*\d+%\]$", line)
            if m:
                chars = m.group(1)
                passed += chars.count(".")
                failed += chars.count("F")
                errors += chars.count("E")
                skipped += chars.count("s") + chars.count("x") + chars.count("X")

    return passed, failed, skipped, errors


def extract_target_file_coverage(
    cov_json_path: Path | str,
    target_file: Path | str,
    *,
    include_branch: bool = False,
) -> tuple[float | None, float | None, list[int]]:
    """Đọc file JSON của pytest-cov và lọc đúng entry của file nguồn mục tiêu `<file.py>` (FR-03, AC-05).

    Tuyệt đối không lấy tổng `totals` bao gồm cả file test.
    """
    c_path = Path(cov_json_path)
    if not c_path.is_file():
        return None, None, []

    try:
        data = json.loads(c_path.read_text(encoding="utf-8"))
    except Exception:
        return None, None, []

    files_map = data.get("files", {})
    if not isinstance(files_map, dict):
        return None, None, []

    target_norm = os.path.normcase(os.path.abspath(str(target_file)))
    target_name = Path(target_file).name.lower()

    matched_entry: dict | None = None
    for recorded_path, entry in files_map.items():
        rec_norm = os.path.normcase(os.path.abspath(recorded_path))
        if rec_norm == target_norm:
            matched_entry = entry
            break

    # Fallback nếu đường dẫn tương đối trong coverage json được ghi từ một working dir khác
    if matched_entry is None:
        for recorded_path, entry in files_map.items():
            if Path(recorded_path).name.lower() == target_name and not Path(recorded_path).name.lower().startswith("test_"):
                matched_entry = entry
                break

    if not isinstance(matched_entry, dict):
        return None, None, []

    summary = matched_entry.get("summary", {})
    if not isinstance(summary, dict):
        return None, None, []

    num_statements = int(summary.get("num_statements", 0))
    covered_lines = int(summary.get("covered_lines", 0))
    if num_statements > 0:
        line_cov = round((covered_lines / num_statements) * 100.0, 2)
    elif "percent_covered" in summary:
        line_cov = round(float(summary["percent_covered"]), 2)
    else:
        line_cov = 100.0

    branch_cov: float | None = None
    if include_branch:
        num_branches = int(summary.get("num_branches", 0))
        covered_branches = int(summary.get("covered_branches", 0))
        if num_branches > 0:
            branch_cov = round((covered_branches / num_branches) * 100.0, 2)
        else:
            branch_cov = 100.0

    missing_lines_raw = matched_entry.get("missing_lines", [])
    missing_lines = (
        sorted(int(x) for x in missing_lines_raw if isinstance(x, int))
        if isinstance(missing_lines_raw, list)
        else []
    )
    return line_cov, branch_cov, missing_lines


def classify_pytest_error(
    *,
    exit_code: int,
    stdout: str,
    stderr: str,
    target_file: Path | str,
    test_file: Path | str,
    timed_out: bool = False,
) -> tuple[ErrorCategory, str | None]:
    """Phân loại lỗi thực thi pytest thành 7 nhóm trước khi gọi LLM sửa test (FR-04, AC-06).

    1. `TIMEOUT`: Quá thời gian thực thi.
    2. `TEST_SYNTAX`: SyntaxError / IndentationError trong file test.
    3. `ENVIRONMENT_ERROR`: ImportError / ModuleNotFoundError / thiếu thư viện.
    4. `SOURCE_ERROR`: Bug runtime nội tại nằm ở frame cuối của file nguồn mục tiêu (không phải AssertionError).
    5. `TEST_FAILURE`: AssertionError hoặc lỗi gọi hàm/kỳ vọng sai từ phía file test.
    6. `COLLECTION_ERROR`: Lỗi collect test khác (exit code 2 hoặc 4).
    7. `UNKNOWN_ERROR`: Không khớp các mẫu trên.
    """
    if timed_out:
        return ErrorCategory.TIMEOUT, "Quá thời gian thực thi bộ test (TimeoutExpired)"

    if exit_code == 0:
        return ErrorCategory.NONE, None

    combined = f"{stdout}\n{stderr}"
    test_path = Path(test_file)
    target_path = Path(target_file)

    # 1. Kiểm tra trực tiếp cú pháp AST của file test
    if test_path.is_file():
        try:
            ast.parse(test_path.read_text(encoding="utf-8"), filename=str(test_path))
        except (SyntaxError, IndentationError) as exc:
            return ErrorCategory.TEST_SYNTAX, f"Lỗi cú pháp trong file test '{test_path.name}': {exc}"

    # 2. Kiểm tra SyntaxError / IndentationError trong output pytest
    if re.search(r"\b(SyntaxError|IndentationError|TabError)\b", combined):
        # Nếu lỗi cú pháp nằm ở file nguồn thì là SOURCE_ERROR, ngược lại ở file test là TEST_SYNTAX
        if target_path.name in combined and test_path.name not in combined:
            return ErrorCategory.SOURCE_ERROR, f"Lỗi cú pháp trong file nguồn '{target_path.name}'"
        return ErrorCategory.TEST_SYNTAX, "Phát hiện SyntaxError / IndentationError trong file test"

    # 3. Kiểm tra ImportError / ModuleNotFoundError -> ENVIRONMENT_ERROR (AC-06)
    import_match = re.search(
        r"\b(ModuleNotFoundError|ImportError)\s*:\s*(.+)",
        combined,
    )
    if import_match:
        return (
            ErrorCategory.ENVIRONMENT_ERROR,
            f"{import_match.group(1)}: {import_match.group(2).strip()}",
        )

    # 4. Phân tích traceback để phân biệt SOURCE_ERROR (bug nội tại mã nguồn) và TEST_FAILURE
    # Tìm tất cả các vị trí file trong traceback dạng `path/to/file.py:123: in func_name`
    tb_frames = re.findall(r"([^\s:<>\"'|?*]+\.py):(\d+):", combined)
    has_assertion_error = bool(re.search(r"\bAssertionError\b", combined))
    has_failed_did_not_raise = "DID NOT RAISE" in combined.upper()

    if has_assertion_error or has_failed_did_not_raise:
        return ErrorCategory.TEST_FAILURE, "AssertionError hoặc kiểm tra ngoại lệ thất bại trong hàm test"

    # Kiểm tra các lỗi runtime nội tại phát sinh tại frame cuối thuộc file nguồn mục tiêu
    internal_exc_match = re.search(
        r"\b(ZeroDivisionError|IndexError|KeyError|AttributeError|NameError|UnboundLocalError|RecursionError|RuntimeError|ArithmeticError|LookupError)\s*:\s*(.*)",
        combined,
    )
    if tb_frames:
        last_frame_file = Path(tb_frames[-1][0]).name.lower()
        target_name_low = target_path.name.lower()
        test_name_low = test_path.name.lower()

        if last_frame_file == target_name_low and internal_exc_match:
            exc_name = internal_exc_match.group(1)
            exc_msg = internal_exc_match.group(2).strip()
            return (
                ErrorCategory.SOURCE_ERROR,
                f"Lỗi nội tại phát sinh từ file mã nguồn '{target_path.name}' ({exc_name}: {exc_msg})",
            )

        # Nếu TypeError phát sinh ở frame cuối bên trong thân hàm nguồn (không phải do test truyền sai số lượng argument ở dòng gọi hàm)
        type_err_match = re.search(r"\bTypeError\s*:\s*(.*)", combined)
        if type_err_match:
            err_msg = type_err_match.group(1).strip()
            is_call_signature_mismatch = any(
                phrase in err_msg
                for phrase in (
                    "positional argument",
                    "keyword argument",
                    "required argument",
                    "takes ",
                    "got an unexpected",
                )
            )
            if last_frame_file == target_name_low and not is_call_signature_mismatch:
                return (
                    ErrorCategory.SOURCE_ERROR,
                    f"TypeError nội tại trong file mã nguồn '{target_path.name}': {err_msg}",
                )
            if last_frame_file == test_name_low or is_call_signature_mismatch:
                return (
                    ErrorCategory.TEST_FAILURE,
                    f"Hàm test gọi sai chữ ký hàm nguồn (TypeError: {err_msg})",
                )

        if last_frame_file == test_name_low:
            return ErrorCategory.TEST_FAILURE, "Lỗi phát sinh tại frame của file test"

    if exit_code in (2, 4):
        return ErrorCategory.COLLECTION_ERROR, f"Pytest không thể thu thập test (exit_code={exit_code})"

    if exit_code == 5:
        return ErrorCategory.TEST_FAILURE, "Không có hàm test nào được thu thập (exit_code=5)"

    return ErrorCategory.UNKNOWN_ERROR, f"Lỗi không xác định khi chạy pytest (exit_code={exit_code})"


def run_tests(
    target_file: Path | str,
    config: AitestConfig,
    *,
    test_file: Path | str | None = None,
    project_root: Path | str | None = None,
    enable_coverage: bool | None = None,
    enable_mutation: bool | None = None,
    timeout_sec: int | None = None,
) -> RunResult:
    """Thực thi bộ unit test cho `<target_file>`, thu thập chỉ số khách quan, coverage và mutation (FR-03)."""
    validated_target = validate_single_python_file(target_file)
    root_dir = Path(project_root).resolve() if project_root else Path.cwd().resolve()
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

    # Đếm số hàm test_* bằng AST (nếu file test hợp lệ cú pháp)
    test_count = 0
    try:
        t_analysis = analyze_test_file(resolved_test)
        test_count = t_analysis.test_count
    except SyntaxError:
        test_count = 0

    use_cov = config.coverage.enabled if enable_coverage is None else enable_coverage
    use_mut = config.mutation.enabled if enable_mutation is None else enable_mutation
    effective_timeout = timeout_sec if timeout_sec is not None else config.timeout_sec

    # Thiết lập PYTHONPATH gồm project_root và thư mục chứa file nguồn (FR-03)
    env = os.environ.copy()
    pythonpath_entries: list[str] = [
        str(root_dir),
        str(validated_target.resolve().parent),
    ]
    if env.get("PYTHONPATH"):
        pythonpath_entries.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(pythonpath_entries))

    tmp_cov_fd, tmp_cov_name = tempfile.mkstemp(prefix=".aitest_cov_", suffix=".json")
    os.close(tmp_cov_fd)
    tmp_cov_path = Path(tmp_cov_name)

    cmd: list[str] = [
        sys.executable,
        "-m",
        "pytest",
        str(resolved_test),
        "--tb=short",
        "-q",
    ]
    if use_cov:
        cov_target_dir = str(validated_target.resolve().parent)
        cmd.extend(
            [
                f"--cov={cov_target_dir}",
                f"--cov-report=json:{tmp_cov_path}",
            ]
        )
        if config.coverage.branch:
            cmd.append("--cov-branch")
    else:
        cmd.extend(["-p", "no:cov"])

    start_ns = time.perf_counter_ns()
    timed_out = False
    stdout_str = ""
    stderr_str = ""
    exit_code = -1

    proc: subprocess.Popen[str] | None = None
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(root_dir),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        try:
            stdout_str, stderr_str = proc.communicate(timeout=float(effective_timeout))
            exit_code = proc.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
            _kill_process_tree(proc.pid)
            try:
                out_rem, err_rem = proc.communicate(timeout=3.0)
                stdout_str = out_rem or ""
                stderr_str = err_rem or ""
            except Exception:
                pass
            exit_code = 124
    finally:
        duration_ms = max(0, int((time.perf_counter_ns() - start_ns) / 1_000_000))

    passed, failed, skipped, errors = (
        (0, 0, 0, 0)
        if timed_out
        else parse_pytest_summary(f"{stdout_str}\n{stderr_str}", exit_code)
    )

    total_executed = passed + failed
    pass_rate: float | None = (
        round(passed / total_executed, 4) if total_executed > 0 else None
    )
    all_tests_passed = (
        not timed_out
        and exit_code == 0
        and passed > 0
        and failed == 0
        and errors == 0
    )

    line_cov: float | None = None
    branch_cov: float | None = None
    missing_lines: list[int] = []
    try:
        if use_cov and not timed_out:
            line_cov, branch_cov, missing_lines = extract_target_file_coverage(
                tmp_cov_path,
                validated_target,
                include_branch=config.coverage.branch,
            )
    finally:
        if tmp_cov_path.exists():
            try:
                tmp_cov_path.unlink()
            except OSError:
                pass

    error_cat = ErrorCategory.NONE
    error_detail: str | None = None
    if not all_tests_passed:
        error_cat, error_detail = classify_pytest_error(
            exit_code=exit_code,
            stdout=stdout_str,
            stderr=stderr_str,
            target_file=validated_target,
            test_file=resolved_test,
            timed_out=timed_out,
        )

    # Chạy Mutation Testing khi và chỉ khi all_tests_passed == True (FR-03 & FR-07)
    mutation_res: MutationResult
    if not use_mut:
        mutation_res = MutationResult(
            mutation_status="disabled",
            mutation_available=False,
            mutation_unavailable_reason="Mutation testing bị tắt qua cấu hình hoặc cờ --no-mutation",
        )
    elif not all_tests_passed:
        mutation_res = MutationResult(
            mutation_status="disabled",
            mutation_available=False,
            mutation_unavailable_reason="Không chạy mutation testing vì bộ test hiện tại chưa pass 100%",
        )
    else:
        mutation_res = run_mutation_testing(
            target_file=validated_target,
            test_file=resolved_test,
            config=config,
            project_root=root_dir,
        )

    return RunResult(
        target_file=str(target_file),
        test_file=str(resolved_test),
        exit_code=exit_code,
        stdout=stdout_str,
        stderr=stderr_str,
        duration_ms=duration_ms,
        all_tests_passed=all_tests_passed,
        tests_passed=passed,
        tests_failed=failed,
        tests_skipped=skipped,
        errors=errors,
        pass_rate=pass_rate,
        line_coverage_pct=line_cov,
        branch_coverage_pct=branch_cov,
        missing_lines=missing_lines,
        test_count=test_count,
        error_category=error_cat,
        error_detail=error_detail,
        mutation=mutation_res,
    )
