"""Unit tests cho aitest.runner và bộ phân loại lỗi 7 nhóm (Giai đoạn 3 - AC-05, AC-06, AC-15, AC-20)."""

from __future__ import annotations

from pathlib import Path

from aitest.config import AitestConfig
from aitest.runner import classify_pytest_error, run_tests
from aitest.schemas import ErrorCategory


def test_runner_extracts_target_coverage_and_missing_lines_ac05(tmp_path: Path) -> None:
    """Kiểm tra AC-05: Trích xuất đúng line_coverage_pct và missing_lines của riêng file nguồn mục tiêu."""
    src_file = tmp_path / "math_ops.py"
    src_file.write_text(
        "def classify_num(x: int) -> str:\n"
        "    if x > 0:\n"
        "        return 'pos'\n"
        "    if x < 0:\n"
        "        return 'neg'\n"
        "    return 'zero'\n",
        encoding="utf-8",
    )
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    test_file = tests_dir / "test_math_ops.py"
    # Chỉ test nhánh x > 0, bỏ trống nhánh x < 0 và zero (dòng 5, 6 chưa phủ)
    test_file.write_text(
        "from math_ops import classify_num\n\n"
        "def test_positive():\n"
        "    assert classify_num(5) == 'pos'\n",
        encoding="utf-8",
    )

    cfg = AitestConfig()
    res = run_tests(
        target_file=src_file,
        config=cfg,
        test_file=test_file,
        project_root=tmp_path,
        enable_coverage=True,
        enable_mutation=False,
    )
    assert res.all_tests_passed is True
    assert res.tests_passed == 1
    assert res.pass_rate == 1.0
    assert res.line_coverage_pct is not None
    assert 40.0 <= res.line_coverage_pct < 100.0
    assert 5 in res.missing_lines or 6 in res.missing_lines


def test_error_classifier_7_categories_ac06(tmp_path: Path) -> None:
    """Kiểm tra AC-06: Phân loại chính xác ENVIRONMENT_ERROR, SOURCE_ERROR, TEST_SYNTAX, TEST_FAILURE, TIMEOUT."""
    target = tmp_path / "service.py"
    target.write_text("def run():\n    return 1\n", encoding="utf-8")
    test_f = tmp_path / "test_service.py"
    test_f.write_text("def test_run():\n    assert True\n", encoding="utf-8")

    # 1. ENVIRONMENT_ERROR (ImportError / ModuleNotFoundError)
    cat, detail = classify_pytest_error(
        exit_code=2,
        stdout="test_service.py:1: in <module>\n    import nonexistent_pkg\nE   ModuleNotFoundError: No module named 'nonexistent_pkg'",
        stderr="",
        target_file=target,
        test_file=test_f,
    )
    assert cat == ErrorCategory.ENVIRONMENT_ERROR
    assert cat.is_fixable_by_llm is False

    # 2. SOURCE_ERROR (Bug nội tại trong file nguồn, ví dụ ZeroDivisionError tại frame cuối của service.py)
    cat_src, _ = classify_pytest_error(
        exit_code=1,
        stdout=(
            "test_service.py:4: in test_run\n"
            "    assert compute(5) == 10\n"
            "service.py:8: in compute\n"
            "    return x / 0\n"
            "E   ZeroDivisionError: division by zero"
        ),
        stderr="",
        target_file=target,
        test_file=test_f,
    )
    assert cat_src == ErrorCategory.SOURCE_ERROR
    assert cat_src.is_fixable_by_llm is False

    # 3. TEST_FAILURE (AssertionError trong file test)
    cat_fail, _ = classify_pytest_error(
        exit_code=1,
        stdout=(
            "test_service.py:4: in test_run\n"
            "    assert run() == 2\n"
            "E   AssertionError: assert 1 == 2"
        ),
        stderr="",
        target_file=target,
        test_file=test_f,
    )
    assert cat_fail == ErrorCategory.TEST_FAILURE
    assert cat_fail.is_fixable_by_llm is True

    # 4. TEST_SYNTAX (Lỗi cú pháp trong file test)
    bad_syntax_test = tmp_path / "test_bad_syntax.py"
    bad_syntax_test.write_text("def test_broken(\n    assert 1 == 1\n", encoding="utf-8")
    cat_syn, _ = classify_pytest_error(
        exit_code=2,
        stdout="SyntaxError: '(' was never closed",
        stderr="",
        target_file=target,
        test_file=bad_syntax_test,
    )
    assert cat_syn == ErrorCategory.TEST_SYNTAX
    assert cat_syn.is_fixable_by_llm is True

    # 5. TIMEOUT
    cat_to, _ = classify_pytest_error(
        exit_code=124,
        stdout="",
        stderr="",
        target_file=target,
        test_file=test_f,
        timed_out=True,
    )
    assert cat_to == ErrorCategory.TIMEOUT
    assert cat_to.is_fixable_by_llm is False


def test_runner_timeout_kills_process_and_sets_null_pass_rate_ac15_ac20(tmp_path: Path) -> None:
    """Kiểm tra AC-15 & AC-20: Tiến trình pytest vượt quá timeout bị hủy sạch, trả TIMEOUT và pass_rate = None."""
    src_file = tmp_path / "slow.py"
    src_file.write_text(
        "import time\n\ndef hang() -> int:\n    time.sleep(10)\n    return 1\n",
        encoding="utf-8",
    )
    test_file = tmp_path / "test_slow.py"
    test_file.write_text(
        "from slow import hang\n\ndef test_hang():\n    assert hang() == 1\n",
        encoding="utf-8",
    )

    cfg = AitestConfig(timeout_sec=1)
    res = run_tests(
        target_file=src_file,
        config=cfg,
        test_file=test_file,
        project_root=tmp_path,
        enable_coverage=False,
        enable_mutation=False,
        timeout_sec=1,
    )
    assert res.all_tests_passed is False
    assert res.error_category == ErrorCategory.TIMEOUT
    assert res.pass_rate is None


def test_classifier_distinguishes_explicit_domain_raise_from_internal_source_bug(
    tmp_path: Path,
) -> None:
    """Kiểm tra P1.7: Lệnh `raise` tường minh trong file nguồn -> TEST_FAILURE; crash ngoài ý muốn -> SOURCE_ERROR."""
    target = tmp_path / "inv.py"
    target.write_text(
        "def get_item(items: dict, key: str) -> int:\n"
        "    if key not in items:\n"
        "        raise KeyError(f'Missing {key}')\n"
        "    return items[key]\n\n"
        "def buggy_ratio(a: int, b: int) -> float:\n"
        "    return a / b\n",
        encoding="utf-8",
    )
    test_f = tmp_path / "test_inv.py"
    test_f.write_text("def test_inv():\n    assert True\n", encoding="utf-8")

    # Ngoại lệ nghiệp vụ tại dòng 3 (lệnh raise KeyError tường minh) -> TEST_FAILURE
    cat_explicit, _ = classify_pytest_error(
        exit_code=1,
        stdout=(
            "test_inv.py:3: in test_inv\n"
            "    get_item({}, 'x')\n"
            "inv.py:3: in get_item\n"
            "    raise KeyError(f'Missing {key}')\n"
            "E   KeyError: 'Missing x'"
        ),
        stderr="",
        target_file=target,
        test_file=test_f,
    )
    assert cat_explicit == ErrorCategory.TEST_FAILURE
    assert cat_explicit.is_fixable_by_llm is True

    # Lỗi chia cho 0 ngoài ý muốn tại dòng 7 (không phải lệnh raise) -> SOURCE_ERROR
    cat_bug, _ = classify_pytest_error(
        exit_code=1,
        stdout=(
            "test_inv.py:6: in test_inv\n"
            "    buggy_ratio(1, 0)\n"
            "inv.py:7: in buggy_ratio\n"
            "    return a / b\n"
            "E   ZeroDivisionError: division by zero"
        ),
        stderr="",
        target_file=target,
        test_file=test_f,
    )
    assert cat_bug == ErrorCategory.SOURCE_ERROR
    assert cat_bug.is_fixable_by_llm is False
