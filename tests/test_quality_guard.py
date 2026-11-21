"""Unit tests cho aitest.quality_guard (Giai đoạn 2 - AC-07, NFR-15)."""

from __future__ import annotations

import pytest

from aitest.quality_guard import check_quality_guard

BASE_TEST_CODE = """\
import pytest
from calculator import add, divide

def test_add():
    assert add(2, 3) == 6

def test_divide():
    assert divide(10, 2) == 5.0
    with pytest.raises(ValueError):
        divide(10, 0)
"""


def test_quality_guard_allows_fixing_wrong_expected_constant_in_eq_ac07() -> None:
    """Cho phép sửa hằng số kỳ vọng bị đoán sai trong phép `==` và ghi log assertion_diff."""
    fixed_code = """\
import pytest
from calculator import add, divide

def test_add():
    assert add(2, 3) == 5

def test_divide():
    assert divide(10, 2) == 5.0
    with pytest.raises(ValueError):
        divide(10, 0)
"""
    res = check_quality_guard(BASE_TEST_CODE, fixed_code)
    assert res.passed is True
    assert res.syntax_ok is True
    assert len(res.assertion_diffs) == 1
    assert res.assertion_diffs[0].classification == "allowed_value_update"
    assert res.assertion_diffs[0].test_name == "test_add"


@pytest.mark.parametrize(
    "weakened_assert",
    [
        "assert True",
        "assert not False",
        "assert len([1, 2]) >= 0",
        "assert add(2, 3) == add(2, 3)",
        "assert add(2, 3) is not None",
        "assert isinstance(add(2, 3), int)",
        "assert add(2, 3) != 0",
        "assert add(2, 3) >= 5",
        "assert add(2, 3) <= 6",
    ],
)
def test_quality_guard_rejects_tautology_and_operator_relaxation_ac07(weakened_assert: str) -> None:
    """Chặn mọi hành vi thay thế phép `==` bằng tautology, `is not None`, hoặc nới lỏng toán tử."""
    bad_code = f"""\
import pytest
from calculator import add, divide

def test_add():
    {weakened_assert}

def test_divide():
    assert divide(10, 2) == 5.0
    with pytest.raises(ValueError):
        divide(10, 0)
"""
    res = check_quality_guard(BASE_TEST_CODE, bad_code)
    assert res.passed is False
    assert any(d.classification == "weakened" for d in res.assertion_diffs)


def test_quality_guard_rejects_deleted_test_or_deleted_pytest_raises_or_added_skip() -> None:
    """Chặn xóa hàm test, xóa khối pytest.raises hoặc gắn thêm @pytest.mark.skip / xfail."""
    # 1. Xóa hàm test_divide
    deleted_fn_code = """\
from calculator import add

def test_add():
    assert add(2, 3) == 5
"""
    res1 = check_quality_guard(BASE_TEST_CODE, deleted_fn_code)
    assert res1.passed is False
    assert "test_divide" in res1.deleted_tests

    # 2. Xóa pytest.raises
    deleted_raises_code = """\
import pytest
from calculator import add, divide

def test_add():
    assert add(2, 3) == 5

def test_divide():
    assert divide(10, 2) == 5.0
    assert divide(6, 2) == 3.0
"""
    res2 = check_quality_guard(BASE_TEST_CODE, deleted_raises_code)
    assert res2.passed is False
    assert any("pytest.raises" in r for r in res2.reasons)

    # 3. Thêm @pytest.mark.skip
    added_skip_code = """\
import pytest
from calculator import add, divide

@pytest.mark.skip(reason="skip failing test")
def test_add():
    assert add(2, 3) == 6

def test_divide():
    assert divide(10, 2) == 5.0
    with pytest.raises(ValueError):
        divide(10, 0)
"""
    res3 = check_quality_guard(BASE_TEST_CODE, added_skip_code)
    assert res3.passed is False
    assert len(res3.added_skips) == 1
