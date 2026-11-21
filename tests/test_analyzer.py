"""Unit tests cho aitest.analyzer (Giai đoạn 2)."""

from __future__ import annotations

from aitest.analyzer import analyze_source_code, analyze_test_code


def test_analyze_source_code_extracts_functions_classes_and_exceptions() -> None:
    src = """\
import math
from typing import Optional

def divide(a: float, b: float = 1.0) -> float:
    \"\"\"Chia hai số thực.\"\"\"
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b

async def fetch_value(key: str) -> Optional[int]:
    if not key:
        raise KeyError("Empty key")
    return 42

class BankAccount:
    \"\"\"Tài khoản ngân hàng.\"\"\"
    def __init__(self, balance: float = 0.0) -> None:
        if balance < 0:
            raise ValueError("Initial balance must be non-negative")
        self.balance = balance

    def withdraw(self, amount: float) -> float:
        if amount > self.balance:
            raise RuntimeError("Insufficient funds")
        self.balance -= amount
        return self.balance
"""
    analysis = analyze_source_code(src, file_path="examples/bank.py")
    assert len(analysis.imports) == 2
    assert len(analysis.functions) == 2

    fn_div = analysis.functions[0]
    assert fn_div.name == "divide"
    assert fn_div.is_async is False
    assert fn_div.return_annotation == "float"
    assert "ValueError" in fn_div.raised_exceptions
    assert len(fn_div.parameters) == 2
    assert fn_div.parameters[1].default == "1.0"

    fn_async = analysis.functions[1]
    assert fn_async.name == "fetch_value"
    assert fn_async.is_async is True
    assert "KeyError" in fn_async.raised_exceptions

    assert len(analysis.classes) == 1
    cls = analysis.classes[0]
    assert cls.name == "BankAccount"
    assert len(cls.methods) == 2
    assert "RuntimeError" in cls.methods[1].raised_exceptions

    summary = "\n".join(analysis.target_summary_lines())
    assert "divide" in summary
    assert "BankAccount" in summary


def test_analyze_test_code_extracts_tests_fixtures_helpers_assertions_and_skips() -> None:
    test_src = """\
import pytest

@pytest.fixture
def sample_data():
    return [1, 2, 3]

def helper_sum(items):
    return sum(items)

def test_normal(sample_data):
    assert helper_sum(sample_data) == 6
    assert len(sample_data) == 3

def test_raises_error():
    with pytest.raises(ValueError):
        raise ValueError("bad")

@pytest.mark.skip(reason="not ready")
def test_skipped():
    assert 1 == 1
"""
    analysis = analyze_test_code(test_src)
    assert analysis.test_count == 3
    assert analysis.test_names == ["test_normal", "test_raises_error", "test_skipped"]
    assert len(analysis.fixtures) == 1
    assert analysis.fixtures[0].name == "sample_data"
    assert len(analysis.helpers) == 1
    assert analysis.helpers[0].name == "helper_sum"

    # Tổng số assertions: 2 trong test_normal + 1 pytest.raises trong test_raises_error + 1 trong test_skipped = 4
    assert analysis.total_assertion_count == 4
    assert analysis.test_functions[1].assertions[0].kind == "pytest_raises"
    assert analysis.test_functions[1].assertions[0].raised_exception == "ValueError"
    assert analysis.test_functions[2].has_skip_or_xfail is True
