"""Unit tests cho aitest.generator & thuật toán Safe Append (Giai đoạn 4 - AC-03, AC-17)."""

from __future__ import annotations

from pathlib import Path

import pytest

from aitest.generator import (
    SafeAppendConflictError,
    TestGenerationSyntaxError,
    safe_merge_and_append,
)
from aitest.utils import resolve_test_file_path


def test_safe_append_dedup_rename_and_backup_ac03(tmp_path: Path) -> None:
    """Kiểm tra AC-03:
    - Giữ nguyên toàn bộ test cũ, tạo file backup.
    - Bỏ qua hàm test trùng tên + cùng nội dung AST.
    - Đổi tên thành test_x__2 khi trùng tên nhưng khác nội dung AST.
    - Hợp nhất import mới không trùng lặp và chèn separator `# --- aitest: appended ... ---`.
    """
    test_file = tmp_path / "tests" / "test_calc.py"
    initial_code = """\
import pytest
from calc import add

@pytest.fixture
def base_num():
    return 10

def test_add_basic():
    assert add(1, 2) == 3
"""
    test_file.parent.mkdir(parents=True, exist_ok=True)
    test_file.write_text(initial_code, encoding="utf-8")

    # Đoạn code mới có:
    # - test_add_basic giống hệt -> phải bị skip
    # - một hàm khác cũng tên là test_add_basic (hoặc trong lần append kế tiếp) khác nội dung -> đổi thành test_add_basic__2
    # - một hàm hoàn toàn mới test_add_negative -> thêm mới
    # - import math mới -> thêm mới; import pytest đã có -> không lặp lại
    new_chunk_1 = """\
import pytest
import math
from calc import add

def test_add_basic():
    assert add(1, 2) == 3

def test_add_negative():
    assert add(-1, -2) == -3
"""
    res1 = safe_merge_and_append(
        test_file=test_file,
        new_code=new_chunk_1,
        mode="append",
        backup_dir=tmp_path / ".aitest_backup",
        log_dir=tmp_path / ".aitest_logs",
        run_id="run-append-1",
    )
    assert res1.backup_path is not None and Path(res1.backup_path).is_file()
    assert res1.skipped_tests == ["test_add_basic"]
    assert res1.added_tests == ["test_add_negative"]
    assert res1.added_imports == ["import math"]

    # Lần append thứ 2: test_add_basic có nội dung KHÁC -> phải tự động đổi tên thành test_add_basic__2
    new_chunk_2 = """\
def test_add_basic():
    assert add(100, 200) == 300
"""
    res2 = safe_merge_and_append(
        test_file=test_file,
        new_code=new_chunk_2,
        mode="append",
        backup_dir=tmp_path / ".aitest_backup",
        log_dir=tmp_path / ".aitest_logs",
        run_id="run-append-2",
    )
    assert res2.renamed_tests == [("test_add_basic", "test_add_basic__2")]
    assert res2.added_tests == ["test_add_basic__2"]

    final_content = test_file.read_text(encoding="utf-8")
    assert "# --- aitest: appended " in final_content
    assert "def test_add_basic():" in final_content
    assert "def test_add_negative():" in final_content
    assert "def test_add_basic__2():" in final_content


def test_safe_append_blocks_fixture_or_helper_conflict_ac03(tmp_path: Path) -> None:
    """Kiểm tra AC-03: Chặn xung đột fixture/helper và giữ nguyên file cũ."""
    test_file = tmp_path / "tests" / "test_item.py"
    orig_code = """\
import pytest

@pytest.fixture
def sample_user():
    return {"name": "Alice"}

def test_user(sample_user):
    assert sample_user["name"] == "Alice"
"""
    test_file.parent.mkdir(parents=True, exist_ok=True)
    test_file.write_text(orig_code, encoding="utf-8")

    conflicting_new_code = """\
import pytest

@pytest.fixture
def sample_user():
    return {"name": "Bob", "role": "admin"}

def test_user_admin(sample_user):
    assert sample_user["role"] == "admin"
"""
    with pytest.raises(SafeAppendConflictError, match="sample_user"):
        safe_merge_and_append(
            test_file=test_file,
            new_code=conflicting_new_code,
            mode="append",
            backup_dir=tmp_path / ".aitest_backup",
            log_dir=tmp_path / ".aitest_logs",
        )
    # File cũ phải được giữ nguyên 100%
    assert test_file.read_text(encoding="utf-8") == orig_code


def test_safe_append_rejects_syntax_error_and_preserves_old_file_ac17(tmp_path: Path) -> None:
    """Kiểm tra AC-17: Khi LLM trả về code lỗi cú pháp -> không ghi đè, giữ nguyên bản cũ và lưu raw log."""
    test_file = tmp_path / "tests" / "test_valid.py"
    orig_code = "def test_existing():\n    assert 1 + 1 == 2\n"
    test_file.parent.mkdir(parents=True, exist_ok=True)
    test_file.write_text(orig_code, encoding="utf-8")

    log_dir = tmp_path / ".aitest_logs"
    with pytest.raises(TestGenerationSyntaxError):
        safe_merge_and_append(
            test_file=test_file,
            new_code="```python\ndef test_broken(\n    assert 1 == 1\n```",
            mode="append",
            backup_dir=tmp_path / ".aitest_backup",
            log_dir=log_dir,
            run_id="run-ac17",
        )

    assert test_file.read_text(encoding="utf-8") == orig_code
    assert any(log_dir.glob("*syntax_error_gen.log"))


def test_safe_append_supports_test_class_and_collision_guard(tmp_path: Path) -> None:
    """Kiểm tra P1.6 (append class Test*) và P0.4 (Collision Guard khi trùng stem khác module)."""
    test_file = tmp_path / "tests" / "test_calc.py"
    test_file.parent.mkdir(parents=True, exist_ok=True)
    test_file.write_text(
        "from examples.calc import add\n\ndef test_base():\n    assert add(1, 1) == 2\n",
        encoding="utf-8",
    )

    class_chunk = """\
class TestExtraCalc:
    def test_add_zero(self):
        assert add(0, 5) == 5
"""
    res = safe_merge_and_append(
        test_file=test_file,
        new_code=class_chunk,
        mode="append",
        backup_dir=tmp_path / ".aitest_backup",
        log_dir=tmp_path / ".aitest_logs",
    )
    assert "TestExtraCalc.test_add_zero" in res.added_tests
    assert "class TestExtraCalc:" in test_file.read_text(encoding="utf-8")

    # Collision Guard: tests/test_analyzer.py đã test `aitest.analyzer`,
    # nên target `examples/analyzer.py` phải tự tách sang `tests/test_examples_analyzer.py`.
    (tmp_path / "examples").mkdir(parents=True, exist_ok=True)
    ex_target = tmp_path / "examples" / "analyzer.py"
    ex_target.write_text("def parse():\n    return 1\n", encoding="utf-8")

    existing_sys_test = tmp_path / "tests" / "test_analyzer.py"
    existing_sys_test.write_text(
        "from aitest.analyzer import analyze_source_code\n\ndef test_sys():\n    assert True\n",
        encoding="utf-8",
    )
    resolved = resolve_test_file_path(ex_target, tests_dir="tests", project_root=tmp_path)
    assert resolved == Path("tests/test_examples_analyzer.py")
