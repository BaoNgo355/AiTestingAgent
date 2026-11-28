"""Unit tests cho aitest.mutation (Giai đoạn 3 - AC-12, AC-22, NFR-02)."""

from __future__ import annotations

from pathlib import Path

from aitest.config import AitestConfig
from aitest.mutation import collect_ast_mutants, run_mutation_testing
from aitest.utils import compute_file_hash


def test_ast_mutation_engine_detects_survived_and_killed_mutants_and_preserves_source_ac12_nfr02(
    tmp_path: Path,
) -> None:
    """Kiểm tra AC-12 & NFR-02:
    - Chạy AST Mutation Engine trên Windows native.
    - Phát hiện survived_mutants khi test thiếu kiểm tra biên, và tăng điểm khi thêm test biên.
    - Bảo toàn 100% hash SHA-256 của file mã nguồn gốc trước/sau khi chạy.
    """
    src_file = tmp_path / "age_checker.py"
    src_file.write_text(
        "def is_adult(age: int) -> bool:\n"
        "    if age >= 18:\n"
        "        return True\n"
        "    return False\n",
        encoding="utf-8",
    )
    hash_before = compute_file_hash(src_file)

    # Bộ test yếu: chỉ kiểm tra age=25, chưa kiểm tra tại biên 18 và < 18
    test_file = tmp_path / "test_age_checker.py"
    test_file.write_text(
        "from age_checker import is_adult\n\n"
        "def test_adult_25():\n"
        "    assert is_adult(25) is True\n",
        encoding="utf-8",
    )

    cfg = AitestConfig()
    weak_res = run_mutation_testing(
        target_file=src_file,
        test_file=test_file,
        config=cfg,
        project_root=tmp_path,
    )
    assert compute_file_hash(src_file) == hash_before
    assert weak_res.mutation_available is True
    assert weak_res.mutation_status == "available"
    assert weak_res.mutation_score_pct is not None
    assert len(weak_res.survived_mutants) >= 1
    assert any(m.line == 2 and ">= -> >" in m.mutation for m in weak_res.survived_mutants)

    # Bộ test mạnh: bổ sung kiểm tra đúng mốc biên 18 và 17
    test_file.write_text(
        "from age_checker import is_adult\n\n"
        "def test_adult_boundary_18():\n"
        "    assert is_adult(18) is True\n\n"
        "def test_minor_17():\n"
        "    assert is_adult(17) is False\n",
        encoding="utf-8",
    )
    strong_res = run_mutation_testing(
        target_file=src_file,
        test_file=test_file,
        config=cfg,
        project_root=tmp_path,
    )
    assert compute_file_hash(src_file) == hash_before
    assert strong_res.mutation_available is True
    assert strong_res.mutation_score_pct == 100.0
    assert len(strong_res.survived_mutants) == 0
    assert (strong_res.mutation_score_pct or 0.0) > (weak_res.mutation_score_pct or 0.0)


def test_mutation_no_valid_mutants_returns_null_score_ac22(tmp_path: Path) -> None:
    """Kiểm tra AC-22: File nguồn không có biểu thức tạo mutant hợp lệ -> mutation_score_pct = None, status = no_valid_mutants."""
    src_file = tmp_path / "empty_fn.py"
    src_file.write_text(
        "def do_nothing() -> None:\n"
        "    \"\"\"Hàm rỗng chỉ có docstring.\"\"\"\n"
        "    pass\n",
        encoding="utf-8",
    )
    test_file = tmp_path / "test_empty_fn.py"
    test_file.write_text(
        "from empty_fn import do_nothing\n\n"
        "def test_noop():\n"
        "    assert do_nothing() is None\n",
        encoding="utf-8",
    )

    cfg = AitestConfig()
    res = run_mutation_testing(
        target_file=src_file,
        test_file=test_file,
        config=cfg,
        project_root=tmp_path,
    )
    assert res.mutation_available is False
    assert res.mutation_status == "no_valid_mutants"
    assert res.mutation_score_pct is None


def test_mutation_augassign_chained_compare_coverage_skip_and_worker_timeout_recovery(
    tmp_path: Path,
) -> None:
    """Kiểm tra P1.5:
    - Sinh mutant cho `AugAssign` (`+=`) và chained `Compare` (`'a' <= ch <= 'z'`).
    - Lọc nhanh bằng `executed_lines` đánh dấu đúng mutant trên dòng chưa phủ là `survived`.
    - Persistent worker xử lý đúng mutant gây vòng lặp vô hạn (`timeout`) và vẫn tiếp tục test các mutant sau.
    """
    sample_src = (
        "def process(ch: str, n: int) -> int:\n"
        "    total = 0\n"
        "    if 'a' <= ch <= 'z':\n"
        "        total += n\n"
        "    return total\n"
    )
    specs = collect_ast_mutants(sample_src)
    mutations = [s.mutation for s in specs]
    assert "+= -> -=" in mutations
    assert "<= -> <" in mutations

    # Kiểm tra timeout recovery trong Persistent Worker:
    # Khi `i += 1` bị mutate thành `i -= 1`, vòng `while i < limit:` thành lặp vô hạn -> timeout,
    # nhưng mutant ở câu `return total` phía sau vẫn chạy đúng.
    loop_src = tmp_path / "loop_mod.py"
    loop_src.write_text(
        "def count_up(limit: int) -> int:\n"
        "    i = 0\n"
        "    total = 0\n"
        "    while i < limit:\n"
        "        total += i\n"
        "        i += 1\n"
        "    if limit < 0:\n"
        "        return -1\n"
        "    return total\n",
        encoding="utf-8",
    )
    loop_test = tmp_path / "test_loop_mod.py"
    loop_test.write_text(
        "from loop_mod import count_up\n\n"
        "def test_count_up():\n"
        "    assert count_up(3) == 3\n",
        encoding="utf-8",
    )
    cfg = AitestConfig()
    cfg.mutation.per_mutant_timeout_sec = 0.8
    # Dòng 8 (`return -1`) chưa được phủ -> truyền executed_lines bỏ qua dòng 8
    res = run_mutation_testing(
        target_file=loop_src,
        test_file=loop_test,
        config=cfg,
        project_root=tmp_path,
        executed_lines={1, 2, 3, 4, 5, 6, 7, 9},
    )
    assert res.mutation_available is True
    assert res.mutation_timeout >= 1
    assert res.mutation_killed is not None and res.mutation_killed >= 1
    assert any(m.line == 8 for m in res.survived_mutants)
