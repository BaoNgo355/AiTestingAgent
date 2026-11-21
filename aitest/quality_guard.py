"""Quality Guard (AST-aware): Kiểm tra hậu sửa test và phân tích AST Assertion Diff (SRS v2.3.0 §7 FR-04, AC-07)."""

from __future__ import annotations

import ast

from aitest.analyzer import AssertionInfo, TestFunctionInfo, analyze_test_code
from aitest.config import QualityGuardConfig
from aitest.schemas import AssertionDiffEntry, GuardResult


def _unparse(node: ast.AST) -> str:
    try:
        return ast.unparse(node).strip()
    except Exception:
        return ""


def _is_len_call(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "len"
        and len(node.args) == 1
    )


def _is_tautology_expr(test_expr: ast.AST) -> tuple[bool, str]:
    """Phát hiện biểu thức điều kiện trong `assert <test_expr>` có phải là tautology/tầm thường không."""
    # 1. Hằng số trực tiếp: assert True, assert 1, assert "abc"
    if isinstance(test_expr, ast.Constant):
        if bool(test_expr.value) is True:
            return True, f"Tautology hằng số luôn đúng (`assert {_unparse(test_expr)}`)"
        return True, f"Hằng số tĩnh trong assert (`assert {_unparse(test_expr)}`)"

    # 2. Phủ định của hằng số sai: assert not False, assert not 0, assert not None
    if isinstance(test_expr, ast.UnaryOp) and isinstance(test_expr.op, ast.Not):
        operand = test_expr.operand
        if isinstance(operand, ast.Constant) and bool(operand.value) is False:
            return True, f"Tautology phủ định hằng số sai (`assert {_unparse(test_expr)}`)"

    # 3. Phép so sánh tầm thường
    if isinstance(test_expr, ast.Compare) and len(test_expr.ops) == 1 and len(test_expr.comparators) == 1:
        left = test_expr.left
        op = test_expr.ops[0]
        right = test_expr.comparators[0]

        # 3a. So sánh hai biểu thức giống hệt nhau: assert x == x, assert 1 == 1
        if isinstance(op, (ast.Eq, ast.Is, ast.GtE, ast.LtE)):
            if ast.dump(left, include_attributes=False) == ast.dump(right, include_attributes=False):
                return True, f"Tautology so sánh một biểu thức với chính nó (`assert {_unparse(test_expr)}`)"

        # 3b. assert len(x) >= 0 hoặc assert 0 <= len(x)
        if (
            isinstance(op, ast.GtE)
            and _is_len_call(left)
            and isinstance(right, ast.Constant)
            and isinstance(right.value, (int, float))
            and right.value <= 0
        ):
            return True, f"Tautology độ dài luôn không âm (`assert {_unparse(test_expr)}`)"

        if (
            isinstance(op, ast.LtE)
            and _is_len_call(right)
            and isinstance(left, ast.Constant)
            and isinstance(left.value, (int, float))
            and left.value <= 0
        ):
            return True, f"Tautology độ dài luôn không âm (`assert {_unparse(test_expr)}`)"

    # 4. Biểu thức boolean `or True`: assert x or True
    if isinstance(test_expr, ast.BoolOp) and isinstance(test_expr.op, ast.Or):
        for val in test_expr.values:
            if isinstance(val, ast.Constant) and bool(val.value) is True:
                return True, f"Tautology chứa nhánh `or True` (`assert {_unparse(test_expr)}`)"

    return False, ""


def _is_existence_or_type_only_check(test_expr: ast.AST) -> tuple[bool, str]:
    """Kiểm tra xem biểu thức có phải chỉ kiểm tra tồn tại (`is not None`, `isinstance`, `bool(x)`) không."""
    if isinstance(test_expr, ast.Compare) and len(test_expr.ops) == 1 and len(test_expr.comparators) == 1:
        op = test_expr.ops[0]
        right = test_expr.comparators[0]
        if isinstance(op, (ast.IsNot, ast.NotEq)) and isinstance(right, ast.Constant) and right.value is None:
            return True, "Chỉ kiểm tra khác None (`is not None` / `!= None`)"

    if isinstance(test_expr, ast.Call) and isinstance(test_expr.func, ast.Name):
        if test_expr.func.id in ("isinstance", "bool", "hasattr", "callable"):
            return True, f"Chỉ kiểm tra kiểu/tồn tại (`{test_expr.func.id}(...)`)"

    return False, ""


def _parse_assert_expr(assertion_source: str) -> ast.AST | None:
    """Trích xuất node `test` bên trong câu lệnh `assert <test>`."""
    try:
        mod = ast.parse(assertion_source)
        if mod.body and isinstance(mod.body[0], ast.Assert):
            return mod.body[0].test
    except SyntaxError:
        pass
    return None


def _compare_single_assertion_pair(
    test_name: str,
    old_a: AssertionInfo,
    new_a: AssertionInfo,
) -> AssertionDiffEntry:
    """Phân tích AST diff giữa một cặp câu lệnh kiểm tra trước và sau khi sửa trong cùng hàm test."""
    if old_a.kind == new_a.kind and old_a.ast_dump == new_a.ast_dump:
        return AssertionDiffEntry(
            test_name=test_name,
            old_assertion=old_a.source,
            new_assertion=new_a.source,
            classification="unchanged",
            reason="Giữ nguyên cấu trúc AST",
        )

    # Nếu cũ là pytest.raises mà mới không còn là pytest.raises
    if old_a.kind == "pytest_raises" and new_a.kind != "pytest_raises":
        return AssertionDiffEntry(
            test_name=test_name,
            old_assertion=old_a.source,
            new_assertion=new_a.source,
            classification="weakened",
            reason="Xóa hoặc thay thế khối kiểm tra ngoại lệ `pytest.raises` bằng assert thường",
        )

    if old_a.kind == "pytest_raises" and new_a.kind == "pytest_raises":
        # Nếu hạ từ một exception cụ thể (ví dụ ValueError) xuống BaseException/Exception chung chung
        old_exc = (old_a.raised_exception or "").strip()
        new_exc = (new_a.raised_exception or "").strip()
        if old_exc not in ("Exception", "BaseException", "") and new_exc in ("Exception", "BaseException"):
            return AssertionDiffEntry(
                test_name=test_name,
                old_assertion=old_a.source,
                new_assertion=new_a.source,
                classification="weakened",
                reason=f"Làm yếu kiểm tra ngoại lệ từ `{old_exc}` xuống `{new_exc}` chung chung",
            )
        return AssertionDiffEntry(
            test_name=test_name,
            old_assertion=old_a.source,
            new_assertion=new_a.source,
            classification="allowed_value_update",
            reason=f"Cập nhật loại ngoại lệ trong pytest.raises ({old_exc} -> {new_exc})",
        )

    old_expr = _parse_assert_expr(old_a.source)
    new_expr = _parse_assert_expr(new_a.source)
    if new_expr is None:
        return AssertionDiffEntry(
            test_name=test_name,
            old_assertion=old_a.source,
            new_assertion=new_a.source,
            classification="weakened",
            reason="Không phân tích được biểu thức assert mới",
        )

    # Kiểm tra tautology trên biểu thức mới
    is_taut, taut_reason = _is_tautology_expr(new_expr)
    if is_taut:
        return AssertionDiffEntry(
            test_name=test_name,
            old_assertion=old_a.source,
            new_assertion=new_a.source,
            classification="weakened",
            reason=taut_reason,
        )

    # Nếu biểu thức cũ là một phép so sánh (Compare)
    if isinstance(old_expr, ast.Compare) and len(old_expr.ops) == 1:
        old_op = old_expr.ops[0]

        # Kiểm tra xem biểu thức mới có bị hạ cấp xuống kiểm tra tồn tại / kiểu không
        is_exist, exist_reason = _is_existence_or_type_only_check(new_expr)
        if is_exist:
            return AssertionDiffEntry(
                test_name=test_name,
                old_assertion=old_a.source,
                new_assertion=new_a.source,
                classification="weakened",
                reason=f"Hạ cấp từ phép so sánh cụ thể xuống: {exist_reason}",
            )

        # Nếu biểu thức cũ là so sánh bằng (==)
        if isinstance(old_op, ast.Eq):
            if not isinstance(new_expr, ast.Compare) or len(new_expr.ops) != 1:
                return AssertionDiffEntry(
                    test_name=test_name,
                    old_assertion=old_a.source,
                    new_assertion=new_a.source,
                    classification="weakened",
                    reason="Hạ cấp từ phép so sánh bằng (`==`) xuống biểu thức không còn phép so sánh",
                )
            new_op = new_expr.ops[0]
            if not isinstance(new_op, ast.Eq):
                return AssertionDiffEntry(
                    test_name=test_name,
                    old_assertion=old_a.source,
                    new_assertion=new_a.source,
                    classification="weakened",
                    reason=(
                        f"Nới lỏng toán tử so sánh từ `==` sang `{new_op.__class__.__name__}` "
                        f"(`{old_a.source}` -> `{new_a.source}`)"
                    ),
                )
            # Cùng là phép so sánh `==` và không phải tautology -> Cho phép sửa giá trị kỳ vọng/tham số
            return AssertionDiffEntry(
                test_name=test_name,
                old_assertion=old_a.source,
                new_assertion=new_a.source,
                classification="allowed_value_update",
                reason="Điều chỉnh giá trị kỳ vọng hoặc tham số đầu vào trong phép so sánh bằng (`==`)",
            )

        # Nếu biểu thức cũ là phép so sánh khác (<, <=, >, >=, !=, in) mà biểu thức mới không còn là Compare
        if not isinstance(new_expr, ast.Compare):
            return AssertionDiffEntry(
                test_name=test_name,
                old_assertion=old_a.source,
                new_assertion=new_a.source,
                classification="weakened",
                reason="Hạ cấp từ biểu thức so sánh xuống biểu thức truthy đơn giản",
            )

    return AssertionDiffEntry(
        test_name=test_name,
        old_assertion=old_a.source,
        new_assertion=new_a.source,
        classification="allowed_value_update",
        reason="Cập nhật biểu thức kiểm tra hợp lệ",
    )


def _analyze_function_assertion_diffs(
    old_fn: TestFunctionInfo,
    new_fn: TestFunctionInfo,
) -> tuple[list[AssertionDiffEntry], list[str]]:
    """So sánh danh sách assertions của một hàm test trước và sau khi sửa."""
    diffs: list[AssertionDiffEntry] = []
    reasons: list[str] = []

    old_raises = [a for a in old_fn.assertions if a.kind == "pytest_raises"]
    new_raises = [a for a in new_fn.assertions if a.kind == "pytest_raises"]
    if len(new_raises) < len(old_raises):
        msg = (
            f"Hàm `{old_fn.name}` bị xóa khối `pytest.raises` "
            f"(trước: {len(old_raises)}, sau: {len(new_raises)})"
        )
        reasons.append(msg)
        diffs.append(
            AssertionDiffEntry(
                test_name=old_fn.name,
                old_assertion=old_raises[-1].source,
                new_assertion="<deleted>",
                classification="weakened",
                reason=msg,
            )
        )

    if len(new_fn.assertions) < len(old_fn.assertions):
        reasons.append(
            f"Hàm `{old_fn.name}` bị giảm số câu lệnh kiểm tra "
            f"(trước: {len(old_fn.assertions)}, sau: {len(new_fn.assertions)})"
        )

    pair_count = min(len(old_fn.assertions), len(new_fn.assertions))
    for idx in range(pair_count):
        entry = _compare_single_assertion_pair(
            old_fn.name,
            old_fn.assertions[idx],
            new_fn.assertions[idx],
        )
        if entry.classification != "unchanged":
            diffs.append(entry)
        if entry.classification == "weakened":
            reasons.append(f"Assertion Weakening tại `{old_fn.name}`: {entry.reason}")

    # Kiểm tra các assertion được thêm mới trong hàm cũng không được là tautology
    for extra_a in new_fn.assertions[pair_count:]:
        if extra_a.kind == "assert":
            expr = _parse_assert_expr(extra_a.source)
            if expr is not None:
                is_taut, taut_reason = _is_tautology_expr(expr)
                if is_taut:
                    diffs.append(
                        AssertionDiffEntry(
                            test_name=new_fn.name,
                            old_assertion="<none>",
                            new_assertion=extra_a.source,
                            classification="weakened",
                            reason=taut_reason,
                        )
                    )
                    reasons.append(f"Tautology mới thêm tại `{new_fn.name}`: {taut_reason}")

    return diffs, reasons


def check_quality_guard(
    old_test_code: str,
    new_test_code: str,
    *,
    guard_config: QualityGuardConfig | None = None,
    allow_skip: bool = False,
) -> GuardResult:
    """Thực thi đầy đủ các quy tắc Quality Guard trên file test trước và sau khi sửa (FR-04, AC-07).

    Kiểm tra:
    1. Cú pháp AST hợp lệ.
    2. Số lượng hàm `test_*` không giảm và không có hàm `test_*` cũ nào bị xóa.
    3. Tổng số câu lệnh `assert` + `pytest.raises` không giảm.
    4. Không thêm mới `@pytest.mark.skip`, `skipif`, `xfail`, hoặc gọi `pytest.skip()` / `pytest.xfail()`.
    5. Phân tích AST Assertion Diff: chặn tautology, chặn hạ cấp xuống `is not None`,
       chặn nới lỏng toán tử `==` sang `!=`/`>=`/`<=`, chặn xóa `pytest.raises`;
       cho phép sửa giá trị hằng số kỳ vọng trong phép `==` (kèm ghi log `assertion_diffs`).
    """
    cfg = guard_config or QualityGuardConfig()

    try:
        new_analysis = analyze_test_code(new_test_code, file_path="<new_test>")
    except SyntaxError as exc:
        return GuardResult(
            passed=False,
            syntax_ok=False,
            reasons=[f"Mã test mới có lỗi cú pháp Python: {exc}"],
        )

    try:
        old_analysis = (
            analyze_test_code(old_test_code, file_path="<old_test>")
            if old_test_code.strip()
            else analyze_test_code("", file_path="<old_test>")
        )
    except SyntaxError:
        # Nếu file test cũ đang bị lỗi cú pháp (TEST_SYNTAX), chỉ kiểm tra file mới không có tautology và skip
        old_analysis = analyze_test_code("", file_path="<old_test>")

    reasons: list[str] = []
    deleted_tests: list[str] = []
    added_skips: list[str] = []
    assertion_diffs: list[AssertionDiffEntry] = []

    old_test_map = {fn.name: fn for fn in old_analysis.test_functions}
    new_test_map = {fn.name: fn for fn in new_analysis.test_functions}

    old_test_count = old_analysis.test_count
    new_test_count = new_analysis.test_count
    old_assert_count = old_analysis.total_assertion_count
    new_assert_count = new_analysis.total_assertion_count

    # 1. Không xóa hàm test cũ & không giảm số lượng hàm test
    for old_name in old_test_map:
        if old_name not in new_test_map:
            deleted_tests.append(old_name)
    if deleted_tests:
        reasons.append(f"Phát hiện hàm test cũ bị xóa: {', '.join(deleted_tests)}")

    if old_test_count > 0 and new_test_count < old_test_count * cfg.min_test_count_ratio:
        reasons.append(
            f"Số lượng hàm test bị giảm từ {old_test_count} xuống {new_test_count}"
        )
    if new_test_count == 0:
        reasons.append("File test mới không chứa bất kỳ hàm `test_*` nào")

    # 2. Tổng số assert + pytest.raises không giảm
    if old_assert_count > 0 and new_assert_count < old_assert_count * cfg.min_assert_count_ratio:
        reasons.append(
            f"Tổng số câu lệnh kiểm tra (assert + pytest.raises) bị giảm từ {old_assert_count} xuống {new_assert_count}"
        )

    # 3. Không thêm mới skip / xfail
    if not allow_skip:
        for mod_skip in new_analysis.module_level_skips:
            if mod_skip not in old_analysis.module_level_skips:
                added_skips.append(f"<module>: {mod_skip}")

        for fn_name, new_fn in new_test_map.items():
            old_fn = old_test_map.get(fn_name)
            old_markers = set(old_fn.skip_markers) if old_fn else set()
            for marker in new_fn.skip_markers:
                if marker not in old_markers:
                    added_skips.append(f"{fn_name}: {marker}")

        if added_skips:
            reasons.append(
                f"Phát hiện thêm mới đánh dấu skip/xfail không được phép: {', '.join(added_skips)}"
            )

    # 4. Phân tích AST Assertion Diff
    if cfg.assertion_ast_diff:
        for fn_name, new_fn in new_test_map.items():
            old_fn = old_test_map.get(fn_name)
            if old_fn is not None:
                fn_diffs, fn_reasons = _analyze_function_assertion_diffs(old_fn, new_fn)
                assertion_diffs.extend(fn_diffs)
                if cfg.reject_on_assertion_weakening:
                    reasons.extend(fn_reasons)
            else:
                # Hàm test mới hoàn toàn (hoặc vừa sửa từ file cũ lỗi syntax): kiểm tra không chứa tautology
                for a in new_fn.assertions:
                    if a.kind == "assert":
                        expr = _parse_assert_expr(a.source)
                        if expr is not None:
                            is_taut, taut_reason = _is_tautology_expr(expr)
                            if is_taut:
                                assertion_diffs.append(
                                    AssertionDiffEntry(
                                        test_name=fn_name,
                                        old_assertion="<new_test>",
                                        new_assertion=a.source,
                                        classification="weakened",
                                        reason=taut_reason,
                                    )
                                )
                                if cfg.reject_on_assertion_weakening:
                                    reasons.append(f"Tautology tại `{fn_name}`: {taut_reason}")

    return GuardResult(
        passed=len(reasons) == 0,
        syntax_ok=True,
        old_test_count=old_test_count,
        new_test_count=new_test_count,
        old_assert_count=old_assert_count,
        new_assert_count=new_assert_count,
        deleted_tests=deleted_tests,
        added_skips=added_skips,
        assertion_diffs=assertion_diffs,
        reasons=reasons,
    )
