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


def _extract_abs_sub_operands(node: ast.AST) -> tuple[ast.AST, ast.AST] | None:
    """Trả về `(a, b)` nếu node có dạng `abs(a - b)` với `a` khác `b` về cấu trúc AST."""
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "abs"
        and len(node.args) == 1
        and isinstance(node.args[0], ast.BinOp)
        and isinstance(node.args[0].op, ast.Sub)
    ):
        left, right = node.args[0].left, node.args[0].right
        if ast.dump(left, include_attributes=False) != ast.dump(right, include_attributes=False):
            return left, right
    return None


def _extract_small_positive_tol(node: ast.AST) -> float | None:
    """Trả về giá trị dung sai dương `0 < tol <= 1e-2` nếu node là hằng số hợp lệ."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        val = float(node.value)
        if 0.0 < val <= 1e-2:
            return val
    return None


def _is_approx_float_check(expr: ast.AST) -> bool:
    """Kiểm tra biểu thức có phải phép so sánh xấp xỉ số thực chặt chẽ (`math.isclose`, `abs(a - b) < 1e-6`) không."""
    if isinstance(expr, ast.Call):
        fn_name = _unparse(expr.func)
        if fn_name in ("math.isclose", "isclose", "np.isclose", "numpy.isclose") and len(expr.args) >= 2:
            if ast.dump(expr.args[0], include_attributes=False) != ast.dump(expr.args[1], include_attributes=False):
                return True
        return False

    if isinstance(expr, ast.Compare) and len(expr.ops) == 1 and len(expr.comparators) == 1:
        op = expr.ops[0]
        left = expr.left
        right = expr.comparators[0]
        if isinstance(op, (ast.Lt, ast.LtE)):
            return _extract_abs_sub_operands(left) is not None and _extract_small_positive_tol(right) is not None
        if isinstance(op, (ast.Gt, ast.GtE)):
            return _extract_small_positive_tol(left) is not None and _extract_abs_sub_operands(right) is not None

    return False


def _is_equivalent_bool_check(new_expr: ast.AST) -> bool:
    """Kiểm tra xem biểu thức mới có phải phép kiểm tra boolean chặt (`is True`, `is False`, `assert expr`, `assert not expr`) không."""
    # `expr is True` hoặc `expr is False`
    if isinstance(new_expr, ast.Compare) and len(new_expr.ops) == 1 and len(new_expr.comparators) == 1:
        op = new_expr.ops[0]
        right = new_expr.comparators[0]
        if isinstance(op, ast.Is) and isinstance(right, ast.Constant) and isinstance(right.value, bool):
            return True
        return False

    # `assert not expr` (không phải tautology, không phải kiểm tra tồn tại/kiểu)
    if isinstance(new_expr, ast.UnaryOp) and isinstance(new_expr.op, ast.Not):
        is_taut, _ = _is_tautology_expr(new_expr)
        is_taut_inner, _ = _is_tautology_expr(new_expr.operand)
        is_exist, _ = _is_existence_or_type_only_check(new_expr.operand)
        return not is_taut and not is_taut_inner and not is_exist

    # `assert expr` (không phải tautology, không phải kiểm tra tồn tại/kiểu)
    is_taut, _ = _is_tautology_expr(new_expr)
    is_exist, _ = _is_existence_or_type_only_check(new_expr)
    return not is_taut and not is_exist


def _effective_assert_count(assertions: list[AssertionInfo]) -> int:
    """Đếm số kiểm tra hiệu dụng (`pytest.raises(..., match=...)` tính thêm 1 kiểm tra message)."""
    return len(assertions) + sum(
        1 for a in assertions if a.kind == "pytest_raises" and a.has_match
    )


def _extract_subject_key(a: AssertionInfo) -> str:
    """Trích xuất khóa biểu thức chủ thể bên trái để ghép cặp assertion thông minh."""
    expr = _parse_assert_expr(a.source)
    if expr is None:
        return ""
    if isinstance(expr, ast.Compare) and len(expr.ops) == 1 and len(expr.comparators) == 1:
        abs_ops = _extract_abs_sub_operands(expr.left)
        if abs_ops is not None:
            return ast.dump(abs_ops[0], include_attributes=False)
        abs_ops_r = _extract_abs_sub_operands(expr.comparators[0])
        if abs_ops_r is not None:
            return ast.dump(abs_ops_r[0], include_attributes=False)
        return ast.dump(expr.left, include_attributes=False)
    if isinstance(expr, ast.Call):
        fn_name = _unparse(expr.func)
        if fn_name in ("math.isclose", "isclose", "np.isclose", "numpy.isclose") and len(expr.args) >= 2:
            return ast.dump(expr.args[0], include_attributes=False)
    if isinstance(expr, ast.UnaryOp) and isinstance(expr.op, ast.Not):
        return ast.dump(expr.operand, include_attributes=False)
    return ast.dump(expr, include_attributes=False)


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
            # Cho phép chuyển sang phép so sánh xấp xỉ số thực chặt (`math.isclose`, `abs(a - b) < 1e-6`)
            if _is_approx_float_check(new_expr):
                return AssertionDiffEntry(
                    test_name=test_name,
                    old_assertion=old_a.source,
                    new_assertion=new_a.source,
                    classification="allowed_value_update",
                    reason="Chuyển sang phép so sánh xấp xỉ số thực có dung sai chặt",
                )

            # Cho phép chuẩn hóa/sửa giá trị kiểm tra boolean (`== True/False` -> `is True/False`, `assert expr`, `assert not expr`)
            if (
                len(old_expr.comparators) == 1
                and isinstance(old_expr.comparators[0], ast.Constant)
                and isinstance(old_expr.comparators[0].value, bool)
                and _is_equivalent_bool_check(new_expr)
            ):
                return AssertionDiffEntry(
                    test_name=test_name,
                    old_assertion=old_a.source,
                    new_assertion=new_a.source,
                    classification="allowed_value_update",
                    reason="Chuẩn hóa phép kiểm tra hằng số boolean tương đương",
                )

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


def _pair_assertions_smart(
    old_items: list[AssertionInfo],
    new_items: list[AssertionInfo],
    *,
    by_exception: bool = False,
) -> tuple[list[tuple[AssertionInfo, AssertionInfo]], list[AssertionInfo], list[AssertionInfo]]:
    """Ghép cặp thông minh hai danh sách assertion cùng loại (`pytest_raises` hoặc `assert`).

    Ưu tiên: (1) Khớp chính xác `ast_dump` -> (2) Khớp cùng `raised_exception` hoặc `_extract_subject_key`
    -> (3) Ghép tuần tự các phần tử còn lại. Trả về `(pairs, unmatched_old, unmatched_new)`.
    """
    used_old: set[int] = set()
    used_new: set[int] = set()
    pairs: list[tuple[AssertionInfo, AssertionInfo]] = []

    # Bước 1: Khớp chính xác ast_dump
    for oi, old_a in enumerate(old_items):
        for ni, new_a in enumerate(new_items):
            if ni not in used_new and old_a.ast_dump == new_a.ast_dump:
                used_old.add(oi)
                used_new.add(ni)
                pairs.append((old_a, new_a))
                break

    # Bước 2: Khớp theo khóa ngoại lệ hoặc biểu thức chủ thể bên trái
    if by_exception:
        for oi, old_a in enumerate(old_items):
            if oi in used_old or not old_a.raised_exception:
                continue
            for ni, new_a in enumerate(new_items):
                if ni not in used_new and new_a.raised_exception == old_a.raised_exception:
                    used_old.add(oi)
                    used_new.add(ni)
                    pairs.append((old_a, new_a))
                    break
    else:
        old_keys = {oi: _extract_subject_key(old_a) for oi, old_a in enumerate(old_items) if oi not in used_old}
        new_keys = {ni: _extract_subject_key(new_a) for ni, new_a in enumerate(new_items) if ni not in used_new}
        for oi, okey in old_keys.items():
            if not okey:
                continue
            for ni, nkey in new_keys.items():
                if ni not in used_new and nkey == okey:
                    used_old.add(oi)
                    used_new.add(ni)
                    pairs.append((old_items[oi], new_items[ni]))
                    break

    # Bước 3: Ghép tuần tự phần còn lại
    rem_old = [old_items[i] for i in range(len(old_items)) if i not in used_old]
    rem_new = [new_items[i] for i in range(len(new_items)) if i not in used_new]
    seq_count = min(len(rem_old), len(rem_new))
    for i in range(seq_count):
        pairs.append((rem_old[i], rem_new[i]))

    return pairs, rem_old[seq_count:], rem_new[seq_count:]


def _extract_absorbed_exc_vars(
    old_fn: TestFunctionInfo,
    raise_pairs: list[tuple[AssertionInfo, AssertionInfo]],
) -> set[str]:
    """Tìm tên biến ngoại lệ (`as exc_info`) từ các khối `pytest.raises` đã nâng cấp lên `match=...`."""
    upgraded_lines = {
        o_r.lineno for o_r, n_r in raise_pairs if not o_r.has_match and n_r.has_match
    }
    if not upgraded_lines or not old_fn.source_code:
        return set()

    exc_vars: set[str] = set()
    try:
        tree = ast.parse(old_fn.source_code)
    except SyntaxError:
        return exc_vars

    for sub in ast.walk(tree):
        if isinstance(sub, (ast.With, ast.AsyncWith)):
            for item in sub.items:
                if item.optional_vars is not None and isinstance(item.optional_vars, ast.Name):
                    # Kiểm tra xem lệnh `with` này nằm trong nhóm `pytest.raises` được nâng cấp lên `match=`
                    ctx_src = _unparse(item.context_expr)
                    if "raises" in ctx_src:
                        exc_vars.add(item.optional_vars.id)
    return exc_vars


def _uses_any_name(source: str, names: set[str]) -> bool:
    if not names:
        return False
    expr = _parse_assert_expr(source)
    if expr is None:
        return False
    return any(isinstance(sub, ast.Name) and sub.id in names for sub in ast.walk(expr))


def _analyze_function_assertion_diffs(
    old_fn: TestFunctionInfo,
    new_fn: TestFunctionInfo,
) -> tuple[list[AssertionDiffEntry], list[str]]:
    """So sánh danh sách assertions của một hàm test trước và sau khi sửa bằng Smart AST Alignment."""
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

    old_raw_count = len(old_fn.assertions)
    new_raw_count = len(new_fn.assertions)
    new_eff_count = _effective_assert_count(new_fn.assertions)
    if new_raw_count < old_raw_count and new_eff_count < old_raw_count:
        reasons.append(
            f"Hàm `{old_fn.name}` bị giảm số câu lệnh kiểm tra "
            f"(trước: {old_raw_count}, sau: {new_raw_count})"
        )

    raise_pairs, _, _ = _pair_assertions_smart(old_raises, new_raises, by_exception=True)
    absorbed_vars = _extract_absorbed_exc_vars(old_fn, raise_pairs)

    old_asserts = [
        a for a in old_fn.assertions
        if a.kind == "assert" and not _uses_any_name(a.source, absorbed_vars)
    ]
    new_asserts = [a for a in new_fn.assertions if a.kind == "assert"]

    assert_pairs, _, extra_new_asserts = _pair_assertions_smart(old_asserts, new_asserts, by_exception=False)

    for old_a, new_a in [*raise_pairs, *assert_pairs]:
        entry = _compare_single_assertion_pair(old_fn.name, old_a, new_a)
        if entry.classification != "unchanged":
            diffs.append(entry)
        if entry.classification == "weakened":
            reasons.append(f"Assertion Weakening tại `{old_fn.name}`: {entry.reason}")

    # Kiểm tra các assertion được thêm mới trong hàm cũng không được là tautology
    for extra_a in extra_new_asserts:
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

    # 2. Tổng số assert + pytest.raises không giảm (công nhận pytest.raises(..., match=...) tương đương thêm 1 kiểm tra message)
    new_effective_assert_count = sum(
        _effective_assert_count(fn.assertions) for fn in new_analysis.test_functions
    )
    if (
        old_assert_count > 0
        and new_assert_count < old_assert_count * cfg.min_assert_count_ratio
        and new_effective_assert_count < old_assert_count * cfg.min_assert_count_ratio
    ):
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
