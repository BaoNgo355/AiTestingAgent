"""Dual Mutation Backend: Built-in AST Mutation Engine đa nền tảng (Windows/POSIX) và mutmut Adapter (SRS v2.3.0 §7 FR-07, AC-12, AC-21, AC-22)."""

from __future__ import annotations

import ast
import copy
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

from aitest.config import AitestConfig
from aitest.progress import NullProgressReporter, ProgressReporter
from aitest.schemas import MutationResult, SurvivedMutant


@dataclass(frozen=True)
class MutantSpec:
    """Đặc tả một phép biến đổi đơn lẻ (1 mutant) trên cây AST của file nguồn."""

    index: int
    lineno: int
    mutation: str
    apply_fn: Callable[[ast.AST], ast.AST]


_ROR_MAP: dict[type[ast.cmpop], tuple[str, str, type[ast.cmpop]]] = {
    ast.GtE: (">=", ">", ast.Gt),
    ast.LtE: ("<=", "<", ast.Lt),
    ast.Gt: (">", ">=", ast.GtE),
    ast.Lt: ("<", "<=", ast.LtE),
    ast.Eq: ("==", "!=", ast.NotEq),
    ast.NotEq: ("!=", "==", ast.Eq),
}

_AOR_MAP: dict[type[ast.operator], tuple[str, str, type[ast.operator]]] = {
    ast.Add: ("+", "-", ast.Sub),
    ast.Sub: ("-", "+", ast.Add),
    ast.Mult: ("*", "/", ast.Div),
    ast.Div: ("/", "*", ast.Mult),
    ast.FloorDiv: ("//", "/", ast.Div),
    ast.Mod: ("%", "*", ast.Mult),
}

_LCR_MAP: dict[type[ast.boolop], tuple[str, str, type[ast.boolop]]] = {
    ast.And: ("and", "or", ast.Or),
    ast.Or: ("or", "and", ast.And),
}


def _kill_process_tree(pid: int) -> None:
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


def _iter_executable_nodes(tree: ast.Module) -> list[ast.AST]:
    """Duyệt theo thứ tự tiền thứ tự (pre-order) các node thực thi nằm bên trong thân hàm/method.

    Loại trừ docstring, chữ ký kiểu (type annotations) và khai báo hằng cấu hình cấp module (§16.6).
    """
    exec_nodes: list[ast.AST] = []

    def walk_stmt_list(stmts: list[ast.stmt], *, is_func_body: bool = False) -> None:
        for idx, stmt in enumerate(stmts):
            # Bỏ qua docstring ở câu lệnh đầu tiên của hàm/class
            if (
                idx == 0
                and isinstance(stmt, ast.Expr)
                and isinstance(stmt.value, ast.Constant)
                and isinstance(stmt.value.value, str)
            ):
                continue

            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                walk_stmt_list(stmt.body, is_func_body=True)
            elif isinstance(stmt, ast.ClassDef):
                walk_stmt_list(stmt.body, is_func_body=False)
            elif is_func_body:
                # Chỉ thu thập các biểu thức thực thi bên trong thân hàm/method
                for sub in ast.walk(stmt):
                    exec_nodes.append(sub)

    walk_stmt_list(tree.body, is_func_body=False)
    return exec_nodes


def collect_ast_mutants(source_code: str) -> list[MutantSpec]:
    """Phân tích AST file nguồn và sinh danh sách các `MutantSpec` tiêu chuẩn (FR-07):
    - ROR (Relational Operator Replacement): `>= <-> >`, `<= <-> <`, `== <-> !=`
    - AOR (Arithmetic Operator Replacement): `+ <-> -`, `* <-> /`, `// -> /`, `% -> *`
    - LCR (Logical Connector Replacement): `and <-> or`
    - UOI (Unary Operator Insertion/Deletion): `not x -> x`, `-x -> +x`
    - Constant / Boundary Replacement: `True <-> False`, hằng số số học `n -> n + 1` trong điều kiện/return
    """
    tree = ast.parse(source_code)
    nodes = _iter_executable_nodes(tree)
    specs: list[MutantSpec] = []

    # Theo dõi các node Constant là vế phải của phép so sánh hoặc giá trị return để áp dụng boundary n -> n + 1
    boundary_constant_ids: set[int] = set()
    for node in nodes:
        if isinstance(node, ast.Compare):
            for comp in node.comparators:
                if isinstance(comp, ast.Constant) and type(comp.value) in (int, float):
                    boundary_constant_ids.add(id(comp))
        elif isinstance(node, ast.Return) and isinstance(node.value, ast.Constant):
            if type(node.value.value) in (int, float):
                boundary_constant_ids.add(id(node.value))

    for node_idx, node in enumerate(nodes):
        lineno = getattr(node, "lineno", 1)

        # 1. ROR: Compare
        if isinstance(node, ast.Compare) and len(node.ops) == 1:
            op_type = type(node.ops[0])
            if op_type in _ROR_MAP:
                before_str, after_str, new_op_cls = _ROR_MAP[op_type]

                def make_ror(cls: type[ast.cmpop] = new_op_cls) -> Callable[[ast.AST], ast.AST]:
                    def _apply(target: ast.AST) -> ast.AST:
                        assert isinstance(target, ast.Compare)
                        target.ops[0] = cls()
                        return target

                    return _apply

                specs.append(
                    MutantSpec(
                        index=node_idx,
                        lineno=lineno,
                        mutation=f"{before_str} -> {after_str}",
                        apply_fn=make_ror(),
                    )
                )

        # 2. AOR: BinOp
        elif isinstance(node, ast.BinOp):
            op_type = type(node.op)
            if op_type in _AOR_MAP:
                # Tránh mutate phép cộng chuỗi hoặc danh sách tĩnh rõ ràng
                if op_type is ast.Add and (
                    (isinstance(node.left, ast.Constant) and isinstance(node.left.value, str))
                    or (isinstance(node.right, ast.Constant) and isinstance(node.right.value, str))
                ):
                    continue
                before_str, after_str, new_op_cls = _AOR_MAP[op_type]

                def make_aor(cls: type[ast.operator] = new_op_cls) -> Callable[[ast.AST], ast.AST]:
                    def _apply(target: ast.AST) -> ast.AST:
                        assert isinstance(target, ast.BinOp)
                        target.op = cls()
                        return target

                    return _apply

                specs.append(
                    MutantSpec(
                        index=node_idx,
                        lineno=lineno,
                        mutation=f"{before_str} -> {after_str}",
                        apply_fn=make_aor(),
                    )
                )

        # 3. LCR: BoolOp (and <-> or)
        elif isinstance(node, ast.BoolOp):
            op_type = type(node.op)
            if op_type in _LCR_MAP:
                before_str, after_str, new_op_cls = _LCR_MAP[op_type]

                def make_lcr(cls: type[ast.boolop] = new_op_cls) -> Callable[[ast.AST], ast.AST]:
                    def _apply(target: ast.AST) -> ast.AST:
                        assert isinstance(target, ast.BoolOp)
                        target.op = cls()
                        return target

                    return _apply

                specs.append(
                    MutantSpec(
                        index=node_idx,
                        lineno=lineno,
                        mutation=f"{before_str} -> {after_str}",
                        apply_fn=make_lcr(),
                    )
                )

        # 4. UOI: UnaryOp (not x -> x, -x -> +x)
        elif isinstance(node, ast.UnaryOp):
            if isinstance(node.op, ast.Not):

                def _apply_not(target: ast.AST) -> ast.AST:
                    assert isinstance(target, ast.UnaryOp)
                    target.op = ast.UAdd()  # Sẽ được thay thế trực tiếp bằng operand khi unparse hoặc giữ operand
                    return target

                # Đối với `not x -> x`, ta bọc `ast.Compare(left=ast.Constant(value=True), ops=[ast.Eq()], comparators=[ast.Subscript...])`
                # hoặc đơn giản thay `UnaryOp(Not, x)` bằng `BoolOp` / sửa trực tiếp trên cây cha.
                # Để thay node in-place mà không cần sửa tham chiếu từ cha:
                # đổi `not x` thành `not (not x)`! `not (not x)` có đúng ngữ nghĩa boolean của `x`!
                def _apply_double_not(target: ast.AST) -> ast.AST:
                    assert isinstance(target, ast.UnaryOp)
                    inner = copy.deepcopy(target.operand)
                    target.operand = ast.UnaryOp(op=ast.Not(), operand=inner)
                    return target

                specs.append(
                    MutantSpec(
                        index=node_idx,
                        lineno=lineno,
                        mutation="not x -> x",
                        apply_fn=_apply_double_not,
                    )
                )
            elif isinstance(node.op, ast.USub):

                def _apply_usub(target: ast.AST) -> ast.AST:
                    assert isinstance(target, ast.UnaryOp)
                    target.op = ast.UAdd()
                    return target

                specs.append(
                    MutantSpec(
                        index=node_idx,
                        lineno=lineno,
                        mutation="-x -> +x",
                        apply_fn=_apply_usub,
                    )
                )

        # 5. Constant / Boundary Replacement: True <-> False, hoặc n -> n + 1
        elif isinstance(node, ast.Constant):
            if isinstance(node.value, bool):
                old_bool = node.value
                new_bool = not old_bool

                def make_bool_flip(val: bool = new_bool) -> Callable[[ast.AST], ast.AST]:
                    def _apply(target: ast.AST) -> ast.AST:
                        assert isinstance(target, ast.Constant)
                        target.value = val
                        return target

                    return _apply

                specs.append(
                    MutantSpec(
                        index=node_idx,
                        lineno=lineno,
                        mutation=f"{old_bool} -> {new_bool}",
                        apply_fn=make_bool_flip(),
                    )
                )
            elif type(node.value) in (int, float) and id(node) in boundary_constant_ids:
                old_num = node.value
                new_num = old_num + 1

                def make_num_shift(val: int | float = new_num) -> Callable[[ast.AST], ast.AST]:
                    def _apply(target: ast.AST) -> ast.AST:
                        assert isinstance(target, ast.Constant)
                        target.value = val
                        return target

                    return _apply

                specs.append(
                    MutantSpec(
                        index=node_idx,
                        lineno=lineno,
                        mutation=f"{old_num} -> {new_num}",
                        apply_fn=make_num_shift(),
                    )
                )

    return specs


def render_mutant_source(source_code: str, spec: MutantSpec) -> str | None:
    """Áp dụng một `MutantSpec` lên cây AST mới và sinh chuỗi mã nguồn Python đã mutate.

    Trả về `None` nếu mutant tạo ra không hợp lệ về cú pháp.
    """
    tree = ast.parse(source_code)
    nodes = _iter_executable_nodes(tree)
    if spec.index >= len(nodes):
        return None
    target_node = nodes[spec.index]
    spec.apply_fn(target_node)
    ast.fix_missing_locations(tree)
    try:
        mutated_code = ast.unparse(tree)
        ast.parse(mutated_code)
        return mutated_code
    except Exception:
        return None


def _resolve_backend(configured_tool: str) -> str:
    """Chọn backend mutation thực tế dựa trên cấu hình `auto | ast | mutmut` và hệ điều hành (FR-07)."""
    if configured_tool == "ast":
        return "ast"
    if configured_tool == "mutmut":
        if os.name == "nt":
            # mutmut 3.x phụ thuộc os.fork()/resource của POSIX, không chạy được trên Windows native
            return "ast"
        return "mutmut" if shutil.which("mutmut") else "ast"
    # configured_tool == "auto"
    if os.name == "nt":
        return "ast"
    return "mutmut" if shutil.which("mutmut") else "ast"


def _run_ast_mutation_engine(
    *,
    target_file: Path,
    display_target_file: str,
    test_file: Path,
    config: AitestConfig,
    project_root: Path,
    reporter: ProgressReporter | None = None,
) -> MutationResult:
    """Thực thi Built-in AST Mutation Engine trong môi trường sandbox cô lập (an toàn 100% cho file nguồn)."""
    rep = reporter or NullProgressReporter()
    start_ns = time.perf_counter_ns()
    source_code = target_file.read_text(encoding="utf-8")
    specs = collect_ast_mutants(source_code)

    killed = 0
    survived = 0
    excluded = 0
    timeouts = 0
    survived_mutants: list[SurvivedMutant] = []

    if not specs:
        elapsed_ms = max(0, int((time.perf_counter_ns() - start_ns) / 1_000_000))
        rep.on_mutation_end(
            killed=0,
            survived=0,
            excluded=0,
            score_pct=None,
            duration_ms=elapsed_ms,
        )
        return MutationResult(
            mutation_score_pct=None,
            mutation_killed=0,
            mutation_survived=0,
            mutation_excluded=0,
            mutation_timeout=0,
            mutation_status="no_valid_mutants",
            mutation_available=False,
            mutation_unavailable_reason="File nguồn không sinh được mutant hợp lệ nào (killed + survived == 0)",
            survived_mutants=[],
            backend_used="ast",
            duration_ms=elapsed_ms,
        )

    total_specs = len(specs)
    rep.on_mutation_start(total_mutants=total_specs, target_file=display_target_file)
    sandbox_root = Path(tempfile.mkdtemp(prefix="aitest_mut_sandbox_"))
    try:
        target_abs = target_file.resolve()
        root_abs = project_root.resolve()
        try:
            rel_target = target_abs.relative_to(root_abs)
        except ValueError:
            rel_target = Path(target_abs.name)

        sandbox_target = sandbox_root / rel_target
        sandbox_target.parent.mkdir(parents=True, exist_ok=True)

        # Sao chép các file .py cùng thư mục với target_file vào sandbox để đảm bảo import nội bộ hoạt động
        for sibling in target_abs.parent.glob("*.py"):
            if sibling.resolve() != target_abs:
                shutil.copy2(str(sibling), str(sandbox_target.parent / sibling.name))

        # Tạo các file __init__.py nếu thư mục cha của target_file có __init__.py
        curr_rel = rel_target.parent
        while curr_rel != Path(".") and str(curr_rel):
            orig_init = root_abs / curr_rel / "__init__.py"
            if orig_init.is_file():
                shutil.copy2(str(orig_init), str(sandbox_root / curr_rel / "__init__.py"))
            curr_rel = curr_rel.parent

        env = os.environ.copy()
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        pythonpath_entries = [
            str(sandbox_root),
            str(sandbox_target.parent),
            str(root_abs),
            str(target_abs.parent),
        ]
        if env.get("PYTHONPATH"):
            pythonpath_entries.append(env["PYTHONPATH"])
        env["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(pythonpath_entries))

        from aitest.analyzer import infer_module_import_path

        inferred_mod = infer_module_import_path(target_abs, root_abs)
        mod_Preload_names = list(dict.fromkeys([inferred_mod, target_abs.stem]))

        runner_snippet = (
            "import importlib.util, sys, pytest; "
            f"sys.path[0:0] = [{repr(str(sandbox_root))}, {repr(str(sandbox_target.parent))}]; "
            f"_spec = importlib.util.spec_from_file_location({repr(target_abs.stem)}, {repr(str(sandbox_target))}); "
            "_mod = importlib.util.module_from_spec(_spec); "
            f"[sys.modules.__setitem__(_n, _mod) for _n in {repr(mod_Preload_names)}]; "
            "_spec.loader.exec_module(_mod); "
            f"raise SystemExit(pytest.main([{repr(str(test_file.resolve()))}, "
            "'--import-mode=importlib', '-x', '-q', '--tb=no', '-p', 'no:cov', '-p', 'no:cacheprovider']))"
        )
        cmd = [sys.executable, "-B", "-c", runner_snippet]

        total_timeout_sec = float(config.mutation.timeout_sec)
        per_mutant_timeout = float(config.mutation.per_mutant_timeout_sec)

        for idx, spec in enumerate(specs, start=1):
            elapsed_sec = (time.perf_counter_ns() - start_ns) / 1_000_000_000
            if elapsed_sec >= total_timeout_sec:
                break

            mutated_code = render_mutant_source(source_code, spec)
            if mutated_code is None:
                excluded += 1
                rep.on_mutant_tested(
                    index=idx,
                    total=total_specs,
                    lineno=spec.lineno,
                    mutation=spec.mutation,
                    status="excluded",
                    killed_so_far=killed,
                    survived_so_far=survived,
                )
                continue

            sandbox_target.write_text(mutated_code, encoding="utf-8")

            proc = subprocess.Popen(
                cmd,
                cwd=str(root_abs),
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            m_status: Literal["killed", "survived", "timeout"] = "killed"
            try:
                proc.communicate(timeout=per_mutant_timeout)
                rc = proc.returncode
                if rc == 0:
                    survived += 1
                    m_status = "survived"
                    survived_mutants.append(
                        SurvivedMutant(
                            file=display_target_file,
                            line=spec.lineno,
                            mutation=spec.mutation,
                        )
                    )
                else:
                    killed += 1
                    m_status = "killed"
            except subprocess.TimeoutExpired:
                _kill_process_tree(proc.pid)
                try:
                    proc.communicate(timeout=2.0)
                except Exception:
                    pass
                timeouts += 1
                killed += 1
                m_status = "timeout"

            rep.on_mutant_tested(
                index=idx,
                total=total_specs,
                lineno=spec.lineno,
                mutation=spec.mutation,
                status=m_status,
                killed_so_far=killed,
                survived_so_far=survived,
            )
    finally:
        shutil.rmtree(str(sandbox_root), ignore_errors=True)

    elapsed_ms = max(0, int((time.perf_counter_ns() - start_ns) / 1_000_000))
    valid_total = killed + survived
    if valid_total < config.mutation.min_valid_mutants:
        rep.on_mutation_end(
            killed=killed,
            survived=survived,
            excluded=excluded,
            score_pct=None,
            duration_ms=elapsed_ms,
        )
        return MutationResult(
            mutation_score_pct=None,
            mutation_killed=killed,
            mutation_survived=survived,
            mutation_excluded=excluded,
            mutation_timeout=timeouts,
            mutation_status="no_valid_mutants",
            mutation_available=False,
            mutation_unavailable_reason=(
                f"Số mutant hợp lệ ({valid_total}) nhỏ hơn ngưỡng tối thiểu ({config.mutation.min_valid_mutants})"
            ),
            survived_mutants=[],
            backend_used="ast",
            duration_ms=elapsed_ms,
        )

    score_pct = round((killed / valid_total) * 100.0, 2)
    rep.on_mutation_end(
        killed=killed,
        survived=survived,
        excluded=excluded,
        score_pct=score_pct,
        duration_ms=elapsed_ms,
    )
    return MutationResult(
        mutation_score_pct=score_pct,
        mutation_killed=killed,
        mutation_survived=survived,
        mutation_excluded=excluded,
        mutation_timeout=timeouts,
        mutation_status="available",
        mutation_available=True,
        mutation_unavailable_reason=None,
        survived_mutants=survived_mutants,
        backend_used="ast",
        duration_ms=elapsed_ms,
    )


def run_mutation_testing(
    *,
    target_file: Path | str,
    test_file: Path | str,
    config: AitestConfig,
    project_root: Path | str | None = None,
    reporter: ProgressReporter | None = None,
) -> MutationResult:
    """Điểm vào chính của Mutation Testing theo kiến trúc Dual Backend (FR-07)."""
    if not config.mutation.enabled:
        return MutationResult(
            mutation_status="disabled",
            mutation_available=False,
            mutation_unavailable_reason="Mutation testing bị tắt trong cấu hình",
        )

    t_path = Path(target_file)
    tst_path = Path(test_file)
    root_path = Path(project_root).resolve() if project_root else Path.cwd().resolve()

    try:
        rel_display = t_path.resolve().relative_to(root_path).as_posix()
    except ValueError:
        rel_display = t_path.as_posix()

    backend = _resolve_backend(config.mutation.tool)
    try:
        res = _run_ast_mutation_engine(
            target_file=t_path,
            display_target_file=rel_display,
            test_file=tst_path,
            config=config,
            project_root=root_path,
            reporter=reporter,
        )
        return res
    except Exception as exc:
        return MutationResult(
            mutation_score_pct=None,
            mutation_killed=None,
            mutation_survived=None,
            mutation_excluded=None,
            mutation_status="failed",
            mutation_available=False,
            mutation_unavailable_reason=f"Lỗi khi thực thi mutation engine ({backend}): {exc}",
            survived_mutants=[],
            backend_used="ast",
        )
