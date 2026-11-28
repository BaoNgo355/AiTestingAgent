"""Dual Mutation Backend: Built-in AST Mutation Engine đa nền tảng (Windows/POSIX) và mutmut Adapter (SRS v2.3.0 §7 FR-07, AC-12, AC-21, AC-22)."""

from __future__ import annotations

import ast
import copy
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

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


def _unit_guard_compare_ids(nodes: list[ast.AST]) -> set[int]:
    """Nhận diện các phép so sánh `<= 1` fast-path guard tương đương với `< 1` để giảm equivalent mutants."""
    ids: set[int] = set()
    for node in nodes:
        if (
            isinstance(node, ast.If)
            and isinstance(node.test, ast.Compare)
            and len(node.test.ops) == 1
            and isinstance(node.test.ops[0], ast.LtE)
            and len(node.test.comparators) == 1
            and isinstance(node.test.comparators[0], ast.Constant)
            and node.test.comparators[0].value == 1
            and len(node.body) == 1
            and isinstance(node.body[0], ast.Return)
            and isinstance(node.body[0].value, ast.Constant)
            and node.body[0].value.value in (1, True)
        ):
            ids.add(id(node.test))
    return ids


def collect_ast_mutants(source_code: str) -> list[MutantSpec]:
    """Phân tích AST file nguồn và sinh danh sách các `MutantSpec` tiêu chuẩn (FR-07):
    - ROR (Relational Operator Replacement): hỗ trợ cả chained comparison (`a <= x <= b`)
    - AOR (Arithmetic Operator Replacement): hỗ trợ cả `BinOp` và `AugAssign` (`+=`, `-=`, `*=`, ...)
    - LCR (Logical Connector Replacement): `and <-> or`
    - UOI (Unary Operator Insertion/Deletion): `not x -> x`, `-x -> +x`
    - Constant / Boundary Replacement: `True <-> False`, hằng số số học `n -> n + 1` (tránh trùng lặp ROR số nguyên)
    """
    tree = ast.parse(source_code)
    nodes = _iter_executable_nodes(tree)
    specs: list[MutantSpec] = []
    unit_guard_ids = _unit_guard_compare_ids(nodes)

    # Theo dõi các node Constant là vế phải của phép so sánh == / != (hoặc số thực) hoặc giá trị return
    # để áp dụng boundary n -> n + 1 (không nhân bản trên <, <=, >, >= với int vì ROR đã phủ)
    boundary_constant_ids: set[int] = set()
    for node in nodes:
        if isinstance(node, ast.Compare):
            is_eq_cmp = all(isinstance(op, (ast.Eq, ast.NotEq)) for op in node.ops)
            for comp in node.comparators:
                if isinstance(comp, ast.Constant):
                    if type(comp.value) is float or (type(comp.value) is int and is_eq_cmp):
                        boundary_constant_ids.add(id(comp))
        elif isinstance(node, ast.Return) and isinstance(node.value, ast.Constant):
            if type(node.value.value) in (int, float):
                boundary_constant_ids.add(id(node.value))

    for node_idx, node in enumerate(nodes):
        lineno = getattr(node, "lineno", 1)

        # 1. ROR: Compare (hỗ trợ cả single và chained comparison như "a" <= ch <= "z")
        if isinstance(node, ast.Compare) and node.ops:
            for op_idx, op_node in enumerate(node.ops):
                op_type = type(op_node)
                if op_type in _ROR_MAP:
                    if op_type is ast.LtE and id(node) in unit_guard_ids:
                        continue
                    before_str, after_str, new_op_cls = _ROR_MAP[op_type]

                    def make_ror(
                        pos: int = op_idx,
                        cls: type[ast.cmpop] = new_op_cls,
                    ) -> Callable[[ast.AST], ast.AST]:
                        def _apply(target: ast.AST) -> ast.AST:
                            assert isinstance(target, ast.Compare)
                            target.ops[pos] = cls()
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
                    (isinstance(node.left, ast.Constant) and isinstance(node.left.value, (str, bytes)))
                    or (isinstance(node.right, ast.Constant) and isinstance(node.right.value, (str, bytes)))
                    or isinstance(node.left, (ast.List, ast.Tuple))
                    or isinstance(node.right, (ast.List, ast.Tuple))
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

        # 2b. AOR cho AugAssign (+=, -=, *=, /=, //=, %=)
        elif isinstance(node, ast.AugAssign):
            op_type = type(node.op)
            if op_type in _AOR_MAP:
                if op_type is ast.Add and (
                    (isinstance(node.value, ast.Constant) and isinstance(node.value.value, (str, bytes)))
                    or isinstance(node.value, (ast.List, ast.Tuple))
                ):
                    continue
                before_str, after_str, new_op_cls = _AOR_MAP[op_type]

                def make_aug_aor(cls: type[ast.operator] = new_op_cls) -> Callable[[ast.AST], ast.AST]:
                    def _apply(target: ast.AST) -> ast.AST:
                        assert isinstance(target, ast.AugAssign)
                        target.op = cls()
                        return target

                    return _apply

                specs.append(
                    MutantSpec(
                        index=node_idx,
                        lineno=lineno,
                        mutation=f"{before_str}= -> {after_str}=",
                        apply_fn=make_aug_aor(),
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


_PERSISTENT_WORKER_SCRIPT = """\
import contextlib
import importlib
import importlib.util
import io
import json
import linecache
import os
import sys
import pytest

_PROTO_OUT = sys.stdout
_SANDBOX_ROOT = sys.argv[1]
_SANDBOX_ROOT_NORM = os.path.normcase(os.path.abspath(_SANDBOX_ROOT))
_SANDBOX_TARGET = sys.argv[2]
_TEST_FILE = sys.argv[3]
_TEST_FILE_NORM = os.path.normcase(os.path.abspath(_TEST_FILE))
_TARGET_STEM = sys.argv[4]
_PRELOAD_NAMES = json.loads(sys.argv[5])
_INFERRED_MOD = sys.argv[6]

_KEEP_PREFIXES = ("pytest", "_pytest", "pluggy", "iniconfig", "packaging", "py.")
_BASE_MODULES = set(sys.modules.keys())

sys.path[0:0] = [_SANDBOX_ROOT, sys.argv[7]]


def _evict_stale_modules() -> None:
    test_stem = sys.argv[8]
    for name in list(sys.modules.keys()):
        if name in _PRELOAD_NAMES or name == test_stem:
            sys.modules.pop(name, None)
            continue
        if name in _BASE_MODULES or name.startswith(_KEEP_PREFIXES):
            continue
        mod = sys.modules.get(name)
        mod_file = getattr(mod, "__file__", None) or ""
        mod_file_norm = os.path.normcase(os.path.abspath(mod_file)) if mod_file else ""
        if (
            mod_file_norm.startswith(_SANDBOX_ROOT_NORM)
            or mod_file_norm == _TEST_FILE_NORM
            or name.split(".", 1)[0] in (_TARGET_STEM, test_stem)
        ):
            sys.modules.pop(name, None)


while True:
    raw_req = sys.stdin.readline()
    if not raw_req:
        break
    line_req = raw_req.strip()
    if not line_req:
        continue
    try:
        req = json.loads(line_req)
    except Exception:
        continue
    if req.get("cmd") == "quit":
        break

    rc = 1
    try:
        _evict_stale_modules()
        importlib.invalidate_caches()
        linecache.clearcache()
        spec = importlib.util.spec_from_file_location(_INFERRED_MOD, _SANDBOX_TARGET)
        if spec is None or spec.loader is None:
            rc = 1
        else:
            mod = importlib.util.module_from_spec(spec)
            for mod_name in _PRELOAD_NAMES:
                sys.modules[mod_name] = mod
            spec.loader.exec_module(mod)
            sink_out = io.StringIO()
            sink_err = io.StringIO()
            with contextlib.redirect_stdout(sink_out), contextlib.redirect_stderr(sink_err):
                rc = int(
                    pytest.main(
                        [
                            _TEST_FILE,
                            "--import-mode=importlib",
                            "-x",
                            "-q",
                            "--tb=no",
                            "-p",
                            "no:cov",
                            "-p",
                            "no:cacheprovider",
                        ]
                    )
                )
    except SystemExit as exc:
        rc = int(exc.code) if isinstance(exc.code, int) else 1
    except BaseException:
        rc = 1

    _PROTO_OUT.write(json.dumps({"rc": rc}) + "\\n")
    _PROTO_OUT.flush()
"""


class _MutantWorkerSession:
    """Quản lý Persistent Python Worker trong sandbox để chạy tuần tự nhiều mutant nhanh và cô lập."""

    def __init__(
        self,
        *,
        worker_script: Path,
        sandbox_root: Path,
        sandbox_target: Path,
        test_file: Path,
        target_stem: str,
        preload_names: list[str],
        inferred_mod: str,
        root_abs: Path,
        env: dict[str, str],
    ) -> None:
        self._cmd = [
            sys.executable,
            "-B",
            str(worker_script),
            str(sandbox_root),
            str(sandbox_target),
            str(test_file.resolve()),
            target_stem,
            json.dumps(preload_names),
            inferred_mod,
            str(sandbox_target.parent),
            test_file.stem,
        ]
        self._cwd = str(root_abs)
        self._env = env
        self._proc: subprocess.Popen[str] | None = None
        self._queue: queue.Queue[str | None] | None = None

    def _ensure_started(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            return
        self._proc = subprocess.Popen(
            self._cmd,
            cwd=self._cwd,
            env=self._env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        q: queue.Queue[str | None] = queue.Queue()
        self._queue = q
        stdout_pipe = self._proc.stdout

        def _reader() -> None:
            try:
                assert stdout_pipe is not None
                for line in stdout_pipe:
                    q.put(line)
            except Exception:
                pass
            finally:
                q.put(None)

        t = threading.Thread(target=_reader, daemon=True)
        t.start()

    def run_once(self, timeout_sec: float) -> tuple[Literal["killed", "survived", "timeout"], int]:
        self._ensure_started()
        assert self._proc is not None and self._proc.stdin is not None and self._queue is not None
        try:
            self._proc.stdin.write('{"cmd":"run"}\n')
            self._proc.stdin.flush()
            resp_line = self._queue.get(timeout=timeout_sec)
        except (BrokenPipeError, OSError, queue.Empty):
            self.kill()
            return "timeout", 124

        if resp_line is None:
            self.kill()
            return "killed", 1

        try:
            payload = json.loads(resp_line)
            rc = int(payload.get("rc", 1))
        except Exception:
            self.kill()
            return "killed", 1

        return ("survived" if rc == 0 else "killed"), rc

    def kill(self) -> None:
        if self._proc is not None:
            if self._proc.poll() is None:
                _kill_process_tree(self._proc.pid)
            try:
                self._proc.communicate(timeout=1.5)
            except Exception:
                pass
            self._proc = None
            self._queue = None

    def close(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            try:
                if self._proc.stdin is not None:
                    self._proc.stdin.write('{"cmd":"quit"}\n')
                    self._proc.stdin.flush()
                self._proc.wait(timeout=1.5)
            except Exception:
                self.kill()
        self._proc = None
        self._queue = None


def _run_ast_mutation_engine(
    *,
    target_file: Path,
    display_target_file: str,
    test_file: Path,
    config: AitestConfig,
    project_root: Path,
    executed_lines: set[int] | None = None,
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
    worker_session: _MutantWorkerSession | None = None
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
        mod_preload_names = list(dict.fromkeys([inferred_mod, target_abs.stem]))

        worker_script = sandbox_root / "_aitest_mut_worker.py"
        worker_script.write_text(_PERSISTENT_WORKER_SCRIPT, encoding="utf-8")
        worker_session = _MutantWorkerSession(
            worker_script=worker_script,
            sandbox_root=sandbox_root,
            sandbox_target=sandbox_target,
            test_file=test_file,
            target_stem=target_abs.stem,
            preload_names=mod_preload_names,
            inferred_mod=inferred_mod,
            root_abs=root_abs,
            env=env,
        )

        total_timeout_sec = float(config.mutation.timeout_sec)
        per_mutant_timeout = float(config.mutation.per_mutant_timeout_sec)
        has_coverage_filter = executed_lines is not None and len(executed_lines) > 0

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

            # Lọc nhanh theo line coverage: dòng chưa từng được thực thi chắc chắn sẽ survived
            if has_coverage_filter and spec.lineno not in executed_lines:  # type: ignore[operator]
                survived += 1
                survived_mutants.append(
                    SurvivedMutant(
                        file=display_target_file,
                        line=spec.lineno,
                        mutation=spec.mutation,
                    )
                )
                rep.on_mutant_tested(
                    index=idx,
                    total=total_specs,
                    lineno=spec.lineno,
                    mutation=spec.mutation,
                    status="survived",
                    killed_so_far=killed,
                    survived_so_far=survived,
                )
                continue

            sandbox_target.write_text(mutated_code, encoding="utf-8")
            m_status, _ = worker_session.run_once(per_mutant_timeout)
            if m_status == "survived":
                survived += 1
                survived_mutants.append(
                    SurvivedMutant(
                        file=display_target_file,
                        line=spec.lineno,
                        mutation=spec.mutation,
                    )
                )
            elif m_status == "timeout":
                timeouts += 1
                killed += 1
            else:
                killed += 1

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
        if worker_session is not None:
            worker_session.close()
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


def _run_mutmut_engine(
    *,
    target_file: Path,
    display_target_file: str,
    test_file: Path,
    config: AitestConfig,
    project_root: Path,
    reporter: ProgressReporter | None = None,
) -> MutationResult:
    """External POSIX adapter cho `mutmut` trên bản sao sandbox cô lập (FR-07)."""
    rep = reporter or NullProgressReporter()
    start_ns = time.perf_counter_ns()
    mutmut_bin = shutil.which("mutmut")
    if not mutmut_bin:
        raise RuntimeError("Không tìm thấy lệnh `mutmut` trong PATH")

    sandbox_root = Path(tempfile.mkdtemp(prefix="aitest_mutmut_sandbox_"))
    try:
        target_abs = target_file.resolve()
        test_abs = test_file.resolve()
        root_abs = project_root.resolve()
        try:
            rel_target = target_abs.relative_to(root_abs)
        except ValueError:
            rel_target = Path(target_abs.name)
        try:
            rel_test = test_abs.relative_to(root_abs)
        except ValueError:
            rel_test = Path("tests") / test_abs.name

        sandbox_target = sandbox_root / rel_target
        sandbox_test = sandbox_root / rel_test
        sandbox_target.parent.mkdir(parents=True, exist_ok=True)
        sandbox_test.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(target_abs), str(sandbox_target))
        shutil.copy2(str(test_abs), str(sandbox_test))

        for sibling in target_abs.parent.glob("*.py"):
            if sibling.resolve() != target_abs:
                shutil.copy2(str(sibling), str(sandbox_target.parent / sibling.name))

        setup_cfg = sandbox_root / "setup.cfg"
        setup_cfg.write_text(
            "[mutmut]\n"
            f"paths_to_mutate={rel_target.as_posix()}\n"
            f"tests_dir={sandbox_test.parent.relative_to(sandbox_root).as_posix()}\n",
            encoding="utf-8",
        )

        env = os.environ.copy()
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["PYTHONPATH"] = os.pathsep.join(
            [str(sandbox_root), str(sandbox_target.parent), env.get("PYTHONPATH", "")]
        ).rstrip(os.pathsep)

        run_proc = subprocess.run(
            [mutmut_bin, "run"],
            cwd=str(sandbox_root),
            env=env,
            capture_output=True,
            text=True,
            timeout=float(config.mutation.timeout_sec),
            check=False,
        )
        res_proc = subprocess.run(
            [mutmut_bin, "results"],
            cwd=str(sandbox_root),
            env=env,
            capture_output=True,
            text=True,
            timeout=30.0,
            check=False,
        )
        if run_proc.returncode not in (0, 2) and res_proc.returncode != 0:
            raise RuntimeError(
                f"mutmut exited with code {run_proc.returncode}: {run_proc.stderr.strip()}"
            )

        results_text = f"{res_proc.stdout}\n{res_proc.stderr}"
        killed = 0
        survived = 0
        timeouts = 0
        excluded = 0
        survived_mutants: list[SurvivedMutant] = []

        for line in results_text.splitlines():
            low = line.strip().lower()
            if not low:
                continue
            if "killed" in low and re.search(r":\s*killed\b|\bkilled\s*:", low):
                killed += 1
            elif "timeout" in low:
                timeouts += 1
                killed += 1
            elif "survived" in low:
                survived += 1
                line_match = re.search(r":(\d+)\b", line)
                ln_num = int(line_match.group(1)) if line_match else 1
                survived_mutants.append(
                    SurvivedMutant(
                        file=display_target_file,
                        line=max(1, ln_num),
                        mutation=line.strip()[:120],
                    )
                )
            elif "skipped" in low or "no tests" in low:
                excluded += 1

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
                mutation_unavailable_reason="mutmut không ghi nhận mutant hợp lệ nào (killed + survived == 0)",
                survived_mutants=[],
                backend_used="mutmut",
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
            backend_used="mutmut",
            duration_ms=elapsed_ms,
        )
    finally:
        shutil.rmtree(str(sandbox_root), ignore_errors=True)


def run_mutation_testing(
    *,
    target_file: Path | str,
    test_file: Path | str,
    config: AitestConfig,
    project_root: Path | str | None = None,
    executed_lines: set[int] | None = None,
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
    if backend == "mutmut":
        try:
            return _run_mutmut_engine(
                target_file=t_path,
                display_target_file=rel_display,
                test_file=tst_path,
                config=config,
                project_root=root_path,
                reporter=reporter,
            )
        except Exception as exc:
            if config.mutation.tool != "auto":
                return MutationResult(
                    mutation_score_pct=None,
                    mutation_killed=None,
                    mutation_survived=None,
                    mutation_excluded=None,
                    mutation_status="failed",
                    mutation_available=False,
                    mutation_unavailable_reason=f"Lỗi khi thực thi mutation engine (mutmut): {exc}",
                    survived_mutants=[],
                    backend_used="mutmut",
                )

    try:
        res = _run_ast_mutation_engine(
            target_file=t_path,
            display_target_file=rel_display,
            test_file=tst_path,
            config=config,
            project_root=root_path,
            executed_lines=executed_lines,
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
