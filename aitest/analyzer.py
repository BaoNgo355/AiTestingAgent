"""Phân tích AST cho file mã nguồn mục tiêu và file unit test (SRS v2.3.0 §7 FR-02, FR-04 & §9.1)."""

from __future__ import annotations

import ast
from pathlib import Path

from pydantic import BaseModel, Field


class ParameterInfo(BaseModel):
    """Thông tin một tham số của hàm hoặc phương thức."""

    name: str
    annotation: str | None = None
    default: str | None = None
    kind: str = "positional_or_keyword"


class FunctionInfo(BaseModel):
    """Thông tin trích xuất từ AST của một hàm hoặc phương thức."""

    name: str
    lineno: int
    end_lineno: int | None = None
    is_async: bool = False
    is_method: bool = False
    class_name: str | None = None
    parameters: list[ParameterInfo] = Field(default_factory=list)
    return_annotation: str | None = None
    decorators: list[str] = Field(default_factory=list)
    docstring: str | None = None
    raised_exceptions: list[str] = Field(default_factory=list)
    raise_conditions: list[str] = Field(default_factory=list)
    signature_str: str = ""


class ClassInfo(BaseModel):
    """Thông tin trích xuất từ AST của một lớp (ClassDef)."""

    name: str
    lineno: int
    end_lineno: int | None = None
    bases: list[str] = Field(default_factory=list)
    decorators: list[str] = Field(default_factory=list)
    docstring: str | None = None
    methods: list[FunctionInfo] = Field(default_factory=list)


class SourceAnalysis(BaseModel):
    """Kết quả phân tích AST toàn diện của một file mã nguồn Python mục tiêu."""

    file_path: str
    module_name: str
    imports: list[str] = Field(default_factory=list)
    functions: list[FunctionInfo] = Field(default_factory=list)
    classes: list[ClassInfo] = Field(default_factory=list)
    total_lines: int = 0

    def target_summary_lines(self) -> list[str]:
        """Tạo danh sách mô tả ngắn gọn các đối tượng cần test (kèm điều kiện raise chính xác) để đưa vào prompt."""
        lines: list[str] = []
        for fn in self.functions:
            prefix = "async def" if fn.is_async else "def"
            ret = f" -> {fn.return_annotation}" if fn.return_annotation else ""
            raises = f" [raises: {', '.join(fn.raised_exceptions)}]" if fn.raised_exceptions else ""
            lines.append(f"- Function `{prefix} {fn.name}{fn.signature_str}{ret}` (line {fn.lineno}){raises}")
            if fn.docstring:
                first_doc = fn.docstring.strip().splitlines()[0]
                lines.append(f"    Doc: {first_doc}")
            if fn.raise_conditions:
                lines.append("    Exact exception conditions (ONLY raise under these conditions):")
                for rc in fn.raise_conditions:
                    lines.append(f"      * {rc}")
        for cls in self.classes:
            bases_str = f"({', '.join(cls.bases)})" if cls.bases else ""
            lines.append(f"- Class `class {cls.name}{bases_str}` (line {cls.lineno})")
            for m in cls.methods:
                prefix = "async def" if m.is_async else "def"
                ret = f" -> {m.return_annotation}" if m.return_annotation else ""
                raises = f" [raises: {', '.join(m.raised_exceptions)}]" if m.raised_exceptions else ""
                lines.append(f"    * Method `{prefix} {m.name}{m.signature_str}{ret}` (line {m.lineno}){raises}")
                if m.raise_conditions:
                    for rc in m.raise_conditions:
                        lines.append(f"        - Raises: {rc}")
        return lines


class AssertionInfo(BaseModel):
    """Thông tin một câu lệnh kiểm tra (`assert` hoặc `with pytest.raises(...)`) trong hàm test."""

    kind: Literal_AssertionKind = "assert"  # type: ignore[valid-type]
    lineno: int
    source: str
    ast_dump: str
    raised_exception: str | None = None
    has_match: bool = False


from typing import Literal

Literal_AssertionKind = Literal["assert", "pytest_raises"]
AssertionInfo.model_rebuild()


class TestFunctionInfo(BaseModel):
    """Thông tin một hàm kiểm thử `test_*` trong file test."""

    __test__ = False

    name: str
    lineno: int
    end_lineno: int | None = None
    is_async: bool = False
    decorators: list[str] = Field(default_factory=list)
    has_skip_or_xfail: bool = False
    skip_markers: list[str] = Field(default_factory=list)
    assertions: list[AssertionInfo] = Field(default_factory=list)
    normalized_body_dump: str = ""
    source_code: str = ""


class HelperOrFixtureInfo(BaseModel):
    """Thông tin một fixture (`@pytest.fixture`) hoặc hàm/lớp helper trong file test."""

    name: str
    lineno: int
    is_fixture: bool = False
    is_class: bool = False
    normalized_dump: str = ""
    source_code: str = ""


class TestFileAnalysis(BaseModel):
    """Kết quả phân tích AST của một file unit test."""

    __test__ = False

    file_path: str
    imports: list[str] = Field(default_factory=list)
    test_functions: list[TestFunctionInfo] = Field(default_factory=list)
    fixtures: list[HelperOrFixtureInfo] = Field(default_factory=list)
    helpers: list[HelperOrFixtureInfo] = Field(default_factory=list)
    module_level_skips: list[str] = Field(default_factory=list)

    @property
    def test_names(self) -> list[str]:
        return [t.name for t in self.test_functions]

    @property
    def test_count(self) -> int:
        return len(self.test_functions)

    @property
    def total_assertion_count(self) -> int:
        return sum(len(t.assertions) for t in self.test_functions)


def _unparse_safe(node: ast.AST | None) -> str | None:
    if node is None:
        return None
    try:
        return ast.unparse(node).strip()
    except Exception:
        return None


def _extract_parameters(args: ast.arguments) -> list[ParameterInfo]:
    params: list[ParameterInfo] = []

    # posonlyargs + args
    posonly_count = len(args.posonlyargs)
    all_pos = list(args.posonlyargs) + list(args.args)
    num_defaults = len(args.defaults)
    default_start = len(all_pos) - num_defaults

    for idx, arg in enumerate(all_pos):
        kind = "positional_only" if idx < posonly_count else "positional_or_keyword"
        default_node = args.defaults[idx - default_start] if idx >= default_start else None
        params.append(
            ParameterInfo(
                name=arg.arg,
                annotation=_unparse_safe(arg.annotation),
                default=_unparse_safe(default_node),
                kind=kind,
            )
        )

    if args.vararg:
        params.append(
            ParameterInfo(
                name=f"*{args.vararg.arg}",
                annotation=_unparse_safe(args.vararg.annotation),
                kind="var_positional",
            )
        )

    for idx, kwarg in enumerate(args.kwonlyargs):
        kw_default = args.kw_defaults[idx] if idx < len(args.kw_defaults) else None
        params.append(
            ParameterInfo(
                name=kwarg.arg,
                annotation=_unparse_safe(kwarg.annotation),
                default=_unparse_safe(kw_default),
                kind="keyword_only",
            )
        )

    if args.kwarg:
        params.append(
            ParameterInfo(
                name=f"**{args.kwarg.arg}",
                annotation=_unparse_safe(args.kwarg.annotation),
                kind="var_keyword",
            )
        )

    return params


def _format_signature(params: list[ParameterInfo]) -> str:
    parts: list[str] = []
    for p in params:
        item = p.name
        if p.annotation:
            item += f": {p.annotation}"
        if p.default is not None:
            item += f" = {p.default}"
        parts.append(item)
    return f"({', '.join(parts)})"


def _extract_raised_exceptions(
    func_node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> tuple[list[str], list[str]]:
    """Trích xuất danh sách tên ngoại lệ và điều kiện `if` chính xác kích hoạt từng lệnh `raise`."""
    exceptions: list[str] = []
    conditions: list[str] = []

    def _describe_raise(exc_node: ast.expr) -> tuple[str | None, str | None]:
        if isinstance(exc_node, ast.Call):
            name = _unparse_safe(exc_node.func)
            msg = None
            if exc_node.args and isinstance(exc_node.args[0], ast.Constant) and isinstance(exc_node.args[0].value, str):
                msg = exc_node.args[0].value
            return name, msg
        if isinstance(exc_node, ast.Name):
            return exc_node.id, None
        return _unparse_safe(exc_node), None

    def _walk_stmts(stmts: list[ast.stmt], active_conds: list[str]) -> None:
        for stmt in stmts:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            if isinstance(stmt, ast.Raise) and stmt.exc is not None:
                name, msg = _describe_raise(stmt.exc)
                if name:
                    if name not in exceptions:
                        exceptions.append(name)
                    cond_text = " and ".join(active_conds) if active_conds else "unconditional"
                    msg_part = f'("{msg}")' if msg else ""
                    entry = f"`{name}{msg_part}` when `{cond_text}`"
                    if entry not in conditions:
                        conditions.append(entry)
            elif isinstance(stmt, ast.If):
                test_str = _unparse_safe(stmt.test) or "condition"
                _walk_stmts(stmt.body, [*active_conds, test_str])
                if stmt.orelse:
                    _walk_stmts(stmt.orelse, [*active_conds, f"not ({test_str})"])
            else:
                for child_field in ("body", "orelse", "finalbody", "handlers"):
                    sub = getattr(stmt, child_field, None)
                    if isinstance(sub, list):
                        for item in sub:
                            if isinstance(item, ast.ExceptHandler):
                                _walk_stmts(item.body, active_conds)
                            elif isinstance(item, ast.stmt):
                                _walk_stmts([item], active_conds)

    _walk_stmts(func_node.body, [])
    return exceptions, conditions


def _extract_function_info(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    *,
    is_method: bool = False,
    class_name: str | None = None,
) -> FunctionInfo:
    params = _extract_parameters(node.args)
    decorators = [d for d in (_unparse_safe(dec) for dec in node.decorator_list) if d]
    raised_excs, raise_conds = _extract_raised_exceptions(node)
    return FunctionInfo(
        name=node.name,
        lineno=node.lineno,
        end_lineno=getattr(node, "end_lineno", None),
        is_async=isinstance(node, ast.AsyncFunctionDef),
        is_method=is_method,
        class_name=class_name,
        parameters=params,
        return_annotation=_unparse_safe(node.returns),
        decorators=decorators,
        docstring=ast.get_docstring(node),
        raised_exceptions=raised_excs,
        raise_conditions=raise_conds,
        signature_str=_format_signature(params),
    )


def infer_module_import_path(file_path: Path | str, project_root: Path | str | None = None) -> str:
    """Suy luận đường dẫn module Python để import từ file nguồn (ví dụ: `examples.calculator` hoặc `calculator`)."""
    target = Path(file_path).resolve()
    root = Path(project_root).resolve() if project_root else Path.cwd().resolve()
    try:
        rel = target.relative_to(root)
        parts = list(rel.with_suffix("").parts)
        if parts and parts[-1] == "__init__":
            parts = parts[:-1]
        if parts:
            return ".".join(parts)
    except ValueError:
        pass
    return target.stem


def analyze_source_code(
    source_code: str,
    file_path: Path | str = "<string>",
    project_root: Path | str | None = None,
) -> SourceAnalysis:
    """Phân tích AST của chuỗi mã nguồn Python mục tiêu."""
    tree = ast.parse(source_code, filename=str(file_path))
    imports: list[str] = []
    functions: list[FunctionInfo] = []
    classes: list[ClassInfo] = []

    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            imp_str = _unparse_safe(node)
            if imp_str and imp_str not in imports:
                imports.append(imp_str)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.append(_extract_function_info(node))
        elif isinstance(node, ast.ClassDef):
            methods: list[FunctionInfo] = []
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    methods.append(
                        _extract_function_info(item, is_method=True, class_name=node.name)
                    )
            bases = [b for b in (_unparse_safe(base) for base in node.bases) if b]
            decorators = [d for d in (_unparse_safe(dec) for dec in node.decorator_list) if d]
            classes.append(
                ClassInfo(
                    name=node.name,
                    lineno=node.lineno,
                    end_lineno=getattr(node, "end_lineno", None),
                    bases=bases,
                    decorators=decorators,
                    docstring=ast.get_docstring(node),
                    methods=methods,
                )
            )

    module_name = (
        infer_module_import_path(file_path, project_root)
        if str(file_path) != "<string>"
        else "target_module"
    )
    return SourceAnalysis(
        file_path=str(file_path),
        module_name=module_name,
        imports=imports,
        functions=functions,
        classes=classes,
        total_lines=len(source_code.splitlines()),
    )


def extract_explicit_raise_lines(source_code: str) -> set[int]:
    """Trả về tập các số dòng chứa lệnh `raise` tường minh trong file mã nguồn."""
    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        return set()

    raise_lines: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Raise):
            start = getattr(node, "lineno", None)
            end = getattr(node, "end_lineno", None) or start
            if start is not None:
                for ln in range(start, (end or start) + 1):
                    raise_lines.add(ln)
    return raise_lines


def analyze_source_file(
    file_path: Path | str,
    project_root: Path | str | None = None,
) -> SourceAnalysis:
    """Đọc và phân tích AST của file mã nguồn Python mục tiêu."""
    path = Path(file_path)
    source = path.read_text(encoding="utf-8")
    return analyze_source_code(source, file_path=path, project_root=project_root)


# =====================================================================
# Phân tích AST cho file Unit Test
# =====================================================================

_SKIP_DECORATOR_KEYWORDS = (
    "pytest.mark.skip",
    "pytest.mark.skipif",
    "pytest.mark.xfail",
    "mark.skip",
    "mark.skipif",
    "mark.xfail",
    "unittest.skip",
    "unittest.skipIf",
    "unittest.expectedFailure",
)

_SKIP_CALL_NAMES = (
    "pytest.skip",
    "pytest.xfail",
    " pytest.importorskip",
    "skip",
    "xfail",
)


def _is_pytest_raises_call(node: ast.AST) -> str | None:
    """Kiểm tra xem node AST có phải lời gọi `pytest.raises(...)` không. Trả về tên exception nếu đúng."""
    if not isinstance(node, ast.Call):
        return None
    func_str = _unparse_safe(node.func) or ""
    if func_str in ("pytest.raises", "raises"):
        if node.args:
            return _unparse_safe(node.args[0]) or "Exception"
        return "Exception"
    return None


def _has_nonempty_match_kwarg(node: ast.AST) -> bool:
    """Kiểm tra xem `pytest.raises(...)` có truyền tham số `match=...` khác rỗng không."""
    if not isinstance(node, ast.Call):
        return False
    for kw in node.keywords:
        if kw.arg == "match":
            if isinstance(kw.value, ast.Constant):
                return kw.value.value not in (None, "")
            return True
    return False


def _is_fixture_decorator(dec: ast.AST) -> bool:
    dec_str = _unparse_safe(dec) or ""
    return (
        dec_str == "pytest.fixture"
        or dec_str.startswith("pytest.fixture(")
        or dec_str == "fixture"
        or dec_str.startswith("fixture(")
    )


def _detect_skip_markers(func_node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    markers: list[str] = []
    for dec in func_node.decorator_list:
        dec_str = _unparse_safe(dec) or ""
        for kw in _SKIP_DECORATOR_KEYWORDS:
            if dec_str == kw or dec_str.startswith(f"{kw}("):
                markers.append(f"@{dec_str}")
                break

    for sub in ast.walk(func_node):
        if isinstance(sub, ast.Call):
            func_name = _unparse_safe(sub.func) or ""
            if func_name in ("pytest.skip", "pytest.xfail", "pytest.importorskip"):
                call_src = _unparse_safe(sub) or func_name
                markers.append(call_src)
    return markers


def _extract_assertions_from_test(
    func_node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> list[AssertionInfo]:
    assertions: list[AssertionInfo] = []

    for sub in ast.walk(func_node):
        if isinstance(sub, ast.Assert):
            src = _unparse_safe(sub) or "assert"
            assertions.append(
                AssertionInfo(
                    kind="assert",
                    lineno=getattr(sub, "lineno", func_node.lineno),
                    source=src,
                    ast_dump=ast.dump(sub.test, include_attributes=False),
                )
            )
        elif isinstance(sub, (ast.With, ast.AsyncWith)):
            for item in sub.items:
                exc_type = _is_pytest_raises_call(item.context_expr)
                if exc_type is not None:
                    src = _unparse_safe(item.context_expr) or f"pytest.raises({exc_type})"
                    assertions.append(
                        AssertionInfo(
                            kind="pytest_raises",
                            lineno=getattr(sub, "lineno", func_node.lineno),
                            source=f"with {src}",
                            ast_dump=ast.dump(item.context_expr, include_attributes=False),
                            raised_exception=exc_type,
                            has_match=_has_nonempty_match_kwarg(item.context_expr),
                        )
                    )
        elif isinstance(sub, ast.Call):
            # Trường hợp gọi trực tiếp pytest.raises(ValueError, func, ...) không qua with
            exc_type = _is_pytest_raises_call(sub)
            if exc_type is not None and len(sub.args) >= 2:
                src = _unparse_safe(sub) or f"pytest.raises({exc_type}, ...)"
                assertions.append(
                    AssertionInfo(
                        kind="pytest_raises",
                        lineno=getattr(sub, "lineno", func_node.lineno),
                        source=src,
                        ast_dump=ast.dump(sub, include_attributes=False),
                        raised_exception=exc_type,
                        has_match=_has_nonempty_match_kwarg(sub),
                    )
                )

    assertions.sort(key=lambda a: a.lineno)
    return assertions


def _normalized_function_body_dump(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    """Sinh chuỗi AST dump chuẩn hóa của thân hàm và decorator (không phụ thuộc tên hàm và số dòng)."""
    dec_dumps = [ast.dump(d, include_attributes=False) for d in node.decorator_list]
    args_dump = ast.dump(node.args, include_attributes=False)
    body_dumps = [ast.dump(stmt, include_attributes=False) for stmt in node.body]
    return f"decorators={dec_dumps}|args={args_dump}|body={body_dumps}"


def analyze_test_code(
    test_code: str,
    file_path: Path | str = "<test_string>",
) -> TestFileAnalysis:
    """Phân tích AST của chuỗi mã nguồn file test pytest."""
    tree = ast.parse(test_code, filename=str(file_path))
    imports: list[str] = []
    test_functions: list[TestFunctionInfo] = []
    fixtures: list[HelperOrFixtureInfo] = []
    helpers: list[HelperOrFixtureInfo] = []
    module_level_skips: list[str] = []

    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            imp_str = _unparse_safe(node)
            if imp_str and imp_str not in imports:
                imports.append(imp_str)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            is_fixture = any(_is_fixture_decorator(d) for d in node.decorator_list)
            src_code = _unparse_safe(node) or ""
            if is_fixture:
                fixtures.append(
                    HelperOrFixtureInfo(
                        name=node.name,
                        lineno=node.lineno,
                        is_fixture=True,
                        is_class=False,
                        normalized_dump=_normalized_function_body_dump(node),
                        source_code=src_code,
                    )
                )
            elif node.name.startswith("test_"):
                skip_markers = _detect_skip_markers(node)
                decorators = [d for d in (_unparse_safe(dec) for dec in node.decorator_list) if d]
                test_functions.append(
                    TestFunctionInfo(
                        name=node.name,
                        lineno=node.lineno,
                        end_lineno=getattr(node, "end_lineno", None),
                        is_async=isinstance(node, ast.AsyncFunctionDef),
                        decorators=decorators,
                        has_skip_or_xfail=len(skip_markers) > 0,
                        skip_markers=skip_markers,
                        assertions=_extract_assertions_from_test(node),
                        normalized_body_dump=_normalized_function_body_dump(node),
                        source_code=src_code,
                    )
                )
            else:
                helpers.append(
                    HelperOrFixtureInfo(
                        name=node.name,
                        lineno=node.lineno,
                        is_fixture=False,
                        is_class=False,
                        normalized_dump=_normalized_function_body_dump(node),
                        source_code=src_code,
                    )
                )
        elif isinstance(node, ast.ClassDef):
            if node.name.startswith("Test"):
                # Hỗ trợ các method test_* bên trong class Test*
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name.startswith("test_"):
                        skip_markers = _detect_skip_markers(item)
                        decorators = [
                            d for d in (_unparse_safe(dec) for dec in item.decorator_list) if d
                        ]
                        qualified_name = f"{node.name}.{item.name}"
                        test_functions.append(
                            TestFunctionInfo(
                                name=qualified_name,
                                lineno=item.lineno,
                                end_lineno=getattr(item, "end_lineno", None),
                                is_async=isinstance(item, ast.AsyncFunctionDef),
                                decorators=decorators,
                                has_skip_or_xfail=len(skip_markers) > 0,
                                skip_markers=skip_markers,
                                assertions=_extract_assertions_from_test(item),
                                normalized_body_dump=_normalized_function_body_dump(item),
                                source_code=_unparse_safe(item) or "",
                            )
                        )
            else:
                helpers.append(
                    HelperOrFixtureInfo(
                        name=node.name,
                        lineno=node.lineno,
                        is_fixture=False,
                        is_class=True,
                        normalized_dump=ast.dump(node, include_attributes=False),
                        source_code=_unparse_safe(node) or "",
                    )
                )
        elif isinstance(node, ast.Assign):
            # Phát hiện pytestmark = pytest.mark.skip / xfail ở cấp module
            assign_src = _unparse_safe(node) or ""
            if "pytestmark" in assign_src and any(
                k in assign_src for k in ("skip", "xfail")
            ):
                module_level_skips.append(assign_src)
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            call_name = _unparse_safe(node.value.func) or ""
            if call_name in ("pytest.skip", "pytest.xfail", "pytest.importorskip"):
                module_level_skips.append(_unparse_safe(node.value) or call_name)

    return TestFileAnalysis(
        file_path=str(file_path),
        imports=imports,
        test_functions=test_functions,
        fixtures=fixtures,
        helpers=helpers,
        module_level_skips=module_level_skips,
    )


def analyze_test_file(file_path: Path | str) -> TestFileAnalysis:
    """Đọc và phân tích AST của một file test trên đĩa."""
    path = Path(file_path)
    source = path.read_text(encoding="utf-8")
    return analyze_test_code(source, file_path=path)
