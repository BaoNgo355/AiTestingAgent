"""Sinh Unit Test mới và thuật toán Safe Append (SRS v2.3.0 §7 FR-02, §8.4, §12.1, AC-02, AC-03, AC-04, AC-17)."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from aitest.analyzer import (
     _normalized_function_body_dump,
    analyze_source_file,
    analyze_test_code,
)
from aitest.config import AitestConfig
from aitest.context import build_gen_prompts
from aitest.llm.client import OllamaClient
from aitest.utils import (
    atomic_write_text,
    backup_file,
    generate_run_id,
    resolve_test_file_path,
    sanitize_code_block,
    save_raw_log,
    utc_now_iso,
    validate_single_python_file,
)


class SafeAppendError(RuntimeError):
    """Ngoại lệ cơ sở cho các lỗi trong quá trình Safe Append."""


class TestGenerationSyntaxError(SafeAppendError):
    """Lỗi khi đoạn mã test do LLM sinh ra không parse được bằng ast.parse (AC-17)."""

    __test__ = False


class SafeAppendConflictError(SafeAppendError):
    """Lỗi khi đoạn mã test mới định nghĩa lại fixture hoặc helper xung đột với file test cũ (AC-03)."""


class SafeAppendResult(BaseModel):
    """Kết quả thực hiện Safe Append hoặc ghi file test."""

    test_file: str
    mode: Literal["append", "overwrite", "created"]
    backup_path: str | None = None
    added_tests: list[str] = Field(default_factory=list)
    skipped_tests: list[str] = Field(default_factory=list)
    renamed_tests: list[tuple[str, str]] = Field(default_factory=list)
    added_imports: list[str] = Field(default_factory=list)


def _rename_function_node(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    new_name: str,
) -> str:
    """Đổi tên một node hàm AST và unparse lại thành mã nguồn Python."""
    node.name = new_name
    return ast.unparse(node).strip()


def safe_merge_and_append(
    *,
    test_file: Path | str,
    new_code: str,
    mode: Literal["append", "overwrite"] = "append",
    backup_dir: Path | str = ".aitest_backup",
    log_dir: Path | str = ".aitest_logs",
    run_id: str | None = None,
    timestamp_iso: str | None = None,
) -> SafeAppendResult:
    """Ghép đoạn mã test mới vào file test hiện có theo đúng quy trình Safe Append (FR-02 & §8.4, AC-03, AC-17).

    Thứ tự:
    1. Sanitize & Parse AST đoạn code mới (`new_code`). Nếu lỗi cú pháp -> lưu raw log vào `.aitest_logs/`,
       không chạm vào file cũ và ném `TestGenerationSyntaxError`.
    2. Nếu file test chưa tồn tại hoặc `mode == 'overwrite'`:
       - Backup file cũ (nếu tồn tại).
       - Kiểm tra có ít nhất 1 hàm `test_*`.
       - Atomic write kèm kiểm tra `ast.parse()`.
    3. Nếu file test đã tồn tại và `mode == 'append'`:
       - Backup file cũ vào `.aitest_backup/<run_id>/`.
       - Parse AST file test cũ (`old_tree`) và đoạn mới (`new_tree`).
       - Kiểm tra xung đột fixture (`@pytest.fixture`) và helper: nếu trùng tên nhưng khác chữ ký/nội dung AST ->
         ném `SafeAppendConflictError`, giữ nguyên file cũ.
       - Lọc các câu lệnh import mới chưa có trong file cũ.
       - Với mỗi hàm `test_*` mới:
         * Chưa tồn tại tên -> thêm mới.
         * Trùng tên + cùng cấu trúc AST thân hàm (`normalized_body_dump`) -> bỏ qua (skip).
         * Trùng tên + khác nội dung -> tự động đổi tên thành `test_x__2` (hoặc `__3`...).
       - Ghép sau dòng phân cách `# --- aitest: appended <ISO8601> ---`.
       - Parse AST toàn bộ file kết quả lần cuối và ghi bằng Atomic Write (`os.replace`).
    """
    effective_run_id = run_id or generate_run_id()
    dest_path = Path(test_file)
    cleaned_new = sanitize_code_block(new_code)

    if not cleaned_new.strip():
        save_raw_log(log_dir, effective_run_id, "empty_test_code", new_code)
        raise TestGenerationSyntaxError("LLM trả về mã test rỗng sau khi sanitize")

    try:
        new_tree = ast.parse(cleaned_new, filename=str(dest_path))
        new_analysis = analyze_test_code(cleaned_new, file_path=dest_path)
    except SyntaxError as exc:
        log_file = save_raw_log(log_dir, effective_run_id, "syntax_error_gen", new_code)
        raise TestGenerationSyntaxError(
            f"Mã test mới có lỗi cú pháp Python (đã lưu raw log tại '{log_file}'): {exc}"
        ) from exc

    # Trường hợp 1: File test chưa tồn tại hoặc chế độ overwrite
    if not dest_path.is_file() or mode == "overwrite":
        existed = dest_path.is_file()
        bak = (
            backup_file(dest_path, backup_root=backup_dir, run_id=effective_run_id)
            if existed
            else None
        )
        atomic_write_text(dest_path, cleaned_new, verify_python_ast=True)
        return SafeAppendResult(
            test_file=str(dest_path),
            mode="overwrite" if existed else "created",
            backup_path=str(bak) if bak else None,
            added_tests=new_analysis.test_names,
            skipped_tests=[],
            renamed_tests=[],
            added_imports=new_analysis.imports,
        )

    # Trường hợp 2: File test đã tồn tại và mode == "append"
    old_code = dest_path.read_text(encoding="utf-8")
    old_analysis = analyze_test_code(old_code, file_path=dest_path)

    # 2a. Kiểm tra xung đột fixture và helper trước khi ghi bất kỳ thứ gì
    old_fixtures = {f.name: f for f in old_analysis.fixtures}
    old_helpers = {h.name: h for h in old_analysis.helpers}

    for nf in new_analysis.fixtures:
        if nf.name in old_helpers:
            raise SafeAppendConflictError(
                f"Xung đột định nghĩa: '{nf.name}' là helper trong file cũ nhưng được định nghĩa lại là @pytest.fixture trong đoạn mới"
            )
        if nf.name in old_fixtures and nf.normalized_dump != old_fixtures[nf.name].normalized_dump:
            raise SafeAppendConflictError(
                f"Xung đột fixture: '@pytest.fixture {nf.name}' đã tồn tại trong file test với nội dung/chữ ký khác"
            )

    for nh in new_analysis.helpers:
        if nh.name in old_fixtures:
            raise SafeAppendConflictError(
                f"Xung đột định nghĩa: '{nh.name}' là @pytest.fixture trong file cũ nhưng được định nghĩa lại là helper trong đoạn mới"
            )
        if nh.name in old_helpers and (
            nh.is_class != old_helpers[nh.name].is_class
            or nh.normalized_dump != old_helpers[nh.name].normalized_dump
        ):
            raise SafeAppendConflictError(
                f"Xung đột helper: '{nh.name}' đã tồn tại trong file test với nội dung/chữ ký khác"
            )

    # 2b. Backup file test cũ
    bak = backup_file(dest_path, backup_root=backup_dir, run_id=effective_run_id)

    # 2c. Lọc imports mới, fixtures/helpers mới và xử lý dedup/rename hàm test_*
    old_imports_set = set(old_analysis.imports)
    added_imports: list[str] = [
        imp for imp in new_analysis.imports if imp not in old_imports_set
    ]

    new_fixture_or_helper_blocks: list[str] = []
    for nf in new_analysis.fixtures:
        if nf.name not in old_fixtures:
            new_fixture_or_helper_blocks.append(nf.source_code)
    for nh in new_analysis.helpers:
        if nh.name not in old_helpers:
            new_fixture_or_helper_blocks.append(nh.source_code)

    old_tests_map = {t.name: t for t in old_analysis.test_functions}
    reserved_names = set(old_tests_map.keys()) | set(old_fixtures.keys()) | set(old_helpers.keys())

    added_tests: list[str] = []
    skipped_tests: list[str] = []
    renamed_tests: list[tuple[str, str]] = []
    new_test_blocks: list[str] = []

    for node in new_tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_"):
            orig_name = node.name
            body_dump = _normalized_function_body_dump(node)

            if orig_name not in reserved_names:
                reserved_names.add(orig_name)
                added_tests.append(orig_name)
                new_test_blocks.append(ast.unparse(node).strip())
            else:
                # Nếu trùng tên với hàm test cũ và thân hàm giống hệt nhau -> skip
                if orig_name in old_tests_map and old_tests_map[orig_name].normalized_body_dump == body_dump:
                    skipped_tests.append(orig_name)
                else:
                    # Trùng tên nhưng khác nội dung -> đổi tên thành test_x__2, test_x__3...
                    suffix = 2
                    while f"{orig_name}__{suffix}" in reserved_names:
                        suffix += 1
                    renamed = f"{orig_name}__{suffix}"
                    reserved_names.add(renamed)
                    renamed_tests.append((orig_name, renamed))
                    added_tests.append(renamed)
                    new_test_blocks.append(_rename_function_node(node, renamed))

    # Nếu không có hàm test mới, fixture/helper mới hay import mới nào cần thêm -> giữ nguyên file cũ
    if not new_test_blocks and not new_fixture_or_helper_blocks:
        return SafeAppendResult(
            test_file=str(dest_path),
            mode="append",
            backup_path=str(bak) if bak else None,
            added_tests=[],
            skipped_tests=skipped_tests,
            renamed_tests=renamed_tests,
            added_imports=[],
        )

    iso_ts = timestamp_iso or utc_now_iso()
    separator = f"# --- aitest: appended {iso_ts} ---"

    appended_parts: list[str] = [separator]
    if added_imports:
        appended_parts.append("\n".join(added_imports))
    if new_fixture_or_helper_blocks:
        appended_parts.append("\n\n".join(new_fixture_or_helper_blocks))
    if new_test_blocks:
        appended_parts.append("\n\n".join(new_test_blocks))

    base_text = old_code.rstrip()
    merged_code = base_text + "\n\n\n" + "\n\n\n".join(appended_parts) + "\n"

    # Kiểm tra ast.parse trên toàn bộ file kết quả và ghi nguyên tử
    atomic_write_text(dest_path, merged_code, verify_python_ast=True)

    return SafeAppendResult(
        test_file=str(dest_path),
        mode="append",
        backup_path=str(bak) if bak else None,
        added_tests=added_tests,
        skipped_tests=skipped_tests,
        renamed_tests=renamed_tests,
        added_imports=added_imports,
    )


def generate_tests(
    target_file: Path | str,
    config: AitestConfig,
    *,
    mode: Literal["append", "overwrite"] = "append",
    no_llm_dedup: bool = False,
    test_file: Path | str | None = None,
    llm_client: OllamaClient | None = None,
    run_id: str | None = None,
    project_root: Path | str | None = None,
) -> SafeAppendResult:
    """Thực thi quy trình `aitest gen <file.py>` (FR-02 & §12.1)."""
    validated_target = validate_single_python_file(target_file)
    root_dir = Path(project_root).resolve() if project_root else Path.cwd().resolve()
    resolved_test = (
        Path(test_file)
        if test_file is not None
        else resolve_test_file_path(validated_target, config.paths.tests_dir)
    )
    if not resolved_test.is_absolute():
        resolved_test = root_dir / resolved_test

    effective_run_id = run_id or generate_run_id()
    source_code = validated_target.read_text(encoding="utf-8")
    source_analysis = analyze_source_file(validated_target, project_root=root_dir)

    existing_test_names: list[str] = []
    if resolved_test.is_file() and mode == "append" and not no_llm_dedup:
        try:
            old_test_analysis = analyze_test_code(
                resolved_test.read_text(encoding="utf-8"),
                file_path=resolved_test,
            )
            existing_test_names = old_test_analysis.test_names
        except SyntaxError:
            existing_test_names = []

    sys_prompt, usr_prompt = build_gen_prompts(
        target_file=validated_target,
        source_code=source_code,
        source_analysis=source_analysis,
        existing_test_names=existing_test_names,
        project_root=root_dir,
    )

    owns_client = llm_client is None
    client = llm_client or OllamaClient(config)
    try:
        llm_resp = client.chat(system_prompt=sys_prompt, user_prompt=usr_prompt)
    finally:
        if owns_client:
            client.close()

    return safe_merge_and_append(
        test_file=resolved_test,
        new_code=llm_resp.content,
        mode=mode,
        backup_dir=root_dir / config.output.backup_dir,
        log_dir=root_dir / config.output.log_dir,
        run_id=effective_run_id,
    )
