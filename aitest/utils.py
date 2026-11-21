"""Các hàm tiện ích cốt lõi của aitest: backup, atomic write, code sanitizer, hash, UUID, timestamp."""

from __future__ import annotations

import ast
import datetime
import hashlib
import json
import os
import re
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any

from pydantic import BaseModel

SINGLE_PY_FILE_ERROR_MSG = "MVP chỉ hỗ trợ một file Python (.py) đơn lẻ"

_FENCED_BLOCK_RE = re.compile(
    r"```(?:python|py)?\s*\r?\n(.*?)```",
    re.DOTALL | re.IGNORECASE,
)


def generate_run_id() -> str:
    """Sinh UUID4 định danh duy nhất cho mỗi phiên chạy."""
    return str(uuid.uuid4())


def utc_now_iso() -> str:
    """Trả về chuỗi thời gian UTC theo chuẩn ISO 8601 (ví dụ: 2026-11-21T04:45:00Z)."""
    return (
        datetime.datetime.now(datetime.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def compute_config_hash(config: BaseModel | dict[str, Any]) -> str:
    """Tính mã băm SHA-256 chuẩn hóa của cấu hình (tuyệt đối loại bỏ khóa bí mật)."""
    if isinstance(config, BaseModel):
        if hasattr(config, "safe_dump"):
            data = config.safe_dump()
        else:
            data = config.model_dump(exclude={"ollama_api_key"})
    else:
        data = {k: v for k, v in config.items() if k != "ollama_api_key"}

    canonical = json.dumps(data, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def compute_file_hash(file_path: Path | str) -> str:
    """Tính mã băm SHA-256 của một file trên đĩa."""
    path = Path(file_path)
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return f"sha256:{h.hexdigest()}"


def sanitize_code_block(raw_text: str) -> str:
    """Làm sạch đầu ra từ LLM để lấy mã nguồn Python thuần túy (AC-04, NFR-08).

    - Nếu chuỗi chứa các khối markdown fence ```python ... ``` hoặc ``` ... ```,
      trích xuất nội dung bên trong.
    - Nếu chỉ có dòng mở/đóng ``` ở đầu hoặc cuối chuỗi, cắt bỏ sạch.
    """
    if not raw_text:
        return ""

    text = raw_text.strip()

    # Trường hợp 1: Có các cặp ```...``` hoàn chỉnh
    matches = _FENCED_BLOCK_RE.findall(text)
    if matches:
        # Nếu chỉ có 1 khối hoặc nhiều khối code, ghép các khối code lại
        cleaned_blocks = [m.strip() for m in matches if m.strip()]
        if cleaned_blocks:
            return "\n\n".join(cleaned_blocks) + "\n"

    # Trường hợp 2: Có dòng ``` ở đầu hoặc cuối nhưng không khớp regex (ví dụ thiếu newline)
    lines = text.splitlines()
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]

    result = "\n".join(lines).strip()
    return (result + "\n") if result else ""


def atomic_write_text(
    target_path: Path | str,
    content: str,
    *,
    encoding: str = "utf-8",
    verify_python_ast: bool = False,
) -> Path:
    """Ghi nội dung vào file theo cơ chế Atomic Write (NFR-12).

    1. Nếu `verify_python_ast=True`, kiểm tra `ast.parse(content)` trước khi chạm vào đĩa.
    2. Ghi ra một file tạm nằm cùng thư mục với `target_path`.
    3. Flush + fsync dữ liệu xuống đĩa.
    4. Dùng `os.replace()` để thay thế nguyên tử.
    """
    if verify_python_ast:
        ast.parse(content, filename=str(target_path))

    dest = Path(target_path)
    dest.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{dest.name}.",
        suffix=".tmp",
        dir=str(dest.parent),
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding=encoding, newline="\n") as tmp_file:
            tmp_file.write(content)
            tmp_file.flush()
            os.fsync(tmp_file.fileno())
        os.replace(str(tmp_path), str(dest))
    except Exception:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        raise
    return dest


def backup_file(
    file_path: Path | str,
    backup_root: Path | str = ".aitest_backup",
    run_id: str | None = None,
    subdir: str | None = None,
) -> Path | None:
    """Sao lưu file hiện có vào `.aitest_backup/<run_id>/[subdir/]<filename>.bak`.

    Trả về đường dẫn file backup, hoặc `None` nếu `file_path` chưa tồn tại.
    """
    src = Path(file_path)
    if not src.is_file():
        return None

    effective_run_id = run_id or generate_run_id()
    dest_dir = Path(backup_root) / effective_run_id
    if subdir:
        dest_dir = dest_dir / subdir
    dest_dir.mkdir(parents=True, exist_ok=True)

    backup_path = dest_dir / f"{src.name}.bak"
    shutil.copy2(str(src), str(backup_path))
    return backup_path


def restore_backup(backup_path: Path | str, target_path: Path | str) -> Path:
    """Khôi phục file từ bản backup bằng cơ chế Atomic Write."""
    b_path = Path(backup_path)
    if not b_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy file backup: {b_path}")
    content = b_path.read_text(encoding="utf-8")
    return atomic_write_text(target_path, content, encoding="utf-8")


def save_raw_log(
    log_dir: Path | str,
    run_id: str,
    label: str,
    content: str,
) -> Path:
    """Lưu raw response từ LLM hoặc thông tin lỗi vào `.aitest_logs/` (NFR-10)."""
    base = Path(log_dir)
    base.mkdir(parents=True, exist_ok=True)
    safe_label = re.sub(r"[^a-zA-Z0-9_.-]+", "_", label).strip("_") or "log"
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    file_path = base / f"{timestamp}_{run_id[:8]}_{safe_label}.log"
    atomic_write_text(file_path, content)
    return file_path


def discover_project_root(start_path: Path | str | None = None) -> Path:
    """Tìm thư mục gốc của dự án (nơi chứa `aitest.yaml`, `pyproject.toml` hoặc `.git`).

    Giúp người dùng dù đứng trong thư mục con (như `examples/`) vẫn tự động dùng chung
    `tests/`, `.aitest_logs/`, `.aitest_backup/` ở thư mục gốc dự án.
    """
    curr = Path(start_path).resolve() if start_path else Path.cwd().resolve()
    if curr.is_file():
        curr = curr.parent

    for directory in (curr, *curr.parents):
        if (
            (directory / "aitest.yaml").is_file()
            or (directory / "pyproject.toml").is_file()
            or (directory / ".git").exists()
        ):
            return directory
    return curr


def ensure_pytest_imported(code: str) -> str:
    """Đảm bảo `import pytest` luôn có mặt ở đầu file nếu mã test có sử dụng biến `pytest`."""
    if not code.strip():
        return code
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code

    has_pytest_import = False
    for node in tree.body:
        if isinstance(node, ast.Import):
            if any(alias.name == "pytest" and alias.asname is None for alias in node.names):
                has_pytest_import = True
                break
        elif isinstance(node, ast.ImportFrom) and node.module == "pytest":
            has_pytest_import = True
            break

    if has_pytest_import:
        return code

    uses_pytest_name = any(
        isinstance(sub, ast.Name) and sub.id == "pytest" for sub in ast.walk(tree)
    )
    if uses_pytest_name:
        return f"import pytest\n{code}"
    return code


def validate_single_python_file(target_path: Path | str) -> Path:
    """Kiểm tra ràng buộc đầu vào: chỉ một file .py tồn tại và parse được bằng ast.parse (FR-02, AC-02)."""
    raw_str = str(target_path).strip()
    if any(ch in raw_str for ch in ("*", "?", "[")):
        raise ValueError(SINGLE_PY_FILE_ERROR_MSG)

    path = Path(raw_str)
    if not path.exists() and not path.is_absolute():
        # Nếu người dùng đứng ở thư mục con (ví dụ examples/) nhưng gõ `examples/calculator.py`
        proj_root = discover_project_root()
        candidate = proj_root / path
        if candidate.exists():
            path = candidate

    if path.exists() and path.is_dir():
        raise ValueError(SINGLE_PY_FILE_ERROR_MSG)
    if path.suffix.lower() != ".py":
        raise ValueError(SINGLE_PY_FILE_ERROR_MSG)
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy file mã nguồn: {path}")

    source = path.read_text(encoding="utf-8")
    try:
        ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        raise SyntaxError(f"File mã nguồn '{path}' có lỗi cú pháp Python: {exc}") from exc

    return path


def resolve_test_file_path(target_file: Path | str, tests_dir: Path | str = "tests") -> Path:
    """Trả về đường dẫn file test tương ứng: `<tests_dir>/test_<stem>.py`."""
    target = Path(target_file)
    return Path(tests_dir) / f"test_{target.stem}.py"
