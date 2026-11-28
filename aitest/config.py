"""Quản lý cấu hình hệ thống aitest theo SRS v4.0.0 §10.

Thứ tự ưu tiên cấu hình:
CLI flags > Biến môi trường AITEST_* (dùng __ cho cấp lồng nhau) / GEMINI_API_KEY > aitest.yaml > Defaults.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

SDK_DEFAULT_MODEL = "gemini-3.8-flash"
OutputFormat = Literal["json", "md", "html"]


def normalize_output_format(v: Any) -> OutputFormat:
    """Chuẩn hóa tên định dạng xuất báo cáo về 'json', 'md' hoặc 'html'."""
    raw = str(v if v is not None else "json").strip().lower()
    if raw == "markdown":
        return "md"
    if raw == "htm":
        return "html"
    if raw == "json":
        return "json"
    if raw == "md":
        return "md"
    if raw == "html":
        return "html"
    raise ValueError(
        f"Định dạng xuất không hợp lệ ('{v}'). Chỉ hỗ trợ: json, md, html."
    )


DEFAULT_AITEST_YAML_TEMPLATE = """\
provider: antigravity
model: null                        # null = dùng mặc định của Google Antigravity SDK (gemini-3.8-flash)
language: python
test_framework: pytest
timeout_sec: 120

coverage:
  enabled: true
  branch: false

mutation:
  enabled: true
  tool: auto                       # auto (Windows -> ast, Linux -> mutmut/ast) | ast | mutmut
  tool_version: ">=3,<4"           # áp dụng khi dùng backend mutmut trên POSIX
  timeout_sec: 300
  per_mutant_timeout_sec: 15.0
  min_valid_mutants: 1             # 0 mutant hợp lệ → mutation_score_pct = null

eval:
  rubric:
    correctness: 0.20
    coverage: 0.15
    mutation: 0.20
    readability: 0.10
    maintainability: 0.10
    isolation: 0.10
    robustness: 0.08
    determinism: 0.07
  threshold_pass: 8.0
  correctness_cap_offset: 2.0
  judge_model: null                # để null để dùng chung model, hoặc chỉ định model riêng cho bước đánh giá

fix:
  max_iters: 3
  allow_skip: false
  quality_guard:
    min_test_count_ratio: 1.0      # không được giảm số hàm test_*
    min_assert_count_ratio: 1.0    # không được giảm tổng số assert / pytest.raises
    assertion_ast_diff: true       # phân tích AST assertion trước/sau
    reject_on_assertion_weakening: true
    reject_on_mutation_drop: true

agent:
  max_iters: 3
  max_llm_calls: 12
  max_total_tokens: 400000
  max_runtime_sec: 900
  early_stop_patience: 2

llm:
  vertex: false                    # true để dùng Vertex AI (Standard ADC hoặc Express api_key)
  project: null                    # GCP project ID khi dùng Vertex Standard ADC
  location: null                   # GCP region (vd: us-central1) khi dùng Vertex Standard ADC
  timeout_sec: 300
  retry: 2

output:
  format: json                     # json | md | html
  log_dir: .aitest_logs
  backup_dir: .aitest_backup

paths:
  tests_dir: tests
"""


def _normalize_nullable_str(v: Any) -> str | None:
    if v is None:
        return None
    if isinstance(v, str):
        stripped = v.strip()
        if not stripped or stripped.lower() in ("null", "none"):
            return None
        return stripped
    return str(v)


class CoverageConfig(BaseModel):
    enabled: bool = True
    branch: bool = False


class MutationConfig(BaseModel):
    enabled: bool = True
    tool: Literal["auto", "ast", "mutmut"] = "auto"
    tool_version: str = ">=3,<4"
    timeout_sec: int = Field(default=300, ge=1)
    per_mutant_timeout_sec: float = Field(default=15.0, gt=0.0)
    min_valid_mutants: int = Field(default=1, ge=1)


class RubricWeights(BaseModel):
    correctness: float = Field(default=0.20, ge=0.0, le=1.0)
    coverage: float = Field(default=0.15, ge=0.0, le=1.0)
    mutation: float = Field(default=0.20, ge=0.0, le=1.0)
    readability: float = Field(default=0.10, ge=0.0, le=1.0)
    maintainability: float = Field(default=0.10, ge=0.0, le=1.0)
    isolation: float = Field(default=0.10, ge=0.0, le=1.0)
    robustness: float = Field(default=0.08, ge=0.0, le=1.0)
    determinism: float = Field(default=0.07, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_weights_sum(self) -> RubricWeights:
        weights_dict = self.to_dict()
        total = sum(weights_dict.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Tổng trọng số rubric phải bằng 1.0 (sai số tối đa 1e-6), nhận được: {total:.6f}"
            )
        return self

    def to_dict(self) -> dict[str, float]:
        return {
            "correctness": self.correctness,
            "coverage": self.coverage,
            "mutation": self.mutation,
            "readability": self.readability,
            "maintainability": self.maintainability,
            "isolation": self.isolation,
            "robustness": self.robustness,
            "determinism": self.determinism,
        }


class EvalConfig(BaseModel):
    rubric: RubricWeights = Field(default_factory=RubricWeights)
    threshold_pass: float = Field(default=8.0, ge=0.0, le=10.0)
    correctness_cap_offset: float = Field(default=2.0, ge=0.0, le=10.0)
    judge_model: str | None = None

    @field_validator("judge_model", mode="before")
    @classmethod
    def _clean_judge_model(cls, v: Any) -> str | None:
        return _normalize_nullable_str(v)


class QualityGuardConfig(BaseModel):
    min_test_count_ratio: float = Field(default=1.0, ge=0.0)
    min_assert_count_ratio: float = Field(default=1.0, ge=0.0)
    assertion_ast_diff: bool = True
    reject_on_assertion_weakening: bool = True
    reject_on_mutation_drop: bool = True


class FixConfig(BaseModel):
    max_iters: int = Field(default=3, ge=1)
    allow_skip: bool = False
    quality_guard: QualityGuardConfig = Field(default_factory=QualityGuardConfig)


class AgentConfig(BaseModel):
    max_iters: int = Field(default=3, ge=1)
    max_llm_calls: int = Field(default=12, ge=1)
    max_total_tokens: int = Field(default=400000, ge=1)
    max_runtime_sec: int = Field(default=900, ge=1)
    early_stop_patience: int = Field(default=2, ge=1)


class LLMConfig(BaseModel):
    api_key: str | None = Field(default=None, repr=False)
    vertex: bool = False
    project: str | None = None
    location: str | None = None
    timeout_sec: int = Field(default=300, ge=1)
    retry: int = Field(default=2, ge=0)

    @field_validator("api_key", "project", "location", mode="before")
    @classmethod
    def _clean_nullable(cls, v: Any) -> str | None:
        return _normalize_nullable_str(v)


class OutputConfig(BaseModel):
    format: OutputFormat = "json"
    log_dir: str = ".aitest_logs"
    backup_dir: str = ".aitest_backup"

    @field_validator("format", mode="before")
    @classmethod
    def _clean_format(cls, v: Any) -> OutputFormat:
        return normalize_output_format(v)


class PathsConfig(BaseModel):
    tests_dir: str = "tests"


class AitestConfig(BaseModel):
    provider: Literal["antigravity"] = "antigravity"
    model: str | None = None
    language: Literal["python"] = "python"
    test_framework: Literal["pytest"] = "pytest"
    timeout_sec: int = Field(default=120, ge=1, description="Timeout mặc định cho pytest (FR-03)")

    coverage: CoverageConfig = Field(default_factory=CoverageConfig)
    mutation: MutationConfig = Field(default_factory=MutationConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)
    fix: FixConfig = Field(default_factory=FixConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)

    @field_validator("model", mode="before")
    @classmethod
    def _validate_model(cls, v: Any) -> str | None:
        return _normalize_nullable_str(v)

    @property
    def effective_model_label(self) -> str:
        """Tên model hiển thị và truy vết (trả về mặc định SDK `gemini-3.8-flash` nếu `model` là None)."""
        return self.model or SDK_DEFAULT_MODEL

    def safe_dump(self) -> dict[str, Any]:
        """Trả về dict cấu hình chuẩn hóa phục vụ tính config_hash (loại bỏ secret `api_key`, NFR-03)."""
        dumped = self.model_dump()
        llm_section = dumped.get("llm")
        if isinstance(llm_section, dict):
            llm_section.pop("api_key", None)
        return dumped


def _parse_dotenv_file(dotenv_path: Path) -> dict[str, str]:
    """Đọc file .env đơn giản mà không ghi đè biến môi trường hệ thống có sẵn."""
    result: dict[str, str] = {}
    if not dotenv_path.is_file():
        return result
    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip()
        # Bỏ inline comment nếu không nằm trong dấu nháy
        if val and val[0] in ("'", '"') and val[-1] == val[0]:
            val = val[1:-1]
        elif "#" in val:
            val = val.split("#", 1)[0].strip()
        if key:
            result[key] = val
    return result


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Gộp đệ quy hai dictionary, ưu tiên giá trị từ override (bỏ qua giá trị None)."""
    merged = copy.deepcopy(base)
    for k, v in override.items():
        if v is None:
            continue
        if isinstance(v, dict) and isinstance(merged.get(k), dict):
            merged[k] = _deep_merge(merged[k], v)
        else:
            merged[k] = v
    return merged


def _set_nested_key(target: dict[str, Any], keys: list[str], value: Any) -> None:
    """Gán giá trị vào dict lồng nhau theo đường dẫn danh sách key."""
    curr = target
    for key in keys[:-1]:
        if key not in curr or not isinstance(curr[key], dict):
            curr[key] = {}
        curr = curr[key]
    curr[keys[-1]] = value


def _extract_env_overrides(env_map: dict[str, str]) -> dict[str, Any]:
    """Trích xuất cấu hình từ `AITEST_*` (dùng `__` cho nested key) và biến chuẩn của Google Antigravity SDK."""
    overrides: dict[str, Any] = {}

    prefix = "AITEST_"
    for raw_key, raw_val in env_map.items():
        if not raw_key.startswith(prefix):
            continue
        stripped = raw_key[len(prefix) :]
        if not stripped:
            continue
        parts = [p.lower() for p in stripped.split("__") if p]
        if not parts:
            continue
        val: Any = raw_val
        low = raw_val.strip().lower()
        if low in ("true", "false"):
            val = low == "true"
        elif low in ("null", "none", ""):
            val = ""
        _set_nested_key(overrides, parts, val)

    llm_ov = overrides.get("llm") if isinstance(overrides.get("llm"), dict) else {}

    if "api_key" not in llm_ov:
        api_key = (env_map.get("GEMINI_API_KEY") or env_map.get("GOOGLE_API_KEY") or "").strip()
        if api_key and api_key != "your_gemini_api_key_here":
            _set_nested_key(overrides, ["llm", "api_key"], api_key)

    if "project" not in llm_ov:
        proj = (env_map.get("GOOGLE_CLOUD_PROJECT") or "").strip()
        if proj:
            _set_nested_key(overrides, ["llm", "project"], proj)

    if "location" not in llm_ov:
        loc = (env_map.get("GOOGLE_CLOUD_LOCATION") or "").strip()
        if loc:
            _set_nested_key(overrides, ["llm", "location"], loc)

    return overrides


def load_config(
    config_path: Path | str | None = None,
    cli_overrides: dict[str, Any] | None = None,
    env: dict[str, str] | None = None,
    dotenv_path: Path | str | None = None,
) -> AitestConfig:
    """Tải và hợp nhất cấu hình theo đúng thứ tự ưu tiên SRS v4.0.0 §10.3:
    CLI flags > Biến môi trường AITEST_* / GEMINI_API_KEY > aitest.yaml > Defaults.
    """
    from aitest.utils import discover_project_root

    project_root = discover_project_root()
    yaml_data: dict[str, Any] = {}
    if config_path is not None:
        resolved_config_path = Path(config_path)
    else:
        cwd_yaml = Path("aitest.yaml")
        resolved_config_path = cwd_yaml if cwd_yaml.is_file() else (project_root / "aitest.yaml")

    if resolved_config_path.is_file():
        loaded = yaml.safe_load(resolved_config_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            yaml_data = loaded

    # Hợp nhất .env (tự dò từ CWD hoặc thư mục gốc dự án) và os.environ (os.environ ưu tiên cao hơn .env)
    dotenv_vars: dict[str, str] = {}
    if env is None:
        if dotenv_path is not None:
            dotenv_vars = _parse_dotenv_file(Path(dotenv_path))
        else:
            root_env = project_root / ".env"
            cwd_env = Path(".env")
            if root_env.is_file():
                dotenv_vars.update(_parse_dotenv_file(root_env))
            if cwd_env.is_file() and cwd_env.resolve() != root_env.resolve():
                dotenv_vars.update(_parse_dotenv_file(cwd_env))

    effective_env: dict[str, str] = {**dotenv_vars, **(env if env is not None else dict(os.environ))}

    env_overrides = _extract_env_overrides(effective_env)

    merged = _deep_merge(yaml_data, env_overrides)
    if cli_overrides:
        merged = _deep_merge(merged, cli_overrides)

    return AitestConfig.model_validate(merged)
