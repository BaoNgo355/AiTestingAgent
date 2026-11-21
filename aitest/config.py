"""Quản lý cấu hình hệ thống aitest theo SRS v2.3.0 §10.

Thứ tự ưu tiên cấu hình:
CLI flags > Biến môi trường AITEST_* (dùng __ cho cấp lồng nhau) > aitest.yaml > Defaults.
Riêng OLLAMA_BASE_URL và OLLAMA_API_KEY cũng được đọc từ biến môi trường hoặc file .env.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, SecretStr, field_validator, model_validator

DEFAULT_AITEST_YAML_TEMPLATE = """\
provider: ollama
base_url: http://localhost:11434
model: qwen2.5-coder:7b
language: python
test_framework: pytest

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
  require_mutation_for_comparable_report: true

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
  max_total_tokens: 200000
  max_runtime_sec: 900
  early_stop_patience: 2

llm:
  temperature: 0.2
  max_tokens: 4096                 # map sang Ollama options.num_predict
  num_ctx: 8192                    # context window
  keep_alive: 5m
  timeout_sec: 300
  retry: 2

output:
  format: json
  log_dir: .aitest_logs
  backup_dir: .aitest_backup

paths:
  tests_dir: tests
"""


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
    require_mutation_for_comparable_report: bool = True


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
    max_total_tokens: int = Field(default=200000, ge=1)
    max_runtime_sec: int = Field(default=900, ge=1)
    early_stop_patience: int = Field(default=2, ge=1)


class LLMConfig(BaseModel):
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, ge=1)
    num_ctx: int = Field(default=8192, ge=512)
    keep_alive: str = "5m"
    timeout_sec: int = Field(default=300, ge=1)
    retry: int = Field(default=2, ge=0)


class OutputConfig(BaseModel):
    format: Literal["json"] = "json"
    log_dir: str = ".aitest_logs"
    backup_dir: str = ".aitest_backup"


class PathsConfig(BaseModel):
    tests_dir: str = "tests"


class AitestConfig(BaseModel):
    provider: Literal["ollama"] = "ollama"
    base_url: str = "http://localhost:11434"
    model: str = "qwen2.5-coder:7b"
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

    # Biến bảo mật đọc từ môi trường, tuyệt đối không xuất vào báo cáo/hash
    ollama_api_key: SecretStr | None = Field(default=None, exclude=True, repr=False)

    @field_validator("base_url")
    @classmethod
    def normalize_base_url(cls, v: str) -> str:
        v = v.strip().rstrip("/")
        if not v:
            raise ValueError("base_url không được để trống")
        return v

    def safe_dump(self) -> dict[str, Any]:
        """Trả về dict cấu hình an toàn (đã loại bỏ hoàn toàn ollama_api_key)."""
        return self.model_dump(exclude={"ollama_api_key"})


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


def _coerce_env_value(raw_val: str) -> Any:
    """Chuyển đổi kiểu cơ bản cho giá trị đọc từ biến môi trường."""
    lower = raw_val.lower().strip()
    if lower in ("true", "yes", "1"):
        return True
    if lower in ("false", "no", "0"):
        return False
    try:
        if "." in raw_val:
            return float(raw_val)
        return int(raw_val)
    except ValueError:
        return raw_val


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


def _extract_env_overrides(env_map: dict[str, str]) -> tuple[dict[str, Any], str | None]:
    """Trích xuất cấu hình từ AITEST_* (dùng __ cho nested key) và OLLAMA_*."""
    overrides: dict[str, Any] = {}
    api_key: str | None = None

    if "OLLAMA_BASE_URL" in env_map and env_map["OLLAMA_BASE_URL"].strip():
        overrides["base_url"] = env_map["OLLAMA_BASE_URL"].strip()

    if "OLLAMA_API_KEY" in env_map and env_map["OLLAMA_API_KEY"].strip():
        api_key = env_map["OLLAMA_API_KEY"].strip()

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
        # Không cho phép parse boolean cho những trường string hoặc số nếu cần,
        # nhưng Pydantic v2 sẽ ép kiểu chuẩn xác từ chuỗi hoặc số.
        # Tuy nhiên với boolean ("true"/"false") cần lưu ý "1"/"0" có thể là số.
        val: Any = raw_val
        if raw_val.lower() in ("true", "false"):
            val = raw_val.lower() == "true"
        _set_nested_key(overrides, parts, val)

    return overrides, api_key


def load_config(
    config_path: Path | str | None = None,
    cli_overrides: dict[str, Any] | None = None,
    env: dict[str, str] | None = None,
    dotenv_path: Path | str | None = None,
) -> AitestConfig:
    """Tải và hợp nhất cấu hình theo đúng thứ tự ưu tiên SRS §10.3:
    CLI flags > Biến môi trường AITEST_* / OLLAMA_* > aitest.yaml > Defaults.
    """
    yaml_data: dict[str, Any] = {}
    resolved_config_path = Path(config_path) if config_path else Path("aitest.yaml")
    if resolved_config_path.is_file():
        loaded = yaml.safe_load(resolved_config_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            yaml_data = loaded

    # Hợp nhất .env và os.environ (os.environ ưu tiên cao hơn .env)
    resolved_dotenv = Path(dotenv_path) if dotenv_path else Path(".env")
    dotenv_vars = _parse_dotenv_file(resolved_dotenv) if env is None else {}
    effective_env: dict[str, str] = {**dotenv_vars, **(env if env is not None else dict(os.environ))}

    env_overrides, api_key = _extract_env_overrides(effective_env)

    merged = _deep_merge(yaml_data, env_overrides)
    if cli_overrides:
        merged = _deep_merge(merged, cli_overrides)

    config = AitestConfig.model_validate(merged)
    if api_key:
        config.ollama_api_key = SecretStr(api_key)
    return config
