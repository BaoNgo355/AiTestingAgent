"""Antigravity SDK Client sử dụng chính thức gói Python `google-antigravity` (SRS v4.0.0 §9.1, §10.4, NFR-02, NFR-03, NFR-05, AC-14, AC-16, AC-18)."""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import re
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, Field, ValidationError

from aitest.config import SDK_DEFAULT_MODEL, AitestConfig
from aitest.schemas import LLMAssessment
from aitest.utils import generate_run_id, save_raw_log

_JSON_FENCE_RE = re.compile(
    r"```(?:json)?\s*\r?\n(.*?)```",
    re.DOTALL | re.IGNORECASE,
)


class AntigravityError(RuntimeError):
    """Ngoại lệ cơ sở cho mọi lỗi giao tiếp với Google Antigravity SDK."""


class AntigravitySDKUnavailableError(AntigravityError):
    """Lỗi chưa cài đặt hoặc không thể import gói `google-antigravity`."""


class AntigravityAuthError(AntigravityError):
    """Lỗi thiếu hoặc sai khóa xác thực (`GEMINI_API_KEY` hoặc cấu hình Vertex AI)."""


class AntigravityModelNotFoundError(AntigravityError):
    """Lỗi model ID không tồn tại hoặc tài khoản chưa được cấp quyền sử dụng."""


class AntigravityQuotaError(AntigravityError):
    """Lỗi vượt hạn mức hoặc giới hạn tốc độ (429 / QUOTA_EXHAUSTED / RESOURCE_EXHAUSTED)."""


class AntigravityServerError(AntigravityError):
    """Lỗi kết nối hoặc máy chủ tạm thời phía Google Antigravity SDK."""


class AntigravityTimeoutError(AntigravityError):
    """Lỗi quá thời gian chờ phản hồi từ Google Antigravity SDK."""


class AntigravitySchemaValidationError(AntigravityError):
    """Lỗi khi đầu ra JSON từ LLM không khớp schema Pydantic sau khi đã retry (AC-16)."""


class LLMResponse(BaseModel):
    """Kết quả trả về từ một lượt gọi Google Antigravity SDK."""

    content: str
    structured_data: dict[str, Any] | None = None
    model_requested: str
    model_actual: str | None = None
    conversation_id: str | None = None
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    thinking_tokens: int = Field(default=0, ge=0)
    cache_read_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)
    latency_ms: int = Field(default=0, ge=0)


@dataclass(frozen=True)
class SDKCallSpec:
    """Đặc tả bất biến cho một lần gọi `google.antigravity.Agent`."""

    system_prompt: str
    full_prompt: str
    model: str | None
    api_key: str | None
    vertex: bool
    project: str | None
    location: str | None
    timeout_sec: float
    response_schema_model: type[BaseModel] | None = None


@dataclass
class SDKTurnResult:
    """Kết quả thô trả về từ một lượt `Agent.chat(...)` trong SDK."""

    text: str
    structured_output: dict[str, Any] | None = None
    model_actual: str | None = None
    conversation_id: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    thinking_tokens: int = 0
    cache_read_tokens: int = 0
    total_tokens: int = 0


SDKRunner = Callable[[SDKCallSpec], SDKTurnResult]


def _strip_json_fences(raw_text: str) -> str:
    text = raw_text.strip()
    match = _JSON_FENCE_RE.search(text)
    if match:
        return match.group(1).strip()
    lines = text.splitlines()
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _build_combined_prompt(
    system_prompt: str,
    user_prompt: str,
    extra_messages: list[dict[str, str]] | None = None,
) -> str:
    parts: list[str] = [user_prompt.strip()]
    if extra_messages:
        parts.append("--- Previous Turn / Validation Repair Context ---")
        for msg in extra_messages:
            role = (msg.get("role") or "user").strip().upper()
            content = (msg.get("content") or "").strip()
            if content:
                parts.append(f"[{role}]\n{content}")
    return "\n\n".join(p for p in parts if p).strip()


def _import_antigravity_sdk() -> tuple[Any, Any, Any]:
    """Import các thành phần cốt lõi của gói `google-antigravity`."""
    try:
        from google.antigravity import Agent, LocalAgentConfig, types
        from google.antigravity.hooks import policy
    except ImportError as exc:
        raise AntigravitySDKUnavailableError(
            "Không thể import gói `google-antigravity`. Hãy cài đặt bằng: `pip install \"google-antigravity>=0.1.24\"`."
        ) from exc
    return Agent, LocalAgentConfig, (types, policy)


def validate_sdk_credentials(config: AitestConfig) -> None:
    """Kiểm tra sớm thông tin xác thực trước khi gọi `google-antigravity` SDK (AC-14)."""
    if config.llm.vertex:
        has_express_key = bool(config.llm.api_key and config.llm.api_key.strip())
        has_standard_adc = bool(config.llm.project and config.llm.location)
        if not has_express_key and not has_standard_adc:
            raise AntigravityAuthError(
                "Đã bật chế độ Vertex AI (`llm.vertex: true`) nhưng thiếu thông tin xác thực. "
                "Vui lòng đặt `GEMINI_API_KEY` (Express Mode) hoặc cấu hình `llm.project` + `llm.location` "
                "(Standard Mode sau khi chạy `gcloud auth application-default login`)."
            )
        return

    if not config.llm.api_key or not config.llm.api_key.strip():
        raise AntigravityAuthError(
            "Chưa tìm thấy khóa xác thực `GEMINI_API_KEY` cho Google Antigravity SDK. "
            "Hãy tạo API key tại https://aistudio.google.com/app/api-keys và thêm vào file `.env`: "
            "`GEMINI_API_KEY=your_api_key`."
        )


def build_sdk_agent_config(spec: SDKCallSpec, app_data_dir: str) -> Any:
    """Khởi tạo `LocalAgentConfig` cô lập an toàn (chặn toàn bộ built-in tools, NFR-02)."""
    _, LocalAgentConfig, (types, policy) = _import_antigravity_sdk()

    kwargs: dict[str, Any] = {
        "system_instructions": types.CustomSystemInstructions(text=spec.system_prompt),
        "capabilities": types.CapabilitiesConfig(
            enabled_tools=[],
            enable_subagents=False,
        ),
        "policies": [policy.deny_all()],
        "app_data_dir": str(Path(app_data_dir).resolve()),
        "retry_config": types.RetryConfig(
            api_retry=types.ModelAPIRetryConfig(max_retries=0),
            model_output_retry=types.ModelOutputRetryConfig(max_retries=1),
        ),
    }
    if spec.model:
        kwargs["model"] = spec.model
    if spec.api_key:
        kwargs["api_key"] = spec.api_key
    if spec.vertex:
        kwargs["vertex"] = True
        if spec.project:
            kwargs["project"] = spec.project
        if spec.location:
            kwargs["location"] = spec.location
    if spec.response_schema_model is not None:
        kwargs["response_schema"] = spec.response_schema_model

    return LocalAgentConfig(**kwargs)


def _classify_sdk_exception(exc: Exception) -> AntigravityError:
    """Ánh xạ ngoại lệ của `google-antigravity` SDK sang phân loại lỗi chuẩn của `aitest` (AC-14, NFR-05)."""
    if isinstance(exc, AntigravityError):
        return exc
    if isinstance(exc, (TimeoutError, asyncio.TimeoutError)):
        return AntigravityTimeoutError("Quá thời gian chờ phản hồi từ Google Antigravity SDK.")

    err_text = str(exc).strip() or exc.__class__.__name__
    low = err_text.lower()
    cls_name = exc.__class__.__name__

    if any(
        marker in low
        for marker in (
            "api_key_invalid",
            "invalid api key",
            "api key not valid",
            "permission_denied",
            "unauthenticated",
            "unauthorized",
            "401",
            "403",
            "default credentials were not found",
        )
    ):
        return AntigravityAuthError(
            "Xác thực Google Antigravity SDK thất bại (API key hoặc quyền Vertex không hợp lệ). "
            "Kiểm tra lại `GEMINI_API_KEY` (https://aistudio.google.com/app/api-keys) hoặc ADC."
        )

    if any(
        marker in low
        for marker in (
            "not found",
            "not recognized",
            "unknown model",
            "invalid model",
            "models/",
            "404",
        )
    ):
        return AntigravityModelNotFoundError(
            f"Model không tồn tại hoặc chưa được hỗ trợ trên cấu hình hiện tại: {err_text[:350]}"
        )

    if any(
        marker in low
        for marker in (
            "quota",
            "rate limit",
            "too many requests",
            "429",
            "resource_exhausted",
            "quota_exhausted",
        )
    ):
        return AntigravityQuotaError(
            f"Đã chạm hạn mức quota hoặc rate limit của Google Antigravity SDK: {err_text[:350]}"
        )

    if "timeout" in low or "timed out" in low:
        return AntigravityTimeoutError(f"Hết thời gian chờ Google Antigravity SDK: {err_text[:350]}")

    if "ValidationError" in cls_name:
        return AntigravityServerError(f"Lỗi cấu hình/validation Google Antigravity SDK: {err_text[:350]}")

    return AntigravityServerError(f"Lỗi từ Google Antigravity SDK ({cls_name}): {err_text[:350]}")


async def _run_sdk_turn_async(spec: SDKCallSpec) -> SDKTurnResult:
    """Thực thi đúng 1 lượt `Agent.chat(...)` trong `app_data_dir` tạm cô lập."""
    Agent, _, (types, _) = _import_antigravity_sdk()

    with tempfile.TemporaryDirectory(prefix="aitest_sdk_", ignore_cleanup_errors=True) as tmp_dir:
        agent_cfg = build_sdk_agent_config(spec, app_data_dir=tmp_dir)
        async with Agent(config=agent_cfg) as agent:
            response = await asyncio.wait_for(
                agent.chat(spec.full_prompt),
                timeout=spec.timeout_sec,
            )

            stop_reason = getattr(response, "stop_reason", None)
            if stop_reason is not None and stop_reason == types.StopReason.QUOTA_EXHAUSTED:
                raise AntigravityQuotaError(
                    "Google Antigravity SDK dừng do hết quota (StopReason.QUOTA_EXHAUSTED)."
                )

            text = (await response.text()) or ""
            structured: dict[str, Any] | None = None
            if spec.response_schema_model is not None:
                raw_struct = await response.structured_output()
                if isinstance(raw_struct, BaseModel):
                    structured = raw_struct.model_dump()
                elif isinstance(raw_struct, dict):
                    structured = raw_struct

            usage = getattr(response, "usage_metadata", None) or getattr(
                getattr(agent, "conversation", None),
                "total_usage",
                None,
            )
            prompt_tok = int(getattr(usage, "prompt_token_count", 0) or 0)
            completion_tok = int(getattr(usage, "candidates_token_count", 0) or 0)
            thinking_tok = int(getattr(usage, "thoughts_token_count", 0) or 0)
            cache_tok = int(getattr(usage, "cached_content_token_count", 0) or 0)
            total_tok = int(
                getattr(usage, "total_token_count", 0)
                or (prompt_tok + completion_tok + thinking_tok)
            )
            conv_id = getattr(agent, "conversation_id", None)
            model_actual = spec.model or SDK_DEFAULT_MODEL

            return SDKTurnResult(
                text=str(text),
                structured_output=structured,
                model_actual=model_actual,
                conversation_id=str(conv_id) if conv_id else None,
                prompt_tokens=prompt_tok,
                completion_tokens=completion_tok,
                thinking_tokens=thinking_tok,
                cache_read_tokens=cache_tok,
                total_tokens=total_tok,
            )


def _default_sdk_runner(spec: SDKCallSpec) -> SDKTurnResult:
    """Chạy hàm async của SDK an toàn ngay cả khi đang nằm bên trong một event loop có sẵn."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(_run_sdk_turn_async(spec))

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(lambda: asyncio.run(_run_sdk_turn_async(spec))).result()


class AntigravityClient:
    """Client giao tiếp với Google Antigravity thông qua `google-antigravity` Python SDK."""

    def __init__(
        self,
        config: AitestConfig,
        *,
        sdk_runner: SDKRunner | None = None,
        backoff_base_sec: float = 1.0,
    ) -> None:
        self.config = config
        self.backoff_base_sec = backoff_base_sec
        self._runner: SDKRunner = sdk_runner or _default_sdk_runner

        # Bộ đếm thống kê phiên làm việc (phục vụ ngân sách Agent và báo cáo tái lập AC-18)
        self.total_calls: int = 0
        self.total_prompt_tokens: int = 0
        self.total_completion_tokens: int = 0
        self.total_thinking_tokens: int = 0
        self.total_cache_read_tokens: int = 0
        self.total_tokens: int = 0
        self.total_latency_ms: int = 0
        self.last_model_actual: str | None = None
        self.last_conversation_id: str | None = None

    def close(self) -> None:
        """Giữ tương thích giao diện context manager."""

    def __enter__(self) -> AntigravityClient:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def chat(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        model_override: str | None = None,
        response_schema_model: type[BaseModel] | None = None,
        extra_messages: list[dict[str, str]] | None = None,
    ) -> LLMResponse:
        """Gửi yêu cầu qua `google.antigravity.Agent` (cô lập bằng `policy.deny_all()` và retry lỗi tạm thời)."""
        validate_sdk_credentials(self.config)

        resolved_model = (
            model_override.strip()
            if (model_override is not None and model_override.strip())
            else self.config.model
        )
        requested_label = resolved_model or SDK_DEFAULT_MODEL
        full_prompt = _build_combined_prompt(system_prompt, user_prompt, extra_messages)

        spec = SDKCallSpec(
            system_prompt=system_prompt,
            full_prompt=full_prompt,
            model=resolved_model,
            api_key=self.config.llm.api_key,
            vertex=self.config.llm.vertex,
            project=self.config.llm.project,
            location=self.config.llm.location,
            timeout_sec=float(self.config.llm.timeout_sec),
            response_schema_model=response_schema_model,
        )

        max_retries = self.config.llm.retry
        last_error: AntigravityError | None = None

        for attempt in range(max_retries + 1):
            start_ns = time.perf_counter_ns()
            try:
                turn_res = self._runner(spec)
                elapsed_ms = max(0, int((time.perf_counter_ns() - start_ns) / 1_000_000))

                model_actual = turn_res.model_actual or requested_label
                prompt_tok = max(0, int(turn_res.prompt_tokens))
                completion_tok = max(0, int(turn_res.completion_tokens))
                thinking_tok = max(0, int(turn_res.thinking_tokens))
                cache_tok = max(0, int(turn_res.cache_read_tokens))
                total_tok = max(
                    0,
                    int(turn_res.total_tokens or (prompt_tok + completion_tok + thinking_tok)),
                )

                self.total_calls += 1
                self.total_prompt_tokens += prompt_tok
                self.total_completion_tokens += completion_tok
                self.total_thinking_tokens += thinking_tok
                self.total_cache_read_tokens += cache_tok
                self.total_tokens += total_tok
                self.total_latency_ms += elapsed_ms
                self.last_model_actual = model_actual
                if turn_res.conversation_id:
                    self.last_conversation_id = turn_res.conversation_id

                return LLMResponse(
                    content=turn_res.text,
                    structured_data=turn_res.structured_output,
                    model_requested=requested_label,
                    model_actual=model_actual,
                    conversation_id=turn_res.conversation_id,
                    prompt_tokens=prompt_tok,
                    completion_tokens=completion_tok,
                    thinking_tokens=thinking_tok,
                    cache_read_tokens=cache_tok,
                    total_tokens=total_tok,
                    latency_ms=elapsed_ms,
                )
            except Exception as exc:
                classified = _classify_sdk_exception(exc)
                if isinstance(
                    classified,
                    (
                        AntigravitySDKUnavailableError,
                        AntigravityAuthError,
                        AntigravityModelNotFoundError,
                    ),
                ):
                    raise classified from exc
                last_error = classified

            if attempt < max_retries:
                sleep_sec = self.backoff_base_sec * (2**attempt)
                if sleep_sec > 0:
                    time.sleep(sleep_sec)

        assert last_error is not None
        raise last_error

    def generate_structured_assessment(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        run_id: str | None = None,
        log_dir: Path | str | None = None,
        model_override: str | None = None,
    ) -> tuple[LLMAssessment, LLMResponse]:
        """Gọi Antigravity SDK với `response_schema=LLMAssessment` và validate lại bằng Pydantic (FR-05, AC-16)."""
        effective_run_id = run_id or generate_run_id()
        effective_log_dir = Path(log_dir or self.config.output.log_dir)
        judge_model = (
            model_override.strip()
            if (model_override is not None and model_override.strip())
            else self.config.eval.judge_model
        )

        def _validate_resp(resp: LLMResponse) -> LLMAssessment:
            if resp.structured_data is not None:
                return LLMAssessment.model_validate(resp.structured_data)
            return LLMAssessment.model_validate_json(_strip_json_fences(resp.content))

        first_resp = self.chat(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model_override=judge_model,
            response_schema_model=LLMAssessment,
        )

        try:
            assessment = _validate_resp(first_resp)
            return assessment, first_resp
        except (ValidationError, ValueError) as first_err:
            raw_attempt1 = (
                json.dumps(first_resp.structured_data, ensure_ascii=False)
                if first_resp.structured_data is not None
                else first_resp.content
            )
            save_raw_log(
                effective_log_dir,
                effective_run_id,
                "eval_schema_invalid_attempt1",
                raw_attempt1,
            )
            repair_messages = [
                {"role": "assistant", "content": raw_attempt1},
                {
                    "role": "user",
                    "content": (
                        "The JSON you returned failed Pydantic schema validation with the following error:\n"
                        f"{first_err}\n\n"
                        "Please return ONLY a valid JSON object matching the exact LLMAssessment schema "
                        "(fields: correctness, readability, maintainability, isolation, robustness, determinism "
                        "as numbers between 0.0 and 10.0, issues as a list of {title, description, suggestion}, "
                        "and summary as a non-empty string)."
                    ),
                },
            ]
            second_resp = self.chat(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model_override=judge_model,
                response_schema_model=LLMAssessment,
                extra_messages=repair_messages,
            )
            try:
                assessment = _validate_resp(second_resp)
                return assessment, second_resp
            except (ValidationError, ValueError) as second_err:
                raw_attempt2 = (
                    json.dumps(second_resp.structured_data, ensure_ascii=False)
                    if second_resp.structured_data is not None
                    else second_resp.content
                )
                log_file = save_raw_log(
                    effective_log_dir,
                    effective_run_id,
                    "eval_schema_invalid_attempt2",
                    raw_attempt2,
                )
                raise AntigravitySchemaValidationError(
                    f"Đầu ra JSON của Antigravity SDK không hợp lệ theo schema LLMAssessment sau khi đã retry. "
                    f"Đã lưu raw response tại '{log_file}'. Chi tiết lỗi: {second_err}"
                ) from second_err
