"""Ollama HTTP Client dựa trên httpx với cơ chế retry, structured output và thống kê token/latency (SRS v2.3.0 §9.1, §10.4, NFR-05, AC-14, AC-16, AC-18)."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel, Field, ValidationError

from aitest.config import AitestConfig
from aitest.schemas import LLMAssessment
from aitest.utils import generate_run_id, save_raw_log


class OllamaError(RuntimeError):
    """Ngoại lệ cơ sở cho mọi lỗi giao tiếp với Ollama."""


class OllamaConnectionError(OllamaError):
    """Lỗi không thể kết nối tới dịch vụ Ollama local/remote."""


class OllamaModelNotFoundError(OllamaError):
    """Lỗi model được yêu cầu chưa tồn tại trên Ollama server (HTTP 404)."""


class OllamaServerError(OllamaError):
    """Lỗi phía máy chủ Ollama (HTTP 5xx)."""


class OllamaTimeoutError(OllamaError):
    """Lỗi quá thời gian chờ phản hồi từ Ollama."""


class OllamaSchemaValidationError(OllamaError):
    """Lỗi khi đầu ra JSON từ LLM không khớp schema Pydantic sau khi đã retry (AC-16)."""


class LLMResponse(BaseModel):
    """Kết quả trả về từ một lần gọi Ollama `/api/chat`."""

    content: str
    model_requested: str
    model_actual: str | None = None
    model_digest: str | None = None
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)
    latency_ms: int = Field(default=0, ge=0)


class OllamaClient:
    """Client giao tiếp với Ollama qua REST API (`GET /api/tags`, `POST /api/chat`)."""

    def __init__(
        self,
        config: AitestConfig,
        *,
        transport: httpx.BaseTransport | None = None,
        backoff_base_sec: float = 1.0,
    ) -> None:
        self.config = config
        self.backoff_base_sec = backoff_base_sec
        self._transport = transport

        headers: dict[str, str] = {"Content-Type": "application/json"}
        if config.ollama_api_key is not None:
            secret_val = config.ollama_api_key.get_secret_value().strip()
            if secret_val:
                headers["Authorization"] = f"Bearer {secret_val}"

        self._client = httpx.Client(
            base_url=config.base_url,
            headers=headers,
            timeout=httpx.Timeout(float(config.llm.timeout_sec)),
            transport=transport,
        )

        # Bộ đếm thống kê phiên làm việc (phục vụ ngân sách Agent và báo cáo tái lập AC-18)
        self.total_calls: int = 0
        self.total_prompt_tokens: int = 0
        self.total_completion_tokens: int = 0
        self.total_tokens: int = 0
        self.total_latency_ms: int = 0
        self.last_model_actual: str | None = None
        self._cached_digest: str | None = None
        self._digest_fetched: bool = False

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> OllamaClient:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def list_models(self) -> list[dict[str, Any]]:
        """Gọi `GET /api/tags` để lấy danh sách các model hiện có trên Ollama."""
        try:
            resp = self._client.get("/api/tags")
            resp.raise_for_status()
            data = resp.json()
            models = data.get("models", [])
            return models if isinstance(models, list) else []
        except httpx.ConnectError as exc:
            raise OllamaConnectionError(
                f"Không thể kết nối tới Ollama tại {self.config.base_url}. "
                f"Vui lòng kiểm tra dịch vụ bằng lệnh `ollama serve`."
            ) from exc
        except httpx.TimeoutException as exc:
            raise OllamaTimeoutError(
                f"Quá thời gian chờ khi kết nối tới Ollama ({self.config.base_url}/api/tags)."
            ) from exc
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            if status >= 500:
                raise OllamaServerError(
                    f"Lỗi máy chủ Ollama (HTTP {status}) khi truy vấn /api/tags."
                ) from exc
            raise OllamaError(f"Lỗi HTTP {status} từ Ollama /api/tags.") from exc

    def get_model_digest(self, model_name: str | None = None, *, force_refresh: bool = False) -> str | None:
        """Tra cứu `model_digest` thực tế từ `/api/tags` (SRS §10.4).

        Nếu không lấy được hoặc không tìm thấy -> trả về `None`, tuyệt đối không tự suy đoán.
        """
        target = (model_name or self.config.model).strip()
        if self._digest_fetched and not force_refresh and target == self.config.model:
            return self._cached_digest

        try:
            models = self.list_models()
        except OllamaError:
            return None

        candidates = {target}
        if ":" not in target:
            candidates.add(f"{target}:latest")

        digest: str | None = None
        for item in models:
            name = str(item.get("name", "")).strip()
            model_field = str(item.get("model", "")).strip()
            if name in candidates or model_field in candidates:
                raw_digest = item.get("digest")
                if isinstance(raw_digest, str) and raw_digest.strip():
                    raw_digest = raw_digest.strip()
                    digest = (
                        raw_digest
                        if raw_digest.startswith("sha256:")
                        else f"sha256:{raw_digest}"
                    )
                    break

        if target == self.config.model:
            self._cached_digest = digest
            self._digest_fetched = True
        return digest

    def chat(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_format: dict[str, Any] | str | None = None,
        extra_messages: list[dict[str, str]] | None = None,
    ) -> LLMResponse:
        """Gửi yêu cầu `POST /api/chat` với cơ chế retry exponential backoff khi gặp lỗi mạng/5xx/timeout."""
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        if extra_messages:
            messages.extend(extra_messages)

        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "stream": False,
            "keep_alive": self.config.llm.keep_alive,
            "options": {
                "temperature": self.config.llm.temperature,
                "num_predict": self.config.llm.max_tokens,
                "num_ctx": self.config.llm.num_ctx,
            },
        }
        if response_format is not None:
            payload["format"] = response_format

        max_retries = self.config.llm.retry
        last_error: OllamaError | None = None

        for attempt in range(max_retries + 1):
            start_ns = time.perf_counter_ns()
            try:
                resp = self._client.post("/api/chat", json=payload)
                if resp.status_code == 404:
                    # 404 Model Not Found -> Không retry, báo lỗi ngay
                    detail = ""
                    try:
                        detail = resp.json().get("error", "")
                    except Exception:
                        detail = resp.text
                    raise OllamaModelNotFoundError(
                        f"Không tìm thấy model '{self.config.model}' trên Ollama ({self.config.base_url}). "
                        f"Chi tiết: {detail}. Gợi ý: chạy `ollama pull {self.config.model}`."
                    )

                if resp.status_code >= 500:
                    raise OllamaServerError(
                        f"Máy chủ Ollama trả về lỗi HTTP {resp.status_code}: {resp.text[:300]}"
                    )

                resp.raise_for_status()
                data = resp.json()

                elapsed_ms = max(0, int((time.perf_counter_ns() - start_ns) / 1_000_000))
                msg_obj = data.get("message", {}) if isinstance(data, dict) else {}
                content = str(msg_obj.get("content", "")) if isinstance(msg_obj, dict) else ""

                model_actual = data.get("model") if isinstance(data.get("model"), str) else None
                prompt_tokens = int(data.get("prompt_eval_count") or 0)
                completion_tokens = int(data.get("eval_count") or 0)
                total_tokens = prompt_tokens + completion_tokens

                digest = self.get_model_digest(model_actual or self.config.model)

                # Cập nhật thống kê
                self.total_calls += 1
                self.total_prompt_tokens += prompt_tokens
                self.total_completion_tokens += completion_tokens
                self.total_tokens += total_tokens
                self.total_latency_ms += elapsed_ms
                if model_actual:
                    self.last_model_actual = model_actual

                return LLMResponse(
                    content=content,
                    model_requested=self.config.model,
                    model_actual=model_actual,
                    model_digest=digest,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    latency_ms=elapsed_ms,
                )

            except OllamaModelNotFoundError:
                raise
            except httpx.ConnectError as exc:
                last_error = OllamaConnectionError(
                    f"Không thể kết nối tới Ollama tại {self.config.base_url}. "
                    f"Vui lòng chạy `ollama serve` và thử lại."
                )
            except httpx.TimeoutException as exc:
                last_error = OllamaTimeoutError(
                    f"Yêu cầu tới Ollama ({self.config.base_url}/api/chat) vượt quá thời gian chờ "
                    f"{self.config.llm.timeout_sec}s."
                )
            except OllamaServerError as exc:
                last_error = exc
            except httpx.HTTPStatusError as exc:
                raise OllamaError(
                    f"Lỗi HTTP {exc.response.status_code} khi gọi Ollama /api/chat: {exc.response.text[:300]}"
                ) from exc

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
    ) -> tuple[LLMAssessment, LLMResponse]:
        """Gọi Ollama với Structured Outputs (`format` = JSON schema của `LLMAssessment`) và validate bằng Pydantic (FR-05, AC-16).

        Nếu parse JSON hoặc validate Pydantic thất bại:
        - Lưu raw response vào `.aitest_logs/`
        - Retry tối đa 1 lần kèm thông báo lỗi validation để LLM sửa JSON.
        - Nếu sau retry vẫn thất bại -> ném `OllamaSchemaValidationError`, tuyệt đối không bịa điểm giả.
        """
        effective_run_id = run_id or generate_run_id()
        effective_log_dir = Path(log_dir or self.config.output.log_dir)
        json_schema = LLMAssessment.model_json_schema()

        first_resp = self.chat(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_format=json_schema,
        )

        try:
            assessment = LLMAssessment.model_validate_json(first_resp.content)
            return assessment, first_resp
        except ValidationError as first_err:
            save_raw_log(
                effective_log_dir,
                effective_run_id,
                "eval_schema_invalid_attempt1",
                first_resp.content,
            )
            repair_messages = [
                {"role": "assistant", "content": first_resp.content},
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
                response_format=json_schema,
                extra_messages=repair_messages,
            )
            try:
                assessment = LLMAssessment.model_validate_json(second_resp.content)
                return assessment, second_resp
            except ValidationError as second_err:
                log_file = save_raw_log(
                    effective_log_dir,
                    effective_run_id,
                    "eval_schema_invalid_attempt2",
                    second_resp.content,
                )
                raise OllamaSchemaValidationError(
                    f"Đầu ra JSON của LLM không hợp lệ theo schema LLMAssessment sau khi đã retry. "
                    f"Đã lưu raw response tại '{log_file}'. Chi tiết lỗi: {second_err}"
                ) from second_err
