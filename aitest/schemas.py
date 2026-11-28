"""Định nghĩa toàn bộ Pydantic v2 schemas cho aitest theo SRS v3.0.0 §8."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

RUBRIC_KEYS: tuple[str, ...] = (
    "correctness",
    "coverage",
    "mutation",
    "readability",
    "maintainability",
    "isolation",
    "robustness",
    "determinism",
)

MutationStatusType = Literal["available", "disabled", "failed", "no_valid_mutants"]


class ErrorCategory(str, Enum):
    """Phân loại lỗi thực thi pytest trước khi gọi LLM sửa test (FR-04)."""

    NONE = "NONE"
    TEST_FAILURE = "TEST_FAILURE"
    TEST_SYNTAX = "TEST_SYNTAX"
    ENVIRONMENT_ERROR = "ENVIRONMENT_ERROR"
    SOURCE_ERROR = "SOURCE_ERROR"
    TIMEOUT = "TIMEOUT"
    COLLECTION_ERROR = "COLLECTION_ERROR"
    UNKNOWN_ERROR = "UNKNOWN_ERROR"

    @property
    def is_fixable_by_llm(self) -> bool:
        """Chỉ TEST_FAILURE và TEST_SYNTAX mới được phép gọi LLM sửa test."""
        return self in (ErrorCategory.TEST_FAILURE, ErrorCategory.TEST_SYNTAX)


class SurvivedMutant(BaseModel):
    file: str
    line: int = Field(ge=1)
    mutation: str


class AssessmentIssue(BaseModel):
    title: str
    description: str
    suggestion: str


class Meta(BaseModel):
    tool: Literal["aitest"] = "aitest"
    version: str
    generated_at: str
    target_file: str
    test_file: str
    provider: Literal["antigravity"] = "antigravity"
    model_requested: str
    model_actual: str | None = None
    judge_model_requested: str | None = None
    judge_model_actual: str | None = None
    vertex: bool = False
    conversation_id: str | None = None
    config_hash: str
    run_id: str


class Execution(BaseModel):
    exit_code: int
    all_tests_passed: bool
    tests_passed: int = Field(ge=0)
    tests_failed: int = Field(ge=0)
    tests_skipped: int = Field(ge=0)
    pass_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    duration_ms: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_execution_invariants(self) -> Execution:
        total_executed = self.tests_passed + self.tests_failed
        if total_executed == 0:
            if self.pass_rate is not None:
                raise ValueError(
                    "Khi tests_passed + tests_failed == 0, pass_rate bắt buộc phải là None (null)"
                )
            if self.all_tests_passed:
                raise ValueError(
                    "Khi không có test nào chạy (tests_passed + tests_failed == 0), "
                    "all_tests_passed bắt buộc phải là False"
                )
        else:
            expected_rate = self.tests_passed / total_executed
            if self.pass_rate is None:
                self.pass_rate = round(expected_rate, 4)
            if self.all_tests_passed and (self.exit_code != 0 or self.tests_failed > 0 or self.tests_passed == 0):
                raise ValueError(
                    "all_tests_passed chỉ được là True khi exit_code == 0, tests_passed > 0 và tests_failed == 0"
                )
        return self


class ObjectiveMetrics(BaseModel):
    line_coverage_pct: float | None = Field(default=None, ge=0.0, le=100.0)
    branch_coverage_pct: float | None = Field(default=None, ge=0.0, le=100.0)
    mutation_score_pct: float | None = Field(default=None, ge=0.0, le=100.0)
    mutation_killed: int | None = Field(default=None, ge=0)
    mutation_survived: int | None = Field(default=None, ge=0)
    mutation_excluded: int | None = Field(default=None, ge=0)
    mutation_status: MutationStatusType
    mutation_available: bool
    mutation_unavailable_reason: str | None = None
    test_count: int = Field(ge=0)
    missing_lines: list[int] = Field(default_factory=list)
    survived_mutants: list[SurvivedMutant] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_mutation_invariants(self) -> ObjectiveMetrics:
        if self.mutation_available:
            if self.mutation_status != "available":
                raise ValueError(
                    "Khi mutation_available == True, mutation_status bắt buộc phải là 'available'"
                )
            if self.mutation_score_pct is None:
                raise ValueError(
                    "Khi mutation_available == True, mutation_score_pct không được là None"
                )
        else:
            if self.mutation_status == "available":
                raise ValueError(
                    "Khi mutation_available == False, mutation_status không được là 'available'"
                )
            if self.mutation_score_pct is not None:
                raise ValueError(
                    "Khi mutation_available == False, mutation_score_pct bắt buộc phải là None"
                )

        if self.mutation_status == "no_valid_mutants":
            valid_total = (self.mutation_killed or 0) + (self.mutation_survived or 0)
            if valid_total != 0:
                raise ValueError(
                    "Khi mutation_status == 'no_valid_mutants', tổng killed + survived phải bằng 0"
                )
            if self.mutation_score_pct is not None or self.mutation_available:
                raise ValueError(
                    "Khi mutation_status == 'no_valid_mutants', mutation_score_pct phải là None "
                    "và mutation_available phải là False"
                )
        return self


class LLMAssessment(BaseModel):
    correctness: float = Field(ge=0.0, le=10.0)
    readability: float = Field(ge=0.0, le=10.0)
    maintainability: float = Field(ge=0.0, le=10.0)
    isolation: float = Field(ge=0.0, le=10.0)
    robustness: float = Field(ge=0.0, le=10.0)
    determinism: float = Field(ge=0.0, le=10.0)
    issues: list[AssessmentIssue] = Field(default_factory=list)
    summary: str = Field(min_length=1)


class EvalReport(BaseModel):
    meta: Meta
    execution: Execution
    objective_metrics: ObjectiveMetrics
    llm_assessment: LLMAssessment
    scores: dict[str, float | None]
    weights: dict[str, float]
    weights_effective: dict[str, float]
    weighted_score: float = Field(ge=0.0, le=10.0)
    threshold_pass: float = Field(ge=0.0, le=10.0)
    passed_eval: bool
    report_comparable: bool
    non_comparable_reason: str | None = None
    correctness_cap_applied: bool

    @model_validator(mode="after")
    def validate_eval_report_invariants(self) -> EvalReport:
        # 1. Tập key của scores, weights và weights_effective phải khớp nhau
        weight_keys = set(self.weights.keys())
        score_keys = set(self.scores.keys())
        effective_keys = set(self.weights_effective.keys())
        if score_keys != weight_keys:
            raise ValueError(
                f"Tập key của scores ({sorted(score_keys)}) phải khớp chính xác với weights ({sorted(weight_keys)})"
            )
        if effective_keys != weight_keys:
            raise ValueError(
                f"Tập key của weights_effective ({sorted(effective_keys)}) phải khớp với weights ({sorted(weight_keys)})"
            )

        # 2. Tổng weights và weights_effective phải bằng 1.0 (sai số 1e-6)
        if any(w < 0.0 for w in self.weights.values()):
            raise ValueError("Trọng số trong weights không được âm")
        if any(w < 0.0 for w in self.weights_effective.values()):
            raise ValueError("Trọng số trong weights_effective không được âm")

        weights_sum = sum(self.weights.values())
        if abs(weights_sum - 1.0) > 1e-6:
            raise ValueError(f"Tổng weights phải bằng 1.0 (sai số 1e-6), nhận được: {weights_sum:.6f}")

        effective_sum = sum(self.weights_effective.values())
        if abs(effective_sum - 1.0) > 1e-6:
            raise ValueError(
                f"Tổng weights_effective phải bằng 1.0 (sai số 1e-6), nhận được: {effective_sum:.6f}"
            )

        # Các tiêu chí có score == None bắt buộc có weight_effective == 0.0
        for k, s in self.scores.items():
            if s is None and self.weights_effective.get(k, 0.0) > 1e-9:
                raise ValueError(f"Tiêu chí '{k}' có score=None nhưng weights_effective[{k}] != 0")
            if s is not None and not (0.0 <= s <= 10.0):
                raise ValueError(f"Điểm scores['{k}'] = {s} phải nằm trong đoạn [0.0, 10.0]")

        # 3. Kiểm tra quy tắc mutation & correctness cap
        if self.objective_metrics.mutation_available:
            if not self.correctness_cap_applied:
                raise ValueError(
                    "Khi mutation_available == True, correctness_cap_applied bắt buộc phải là True"
                )
        else:
            if self.correctness_cap_applied:
                raise ValueError(
                    "Khi mutation_available == False, correctness_cap_applied bắt buộc phải là False"
                )
            if self.scores.get("mutation") is not None:
                raise ValueError(
                    "Khi mutation_available == False, scores['mutation'] bắt buộc phải là None"
                )
            if self.report_comparable:
                raise ValueError(
                    "Khi mutation_available == False (bao gồm no_valid_mutants), "
                    "report_comparable bắt buộc phải là False"
                )

        # 4. Nếu bất kỳ chỉ số nào trong scores bị None -> report_comparable phải là False
        if any(v is None for v in self.scores.values()) and self.report_comparable:
            raise ValueError(
                "Khi có chỉ số trong scores là None (phải renormalize trọng số), "
                "report_comparable bắt buộc phải là False"
            )

        # 5. Khi report_comparable == False -> non_comparable_reason bắt buộc là chuỗi khác rỗng
        if not self.report_comparable:
            if not self.non_comparable_reason or not self.non_comparable_reason.strip():
                raise ValueError(
                    "Khi report_comparable == False, non_comparable_reason bắt buộc phải là chuỗi khác rỗng"
                )
        else:
            if self.non_comparable_reason is not None:
                raise ValueError(
                    "Khi report_comparable == True, non_comparable_reason phải là None"
                )

        # 6. Kiểm tra passed_eval nhất quán với all_tests_passed và threshold_pass
        expected_passed_eval = (
            self.execution.all_tests_passed and self.weighted_score >= self.threshold_pass
        )
        if self.passed_eval != expected_passed_eval:
            raise ValueError(
                f"passed_eval ({self.passed_eval}) không khớp với "
                f"(all_tests_passed={self.execution.all_tests_passed} and "
                f"weighted_score={self.weighted_score} >= threshold_pass={self.threshold_pass})"
            )

        return self


def clamp_score_10(pct: float | None) -> float | None:
    """Quy đổi phần trăm (0..100) sang thang điểm 0..10: clamp(pct / 10.0, 0.0, 10.0)."""
    if pct is None:
        return None
    return max(0.0, min(10.0, pct / 10.0))


def build_eval_report(
    *,
    meta: Meta,
    execution: Execution,
    objective_metrics: ObjectiveMetrics,
    llm_assessment: LLMAssessment,
    weights: dict[str, float],
    threshold_pass: float = 8.0,
    correctness_cap_offset: float = 2.0,
) -> EvalReport:
    """Tính toán toàn bộ trường có thẩm quyền từ chương trình và đóng gói EvalReport (FR-05 & §8.1).

    Đảm bảo:
    - coverage_score và mutation_score_10 tính từ chỉ số công cụ.
    - correctness bị chặn trên bởi min(correctness_llm, mutation_score_10 + offset) khi mutation khả dụng.
    - weights_effective chuẩn hóa lại trên các tiêu chí có điểm != None.
    - weighted_score tính từ điểm gốc CHƯA làm tròn, chỉ làm tròn 2 chữ số thập phân ở bước xuất cuối.
    """
    coverage_raw = clamp_score_10(objective_metrics.line_coverage_pct)
    mutation_raw = (
        clamp_score_10(objective_metrics.mutation_score_pct)
        if objective_metrics.mutation_available
        else None
    )

    if objective_metrics.mutation_available and mutation_raw is not None:
        correctness_raw = min(
            llm_assessment.correctness,
            mutation_raw + correctness_cap_offset,
        )
        correctness_cap_applied = True
    else:
        correctness_raw = llm_assessment.correctness
        correctness_cap_applied = False

    raw_scores: dict[str, float | None] = {
        "correctness": correctness_raw,
        "coverage": coverage_raw,
        "mutation": mutation_raw,
        "readability": llm_assessment.readability,
        "maintainability": llm_assessment.maintainability,
        "isolation": llm_assessment.isolation,
        "robustness": llm_assessment.robustness,
        "determinism": llm_assessment.determinism,
    }

    # Chuẩn hóa weights_effective trên các tiêu chí có điểm != None
    available_weight_sum = sum(
        w for k, w in weights.items() if raw_scores.get(k) is not None
    )
    if available_weight_sum <= 0.0:
        raise ValueError("Không có tiêu chí nào khả dụng để tính weights_effective")

    weights_effective_raw: dict[str, float] = {}
    for k, w in weights.items():
        if raw_scores.get(k) is None:
            weights_effective_raw[k] = 0.0
        else:
            weights_effective_raw[k] = w / available_weight_sum

    # Tính weighted_score từ điểm gốc CHƯA làm tròn và trọng số hiệu dụng gốc
    unrounded_weighted = sum(
        (raw_scores[k] or 0.0) * weights_effective_raw[k]
        for k in weights
        if raw_scores[k] is not None
    )
    weighted_score = round(unrounded_weighted, 2)

    # Xác định report_comparable và non_comparable_reason
    reasons: list[str] = []
    if not objective_metrics.mutation_available:
        detail = objective_metrics.mutation_unavailable_reason or (
            f"mutation_status={objective_metrics.mutation_status}"
        )
        reasons.append(f"Thiếu chỉ số mutation ({detail})")
    if coverage_raw is None:
        reasons.append("Thiếu chỉ số line coverage")

    report_comparable = len(reasons) == 0 and all(v is not None for v in raw_scores.values())
    non_comparable_reason = "; ".join(reasons) if not report_comparable else None
    if not report_comparable and not non_comparable_reason:
        non_comparable_reason = "Báo cáo thiếu một hoặc nhiều chỉ số khách quan"

    # Làm tròn điểm thành phần và weights_effective để xuất JSON sạch nhưng vẫn bảo toàn tổng = 1.0
    rounded_scores: dict[str, float | None] = {
        k: (round(v, 2) if v is not None else None) for k, v in raw_scores.items()
    }
    rounded_weights_effective: dict[str, float] = {
        k: round(v, 6) for k, v in weights_effective_raw.items()
    }
    # Bù sai số làm tròn nhỏ vào phần tử lớn nhất để tổng weights_effective khớp đúng 1.0
    diff = 1.0 - sum(rounded_weights_effective.values())
    if abs(diff) > 0:
        nonzero_keys = [k for k, v in rounded_weights_effective.items() if v > 0]
        if nonzero_keys:
            max_key = max(nonzero_keys, key=lambda k: rounded_weights_effective[k])
            rounded_weights_effective[max_key] = round(
                rounded_weights_effective[max_key] + diff, 6
            )

    passed_eval = execution.all_tests_passed and (weighted_score >= threshold_pass)

    return EvalReport(
        meta=meta,
        execution=execution,
        objective_metrics=objective_metrics,
        llm_assessment=llm_assessment,
        scores=rounded_scores,
        weights=weights,
        weights_effective=rounded_weights_effective,
        weighted_score=weighted_score,
        threshold_pass=threshold_pass,
        passed_eval=passed_eval,
        report_comparable=report_comparable,
        non_comparable_reason=non_comparable_reason,
        correctness_cap_applied=correctness_cap_applied,
    )


# =====================================================================
# Các Model nội bộ phục vụ Runner, Mutation, Quality Guard, Agent Loop
# =====================================================================


class MutationResult(BaseModel):
    """Kết quả thực thi Mutation Testing (FR-07)."""

    mutation_score_pct: float | None = Field(default=None, ge=0.0, le=100.0)
    mutation_killed: int | None = Field(default=None, ge=0)
    mutation_survived: int | None = Field(default=None, ge=0)
    mutation_excluded: int | None = Field(default=None, ge=0)
    mutation_timeout: int = Field(default=0, ge=0)
    mutation_status: MutationStatusType = "disabled"
    mutation_available: bool = False
    mutation_unavailable_reason: str | None = None
    survived_mutants: list[SurvivedMutant] = Field(default_factory=list)
    backend_used: Literal["ast", "mutmut", "none"] = "none"
    duration_ms: int = Field(default=0, ge=0)


class RunResult(BaseModel):
    """Kết quả thực thi pytest + coverage + phân loại lỗi (FR-03 & FR-04)."""

    target_file: str
    test_file: str
    exit_code: int
    stdout: str = ""
    stderr: str = ""
    duration_ms: int = Field(default=0, ge=0)
    all_tests_passed: bool = False
    tests_passed: int = Field(default=0, ge=0)
    tests_failed: int = Field(default=0, ge=0)
    tests_skipped: int = Field(default=0, ge=0)
    errors: int = Field(default=0, ge=0)
    pass_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    line_coverage_pct: float | None = Field(default=None, ge=0.0, le=100.0)
    branch_coverage_pct: float | None = Field(default=None, ge=0.0, le=100.0)
    missing_lines: list[int] = Field(default_factory=list)
    executed_lines: list[int] = Field(default_factory=list)
    test_count: int = Field(default=0, ge=0)
    error_category: ErrorCategory = ErrorCategory.NONE
    error_detail: str | None = None
    mutation: MutationResult | None = None

    def to_execution(self) -> Execution:
        return Execution(
            exit_code=self.exit_code,
            all_tests_passed=self.all_tests_passed,
            tests_passed=self.tests_passed,
            tests_failed=self.tests_failed,
            tests_skipped=self.tests_skipped,
            pass_rate=self.pass_rate,
            duration_ms=self.duration_ms,
        )

    def to_objective_metrics(self) -> ObjectiveMetrics:
        mut = self.mutation or MutationResult(
            mutation_status="disabled",
            mutation_available=False,
            mutation_unavailable_reason="Mutation testing bị tắt hoặc chưa chạy",
        )
        return ObjectiveMetrics(
            line_coverage_pct=self.line_coverage_pct,
            branch_coverage_pct=self.branch_coverage_pct,
            mutation_score_pct=mut.mutation_score_pct,
            mutation_killed=mut.mutation_killed,
            mutation_survived=mut.mutation_survived,
            mutation_excluded=mut.mutation_excluded,
            mutation_status=mut.mutation_status,
            mutation_available=mut.mutation_available,
            mutation_unavailable_reason=mut.mutation_unavailable_reason,
            test_count=self.test_count,
            missing_lines=list(self.missing_lines),
            survived_mutants=mut.survived_mutants,
        )


class AssertionDiffEntry(BaseModel):
    """Chi tiết sự thay đổi của một câu lệnh assert trong một hàm test (FR-04)."""

    test_name: str
    old_assertion: str
    new_assertion: str
    classification: Literal["unchanged", "allowed_value_update", "weakened"]
    reason: str


class GuardResult(BaseModel):
    """Kết quả kiểm tra Quality Guard hậu sửa test (FR-04)."""

    passed: bool
    syntax_ok: bool = True
    old_test_count: int = Field(default=0, ge=0)
    new_test_count: int = Field(default=0, ge=0)
    old_assert_count: int = Field(default=0, ge=0)
    new_assert_count: int = Field(default=0, ge=0)
    deleted_tests: list[str] = Field(default_factory=list)
    added_skips: list[str] = Field(default_factory=list)
    assertion_diffs: list[AssertionDiffEntry] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)


class AgentIterationLog(BaseModel):
    """Lịch sử một bước/vòng lặp trong Agent Loop (FR-06)."""

    iteration: int = Field(ge=0)
    action: Literal["gen", "fix", "improve", "eval"]
    metrics_before: dict[str, Any] = Field(default_factory=dict)
    metrics_after: dict[str, Any] = Field(default_factory=dict)
    llm_calls: int = Field(default=0, ge=0)
    tokens_used: int = Field(default=0, ge=0)
    accepted: bool = True
    reason: str = ""
