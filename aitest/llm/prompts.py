"""Bộ Jinja2 Prompt Templates cho 4 tác vụ LLM: gen, fix, improve, eval (SRS v2.3.0 §17 Phụ lục A, B, C, D)."""

from __future__ import annotations

from typing import Any

from jinja2 import Environment, StrictUndefined

_JINJA_ENV = Environment(
    undefined=StrictUndefined,
    trim_blocks=True,
    lstrip_blocks=True,
    autoescape=False,
)

# =====================================================================
# Phụ lục A: Prompt `gen` (Sinh bộ test ban đầu)
# =====================================================================

GEN_SYSTEM_PROMPT = (
    "You are an expert Python QA engineer writing unit tests with pytest. "
    "Return ONLY valid, executable raw Python test code. "
    "Do NOT wrap code in markdown fences, and do NOT include explanations."
)

GEN_USER_TEMPLATE = _JINJA_ENV.from_string(
    """\
Target file path: {{ target_file }}
Python module import path: `{{ module_name }}` (import target symbols using `from {{ module_name }} import ...`)

### Source Code
{{ source_code }}

### Extracted AST Targets to Test
{% if ast_summary_lines %}
{% for line in ast_summary_lines %}
{{ line }}
{% endfor %}
{% else %}
- (No top-level functions or classes detected)
{% endif %}

{% if existing_test_names %}
### Existing Test Functions (DO NOT duplicate these test names)
{% for tname in existing_test_names %}
- `{{ tname }}`
{% endfor %}
{% endif %}

### Requirements
1. Write comprehensive `pytest` unit tests covering:
   - Normal / happy paths with exact expected values (`assert actual == expected`).
   - Boundary and edge conditions (0, negative numbers, empty strings/collections, exact threshold boundaries such as `<` vs `<=` and `>` vs `>=`).
   - Error and exception paths using `with pytest.raises(SpecificException):`.
2. Ensure every test function is deterministic, isolated, and named `test_< descriptive_name >`.
3. Never use trivial tautologies (`assert True`, `assert x is not None` when a concrete value can be asserted) or `@pytest.mark.skip` / `xfail`.
4. Return ONLY raw Python code ready to be parsed by `ast.parse()`.
"""
)

# =====================================================================
# Phụ lục B: Prompt `fix` (Sửa lỗi test không làm yếu chất lượng)
# =====================================================================

FIX_SYSTEM_PROMPT = (
    "You are an expert Python test maintainer. "
    "Fix ONLY the provided pytest test file so that it passes against the source code. "
    "Do NOT modify the source code. Do NOT delete any existing test functions. "
    "Do NOT add skip/xfail markers. Do NOT weaken assertions into tautologies "
    "(such as assert True or assert x is not None). "
    "Return the COMPLETE fixed test file as raw Python code."
)

FIX_USER_TEMPLATE = _JINJA_ENV.from_string(
    """\
Target file path: {{ target_file }}
Python module import path: `{{ module_name }}`
Classified Error Category: {{ error_category }}

### Reference Source Code (READ-ONLY — represents true behavior)
{{ source_code }}

### Current Failing Test File (`{{ test_file }}`)
{{ current_test_code }}

### Pytest Output & Traceback
{{ pytest_output }}

### Strict Quality Guard Rules (Violations will be automatically rejected!)
1. Keep ALL existing test functions ({{ existing_test_names | join(', ') }}). Do NOT delete or rename existing `test_*` functions.
2. Keep at least the same total number of `assert` statements and `with pytest.raises(...)` blocks.
3. Do NOT add `@pytest.mark.skip`, `@pytest.mark.skipif`, `@pytest.mark.xfail`, or call `pytest.skip()` / `pytest.xfail()`.
4. Do NOT weaken any `assert x == expected` into `assert x is not None`, `assert isinstance(...)`, `assert True`, `assert len(x) >= 0`, or relax `==` into `!=`, `>=`, `<=`.
5. Only fix syntax errors, wrong function call arguments, or miscalculated expected constant values in `==` so that they match the true logic of the source code.
6. Return the COMPLETE fixed test file as raw Python code without markdown fences.
"""
)

# =====================================================================
# Phụ lục C: Prompt `improve` (Bổ sung test diệt Survived Mutants)
# =====================================================================

IMPROVE_SYSTEM_PROMPT = (
    "You are an expert Python mutation-testing engineer. "
    "Write NEW pytest test functions (named test_*) specifically designed to kill the listed "
    "surviving mutants and cover the missing lines. "
    "Return ONLY raw Python code containing the new test functions (and any new imports needed). "
    "Do NOT repeat existing test functions, and do NOT include markdown fences or explanations."
)

IMPROVE_USER_TEMPLATE = _JINJA_ENV.from_string(
    """\
Target file path: {{ target_file }}
Python module import path: `{{ module_name }}` (import target symbols using `from {{ module_name }} import ...`)

### Source Code (with line numbers)
{{ numbered_source_code }}

### Existing Test Functions (Do NOT repeat or redefine these names)
{% if existing_test_names %}
{% for tname in existing_test_names %}
- `{{ tname }}`
{% endfor %}
{% else %}
- (None)
{% endif %}

### Surviving Mutants to Kill
{% if survived_mutants_details %}
{% for m in survived_mutants_details %}
- Line {{ m.line }}: mutation `{{ m.mutation }}` | Original line: `{{ m.source_line }}`
{% endfor %}
{% else %}
- (No surviving mutants listed)
{% endif %}

### Uncovered / Missing Lines to Cover
{% if missing_lines %}
Lines: {{ missing_lines | join(', ') }}
{% else %}
None (100% line coverage achieved)
{% endif %}

### Instructions
1. For each surviving mutant (for example, `line 12: >= -> >` or `+ -> -`), design a new `test_*` function that passes the exact boundary input where the original code and the mutated code produce DIFFERENT results.
2. Assert the exact expected output using `assert actual == expected` or `with pytest.raises(SpecificException):` so the mutant is killed.
3. Choose unique function names (`test_<target>_mutant_line_<N>_...`) that do not collide with existing test functions.
4. Return ONLY raw Python code containing the new `test_*` functions (and necessary imports).
"""
)

# =====================================================================
# Phụ lục D: Prompt `eval` (Chấm điểm chủ quan LLM-as-judge)
# =====================================================================

EVAL_SYSTEM_PROMPT = (
    "You are a strict software quality assurance reviewer. "
    "Evaluate the unit test suite against the source code on subjective quality criteria (0.0 to 10.0). "
    "Return ONLY a valid JSON object matching the required schema. "
    "Do NOT compute weighted_score, coverage score, or mutation score."
)

EVAL_USER_TEMPLATE = _JINJA_ENV.from_string(
    """\
Target file path: {{ target_file }}
Test file path: {{ test_file }}

### Source Code
{{ source_code }}

### Unit Test Suite
{{ test_code }}

### Objective Execution Evidence (Measured by Tools)
- all_tests_passed: {{ all_tests_passed }}
- tests_passed: {{ tests_passed }} / tests_failed: {{ tests_failed }} / tests_skipped: {{ tests_skipped }}
- test_count: {{ test_count }}
- line_coverage_pct: {{ line_coverage_pct if line_coverage_pct is not none else "null" }}
- mutation_status: {{ mutation_status }}
- mutation_score_pct: {{ mutation_score_pct if mutation_score_pct is not none else "null" }}
- survived_mutants_count: {{ survived_mutants_count }}
{% if survived_mutants_preview %}
- survived_mutants sample:
{% for m in survived_mutants_preview %}
  * line {{ m.line }}: {{ m.mutation }}
{% endfor %}
{% endif %}

### Scoring Rubric (Scale 0.0 to 10.0 for each criterion)
1. `correctness` (0.0 - 10.0): Accuracy of assertions, verification of exact expected values and boundary conditions rather than weak checks.
2. `readability` (0.0 - 10.0): Clear descriptive `test_*` names, clean Arrange-Act-Assert (AAA) structure.
3. `maintainability` (0.0 - 10.0): Low duplication, appropriate modularization, clean setup.
4. `isolation` (0.0 - 10.0): Tests are independent, do not share mutable global state or depend on execution order.
5. `robustness` (0.0 - 10.0): Thorough checking of edge cases, invalid inputs, and exceptions (`pytest.raises`).
6. `determinism` (0.0 - 10.0): Free from unseeded randomness, wall-clock timing dependencies, or external network/filesystem leaks.

Provide concrete weaknesses in `issues` (each with `title`, `description`, `suggestion` in Vietnamese) and an overall summary in `summary` (in Vietnamese).
"""
)


def render_gen_prompt(**context: Any) -> tuple[str, str]:
    return GEN_SYSTEM_PROMPT, GEN_USER_TEMPLATE.render(**context)


def render_fix_prompt(**context: Any) -> tuple[str, str]:
    return FIX_SYSTEM_PROMPT, FIX_USER_TEMPLATE.render(**context)


def render_improve_prompt(**context: Any) -> tuple[str, str]:
    return IMPROVE_SYSTEM_PROMPT, IMPROVE_USER_TEMPLATE.render(**context)


def render_eval_prompt(**context: Any) -> tuple[str, str]:
    return EVAL_SYSTEM_PROMPT, EVAL_USER_TEMPLATE.render(**context)
