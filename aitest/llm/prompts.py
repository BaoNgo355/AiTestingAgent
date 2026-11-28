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
1. Always include `import pytest` at the very top of the file.
2. Write comprehensive `pytest` unit tests covering:
   - Normal / happy paths with exact expected values (`assert actual == expected`). Double-check all arithmetic calculations against the source code logic.
   - Boundary and edge conditions (0, negative numbers, empty strings/collections, exact threshold boundaries such as `<` vs `<=` and `>` vs `>=`).
   - Error and exception paths using `with pytest.raises(SpecificException):` where the arguments passed strictly match the exact `if ...: raise SpecificException(...)` branch in the source code (for example, distinguish `TypeError` for wrong type vs `ValueError` for invalid value).
3. Write focused, granular `test_*` functions separating normal cases, boundary cases, and distinct exception cases into separate functions (instead of packing many unrelated assertions into one giant test function), so each failure is isolated.
4. Ensure every test function is deterministic, isolated, and named `test_<descriptive_name>`.
5. Never use trivial tautologies (`assert True`, `assert x is not None` when a concrete value can be asserted) or `@pytest.mark.skip` / `xfail`.
6. Return ONLY raw Python code ready to be parsed by `ast.parse()`.
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

{% if ast_summary_lines is defined and ast_summary_lines %}
### Extracted AST Targets & Exact Exception Conditions
{% for line in ast_summary_lines %}
{{ line }}
{% endfor %}
{% endif %}

### Current Failing Test File (`{{ test_file }}`)
{{ current_test_code }}

### Pytest Output & Traceback
{{ pytest_output }}

### How to Fix the Failing Tests (MUST make actual code changes!)
1. Ensure `import pytest` and all necessary module imports are at the top of the file.
2. For any `AssertionError: assert <actual> == <expected>` in the traceback: update the expected value in `assert ... == <actual>` (or fix the input arguments) so it matches the true output of the reference source code.
3. For any failing `with pytest.raises(ExpectedException):` block (either `DID NOT RAISE` or raised a different exception):
   - Do NOT delete the `with pytest.raises(ExpectedException):` block!
   - Instead, change the arguments passed to the function inside `with pytest.raises(ExpectedException):` so that it actually triggers `ExpectedException` according to the `raise` conditions in the Reference Source Code (e.g., pass a wrong type like `1.5` or `"bad"` to trigger `TypeError`, or pass all-zero weights `[0, 0, 0]` / negative values `-1` to trigger `ValueError`).
4. Keep ALL existing test functions ({{ existing_test_names | join(', ') }}). Do NOT delete or rename existing `test_*` functions.
5. Keep at least the same total number of `assert` statements and `with pytest.raises(...)` blocks.
6. Do NOT add `@pytest.mark.skip`, `skipif`, `xfail`, or weaken `==` into `is not None`, `assert True`, `!=`, `>=`, `<=`.
7. Return the COMPLETE fixed test file as raw Python code without markdown fences.
"""
)

FIX_TARGETED_SYSTEM_PROMPT = (
    "You are an expert Python test maintainer. "
    "Fix ONLY the failing pytest test function(s) provided below so they pass against the source code. "
    "Do NOT modify the source code. Do NOT delete any failing test functions. "
    "Do NOT add skip/xfail markers. Do NOT weaken assertions into tautologies "
    "(such as assert True or assert x is not None). "
    "Return ONLY the fixed test function(s) (and any newly required import lines) as raw Python code."
)

FIX_TARGETED_USER_TEMPLATE = _JINJA_ENV.from_string(
    """\
Target file path: {{ target_file }}
Python module import path: `{{ module_name }}`
Classified Error Category: {{ error_category }}

### Reference Source Code (READ-ONLY — represents true behavior)
{{ source_code }}

{% if ast_summary_lines is defined and ast_summary_lines %}
### Extracted AST Targets & Exact Exception Conditions
{% for line in ast_summary_lines %}
{{ line }}
{% endfor %}
{% endif %}

{% if header_context %}
### Existing Test File Imports / Fixtures / Helpers (READ-ONLY — reuse as needed)
{{ header_context }}
{% endif %}

### Failing Test Function(s) to Fix (`{{ test_file }}`)
{{ failing_tests_code }}

### Pytest Output & Traceback
{{ pytest_output }}

### How to Fix (MUST make actual code changes!)
1. For any `AssertionError: assert <actual> == <expected>`: update the expected value (or input arguments) so it matches the true output of the Reference Source Code (use `pytest.approx(...)` or `math.isclose(...)` if floating-point precision differs).
2. For any failing `with pytest.raises(ExpectedException):`: do NOT delete it; instead adjust the arguments inside the `with` block so they actually trigger `ExpectedException` according to the Reference Source Code.
3. Keep the exact function name(s): {{ failing_test_names | join(', ') }}.
4. Keep at least the same number of `assert` statements and `with pytest.raises(...)` blocks in each function, and do NOT add `@pytest.mark.skip`, `skipif`, `xfail`, or weaken `==` into `is not None` / `assert True` / `!=` / `>=` / `<=`.
5. Return ONLY the fixed test function(s) (and any new imports needed, e.g. `import math`) as raw Python code without markdown fences.
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

{% if (existing_fixtures is defined and existing_fixtures) or (existing_helpers is defined and existing_helpers) %}
### Existing Fixtures & Helpers (REUSE these directly — DO NOT redefine them!)
{% if existing_fixtures is defined and existing_fixtures %}
{% for fix_src in existing_fixtures %}
{{ fix_src }}

{% endfor %}
{% endif %}
{% if existing_helpers is defined and existing_helpers %}
{% for help_src in existing_helpers %}
{{ help_src }}

{% endfor %}
{% endif %}
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


def render_targeted_fix_prompt(**context: Any) -> tuple[str, str]:
    return FIX_TARGETED_SYSTEM_PROMPT, FIX_TARGETED_USER_TEMPLATE.render(**context)


def render_improve_prompt(**context: Any) -> tuple[str, str]:
    return IMPROVE_SYSTEM_PROMPT, IMPROVE_USER_TEMPLATE.render(**context)


def render_eval_prompt(**context: Any) -> tuple[str, str]:
    return EVAL_SYSTEM_PROMPT, EVAL_USER_TEMPLATE.render(**context)
