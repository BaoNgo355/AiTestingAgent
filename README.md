# aitest — AI Testing Agent

[![Python](https://img.shields.io/badge/Python-%3E%3D3.10-blue.svg)](https://www.python.org/)
[![Version](https://img.shields.io/badge/version-4.0.0-green.svg)](pyproject.toml)
[![Test Framework](https://img.shields.io/badge/tests-pytest-orange.svg)](https://docs.pytest.org/)
[![LLM Provider](https://img.shields.io/badge/LLM-Google%20Antigravity%20SDK-purple.svg)](https://antigravity.google/)

**`aitest`** là công cụ dòng lệnh (CLI AI Agent) hỗ trợ tự động **sinh, chạy, sửa lỗi, cải thiện và đánh giá chất lượng Unit Test Python** bằng **Google Antigravity Python SDK** (`google-antigravity`, kết nối Gemini Developer API qua `GEMINI_API_KEY` hoặc Vertex AI).

Hệ thống kết hợp phân tích tĩnh (AST), đo lường khách quan (`pytest`, `pytest-cov`, Mutation Testing) và chấm điểm theo rubric có kiểm soát (LLM-as-judge với JSON Schema) để tạo ra bộ unit test có độ phủ và khả năng phát hiện lỗi thực sự cao, đồng thời bảo đảm an toàn tuyệt đối cho mã nguồn gốc.

---

## Mục lục

- [Tính năng chính](#tính-năng-chính)
- [Kiến trúc & Nguyên lý hoạt động](#kiến-trúc--nguyên-lý-hoạt-động)
- [Yêu cầu hệ thống](#yêu-cầu-hệ-thống)
- [Cài đặt & Chuẩn bị](#cài-đặt--chuẩn-bị)
- [Hướng dẫn sử dụng CLI](#hướng-dẫn-sử-dụng-cli)
- [Cấu hình (`aitest.yaml` & `.env`)](#cấu-hình-aitestyaml--env)
- [Cấu trúc thư mục](#cấu-trúc-thư-mục)
- [Thực nghiệm & Kiểm thử](#thực-nghiệm--kiểm-thử)

---

## Tính năng chính

- **Phân tích mã nguồn bằng AST**: Trích xuất hàm, lớp, phương thức, chữ ký, chú thích kiểu và điều kiện ngoại lệ (`if ...: raise ...`) từ file Python đơn lẻ (`*.py`).
- **Tích hợp Google Antigravity Python SDK (`google-antigravity`)**:
  - Sử dụng trực tiếp `Agent` + `LocalAgentConfig` của SDK (mặc định `gemini-4-hydrogen`), hỗ trợ cả `GEMINI_API_KEY` (Google AI Studio) lẫn **Vertex AI**.
  - Cô lập quyền an toàn bằng `policy.deny_all()`, tắt toàn bộ built-in tools/subagents và trỏ `app_data_dir` vào thư mục tạm riêng để bảo đảm chỉ `aitest` điều phối file test; bước `eval` ép đầu ra chuẩn Pydantic `LLMAssessment` qua `response_schema`.
  - Hỗ trợ tùy chọn `--model` và `--judge-model` (tách model chấm điểm riêng).
- **Sinh Unit Test an toàn (Safe Append)**: Hỗ trợ chế độ `append` (mặc định, tự động dedup bằng AST cho cả hàm `test_*` và lớp `Test*`) hoặc `overwrite`.
- **Đo lường khách quan (`pytest-cov` & Dual Mutation Engine)**:
  - Đo Line Coverage riêng cho file nguồn mục tiêu.
  - Built-in AST Mutation Engine đa nền tảng (Windows/Linux/macOS) tối ưu tốc độ bằng persistent worker và lọc theo dòng đã thực thi, kèm adapter `mutmut` trên POSIX.
- **Tự động sửa lỗi có kiểm soát (`Targeted Fix` + `Quality Guard`)**:
  - Phân loại lỗi 7 nhóm trước khi gọi LLM; chỉ sửa `TEST_FAILURE` và `TEST_SYNTAX`.
  - Khoanh vùng sửa riêng hàm test bị fail (**Targeted Function-level Fix**) để giữ nguyên các test đang pass.
  - Kiểm tra **AST Assertion Diff** chống làm yếu câu lệnh `assert` (*assertion weakening*) và tự động rollback nếu vi phạm hoặc làm giảm mutation score.
- **Đánh giá chất lượng 2 tầng & Báo cáo trực quan (`JSON` / `Markdown` / `HTML Dashboard`)**:
  - **Tầng khách quan**: Tỷ lệ pass/fail, line coverage (`missing_lines`), mutation score (`survived_mutants` kèm ngữ cảnh dòng mã nguồn).
  - **Tầng chủ quan**: Rubric 8 tiêu chí (`correctness` bị chặn trên bởi `mutation_score_10 + 2.0`).
  - **Smart TTY & Đa định dạng**: Gõ trực tiếp trên Terminal hiển thị Rich Dashboard 8 tiêu chí, bảng biến thiên $\Delta$ Agent và góp ý LLM (ẩn JSON thô); khi pipe/CI/`-q` tự động xuất JSON chuẩn Pydantic v2 ra `stdout`; hỗ trợ `--format json|md|html` và tự suy luận từ đuôi `-o` (file `.html` độc lập có biểu đồ SVG Radar, Source Code Viewer tô màu và Agent Timeline).
- **Vòng lặp Agent tự động**: Thực thi khép kín `gen -> run -> fix -> mutation -> eval -> improve` với **Selective Per-Function Acceptance**, kiểm soát ngân sách và chế độ `--baseline` để đối sánh thực nghiệm.

---

## Kiến trúc & Nguyên lý hoạt động

```text
AST Analyzer -> Sinh test (gen) -> Chạy test (pytest + cov)
      ^                                       |
      |                                       v
Cải thiện (improve) <- Đánh giá (eval) <- Sửa lỗi (fix) / Mutation
```

---

## Yêu cầu hệ thống

- **Python**: `>= 3.10`
- **Google Antigravity SDK (`google-antigravity>=0.1.24`)**: tự động cài khi chạy `pip install -e .`
- **Xác thực**: biến môi trường hoặc `.env` chứa `GEMINI_API_KEY` (tạo tại Google AI Studio) hoặc cấu hình Vertex AI
- **Model mặc định**: `gemini-4-hydrogen` (có thể ghi đè qua `--model` / `--judge-model`)

---

## Cài đặt & Chuẩn bị

### 1. Clone repository và tạo môi trường ảo

```bash
git clone https://github.com/BaoNgo355/AiTestingAgent.git
cd AiTestingAgent

python -m venv .venv
```

Kích hoạt môi trường ảo:

- **Windows (PowerShell):**
  ```powershell
  .venv\Scripts\Activate.ps1
  ```
- **Linux / macOS:**
  ```bash
  source .venv/bin/activate
  ```

### 2. Cài đặt gói `aitest`

```bash
pip install -e .
```

### 3. Thiết lập khóa xác thực (`GEMINI_API_KEY`)

Tạo API key tại https://aistudio.google.com/app/api-keys và lưu vào file `.env` (file `.env` đã được đưa vào `.gitignore`):

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

### 4. Khởi tạo cấu hình trong thư mục làm việc

```bash
aitest init
```

Lệnh này sẽ tạo `aitest.yaml`, `.env.example`, cập nhật `.gitignore` (thêm `.env`, `.aitest_backup/`, `.aitest_logs/`) và kiểm tra `pytest`, `pytest-cov` cùng trạng thái gói `google-antigravity` / khóa xác thực.

---

## Hướng dẫn sử dụng CLI

Xem danh sách lệnh hoặc phiên bản:

```bash
aitest --help
aitest --version
```

### 1. `aitest init` — Khởi tạo dự án

```bash
aitest init [--force]
```

### 2. `aitest gen` — Sinh Unit Test

```bash
aitest gen examples/calculator.py
aitest gen examples/calculator.py --mode overwrite --model gemini-4-hydrogen
```

Các tùy chọn chính:
- `--mode`, `-m`: `append` (mặc định - Safe Append) hoặc `overwrite`.
- `--no-llm-dedup`: Không gửi danh sách tên hàm test hiện có vào prompt LLM.
- `--test-file`: Chỉ định đường dẫn file test đích.
- `--model`: Ghi đè model ID của Google Antigravity SDK.
- `--quiet` / `-q`, `--verbose`: Điều chỉnh hiển thị tiến độ trên `stderr`.

### 3. `aitest run` — Chạy test, Coverage & Mutation Testing

```bash
aitest run examples/calculator.py
aitest run examples/calculator.py --no-mutation --timeout 120
```

Các tùy chọn chính:
- `--cov / --no-cov`: Bật/tắt đo line coverage của riêng file nguồn mục tiêu.
- `--mutation / --no-mutation`: Bật/tắt Mutation Testing (chỉ chạy khi bộ test pass 100%).
- `--timeout`, `-t`: Giới hạn thời gian chạy `pytest` (giây, mặc định `120`).

### 4. `aitest fix` — Tự động sửa lỗi file test

```bash
aitest fix examples/calculator.py
aitest fix examples/calculator.py --max-iters 3 --dry-run
```

Các tùy chọn chính:
- `--max-iters`: Số vòng lặp sửa lỗi tối đa (mặc định `3`).
- `--dry-run`: Chỉ kiểm tra Quality Guard và hiển thị unified diff, không ghi đè file test.
- `--model`: Ghi đè model ID của Google Antigravity SDK.

### 5. `aitest eval` — Đánh giá chất lượng Unit Test

```bash
aitest eval examples/calculator.py
aitest eval examples/calculator.py -o eval_report.html
aitest eval examples/calculator.py --judge-model gemini-3.8-pro --format json -o eval_report.json
```

Các tùy chọn chính:
- `--format`, `-f`: Định dạng xuất báo cáo (`json`, `md`, `html`; mặc định tự động theo Smart TTY hoặc đuôi file `-o`).
- `--output`, `-o`: Đường dẫn lưu file báo cáo (`.json`, `.md`, `.html`).
- `--no-mutation`: Tắt Mutation Testing (báo cáo được đánh dấu `report_comparable = false`).
- `--model`, `--judge-model`: Tùy biến model sinh và model chấm điểm rubric.

### 6. `aitest agent` — Chạy vòng lặp Agent hoàn chỉnh

```bash
aitest agent examples/calculator.py
aitest agent examples/calculator.py --max-iters 3 --threshold 8.0 --judge-model gemini-3.8-pro -o agent_report.html
aitest agent examples/calculator.py --baseline -o baseline_report.json
```

Các tùy chọn chính:
- `--max-iters`: Số vòng lặp Agent tối đa (mặc định `3`).
- `--threshold`: Ngưỡng `weighted_score` để đạt yêu cầu (mặc định `8.0`).
- `--max-llm-calls`: Ngân sách số lần gọi LLM tối đa (mặc định `12`).
- `--max-total-tokens`: Ngân sách tổng token tối đa (mặc định `400000`).
- `--max-runtime-sec`: Giới hạn thời gian chạy tối đa tính bằng giây (mặc định `900`).
- `--baseline`: Chạy nhánh đối sánh Baseline A (sinh test + sửa lỗi nếu có + đánh giá 1 lần, bỏ qua vòng lặp cải thiện).
- `--format`, `-f`: Định dạng xuất báo cáo (`json`, `md`, `html`).
- `--output`, `-o`: Lưu báo cáo cuối cùng ra file (`.json`, `.md`, `.html`).
- `--model`, `--judge-model`, `--test-file`, `--quiet`, `--verbose`.

---

## Cấu hình (`aitest.yaml` & `.env`)

Tham khảo [`aitest.yaml.example`](aitest.yaml.example) và [`.env.example`](.env.example):

- **Biến môi trường (`.env`)**:
  ```env
  GEMINI_API_KEY=your_gemini_api_key_here
  # AITEST_MODEL=gemini-4-hydrogen
  ```
- **File `aitest.yaml`**: Cho phép tùy chỉnh `model` (mặc định `gemini-4-hydrogen`), `timeout_sec`, `coverage`, `mutation`, trọng số `eval.rubric`, `eval.judge_model`, ngưỡng `threshold_pass`, quy tắc `fix.quality_guard`, ngân sách `agent` và tham số `llm` (`vertex`, `project`, `location`, `timeout_sec`, `retry`).

---

## Cấu trúc thư mục

```text
AiTestingAgent/
├── aitest/                  # Mã nguồn chính của gói CLI aitest
│   ├── llm/                 # Google Antigravity SDK Client & Jinja2 prompts
│   ├── agent.py             # Vòng lặp Agent tự động & nhánh Baseline
│   ├── analyzer.py          # Phân tích mã nguồn Python bằng AST
│   ├── cli.py               # Định nghĩa 6 lệnh CLI (Typer + Rich)
│   ├── config.py            # Tải cấu hình từ CLI, env, aitest.yaml
│   ├── context.py           # Xây dựng ngữ cảnh cho LLM
│   ├── evaluator.py         # Đánh giá 2 tầng & chấm điểm rubric
│   ├── fixer.py             # Sửa lỗi test khoanh vùng & Quality Guard
│   ├── generator.py         # Sinh test & Safe Append
│   ├── mutation.py          # Dual Mutation Backend (AST Engine / mutmut)
│   ├── output.py            # Rich Dashboard, Smart TTY & điều phối xuất báo cáo
│   ├── progress.py          # Hiển thị tiến trình thực thi Rich
│   ├── quality_guard.py     # Kiểm tra AST Assertion Diff & chống làm yếu test
│   ├── reporting.py         # Kết xuất báo cáo Markdown, HTML + SVG & CSV thực nghiệm
│   ├── runner.py            # Thực thi pytest & thu thập coverage
│   ├── schemas.py           # Các schema dữ liệu Pydantic
│   └── utils.py             # Tiện ích (atomic write, backup, logging...)
├── docs/                    # Tài liệu đặc tả (SRS v4.0.0, kế hoạch dự án, scripts)
├── examples/                # Các file Python mẫu để thử nghiệm & đánh giá
├── scripts/                 # Kịch bản chạy thực nghiệm đối sánh
│   └── run_experiment.py
├── tests/                   # Bộ unit test cho chính hệ thống aitest
├── .env.example             # Mẫu biến môi trường
├── aitest.yaml.example      # Mẫu file cấu hình
└── pyproject.toml           # Khai báo package & dependencies
```

---

## Thực nghiệm & Kiểm thử

### Chạy bộ unit test của dự án

```bash
pytest tests/ -v
```

### Chạy kịch bản thực nghiệm đối sánh Baseline A vs Agent B

```bash
python scripts/run_experiment.py --runs 3 --max-iters 3 --threshold 8.0
```

Mặc định kịch bản sẽ chạy trên 5 file mẫu trong `examples/` và xuất đầy đủ 4 định dạng báo cáo: `experiment_report.json`, `experiment_report.md`, `experiment_report.html` (kèm biểu đồ cột SVG so sánh Baseline A vs Agent B) và `experiment_summary.csv` (UTF-8 BOM cho Excel) với bảng Mean ± SD, chênh lệch $\Delta$ Mutation Score theo file và tỷ lệ *Win / Tie / Loss*.
