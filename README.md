# aitest — AI Testing Agent

[![Python](https://img.shields.io/badge/Python-%3E%3D3.10-blue.svg)](https://www.python.org/)
[![Version](https://img.shields.io/badge/version-2.3.0-green.svg)](pyproject.toml)
[![Test Framework](https://img.shields.io/badge/tests-pytest-orange.svg)](https://docs.pytest.org/)
[![LLM Provider](https://img.shields.io/badge/LLM-Ollama%20Local-purple.svg)](https://ollama.com/)

**`aitest`** là công cụ dòng lệnh (CLI AI Agent) hỗ trợ tự động **sinh, chạy, sửa lỗi, cải thiện và đánh giá chất lượng Unit Test Python** bằng mô hình ngôn ngữ lớn (LLM) chạy local qua **Ollama**.

Hệ thống kết hợp phân tích tĩnh (AST), đo lường khách quan (`pytest`, `pytest-cov`, Mutation Testing) và chấm điểm theo rubric (LLM-as-judge) để tạo ra bộ unit test có độ phủ và khả năng phát hiện lỗi thực sự cao, đồng thời bảo đảm an toàn tuyệt đối cho mã nguồn gốc.

---

## Mục lục

- [Tính năng chính](#tính-năng-chính)
- [Kiến trúc & Nguyên lý hoạt động](#kiến-trúc--nguyên-lý-hoạt-động)
- [Yêu cầu hệ thống](#yêu-cầu-hệ-thống)
- [Cài đặt](#cài-đặt)
- [Hướng dẫn sử dụng CLI](#hướng-dẫn-sử-dụng-cli)
- [Cấu hình (`aitest.yaml` & `.env`)](#cấu-hình-aitestyaml--env)
- [Cấu trúc thư mục](#cấu-trúc-thư-mục)
- [Thực nghiệm & Kiểm thử](#thực-nghiệm--kiểm-thử)

---

## Tính năng chính

- **Phân tích mã nguồn bằng AST**: Tự động trích xuất thông tin hàm, lớp, tham số và ngữ cảnh từ file Python đơn lẻ (`*.py`) trước khi gửi cho LLM.
- **Sinh Unit Test an toàn (Safe Append)**: Sinh test cho `pytest`, hỗ trợ chế độ `append` (mặc định, tự động loại bỏ hàm test trùng lặp bằng AST) hoặc `overwrite`.
- **Đo lường khách quan (`pytest-cov` & Mutation Testing)**:
  - Đo Line Coverage riêng cho file nguồn mục tiêu.
  - Hỗ trợ **Dual Mutation Backend** (`auto`): sử dụng Built-in AST Mutation Engine đa nền tảng (Windows/Linux/macOS) hoặc `mutmut` trên môi trường POSIX.
- **Tự động sửa lỗi có kiểm soát (`Quality Guard`)**:
  - Chỉ chỉnh sửa file test, **không bao giờ** tự ý sửa mã nguồn gốc.
  - Kiểm tra **AST Assertion Diff** để ngăn chặn hành vi làm yếu câu lệnh kiểm tra (*assertion weakening*) hoặc làm giảm số lượng test/assertion.
- **Đánh giá chất lượng 2 tầng (`EvalReport` JSON)**:
  - **Tầng khách quan**: Tỷ lệ pass/fail, line coverage, mutation score.
  - **Tầng chủ quan (LLM-as-judge)**: Chấm điểm theo rubric 8 tiêu chí (`correctness`, `coverage`, `mutation`, `readability`, `maintainability`, `isolation`, `robustness`, `determinism`) và xuất báo cáo JSON chuẩn Pydantic.
- **Vòng lặp Agent tự động**: Thực thi khép kín `gen -> run -> fix -> mutation -> eval -> improve` với kiểm soát ngân sách (số vòng lặp, số lần gọi LLM, tổng token, thời gian chạy) và hỗ trợ chế độ `--baseline` để đối sánh.

---

## Kiến trúc & Nguyên lý hoạt động

```text
Phân tích mã nguồn (AST) → Sinh test (gen) → Chạy test (pytest + cov)
        ↑                                              │
        │                                              ▼
  Cải thiện (improve) ← Đánh giá 2 tầng (eval) ← Sửa lỗi (fix) / Mutation Testing
```

---

## Yêu cầu hệ thống

- **Python**: `>= 3.10`
- **Ollama**: Đã cài đặt và đang chạy local (mặc định tại `http://localhost:11434`)
- **Model LLM khuyến nghị**: `qwen2.5-coder:7b` (hoặc model coding khác trên Ollama)

---

## Cài đặt

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

### 3. Chuẩn bị Ollama

```bash
ollama serve
ollama pull qwen2.5-coder:7b
```

### 4. Khởi tạo cấu hình trong thư mục làm việc

```bash
aitest init
```

Lệnh này sẽ tạo `aitest.yaml`, `.env.example`, cập nhật `.gitignore` (thêm `.env`, `.aitest_backup/`, `.aitest_logs/`) và kiểm tra kết nối tới Ollama cùng các thư viện `pytest`, `pytest-cov`.

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
aitest gen examples/calculator.py --mode overwrite --model qwen2.5-coder:7b
```

Các tùy chọn chính:
- `--mode`, `-m`: `append` (mặc định - Safe Append) hoặc `overwrite`.
- `--no-llm-dedup`: Không gửi danh sách tên hàm test hiện có vào prompt LLM (AST vẫn dedup khi ghi file).
- `--model`: Ghi đè tên model Ollama trong `aitest.yaml`.

### 3. `aitest run` — Chạy test, Coverage & Mutation Testing

```bash
aitest run examples/calculator.py
aitest run examples/calculator.py --no-mutation --timeout 120
```

Các tùy chọn chính:
- `--cov / --no-cov`: Bật/tắt đo line coverage.
- `--mutation / --no-mutation`: Bật/tắt Mutation Testing (chỉ chạy khi bộ test pass 100%).
- `--timeout`, `-t`: Giới hạn thời gian chạy `pytest` (giây).

### 4. `aitest fix` — Tự động sửa lỗi file test

```bash
aitest fix examples/calculator.py
aitest fix examples/calculator.py --max-iters 3 --dry-run
```

Các tùy chọn chính:
- `--max-iters`: Số vòng lặp sửa lỗi tối đa (mặc định `3`).
- `--dry-run`: Chỉ kiểm tra Quality Guard và hiển thị unified diff, không ghi đè file test.
- `--model`: Ghi đè tên model Ollama.

### 5. `aitest eval` — Đánh giá chất lượng Unit Test

```bash
aitest eval examples/calculator.py -o eval_report.json
aitest eval examples/calculator.py --no-mutation
```

Các tùy chọn chính:
- `--format`: Định dạng xuất báo cáo (hiện hỗ trợ `json`).
- `--output`, `-o`: Đường dẫn lưu file báo cáo JSON.
- `--no-mutation`: Tắt Mutation Testing (báo cáo được đánh dấu `report_comparable = false`).
- `--model`: Ghi đè tên model Ollama.

### 6. `aitest agent` — Chạy vòng lặp Agent hoàn chỉnh

```bash
aitest agent examples/calculator.py
aitest agent examples/calculator.py --max-iters 3 --threshold 8.0 -o agent_report.json
aitest agent examples/calculator.py --baseline
```

Các tùy chọn chính:
- `--max-iters`: Số vòng lặp Agent tối đa (mặc định `3`).
- `--threshold`: Ngưỡng `weighted_score` để đạt yêu cầu (mặc định `8.0`).
- `--max-llm-calls`: Ngân sách số lần gọi LLM tối đa (mặc định `12`).
- `--max-total-tokens`: Ngân sách tổng token tối đa (mặc định `200000`).
- `--max-runtime-sec`: Giới hạn thời gian chạy tối đa tính bằng giây (mặc định `900`).
- `--baseline`: Chạy nhánh đối sánh Baseline A (sinh test + sửa lỗi nếu có + đánh giá 1 lần, bỏ qua vòng lặp cải thiện).
- `--output`, `-o`: Lưu báo cáo `EvalReport` JSON cuối cùng ra file.
- `--model`: Ghi đè tên model Ollama.

---

## Cấu hình (`aitest.yaml` & `.env`)

Bạn có thể tham khảo file mẫu [`aitest.yaml.example`](aitest.yaml.example) và [`.env.example`](.env.example):

- **Biến môi trường (`.env`)**:
  ```env
  OLLAMA_BASE_URL=http://localhost:11434
  # OLLAMA_API_KEY=...   # chỉ cần nếu kết nối tới Ollama remote có bật xác thực
  ```
- **File `aitest.yaml`**: Cho phép tùy chỉnh `model`, `coverage`, `mutation` (`tool: auto | ast | mutmut`, `timeout_sec`, `per_mutant_timeout_sec`), trọng số `eval.rubric`, ngưỡng `threshold_pass`, quy tắc `fix.quality_guard`, ngân sách `agent` và tham số `llm` (`temperature`, `max_tokens`, `num_ctx`, `keep_alive`, `retry`).

---

## Cấu trúc thư mục

```text
AiTestingAgent/
├── aitest/                  # Mã nguồn chính của gói CLI aitest
│   ├── llm/                 # Ollama client & prompt templates
│   ├── agent.py             # Vòng lặp Agent tự động & nhánh Baseline
│   ├── analyzer.py          # Phân tích mã nguồn Python bằng AST
│   ├── cli.py               # Định nghĩa 6 lệnh CLI (Typer + Rich)
│   ├── config.py            # Tải cấu hình từ aitest.yaml & .env
│   ├── context.py           # Xây dựng ngữ cảnh cho LLM
│   ├── evaluator.py         # Đánh giá 2 tầng & chấm điểm rubric
│   ├── fixer.py             # Sửa lỗi test có kiểm soát
│   ├── generator.py         # Sinh test & Safe Append
│   ├── mutation.py          # Dual Mutation Backend (AST Engine / mutmut)
│   ├── output.py            # Hiển thị bảng kết quả & xuất JSON
│   ├── progress.py          # Hiển thị tiến trình thực thi
│   ├── quality_guard.py     # Kiểm tra AST Assertion Diff & chống làm yếu test
│   ├── runner.py            # Thực thi pytest & thu thập coverage
│   ├── schemas.py           # Các schema dữ liệu Pydantic
│   └── utils.py             # Tiện ích (atomic write, backup, logging...)
├── docs/                    # Tài liệu đặc tả (SRS, kế hoạch dự án)
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
pytest
```

### Chạy kịch bản thực nghiệm đối sánh Baseline A vs Agent B

```bash
python scripts/run_experiment.py --runs 3 --max-iters 3 --threshold 8.0
```

Mặc định kịch bản sẽ chạy trên các file mẫu trong `examples/`: `calculator.py`, `string_utils.py`, `date_scheduler.py`, `inventory.py`, `bank_account.py`.
