# Contributing to `llm-drift-bench`

Thank you for your interest in contributing to **`llm-drift-bench`**! We aim to keep this package fast, modular, strictly typed, and zero-bloat.

---

## 1. Development Environment Setup

### Prerequisites
- Python **3.10+**
- `git`

### Clone and Install in Editable Mode
```bash
git clone https://github.com/anant-goel/llm-drift-bench.git
cd llm-drift-bench
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

---

## 2. Running Tests, Linting & Type Checking

Before opening a Pull Request, verify that all checks pass locally:

### Run the Test Suite with Coverage
```bash
pytest --cov=llm_drift_bench --cov-report=term-missing --cov-fail-under=85
```

You can also run the test suite with zero external packages using Python's built-in `unittest` runner:
```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

### Run Ruff Linter
```bash
ruff check src tests
```

### Run Mypy Strict Type Checking
```bash
mypy src/llm_drift_bench
```

---

## 3. Pull Request & Commit Guidelines

1. **Zero-Bloat Philosophy**: Do not add third-party runtime dependencies to `[project].dependencies` unless strictly necessary; prefer standard library implementations or optional extras.
2. **Atomic Conventional Commits**: Every logical change (schema addition, metric function, CLI flag, test file, or documentation update) should have an atomic commit following [Conventional Commits](https://www.conventionalcommits.org/):
   - `feat(metrics): ...`
   - `fix(validator): ...`
   - `test(engine): ...`
   - `docs: ...`
   - `ci: ...`
3. **Test Coverage**: New features and bug fixes must include unit tests in `tests/` maintaining `>85%` total coverage.
4. **Strict Typing**: All functions and methods in `src/llm_drift_bench/` must include full Python type annotations (`from __future__ import annotations`) and pass `mypy --strict`.

---

## 4. `good first issue` Roadmap

Looking for a place to start contributing? Here is our curated list of `good first issue` enhancements:

- [ ] **`[good first issue]` Support `uniqueItems` and `multipleOf` in `validator.py`**
  - Extend `validate_against_schema` in `src/llm_drift_bench/validator.py` to validate JSON Schema `uniqueItems: true` on arrays and `multipleOf` on numeric fields, with corresponding unit tests in `tests/test_validator.py`.
- [ ] **`[good first issue]` Markdown Report Export (`--format markdown`)**
  - Add a `format_report_markdown(report: DriftBenchmarkReport) -> str` helper in `src/llm_drift_bench/engine.py` and expose `--format markdown` in `llm-drift-bench run` so CI workflows can post benchmark tables directly to GitHub Pull Request comments.
- [ ] **`[good first issue]` Per-Category Drift Breakdown in `DriftBenchmarkReport`**
  - Group `SampleDriftResult` items by `category` and include per-category mean drift and conformity rates in `DriftBenchmarkReport.to_dict()`.
- [ ] **`[good first issue]` CSV / TSV Export Option for Sample Results**
  - Add `--csv <path>` to `llm-drift-bench run` using Python's standard library `csv` module to export per-prompt drift scores for spreadsheet analysis.
- [ ] **`[good first issue]` JSON Schema `anyOf` / `oneOf` Combinator Support**
  - Implement recursive support for `anyOf` and `oneOf` schema combinators in `src/llm_drift_bench/validator.py`.
