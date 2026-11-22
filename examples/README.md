# Reproducible Offline Examples

Handwritten demonstration fixtures for trying every core `llm-drift-bench` workflow offline without API keys, GPUs, or network calls. No timings or token counts are invented.

## Files in this directory

| File | Description |
| :--- | :--- |
| `extraction-suite.json` | Two required-field JSON Schema prompt cases (`invoice-01`, `ticket-01`). |
| `baseline.jsonl` | Reference baseline responses (`demo-reference`). |
| `candidate-pass.jsonl` | Identical responses under a passing candidate label (`demo-candidate-pass`). |
| `candidate-drift.jsonl` | Regressed responses: shifted invoice total + dropped required `priority` field (`demo-candidate-drift`). |
| `expected-pass.json` | Canonical saved engine report for the passing run (checked by `tests/test_examples.py`). |
| `expected-drift.json` | Canonical saved engine report for the failing run (checked by `tests/test_examples.py`). |
| `drift.toml` | Sample evaluation policy demonstrating thresholds, metric weights, and per-field rules (`ignore`, `tolerance_abs`). |
| `demo_suite.json` | Original legacy prompt suite (kept separate; do not mix its prompt IDs with `extraction-suite.json`). |

---

## Try the CLI workflows

Run from the repository root (substitute `.\.venv\Scripts\python.exe` on Windows PowerShell or `.venv/bin/python` on Linux/macOS):

### 1. Basic pass / fail drift comparison & diagnosis
```bash
# Passing comparison -> exit 0
python -m llm_drift_bench run -b examples/baseline.jsonl -c examples/candidate-pass.jsonl -s examples/extraction-suite.json

# Regressed comparison -> exit 1
python -m llm_drift_bench run -b examples/baseline.jsonl -c examples/candidate-drift.jsonl -s examples/extraction-suite.json

# Explain why the failing report failed -> exit 0
python -m llm_drift_bench diagnose -i examples/expected-drift.json
```

### 2. Evaluate with `drift.toml` policy
```bash
python -m llm_drift_bench run \
  -b examples/baseline.jsonl \
  -c examples/candidate-pass.jsonl \
  -s examples/extraction-suite.json \
  --config examples/drift.toml
```

### 3. Visual reports: GitHub PR comments, HTML JSON diffs & SVG Pareto dashboard
```bash
# Compact GitHub PR comment showing deltas vs the passing report
python -m llm_drift_bench report -i examples/expected-drift.json --format pr-comment --previous examples/expected-pass.json

# Offline HTML report with opt-in collapsible side-by-side JSON field diffs
python -m llm_drift_bench run \
  -b examples/baseline.jsonl -c examples/candidate-drift.jsonl -s examples/extraction-suite.json \
  --format html --include-diffs --no-fail-on-drift > drift-diffs.html

# Rank candidates in a Rich table and export an offline SVG Pareto HTML dashboard
python -m llm_drift_bench compare-reports examples/expected-pass.json examples/expected-drift.json \
  --sort-by passed --format rich --html-output comparison.html
```

### 4. Auto-infer JSON Schemas & calibrate a `drift.toml`
```bash
# Infer strict JSON Schemas straight from baseline.jsonl, then validate against them
python -m llm_drift_bench infer-schema -i examples/baseline.jsonl -o inferred-prompts.json --strict-additional
python -m llm_drift_bench validate -i examples/baseline.jsonl -p inferred-prompts.json --strict

# Calibrate recommended thresholds from a saved report
python -m llm_drift_bench calibrate examples/expected-pass.json
```

### 5. Drift history ledger & trend charts
```bash
python -m llm_drift_bench history record --ledger demo-history.jsonl -i examples/expected-pass.json --tag q8
python -m llm_drift_bench history record --ledger demo-history.jsonl -i examples/expected-drift.json --tag q4
python -m llm_drift_bench history show --ledger demo-history.jsonl --format text
python -m llm_drift_bench history show --ledger demo-history.jsonl --format html -o trends.html
```

---

## Use the batch Python API

Evaluate multiple candidate captures against a shared baseline and inspect per-prompt transitions (`regressed` / `recovered`):

```python
from llm_drift_bench import evaluate_many, load_model_samples, load_prompt_cases, prompt_changes

reports = evaluate_many(
    load_model_samples("examples/baseline.jsonl"),
    [
        ("reference-like", load_model_samples("examples/candidate-pass.jsonl")),
        ("changed", load_model_samples("examples/candidate-drift.jsonl")),
    ],
    suite=load_prompt_cases("examples/extraction-suite.json"),
)
for change in prompt_changes(reports, against="reference-like"):
    print(change.candidate, change.prompt_id, change.state)
```

Expected output:

```text
changed invoice-01 regressed
changed ticket-01 regressed
```
