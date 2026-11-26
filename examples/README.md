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
| `conversations-baseline.jsonl` | Multi-turn baseline conversation transcripts (`conv-support-01`, `conv-triage-02`) for `check-conversations`. |
| `conversations-candidate-drift.jsonl` | Multi-turn candidate transcripts demonstrating turn-1 stability, turn-2 drift, and turn-3 refusal flip + JSON drop. |
| `case-study/` | End-to-end FP16 -> Q8 -> Q4 offline quantization case study fixtures (see [docs/case-study-quantization.md](../docs/case-study-quantization.md)). |
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

# Explain why the failing report failed & inspect exact field-level diffs -> exit 0
python -m llm_drift_bench diagnose -i examples/expected-drift.json
python -m llm_drift_bench diff -b examples/baseline.jsonl -c examples/candidate-drift.jsonl --only-changed

# Run pluggable scorers or export metrics to CSV
python -m llm_drift_bench score -b examples/baseline.jsonl -c examples/candidate-drift.jsonl
python -m llm_drift_bench report -i examples/expected-drift.json --format csv -o drift-metrics.csv
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
python -m llm_drift_bench history check --ledger demo-history.jsonl --window 1 --max-drift-increase 0.05
```

### 6. Multi-turn conversation trajectory drift
```bash
# Pre-flight validation of conversation transcripts & conversation_id alignment
python -m llm_drift_bench doctor \
  --baseline-conversations examples/conversations-baseline.jsonl \
  --candidate-conversations examples/conversations-candidate-drift.jsonl

# Detect turn-by-turn divergence, escalation slope, and refusal flips -> exit 1
python -m llm_drift_bench check-conversations \
  -b examples/conversations-baseline.jsonl \
  -c examples/conversations-candidate-drift.jsonl \
  --format html -o conversation-drift.html
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
