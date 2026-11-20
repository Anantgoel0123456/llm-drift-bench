# Reproducible offline demo

These handwritten fixtures demonstrate behavior. They are not measured outputs
from an actual model, and no timings or token counts are invented.

- `extraction-suite.json`: two required-field extraction cases.
- `baseline.jsonl`: the reference responses.
- `candidate-pass.jsonl`: identical response text under a different model label.
- `candidate-drift.jsonl`: a changed invoice total and a missing required label.
- `expected-pass.json` and `expected-drift.json`: engine reports from this checkout.

From the repository root after installing the development checkout:

```powershell
& .\.venv\Scripts\python.exe -m llm_drift_bench run -b examples/baseline.jsonl -c examples/candidate-pass.jsonl -p examples/extraction-suite.json
# Expected: PASSED, exit 0
& .\.venv\Scripts\python.exe -m llm_drift_bench run -b examples/baseline.jsonl -c examples/candidate-drift.jsonl -p examples/extraction-suite.json
# Expected: FAILED, exit 1
```

The original `demo_suite.json` remains a separate legacy suite. Do not mix its
prompt IDs with these extraction fixtures. The test suite checks the new
fixtures against their saved engine reports so changes remain reviewable.

## Use the batch API from Python

Run this example from the repository root. The two candidates share a baseline
and suite. Labels identify runs in the output and can differ from model names.

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

Expected rows are `changed invoice-01 regressed` and
`changed ticket-01 regressed`.
