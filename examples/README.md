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
