<div align="center">

# llm-drift-bench

**Catch silent regressions before you swap or quantize your LLM.**

Compare captured outputs. Check structured responses. Gate your next model change.

[![PyPI](https://img.shields.io/pypi/v/llm-drift-bench)](https://pypi.org/project/llm-drift-bench/)
[![CI](https://github.com/anant-goel/llm-drift-bench/actions/workflows/ci.yml/badge.svg)](https://github.com/anant-goel/llm-drift-bench/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

[Quickstart](#quickstart) · [Examples](examples/README.md) · [Architecture](docs/architecture.md) · [Roadmap](docs/roadmap.md) · [Contribute](CONTRIBUTING.md)

</div>

> **Release status:** 0.1.0 has been released. This branch prepares **0.2.0**
> (`0.2.0.dev0`). The commands and behavior below describe this development
> checkout. Read the [migration notes](docs/migration-0.2.md) before upgrading.

## A faster model is only useful if its outputs still work

A quantized model can return valid JSON while changing a price, dropping a field,
or choosing a different label. A prompt revision can change output formatting
without improving the underlying task. A model update can affect both.

`llm-drift-bench` compares saved baseline and candidate responses by prompt ID
and makes those changes inspectable. It runs locally, with no model API calls,
API keys, embedding downloads, or GPU requirement. Its runtime dependencies are
Pydantic and Rich, plus their transitive dependencies.

| Check | What it tells you |
| --- | --- |
| JSON conformity | Whether responses parse and satisfy the supported schema subset |
| Text similarity | Token, n-gram, and character-level agreement with your baseline |
| JSON field agreement | Changes to structure, exact leaf values, and numeric values |
| Recorded latency | P50/P95 differences when complete measurements are supplied |
| CI gates | Whether a sample or suite exceeds your configured thresholds |

**Agreement is not correctness.** A wrong baseline can agree perfectly with a
wrong candidate. These metrics are lexical and structural, not a semantic judge.

## Quickstart

Try the development checkout from PowerShell with Python 3.10 or newer:

```powershell
git clone https://github.com/anant-goel/llm-drift-bench.git
Set-Location llm-drift-bench
python -m venv .venv
& .\.venv\Scripts\python.exe -m pip install -e .
& .\.venv\Scripts\python.exe -m llm_drift_bench --version
& .\.venv\Scripts\python.exe -m llm_drift_bench run -b examples/baseline.jsonl -c examples/candidate-pass.jsonl -p examples/extraction-suite.json
```

Expected: `0.2.0.dev0`, followed by a **PASSED** comparison and exit code `0`.
On Linux/macOS, replace `.\.venv\Scripts\python.exe` with `.venv/bin/python`.
No virtual-environment activation is needed.

Now catch a regression:

```powershell
& .\.venv\Scripts\python.exe -m llm_drift_bench run -b examples/baseline.jsonl -c examples/candidate-drift.jsonl -p examples/extraction-suite.json --output drift-report.json
$LASTEXITCODE # Expected: 1
```

These fixtures are **handwritten demonstrations**, not measurements of any real
model or quantization. [Inspect the inputs and expected reports](examples/README.md).

To install the published version instead:

```powershell
python -m pip install "llm-drift-bench==0.1.0"
python -m llm_drift_bench --help
```

Published 0.1.0 does not include all changes documented on this branch.

## Bring your own captures

Create a JSON or JSONL file for each comparison target. Each record contains a
unique, nonempty `prompt_id` and the captured response as an `output` string:

```json
{"prompt_id":"invoice-01","model_name":"my-baseline","output":"{\"total\":100}"}
```

Optionally include measured `latency_ms` and `tokens_generated`. Missing
measurements remain unavailable; the tool does not estimate them from text.
Baseline and candidate must have exactly matching prompt IDs. A supplied suite
must cover the same IDs, with no duplicates or extra cases.

A suite associates those IDs with prompts and schemas:

```json
[{"prompt_id":"invoice-01","prompt":"Extract the invoice total.","schema":{"type":"object","required":["total"],"properties":{"total":{"type":"number","minimum":0}}}}]
```

JSON Schema support is intentionally limited. Unsupported keywords fail
explicitly. See [current limitations](docs/current-limitations.md) before using
complex schemas or untrusted regex patterns.

## Use it in CI

```powershell
& .\.venv\Scripts\python.exe -m llm_drift_bench run -b baseline.jsonl -c candidate.jsonl -p prompts.json --max-drift 0.15 --min-conformity 0.98 --max-numeric-drift 0.05 --max-latency-regression 25 --format json --output drift-report.json
exit $LASTEXITCODE
```

| Exit code | Meaning |
| --- | --- |
| `0` | Passed |
| `1` | Drift or a threshold failure |
| `2` | Usage or input error |

`--fail-on-drift` explicitly enables the default. `--no-fail-on-drift` makes a
run informational, but its report still records failure. Input errors remain
errors. A failed individual sample fails the suite even if the mean is low.

| Threshold | Default |
| --- | ---: |
| `--max-drift` | 0.25 |
| `--min-conformity` | 0.95 |
| `--max-numeric-drift` | 0.15 |
| `--max-latency-regression` | 50 percent |

Calibrate thresholds using your own tasks. Missing timings skip the latency
gate, so a passing report does not necessarily certify latency performance.

## Python API

```python
from llm_drift_bench import DriftBenchmarkEngine, ModelSample, PromptCase

suite = [PromptCase(prompt_id="invoice-01", prompt="Extract total.",
                    schema={"type": "object", "required": ["total"]})]
baseline = [ModelSample(prompt_id="invoice-01", output='{"total":100}')]
candidate = [ModelSample(prompt_id="invoice-01", output='{"total":100}')]
report = DriftBenchmarkEngine().evaluate(baseline, candidate, suite)
assert report.passed, report.failure_reasons
```

## Commands and reporting

| Command | Purpose |
| --- | --- |
| `run` | Compare captured baseline and candidate outputs |
| `validate --strict` | Fail when captured outputs do not conform |
| `report` | Render an existing result |
| `init-sample --dir demo` | Create illustrative starter files |

Use `run --format json` or the default table for canonical engine scores.
Rich/Markdown currently use a second evaluator and can disagree with engine
scores. Unifying them is a **0.2.0 release blocker**, tracked in the roadmap.
For all options, run `python -m llm_drift_bench <command> --help`.

## Help make model changes safer

Useful contributions include a small failing capture, a reproducible bug report,
a tested edge-case fix, or clearer documentation. Start with the
[contributor guide](CONTRIBUTING.md) and [roadmap](docs/roadmap.md).

If the tool catches a real regression, share a sanitized reproduction and the
benchmark version. Reproducible examples help other users more than a score
without its inputs.

[Changelog](CHANGELOG.md) · [Migration guide](docs/migration-0.2.md) · [MIT license](LICENSE)
