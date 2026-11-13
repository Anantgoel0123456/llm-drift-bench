# Moving to 0.2.0

0.2.0 is in development. Installing 0.1.0 does not provide these changes.

| Area | Development behavior | Migration |
| --- | --- | --- |
| Exit codes | 0 pass, 1 drift, 2 input error | Handle failed comparisons explicitly |
| Prompt matching | Exact ID sets; no duplicates | Capture every expected prompt on both sides |
| File inputs | Missing files error | Supply captured outputs, not model identifiers |
| Missing measurements | null or n/a | Record real timings and token counts |
| Edit distance | Unicode characters, maximum 16,384 each | Recalibrate thresholds with representative captures |
| Schema subset | Unsupported keywords fail | Review schemas before upgrading |
| Validation | Empty input errors; {} overrides global schema | Supply actual samples and intentional schemas |

Use `--no-fail-on-drift` for informational runs. Reports still record failure.
Keep raw captures and record the benchmark version alongside results. A scoring
change does not prove that the underlying model improved.

Use `run --format json --output report.json` for canonical engine results.
Every CLI run format now renders the same engine result. Rich/Markdown scores
therefore change from the previous alternate-evaluator display, not from a new
engine scoring formula. HTML exports are available for engine reports.

`load_report` and `load_benchmark_result` now return `DriftBenchmarkReport` for
engine files and `BenchmarkResult` for legacy evaluator files. Use `isinstance`
when consuming both. Incomplete historical files that previously received
invented defaults now raise `ValueError`. Complete old engine reports may lack
thresholds; these remain unavailable rather than assuming defaults.

Saved numeric values retain the existing four-decimal score/two-decimal latency
serialization. Rendering does not recompute pass/fail decisions from rounded
scores. New engine reports include the thresholds that produced the decision.
