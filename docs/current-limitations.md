# Current limitations

Applies to the 0.2.0 development checkout.


Levenshtein similarity always compares characters (Unicode code points),
including whitespace and case. Each response is limited to 16,384 characters
for this metric. Longer responses raise an error (CLI exit code `2`), rather
than being truncated or scored using a different unit. Unicode normalization
is not applied.

Commands are `run`, `report`, `validate`, and `init-sample`.
`--suite` and `--prompts` are aliases for the suite path.

Use existing captured-output files for both comparison targets.
Missing files, directory paths, and model identifiers are input errors
(CLI exit code `2`). The tool does not call model endpoints or generate
responses. The library also accepts explicit sequences of `ModelSample`.
`init-sample` creates illustrative fixtures, not measured model outputs.

Baseline and candidate must contain exactly the same non-empty prompt IDs,
with no duplicates. An explicitly supplied suite must cover exactly those
IDs and cannot be empty. The `validate --prompts` command applies the same
suite-coverage rule. Matching errors return CLI exit code `2`.
Comparison results are ordered by prompt ID, independently of input order.
For library callers, `evaluate_suite([], baseline, candidate)` still means
infer the prompts from the complete matching output sets; it never selects
only their intersection.

All CLI run formats use the same engine report without reevaluation. The legacy
evaluate_suite library API retains different scoring semantics and must not be
compared directly with engine scores.

Missing measurements appear as `n/a` in reports and `null` in JSON.
Throughput requires an explicit `tokens_generated` count and positive measured
`latency_ms`; text token estimates are not substituted. Zero latency remains
a recorded value, but cannot produce throughput or a baseline-relative
percentage regression.

Suite means and latency percentiles require complete measurements for that
side of the comparison. If either side lacks timings, the latency regression
check is skipped; a passing result does not certify latency performance.
Other drift and conformity thresholds still apply.

Saved engine reports are loaded as engine reports, without invented semantic
or exact-match values. Reports missing required fields are rejected. Older
complete reports without thresholds retain unknown thresholds. Legacy evaluator
reports remain readable in JSON, Rich, and Markdown; HTML requires an engine
report. The engine does not store throughput or latency means, so its renderers
do not display them.
