# Current limitations

Applies to the 0.2.0 development checkout.

## Numeric output policy

Output parsing rejects `NaN`, `Infinity`, and `-Infinity`, which are not JSON
number literals. It also rejects otherwise legal literals such as `1e400`
whose values exceed the finite floating-point range supported by our metrics.
This range restriction is a library policy, not a claim that those literals
violate JSON grammar. The same checks apply to plain, fenced, and prose-wrapped
outputs. A rejected candidate fails validation and has no JSON numeric score.

Direct calls to `json_numeric_drift` raise `ValueError` for nonfinite or
out-of-range numbers, including numbers at unmatched paths. Numeric metrics
use floating-point arithmetic, so integer differences above floating-point
precision may be rounded. Exact field matching still compares parsed integers
without converting them to floats. These metrics are not exact financial arithmetic.

## Output contract

A captured `output` (or `response`) must be a JSON string. Objects, arrays, numbers,
booleans and `null` are rejected, never converted to Python text, because converting
would change what the text metrics compare and could turn `null` into the string
`None`. If your logger stores parsed JSON, call `json.dumps` on it before capture.
Exact text metrics depend on the serialization you choose (spacing, key order), so
use the same serializer for baseline and candidate.

Duplicate object keys are rejected wherever JSON is parsed: dataset files, saved
reports, and every output-extraction mode. The JSON Schema `integer` type accepts
finite numbers without a fractional part, so `1.0` and `1e2` conform while `1.5`
and `true` do not. Schema limits `minLength`, `maxLength`, `minItems`, and
`maxItems` also accept non-negative integral floats such as `3.0`. Fractional,
negative, nonfinite, boolean, and string values are rejected.

## Saved reports

`load_report` rejects reports whose parts disagree, including out-of-range scores,
means that do not match the per-prompt results, and a passing verdict that
contradicts recorded thresholds. Checks allow for the four-decimal score and
two-decimal latency rounding used when reports are saved. Conformity rates are
checked for plausibility against per-prompt outcomes; the file does not record
which prompts had schemas, so the exact denominator cannot be recomputed.

## Evaluation scope

Dataset, schema and saved-report files must be UTF-8; an initial UTF-8 BOM and
Windows line endings are accepted. Each file has a 16 MiB byte limit, enforced
during reading. There is no separate record-count limit. The file byte cap bounds
the number of records that can fit. For JSONL, record errors identify the physical
line, including blank lines in the count. For JSON arrays and wrappers, record
numbers start at one. Syntax errors include the decoder line and column. Duplicate JSON keys and nonfinite
numbers in these files are rejected rather than silently overwritten or accepted.

Supported schema keywords are checked for valid shapes before evaluating data,
including rules for absent properties. Invalid or unsupported regular expressions
are reported as schema issues before matching, including in absent properties.

Patterns use a conservative subset of Python regex: literals, character classes,
dot, anchors, escaped punctuation, and `\d`, `\D`, `\s`, `\S`, `\w`, `\W`.
At most one `*` or `+` repetition is allowed, and that expression must start with
`^`. Groups, alternation, backreferences, lookarounds, counted repetitions,
optional repetitions, and lazy/possessive repetitions are rejected explicitly.
For example, `^INV-\d+$` is supported; `(a+)+$` and `a+a+$` are not.
Patterns are limited to 256 characters; matching input is limited to 16,384
Unicode characters. Larger input produces a `pattern_limit` issue, without
truncation. This deliberately restricts compatibility rather than running
arbitrary Python regular expressions with unbounded backtracking. It is not a
general-purpose JSON Schema regex implementation.

Metric paths retain dot notation for simple ASCII identifier keys and use JSON-quoted
brackets for other object keys, for example `$["a.b"]` versus `$.a.b`.
This prevents literal keys from colliding with nested fields. Saved older reports
are not recalculated; comparisons involving special keys may change on rerun.

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
