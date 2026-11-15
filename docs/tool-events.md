# Offline read-before-edit checks

Available in the Python API and the `check-tools` CLI command (not in published
0.1.0). This complements output comparison: matching final text does not establish that
an agent followed a required workflow. No live tracing is provided.

```python
from llm_drift_bench import ToolEvent, check_read_before_edit

events = [
    ToolEvent(0, "read", "guide@revision-1", start=0, end=1000),
    ToolEvent(1, "edit", "app.py@revision-1"),
]
result = check_read_before_edit(
    events,
    resource="guide@revision-1",
    required_start=0,
    required_end=45000,
    trace_complete=True,
)
assert result.status == "FAIL"
assert result.covered_characters == 1000
print(result.reason)
```

## Trace file format

`check-tools` reads normalized traces: a `.jsonl` file with one event per line, or a
`.json` array. Fields:

| Field | Required | Meaning |
| --- | --- | --- |
| `kind` | yes | `"read"` (a completed read) or `"edit"` (an edit invocation) |
| `resource` | yes | Exact revision identifier, for example `guide.md@sha256:1f2e` |
| `sequence` | all or none | Unique non-negative integer; if omitted everywhere, file order is used |
| `success` | no | Boolean, default `true`; failed reads give no coverage |
| `start`, `end` | reads only | Zero-based Unicode character offsets actually returned, end exclusive; give both or neither |

Unknown fields, duplicate keys, `NaN`, floats and strings where integers belong are
rejected with the offending line number rather than guessed at. See
`examples/tool-trace/trace.jsonl`.

This tool does not parse raw agent transcripts. Vendor logs differ, and most report
line ranges or truncated text, from which exact character coverage cannot be
derived without the file content. Write a small adapter that emits the records above,
and mark reads whose returned range you cannot establish by omitting `start` and
`end`.

## Command line

```console
$ llm-drift-bench check-tools -i trace.jsonl --resource guide.md@sha256:1f2e \
    --end 4000 --trace-complete -o evidence.json
read_before_edit: FAIL
Resource: guide.md@sha256:1f2e
Required: [0, 4000) (4000 characters); returned before first edit: 1000
First edit sequence: 2
Trace attested complete: yes
Reason: Required content was not fully returned before the first edit.
```

`--end` is the exclusive end of the required range (typically the revision's length)
and `--start` defaults to 0. `--trace-complete` is your attestation that the trace
holds every read and edit of the run; without it the result is `UNKNOWN`.

| Exit code | Meaning |
| --- | --- |
| 0 | PASS |
| 1 | FAIL |
| 3 | UNKNOWN: decide explicitly in your script; do not treat it as PASS |
| 2 | Error: unreadable or malformed trace, invalid range |

`-o` saves the verdict with the question it answered (resource, range, attestation).
Loading a saved file re-checks that the verdict agrees with its own evidence, so a
file edited to say `PASS` without full coverage or an attested trace is rejected.

## In reports

```console
$ llm-drift-bench run -b base.jsonl -c cand.jsonl --format html \
    --tool-check evidence.json > report.html
$ llm-drift-bench report -i report.json -f markdown --tool-check evidence.json
```

`--tool-check` is repeatable and works with markdown and html output of `run` and
`report`. It adds a separate **Tool evidence** section. It never changes the drift
verdict, pass/fail, or the exit code of `run`. Legacy reports cannot carry tool
evidence.

Adapters must supply read **completion** events and edit **invocation** events
with unique integer sequence numbers from one run. Do not order overlapping
operations only by their start time: a read must finish before editing starts.
An attempted edit counts even when it fails. Include every edit, including shell
and patch operations, or leave `trace_complete=False`.

Resources match by exact identifier. Include a content hash or revision and
normalize path aliases in your adapter. This checker does not open local files,
resolve symlinks, parse arbitrary shell commands, or verify collector integrity.
Do not mix runs or concatenate traces without preserving ordering and identity.

Read ranges are zero-based Unicode character offsets with exclusive ends,
measured against that exact resource revision. Record what was actually returned,
not the requested range or a tool's claimed byte count. Truncated or transformed
content whose offsets cannot be established must use unknown ranges. Failed
reads provide no coverage; repeated and overlapping ranges count only once.

Results:

- `PASS`: the complete required interval was returned before the first edit,
  and the caller attests to a complete trace.
- `FAIL`: a complete trace shows an edit before sufficient content was returned.
- `UNKNOWN`: trace completeness is unconfirmed, no edit occurred, or a successful
  read with unknown coverage could change the conclusion.

This is evidence of content delivery, not proof of comprehension or correctness.
The verdict is separate from composite output drift. CI callers should explicitly
decide whether UNKNOWN blocks their workflow; do not treat it as PASS.

Its `to_score()` adapter returns a local `EvidenceScore` dataclass. No vendored
SDK source is used or packaged. No SDK installation, tracing server, API key,
or network request is required for offline checks.

```python
score = result.to_score()
print(score.to_dict())
```

PASS maps to 1.0 and FAIL to 0.0. UNKNOWN sets `scoring_failed=True` and exports
`value: null` through `to_dict()`. Use `measured_value`, not the inherited numeric
placeholder, when aggregating. These are workflow evidence scores, not composite
drift scores or confidence estimates.

## Installed Opik integration

Install `llm-drift-bench[opik]` to use `result.to_score().to_opik()`.
The adapter imports `opik.evaluation.metrics.score_result` only when explicitly
called, and returns that installed SDK's `ScoreResult`. It does not send results
to a server. UNKNOWN raises `ValueError` before importing the SDK. Missing Opik
produces an installation hint; missing dependencies inside a broken SDK installation
are propagated rather than hidden by a fallback. The extra requires
`opik>=1.10.24,<3`: earlier releases do not accept the `category_name` argument.
CI runs `tests/test_opik_integration.py` against that minimum and the newest
release within the supported range. Re-test against your chosen version before deploying;
versions outside the
tested range are not supported.

## Vendor transcript adapters are pending

There is no Claude Code or OpenAI transcript importer yet. A real adapter needs
a sanitized sample log and a documented mapping from tool requests/results to
the normalized events above. Line ranges alone do not establish exact character
coverage without the corresponding file content and revision. An adapter must
keep unknown coverage unknown instead of estimating it as a successful read.
