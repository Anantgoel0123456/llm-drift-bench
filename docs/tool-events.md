# Offline read-before-edit checks

Available in the development Python API, not the CLI or published 0.1.0.
This complements output comparison: matching final text does not establish that
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
are propagated rather than hidden by a fallback. Test against your chosen Opik
version before deploying an integration; the extra is optional and not version-pinned.
