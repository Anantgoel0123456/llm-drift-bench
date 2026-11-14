# Offline read-before-edit checks

Available in the development Python API, not the CLI or published 0.1.0.
This complements output comparison: matching final text does not establish that
an agent followed a required workflow. No live tracing or Opik integration is provided.

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

This implementation is original code. The design review of Opik's structured
metric results and trajectory evaluator informed the scope; no Opik source was copied.
