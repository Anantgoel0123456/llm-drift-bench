"""Behavioral checks on read completions and edit invocations."""

from pathlib import Path

import pytest

from llm_drift_bench.tool_events import ToolEvent, check_read_before_edit


def test_overlapping_reads_count_once_and_input_order_does_not_matter() -> None:
    events = [
        ToolEvent(4, "edit", "code@v1"),
        ToolEvent(3, "read", "guide@v1", start=5, end=12),
        ToolEvent(1, "read", "guide@v1", start=0, end=7),
        ToolEvent(2, "read", "guide@v1", start=0, end=7),
    ]
    result = check_read_before_edit(
        events, resource="guide@v1", required_start=0, required_end=10, trace_complete=True,
    )
    assert result.status == "PASS"
    assert result.covered_characters == result.required_characters == 10
    assert result.first_edit_sequence == 4


@pytest.mark.parametrize("read", [
    ToolEvent(2, "read", "guide@v1", start=0, end=10),  # after edit
    ToolEvent(0, "read", "guide@v2", start=0, end=10),  # different revision
    ToolEvent(0, "read", "guide@v1", success=False, start=0, end=10),
    ToolEvent(0, "read", "guide@v1", start=0, end=5),
    ToolEvent(0, "read", "guide@v1", start=10, end=20),
])
def test_missing_coverage_fails_even_if_edit_itself_failed(read: ToolEvent) -> None:
    result = check_read_before_edit(
        [read, ToolEvent(1, "edit", "code@v1", success=False)], resource="guide@v1",
        required_start=0, required_end=10, trace_complete=True,
    )
    assert result.status == "FAIL"


def test_unknown_trace_or_range_never_becomes_a_false_pass() -> None:
    read = ToolEvent(0, "read", "guide@v1", start=0, end=10)
    edit = ToolEvent(1, "edit", "code@v1")
    for events, complete in [([read, edit], False), ([read], True),
                             ([ToolEvent(0, "read", "guide@v1"), edit], True)]:
        result = check_read_before_edit(
            events, resource="guide@v1", required_start=0, required_end=10,
            trace_complete=complete,
        )
        assert result.status == "UNKNOWN"


def test_disjoint_ranges_leave_a_gap() -> None:
    result = check_read_before_edit(
        [ToolEvent(0, "read", "g", start=0, end=4),
         ToolEvent(1, "read", "g", start=6, end=10), ToolEvent(2, "edit", "code")],
        resource="g", required_start=2, required_end=8, trace_complete=True,
    )
    assert result.status == "FAIL"
    assert result.covered_characters == 4


def test_duplicate_sequence_rejected() -> None:
    with pytest.raises(ValueError, match="unique"):
        check_read_before_edit(
            [ToolEvent(0, "read", "g"), ToolEvent(0, "edit", "code")],
            resource="g", required_start=0, required_end=10,
        )


@pytest.mark.parametrize(("start", "end"), [(0, 0), (-1, 3), (4, 2), (True, 3)])
def test_invalid_required_ranges_rejected(start: int, end: int) -> None:
    with pytest.raises(ValueError):
        check_read_before_edit([], resource="g", required_start=start, required_end=end)


def test_invalid_events_rejected() -> None:
    with pytest.raises(ValueError):
        ToolEvent(-1, "read", "g")
    with pytest.raises(ValueError):
        ToolEvent(0, "read", " ")
    with pytest.raises(ValueError):
        ToolEvent(0, "read", "g", start=0)
    with pytest.raises(ValueError):
        ToolEvent(0, "edit", "g", start=0, end=1)


def test_max_tool_calls_allowed_resources_and_jwt_db_secrets() -> None:
    from llm_drift_bench.tool_events import (
        check_allowed_resources,
        check_max_tool_calls,
        check_no_secret_in_tool_args,
    )

    events = [
        ToolEvent(0, "read", "src/app.py@1", start=0, end=10),
        ToolEvent(1, "edit", "src/app.py@1"),
        ToolEvent(2, "edit", "etc/passwd@1"),
    ]
    assert check_max_tool_calls(events, max_calls=2).status == "FAIL"
    assert check_max_tool_calls(events, max_calls=3, trace_complete=False).status == "UNKNOWN"
    assert check_max_tool_calls(events, max_calls=3, trace_complete=True).status == "PASS"

    assert check_allowed_resources(events, allowed_prefixes=["src/"]).status == "FAIL"
    assert (
        check_allowed_resources(events[:2], allowed_prefixes=["src/"], trace_complete=True).status
        == "PASS"
    )
    assert (
        check_allowed_resources(events[:2], allowed_prefixes=["src/"], trace_complete=False).status
        == "UNKNOWN"
    )

    jwt = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.signaturevalue12345"
    db = "postgresql://admin:s3cr3tpass@db.internal:5432/prod"
    sec = check_no_secret_in_tool_args([jwt, db])
    assert sec.status == "FAIL"
    assert "jwt_token" in sec.reason and "database_url" in sec.reason
    assert "s3cr3tpass" not in sec.reason

    with pytest.raises(ValueError, match="non-negative integer"):
        check_max_tool_calls(events, max_calls=True)
    with pytest.raises(ValueError, match="non-empty strings"):
        check_allowed_resources(events, allowed_prefixes=[])

    # Cover remaining validation guards & EvidenceScore bridge (100% tool_events.py)
    from llm_drift_bench.tool_events import check_max_tool_retries, check_read_before_edit

    bad_bool: object = "yes"
    with pytest.raises(ValueError, match="read or edit"):
        ToolEvent(0, "delete", "a@1")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="boolean"):
        ToolEvent(0, "read", "a@1", success=bad_bool)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="boolean"):
        check_read_before_edit(
            events,
            resource="a@1",
            required_start=0,
            required_end=10,
            trace_complete=bad_bool,  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="non-negative integer"):
        check_max_tool_retries(events, max_failed_calls=True)
    with pytest.raises(ValueError, match="boolean"):
        check_max_tool_retries(
            events, max_failed_calls=1, trace_complete=bad_bool  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="boolean"):
        check_no_secret_in_tool_args(["ok"], trace_complete=bad_bool)  # type: ignore[arg-type]
    assert check_no_secret_in_tool_args(["clean"], trace_complete=False).status == "UNKNOWN"
    with pytest.raises(ValueError, match="boolean"):
        check_max_tool_calls(
            events, max_calls=5, trace_complete=bad_bool  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="boolean"):
        check_allowed_resources(
            events, allowed_prefixes=["src/"], trace_complete=bad_bool  # type: ignore[arg-type]
        )
    score = check_max_tool_calls(events, max_calls=5, trace_complete=True).to_score()
    assert score.value == 1.0 and score.scoring_failed is False


def test_compare_tool_traces_detects_call_retry_and_edit_scope_drift() -> None:
    from llm_drift_bench.tool_events import compare_tool_traces, format_tool_trace_diff_text

    base = [
        ToolEvent(0, "read", "src/a.py@1", start=0, end=20),
        ToolEvent(1, "edit", "src/a.py@1"),
    ]
    cand = [
        ToolEvent(0, "read", "src/a.py@1", start=0, end=20),
        ToolEvent(1, "read", "src/b.py@1", start=0, end=10, success=False),
        ToolEvent(2, "edit", "src/a.py@1"),
        ToolEvent(3, "edit", "src/b.py@1"),
        ToolEvent(4, "edit", "src/c.py@1"),
        ToolEvent(5, "edit", "src/d.py@1"),
        ToolEvent(6, "edit", "src/e.py@1"),
    ]
    diff = compare_tool_traces(
        base,
        cand,
        max_call_increase=2,
        max_failed_increase=0,
        fail_on_new_edit_resources=True,
    )
    assert diff.passed is False
    assert diff.call_delta == 5 and diff.failed_delta == 1
    assert len(diff.added_edit_resources) == 4
    assert len(diff.failure_reasons) == 3
    assert "+1 more" in diff.failure_reasons[2]
    assert diff.to_dict()["passed"] is False

    txt = format_tool_trace_diff_text(diff)
    assert "[FAILED]" in txt and "New edit targets:" in txt

    # Identical traces & removed resource coverage
    same_diff = compare_tool_traces(base, base)
    assert same_diff.passed is True and same_diff.action_jaccard == 1.0
    assert "[PASSED]" in format_tool_trace_diff_text(same_diff)

    removed_diff = compare_tool_traces(base, [base[0]])
    assert removed_diff.removed_resources == ()  # still read src/a.py@1
    removed_other = compare_tool_traces(base, [ToolEvent(0, "read", "src/z.py@1")])
    assert "Removed         :" in format_tool_trace_diff_text(removed_other)

    # Validation errors
    with pytest.raises(ValueError, match="max_call_increase"):
        compare_tool_traces(base, base, max_call_increase=-1)
    with pytest.raises(ValueError, match="max_failed_increase"):
        compare_tool_traces(base, base, max_failed_increase=True)
    with pytest.raises(ValueError, match="fail_on_new_edit_resources"):
        compare_tool_traces(base, base, fail_on_new_edit_resources="yes")  # type: ignore[arg-type]


def test_tool_trace_diff_markdown_html_and_cli(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from llm_drift_bench.cli import main
    from llm_drift_bench.tool_events import (
        compare_tool_traces,
        export_tool_trace_diff_html,
        format_tool_trace_diff_markdown,
    )

    base = [
        ToolEvent(0, "read", "src/a.py@1", start=0, end=20),
        ToolEvent(1, "edit", "src/a.py@1"),
        ToolEvent(2, "read", "src/old.py@1", start=0, end=10),
    ]
    cand = [
        ToolEvent(0, "read", "src/a.py@1", start=0, end=20),
        ToolEvent(1, "edit", "src/new<x>.py@1", success=False),
    ]
    diff = compare_tool_traces(
        base,
        cand,
        max_failed_increase=0,
        fail_on_new_edit_resources=True,
    )
    md = format_tool_trace_diff_markdown(diff)
    assert "## Agent Tool-Trace Drift (FAIL)" in md
    assert "**Added resources:**" in md
    assert "**Removed resources:**" in md
    assert "**New edit targets:**" in md
    assert "### Failure Reasons" in md

    html = export_tool_trace_diff_html(diff)
    assert "<!DOCTYPE html>" in html and "<script" not in html.lower()
    assert "src/new&lt;x&gt;.py@1" in html
    assert "Failure Reasons" in html

    same_diff = compare_tool_traces(base, base)
    assert "## Agent Tool-Trace Drift (PASS)" in format_tool_trace_diff_markdown(same_diff)
    assert "Failure Reasons" not in export_tool_trace_diff_html(same_diff)

    base_file = tmp_path / "base.jsonl"
    cand_file = tmp_path / "cand.jsonl"
    base_file.write_text(
        '{"kind": "read", "resource": "src/a.py@1", "start": 0, "end": 10}\n',
        encoding="utf-8",
    )
    cand_file.write_text(
        '{"kind": "read", "resource": "src/a.py@1", "start": 0, "end": 10}\n'
        '{"kind": "edit", "resource": "src/b.py@1"}\n',
        encoding="utf-8",
    )
    md_out = tmp_path / "diff.md"
    html_out = tmp_path / "diff.html"
    assert (
        main(
            [
                "diff-tools",
                "-b",
                str(base_file),
                "-c",
                str(cand_file),
                "--format",
                "markdown",
                "-o",
                str(md_out),
            ]
        )
        == 0
    )
    assert "## Agent Tool-Trace Drift (PASS)" in md_out.read_text(encoding="utf-8")
    assert "## Agent Tool-Trace Drift (PASS)" in capsys.readouterr().out

    assert (
        main(
            [
                "diff-tools",
                "-b",
                str(base_file),
                "-c",
                str(cand_file),
                "--fail-on-new-edit-resources",
                "--format",
                "html",
                "-o",
                str(html_out),
            ]
        )
        == 1
    )
    assert "Wrote agent tool-trace drift report" in capsys.readouterr().out
    assert "<!DOCTYPE html>" in html_out.read_text(encoding="utf-8")


def test_shipped_example_tool_traces_detect_expected_drift() -> None:
    from llm_drift_bench.tool_events import compare_tool_traces
    from llm_drift_bench.tool_log import load_tool_events

    examples = Path(__file__).resolve().parents[1] / "examples"
    base = load_tool_events(examples / "trace-baseline.jsonl")
    drift = load_tool_events(examples / "trace-candidate-drift.jsonl")

    same = compare_tool_traces(base, base, fail_on_new_edit_resources=True)
    assert same.passed is True and same.action_jaccard == 1.0

    diff = compare_tool_traces(
        base,
        drift,
        max_failed_increase=0,
        fail_on_new_edit_resources=True,
    )
    assert diff.passed is False
    assert diff.call_delta == 2 and diff.failed_delta == 1
    assert diff.added_edit_resources == ("config/production.env@v1",)


def test_export_tool_trace_pr_comment_and_cli(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from llm_drift_bench.cli import main
    from llm_drift_bench.tool_events import compare_tool_traces, export_tool_trace_pr_comment
    from llm_drift_bench.tool_log import load_tool_events

    examples = Path(__file__).resolve().parents[1] / "examples"
    base = load_tool_events(examples / "trace-baseline.jsonl")
    pass_md = export_tool_trace_pr_comment(compare_tool_traces(base, base))
    assert "<!-- llm-drift-bench:tool-trace -->" in pass_md
    assert "**PASS**" in pass_md and "<details>" not in pass_md

    assert (
        main(
            [
                "diff-tools",
                "-b",
                str(examples / "trace-baseline.jsonl"),
                "-c",
                str(examples / "trace-candidate-drift.jsonl"),
                "--fail-on-new-edit-resources",
                "--format",
                "pr-comment",
            ]
        )
        == 1
    )
    fail_md = capsys.readouterr().out
    assert "**FAIL**" in fail_md and "config/production.env@v1" in fail_md



