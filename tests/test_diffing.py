"""Tests for semantic JSON/text diffing and the diff CLI subcommand."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_drift_bench.cli import main
from llm_drift_bench.diffing import diff_captures, diff_sample_pair, format_diffs_text
from llm_drift_bench.models import ModelSample
from llm_drift_bench.policy import DriftPolicy, FieldRule


def test_diff_sample_pair_json_changes_and_policy_filtering() -> None:
    base = ModelSample(
        prompt_id="inv-1",
        output=json.dumps({
            "total": 100.0,
            "priority": "high",
            "status": "paid",
            "score": 0.90,
            "timestamp": "2026-01-01",
            "tags": ["a", "b"],
        }),
    )
    cand = ModelSample(
        prompt_id="inv-1",
        output=json.dumps({
            "total": 180.0,
            "status": 200,
            "score": 0.92,
            "timestamp": "2026-11-23",
            "tags": ["b", "a"],
            "note": "extra",
        }),
    )
    policy = DriftPolicy(
        field_rules={
            "$.timestamp": FieldRule(path="$.timestamp", ignore=True),
            "$.score": FieldRule(path="$.score", tolerance_abs=0.05),
            "$.tags": FieldRule(path="$.tags", unordered_array=True),
        }
    )

    diff = diff_sample_pair(base, cand, policy=policy)
    assert diff.mode == "json"
    assert diff.changed is True
    assert diff.ignored_paths == ("$.timestamp",)

    by_path = {c.path: c for c in diff.field_changes}
    assert set(by_path) == {"$.note", "$.priority", "$.status", "$.total"}
    assert by_path["$.note"].kind == "added"
    assert by_path["$.priority"].kind == "removed"
    assert by_path["$.status"].kind == "type_changed"
    assert by_path["$.total"].kind == "changed"
    assert by_path["$.total"].numeric_rel_drift == pytest.approx(0.4444, abs=1e-4)

    rendered = format_diffs_text([diff])
    assert "+ $.note" in rendered
    assert "- $.priority" in rendered
    assert "type changed" in rendered
    assert "rel drift 0.4444" in rendered
    assert "ignored 1 path(s)" in rendered


def test_diff_sample_pair_plain_text_and_errors() -> None:
    b = ModelSample(prompt_id="p1", output="line one\nline two")
    c = ModelSample(prompt_id="p1", output="line one\nline TWO")
    diff = diff_sample_pair(b, c)
    assert diff.mode == "text"
    assert diff.changed is True
    assert any("-line two" in line for line in diff.unified_diff)
    assert any("+line TWO" in line for line in diff.unified_diff)

    same = diff_sample_pair(b, b)
    assert same.changed is False
    assert "[UNCHANGED] p1 (text)" in format_diffs_text([same])

    with pytest.raises(ValueError, match="Prompt ID mismatch"):
        diff_sample_pair(b, ModelSample(prompt_id="p2", output="x"))


def test_diff_captures_filters_and_validation() -> None:
    base = [
        ModelSample(prompt_id="p1", output='{"a": 1}'),
        ModelSample(prompt_id="p2", output='{"a": 2}'),
    ]
    cand = [
        ModelSample(prompt_id="p1", output='{"a": 1}'),
        ModelSample(prompt_id="p2", output='{"a": 9}'),
    ]
    only_changed = diff_captures(base, cand, only_changed=True)
    assert [d.prompt_id for d in only_changed] == ["p2"]

    filtered = diff_captures(base, cand, prompt_ids=["p1"])
    assert [d.prompt_id for d in filtered] == ["p1"]
    assert filtered[0].changed is False

    with pytest.raises(ValueError, match="Unknown prompt_id"):
        diff_captures(base, cand, prompt_ids=["missing"])
    with pytest.raises(ValueError, match="non-empty"):
        diff_captures([], cand)


def test_cli_diff_on_examples_text_and_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    examples = Path(__file__).resolve().parents[1] / "examples"
    base = examples / "baseline.jsonl"
    drift = examples / "candidate-drift.jsonl"

    assert main(["diff", "-b", str(base), "-c", str(drift), "--only-changed"]) == 0
    out = capsys.readouterr().out
    assert "Diff summary: 2 of 2 prompt(s) changed" in out
    assert "invoice-01" in out
    assert "ticket-01" in out

    out_json = tmp_path / "diff.json"
    rc = main([
        "diff",
        "-b",
        str(base),
        "-c",
        str(drift),
        "--prompt-id",
        "invoice-01",
        "--limit",
        "1",
        "--format",
        "json",
        "-o",
        str(out_json),
    ])
    assert rc == 0
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["total_matched"] == 1
    assert payload["changed_prompts"] == 1
    assert payload["diffs"][0]["prompt_id"] == "invoice-01"

    assert main(["diff", "-b", str(base), "-c", str(drift), "--limit", "0"]) == 2


def test_diff_empty_containers_do_not_report_spurious_root_removal() -> None:
    d_obj = diff_sample_pair(
        ModelSample(prompt_id="p1", output="{}"),
        ModelSample(prompt_id="p1", output='{"a": 1}'),
    )
    assert [(c.path, c.kind) for c in d_obj.field_changes] == [("$.a", "added")]
    assert d_obj.summary_counts() == {"added": 1, "removed": 0, "changed": 0, "type_changed": 0}

    d_arr = diff_sample_pair(
        ModelSample(prompt_id="p1", output="[10]"),
        ModelSample(prompt_id="p1", output="[]"),
    )
    assert [(c.path, c.kind) for c in d_arr.field_changes] == [("$[0]", "removed")]


def test_format_diffs_markdown_renders_json_tables_and_text_diffs() -> None:
    from llm_drift_bench.diffing import format_diffs_markdown

    d_json = diff_sample_pair(
        ModelSample(prompt_id="inv-1", output='{"total": 100, "note": "a|b"}'),
        ModelSample(prompt_id="inv-1", output='{"total": 150, "extra": true}'),
    )
    d_text = diff_sample_pair(
        ModelSample(prompt_id="txt-1", output="hello world"),
        ModelSample(prompt_id="txt-1", output="hello earth"),
    )
    d_same = diff_sample_pair(
        ModelSample(prompt_id="same-1", output='{"x": 1}'),
        ModelSample(prompt_id="same-1", output='{"x": 1}'),
    )
    md = format_diffs_markdown([d_json, d_text, d_same])
    assert "**Changed prompts:** `2/3`" in md
    assert "| `$.total` | **changed**" in md
    assert "a&#124;b" in md and "a|b" not in md  # pipe escaped safely inside GFM table
    assert "```diff" in md
    assert "Unchanged." in md

    with pytest.raises(ValueError, match="positive integer"):
        format_diffs_markdown([d_json], max_changes_per_prompt=0)

    # Empty lists, text diff formatting, string-value changes, omission & ignored notes
    assert "No matching prompts" in format_diffs_text([])
    assert "No matching prompts" in format_diffs_markdown([])
    assert "[CHANGED] txt-1 (text)" in format_diffs_text([d_text])

    pol = DriftPolicy(field_rules={"$.ts": FieldRule(path="$.ts", ignore=True)})
    d_str = diff_sample_pair(
        ModelSample(prompt_id="s1", output='{"a": "old", "b": "x", "ts": 1}'),
        ModelSample(prompt_id="s1", output='{"a": "new", "b": "y", "ts": 2}'),
        policy=pol,
    )
    md_trunc = format_diffs_markdown([d_str], max_changes_per_prompt=1)
    assert "additional field change(s) omitted" in md_trunc
    assert "Ignored 1 path(s) via policy" in md_trunc

    dup_s = ModelSample(prompt_id="p1", output='{"a": 1}')
    with pytest.raises(ValueError, match="duplicate prompt_id"):
        diff_captures([dup_s, dup_s], [dup_s])


