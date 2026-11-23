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
