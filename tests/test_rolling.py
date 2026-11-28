"""Tests for rolling-baseline cross-window JSON field distribution drift analysis."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_drift_bench.models import ModelSample, PromptCase
from llm_drift_bench.policy import DriftPolicy, FieldRule
from llm_drift_bench.rolling import (
    analyze_rolling_distribution,
    export_rolling_html,
    format_rolling_markdown,
    format_rolling_text,
)


def _make_window(
    prefix: str,
    rows: list[tuple[str, float, str]],
) -> list[ModelSample]:
    return [
        ModelSample(
            prompt_id=f"{prefix}_{i}",
            model_name="prod",
            output=json.dumps({"label": lbl, "score": score, "req_id": f"{prefix}-{i}"}),
            metadata={"route": route, "category": "classification"},
        )
        for i, (lbl, score, route) in enumerate(rows)
    ]


def test_rolling_baseline_jitter_and_shift_detection(tmp_path: Path) -> None:
    # Three prior windows with natural slight variation (~0.10 TVD jitter)
    win1 = _make_window(
        "d1",
        [("high", 0.90, "r1"), ("high", 0.88, "r1"), ("low", 0.20, "r2"), ("high", 0.85, "r2")],
    )
    win2 = _make_window(
        "d2",
        [("high", 0.91, "r1"), ("high", 0.86, "r1"), ("low", 0.22, "r2"), ("low", 0.25, "r2")],
    )
    win3 = _make_window(
        "d3",
        [("high", 0.89, "r1"), ("high", 0.87, "r1"), ("low", 0.21, "r2"), ("high", 0.84, "r2")],
    )
    # Candidate has DIFFERENT prompt_ids and sample count, plus real shift across both routes!
    cand_shifted = _make_window(
        "d4",
        [
            ("low", 0.15, "r1"),
            ("low", 0.12, "r1"),
            ("low", 0.18, "r1"),
            ("low", 0.10, "r2"),
            ("low", 0.14, "r2"),
            ("low", 0.16, "r2"),
        ],
    )
    policy = DriftPolicy(field_rules={"$.req_id": FieldRule(path="$.req_id", ignore=True)})

    rep = analyze_rolling_distribution(
        [win1, win2, win3],
        cand_shifted,
        window_labels=["mon", "tue", "wed"],
        group_by="metadata:route",
        policy=policy,
        shift_threshold=0.15,
        max_tvd=0.30,
        fail_on_widespread_shift=True,
        max_jitter_multiplier=2.0,
    )
    assert rep.passed is False
    assert rep.window_count == 3
    assert rep.baseline_sample_count == 12
    assert rep.candidate_sample_count == 6
    assert rep.baseline_jitter_tvd == pytest.approx(0.25)
    assert rep.overall.max_tvd > 0.60
    assert rep.segmented is not None
    assert rep.segmented.shift_scope == "widespread"
    assert set(rep.segmented.shifted_segments) == {"r1", "r2"}
    assert any("Widespread rolling shift" in msg for msg in rep.failure_reasons)
    assert any("rolling baseline jitter" in msg for msg in rep.failure_reasons)

    as_dict = rep.to_dict()
    assert as_dict["passed"] is False
    assert as_dict["window_labels"] == ["mon", "tue", "wed"]
    assert "segmented" in as_dict

    txt = format_rolling_text(rep)
    assert "Rolling Baseline Distribution Drift [FAILED]" in txt
    assert "Threshold failures:" in txt
    md = format_rolling_markdown(rep)
    assert "# Rolling Baseline Distribution Drift: FAILED" in md
    assert "| mon |" in md and "## Segments" in md

    html_out = tmp_path / "dash" / "rolling.html"
    html = export_rolling_html(rep, output_path=html_out)
    assert html_out.read_text(encoding="utf-8") == html
    assert "Rolling Baseline Distribution Drift" in html
    assert "FAILED" in html and "Threshold Failures" in html
    assert "<script" not in html.lower()


def test_rolling_baseline_single_window_category_and_passing_case() -> None:
    win1 = [
        ModelSample(prompt_id="p1", output='{"tier": "gold", "val": 10}'),
        ModelSample(prompt_id="p2", output='{"tier": "gold", "val": 12}'),
    ]
    cand = [
        ModelSample(prompt_id="p1", output='{"tier": "gold", "val": 10}'),
        ModelSample(prompt_id="p2", output='{"tier": "gold", "val": 12}'),
    ]
    cases = [
        PromptCase(prompt_id="p1", category="billing", prompt="a"),
        PromptCase(prompt_id="p2", category="billing", prompt="b"),
    ]
    rep = analyze_rolling_distribution(
        [win1],
        cand,
        group_by="category",
        prompt_cases=cases,
        max_tvd=0.05,
        max_jsd=0.05,
        max_wasserstein=0.05,
    )
    assert rep.passed is True
    assert rep.baseline_jitter_tvd is None
    assert rep.baseline_jitter_wasserstein is None
    assert rep.segmented is not None and "billing" in rep.segmented.segments
    assert "PASSED" in format_rolling_text(rep)
    assert "PASSED" in format_rolling_markdown(rep)


def test_rolling_baseline_validation_and_threshold_edges() -> None:
    s_ok = [ModelSample(prompt_id="p1", output='{"a": "x", "n": 1}')]
    s_shift = [ModelSample(prompt_id="p9", output='{"a": "y", "n": 10}')]
    s_mid = [ModelSample(prompt_id="p2", output='{"a": "x", "n": 2}')]

    # Overall threshold failures without group_by + jitter wasserstein failure
    rep_fail = analyze_rolling_distribution(
        [s_ok, s_mid],
        s_shift,
        max_tvd=0.1,
        max_jsd=0.1,
        max_wasserstein=0.1,
        max_jitter_multiplier=0.5,
    )
    assert rep_fail.passed is False
    assert len(rep_fail.failure_reasons) >= 3

    # Segment jsd / wasserstein failure + non-JSON skipped cleanly
    s_text = [ModelSample(prompt_id="txt", output="not json")]
    rep_non_json = analyze_rolling_distribution([s_text], s_text)
    assert rep_non_json.overall.comparable_json_prompts == 0

    rep_seg = analyze_rolling_distribution(
        [s_ok],
        s_shift,
        group_by="metadata:route",
        max_jsd=0.1,
        max_wasserstein=0.1,
    )
    assert rep_seg.passed is False
    assert rep_seg.segmented is not None and "(unassigned)" in rep_seg.segmented.segments

    # Input validation guards
    with pytest.raises(ValueError, match="at least one window"):
        analyze_rolling_distribution([], s_ok)
    with pytest.raises(ValueError, match="candidate_samples"):
        analyze_rolling_distribution([s_ok], [])
    with pytest.raises(ValueError, match="window #1 is empty"):
        analyze_rolling_distribution([[]], s_ok)
    with pytest.raises(ValueError, match="window_labels length"):
        analyze_rolling_distribution([s_ok], s_ok, window_labels=["a", "b"])
    with pytest.raises(ValueError, match="unique non-empty"):
        analyze_rolling_distribution([s_ok, s_ok], s_ok, window_labels=["w", "w"])
    with pytest.raises(ValueError, match="shift_threshold"):
        analyze_rolling_distribution([s_ok], s_ok, shift_threshold=0.0)
    with pytest.raises(ValueError, match="max_jitter_multiplier"):
        analyze_rolling_distribution([s_ok], s_ok, max_jitter_multiplier=0.0)
    with pytest.raises(ValueError, match="requires group_by"):
        analyze_rolling_distribution([s_ok], s_ok, fail_on_widespread_shift=True)
    with pytest.raises(ValueError, match="between 0.0 and 1.0"):
        analyze_rolling_distribution([s_ok], s_ok, max_tvd=2.0)

    # Fallback segment resolution: sample.metadata["category"] when suite omitted,
    # and PromptCase.metadata[key] when sample.metadata omits the key
    from llm_drift_bench.models import PromptCase

    cat_sample = [
        ModelSample(prompt_id="p1", output='{"label": "a"}', metadata={"category": "billing"})
    ]
    rep_cat = analyze_rolling_distribution([cat_sample], cat_sample, group_by="category")
    assert rep_cat.segmented is not None and "billing" in rep_cat.segmented.segments

    plain_sample = [ModelSample(prompt_id="p1", output='{"label": "a"}')]
    cases = [PromptCase(prompt_id="p1", prompt="x", metadata={"tenant": "acme"})]
    rep_meta = analyze_rolling_distribution(
        [plain_sample], plain_sample, group_by="metadata:tenant", prompt_cases=cases
    )
    assert rep_meta.segmented is not None and "acme" in rep_meta.segmented.segments


def test_export_rolling_pr_comment_and_cli(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from pathlib import Path

    from llm_drift_bench.cli import main
    from llm_drift_bench.rolling import export_rolling_pr_comment

    s_a = [ModelSample("p1", '{"label": "a"}', metadata={"route": "r1"})]
    s_b = [ModelSample("p2", '{"label": "b"}', metadata={"route": "r1"})]
    pass_md = export_rolling_pr_comment(analyze_rolling_distribution([s_a], s_a))
    assert "<!-- llm-drift-bench:rolling-distribution -->" in pass_md
    assert "**PASS**" in pass_md and "<details>" not in pass_md

    fail_rep = analyze_rolling_distribution(
        [s_a, s_a], s_b, group_by="metadata:route", max_tvd=0.1
    )
    fail_md = export_rolling_pr_comment(fail_rep)
    assert "**FAIL**" in fail_md and "Blast radius" in fail_md and "<details>" in fail_md

    examples = Path(__file__).resolve().parents[1] / "examples"
    assert (
        main(
            [
                "rolling-distribution",
                "-w",
                str(examples / "baseline.jsonl"),
                "-c",
                str(examples / "candidate-pass.jsonl"),
                "--format",
                "pr-comment",
            ]
        )
        == 0
    )
    assert "### Rolling-Baseline Distribution Drift: **PASS**" in capsys.readouterr().out

