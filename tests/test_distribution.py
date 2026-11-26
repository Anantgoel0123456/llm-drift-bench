"""Tests for cross-prompt JSON field distribution drift analysis."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_drift_bench.distribution import (
    analyze_field_distributions,
    analyze_segmented_distributions,
    export_distribution_html,
    export_segmented_distribution_html,
    format_distribution_markdown,
    format_distribution_text,
    format_segmented_distribution_markdown,
    format_segmented_distribution_text,
)
from llm_drift_bench.models import ModelSample, PromptCase
from llm_drift_bench.policy import DriftPolicy, FieldRule


def _samples(payloads: list[dict[str, object]], model: str) -> list[ModelSample]:
    return [
        ModelSample(
            prompt_id=f"p{i}",
            output=json.dumps(payload),
            model_name=model,
        )
        for i, payload in enumerate(payloads)
    ]


def test_distribution_catches_label_flatten_and_numeric_score_slide() -> None:
    # All outputs are valid JSON with identical keys/types, yet:
    # - $.label slides from 90% "high" / 10% "low" -> 50% "high" / 50% "low" (TVD = 0.40)
    # - $.score slides from near top [0.85..0.95] -> middle [0.45..0.55]
    base_payloads = [
        {"label": "high" if i < 9 else "low", "score": 0.85 + (i % 3) * 0.05}
        for i in range(10)
    ]
    cand_payloads = [
        {"label": "high" if i < 5 else "low", "score": 0.45 + (i % 3) * 0.05}
        for i in range(10)
    ]
    report = analyze_field_distributions(
        _samples(base_payloads, "base"),
        _samples(cand_payloads, "cand"),
    )
    assert report.total_prompts == 10
    assert report.comparable_json_prompts == 10
    assert len(report.categorical_fields) == 1
    assert len(report.numeric_fields) == 1

    cat = report.categorical_fields[0]
    assert cat.path == "$.label"
    assert cat.tvd == pytest.approx(0.4, abs=1e-6)
    assert 0.1 < cat.jsd < 0.3
    assert cat.baseline_shares["high"] == pytest.approx(0.9)
    assert cat.candidate_shares["high"] == pytest.approx(0.5)
    assert cat.top_shifts[0][0] == "high"
    assert cat.top_shifts[0][3] == pytest.approx(-0.4)

    num = report.numeric_fields[0]
    assert num.path == "$.score"
    assert num.mean_delta == pytest.approx(-0.4, abs=1e-6)
    assert num.median_delta == pytest.approx(-0.4, abs=1e-6)
    assert num.wasserstein_1d == pytest.approx(0.4, abs=1e-6)
    assert num.normalized_wasserstein == pytest.approx(0.8, abs=1e-6)

    assert report.max_tvd == pytest.approx(0.4, abs=1e-6)
    assert report.max_normalized_wasserstein == pytest.approx(0.8, abs=1e-6)
    as_dict = report.to_dict()
    assert as_dict["categorical_fields"][0]["path"] == "$.label"
    assert as_dict["numeric_fields"][0]["path"] == "$.score"


def test_distribution_identical_captures_yield_zero_drift() -> None:
    payloads = [
        {"ok": True, "note": None, "val": 10},
        {"ok": False, "note": "x", "val": 10},
    ]
    report = analyze_field_distributions(
        _samples(payloads, "base"),
        _samples(payloads, "cand"),
    )
    assert report.max_tvd == 0.0
    assert report.max_jsd == 0.0
    assert report.max_normalized_wasserstein == 0.0


def test_distribution_respects_policy_ignores_and_collapses_array_indices() -> None:
    base_payloads = [
        {"request_id": "a1", "items": [{"tag": "x", "qty": 1}, {"tag": "x", "qty": 2}]},
        {"request_id": "a2", "items": [{"tag": "x", "qty": 1}, {"tag": "y", "qty": 2}]},
    ]
    cand_payloads = [
        {"request_id": "b1", "items": [{"tag": "y", "qty": 10}, {"tag": "y", "qty": 20}]},
        {"request_id": "b2", "items": [{"tag": "y", "qty": 10}, {"tag": "y", "qty": 20}]},
    ]
    policy = DriftPolicy(field_rules={"$.request_id": FieldRule(path="$.request_id", ignore=True)})
    collapsed = analyze_field_distributions(
        _samples(base_payloads, "base"),
        _samples(cand_payloads, "cand"),
        policy=policy,
        collapse_array_indices=True,
    )
    cat_paths = [c.path for c in collapsed.categorical_fields]
    num_paths = [n.path for n in collapsed.numeric_fields]
    assert "$.request_id" not in cat_paths
    assert cat_paths == ["$.items[*].tag"]
    assert num_paths == ["$.items[*].qty"]
    assert collapsed.categorical_fields[0].baseline_count == 4

    uncollapsed = analyze_field_distributions(
        _samples(base_payloads, "base"),
        _samples(cand_payloads, "cand"),
        policy=policy,
        collapse_array_indices=False,
    )
    assert [c.path for c in uncollapsed.categorical_fields] == [
        "$.items[0].tag",
        "$.items[1].tag",
    ]


def test_distribution_formatters_and_html_escape_safety(tmp_path: Path) -> None:
    base = _samples([{"cat": "<script>alert(1)</script>|pipe", "score": 1.0}], "b")
    cand = _samples([{"cat": "safe", "score": 2.0}], "c")
    report = analyze_field_distributions(base, cand)

    text = format_distribution_text(report)
    assert "$.cat" in text and "$.score" in text

    md = format_distribution_markdown(report)
    assert "&#124;" in md
    assert "<script>" not in md

    out_html = tmp_path / "dist.html"
    html = export_distribution_html(report, out_html)
    assert out_html.read_text(encoding="utf-8") == html
    assert "default-src 'none'" in html
    assert "<script>" not in html.lower()
    assert "&lt;script&gt;" in html


def test_distribution_validation_errors_and_non_json_outputs() -> None:
    s1 = [ModelSample(prompt_id="p1", output="plain text", model_name="b")]
    s2 = [ModelSample(prompt_id="p1", output="other text", model_name="c")]
    empty_report = analyze_field_distributions(s1, s2)
    assert empty_report.comparable_json_prompts == 0
    assert empty_report.max_tvd == 0.0
    assert "No shared JSON leaf fields" in format_distribution_text(empty_report)

    with pytest.raises(ValueError, match="non-empty"):
        analyze_field_distributions([], s2)
    with pytest.raises(ValueError, match="do not match"):
        analyze_field_distributions(
            s1,
            [ModelSample(prompt_id="p2", output="x", model_name="c")],
        )


def test_segmented_distribution_isolates_single_route_shift_before_totals(tmp_path: Path) -> None:
    # route_a shifts 100% ("high" -> "low"), while route_b and route_c stay completely unchanged.
    # Global TVD is diluted to 1/3 (0.333), whereas segmenting by metadata:route isolates TVD=1.0!
    base = [
        ModelSample(
            prompt_id="a1", output='{"label": "high", "score": 0.9}',
            model_name="b", metadata={"route": "route_a"},
        ),
        ModelSample(
            prompt_id="a2", output='{"label": "high", "score": 0.9}',
            model_name="b", metadata={"route": "route_a"},
        ),
        ModelSample(
            prompt_id="b1", output='{"label": "high", "score": 0.9}',
            model_name="b", metadata={"route": "route_b"},
        ),
        ModelSample(
            prompt_id="b2", output='{"label": "high", "score": 0.9}',
            model_name="b", metadata={"route": "route_b"},
        ),
        ModelSample(
            prompt_id="c1", output='{"label": "high", "score": 0.9}',
            model_name="b", metadata={"route": "route_c"},
        ),
        ModelSample(
            prompt_id="c2", output='{"label": "high", "score": 0.9}',
            model_name="b", metadata={"route": "route_c"},
        ),
    ]
    cand = [
        ModelSample(
            prompt_id="a1", output='{"label": "low", "score": 0.2}',
            model_name="c", metadata={"route": "route_a"},
        ),
        ModelSample(
            prompt_id="a2", output='{"label": "low", "score": 0.2}',
            model_name="c", metadata={"route": "route_a"},
        ),
        ModelSample(
            prompt_id="b1", output='{"label": "high", "score": 0.9}',
            model_name="c", metadata={"route": "route_b"},
        ),
        ModelSample(
            prompt_id="b2", output='{"label": "high", "score": 0.9}',
            model_name="c", metadata={"route": "route_b"},
        ),
        ModelSample(
            prompt_id="c1", output='{"label": "high", "score": 0.9}',
            model_name="c", metadata={"route": "route_c"},
        ),
        ModelSample(
            prompt_id="c2", output='{"label": "high", "score": 0.9}',
            model_name="c", metadata={"route": "route_c"},
        ),
    ]

    seg = analyze_segmented_distributions(base, cand, group_by="metadata:route")
    assert seg.worst_segment_name == "route_a"
    assert seg.worst_segment_tvd == pytest.approx(1.0)
    assert seg.segments["route_a"].max_tvd == pytest.approx(1.0)
    assert seg.segments["route_b"].max_tvd == 0.0
    assert seg.segments["route_c"].max_tvd == 0.0
    assert seg.overall.max_tvd == pytest.approx(1 / 3, abs=1e-4)
    assert seg.shift_scope == "localized"
    assert seg.shifted_segments == ("route_a",)
    assert seg.to_dict()["worst_segment_name"] == "route_a"
    assert seg.to_dict()["shift_scope"] == "localized"

    txt = format_segmented_distribution_text(seg)
    assert "[route_a]" in txt and "Overall totals:" in txt and "Blast radius : localized" in txt
    md = format_segmented_distribution_markdown(seg)
    assert "| route&#95;a |" in md and "## Overall totals" in md and "**localized**" in md
    html_file = tmp_path / "seg.html"
    html = export_segmented_distribution_html(seg, html_file)
    assert html_file.read_text(encoding="utf-8") == html
    assert "default-src 'none'" in html and "<script" not in html.lower()
    assert "<strong>localized</strong>" in html

    # All routes shifting together -> widespread (likely model/provider shift)
    cand_all_shifted = [
        ModelSample(
            prompt_id=s.prompt_id,
            output='{"label": "low", "score": 0.2}',
            model_name="c",
            metadata=dict(s.metadata),
        )
        for s in base
    ]
    wide = analyze_segmented_distributions(base, cand_all_shifted, group_by="metadata:route")
    assert wide.shift_scope == "widespread"
    assert set(wide.shifted_segments) == {"route_a", "route_b", "route_c"}

    # No route shifting -> none
    clean_seg = analyze_segmented_distributions(base, base, group_by="metadata:route")
    assert clean_seg.shift_scope == "none" and clean_seg.shifted_segments == ()

    # 2 of 4 routes shifting -> partial
    base4 = base + [
        ModelSample(
            prompt_id="d1", output='{"label": "high", "score": 0.9}',
            model_name="b", metadata={"route": "route_d"},
        ),
    ]
    cand4 = cand + [
        ModelSample(
            prompt_id="d1", output='{"label": "low", "score": 0.2}',
            model_name="c", metadata={"route": "route_d"},
        ),
    ]
    partial = analyze_segmented_distributions(base4, cand4, group_by="metadata:route")
    assert partial.shift_scope == "partial"
    assert set(partial.shifted_segments) == {"route_a", "route_d"}

    # Single segment only -> single_segment
    one = analyze_segmented_distributions(base[:2], cand[:2], group_by="metadata:route")
    assert one.shift_scope == "single_segment"

    with pytest.raises(ValueError, match="shift_threshold"):
        analyze_segmented_distributions(base, cand, group_by="metadata:route", shift_threshold=0.0)

    # Category grouping via prompt_cases
    cases = [
        PromptCase(
            prompt_id=s.prompt_id,
            prompt="p",
            category="vip" if s.prompt_id.startswith("a") else "std",
        )
        for s in base
    ]
    by_cat = analyze_segmented_distributions(base, cand, group_by="category", prompt_cases=cases)
    assert by_cat.worst_segment_name == "vip"
    assert by_cat.segments["std"].max_tvd == 0.0

    with pytest.raises(ValueError, match="Unsupported group_by"):
        analyze_segmented_distributions(base, cand, group_by="unknown")
    with pytest.raises(ValueError, match="non-empty metadata key"):
        analyze_segmented_distributions(base, cand, group_by="metadata:")
    with pytest.raises(ValueError, match="category"):
        analyze_segmented_distributions(base, cand, group_by="   ")
    with pytest.raises(ValueError, match="non-empty"):
        analyze_segmented_distributions([], cand, group_by="category")
    with pytest.raises(ValueError, match="cannot be empty"):
        analyze_segmented_distributions(base, cand, group_by="category", prompt_cases=[])

    # Category from sample.metadata without prompt_cases & metadata fallback to prompt_cases
    m_base = [ModelSample(prompt_id="p1", output='{"a": "x"}', metadata={"category": "billing"})]
    m_cand = [ModelSample(prompt_id="p1", output='{"a": "x"}', metadata={"category": "billing"})]
    by_meta_cat = analyze_segmented_distributions(m_base, m_cand, group_by="category")
    assert "billing" in by_meta_cat.segments

    c_cases = [PromptCase(prompt_id="p1", prompt="hi", metadata={"tenant": "acme"})]
    by_case_meta = analyze_segmented_distributions(
        m_base, m_cand, group_by="metadata:tenant", prompt_cases=c_cases
    )
    assert "acme" in by_case_meta.segments


