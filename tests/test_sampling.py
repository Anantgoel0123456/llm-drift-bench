"""Tests for deterministic stratified suite subsampling and capture slicing."""

from __future__ import annotations

import pytest

from llm_drift_bench.models import ModelSample, PromptCase
from llm_drift_bench.sampling import (
    filter_prompt_suite,
    sample_prompt_suite,
    slice_samples_for_suite,
)


def test_stratified_sampling_is_deterministic_and_preserves_categories() -> None:
    cases = (
        [PromptCase(prompt_id=f"ext-{i}", prompt="e", category="extraction") for i in range(6)]
        + [
            PromptCase(prompt_id=f"cls-{i}", prompt="c", category="classification")
            for i in range(3)
        ]
        + [PromptCase(prompt_id="code-0", prompt="x", category="coding")]
    )
    sub_a = sample_prompt_suite(cases, size=5, seed=42)
    sub_b = sample_prompt_suite(cases, size=5, seed=42)
    assert sub_a == sub_b
    assert len(sub_a) == 5
    cats = {c.category for c in sub_a}
    assert cats == {"extraction", "classification", "coding"}

    unstrat = sample_prompt_suite(cases, size=4, seed=7, stratify_by_category=False)
    assert len(unstrat) == 4
    assert len(sample_prompt_suite(cases, size=10)) == 10

    with pytest.raises(ValueError, match="exceeds suite size"):
        sample_prompt_suite(cases, size=11)
    with pytest.raises(ValueError, match="positive integer"):
        sample_prompt_suite(cases, size=True)


def test_slice_samples_for_suite_aligns_and_rejects_missing() -> None:
    cases = [PromptCase(prompt_id="p2", prompt="2"), PromptCase(prompt_id="p1", prompt="1")]
    samples = [
        ModelSample(prompt_id="p3", output="3"),
        ModelSample(prompt_id="p1", output="1"),
        ModelSample(prompt_id="p2", output="2"),
    ]
    sliced = slice_samples_for_suite(samples, cases)
    assert [s.prompt_id for s in sliced] == ["p1", "p2"]

    with pytest.raises(ValueError, match="missing prompt_id"):
        slice_samples_for_suite(samples[:1], cases)
    with pytest.raises(ValueError, match="duplicate prompt_id"):
        slice_samples_for_suite([samples[0], samples[0]], cases)


def test_sampling_edge_branches_and_filter_prompt_suite() -> None:
    cases = (
        [
            PromptCase(
                prompt_id=f"big-{i}",
                prompt="b",
                category="big",
                schema={"type": "object"} if i % 2 == 0 else None,
            )
            for i in range(8)
        ]
        + [PromptCase(prompt_id="s1", prompt="s", category="small1")]
        + [PromptCase(prompt_id="s2", prompt="s", category="small2")]
        + [PromptCase(prompt_id="s3", prompt="s", category="small3")]
    )

    # size < number of categories selects from largest proportional buckets cleanly.
    tiny = sample_prompt_suite(cases, size=2, seed=1)
    assert len(tiny) == 2

    # Triggers quota trim branch (4 categories, size=4 -> big exact ~2.9 -> base=2, +3*1 = 5 > 4).
    exact_cats = sample_prompt_suite(cases, size=4, seed=1)
    assert len(exact_cats) == 4
    assert {c.category for c in exact_cats} == {"big", "small1", "small2", "small3"}

    with pytest.raises(ValueError, match="empty"):
        sample_prompt_suite([], size=1)
    with pytest.raises(ValueError, match="duplicate prompt_id"):
        sample_prompt_suite([cases[0], cases[0]], size=1)
    with pytest.raises(ValueError, match="boolean"):
        sample_prompt_suite(cases, size=2, seed=True)

    filtered = filter_prompt_suite(
        cases,
        categories=["big"],
        require_schema=True,
        exclude_ids=["big-0"],
    )
    assert [c.prompt_id for c in filtered] == ["big-2", "big-4", "big-6"]

    with pytest.raises(ValueError, match="Unknown category"):
        filter_prompt_suite(cases, categories=["nonexistent"])
    with pytest.raises(ValueError, match="empty"):
        filter_prompt_suite([])

