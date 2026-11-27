"""Tests for deterministic quantization and upgrade quality regression checks."""

from __future__ import annotations

import pytest

from llm_drift_bench.models import ModelSample
from llm_drift_bench.quality_checks import (
    detect_repetition_loop,
    format_quality_report_text,
    is_likely_refusal,
    run_quality_checks,
)


def test_detect_repetition_loop_catches_line_and_ngram_loops() -> None:
    line_loop = "\n".join(["Step 1: verify the invoice total carefully."] * 4)
    hit, detail = detect_repetition_loop(line_loop)
    assert hit is True and detail is not None and "Repeated line" in detail

    ngram_loop = "The server failed to connect to the database. " * 4
    hit2, detail2 = detect_repetition_loop(ngram_loop)
    assert hit2 is True and detail2 is not None and "5-word phrase" in detail2

    clean = (
        "First verify the invoice number against the purchase order. "
        "Next confirm the subtotal, shipping fee, and tax rate before approval."
    )
    assert detect_repetition_loop(clean) == (False, None)


def test_is_likely_refusal_handles_standard_and_smart_quotes() -> None:
    assert is_likely_refusal("I\u2019m sorry, I cannot assist with that request.") is True
    assert is_likely_refusal("As an AI language model, I have limitations.") is True
    assert is_likely_refusal('{"status": "ok", "total": 42}') is False
    # Refusal phrase buried far past opening shouldn't falsely flag a normal essay.
    essay = ("Detailed technical overview of authorization checks. " * 10) + (
        "Sometimes models say I cannot assist with unsafe prompts."
    )
    assert is_likely_refusal(essay) is False


def test_run_quality_checks_detects_all_four_regression_kinds() -> None:
    base = [
        ModelSample(
            prompt_id="p_loop",
            output="Check the config file and restart the worker cleanly.",
            model_name="fp16",
        ),
        ModelSample(
            prompt_id="p_refuse",
            output="Here is the SQL query to aggregate monthly revenue by region.",
            model_name="fp16",
        ),
        ModelSample(
            prompt_id="p_unblock",
            output="I'm sorry, I cannot fulfill that request.",
            model_name="fp16",
        ),
        ModelSample(
            prompt_id="p_trunc",
            output="This detailed response explains all five migration phases and rollback steps.",
            model_name="fp16",
        ),
        ModelSample(
            prompt_id="p_blowup",
            output="Summary: deployment succeeded cleanly on all three nodes.",
            model_name="fp16",
        ),
        ModelSample(
            prompt_id="p_ok",
            output="The invoice total is $120.50 due on Friday.",
            model_name="fp16",
        ),
    ]
    cand = [
        ModelSample(
            prompt_id="p_loop",
            output="Check the config file and restart the worker cleanly. " * 4,
            model_name="q4",
        ),
        ModelSample(
            prompt_id="p_refuse",
            output="I'm sorry, but I cannot assist with that request.",
            model_name="q4",
        ),
        ModelSample(
            prompt_id="p_unblock",
            output="Here is the direct answer you asked for.",
            model_name="q4",
        ),
        ModelSample(
            prompt_id="p_trunc",
            output="Done.",
            model_name="q4",
        ),
        ModelSample(
            prompt_id="p_blowup",
            output=" ".join(f"Section {i} notes deployment status ok." for i in range(15)),
            model_name="q4",
        ),
        ModelSample(
            prompt_id="p_ok",
            output="The invoice total is $120.50 due Friday.",
            model_name="q4",
        ),
    ]

    report = run_quality_checks(base, cand)
    assert report.passed is False
    assert report.total_samples == 6
    assert report.flagged_prompts == 5
    by_prompt = {(f.prompt_id, f.check): f.severity for f in report.findings}
    assert by_prompt[("p_loop", "repetition_loop")] == "error"
    assert by_prompt[("p_refuse", "refusal_flip")] == "error"
    assert by_prompt[("p_unblock", "refusal_flip")] == "warning"
    assert by_prompt[("p_trunc", "length_truncation")] == "error"
    assert by_prompt[("p_blowup", "length_blowup")] == "warning"
    assert all(f.prompt_id != "p_ok" for f in report.findings)

    as_dict = report.to_dict()
    assert as_dict["passed"] is False
    assert as_dict["counts_by_check"]["refusal_flip"] == 2
    text = format_quality_report_text(report)
    assert "ISSUES DETECTED" in text and "p_loop" in text


def test_quality_checks_clean_pair_and_validation() -> None:
    same = [ModelSample(prompt_id="p1", output="Healthy deterministic output.", model_name="m")]
    clean = run_quality_checks(same, same)
    assert clean.passed is True
    assert "PASSED" in format_quality_report_text(clean)

    with pytest.raises(ValueError, match="max_length_ratio"):
        run_quality_checks(same, same, max_length_ratio=1.0)
    with pytest.raises(ValueError, match="min_length_ratio"):
        run_quality_checks(same, same, min_length_ratio=1.0)
    with pytest.raises(ValueError, match="min_baseline_chars"):
        run_quality_checks(same, same, min_baseline_chars=0)
    with pytest.raises(ValueError, match="ngram_words"):
        detect_repetition_loop("abc", ngram_words=1)
    with pytest.raises(ValueError, match="min_repeats"):
        detect_repetition_loop("abc", min_repeats=2)
    with pytest.raises(ValueError, match="non-empty"):
        run_quality_checks([], same)
    # Overlapping 5-word n-gram occurrences collapse below min_repeats (no false loop)
    assert detect_repetition_loop("word " * 8, ngram_words=5, min_repeats=4) == (False, None)

