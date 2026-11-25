"""Tests for the unified offline audit report and executive dashboard."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from llm_drift_bench.audit import (
    export_audit_html,
    format_audit_markdown,
    format_audit_text,
    run_full_audit,
)
from llm_drift_bench.models import ModelSample
from llm_drift_bench.packs import get_pack
from llm_drift_bench.policy import load_policy


def test_full_audit_passes_clean_pack_and_fails_drifted_pack(tmp_path: Path) -> None:
    pack = get_pack("extraction")
    policy_file = tmp_path / "drift.toml"
    policy_file.write_text(pack.policy_toml, encoding="utf-8")
    policy = load_policy(policy_file)

    ok_audit = run_full_audit(
        pack.baseline,
        pack.candidate_pass,
        prompt_cases=pack.prompts,
        policy=policy,
        max_tvd=0.1,
        max_wasserstein=0.1,
    )
    assert ok_audit.passed is True
    assert ok_audit.failure_reasons == ()
    assert ok_audit.cost.complete is True
    assert "PASSED" in format_audit_text(ok_audit)
    assert "# Unified Drift Audit: PASSED" in format_audit_markdown(ok_audit)

    bad_audit = run_full_audit(
        pack.baseline,
        pack.candidate_drift,
        prompt_cases=pack.prompts,
        policy=policy,
        max_tvd=0.1,
        max_wasserstein=0.1,
    )
    assert bad_audit.passed is False
    assert any("Distribution max_tvd" in r for r in bad_audit.failure_reasons)
    assert bad_audit.to_dict()["passed"] is False

    html_out = tmp_path / "audit.html"
    html = export_audit_html(bad_audit, html_out)
    assert html_out.read_text(encoding="utf-8") == html
    assert "default-src 'none'" in html
    assert "<script" not in html.lower()
    assert "FAILED" in html


def test_full_audit_quality_and_logprobs_gates() -> None:
    pack = get_pack("coding")
    coding_audit = run_full_audit(
        pack.baseline,
        pack.candidate_drift,
        prompt_cases=pack.prompts,
        fail_on_quality_issues=True,
    )
    assert coding_audit.passed is False
    assert any("Quality regressions" in r for r in coding_audit.failure_reasons)
    assert "## Quality findings" in format_audit_markdown(coding_audit)

    # Incomplete logprobs with threshold must fail honestly
    lp_missing = run_full_audit(
        pack.baseline,
        pack.candidate_pass,
        prompt_cases=pack.prompts,
        max_top_k_jsd=0.2,
    )
    assert lp_missing.passed is False
    assert any("Logprobs coverage incomplete" in r for r in lp_missing.failure_reasons)

    # Complete logprobs breaching threshold fails with exact reason
    lp_base = [
        ModelSample(
            prompt_id="p1",
            output="yes",
            model_name="fp16",
            metadata={
                "logprobs": [
                    {
                        "token": "yes",
                        "logprob": math.log(0.99),
                        "top_logprobs": [
                            {"token": "yes", "logprob": math.log(0.99)},
                            {"token": "no", "logprob": math.log(0.01)},
                        ],
                    }
                ]
            },
        )
    ]
    lp_cand = [
        ModelSample(
            prompt_id="p1",
            output="yes",
            model_name="q4",
            metadata={
                "logprobs": [
                    {
                        "token": "yes",
                        "logprob": math.log(0.55),
                        "top_logprobs": [
                            {"token": "yes", "logprob": math.log(0.55)},
                            {"token": "no", "logprob": math.log(0.45)},
                        ],
                    }
                ]
            },
        )
    ]
    lp_fail = run_full_audit(lp_base, lp_cand, max_entropy_delta=0.2, max_top_k_jsd=0.1)
    assert lp_fail.passed is False
    assert any("mean_entropy_delta_bits" in r for r in lp_fail.failure_reasons)
    assert any("max_top_k_jsd" in r for r in lp_fail.failure_reasons)

    with pytest.raises(ValueError, match="max_tvd"):
        run_full_audit(lp_base, lp_cand, max_tvd=1.5)
    with pytest.raises(ValueError, match="max_entropy_delta"):
        run_full_audit(lp_base, lp_cand, max_entropy_delta=-0.1)
