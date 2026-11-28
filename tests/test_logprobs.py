"""Tests for optional top_logprobs token entropy and JS divergence drift analysis."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from llm_drift_bench.logprobs import analyze_logprobs_drift, format_logprobs_text
from llm_drift_bench.models import ModelSample


def _lp(prob: float) -> float:
    return math.log(prob)


def test_logprobs_detects_pre_flip_uncertainty_shift_with_identical_text() -> None:
    # Both models emit the exact same greedy text '{"label": "high"}',
    # yet baseline is 99% confident while candidate is nearly 50/50!
    base = [
        ModelSample(
            prompt_id="p1",
            output='{"label": "high"}',
            model_name="fp16",
            metadata={
                "logprobs": [
                    {
                        "token": "high",
                        "logprob": _lp(0.99),
                        "top_logprobs": [
                            {"token": "high", "logprob": _lp(0.99)},
                            {"token": "low", "logprob": _lp(0.01)},
                        ],
                    }
                ]
            },
        )
    ]
    cand = [
        ModelSample(
            prompt_id="p1",
            output='{"label": "high"}',
            model_name="q4",
            metadata={
                "logprobs": [
                    {
                        "token": "high",
                        "logprob": _lp(0.55),
                        "top_logprobs": [
                            {"token": "high", "logprob": _lp(0.55)},
                            {"token": "low", "logprob": _lp(0.45)},
                        ],
                    }
                ]
            },
        )
    ]

    summary = analyze_logprobs_drift(base, cand)
    assert summary.complete is True
    assert summary.comparable_samples == 1
    assert summary.mean_baseline_entropy_bits is not None
    assert summary.mean_candidate_entropy_bits is not None
    assert summary.mean_entropy_delta_bits is not None
    assert summary.mean_baseline_entropy_bits < 0.12
    assert summary.mean_candidate_entropy_bits > 0.95
    assert summary.mean_entropy_delta_bits > 0.85
    assert summary.mean_top_k_jsd is not None and summary.mean_top_k_jsd > 0.20
    assert summary.mean_logprob_delta is not None and summary.mean_logprob_delta < -0.5

    as_dict = summary.to_dict()
    assert as_dict["complete"] is True
    assert as_dict["results"][0]["prompt_id"] == "p1"
    rendered = format_logprobs_text(summary)
    assert "complete" in rendered and "p1:" in rendered


def test_logprobs_identical_distributions_yield_zero_drift() -> None:
    meta = {
        "logprobs": [
            {
                "token": "yes",
                "logprob": _lp(0.8),
                "top_logprobs": [
                    {"token": "yes", "logprob": _lp(0.8)},
                    {"token": "no", "logprob": _lp(0.15)},
                ],
            }
        ]
    }
    base = [ModelSample(prompt_id="p1", output="yes", model_name="b", metadata=meta)]
    cand = [ModelSample(prompt_id="p1", output="yes", model_name="c", metadata=meta)]
    summary = analyze_logprobs_drift(base, cand)
    assert summary.complete is True
    assert summary.mean_entropy_delta_bits == pytest.approx(0.0, abs=1e-9)
    assert summary.mean_top_k_jsd == pytest.approx(0.0, abs=1e-9)
    assert summary.mean_logprob_delta == pytest.approx(0.0, abs=1e-9)


def test_logprobs_missing_metadata_reports_incomplete_without_fabricating() -> None:
    base = [
        ModelSample(prompt_id="p1", output="ok", model_name="b"),
        ModelSample(
            prompt_id="p2",
            output="ok",
            model_name="b",
            metadata={"logprobs": [{"token": "ok", "logprob": _lp(0.9)}]},
        ),
    ]
    cand = [
        ModelSample(prompt_id="p1", output="ok", model_name="c"),
        ModelSample(
            prompt_id="p2",
            output="ok",
            model_name="c",
            metadata={"logprobs": [{"token": "ok", "logprob": _lp(0.5)}]},
        ),
    ]
    summary = analyze_logprobs_drift(base, cand)
    assert summary.total_samples == 2
    assert summary.comparable_samples == 1
    assert summary.complete is False
    assert summary.results[0].available is False
    assert summary.results[1].available is True
    text = format_logprobs_text(summary)
    assert "incomplete" in text
    assert "p1: unavailable" in text

    # None available -> all aggregates remain None
    empty_summary = analyze_logprobs_drift([base[0]], [cand[0]])
    assert empty_summary.complete is False
    assert empty_summary.comparable_samples == 0
    assert empty_summary.mean_top_k_jsd is None
    assert "n/a" in format_logprobs_text(empty_summary)


def test_logprobs_validation_errors() -> None:
    s1 = [ModelSample(prompt_id="p1", output="a", model_name="b")]
    with pytest.raises(ValueError, match="non-empty"):
        analyze_logprobs_drift([], s1)
    with pytest.raises(ValueError, match="do not match"):
        analyze_logprobs_drift(s1, [ModelSample(prompt_id="p2", output="a", model_name="c")])

    # Malformed token entries, >1.0 probability renormalization, and all-invalid logprobs list
    noisy_b = [
        ModelSample(
            prompt_id="p1",
            output="ok",
            model_name="b",
            metadata={
                "logprobs": [
                    "not-a-dict",
                    {"token": "skip", "logprob": 2.5},
                    {
                        "token": "yes",
                        "logprob": _lp(0.8),
                        "top_logprobs": [
                            "bad-cand",
                            {"token": "yes", "logprob": _lp(0.8)},
                            {"token": "no", "logprob": _lp(0.5)},
                        ],
                    },
                ]
            },
        ),
        ModelSample(
            prompt_id="p2",
            output="ok",
            model_name="b",
            metadata={"logprobs": ["bad", {"token": "x", "logprob": 5.0}]},
        ),
    ]
    noisy_c = [
        ModelSample(
            prompt_id="p1",
            output="ok",
            model_name="c",
            metadata={"logprobs": [{"token": "yes", "logprob": _lp(0.8)}]},
        ),
        ModelSample(
            prompt_id="p2",
            output="ok",
            model_name="c",
            metadata={"logprobs": [{"token": "x", "logprob": _lp(0.8)}]},
        ),
    ]
    noisy_summary = analyze_logprobs_drift(noisy_b, noisy_c)
    assert noisy_summary.comparable_samples == 1
    assert noisy_summary.complete is False


def test_logprobs_markdown_html_and_cli(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from llm_drift_bench.cli import main
    from llm_drift_bench.logprobs import export_logprobs_html, format_logprobs_markdown

    base = [
        ModelSample(
            prompt_id="p1",
            output="yes",
            model_name="b",
            metadata={"logprobs": [{"token": "yes", "logprob": _lp(0.95)}]},
        ),
        ModelSample(prompt_id="p2", output="no", model_name="b"),
    ]
    cand = [
        ModelSample(
            prompt_id="p1",
            output="yes",
            model_name="c",
            metadata={"logprobs": [{"token": "yes", "logprob": _lp(0.55)}]},
        ),
        ModelSample(prompt_id="p2", output="no", model_name="c"),
    ]
    summary = analyze_logprobs_drift(base, cand)
    md = format_logprobs_markdown(summary)
    assert "## Logprobs Distribution & Uncertainty Drift (Incomplete)" in md
    assert "| `p1` | OK |" in md and "| `p2` | N/A |" in md

    html_file = tmp_path / "lp.html"
    html = export_logprobs_html(summary, output_path=html_file)
    assert html_file.read_text(encoding="utf-8") == html
    assert "Content-Security-Policy" in html and "<script" not in html.lower()
    assert "INCOMPLETE" in html and "<code>p1</code>" in html

    b_file = tmp_path / "b.jsonl"
    c_file = tmp_path / "c.jsonl"
    b_file.write_text(
        '{"prompt_id": "p1", "output": "yes", "metadata": {"logprobs": '
        '[{"token": "yes", "logprob": -0.05}]}}\n',
        encoding="utf-8",
    )
    c_file.write_text(
        '{"prompt_id": "p1", "output": "yes", "metadata": {"logprobs": '
        '[{"token": "yes", "logprob": -0.60}]}}\n',
        encoding="utf-8",
    )
    assert (
        main(
            [
                "logprobs",
                "-b",
                str(b_file),
                "-c",
                str(c_file),
                "--max-entropy-delta",
                "0.01",
                "--format",
                "markdown",
            ]
        )
        == 1
    )
    md_out = capsys.readouterr().out
    assert "## Logprobs Distribution & Uncertainty Drift (Complete)" in md_out
    assert "**Threshold failures:**" in md_out

    cli_html = tmp_path / "cli_lp.html"
    assert (
        main(
            [
                "logprobs",
                "-b",
                str(b_file),
                "-c",
                str(c_file),
                "--format",
                "html",
                "-o",
                str(cli_html),
            ]
        )
        == 0
    )
    assert "Wrote logprobs drift report" in capsys.readouterr().out
    assert "COMPLETE" in cli_html.read_text(encoding="utf-8")

    cli_md_out = tmp_path / "cli_lp.md"
    assert (
        main(
            [
                "logprobs",
                "-b",
                str(b_file),
                "-c",
                str(c_file),
                "--format",
                "markdown",
                "-o",
                str(cli_md_out),
            ]
        )
        == 0
    )
    assert "## Logprobs" in cli_md_out.read_text(encoding="utf-8")
    capsys.readouterr()
