"""Tests for downstream consumer validation and rejection-rate creep analysis."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from llm_drift_bench.consumer import (
    ConsumerDecision,
    compare_consumer_decisions,
    evaluate_consumer_rejection,
    format_consumer_markdown,
    format_consumer_text,
    load_consumer_decisions,
    load_validator_callable,
)
from llm_drift_bench.models import ModelSample


def test_evaluate_consumer_rejection_catches_business_rule_creep() -> None:
    # All JSON outputs are structurally valid, but downstream requires total == subtotal + tax.
    def _invoice_invariant(sample: ModelSample) -> tuple[bool, str | None]:
        data = json.loads(sample.output)
        if abs((data["subtotal"] + data["tax"]) - data["total"]) > 1e-6:
            return False, "total != subtotal + tax"
        return True, None

    base = [
        ModelSample(prompt_id="p1", output='{"subtotal": 100, "tax": 10, "total": 110}'),
        ModelSample(prompt_id="p2", output='{"subtotal": 50, "tax": 5, "total": 55}'),
        ModelSample(prompt_id="p3", output='{"subtotal": 20, "tax": 2, "total": 22}'),
        ModelSample(prompt_id="p4", output='{"subtotal": 10, "tax": 1, "total": 11}'),
    ]
    cand = [
        ModelSample(prompt_id="p1", output='{"subtotal": 100, "tax": 10, "total": 110}'),
        ModelSample(prompt_id="p2", output='{"subtotal": 50, "tax": 5, "total": 60}'),
        ModelSample(prompt_id="p3", output='{"subtotal": 20, "tax": 2, "total": 25}'),
        ModelSample(prompt_id="p4", output='{"subtotal": 10, "tax": 1, "total": 11}'),
    ]

    comp = evaluate_consumer_rejection(
        base,
        cand,
        _invoice_invariant,
        max_rejection_rate=0.25,
        max_rejection_increase=0.10,
    )
    assert comp.passed is False
    assert comp.baseline_rejection_rate == 0.0
    assert comp.candidate_rejection_rate == pytest.approx(0.5)
    assert comp.rejection_rate_delta == pytest.approx(0.5)
    assert comp.newly_rejected_ids == ("p2", "p3")
    assert comp.newly_accepted_ids == ()
    assert comp.candidate_reasons["p2"] == "total != subtotal + tax"
    assert len(comp.failure_reasons) == 2

    as_dict = comp.to_dict()
    assert as_dict["passed"] is False and as_dict["newly_rejected_ids"] == ["p2", "p3"]
    assert "FAILED" in format_consumer_text(comp)
    md = format_consumer_markdown(comp)
    assert "# Downstream Consumer Rejection Check: FAILED" in md and "| p2 |" in md


def test_load_consumer_decisions_jsonl_and_json_plus_dynamic_validator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base_file = tmp_path / "base_dec.jsonl"
    cand_file = tmp_path / "cand_dec.json"
    base_file.write_text(
        json.dumps({"prompt_id": "p1", "accepted": False, "reason": "old bug"}) + "\n"
        + json.dumps({"prompt_id": "p2", "accepted": True}) + "\n",
        encoding="utf-8",
    )
    cand_file.write_text(
        json.dumps(
            {
                "decisions": [
                    {"prompt_id": "p1", "accepted": True},
                    {"prompt_id": "p2", "accepted": True},
                ]
            }
        ),
        encoding="utf-8",
    )
    b_dec = load_consumer_decisions(base_file)
    c_dec = load_consumer_decisions(cand_file)
    comp = compare_consumer_decisions(b_dec, c_dec, max_rejection_increase=0.05)
    assert comp.passed is True
    assert comp.newly_accepted_ids == ("p1",)
    assert comp.rejection_rate_delta == pytest.approx(-0.5)
    assert "PASSED" in format_consumer_text(comp)

    # Dynamic module:callable loader
    mod_file = tmp_path / "custom_consumer_mod.py"
    mod_file.write_text(
        "def accept_nonempty(sample):\n    return bool(sample.output.strip())\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    try:
        fn = load_validator_callable("custom_consumer_mod:accept_nonempty")
        res = evaluate_consumer_rejection(
            [ModelSample(prompt_id="p1", output="ok")],
            [ModelSample(prompt_id="p1", output="   ")],
            fn,
            max_rejection_rate=0.5,
        )
        assert res.passed is False and res.newly_rejected_ids == ("p1",)
    finally:
        sys.modules.pop("custom_consumer_mod", None)


def test_consumer_validation_errors(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="boolean"):
        ConsumerDecision(prompt_id="p1", accepted=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="module.path:callable_name"):
        load_validator_callable("no_colon_here")
    with pytest.raises(ValueError, match="not callable"):
        load_validator_callable("json:__name__")
    with pytest.raises(TypeError, match="bool or"):
        evaluate_consumer_rejection(
            [ModelSample(prompt_id="p1", output="a")],
            [ModelSample(prompt_id="p1", output="a")],
            lambda _s: "bad",  # type: ignore[arg-type, return-value]
        )

    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        load_consumer_decisions(empty)


def test_consumer_json_array_to_dict_and_full_error_matrix(tmp_path: Path) -> None:
    # ConsumerDecision.to_dict with and without reason + validation errors
    d_plain = ConsumerDecision(prompt_id="p1", accepted=True)
    assert d_plain.to_dict() == {"prompt_id": "p1", "accepted": True}
    d_reason = ConsumerDecision(prompt_id="p2", accepted=False, reason="bad_code")
    assert d_reason.to_dict() == {"prompt_id": "p2", "accepted": False, "reason": "bad_code"}

    with pytest.raises(ValueError, match="non-empty string"):
        ConsumerDecision(prompt_id="   ", accepted=True)
    with pytest.raises(ValueError, match="string or None"):
        ConsumerDecision(prompt_id="p1", accepted=False, reason=99)  # type: ignore[arg-type]

    # Top-level JSON array decision log
    arr_file = tmp_path / "arr.json"
    arr_file.write_text(
        json.dumps([{"prompt_id": "p1", "accepted": True}]),
        encoding="utf-8",
    )
    loaded = load_consumer_decisions(arr_file)
    assert len(loaded) == 1 and loaded[0].prompt_id == "p1"

    # Malformed decision files & rows
    for name, body, match in (
        ("bad_top.json", '{"wrong": 1}', "JSON array"),
        ("empty_arr.json", "[]", "no records"),
        ("non_dict.json", "[123]", "JSON object"),
        ("blank_id.jsonl", '{"prompt_id": " ", "accepted": true}\n', "prompt_id"),
        ("bad_acc.jsonl", '{"prompt_id": "p1", "accepted": "yes"}\n', "boolean"),
        ("bad_reason.jsonl", '{"prompt_id": "p1", "accepted": false, "reason": 7}\n', "reason"),
    ):
        bad = tmp_path / name
        bad.write_text(body, encoding="utf-8")
        with pytest.raises(ValueError, match=match):
            load_consumer_decisions(bad)

    # Empty collections, out-of-range limits, non-callable validator, and blank validator spec
    with pytest.raises(ValueError, match="non-empty"):
        compare_consumer_decisions([], [d_plain])
    with pytest.raises(ValueError, match="between 0.0 and 1.0"):
        compare_consumer_decisions([d_plain], [d_plain], max_rejection_rate=1.5)
    with pytest.raises(TypeError, match="callable"):
        evaluate_consumer_rejection(
            [ModelSample(prompt_id="p1", output="a")],
            [ModelSample(prompt_id="p1", output="a")],
            "not_fn",  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="non-empty"):
        evaluate_consumer_rejection([], [ModelSample(prompt_id="p1", output="a")], lambda _s: True)
    with pytest.raises(ValueError, match="non-empty module"):
        load_validator_callable(":fn")

