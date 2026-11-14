"""Verify the local result contract and optional SDK adapter."""

import json
from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from llm_drift_bench import EvidenceScore, ToolCheckResult


@pytest.mark.parametrize(("status", "expected"), [("PASS", 1.0), ("FAIL", 0.0),
                                                   ("UNKNOWN", None)])
def test_verdict_adapter_serializes_missing_scores(status, expected) -> None:
    result = ToolCheckResult(status, "Evidence explanation", 5, 10, 2)
    score = result.to_score()
    assert isinstance(score, EvidenceScore)
    assert score.reason == result.reason
    assert score.measured_value == expected
    assert score.scoring_failed == (status == "UNKNOWN")
    payload = json.loads(json.dumps(score.to_dict()))
    assert payload["value"] == expected
    assert payload["metadata"]["status"] == status
    assert payload["metadata"]["covered_characters"] == 5


def test_result_constructor_defaults_and_mutability_are_preserved() -> None:
    score = EvidenceScore("example", 0.5)
    assert asdict(score) == {
        "name": "example", "value": 0.5, "reason": None,
        "category_name": None, "metadata": None, "scoring_failed": False,
    }
    score.reason = "explanation"
    assert score.to_dict()["reason"] == "explanation"


def test_installed_sdk_adapter_uses_public_class() -> None:
    constructor = Mock(return_value=object())
    score = EvidenceScore("check", 1.0, reason="ok", metadata={"status": "PASS"})
    with patch("llm_drift_bench.scores.import_module",
               return_value=SimpleNamespace(ScoreResult=constructor)) as imported:
        result = score.to_opik()
    imported.assert_called_once_with("opik.evaluation.metrics.score_result")
    assert result is constructor.return_value
    constructor.assert_called_once_with(
        name="check", value=1.0, reason="ok", category_name=None, metadata={"status": "PASS"},
    )
    assert constructor.call_args.kwargs["metadata"] is not score.metadata


def test_unknown_evidence_does_not_import_or_export_sdk_score() -> None:
    with (
        patch("llm_drift_bench.scores.import_module") as imported,
        pytest.raises(ValueError, match="unavailable"),
    ):
        EvidenceScore("check", 0.0, scoring_failed=True).to_opik()
    imported.assert_not_called()


def test_missing_sdk_explains_installation_but_broken_sdk_errors_propagate() -> None:
    for name in ("opik", "missing_internal_dependency"):
        error = ModuleNotFoundError("missing", name=name)
        with (
            patch("llm_drift_bench.scores.import_module", side_effect=error),
            pytest.raises(ImportError) as caught,
        ):
            EvidenceScore("check", 0.0).to_opik()
        if name == "opik":
            assert "llm-drift-bench[opik]" in str(caught.value)
        else:
            assert caught.value is error
