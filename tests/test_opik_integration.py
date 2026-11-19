"""Exercise EvidenceScore.to_opik() against the real, installed Opik SDK.

These tests are skipped when Opik is not installed. CI sets REQUIRE_OPIK=1 in the
dedicated job so a missing or broken install fails instead of silently skipping.
"""

import dataclasses
import os
from importlib.metadata import version

import pytest

from llm_drift_bench import EvidenceScore, ToolCheckResult

if os.environ.get("REQUIRE_OPIK") == "1":
    import opik  # noqa: F401  (a missing install must fail, not skip)
else:
    pytest.importorskip("opik")

from opik.evaluation.metrics.score_result import ScoreResult  # noqa: E402


def test_sdk_class_accepts_every_field_the_adapter_passes() -> None:
    fields = {field.name for field in dataclasses.fields(ScoreResult)}
    assert {"name", "value", "reason", "category_name", "metadata"} <= fields, version("opik")


def test_adapter_returns_the_real_public_score_result() -> None:
    metadata = {"status": "PASS", "covered_characters": 5, "ranges": [[0, 5]]}
    score = EvidenceScore("read_before_edit", 1.0, reason="all covered",
                          category_name="tool-evidence", metadata=metadata)
    exported = score.to_opik()
    assert type(exported) is ScoreResult
    assert exported.name == "read_before_edit"
    assert exported.value == 1.0
    assert exported.reason == "all covered"
    assert exported.category_name == "tool-evidence"
    assert exported.metadata == metadata
    assert exported.metadata is not metadata
    assert exported.metadata["ranges"] is not metadata["ranges"]
    assert exported.scoring_failed is False


def test_optional_fields_default_through_the_real_sdk() -> None:
    exported = EvidenceScore("check", 0.0).to_opik()
    assert (exported.reason, exported.category_name, exported.metadata) == (None, None, None)


@pytest.mark.parametrize(("status", "value"), [("PASS", 1.0), ("FAIL", 0.0)])
def test_tool_check_results_export_measured_verdicts(status: str, value: float) -> None:
    exported = ToolCheckResult(status, "explanation", 5, 10, 2).to_score().to_opik()
    assert type(exported) is ScoreResult
    assert exported.value == value
    assert exported.metadata is not None and exported.metadata["status"] == status


def test_unknown_evidence_is_never_exported_as_a_numeric_score() -> None:
    result = ToolCheckResult("UNKNOWN", "no evidence", 0, 0, 0)
    with pytest.raises(ValueError, match="unavailable"):
        result.to_score().to_opik()
