"""Local evidence results and an optional external SDK export adapter."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from importlib import import_module
from typing import Any


@dataclass
class EvidenceScore:
    """A named observation with explanation and explicit availability.

    The numeric value is a placeholder when scoring_failed is true.
    Use measured_value or to_dict when consuming scores so unavailable evidence
    cannot be mistaken for a measured zero.
    """

    name: str
    value: float
    reason: str | None = None
    category_name: str | None = None
    metadata: dict[str, Any] | None = None
    scoring_failed: bool = False

    @property
    def measured_value(self) -> float | None:
        return None if self.scoring_failed else self.value

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["value"] = self.measured_value
        return result

    def to_opik(self) -> Any:
        """Construct the installed Opik SDK's public ScoreResult, on demand.

        No vendored class is returned by this adapter. Unknown evidence is
        rejected so SDK versions cannot accidentally aggregate a placeholder zero.
        """
        if self.scoring_failed:
            raise ValueError("Cannot export unavailable evidence as an Opik numeric score.")
        try:
            module = import_module("opik.evaluation.metrics.score_result")
        except ModuleNotFoundError as exc:
            if exc.name == "opik":
                raise ImportError(
                    "Install llm-drift-bench[opik] to export installed SDK results."
                ) from exc
            raise
        return module.ScoreResult(
            name=self.name, value=self.value, reason=self.reason,
            category_name=self.category_name,
            metadata=deepcopy(self.metadata),
        )
