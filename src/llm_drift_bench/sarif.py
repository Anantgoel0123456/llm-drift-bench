"""SARIF 2.1.0 exporter for GitHub Code Scanning and IDE diagnostic viewers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from llm_drift_bench import __version__
from llm_drift_bench.models import DriftBenchmarkReport

_SARIF_SCHEMA = (
    "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/main/"
    "sarif-2.1/schema/sarif-schema-2.1.0.json"
)

_SARIF_RULES: list[dict[str, Any]] = [
    {
        "id": "DRIFT001",
        "name": "SchemaConformityFailure",
        "shortDescription": {"text": "Candidate response failed JSON or JSON Schema validation."},
        "defaultConfiguration": {"level": "error"},
    },
    {
        "id": "DRIFT002",
        "name": "NumericDriftExceeded",
        "shortDescription": {"text": "Shared numeric JSON fields exceeded drift threshold."},
        "defaultConfiguration": {"level": "error"},
    },
    {
        "id": "DRIFT003",
        "name": "CompositeDriftExceeded",
        "shortDescription": {"text": "Prompt composite output drift exceeded threshold."},
        "defaultConfiguration": {"level": "error"},
    },
    {
        "id": "DRIFT004",
        "name": "SuiteGateFailure",
        "shortDescription": {"text": "Aggregate benchmark suite threshold gate failed."},
        "defaultConfiguration": {"level": "error"},
    },
]


def _partial_fingerprint(rule_id: str, subject: str, baseline: str, candidate: str) -> str:
    payload = f"{rule_id}\0{subject}\0{baseline}\0{candidate}".encode()
    return hashlib.sha256(payload).hexdigest()[:32]


def export_sarif_report(
    report: DriftBenchmarkReport,
    output_path: str | Path | None = None,
    *,
    artifact_uri: str = "prompts.json",
) -> str:
    """Export canonical DriftBenchmarkReport findings as OASIS SARIF 2.1.0 JSON.

    Excludes raw captured response bodies to keep CI annotations safe and compact.
    """
    if not isinstance(report, DriftBenchmarkReport):
        raise ValueError("SARIF export requires a canonical engine report.")
    if not isinstance(artifact_uri, str) or not artifact_uri.strip():
        raise ValueError("artifact_uri must be a non-empty string.")

    results: list[dict[str, Any]] = []
    thresholds = report.thresholds

    for row in sorted(report.results, key=lambda item: item.prompt_id):
        if not row.drifted:
            continue
        loc = [
            {
                "physicalLocation": {"artifactLocation": {"uri": artifact_uri}},
                "logicalLocations": [
                    {
                        "name": row.prompt_id,
                        "fullyQualifiedName": f"{row.category}/{row.prompt_id}",
                        "kind": "prompt",
                    }
                ],
            }
        ]
        props: dict[str, Any] = {
            "prompt_id": row.prompt_id,
            "category": row.category,
            "composite_drift_score": round(row.composite_drift_score, 4),
            "candidate_passed": row.candidate_validation.passed,
        }
        if row.json_numeric_drift is not None:
            props["json_numeric_drift"] = round(row.json_numeric_drift, 4)

        if not row.candidate_validation.passed:
            rule_id = "DRIFT001"
            issue_summary = "; ".join(
                f"{iss.path}: {iss.rule}" for iss in row.candidate_validation.issues
            ) or "invalid JSON"
            msg = (
                f"Prompt {row.prompt_id!r} failed JSON/schema validation "
                f"({issue_summary}); composite drift {row.composite_drift_score:.4f}."
            )
        elif (
            thresholds is not None
            and row.json_numeric_drift is not None
            and row.json_numeric_drift > thresholds.max_numeric_relative_drift
        ):
            rule_id = "DRIFT002"
            msg = (
                f"Prompt {row.prompt_id!r} numeric relative drift "
                f"{row.json_numeric_drift:.4f} exceeded threshold "
                f"{thresholds.max_numeric_relative_drift:.4f}."
            )
        else:
            rule_id = "DRIFT003"
            msg = (
                f"Prompt {row.prompt_id!r} composite drift "
                f"{row.composite_drift_score:.4f} exceeded drift threshold."
            )
        results.append({
            "ruleId": rule_id,
            "level": "error",
            "message": {"text": msg},
            "locations": loc,
            "partialFingerprints": {
                "primaryLocationLineHash": _partial_fingerprint(
                    rule_id, row.prompt_id, report.baseline_model, report.candidate_model
                )
            },
            "properties": props,
        })

    if not report.passed:
        for reason in report.failure_reasons or ("Benchmark thresholds failed.",):
            results.append({
                "ruleId": "DRIFT004",
                "level": "error",
                "message": {"text": reason},
                "locations": [
                    {
                        "physicalLocation": {"artifactLocation": {"uri": artifact_uri}},
                        "logicalLocations": [{"name": "benchmark_gate", "kind": "suite"}],
                    }
                ],
                "partialFingerprints": {
                    "primaryLocationLineHash": _partial_fingerprint(
                        "DRIFT004", reason, report.baseline_model, report.candidate_model
                    )
                },
                "properties": {
                    "baseline_model": report.baseline_model,
                    "candidate_model": report.candidate_model,
                    "mean_composite_drift": round(report.mean_composite_drift, 4),
                    "candidate_json_conformity_rate": round(
                        report.candidate_json_conformity_rate, 4
                    ),
                },
            })

    sarif_doc: dict[str, Any] = {
        "$schema": _SARIF_SCHEMA,
        "version": "2.1.0",
        "runs": [
            {
                "tool": {
                    "driver": {
                        "name": "llm-drift-bench",
                        "version": __version__,
                        "informationUri": "https://github.com/anant-goel/llm-drift-bench",
                        "rules": _SARIF_RULES,
                    }
                },
                "invocations": [{"executionSuccessful": True}],
                "properties": {
                    "baseline_model": report.baseline_model,
                    "candidate_model": report.candidate_model,
                    "passed": report.passed,
                    "total_samples": report.total_samples,
                    "drifted_samples": report.drifted_samples,
                },
                "results": results,
            }
        ],
    }
    serialized = json.dumps(sarif_doc, indent=2) + "\n"
    if output_path is not None:
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(serialized, encoding="utf-8")
    return serialized
