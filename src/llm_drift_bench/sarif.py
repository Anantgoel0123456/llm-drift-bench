"""SARIF 2.1.0 exporter for GitHub Code Scanning and IDE diagnostic viewers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from llm_drift_bench import __version__
from llm_drift_bench.conversation import ConversationDriftReport
from llm_drift_bench.models import DriftBenchmarkReport
from llm_drift_bench.tool_events import ToolTraceDiff

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
                        "informationUri": "https://github.com/Anantgoel0123456/llm-drift-bench",
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


_CONV_SARIF_RULES: list[dict[str, Any]] = [
    {
        "id": "CONV001",
        "name": "ConversationRefusalFlip",
        "shortDescription": {
            "text": "Candidate refused a multi-turn prompt that baseline answered."
        },
        "defaultConfiguration": {"level": "error"},
    },
    {
        "id": "CONV002",
        "name": "ConversationTrajectoryDrift",
        "shortDescription": {
            "text": "Multi-turn conversation exceeded turn drift or turn-count gates."
        },
        "defaultConfiguration": {"level": "error"},
    },
    {
        "id": "CONV003",
        "name": "ConversationGateFailure",
        "shortDescription": {"text": "Aggregate multi-turn conversation trajectory gate failed."},
        "defaultConfiguration": {"level": "error"},
    },
]


def export_conversation_sarif_report(
    report: ConversationDriftReport,
    output_path: str | Path | None = None,
    *,
    artifact_uri: str = "conversations.jsonl",
) -> str:
    """Export ConversationDriftReport findings as OASIS SARIF 2.1.0 JSON without raw turn text."""
    if not isinstance(report, ConversationDriftReport):
        raise ValueError("Conversation SARIF export requires a ConversationDriftReport.")
    if not isinstance(artifact_uri, str) or not artifact_uri.strip():
        raise ValueError("artifact_uri must be a non-empty string.")

    results: list[dict[str, Any]] = []
    for conv in sorted(report.conversations, key=lambda item: item.conversation_id):
        if not conv.drifted:
            continue
        rule_id = "CONV001" if conv.refusal_flip_turns else "CONV002"
        if conv.refusal_flip_turns:
            turns = ", ".join(str(t) for t in conv.refusal_flip_turns)
            msg = (
                f"Conversation {conv.conversation_id!r} flipped to refusal at turn(s) {turns} "
                f"(max drift {conv.max_turn_drift:.4f})."
            )
        else:
            msg = (
                f"Conversation {conv.conversation_id!r} exceeded trajectory drift gate "
                f"(max drift {conv.max_turn_drift:.4f}, first div turn "
                f"{conv.first_divergence_turn})."
            )
        results.append(
            {
                "ruleId": rule_id,
                "level": "error",
                "message": {"text": msg},
                "locations": [
                    {
                        "physicalLocation": {"artifactLocation": {"uri": artifact_uri}},
                        "logicalLocations": [
                            {"name": conv.conversation_id, "kind": "conversation"}
                        ],
                    }
                ],
                "partialFingerprints": {
                    "primaryLocationLineHash": _partial_fingerprint(
                        rule_id, conv.conversation_id, "conversations", artifact_uri
                    )
                },
                "properties": {
                    "conversation_id": conv.conversation_id,
                    "max_turn_drift": round(conv.max_turn_drift, 4),
                    "first_divergence_turn": conv.first_divergence_turn,
                    "refusal_flip_turns": list(conv.refusal_flip_turns),
                },
            }
        )

    if not report.passed:
        for reason in report.failure_reasons or ("Conversation trajectory gate failed.",):
            results.append(
                {
                    "ruleId": "CONV003",
                    "level": "error",
                    "message": {"text": reason},
                    "locations": [
                        {
                            "physicalLocation": {"artifactLocation": {"uri": artifact_uri}},
                            "logicalLocations": [{"name": "conversation_gate", "kind": "suite"}],
                        }
                    ],
                    "partialFingerprints": {
                        "primaryLocationLineHash": _partial_fingerprint(
                            "CONV003", reason, "conversations", artifact_uri
                        )
                    },
                    "properties": {
                        "mean_trajectory_drift": round(report.mean_trajectory_drift, 4),
                        "total_refusal_flip_turns": report.total_refusal_flip_turns,
                    },
                }
            )

    sarif_doc: dict[str, Any] = {
        "$schema": _SARIF_SCHEMA,
        "version": "2.1.0",
        "runs": [
            {
                "tool": {
                    "driver": {
                        "name": "llm-drift-bench",
                        "version": __version__,
                        "informationUri": "https://github.com/Anantgoel0123456/llm-drift-bench",
                        "rules": _CONV_SARIF_RULES,
                    }
                },
                "invocations": [{"executionSuccessful": True}],
                "properties": {
                    "passed": report.passed,
                    "total_conversations": report.total_conversations,
                    "drifted_conversations": report.drifted_conversations,
                    "mean_trajectory_drift": round(report.mean_trajectory_drift, 4),
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


_TOOL_SARIF_RULES: list[dict[str, Any]] = [
    {
        "id": "TOOL001",
        "name": "ToolTraceUnexpectedEdit",
        "shortDescription": {
            "text": "Candidate agent edited a resource not edited in the baseline trace."
        },
        "defaultConfiguration": {"level": "error"},
    },
    {
        "id": "TOOL002",
        "name": "ToolTraceDriftGateFailure",
        "shortDescription": {
            "text": "Agent tool-trace comparison failed a configured drift gate."
        },
        "defaultConfiguration": {"level": "error"},
    },
]


def export_tool_trace_sarif_report(
    diff: ToolTraceDiff,
    output_path: str | Path | None = None,
    *,
    artifact_uri: str = "candidate-trace.jsonl",
) -> str:
    """Export agent ToolTraceDiff findings as OASIS SARIF 2.1.0 JSON."""
    if not isinstance(diff, ToolTraceDiff):
        raise ValueError("Tool-trace SARIF export requires a ToolTraceDiff.")
    if not isinstance(artifact_uri, str) or not artifact_uri.strip():
        raise ValueError("artifact_uri must be a non-empty string.")

    results: list[dict[str, Any]] = []
    if not diff.passed:
        edit_failed = any("edited new resource" in r for r in diff.failure_reasons)
        if edit_failed:
            for resource in diff.added_edit_resources:
                results.append(
                    {
                        "ruleId": "TOOL001",
                        "level": "error",
                        "message": {
                            "text": (
                                f"Candidate agent edited new resource {resource!r} "
                                "not edited in baseline trace."
                            )
                        },
                        "locations": [
                            {
                                "physicalLocation": {"artifactLocation": {"uri": artifact_uri}},
                                "logicalLocations": [{"name": resource, "kind": "resource"}],
                            }
                        ],
                        "partialFingerprints": {
                            "primaryLocationLineHash": _partial_fingerprint(
                                "TOOL001", resource, "tool-trace", artifact_uri
                            )
                        },
                        "properties": {
                            "resource": resource,
                            "action_jaccard": round(diff.action_jaccard, 4),
                        },
                    }
                )
        for reason in diff.failure_reasons or ("Agent tool-trace drift check failed.",):
            results.append(
                {
                    "ruleId": "TOOL002",
                    "level": "error",
                    "message": {"text": reason},
                    "locations": [
                        {
                            "physicalLocation": {"artifactLocation": {"uri": artifact_uri}},
                            "logicalLocations": [{"name": "tool_trace_gate", "kind": "suite"}],
                        }
                    ],
                    "partialFingerprints": {
                        "primaryLocationLineHash": _partial_fingerprint(
                            "TOOL002", reason, "tool-trace", artifact_uri
                        )
                    },
                    "properties": {
                        "baseline_calls": diff.baseline_calls,
                        "candidate_calls": diff.candidate_calls,
                        "call_delta": diff.call_delta,
                        "failed_delta": diff.failed_delta,
                        "action_jaccard": round(diff.action_jaccard, 4),
                    },
                }
            )

    sarif_doc: dict[str, Any] = {
        "$schema": _SARIF_SCHEMA,
        "version": "2.1.0",
        "runs": [
            {
                "tool": {
                    "driver": {
                        "name": "llm-drift-bench",
                        "version": __version__,
                        "informationUri": "https://github.com/Anantgoel0123456/llm-drift-bench",
                        "rules": _TOOL_SARIF_RULES,
                    }
                },
                "invocations": [{"executionSuccessful": True}],
                "properties": {
                    "passed": diff.passed,
                    "baseline_calls": diff.baseline_calls,
                    "candidate_calls": diff.candidate_calls,
                    "call_delta": diff.call_delta,
                    "failed_delta": diff.failed_delta,
                    "action_jaccard": round(diff.action_jaccard, 4),
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
