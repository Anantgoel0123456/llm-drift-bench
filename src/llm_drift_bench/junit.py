"""JUnit XML export of canonical engine decisions without captured response bodies."""

from __future__ import annotations

import xml.etree.ElementTree as ET

from llm_drift_bench.conversation import ConversationDriftReport
from llm_drift_bench.models import DriftBenchmarkReport


def _xml_text(value: str) -> str:
    """Replace characters forbidden by XML 1.0, including lone surrogates."""
    return "".join(
        char if (char in "\t\n\r" or 0x20 <= ord(char) <= 0xD7FF
                 or 0xE000 <= ord(char) <= 0xFFFD
                 or 0x10000 <= ord(char) <= 0x10FFFF) else "\ufffd"
        for char in value
    )


def export_junit_report(report: DriftBenchmarkReport) -> str:
    """Return XML with one case per prompt plus one overall benchmark gate.

    The extra gate preserves suite-only failures such as latency regression.
    No execution times are fabricated from model latency. Legacy evaluator
    reports are rejected because they do not retain the engine's gate verdict.
    """
    if not isinstance(report, DriftBenchmarkReport):
        raise ValueError("JUnit export requires a canonical engine report.")
    failures = sum(result.drifted for result in report.results) + int(not report.passed)
    root = ET.Element("testsuite", {
        "name": "llm-drift-bench", "tests": str(len(report.results) + 1),
        "failures": str(failures), "errors": "0", "skipped": "0",
    })
    for result in sorted(report.results, key=lambda result: result.prompt_id):
        case = ET.SubElement(root, "testcase", {
            "classname": "llm_drift_bench.prompts", "name": _xml_text(result.prompt_id),
        })
        if result.drifted:
            ET.SubElement(case, "failure", {
                "type": "DriftDetected", "message": "Prompt drift check failed",
            }).text = (
                f"Composite drift: {result.composite_drift_score:.4f}; "
                f"candidate schema passed: {result.candidate_validation.passed}."
            )
    gate = ET.SubElement(root, "testcase", {
        "classname": "llm_drift_bench.suite", "name": "benchmark_gate",
    })
    if not report.passed:
        ET.SubElement(gate, "failure", {
            "type": "BenchmarkFailed", "message": "Benchmark thresholds failed",
        }).text = _xml_text("\n".join(report.failure_reasons) or "Benchmark failed.")
    props = ET.SubElement(root, "properties")
    for prop_name, prop_value in (
        ("baseline_model", _xml_text(report.baseline_model)),
        ("candidate_model", _xml_text(report.candidate_model)),
        ("mean_composite_drift", f"{report.mean_composite_drift:.4f}"),
        ("candidate_json_conformity_rate", f"{report.candidate_json_conformity_rate:.4f}"),
    ):
        ET.SubElement(props, "property", {"name": prop_name, "value": prop_value})
    ET.indent(root, space="  ")
    serialized: bytes = ET.tostring(root, encoding="us-ascii", xml_declaration=True)
    return serialized.decode("ascii") + "\n"


def export_conversation_junit_report(report: ConversationDriftReport) -> str:
    """Export a ConversationDriftReport as JUnit XML without leaking raw turn contents."""
    if not isinstance(report, ConversationDriftReport):
        raise ValueError("Conversation JUnit export requires a ConversationDriftReport.")
    failures = sum(c.drifted for c in report.conversations) + int(not report.passed)
    root = ET.Element(
        "testsuite",
        {
            "name": "llm-drift-bench-conversations",
            "tests": str(len(report.conversations) + 1),
            "failures": str(failures),
            "errors": "0",
            "skipped": "0",
        },
    )
    for conv in sorted(report.conversations, key=lambda item: item.conversation_id):
        case = ET.SubElement(
            root,
            "testcase",
            {
                "classname": "llm_drift_bench.conversations",
                "name": _xml_text(conv.conversation_id),
            },
        )
        if conv.drifted:
            ET.SubElement(
                case,
                "failure",
                {
                    "type": "ConversationDriftDetected",
                    "message": "Multi-turn trajectory drift check failed",
                },
            ).text = (
                f"Max turn drift: {conv.max_turn_drift:.4f}; "
                f"first divergence turn: {conv.first_divergence_turn}; "
                f"refusal flip turns: {list(conv.refusal_flip_turns)}."
            )
    gate = ET.SubElement(
        root,
        "testcase",
        {"classname": "llm_drift_bench.suite", "name": "conversation_gate"},
    )
    if not report.passed:
        ET.SubElement(
            gate,
            "failure",
            {
                "type": "ConversationGateFailed",
                "message": "Conversation trajectory gate failed",
            },
        ).text = _xml_text("\n".join(report.failure_reasons) or "Conversation gate failed.")
    props = ET.SubElement(root, "properties")
    for prop_name, prop_value in (
        ("total_conversations", str(report.total_conversations)),
        ("drifted_conversations", str(report.drifted_conversations)),
        ("mean_trajectory_drift", f"{report.mean_trajectory_drift:.4f}"),
        ("total_refusal_flip_turns", str(report.total_refusal_flip_turns)),
    ):
        ET.SubElement(props, "property", {"name": prop_name, "value": prop_value})
    ET.indent(root, space="  ")
    serialized: bytes = ET.tostring(root, encoding="us-ascii", xml_declaration=True)
    return serialized.decode("ascii") + "\n"
