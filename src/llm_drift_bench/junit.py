"""JUnit XML export of canonical engine decisions without captured response bodies."""

from __future__ import annotations

import xml.etree.ElementTree as ET

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
