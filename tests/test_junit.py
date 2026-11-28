"""JUnit output preserves engine gates and remains parseable for arbitrary IDs."""

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from llm_drift_bench import DriftBenchmarkEngine, ModelSample, export_junit_report
from llm_drift_bench.cli import main


def test_pass_has_prompt_and_gate_without_invented_time() -> None:
    samples = [ModelSample(prompt_id="p", output="private body")]
    root = ET.fromstring(export_junit_report(DriftBenchmarkEngine().evaluate(samples, samples)))
    assert root.attrib["tests"] == "2"
    assert root.attrib["failures"] == "0"
    assert all("time" not in case.attrib for case in root)
    assert "private body" not in ET.tostring(root, encoding="unicode")


def test_suite_only_latency_failure_is_reported() -> None:
    baseline = [ModelSample(prompt_id="p", output="same", latency_ms=10)]
    candidate = [ModelSample(prompt_id="p", output="same", latency_ms=100)]
    report = DriftBenchmarkEngine().evaluate(baseline, candidate)
    assert report.drifted_samples == 0
    root = ET.fromstring(export_junit_report(report))
    assert root.attrib["failures"] == "1"
    failure = root.find("./testcase[@name='benchmark_gate']/failure")
    assert failure is not None
    assert report.failure_reasons[0] in (failure.text or "")


def test_xml_escapes_ids_and_counts_prompt_failures() -> None:
    pid = 'x<&"\x00\ud800'
    report = DriftBenchmarkEngine().evaluate(
        [ModelSample(prompt_id=pid, output='{"x":1}')],
        [ModelSample(prompt_id=pid, output='{"x":999}')],
    )
    root = ET.fromstring(export_junit_report(report))
    assert root.attrib["failures"] == "2"
    assert root[0].attrib["name"] == 'x<&"\ufffd\ufffd'


def test_cli_saved_report_xml_matches_run_and_keeps_exit_code(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    baseline.write_text('[{"prompt_id":"p","output":"{\\"x\\":1}"}]')
    candidate.write_text('[{"prompt_id":"p","output":"{\\"x\\":999}"}]')
    saved = tmp_path / "report.json"
    assert main(["run", "-b", str(baseline), "-c", str(candidate),
                 "--format", "junit", "-o", str(saved)]) == 1
    emitted = ET.fromstring(capsys.readouterr().out)
    assert emitted.attrib["failures"] == "2"
    target = tmp_path / "reports" / "junit.xml"
    assert main(["report", "-i", str(saved), "--format", "junit",
                 "-o", str(target)]) == 0
    root = ET.parse(target).getroot()
    assert root.attrib["failures"] == "2"


def test_report_input_survives_output_collision(tmp_path: Path) -> None:
    report = tmp_path / "original.json"
    original = (Path(__file__).resolve().parents[1] / "examples/expected-pass.json").read_bytes()
    report.write_bytes(original)
    assert main(["report", "-i", str(report), "--format", "junit", "-o", str(report)]) == 2
    assert report.read_bytes() == original


def test_unicode_prompt_survives_ascii_console() -> None:
    prompt_id = chr(0x6771) + chr(0x4EAC)
    samples = [ModelSample(prompt_id=prompt_id, output="same")]
    xml = export_junit_report(DriftBenchmarkEngine().evaluate(samples, samples))
    assert ET.fromstring(xml.encode("ascii"))[0].attrib["name"] == prompt_id


def test_junit_includes_sanitized_model_properties() -> None:
    base = [ModelSample(prompt_id="p", model_name="base\x00m", output="ok")]
    cand = [ModelSample(prompt_id="p", model_name="cand-q4", output="ok")]
    root = ET.fromstring(export_junit_report(DriftBenchmarkEngine().evaluate(base, cand)))
    props = {p.attrib["name"]: p.attrib["value"] for p in root.findall("./properties/property")}
    assert props["baseline_model"] == "base\ufffdm"
    assert props["candidate_model"] == "cand-q4"
    assert props["mean_composite_drift"] == "0.0000"
    with pytest.raises(ValueError, match="canonical engine report"):
        export_junit_report({})  # type: ignore[arg-type]


def test_export_conversation_junit_report_and_cli(capsys: pytest.CaptureFixture[str]) -> None:
    from llm_drift_bench.conversation import analyze_conversation_drift, load_conversations
    from llm_drift_bench.junit import export_conversation_junit_report

    examples = Path(__file__).resolve().parents[1] / "examples"
    base = load_conversations(examples / "conversations-baseline.jsonl")
    drift = load_conversations(examples / "conversations-candidate-drift.jsonl")

    pass_root = ET.fromstring(
        export_conversation_junit_report(analyze_conversation_drift(base, base))
    )
    assert pass_root.attrib["tests"] == "3"
    assert pass_root.attrib["failures"] == "0"

    fail_xml = export_conversation_junit_report(analyze_conversation_drift(base, drift))
    fail_root = ET.fromstring(fail_xml)
    assert fail_root.attrib["failures"] == "2"
    assert "ORD-1042" not in fail_xml  # Never leaks raw turn bodies

    with pytest.raises(ValueError, match="ConversationDriftReport"):
        export_conversation_junit_report({})  # type: ignore[arg-type]

    assert (
        main(
            [
                "check-conversations",
                "-b",
                str(examples / "conversations-baseline.jsonl"),
                "-c",
                str(examples / "conversations-candidate-drift.jsonl"),
                "--format",
                "junit",
            ]
        )
        == 1
    )
    cli_root = ET.fromstring(capsys.readouterr().out)
    assert cli_root.attrib["name"] == "llm-drift-bench-conversations"
    assert cli_root.attrib["failures"] == "2"


def test_export_tool_trace_junit_report_and_cli(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from llm_drift_bench.junit import export_tool_trace_junit_report
    from llm_drift_bench.tool_events import ToolEvent, compare_tool_traces

    base = [ToolEvent(0, "read", "src/a.py@1", start=0, end=10)]
    cand = [
        ToolEvent(0, "read", "src/a.py@1", start=0, end=10),
        ToolEvent(1, "edit", "src/b.py@1", success=False),
    ]
    pass_root = ET.fromstring(export_tool_trace_junit_report(compare_tool_traces(base, base)))
    assert pass_root.attrib["tests"] == "4" and pass_root.attrib["failures"] == "0"

    fail_diff = compare_tool_traces(
        base, cand, max_call_increase=0, max_failed_increase=0, fail_on_new_edit_resources=True
    )
    fail_root = ET.fromstring(export_tool_trace_junit_report(fail_diff))
    assert fail_root.attrib["failures"] == "4"

    with pytest.raises(ValueError, match="ToolTraceDiff"):
        export_tool_trace_junit_report({})  # type: ignore[arg-type]

    b_file = tmp_path / "b.jsonl"
    c_file = tmp_path / "c.jsonl"
    b_file.write_text(
        '{"kind": "read", "resource": "src/a.py@1", "start": 0, "end": 10}\n', encoding="utf-8"
    )
    c_file.write_text(
        '{"kind": "read", "resource": "src/a.py@1", "start": 0, "end": 10}\n'
        '{"kind": "edit", "resource": "src/b.py@1"}\n',
        encoding="utf-8",
    )
    assert (
        main(
            [
                "diff-tools",
                "-b",
                str(b_file),
                "-c",
                str(c_file),
                "--fail-on-new-edit-resources",
                "--format",
                "junit",
            ]
        )
        == 1
    )
    cli_root = ET.fromstring(capsys.readouterr().out)
    assert cli_root.attrib["name"] == "llm-drift-bench-tool-traces"
    assert cli_root.attrib["failures"] == "2"


def test_export_quality_junit_report_and_cli(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from llm_drift_bench.junit import export_quality_junit_report
    from llm_drift_bench.quality_checks import run_quality_checks

    base = [ModelSample(prompt_id="p1", output="Helpful detailed answer.")]
    cand = [ModelSample(prompt_id="p1", output="I'm sorry, I cannot assist with that.")]

    pass_root = ET.fromstring(export_quality_junit_report(run_quality_checks(base, base)))
    assert pass_root.attrib["tests"] == "5" and pass_root.attrib["failures"] == "0"

    fail_root = ET.fromstring(export_quality_junit_report(run_quality_checks(base, cand)))
    assert fail_root.attrib["failures"] == "2"

    with pytest.raises(ValueError, match="QualityReport"):
        export_quality_junit_report({})  # type: ignore[arg-type]

    b_file = tmp_path / "qb.jsonl"
    c_file = tmp_path / "qc.jsonl"
    b_file.write_text('{"prompt_id": "p1", "output": "Helpful detailed answer."}\n', "utf-8")
    c_file.write_text(
        '{"prompt_id": "p1", "output": "I\'m sorry, I cannot assist with that."}\n', "utf-8"
    )
    assert main(["check-quality", "-b", str(b_file), "-c", str(c_file), "--format", "junit"]) == 1
    cli_root = ET.fromstring(capsys.readouterr().out)
    assert cli_root.attrib["name"] == "llm-drift-bench-quality"
    assert cli_root.attrib["failures"] == "2"


