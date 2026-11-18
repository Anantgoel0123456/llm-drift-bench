import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from llm_drift_bench.cli import main

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


@pytest.mark.parametrize(("candidate", "code", "verdict"), [
    ("candidate-pass.jsonl", 0, "PASSED"), ("candidate-drift.jsonl", 1, "FAILED"),
])
def test_job_summary_preserves_text_and_verdict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, candidate: str, code: int, verdict: str,
) -> None:
    summary = tmp_path / "summary.md"
    summary.write_text("Earlier check\n")
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    args = ["run", "-s", str(EXAMPLES / "extraction-suite.json"),
            "-b", str(EXAMPLES / "baseline.jsonl"), "-c", str(EXAMPLES / candidate)]
    junit = tmp_path / "reports" / "junit.xml"
    saved = tmp_path / "report.json"
    assert main([*args, "--github-summary", "--junit-output", str(junit), "-o", str(saved)]) == code
    assert json.loads(saved.read_text())["passed"] == (code == 0)
    assert int(ET.parse(junit).getroot().attrib["failures"]) == (0 if code == 0 else 3)
    assert summary.read_text().startswith(f"Earlier check\n\n## LLM drift comparison: {verdict}")


def test_check_tools_preserves_input(tmp_path: Path) -> None:
    original = (EXAMPLES / "tool-trace" / "trace.jsonl").read_bytes()
    trace = tmp_path / "trace.jsonl"
    trace.write_bytes(original)
    assert main(["check-tools", "-i", str(trace), "--resource", "guide", "--end", "10",
                 "-o", str(trace)]) == 2
    assert trace.read_bytes() == original


def test_run_rejects_shared_json_and_junit_destination(tmp_path: Path) -> None:
    target = tmp_path / "combined"
    baseline = str(EXAMPLES / "baseline.jsonl")
    assert main(["run", "-b", baseline, "-c", baseline, "-o", str(target),
                 "--junit-output", str(target)]) == 2
    assert not target.exists()
