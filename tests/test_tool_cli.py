"""check-tools command and Tool evidence in saved reports."""

import json
from pathlib import Path

import pytest

from llm_drift_bench import (
    DriftBenchmarkEngine,
    ModelSample,
    check_trace,
    evaluate_suite,
    export_html_report,
    export_markdown_report,
    load_tool_events,
)
from llm_drift_bench.cli import EXIT_TOOL_UNKNOWN, main

ROOT = Path(__file__).resolve().parents[1]
TRACE = str(ROOT / "examples" / "tool-trace" / "trace.jsonl")
RES = "guide.md@sha256:1f2e"


def _check(*extra: str) -> list[str]:
    return ["check-tools", "-i", TRACE, "--resource", RES, *extra]


@pytest.mark.parametrize(
    ("extra", "code", "status"),
    [
        (("--end", "1000", "--trace-complete"), 0, "PASS"),
        (("--end", "4000", "--trace-complete"), 1, "FAIL"),
        (("--end", "1000"), EXIT_TOOL_UNKNOWN, "UNKNOWN"),
        (("--end", "4000"), EXIT_TOOL_UNKNOWN, "UNKNOWN"),
    ],
)
def test_exit_codes_distinguish_pass_fail_and_unknown(
    capsys: pytest.CaptureFixture[str], extra: tuple[str, ...], code: int, status: str,
) -> None:
    assert main(_check(*extra)) == code
    out = capsys.readouterr().out
    assert out.startswith(f"read_before_edit: {status}")
    assert RES in out


def test_text_output_reports_the_question_and_the_evidence(
    capsys: pytest.CaptureFixture[str],
) -> None:
    main(_check("--start", "10", "--end", "4000", "--trace-complete"))
    out = capsys.readouterr().out
    assert "[10, 4000)" in out and "3990 characters" in out
    assert "returned before first edit: 990" in out
    assert "First edit sequence: 2" in out and "attested complete: yes" in out


def test_json_stdout_matches_the_saved_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    out_file = tmp_path / "nested" / "check.json"
    code = main(
        _check("--end", "4000", "--trace-complete", "--format", "json", "-o", str(out_file))
    )
    assert code == 1
    printed = json.loads(capsys.readouterr().out)
    assert printed == json.loads(out_file.read_text(encoding="utf-8"))
    assert printed["status"] == "FAIL" and printed["covered_characters"] == 1000


@pytest.mark.parametrize(
    "args",
    [
        ["--end", "0"],
        ["--start", "5", "--end", "5"],
        ["--start", "-1", "--end", "5"],
    ],
)
def test_invalid_ranges_are_errors_not_verdicts(
    capsys: pytest.CaptureFixture[str], args: list[str],
) -> None:
    assert main(_check(*args)) == 2
    assert "Error:" in capsys.readouterr().err


def test_missing_and_malformed_traces_are_errors(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["check-tools", "-i", str(tmp_path / "none.jsonl"),
                 "--resource", RES, "--end", "10"]) == 2
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"kind":"read","resource":"a","bogus":1}\n', encoding="utf-8")
    assert main(["check-tools", "-i", str(bad), "--resource", RES, "--end", "10"]) == 2
    assert "unknown event field" in capsys.readouterr().err


def _drift_report(tmp_path: Path, *, passing: bool) -> Path:
    base = [ModelSample(prompt_id="p", output='{"n": 1}')]
    cand = base if passing else [ModelSample(prompt_id="p", output='{"n": 9, "x": [1]}')]
    path = tmp_path / "report.json"
    path.write_text(DriftBenchmarkEngine().evaluate(base, cand).to_json(), encoding="utf-8")
    return path


def _saved_check(tmp_path: Path, name: str = "check.json", *extra: str) -> Path:
    path = tmp_path / name
    main(_check("--end", "4000", "--trace-complete", "-o", str(path), *extra))
    return path


@pytest.mark.parametrize("fmt", ["html", "markdown"])
def test_report_embeds_tool_evidence_without_changing_the_verdict(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], fmt: str,
) -> None:
    report = _drift_report(tmp_path, passing=True)
    check = _saved_check(tmp_path)
    capsys.readouterr()
    assert main(["report", "-i", str(report), "-f", fmt]) == 0
    plain = capsys.readouterr().out
    assert main(["report", "-i", str(report), "-f", fmt, "--tool-check", str(check)]) == 0
    with_tools = capsys.readouterr().out
    assert "Tool evidence" not in plain
    assert "Tool evidence" in with_tools and RES in with_tools and "FAIL" in with_tools
    assert "does not change the drift verdict" in with_tools
    assert "PASSED" in with_tools and "FAILED" not in with_tools


def test_run_attaches_evidence_and_keeps_its_own_exit_code(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    base = tmp_path / "base.jsonl"
    cand = tmp_path / "cand.jsonl"
    base.write_text('{"prompt_id":"p","output":"{\\"n\\": 1}"}\n', encoding="utf-8")
    cand.write_text('{"prompt_id":"p","output":"{\\"n\\": 1}"}\n', encoding="utf-8")
    passing = _saved_check(tmp_path)
    capsys.readouterr()
    args = ["run", "-b", str(base), "-c", str(cand), "--format", "html",
            "--tool-check", str(passing)]
    assert main(args) == 0
    assert "Tool evidence" in capsys.readouterr().out


def test_tool_check_rejects_unsupported_formats_and_legacy_reports(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    report = _drift_report(tmp_path, passing=True)
    check = _saved_check(tmp_path)
    capsys.readouterr()
    for fmt in ("json", "table", "console"):
        assert main(["report", "-i", str(report), "-f", fmt, "--tool-check", str(check)]) == 2
        assert "markdown or html" in capsys.readouterr().err
    legacy = evaluate_suite(
        [], [ModelSample(prompt_id="a", output="x")], [ModelSample(prompt_id="a", output="y")],
    )
    record = check_trace(load_tool_events(TRACE), resource=RES, required_start=0,
                         required_end=10, trace_complete=True)
    with pytest.raises(ValueError, match="engine reports"):
        export_markdown_report(legacy, tool_checks=[record])


def test_a_contradictory_saved_check_is_refused_by_report(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    report = _drift_report(tmp_path, passing=True)
    check = _saved_check(tmp_path)
    forged = json.loads(check.read_text(encoding="utf-8"))
    forged["status"] = "PASS"
    check.write_text(json.dumps(forged), encoding="utf-8")
    capsys.readouterr()
    assert main(["report", "-i", str(report), "-f", "html", "--tool-check", str(check)]) == 2
    assert "PASS requires" in capsys.readouterr().err


def test_untrusted_resource_names_are_escaped_in_html_and_markdown(tmp_path: Path) -> None:
    hostile = "<script>alert(1)</script>|`x`"
    record = check_trace(
        load_tool_events(TRACE), resource=hostile, required_start=0, required_end=5,
        trace_complete=True,
    )
    report = DriftBenchmarkEngine().evaluate(
        [ModelSample(prompt_id="p", output="{}")], [ModelSample(prompt_id="p", output="{}")],
    )
    page = export_html_report(report, tool_checks=[record])
    assert "<script>" not in page and "&lt;script&gt;" in page
    assert "Content-Security-Policy" in page
    markdown = export_markdown_report(report, tool_checks=[record])
    assert "<script>" not in markdown and "&#60;script&#62;" in markdown
    row = next(line for line in markdown.splitlines() if "&#60;script" in line)
    assert row.count("|") == 8


def test_unknown_evidence_is_displayed_as_unknown() -> None:
    record = check_trace(load_tool_events(TRACE), resource=RES, required_start=0, required_end=10)
    report = DriftBenchmarkEngine().evaluate(
        [ModelSample(prompt_id="p", output="{}")], [ModelSample(prompt_id="p", output="{}")],
    )
    page = export_html_report(report, tool_checks=[record])
    assert "UNKNOWN" in page and "Trace attested" in page and ">no<" in page
