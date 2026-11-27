"""Tests for the offline static SVG drift status badge generator."""

from __future__ import annotations

from pathlib import Path

import pytest

from llm_drift_bench.badge import export_drift_badge_svg
from llm_drift_bench.report_io import load_engine_report

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def test_export_drift_badge_svg_pass_fail_escaping_and_file_write(tmp_path: Path) -> None:
    passed = load_engine_report(EXAMPLES / "expected-pass.json")
    drifted = load_engine_report(EXAMPLES / "expected-drift.json")

    pass_out = tmp_path / "badges" / "pass.svg"
    pass_svg = export_drift_badge_svg(passed, output_path=pass_out)
    assert pass_out.read_text(encoding="utf-8") == pass_svg
    assert pass_svg.startswith("<svg ")
    assert "#2ea44f" in pass_svg
    assert "PASS 0.00" in pass_svg
    assert "<script" not in pass_svg.lower()

    fail_svg = export_drift_badge_svg(drifted, label="prod <q4>")
    assert "#d73a49" in fail_svg
    assert "FAIL 0.67" in fail_svg
    assert "prod &lt;q4&gt;" in fail_svg

    with pytest.raises(TypeError, match="DriftBenchmarkReport"):
        export_drift_badge_svg({"passed": True})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-empty"):
        export_drift_badge_svg(passed, label="  ")


def test_export_audit_and_conversation_badge_svg(tmp_path: Path) -> None:
    from llm_drift_bench.audit import run_full_audit
    from llm_drift_bench.badge import export_audit_badge_svg, export_conversation_badge_svg
    from llm_drift_bench.conversation import (
        ConversationRecord,
        ConversationTurn,
        analyze_conversation_drift,
    )
    from llm_drift_bench.models import ModelSample

    same = [ModelSample(prompt_id="p1", output='{"x": 1}')]
    diff = [ModelSample(prompt_id="p1", output="bad")]
    ok_audit = run_full_audit(same, same)
    fail_audit = run_full_audit(same, diff)

    svg_ok = export_audit_badge_svg(ok_audit, output_path=tmp_path / "audit-ok.svg")
    assert "#2ea44f" in svg_ok and "PASS 0.00" in svg_ok
    svg_fail = export_audit_badge_svg(fail_audit)
    assert "#d73a49" in svg_fail and "FAIL (" in svg_fail
    with pytest.raises(TypeError, match="AuditReport"):
        export_audit_badge_svg({})  # type: ignore[arg-type]

    c_ok = ConversationRecord("c1", (ConversationTurn("assistant", '{"ok": true}'),))
    c_bad = ConversationRecord(
        "c1", (ConversationTurn("assistant", "I'm sorry, I cannot assist with that."),)
    )
    conv_pass = analyze_conversation_drift([c_ok], [c_ok])
    conv_fail = analyze_conversation_drift([c_ok], [c_bad])
    assert "PASS 0.00" in export_conversation_badge_svg(conv_pass)
    assert "#d73a49" in export_conversation_badge_svg(conv_fail)
    with pytest.raises(TypeError, match="ConversationDriftReport"):
        export_conversation_badge_svg({})  # type: ignore[arg-type]

