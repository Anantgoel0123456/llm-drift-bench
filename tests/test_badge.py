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


def test_export_badge_from_json_and_cli_auto_detects_report_types(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import json

    from llm_drift_bench.audit import run_full_audit
    from llm_drift_bench.badge import export_badge_from_json
    from llm_drift_bench.cli import main
    from llm_drift_bench.conversation import analyze_conversation_drift, load_conversations
    from llm_drift_bench.models import ModelSample

    # Canonical report via CLI
    assert main(["badge", "-i", str(EXAMPLES / "expected-pass.json")]) == 0
    out_canon = capsys.readouterr().out
    assert "llm-drift: PASS 0.00" in out_canon

    # Audit report JSON (pass & fail)
    same = [ModelSample(prompt_id="p1", output='{"x": 1}')]
    diff = [ModelSample(prompt_id="p1", output="bad")]
    audit_pass_file = tmp_path / "audit-pass.json"
    audit_fail_file = tmp_path / "audit-fail.json"
    audit_pass_file.write_text(
        json.dumps(run_full_audit(same, same).to_dict()), encoding="utf-8"
    )
    audit_fail_file.write_text(
        json.dumps(run_full_audit(same, diff).to_dict()), encoding="utf-8"
    )
    assert main(["badge", "-i", str(audit_pass_file)]) == 0
    assert "llm-audit: PASS 0.00" in capsys.readouterr().out
    assert "FAIL (" in export_badge_from_json(audit_fail_file, label="release-audit")

    # Conversation report JSON
    base_c = load_conversations(EXAMPLES / "conversations-baseline.jsonl")
    drift_c = load_conversations(EXAMPLES / "conversations-candidate-drift.jsonl")
    conv_file = tmp_path / "conv-drift.json"
    conv_file.write_text(
        json.dumps(analyze_conversation_drift(base_c, drift_c).to_dict()),
        encoding="utf-8",
    )
    badge_out = tmp_path / "conv.svg"
    assert main(["badge", "-i", str(conv_file), "-o", str(badge_out)]) == 0
    assert "Wrote drift status badge" in capsys.readouterr().out
    svg_text = badge_out.read_text(encoding="utf-8")
    assert "conversations: FAIL" in svg_text and "#d73a49" in svg_text

    # Validation guards
    not_obj = tmp_path / "list.json"
    not_obj.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="object"):
        export_badge_from_json(not_obj)

    bad_audit = tmp_path / "bad_audit.json"
    bad_audit.write_text(
        json.dumps({"benchmark": {}, "distribution": {}, "passed": "yes"}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="AuditReport"):
        export_badge_from_json(bad_audit)

    bad_metric = tmp_path / "bad_metric.json"
    bad_metric.write_text(
        json.dumps(
            {"benchmark": {"mean_composite_drift": True}, "distribution": {}, "passed": True}
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="finite number"):
        export_badge_from_json(bad_metric)

    bad_conv = tmp_path / "bad_conv.json"
    bad_conv.write_text(
        json.dumps({"conversations": [], "mean_trajectory_drift": 0.1, "passed": "no"}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="ConversationDriftReport"):
        export_badge_from_json(bad_conv)


def test_export_tool_trace_badge_svg_and_cli_auto_detect(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import json

    from llm_drift_bench.badge import export_badge_from_json, export_tool_trace_badge_svg
    from llm_drift_bench.cli import main
    from llm_drift_bench.tool_events import compare_tool_traces
    from llm_drift_bench.tool_log import load_tool_events

    base = load_tool_events(EXAMPLES / "trace-baseline.jsonl")
    drift = load_tool_events(EXAMPLES / "trace-candidate-drift.jsonl")

    pass_svg = export_tool_trace_badge_svg(compare_tool_traces(base, base))
    assert "tool-trace: PASS 1.00" in pass_svg and "#2ea44f" in pass_svg

    fail_diff = compare_tool_traces(base, drift, fail_on_new_edit_resources=True)
    fail_svg = export_tool_trace_badge_svg(fail_diff, output_path=tmp_path / "tt.svg")
    assert "tool-trace: FAIL" in fail_svg and "#d73a49" in fail_svg

    with pytest.raises(TypeError, match="ToolTraceDiff"):
        export_tool_trace_badge_svg({})  # type: ignore[arg-type]

    tt_json = tmp_path / "trace-diff.json"
    tt_json.write_text(json.dumps(fail_diff.to_dict()), encoding="utf-8")
    assert main(["badge", "-i", str(tt_json)]) == 0
    assert "tool-trace: FAIL" in capsys.readouterr().out

    bad_tt = tmp_path / "bad_tt.json"
    bad_tt.write_text(
        json.dumps(
            {
                "baseline_calls": 1,
                "candidate_calls": 1,
                "action_jaccard": 1.0,
                "passed": "nope",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="ToolTraceDiff"):
        export_badge_from_json(bad_tt)


def test_export_consumer_and_rolling_badge_svg_and_cli_auto_detect(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import json

    from llm_drift_bench.badge import (
        export_badge_from_json,
        export_consumer_badge_svg,
        export_rolling_badge_svg,
    )
    from llm_drift_bench.cli import main
    from llm_drift_bench.consumer import ConsumerDecision, compare_consumer_decisions
    from llm_drift_bench.models import ModelSample
    from llm_drift_bench.rolling import analyze_rolling_distribution

    cons_ok = compare_consumer_decisions(
        [ConsumerDecision("p1", True)], [ConsumerDecision("p1", True)]
    )
    cons_fail = compare_consumer_decisions(
        [ConsumerDecision("p1", True)],
        [ConsumerDecision("p1", False, "err")],
        max_rejection_increase=0.1,
    )
    assert "consumer: PASS 0.0%" in export_consumer_badge_svg(cons_ok)
    assert "#d73a49" in export_consumer_badge_svg(cons_fail, output_path=tmp_path / "c.svg")
    with pytest.raises(TypeError, match="ConsumerComparison"):
        export_consumer_badge_svg({})  # type: ignore[arg-type]

    cons_json = tmp_path / "cons.json"
    cons_json.write_text(json.dumps(cons_fail.to_dict()), encoding="utf-8")
    assert main(["badge", "-i", str(cons_json)]) == 0
    assert "consumer: FAIL 100.0%" in capsys.readouterr().out

    bad_cons = tmp_path / "bad_cons.json"
    bad_cons.write_text(
        json.dumps(
            {
                "baseline_rejection_rate": 0.0,
                "candidate_rejection_rate": 0.1,
                "passed": "nope",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="ConsumerComparison"):
        export_badge_from_json(bad_cons)

    s1 = [ModelSample("p1", '{"k": "a"}')]
    s2 = [ModelSample("p1", '{"k": "b"}')]
    roll_ok = analyze_rolling_distribution([s1], s1, max_tvd=0.1)
    roll_fail = analyze_rolling_distribution([s1], s2, max_tvd=0.1)
    assert "rolling-dist: PASS TVD 0.00" in export_rolling_badge_svg(roll_ok)
    assert "#d73a49" in export_rolling_badge_svg(roll_fail, output_path=tmp_path / "r.svg")
    with pytest.raises(TypeError, match="RollingWindowReport"):
        export_rolling_badge_svg({})  # type: ignore[arg-type]

    roll_json = tmp_path / "roll.json"
    roll_json.write_text(json.dumps(roll_ok.to_dict()), encoding="utf-8")
    assert main(["badge", "-i", str(roll_json)]) == 0
    assert "rolling-dist: PASS TVD 0.00" in capsys.readouterr().out

    bad_roll = tmp_path / "bad_roll.json"
    bad_roll.write_text(
        json.dumps({"window_count": 1, "overall": "bad", "passed": True}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="RollingWindowReport"):
        export_badge_from_json(bad_roll)

