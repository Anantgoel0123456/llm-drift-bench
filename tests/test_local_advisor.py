"""Exercise the inference boundary independently of hardware availability."""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from llm_drift_bench import Finding
from llm_drift_bench.local_advisor import available_devices, generate_advice


@pytest.mark.parametrize("device", ["CPU", "GPU", "NPU"])
def test_local_inference_routes_only_summarized_findings(tmp_path: Path, device: str) -> None:
    sdk = Mock()
    pipeline = sdk.LLMPipeline.return_value
    pipeline.generate.return_value = " numeric_change "
    finding = Finding("numeric_change", "secret-id", "secret-fields", "secret-suggestion")
    with patch("llm_drift_bench.local_advisor.importlib.import_module", return_value=sdk):
        assert generate_advice([finding, Finding("schema", None, "", "")], tmp_path, device) == (
            "Model-selected checks:\n- secret-suggestion"
        )
    sdk.LLMPipeline.assert_called_once_with(str(tmp_path.resolve()), device)
    args, kwargs = pipeline.generate.call_args
    assert "numeric_change=1" in args[0] and "secret" not in args[0]
    if device == "CPU":
        sdk.StructuredOutputConfig.assert_called_once_with(regex="(numeric_change|schema)")
        assert kwargs.pop("structured_output_config") is sdk.StructuredOutputConfig.return_value
    assert kwargs == {"max_new_tokens": 32, "do_sample": False}


def test_discovery_uses_runtime_device_list() -> None:
    runtime = Mock()
    runtime.Core.return_value.available_devices = ["NPU", "CPU"]
    with patch("llm_drift_bench.local_advisor.importlib.import_module", return_value=runtime):
        assert available_devices() == ["CPU", "NPU"]


def test_missing_sdk_explains_installation(tmp_path: Path) -> None:
    with (
        patch("llm_drift_bench.local_advisor.importlib.import_module", side_effect=ImportError),
        pytest.raises(ValueError, match="ai extra"),
    ):
        generate_advice(
            [Finding("schema", None, "", ""), Finding("drift", None, "", "")], tmp_path, "GPU"
        )


@pytest.mark.parametrize(
    "reply", ["Measure accuracy and speed.", "schema,unknown", "schema,schema"]
)
def test_demo_rejects_ungrounded_model_advice(tmp_path: Path, reply: str) -> None:
    from llm_drift_bench import diagnose_report
    from llm_drift_bench.models import DriftBenchmarkReport
    from llm_drift_bench.report_io import load_report

    report = load_report(Path(__file__).resolve().parents[1] / "examples" / "expected-drift.json")
    assert isinstance(report, DriftBenchmarkReport)
    sdk = Mock()
    sdk.LLMPipeline.return_value.generate.return_value = reply
    with patch("llm_drift_bench.local_advisor.importlib.import_module", return_value=sdk):
        advice = generate_advice(diagnose_report(report), tmp_path, "CPU")
    assert advice.startswith("Model selection unavailable.")
    assert "required field" in advice and "decimal placement" in advice
    assert reply not in advice


@pytest.mark.parametrize("code", ["schema", "numeric_change", "missing_latency"])
def test_kaggle_cases_have_one_actionable_choice(tmp_path: Path, code: str) -> None:
    findings = [Finding(code, None, "", "Check this.")]
    if code != "missing_latency":
        findings.insert(0, Finding("suite", None, "", "Check gate."))
    with patch("llm_drift_bench.local_advisor.importlib.import_module") as load:
        assert generate_advice(findings, tmp_path, "CPU") == "Report-selected check:\n- Check this."
    load.assert_not_called()


@pytest.mark.parametrize("code", ["schema", "numeric_change"])
@pytest.mark.parametrize(
    ("reply", "heading"),
    [
        ('["CODE","drift"]', "Model-selected checks:"),
        ('["CODE","unknown"]', "Model selection unavailable. Showing report checks:"),
        ("An unrelated recommendation.", "Model selection unavailable. Showing report checks:"),
    ],
)
def test_selection_or_fallback_for_suite_first_input(
    tmp_path: Path, code: str, reply: str, heading: str
) -> None:
    findings = [Finding("suite", None, "", "Check gate."), Finding(code, "id", "", "Check field.")]
    findings.append(Finding("drift", "second", "", "Compare responses."))
    sdk = Mock()
    sdk.LLMPipeline.return_value.generate.return_value = reply.replace("CODE", code)
    with patch("llm_drift_bench.local_advisor.importlib.import_module", return_value=sdk):
        expected = [heading, "- Check field.", "- Compare responses."]
        if heading.startswith("Model selection unavailable"):
            expected.append("- Check gate.")
        assert generate_advice(findings, tmp_path, "GPU").splitlines() == expected


def test_local_advisor_parse_guards_device_validation_and_missing_extra(tmp_path: Path) -> None:
    from llm_drift_bench.local_advisor import _parse_selection, available_devices

    assert _parse_selection(123) == []
    assert _parse_selection("[invalid json") == []
    assert _parse_selection("[1, 2, 3]") == []

    with pytest.raises(ValueError, match="CPU, GPU or NPU"):
        generate_advice([], tmp_path, "TPU")
    with pytest.raises(ValueError, match="existing directory"):
        generate_advice([], tmp_path / "missing_model_dir", "CPU")
    assert generate_advice([], tmp_path, "CPU") == "No diagnostic checks to prioritize."

    with (
        patch("llm_drift_bench.local_advisor.importlib.import_module", side_effect=ImportError),
        pytest.raises(ValueError, match="Install the ai extra"),
    ):
        available_devices()

