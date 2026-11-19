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
        assert generate_advice([finding], tmp_path, device) == (
            "Model-selected checks:\n- secret-suggestion"
        )
    sdk.LLMPipeline.assert_called_once_with(str(tmp_path.resolve()), device)
    args, kwargs = pipeline.generate.call_args
    assert "numeric_change=1" in args[0] and "secret" not in args[0]
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
        generate_advice([Finding("schema", None, "", "Check the schema.")], tmp_path, "GPU")


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
