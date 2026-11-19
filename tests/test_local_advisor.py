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
    pipeline.generate.return_value = " Compare units. "
    finding = Finding("numeric_change", "secret-id", "secret-fields", "secret-suggestion")
    with patch("llm_drift_bench.local_advisor.importlib.import_module", return_value=sdk):
        assert generate_advice([finding], tmp_path, device) == "Compare units."
    sdk.LLMPipeline.assert_called_once_with(str(tmp_path.resolve()), device)
    args, kwargs = pipeline.generate.call_args
    assert "numeric_change=1" in args[0] and "secret" not in args[0]
    assert kwargs == {"max_new_tokens": 160, "do_sample": False}


def test_discovery_uses_runtime_device_list() -> None:
    runtime = Mock()
    runtime.Core.return_value.available_devices = ["NPU", "CPU"]
    with patch("llm_drift_bench.local_advisor.importlib.import_module", return_value=runtime):
        assert available_devices() == ["CPU", "NPU"]


def test_missing_sdk_explains_installation(tmp_path: Path) -> None:
    with patch("llm_drift_bench.local_advisor.importlib.import_module", side_effect=ImportError):
        with pytest.raises(ValueError, match="ai extra"):
            generate_advice([], tmp_path, "GPU")
