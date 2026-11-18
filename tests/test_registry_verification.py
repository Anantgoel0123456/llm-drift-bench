"""Verify registry download orchestration without network access or publishing."""

import runpy
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "verify_registry.py"


@pytest.mark.parametrize("timeout", [False, True])
def test_exact_testpypi_wheel_is_checked_after_retry(timeout: bool) -> None:
    calls = []

    def execute(command, **kwargs):
        calls.append(command)
        if "download" in command:
            if len(calls) == 1:
                if timeout:
                    raise subprocess.TimeoutExpired(command, 180)
                return subprocess.CompletedProcess(command, 1)
            destination = Path(command[command.index("--dest") + 1])
            (destination / "llm_drift_bench-0.2.0-py3-none-any.whl").write_bytes(b"fixture")
        return subprocess.CompletedProcess(command, 0)

    with (
        patch.object(sys, "argv", [str(SCRIPT), "testpypi", "0.2.0"]),
        patch("subprocess.run", side_effect=execute),
        patch("time.sleep") as sleep,
    ):
        runpy.run_path(str(SCRIPT), run_name="__main__")
    assert len(calls) == 3
    assert "https://test.pypi.org/simple/" in calls[0]
    assert "--no-deps" in calls[0]
    assert "llm-drift-bench==0.2.0" in calls[0]
    assert calls[-1][1].endswith("check_installed.py")
    sleep.assert_called_once_with(15)


def test_failed_download_never_runs_installation() -> None:
    with (
        patch.object(sys, "argv", [str(SCRIPT), "pypi", "0.2.0"]),
        patch("subprocess.run", return_value=subprocess.CompletedProcess([], 1)) as execute,
        patch("time.sleep"),
        pytest.raises(SystemExit, match="three attempts"),
    ):
        runpy.run_path(str(SCRIPT), run_name="__main__")
    assert execute.call_count == 3


def test_invalid_version_does_not_start_pip() -> None:
    with (
        patch.object(sys, "argv", [str(SCRIPT), "pypi", "https://example.com"]),
        patch("subprocess.run") as execute,
        pytest.raises(SystemExit),
    ):
        runpy.run_path(str(SCRIPT), run_name="__main__")
    execute.assert_not_called()
