"""Check distribution validation against small standalone archives."""

import runpy
import tarfile
import zipfile
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_installed.py"
CHECK = runpy.run_path(str(SCRIPT))["check_archive_contents"]
ROOT = "llm_drift_bench-0.3.0"
DIST_INFO = ROOT + ".dist-info"


def test_directory_cannot_stand_in_for_marker(tmp_path: Path) -> None:
    archive = tmp_path / (ROOT + ".tar.gz")
    with tarfile.open(archive, "w:gz") as stream:
        entry = tarfile.TarInfo(f"{ROOT}/src/llm_drift_bench/py.typed")
        entry.type = tarfile.DIRTYPE
        stream.addfile(entry)
    with pytest.raises(SystemExit, match="py.typed"):
        CHECK(archive)


def test_unrelated_nested_marker_does_not_validate_wheel(tmp_path: Path) -> None:
    archive = tmp_path / (ROOT + "-py3-none-any.whl")
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr("examples/llm_drift_bench/py.typed", "")
    with pytest.raises(SystemExit, match="py.typed"):
        CHECK(archive)


def test_wheel_requires_license_under_its_own_metadata(tmp_path: Path) -> None:
    archive = tmp_path / (ROOT + "-py3-none-any.whl")
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr("llm_drift_bench/py.typed", "")
        stream.writestr("unrelated.dist-info/licenses/LICENSE", "other")
    with pytest.raises(SystemExit, match="LICENSE"):
        CHECK(archive)
