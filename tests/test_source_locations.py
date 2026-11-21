"""Actionable input errors preserve locations without printing capture content."""

from pathlib import Path

import pytest

from llm_drift_bench._file_io import read_utf8


@pytest.mark.parametrize("prefix", [b"", b"\xef\xbb\xbf"])
def test_invalid_byte_offset_includes_bom(tmp_path: Path, prefix: bytes) -> None:
    path = tmp_path / "input.json"
    path.write_bytes(prefix + b"abc\xff")
    with pytest.raises(ValueError, match=rf"offset {len(prefix) + 3}"):
        read_utf8(path)
