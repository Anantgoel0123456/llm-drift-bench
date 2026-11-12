"""Module entrypoint for running `python -m llm_drift_bench`."""

from __future__ import annotations

import sys

from llm_drift_bench.cli import main

if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
