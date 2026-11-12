"""Command-line interface for llm-drift-bench."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import sys

from llm_drift_bench import __version__


def build_parser() -> argparse.ArgumentParser:
    """Build and return the top-level CLI ArgumentParser."""
    parser = argparse.ArgumentParser(
        prog="llm-drift-bench",
        description=(
            "Lightweight, zero-bloat CLI to benchmark LLM outputs, validate JSON "
            "conformity, and detect output drift across model updates or quantizations."
        ),
    )
    parser.add_argument(
        "-V",
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    parser.add_subparsers(dest="command", metavar="<command>")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Main entrypoint for the llm-drift-bench CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "command", None) is None:
        parser.print_help()
        return 0
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
