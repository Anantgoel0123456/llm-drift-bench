"""Numeric limits shared by parsing and numeric comparison."""

from __future__ import annotations

import math


def finite_number(value: int | float) -> float:
    """Convert to a finite float, rejecting values outside the metric's range."""
    try:
        number = float(value)
    except OverflowError as exc:
        raise ValueError("Number exceeds the supported finite floating-point range.") from exc
    if not math.isfinite(number):
        raise ValueError("Numbers must be finite and within the supported floating-point range.")
    return number


def parse_finite_float(value: str) -> float:
    return finite_number(float(value))


def parse_finite_int(value: str) -> int:
    number = int(value)
    finite_number(number)
    return number


def reject_constant(value: str) -> None:
    raise ValueError(f"Nonstandard JSON numeric constant: {value}.")
