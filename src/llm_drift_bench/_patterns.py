"""Conservative regex subset with no branching or nested repetition."""

from __future__ import annotations

MAX_PATTERN_LENGTH = 256
MAX_PATTERN_INPUT = 16_384


def pattern_policy_error(pattern: str) -> str | None:
    """Check syntax features after re.compile has validated the expression.

    At most one variable repetition is allowed, and only in a start-anchored
    expression. No groups, alternation, counted repetition or backreferences
    are supported. This avoids backtracking trees and repeated unanchored scans.
    """
    if len(pattern) > MAX_PATTERN_LENGTH:
        return f"pattern exceeds {MAX_PATTERN_LENGTH} characters."
    in_class = False
    repeats = 0
    index = 0
    while index < len(pattern):
        char = pattern[index]
        if char == "\\":
            index += 1
            escaped = pattern[index]
            if escaped.isalnum() and escaped not in "dDsSwW":
                return "Only character-class escapes and escaped punctuation are supported."
        elif in_class:
            if char == "]":
                in_class = False
        elif char == "[":
            in_class = True
        elif char in "()|{}?":
            return "Groups, alternation, counted/optional/lazy repetition are unsupported."
        elif char in "*+":
            repeats += 1
            if repeats > 1 or not pattern.startswith("^"):
                return "Only one repetition is supported, in a pattern beginning with ^."
        index += 1
    return None
