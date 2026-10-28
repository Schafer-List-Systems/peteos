"""Utilities for sanitizing text and data structures with problematic Unicode."""

from __future__ import annotations

import ftfy

from peteos.utils import get_logger

_logger = get_logger(__name__)


def _sanitize_text(text: str) -> str:
    """Sanitize text by repairing lone surrogates in-place.

    Some LLM outputs emit unpaired Unicode surrogates (e.g. \\uD83D \\uDCCB
    for an emoji) which decode fine but corrupt downstream UTF-8 serialization.
    ftfy repairs them by pairing back to proper Unicode code points.

    Args:
        text: The raw text string that may contain lone surrogates.

    Returns:
        The sanitized text with surrogates repaired where needed.
    """
    # Apply ftfy repair — this always returns a string (possibly the same object).
    fixed = ftfy.fix_text(text)

    # A change means the text had bad surrogates — warn and return the fixed copy.
    if fixed != text:
        _logger.warning(
            "[sanitizer] _sanitize_text: text repairs applied — "
            "malformed Unicode was detected and fixed",
        )
        return fixed

    # No change — return the original reference so no new string object is allocated.
    return text


def sanitize_encoding(obj: str | list | dict) -> tuple[str | None, int]:
    """Recursively sanitize a data structure by repairing lone surrogates in all strings.

    Walks lists and dicts in-place. For string leaves, applies _sanitize_text.
    Returns (replacement_string, 1) if the string was fixed, (None, 0) otherwise.
    For containers, mutates in-place and returns (None, total_replacement_count).

    Args:
        obj: The object to sanitize — may be a string, list, or dict.

    Returns:
        A tuple of (replacement, count). For strings: the fixed string or None,
        and 1 or 0. For containers: always None and the total count of fields
        that were repaired in-place.
    """
    # For a string leaf, apply _sanitize_text and return the fixed string or None.
    if isinstance(obj, str):
        fixed = _sanitize_text(obj)
        return (fixed, 1) if obj is not fixed else (None, 0)

    # For a list, iterate and recurse into each element in-place.
    if isinstance(obj, list):
        total = 0
        for i, item in enumerate(obj):
            # Recurse into each item — returns a fixed value and the count from this branch.
            result, count = sanitize_encoding(item)

            # If a replacement was returned, write it back in-place.
            if result is not None:
                obj[i] = result

            # Count already includes any string fixed in this branch — no extra increment.
            total += count

        return None, total

    # For a dict, iterate and recurse into each value in-place.
    if isinstance(obj, dict):
        total = 0
        for key, value in obj.items():
            # Recurse into each value — returns a fixed value and the count from this branch.
            result, count = sanitize_encoding(value)

            # If a replacement was returned, write it back in-place.
            if result is not None:
                obj[key] = result

            # Count already includes any string fixed in this branch — no extra increment.
            total += count

        return None, total

    # Non-container values are left as-is — no replacements needed.
    return None, 0
