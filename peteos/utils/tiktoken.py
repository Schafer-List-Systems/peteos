"""Tiktoken utilities for counting tokens."""

import tiktoken


def count_tiktoken(text: str, encoding: str = "cl100k_base") -> int:
    """
    Count tokens in a string using tiktoken.

    Args:
        text: The text to count tokens for
        encoding: The tiktoken encoding to use (default: cl100k_base)

    Returns:
        The token count as an integer

    Raises:
        ImportError: If tiktoken is not installed
        ValueError: If the encoding is not recognized
    """
    try:
        encoder = tiktoken.get_encoding(encoding)
    except KeyError as e:
        raise ValueError(f"Unknown encoding: {encoding}") from e

    return len(encoder.encode(text))
