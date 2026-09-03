"""OAP tests for BashWorkspace.

These tests require a live LLM backend configured in peteos.json.
"""

from __future__ import annotations

import pytest

from peteos.agentic_objects.bash_workspace import BashWorkspace
from peteos.oap.error import Error
from peteos.oap.token_counter import RecursiveTokenCounter
from dataclasses import dataclass


@dataclass
class HashResult:
    """Schema for file hash output."""
    hex_digest: str


class TestBashWorkspaceIntegration:
    """End-to-end integration tests using invoke_agent with a live backend."""

    @pytest.mark.oap
    async def test_agent_counts_lines(self):
        """Agent writes a file, runs wc -l, returns count."""
        obj = BashWorkspace()
        result = await obj.invoke_agent(
            prompt=(
                "Write file 'lines.txt' with:\n"
                "red\ngreen\nblue\nyellow\npurple\n\n"
                "Count the lines with bash and return the count."
            ),
            output_schema=int,
            timeout=90,
        )
        assert isinstance(result, (int, Error)), f"Expected int, got {type(result).__name__}: {result}"
        assert result == 5

    @pytest.mark.oap
    async def test_agent_counts_words(self):
        """Agent writes a file, counts words, returns total."""
        obj = BashWorkspace()
        result = await obj.invoke_agent(
            prompt=(
                "Write file 'text.txt' with:\n"
                "the quick brown fox jumps over the lazy dog\n\n"
                "Count the words and return the number."
            ),
            output_schema=int,
            timeout=90,
        )
        assert isinstance(result, (int, Error)), f"Expected int, got {type(result).__name__}: {result}"
        assert result == 9

    @pytest.mark.oap
    async def test_agent_grep_files(self):
        """Agent writes a file, greps for lines starting with 'a', returns list."""
        obj = BashWorkspace()
        result = await obj.invoke_agent(
            prompt=(
                "Write file 'words.txt' with:\n"
                "apple\nbanana\napricot\ncherry\navocado\n\n"
                "Find all lines starting with 'a', sort them, "
                "and return the result as a list."
            ),
            output_schema=list,
            timeout=90,
        )
        assert isinstance(result, (list, Error)), f"Expected list, got {type(result).__name__}: {result}"
        assert isinstance(result, list)
        assert sorted(result) == sorted(["apple", "apricot", "avocado"])

    @pytest.mark.oap
    async def test_agent_sha256_hash(self):
        """Agent writes a file, computes sha256 hash, returns via HashResult."""
        obj = BashWorkspace()
        result = await obj.invoke_agent(
            prompt=(
                "Write file 'hashme.txt' with content 'hello'. "
                "Compute its sha256 hash. "
                "Return a HashResult with the hex digest."
            ),
            output_schema=HashResult,
            timeout=None,
        )
        assert isinstance(result, (HashResult, Error)), f"Expected HashResult, got {type(result).__name__}: {result}"
        assert isinstance(result, HashResult)
        assert len(result.hex_digest) == 64
        assert result.hex_digest == "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"

    @pytest.mark.oap
    async def test_agent_sha256_hash_with_token_counting(self):
        """Same as test_agent_sha256_hash but with RecursiveTokenCounter tracking."""
        obj = BashWorkspace()
        counter, hooks = RecursiveTokenCounter.make_counter()
        result = await obj.invoke_agent(
            prompt=(
                "Write file 'hashme.txt' with content 'hello'. "
                "Compute its sha256 hash. "
                "Return a HashResult with the hex digest."
            ),
            output_schema=HashResult,
            timeout=None,
            hooks=hooks,
        )
        assert isinstance(result, (HashResult, Error)), f"Expected HashResult, got {type(result).__name__}: {result}"
        assert isinstance(result, HashResult)
        assert len(result.hex_digest) == 64
        assert result.hex_digest == "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
        assert counter.total_delta <= 2000, f"Token count {counter.total_delta} exceeds 2000"
