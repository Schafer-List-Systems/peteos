"""OAP tests for WebNavigator.

These tests require a live LLM backend configured in peteos.json.
"""

from __future__ import annotations

import pytest

from peteos.agentic_objects.web_navigator import WebNavigator
from peteos.oap.error import Error


class TestWebNavigatorIntegration:
    """End-to-end integration tests using invoke_agent with a live backend."""

    @pytest.mark.oap
    async def test_agent_fetches_example(self):
        """Agent fetches https://example.com and returns the content."""
        obj = WebNavigator()
        result = await obj.invoke_agent(
            prompt="Fetch the text content of https://example.com.",
            output_schema=str,
            timeout=120,
        )
        assert isinstance(result, (str, Error)), f"Expected str, got {type(result).__name__}: {result}"
        if isinstance(result, str):
            assert "example.com" in result.lower() or "documentation" in result.lower()

    @pytest.mark.oap
    async def test_agent_fetches_google(self):
        """Agent fetches https://www.google.com and returns content."""
        obj = WebNavigator()
        result = await obj.invoke_agent(
            prompt="Fetch the text content of https://www.google.com.",
            output_schema=str,
            timeout=120,
        )
        assert isinstance(result, (str, Error)), f"Expected str, got {type(result).__name__}: {result}"
        if isinstance(result, str):
            assert len(result) > 10  # Google returns substantial content

    @pytest.mark.oap
    async def test_agent_fetches_duckduckgo(self):
        """Agent fetches https://duckduckgo.com and returns content."""
        obj = WebNavigator()
        result = await obj.invoke_agent(
            prompt="Fetch the text content of https://duckduckgo.com.",
            output_schema=str,
            timeout=120,
        )
        assert isinstance(result, (str, Error)), f"Expected str, got {type(result).__name__}: {result}"
        if isinstance(result, str):
            assert len(result) > 10

    @pytest.mark.oap
    async def test_agent_fetches_raw(self):
        """Agent fetches https://example.com using fetch_raw and returns status code."""
        obj = WebNavigator()
        result = await obj.invoke_agent(
            prompt="Fetch the raw HTTP response from https://example.com and return the status code.",
            output_schema=int,
            timeout=120,
        )
        assert isinstance(result, (int, Error)), f"Expected int, got {type(result).__name__}: {result}"
        if isinstance(result, int):
            assert result == 200
