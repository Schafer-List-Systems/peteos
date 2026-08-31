"""Test for OAP prompt-based agentic interaction via HelloPete."""

from peteos.oap.agentic_object import AgenticObject


class HelloPete(AgenticObject):
    """You are Pete, a helpful assistant."""


async def test_hello_pete_prompt():
    pete = HelloPete()
    result = await pete.invoke_agent("Hello, what's your name?")
    assert isinstance(result, str)
    assert "Pete" in result
