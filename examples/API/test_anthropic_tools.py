"""Test Anthropic tool call handling."""

import asyncio
from peteos.chatbot import GenericChatBotResponse, AnthropicChatBot


async def test_anthropic_tool_use():
    """Test Anthropic tool_use block produces unified format matching OpenAI.

    Both providers should produce IDENTICAL uniform format:
    {"tool_calls": [{"index": 0, "type": "...", "id": "...", "name": "...", "arguments": "..."}]}
    """
    async def mock_stream():
        # content_block_start: tool_use with id, name (top-level index for Anthropic)
        yield 'data: {"index": 0, "type": "content_block_start", "content_block": {"type": "tool_use", "id": "toolu_123", "name": "calculate"}}'
        # content_block_delta: partial_json accumulation
        # The inner JSON needs proper escaping for JSON-in-JSON
        yield 'data: {"index": 0, "type": "content_block_delta", "delta": {"type": "input_json_delta", "partial_json": "{\\"expr\\": \\"2+2\\"}"}}'
        yield '[DONE]'

    response = GenericChatBotResponse(mock_stream(), AnthropicChatBot.RESPONSE_TRANSLATIONS)
    async for _ in response:
        pass

    print("=== Anthropic Tool Call (Unified Format) ===")
    print(f"Response data: {response.data}")

    # Verify unified format - tool_calls array structure matching OpenAI
    assert "tool_calls" in response.data, f"Expected 'tool_calls' key, got keys: {response.data.keys()}"

    tool_calls = response.data.get("tool_calls")
    assert isinstance(tool_calls, list), f"Expected tool_calls to be a list, got {type(tool_calls)}"
    assert len(tool_calls) > 0, "Expected at least one tool call in array"

    # First tool call should have index field for delta merge
    first_tool = tool_calls[0]
    assert "index" in first_tool, f"Expected 'index' field in tool_call, got: {first_tool}"
    assert first_tool["index"] == 0, f"Expected index=0, got {first_tool['index']}"

    # Verify all fields are present in unified format
    assert first_tool.get("type") == "tool_use", f"Expected type='tool_use', got {first_tool.get('type')}"
    assert first_tool.get("id") == "toolu_123", f"Expected id='toolu_123', got {first_tool.get('id')}"
    assert first_tool.get("name") == "calculate", f"Expected name='calculate', got {first_tool.get('name')}"
    assert first_tool.get("arguments") == '{"expr": "2+2"}', f"Expected arguments, got {first_tool.get('arguments')}"

    print("✅ Anthropic tool_use block produces unified format matching OpenAI")


if __name__ == "__main__":
    asyncio.run(test_anthropic_tool_use())