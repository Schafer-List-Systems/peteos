"""
Test script to verify Anthropic API integration with delta merge.
Tests the translation table against the captured SSE stream.
"""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from peteos.utils.dict_path import (
    translate_delta_event,
    merge_delta_into_target
)


def test_anthropic_translation():
    """Test translation of Anthropic events."""
    print("Testing Anthropic translation...")

    # Sample Anthropic events from the stream
    events = [
        # content_block_start for thinking (index 0)
        {
            "type": "content_block_start",
            "content_block": {"type": "thinking", "thinking": ""},
            "index": 0
        },
        # content_block_delta for thinking
        {
            "type": "content_block_delta",
            "delta": {"type": "thinking_delta", "thinking": "The"},
            "index": 0
        },
        # content_block_start for tool_use (index 1)
        {
            "type": "content_block_start",
            "content_block": {"type": "tool_use"},
            "index": 1
        },
        # content_block_delta for tool_use arguments
        {
            "type": "content_block_delta",
            "delta": {"type": "input_json_delta", "partial_json": '{"expression": "'},
            "index": 1
        },
        # message_delta for stop reason
        {
            "type": "message_delta",
            "delta": {"stop_reason": "tool_use"},
            "usage": {"input_tokens": 294, "output_tokens": 123}
        }
    ]

    translations = {
        "content_block_delta.delta.thinking_delta.thinking": "reasoning",
        "content_block_delta.delta.input_json_delta.partial_json": "tool_arguments",
        "content_block_start.content_block.type": "tool_type",
    }

    # Test each translation
    # Thinking delta
    result = translate_delta_event(events[1], translations)
    assert "reasoning" in result, f"Expected 'reasoning' in {result}"
    assert result["reasoning"] == "The", f"Expected 'The', got {result['reasoning']}"
    print(f"  Thinking delta: {result}")

    # Tool start
    result = translate_delta_event(events[2], translations)
    assert "tool_type" in result, f"Expected 'tool_type' in {result}"
    assert result["tool_type"] == "tool_use", f"Expected 'tool_use', got {result['tool_type']}"
    print(f"  Tool start: {result}")

    # Tool arguments
    result = translate_delta_event(events[3], translations)
    assert "tool_arguments" in result, f"Expected 'tool_arguments' in {result}"
    assert result["tool_arguments"] == '{"expression": "', f"Expected '{{\"expression\": \"', got {result['tool_arguments']}"
    print(f"  Tool arguments: {result}")

    print("✓ Anthropic translation passed")


def test_anthropic_merge():
    """Test merging Anthropic events."""
    print("\nTesting Anthropic merge...")

    translations = {
        "content_block_delta.delta.thinking_delta.thinking": "reasoning",
        "content_block_delta.delta.input_json_delta.partial_json": "tool_arguments",
    }

    # Simulate stream processing
    target = {}

    # Process thinking deltas
    thinking_chunks = [
        "The user wants me",
        " to calculate",
        ":",
        " 2**16 + 32 * 15 - 100"
    ]

    for chunk in thinking_chunks:
        event = {
            "type": "content_block_delta",
            "delta": {"type": "thinking_delta", "thinking": chunk},
            "index": 0
        }
        translated = translate_delta_event(event, translations)
        merge_delta_into_target(target, translated)

    assert target["reasoning"] == "The user wants me to calculate:  2**16 + 32 * 15 - 100"
    print(f"  Accumulated reasoning: {target['reasoning']!r}")

    # Process tool arguments
    tool_args = [
        '{"expression": "2',
        '**16',
        ' +',
        ' 32',
        ' *',
        ' 15',
        ' -',
        ' 100"}'
    ]

    for arg in tool_args:
        event = {
            "type": "content_block_delta",
            "delta": {"type": "input_json_delta", "partial_json": arg},
            "index": 1
        }
        translated = translate_delta_event(event, translations)
        merge_delta_into_target(target, translated)

    expected_json = '{"expression": "2**16 + 32 * 15 - 100"}'
    assert target["tool_arguments"] == expected_json
    print(f"  Accumulated tool_arguments: {target['tool_arguments']!r}")

    print("✓ Anthropic merge passed")


def test_from_stream_file():
    """Test parsing actual stream file."""
    print("\nTesting from stream file...")

    stream_path = "/home/frygge/projects/private/peteos/examples/API/Anthropic/raw_sse_stream.txt"

    translations = {
        "content_block_delta.delta.thinking_delta.thinking": "reasoning",
        "content_block_delta.delta.input_json_delta.partial_json": "tool_arguments",
        "content_block_start.content_block.type": "tool_type",
        "message_delta.delta.stop_reason": "stop_reason",
    }

    target = {}

    with open(stream_path, "r") as f:
        line_buffer = ""
        for line in f:
            line = line.rstrip('\n')

            if line.startswith("data: "):
                data_str = line[6:]
                if data_str.strip():
                    try:
                        event = json.loads(data_str)
                        translated = translate_delta_event(event, translations)
                        merge_delta_into_target(target, translated)
                    except json.JSONDecodeError:
                        pass

    print(f"  Final data: {target}")
    print(f"  reasoning length: {len(target.get('reasoning', ''))} chars")
    print(f"  tool_arguments: {target.get('tool_arguments', '')[:50]!r}...")
    print(f"  stop_reason: {target.get('stop_reason')}")

    assert "reasoning" in target, "reasoning missing from accumulated data"
    assert "tool_arguments" in target, "tool_arguments missing from accumulated data"
    assert target["stop_reason"] == "tool_use"

    print("✓ Stream file parsing passed")


if __name__ == '__main__':
    print("=" * 60)
    print("Anthropic API Integration Tests")
    print("=" * 60)

    test_anthropic_translation()
    test_anthropic_merge()
    test_from_stream_file()

    print("\n" + "=" * 60)
    print("All Anthropic tests passed!")
    print("=" * 60)