"""Test script for delta merge functionality."""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from peteos.utils.dict_path import (
    merge_delta_into_target,
    translate_delta_event,
    extract_with_indices,
    propagate_indices
)


def test_merge_delta():
    """Test merge_delta_into_target with realistic SSE deltas."""
    print("Testing merge_delta_into_target...")

    target = {}

    # First delta: role
    merge_delta_into_target(target, {'role': 'assistant'})
    assert target['role'] == 'assistant', f"Expected 'assistant', got {target.get('role')}"

    # Second delta: reasoning (string concatenation)
    merge_delta_into_target(target, {'reasoning': 'The'})
    assert target['reasoning'] == 'The', f"Expected 'The', got {target.get('reasoning')}"

    merge_delta_into_target(target, {'reasoning': ' user'})
    assert target['reasoning'] == 'The user', f"Expected 'The user', got {target.get('reasoning')}"

    merge_delta_into_target(target, {'reasoning': ' is'})
    assert target['reasoning'] == 'The user is', f"Expected 'The user is', got {target.get('reasoning')}"

    # Third delta: tool_calls (index-based array merge)
    merge_delta_into_target(target, {
        'tool_calls': [{
            'index': 0,
            'id': 'call_06042d4619a64303b5442d45',
            'type': 'function',
            'function': {'name': 'calculate', 'arguments': ''}
        }]
    })
    assert 'tool_calls' in target, "tool_calls not in target"
    assert target['tool_calls'][0]['id'] == 'call_06042d4619a64303b5442d45'
    assert target['tool_calls'][0]['function']['name'] == 'calculate'

    # Fourth delta: tool_calls arguments (merge into existing)
    merge_delta_into_target(target, {
        'tool_calls': [{
            'index': 0,
            'function': {'arguments': '{'}
        }]
    })
    assert target['tool_calls'][0]['function']['arguments'] == '{'

    # Fifth delta: tool_calls more arguments
    merge_delta_into_target(target, {
        'tool_calls': [{
            'index': 0,
            'function': {'arguments': '"expression": "2**16 + 32 * 15 - 100"'}
        }]
    })
    expected_args = '{"expression": "2**16 + 32 * 15 - 100"'
    assert target['tool_calls'][0]['function']['arguments'] == expected_args, \
        f"Expected {expected_args}, got {target['tool_calls'][0]['function']['arguments']}"

    # Sixth delta: tool_calls final brace
    merge_delta_into_target(target, {
        'tool_calls': [{
            'index': 0,
            'function': {'arguments': '}'}
        }]
    })
    expected_args = '{"expression": "2**16 + 32 * 15 - 100"}'
    assert target['tool_calls'][0]['function']['arguments'] == expected_args, \
        f"Expected {expected_args}, got {target['tool_calls'][0]['function']['arguments']}"

    print("✓ merge_delta_into_target passed")


def test_error_cases():
    """Test error handling."""
    print("\nTesting error cases...")

    # Array without index should raise
    try:
        merge_delta_into_target([], [{'function': {'name': 'test'}}])
        print("✗ Should have raised for array without index")
        sys.exit(1)
    except ValueError as e:
        if "missing 'index'" in str(e):
            print("✓ Correctly raised error for array without index")
        else:
            print(f"✗ Wrong error: {e}")
            sys.exit(1)

    # Number should raise
    try:
        target = {}
        merge_delta_into_target(target, {'temperature': 0.7})
        print("✗ Should have raised for numeric field")
        sys.exit(1)
    except ValueError as e:
        if "Numeric field" in str(e):
            print("✓ Correctly raised error for numeric field")
        else:
            print(f"✗ Wrong error: {e}")
            sys.exit(1)


def test_translate_delta():
    """Test translation of SSE event to uniform format."""
    print("\nTesting translate_delta_event...")

    # Simulate OpenAI SSE delta event
    event = {
        'choices': [{
            'index': 0,
            'delta': {
                'tool_calls': [{
                    'index': 0,
                    'function': {'name': 'calculate', 'arguments': '{'}
                }]
            }
        }]
    }

    translations = {
        'choices[*].delta.tool_calls': 'tool_calls',
        'choices[*].delta.role': 'role'
    }

    translated = translate_delta_event(event, translations)
    assert 'tool_calls' in translated, "tool_calls not translated"
    assert translated['tool_calls'][0]['index'] == 0, "index not preserved"
    assert translated['tool_calls'][0]['function']['name'] == 'calculate'
    print("✓ translate_delta_event passed")


def test_extract_with_indices():
    """Test index extraction from paths."""
    print("\nTesting extract_with_indices...")

    # Nested structure: choices[*].delta.content
    event = {
        'choices': [{
            'index': 0,
            'delta': {
                'content': 'Hello',
                'tool_calls': [{
                    'index': 0,
                    'function': {'name': 'test'}
                }]
            }
        }]
    }

    # Extract content using wildcard path (extract from first choice)
    result = extract_with_indices(event, ['choices[*]', 'delta', 'content'])
    assert result == 'Hello', f"Expected 'Hello', got {result}"

    # Extract tool_calls with wildcard and index preservation
    result = extract_with_indices(event, ['choices[*]', 'delta', 'tool_calls'])
    assert result is not None, "tool_calls not extracted"
    assert result[0]['index'] == 0, "index not preserved"
    print("✓ extract_with_indices passed")


def test_sse_stream_simulation():
    """Simulate the full Qwen3.5-35B SSE stream."""
    print("\nTesting full SSE stream simulation...")

    # Full SSE deltas from the stream
    deltas = [
        {'role': 'assistant'},
        {'reasoning': 'The'},
        {'reasoning': ' user'},
        {'reasoning': ' is'},
        {'reasoning': ' asking'},
        {'reasoning': ' me'},
        {'reasoning': ' to'},
        {'reasoning': ' calculate'},
        {'reasoning': ' a'},
        {'reasoning': ' mathematical'},
        {'reasoning': ' expression'},
        {'reasoning': ':'},
        {'reasoning': ' '},
        {'reasoning': '2'},
        {'reasoning': '**'},
        {'reasoning': '1'},
        {'reasoning': '6'},
        {'reasoning': ' +'},
        {'reasoning': ' '},
        {'reasoning': '3'},
        {'reasoning': '2'},
        {'reasoning': ' *'},
        {'reasoning': ' '},
        {'reasoning': '1'},
        {'reasoning': '5'},
        {'reasoning': ' -'},
        {'reasoning': ' '},
        {'reasoning': '1'},
        {'reasoning': '0'},
        {'reasoning': '0'},
        {'reasoning': '\n\n'},
        {'reasoning': 'This'},
        {'reasoning': ' is'},
        {'reasoning': ' an'},
        {'reasoning': ' arithmetic'},
        {'reasoning': ' expression'},
        {'reasoning': ' that'},
        {'reasoning': ' I'},
        {'reasoning': ' can'},
        {'reasoning': ' calculate'},
        {'reasoning': ' using'},
        {'reasoning': ' the'},
        {'reasoning': ' calculate'},
        {'reasoning': ' function'},
        {'reasoning': '.'},
        {'reasoning': ' Let'},
        {'reasoning': ' me'},
        {'reasoning': ' construct'},
        {'reasoning': ' the'},
        {'reasoning': ' expression'},
        {'reasoning': ' "'},
        {'reasoning': '2'},
        {'reasoning': '**'},
        {'reasoning': '1'},
        {'reasoning': '6'},
        {'reasoning': ' +'},
        {'reasoning': ' '},
        {'reasoning': '3'},
        {'reasoning': '2'},
        {'reasoning': ' *'},
        {'reasoning': ' '},
        {'reasoning': '1'},
        {'reasoning': '5'},
        {'reasoning': ' -'},
        {'reasoning': ' '},
        {'reasoning': '1'},
        {'reasoning': '0'},
        {'reasoning': '0'},
        {'reasoning': '"'},
        {'reasoning': '\n\n'},
        {'reasoning': 'Let'},
        {'reasoning': ' me'},
        {'reasoning': ' use'},
        {'reasoning': ' the'},
        {'reasoning': ' calculate'},
        {'reasoning': ' function'},
        {'reasoning': ' to'},
        {'reasoning': ' compute'},
        {'reasoning': ' this'},
        {'reasoning': '.'},
        {'reasoning': '\n'},
        {'content': '\n\n'},
        {'tool_calls': [{
            'index': 0,
            'id': 'call_06042d4619a64303b5442d45',
            'type': 'function',
            'function': {'name': 'calculate', 'arguments': ''}
        }]},
        {'tool_calls': [{
            'index': 0,
            'function': {'arguments': '{'}
        }]},
        {'tool_calls': [{
            'index': 0,
            'function': {'arguments': '"expression": "2**16 + 32 * 15 - 100"'}
        }]},
        {'tool_calls': [{
            'index': 0,
            'function': {'arguments': '}'}
        }]},
        {'content': ''}
    ]

    target = {}
    for delta in deltas:
        merge_delta_into_target(target, delta)

    # Verify accumulated results
    assert 'role' in target, "role missing"
    assert 'reasoning' in target, "reasoning missing"
    assert 'tool_calls' in target, "tool_calls missing"
    assert 'content' in target, "content missing"

    print(f"  role: {target['role']!r}")
    print(f"  reasoning length: {len(target['reasoning'])} chars")
    print(f"  tool_calls: {target['tool_calls']}")
    print(f"  content: {target['content']!r}")
    print("✓ Full SSE stream simulation passed")


if __name__ == '__main__':
    print("=" * 60)
    print("Delta Merge Functionality Tests")
    print("=" * 60)

    test_merge_delta()
    test_error_cases()
    test_translate_delta()
    test_extract_with_indices()
    test_sse_stream_simulation()

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
