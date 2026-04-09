"""Unit tests for delta merge functionality in peteos/utils/delta_merge.py."""

import pytest
from peteos.utils.delta_merge import (
    merge_delta_into_target,
    translate_delta_event,
)


class TestMergeDeltaIntoTarget:
    """Tests for the core merge_delta_into_target function."""

    def test_simple_string_concatenation(self):
        """Test that strings are concatenated during merge."""
        target = {}
        merge_delta_into_target(target, {"text": "Hello"})
        merge_delta_into_target(target, {"text": " World"})
        assert target["text"] == "Hello World"

    def test_simple_overwrite(self):
        """Test that non-string values are overwritten."""
        target = {}
        merge_delta_into_target(target, {"id": "chatcmpl-123"})
        assert target["id"] == "chatcmpl-123"

    def test_nested_dict_merge(self):
        """Test recursive merging of nested dictionaries."""
        target = {}
        merge_delta_into_target(
            target,
            {
                "tool_calls": {
                    "function": {"name": "calculate"}
                }
            },
        )
        assert target["tool_calls"]["function"]["name"] == "calculate"

    def test_array_with_index(self):
        """Test array merge with index field for positioning."""
        target = {"tool_calls": []}
        merge_delta_into_target(
            target,
            {
                "tool_calls": [
                    {
                        "index": 0,
                        "function": {"name": "calculate"}
                    }
                ]
            },
        )
        assert len(target["tool_calls"]) == 1
        assert target["tool_calls"][0]["function"]["name"] == "calculate"

    def test_array_index_expansion(self):
        """Test that arrays expand to accommodate index."""
        target = {"items": []}
        merge_delta_into_target(
            target,
            {
                "items": [
                    {
                        "index": 2,
                        "value": "third item"
                    }
                ]
            },
        )
        assert len(target["items"]) == 3
        assert target["items"][0] == {}
        assert target["items"][1] == {}
        assert target["items"][2]["value"] == "third item"

    def test_array_item_index_stripped(self):
        """Test that index field is not accumulated into result."""
        target = {"items": []}
        merge_delta_into_target(
            target,
            {
                "items": [
                    {
                        "index": 0,
                        "function": {"name": "test"}
                    }
                ]
            },
        )
        assert "index" not in target["items"][0]

    def test_array_missing_index_raises_error(self):
        """Test that array items without index field raise ValueError."""
        with pytest.raises(ValueError, match="missing 'index' field"):
            merge_delta_into_target(
                target=[],
                delta=[{"function": {"name": "test"}}],
            )

    def test_string_accumulation_in_array_item(self):
        """Test string concatenation within array items."""
        target = {"items": []}
        merge_delta_into_target(
            target,
            {
                "items": [
                    {
                        "index": 0,
                        "arguments": "{"
                    }
                ]
            },
        )
        merge_delta_into_target(
            target,
            {
                "items": [
                    {
                        "index": 0,
                        "arguments": '"expr":'
                    }
                ]
            },
        )
        assert target["items"][0]["arguments"] == '{"expr":'

    def test_numeric_value_raises_error(self):
        """Test that numeric values raise ValueError."""
        with pytest.raises(ValueError, match="Numeric field not handled"):
            merge_delta_into_target({}, {"count": 42})

    def test_standalone_scalar_raises_error(self):
        """Test that standalone scalars raise ValueError."""
        with pytest.raises(ValueError, match="Standalone scalar"):
            merge_delta_into_target(target={}, delta="hello")

    def test_multiple_array_items(self):
        """Test merging multiple array items at different indices."""
        target = {"items": []}
        merge_delta_into_target(
            target,
            {
                "items": [
                    {"index": 0, "name": "first"},
                    {"index": 1, "name": "second"}
                ]
            },
        )
        assert len(target["items"]) == 2
        assert target["items"][0]["name"] == "first"
        assert target["items"][1]["name"] == "second"

    def test_recurse_into_nested_array(self):
        """Test recursive merge into nested arrays."""
        target = {}
        merge_delta_into_target(
            target,
            {
                "tool_calls": [
                    {"index": 0, "function": {"name": "calc"}}
                ]
            },
        )
        assert target["tool_calls"][0]["function"]["name"] == "calc"

    def test_empty_delta(self):
        """Test that empty delta doesn't change target."""
        target = {"existing": "value"}
        merge_delta_into_target(target, {})
        assert target == {"existing": "value"}


class TestTranslateDeltaEvent:
    """Tests for the translate_delta_event function."""

    def test_simple_path_translation(self):
        """Test basic path translation from source to target key."""
        event = {
            "choices": [
                {
                    "delta": {"content": "Hello"}
                }
            ]
        }
        result = translate_delta_event(
            event,
            {"choices[*].delta.content": "text"}
        )
        assert result["text"] == "Hello"

    def test_multiple_path_translation(self):
        """Test translation of multiple paths in one event."""
        event = {
            "choices": [
                {
                    "delta": {
                        "reasoning": "Thinking",
                        "content": "Answer"
                    }
                }
            ]
        }
        result = translate_delta_event(
            event,
            {
                "choices[*].delta.reasoning": "reasoning",
                "choices[*].delta.content": "text"
            }
        )
        assert result["reasoning"] == "Thinking"
        assert result["text"] == "Answer"

    def test_tool_calls_translation(self):
        """Test translation of tool_calls with index preservation."""
        event = {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "function": {"name": "calculate"}
                            }
                        ]
                    }
                }
            ]
        }
        result = translate_delta_event(
            event,
            {"choices[*].delta.tool_calls": "tool_calls"}
        )
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["index"] == 0
        assert result["tool_calls"][0]["function"]["name"] == "calculate"

    def test_nested_index_propagation(self):
        """Test that nested index fields are preserved."""
        event = {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {"index": 0, "function": {"name": "test"}}
                        ]
                    }
                }
            ]
        }
        result = translate_delta_event(
            event,
            {"choices[*].delta.tool_calls": "tool_calls"}
        )
        # The nested tool_calls should preserve its index
        assert result["tool_calls"][0]["index"] == 0

    def test_missing_path_returns_none(self):
        """Test that missing paths return None and are not included."""
        event = {"choices": [{"delta": {"content": "Hello"}}]}
        result = translate_delta_event(
            event,
            {"choices[*].delta.missing_field": "missing"}
        )
        assert "missing" not in result

    def test_empty_event(self):
        """Test translation of empty event."""
        result = translate_delta_event({}, {"path.to.field": "target"})
        assert result == {}


class TestDeltaIntegration:
    """Integration tests combining translation and merge."""

    def test_full_openai_tool_call_flow(self):
        """Test complete OpenAI tool call accumulation flow."""
        target = {}

        # Event 1: Tool call name
        event1 = {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {"index": 0, "function": {"name": "calculate"}}
                        ]
                    }
                }
            ]
        }
        translated1 = translate_delta_event(event1, {"choices[*].delta.tool_calls": "tool_calls"})
        merge_delta_into_target(target, translated1)

        # Event 2: Tool call arguments (partial)
        event2 = {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {"index": 0, "function": {"arguments": '{"expr":'}}
                        ]
                    }
                }
            ]
        }
        translated2 = translate_delta_event(event2, {"choices[*].delta.tool_calls": "tool_calls"})
        merge_delta_into_target(target, translated2)

        # Event 3: Tool call arguments (complete)
        event3 = {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {"index": 0, "function": {"arguments": '"2+2"}'}}
                        ]
                    }
                }
            ]
        }
        translated3 = translate_delta_event(event3, {"choices[*].delta.tool_calls": "tool_calls"})
        merge_delta_into_target(target, translated3)

        assert target["tool_calls"][0]["function"]["name"] == "calculate"
        assert target["tool_calls"][0]["function"]["arguments"] == '{"expr":"2+2"}'

    def test_reasoning_and_text_accumulation(self):
        """Test accumulation of reasoning and text fields."""
        target = {}

        # Reasoning delta
        event1 = {
            "choices": [
                {"delta": {"reasoning": "Let me think"}}
            ]
        }
        translated1 = translate_delta_event(event1, {"choices[*].delta.reasoning": "reasoning"})
        merge_delta_into_target(target, translated1)

        # Text delta
        event2 = {
            "choices": [
                {"delta": {"content": "Hello"}}
            ]
        }
        translated2 = translate_delta_event(event2, {"choices[*].delta.content": "text"})
        merge_delta_into_target(target, translated2)

        assert target["reasoning"] == "Let me think"
        assert target["text"] == "Hello"


class TestEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_deeply_nested_structure(self):
        """Test merging deeply nested structures."""
        target = {}
        merge_delta_into_target(
            target,
            {
                "level1": {
                    "level2": {
                        "level3": {
                            "value": "deep"
                        }
                    }
                }
            },
        )
        assert target["level1"]["level2"]["level3"]["value"] == "deep"

    def test_mixed_array_and_dict(self):
        """Test merging mixed array and dict structures."""
        target = {}
        merge_delta_into_target(
            target,
            {
                "items": [
                    {"index": 0, "name": "item1"},
                    {"index": 1, "data": {"nested": "value"}}
                ]
            },
        )
        assert len(target["items"]) == 2
        assert target["items"][0]["name"] == "item1"
        assert target["items"][1]["data"]["nested"] == "value"

    def test_string_accumulation(self):
        """Test that strings are always concatenated, not overwritten."""
        target = {"count": "42"}
        merge_delta_into_target(target, {"count": "forty-two"})
        assert target["count"] == "42forty-two"
