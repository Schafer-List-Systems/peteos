"""Tests for GeminiChatBot._build_body."""

from unittest.mock import MagicMock

import pytest

from peteos.utils import json

from peteos.chatbot.geminichatbot import GeminiChatBot
from peteos.chatbot.chatbotconfig import ChatBotConfig
from peteos.conversation.message import Message, ContentPart
from peteos.conversation.context import Context


class TestGeminiToolResultMatching:
    """Verify tool_result messages are matched to the correct functionCall name."""

    @pytest.fixture
    def chatbot(self):
        http_client = MagicMock()
        config = ChatBotConfig(
            name="gemini",
            model="gemini-pro",
            api_key="test-key",
            url="https://generativelanguage.googleapis.com",
        )
        return GeminiChatBot(http_client, config)

    def test_tool_result_matches_tool_use_from_model_response(self, chatbot):
        """tool_result should find the function name from the preceding assistant message's tool_use part."""
        assistant_msg = Message.create(
            role="assistant",
            content_parts=[
                ContentPart.create_tool_use(
                    call_id="test-1",
                    name="bash_exec",
                    arguments=json.dumps({"command": "ls"}),
                )
            ],
        )
        tool_result_msg = Message.create(
            role="tool_result",
            content_parts=[
                ContentPart.create_tool_result(
                    call_id="test-1",
                    content="file1 file2",
                )
            ],
        )
        context = MagicMock(spec=Context)
        context.messages = [assistant_msg, tool_result_msg]

        body = chatbot._build_body(context)

        # Find the user message that should contain functionResponse
        user_msgs = [m for m in body["contents"] if m["role"] == "user"]
        assert len(user_msgs) == 1

        parts = user_msgs[0]["parts"]
        assert len(parts) == 1
        assert "functionResponse" in parts[0]
        assert parts[0]["functionResponse"]["name"] == "bash_exec"
        assert parts[0]["functionResponse"]["response"]["content"] == "file1 file2"

    def test_tool_result_after_tool_use_and_text(self, chatbot):
        """tool_result should match tool_use even when model message has text + tool_use."""
        assistant_msg = Message.create(
            role="assistant",
            content_parts=[
                ContentPart.create_text("Let me run a command."),
                ContentPart.create_tool_use(
                    call_id="test-2",
                    name="bash_exec",
                    arguments=json.dumps({"command": "pwd"}),
                ),
            ],
        )
        tool_result_msg = Message.create(
            role="tool_result",
            content_parts=[
                ContentPart.create_tool_result(call_id="test-2", content="/home"),
            ],
        )
        context = MagicMock(spec=Context)
        context.messages = [assistant_msg, tool_result_msg]

        body = chatbot._build_body(context)

        user_msgs = [m for m in body["contents"] if m["role"] == "user"]
        assert len(user_msgs) == 1
        # Should contain both text and functionResponse parts
        func_parts = [p for p in user_msgs[0]["parts"] if "functionResponse" in p]
        assert len(func_parts) == 1
        assert func_parts[0]["functionResponse"]["name"] == "bash_exec"

    def test_tool_result_without_matching_functionCall_is_skipped(self, chatbot):
        """If a tool_result has no matching functionCall, log a warning and skip it."""
        tool_result_msg = Message.create(
            role="tool_result",
            content_parts=[
                ContentPart.create_tool_result(call_id="x", content="err"),
            ],
        )
        context = MagicMock(spec=Context)
        context.messages = [tool_result_msg]

        body = chatbot._build_body(context)

        # No contents at all — the unmatched tool_result was skipped
        assert "contents" not in body or len(body.get("contents", [])) == 0

    def test_multiple_tool_results_in_sequence(self, chatbot):
        """Multiple consecutive tool_results should each append to user message,
        matching the order of functionCalls."""
        assistant_msg = Message.create(
            role="assistant",
            content_parts=[
                ContentPart.create_tool_use(call_id="a", name="tool_a", arguments="{}"),
                ContentPart.create_tool_use(call_id="b", name="tool_b", arguments="{}"),
            ],
        )
        tool_result_a = Message.create(
            role="tool_result",
            content_parts=[
                ContentPart.create_tool_result(call_id="a", content="result_a"),
            ],
        )
        tool_result_b = Message.create(
            role="tool_result",
            content_parts=[
                ContentPart.create_tool_result(call_id="b", content="result_b"),
            ],
        )
        context = MagicMock(spec=Context)
        context.messages = [assistant_msg, tool_result_a, tool_result_b]

        body = chatbot._build_body(context)

        # Should have: 1 model message, 1 user message (all functionResponses merged)
        assistant_msgs = [m for m in body["contents"] if m["role"] == "model"]
        user_msgs = [m for m in body["contents"] if m["role"] == "user"]
        assert len(assistant_msgs) == 1
        assert len(user_msgs) == 1
        assert len(user_msgs[0]["parts"]) == 2
        assert user_msgs[0]["parts"][0]["functionResponse"]["name"] == "tool_a"
        assert user_msgs[0]["parts"][1]["functionResponse"]["name"] == "tool_b"

    def test_full_conversation_roundtrip(self, chatbot):
        """Test a full multi-turn conversation with alternating roles."""
        user_msg = Message.create(
            role="user",
            content_parts=[ContentPart.create_text("What's 2+3?")],
        )
        assistant_msg = Message.create(
            role="assistant",
            content_parts=[
                ContentPart.create_tool_use(call_id="t1", name="add", arguments='{"a": 2, "b": 3}'),
            ],
        )
        tool_result_msg = Message.create(
            role="tool_result",
            content_parts=[ContentPart.create_tool_result(call_id="t1", content="5")],
        )
        # Model's next response
        assistant_msg2 = Message.create(
            role="assistant",
            content_parts=[ContentPart.create_text("The answer is 5.")],
        )
        context = MagicMock(spec=Context)
        context.messages = [user_msg, assistant_msg, tool_result_msg, assistant_msg2]

        body = chatbot._build_body(context)

        contents = body["contents"]
        # Should alternate: user -> model -> user -> model
        roles = [m["role"] for m in contents]
        assert roles == ["user", "model", "user", "model"]
