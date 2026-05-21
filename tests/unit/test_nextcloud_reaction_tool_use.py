"""Test that tool_use content parts are properly extracted for reaction matching.

Anthropic's API returns tool call content parts with type='tool_use', not
'tool_call' or 'tool_calls'. The send() method tracks tool_call_id in _sent_parts
so the reaction handler can find pending tool calls.
"""

import asyncio
import uuid
from unittest.mock import MagicMock

import pytest

from peteos.channels.nextcloud_talk_channel import NextcloudTalkChannel
from peteos.chatbot import Message, ContentPart


class MockAgent:
    """Minimal mock agent for testing channel send/reaction."""

    def __init__(self):
        self._sessions = {}
        self._channels = {}

    def register_channel(self, channel):
        self._channels[channel.name] = channel

    def register_session(self, session_uuid, session):
        self._sessions[session_uuid] = session

    def get_session(self, session_uuid):
        return self._sessions.get(session_uuid)


@pytest.fixture
def channel():
    """Create a channel without starting the web server."""
    agent = MockAgent()
    config = {
        "nextcloud_url": "http://test",
        "bot_name": "test",
        "bot_secret": "secret",
        "default_role": "test",
    }
    ch = NextcloudTalkChannel(name="test", agent=agent, config=config)
    session_uuid = uuid.uuid4()
    ch._session_conversations[session_uuid] = "fake_token"
    ch._rooms["fake_token"] = session_uuid
    # Register a mock session so get_session returns a valid mock
    mock_session = MagicMock()
    mock_session.is_tool_call_pending.return_value = False
    agent._sessions[session_uuid] = mock_session
    return ch


def _make_tool_use_part(tool_id="call_abc123"):
    """Create a tool_use content part like Anthropic returns."""
    return ContentPart(
        part_type="tool_use",
        id=tool_id,
        name="eval_python",
        arguments='{"python_string": "import random; _result = 42"}',
    )


@pytest.mark.asyncio
async def test_send_tool_use_extracts_tool_call_ids(channel):
    """Verify that a tool_use part's id is extracted into _sent_parts."""
    part = _make_tool_use_part("call_abc123")
    message = Message(role="assistant", content=[part])
    session_uuid = list(channel._session_conversations.keys())[0]

    await channel.send(message, session_uuid=session_uuid)

    # Check _sent_parts has the tool_call_id
    found = any(part.get("tool_call_id") == "call_abc123" for part in channel._sent_parts)
    assert found, (
        f"Expected tool_call_id='call_abc123' in _sent_parts, "
        f"but got: {channel._sent_parts}"
    )


@pytest.mark.asyncio
async def test_send_tool_use_adds_to_sent_parts(channel):
    """Verify that sent tool_use messages are tracked in _sent_parts."""
    part = _make_tool_use_part("call_xyz789")
    message = Message(role="assistant", content=[part])
    session_uuid = list(channel._session_conversations.keys())[0]

    await channel.send(message, session_uuid=session_uuid)

    found = False
    for sent in channel._sent_parts:
        if sent.get("tool_call_id") == "call_xyz789":
            found = True
            break
    assert found, f"No entry in _sent_parts with tool_call_id='call_xyz789'"


@pytest.mark.asyncio
async def test_tool_use_reaction_handler_can_find_pending_tool(channel):
    """Verify that the reaction matching flow works with tool_use parts."""
    tool_call_id = "call_test999"
    part = ContentPart(
        part_type="tool_use",
        id=tool_call_id,
        name="eval_python",
        arguments="{}",
    )
    message = Message(role="assistant", content=[part])
    session_uuid = list(channel._session_conversations.keys())[0]

    await channel.send(message, session_uuid=session_uuid)

    extracted_ids = [p.get("tool_call_id") for p in channel._sent_parts if p.get("tool_call_id")]
    assert tool_call_id in extracted_ids, (
        f"Reaction handler needs tool_call_id {tool_call_id}, got {extracted_ids}"
    )