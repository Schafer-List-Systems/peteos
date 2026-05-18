"""Test that tool_use content parts are properly extracted for reaction matching.

Anthropic's API returns tool call content parts with type='tool_use', not
'tool_call' or 'tool_calls'. The send() method must extract the tool_call_id
from these parts so the reaction handler can find pending tool calls.
"""

import asyncio
import uuid
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
    """Verify that a tool_use part's id is extracted into _tool_call_ids."""
    part = _make_tool_use_part("call_abc123")
    message = Message(role="assistant", content=[part])
    session_uuid = list(channel._session_conversations.keys())[0]

    await channel.send(message, session_uuid=session_uuid)

    reference_id = message.get_id()
    assert reference_id in channel._tool_call_ids, (
        f"Expected tool_call_ids to be populated for ref={reference_id}, "
        f"but _tool_call_ids keys are: {list(channel._tool_call_ids.keys())}"
    )
    assert channel._tool_call_ids[reference_id] == ["call_abc123"], (
        f"Expected ['call_abc123'], got {channel._tool_call_ids[reference_id]}"
    )


@pytest.mark.asyncio
async def test_send_tool_use_adds_to_sent_messages(channel):
    """Verify that sent tool_use messages have tool_call_ids in _sent_messages."""
    part = _make_tool_use_part("call_xyz789")
    message = Message(role="assistant", content=[part])
    session_uuid = list(channel._session_conversations.keys())[0]

    await channel.send(message, session_uuid=session_uuid)

    reference_id = message.get_id()
    found = False
    for sent in channel._sent_messages:
        if sent.get("referenceId") == reference_id:
            found = True
            assert sent.get("tool_call_ids") == ["call_xyz789"], (
                f"Expected tool_call_ids=['call_xyz789'], got {sent.get('tool_call_ids')}"
            )
            break
    assert found, f"No entry in _sent_messages for ref={reference_id}"


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

    reference_id = message.get_id()
    extracted_ids = channel._tool_call_ids.get(reference_id, [])
    assert extracted_ids == [tool_call_id], (
        f"Reaction handler needs tool_call_id {tool_call_id}, got {extracted_ids}"
    )