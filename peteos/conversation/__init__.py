"""Conversation — the data model for sessions, contexts, and messages.

A session contains one or more contexts, each representing a single step
or turn in the interaction.  A context is a list of serialised messages
(``ContentPart`` objects) with support for forked branches, content-addressed
storage, and hook-based materialisation of dynamic content.

Classes: ``ContentPart``, ``Message``, ``MessageRegistry``, ``Context``,
``Session``, ``SystemPromptMessage``, ``ToolDefinitionsMessage``.
"""

from peteos.conversation.message import ContentPart, Message
from peteos.conversation.message_registry import MessageRegistry
from peteos.conversation.context import Context
from peteos.conversation.session import Session
from peteos.conversation.system_prompt_message import SystemPromptMessage
from peteos.conversation.tool_definitions_message import ToolDefinitionsMessage

__all__ = [
    "ContentPart",
    "Message",
    "MessageRegistry",
    "Context",
    "Session",
    "SystemPromptMessage",
    "ToolDefinitionsMessage",
]