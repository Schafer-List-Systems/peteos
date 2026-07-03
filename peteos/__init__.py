# Peteos - Agentic application framework

import asyncio
import importlib
import pkgutil

from peteos.chatbot import ChatBotManager

try:
    loop = asyncio.get_running_loop()
except RuntimeError:
    asyncio.run(ChatBotManager.load_from_config())

from peteos.utils.activeclass import ActiveClass
from peteos.chatbot import (
    ChatBot,
    OpenAIChatBot,
    AnthropicChatBot,
    ChatBotManager,
    BackendInfo,
    ChatBotResponse,
    GenericChatBotResponse,
    AnthropicChatBotResponse,
    Message,
    HTTPClient,
)

__all__ = [
    "ActiveClass",
    "AnthropicChatBot",
    "AnthropicChatBotResponse",
    "BackendInfo",
    "ChatBot",
    "ChatBotManager",
    "ChatBotResponse",
    "HTTPClient",
    "Message",
    "OpenAIChatBot",
]

try:
    from peteos.agent import Agent
    from peteos.executionenvironment import ExecutionEnvironment
    from peteos.replexecutionenvironment import REPLExecutionEnvironment
    from peteos.role import Role
    from peteos.rolemanager import RoleManager
    from peteos.session import Session, invoke_agent, _extract_last_assistant_text

    __all__.extend([
        "Agent",
        "ExecutionEnvironment",
        "invoke_agent",
        "REPLExecutionEnvironment",
        "Role",
        "RoleManager",
        "Session",
        "_extract_last_assistant_text",
    ])
except ImportError:
    pass
