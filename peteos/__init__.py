# Peteos - Agentic application framework

import asyncio

from peteos.config import ConfigManager
from peteos.utils.activeclass import ActiveClass
from peteos.chatbot import (
    ChatBot,
    OpenAIChatBot,
    AnthropicChatBot,
    ChatBotManager,
    BackendInfo,
    ChatBotResponse,
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
    "ConfigManager",
    "HTTPClient",
    "Message",
    "OpenAIChatBot",
]

# Bootstrap backends and roles from peteos.json at import time.
# asyncio.run() cannot be called inside a running event loop, so we detect
# whether one exists first; if not, create one and run ConfigManager.init().
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.run(ConfigManager.init())


try:
    from peteos.persona.agent import Agent
    from peteos.engine.executionenvironment import ExecutionEnvironment
    from peteos.persona.role import Role
    from peteos.persona.rolemanager import RoleManager
    from peteos.conversation.session import Session

    __all__.extend([
        "Agent",
        "ExecutionEnvironment",
        "Role",
        "RoleManager",
        "Session",
    ])
except ImportError:
    pass
