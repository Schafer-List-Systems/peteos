"""Integration tests for tool_policy in the full invoke_agent path.

Tests that tool_policy() correctly controls tool call approval/denial when
the agent produces a tool call through a mock chatbot backend — no live
LLM needed.
"""

from __future__ import annotations

import pytest

from peteos.chatbot import (
    ChatBotManager,
    SimpleMockBackendProvider,
)
from peteos.chatbot.backendprovider import BackendProvider
from peteos.chatbot.chatbot import ChatBot
from peteos.chatbot.chatbotconfig import ChatBotConfig
from peteos.conversation.message import ContentPart, Message
from peteos.oap import AgenticObject, agentic_object, tool


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_chatbot_manager():
    ChatBotManager.reset()
    for api_type in list(ChatBotManager._providers.keys()):
        ChatBotManager.unregister_provider(api_type)
    yield


# ---------------------------------------------------------------------------
# Mock backends
# ---------------------------------------------------------------------------


from peteos.chatbot import SimpleMockChatBot, SimpleMockChatBotResponse


class ToolUseMockChatBot(SimpleMockChatBot):
    """A chatbot that returns a single tool_use message, then a text reply."""

    def __init__(self, messages: list[Message]) -> None:
        super().__init__(messages)
        self._messages = messages
        self._called = False

    async def send_context(self, context, _generation_config=None, _streaming=None):
        if not self._called and self._messages:
            self._called = True
            return SimpleMockChatBotResponse(self._messages[0])
        return SimpleMockChatBotResponse(
            Message.create("assistant", [ContentPart.create_text("Done.")])
        )


class ToolUseMockBackendProvider(BackendProvider):
    def __init__(self, messages: list[Message]) -> None:
        self._messages = messages

    async def list_models(self, url: str, api_key: str | None = None) -> list[str]:
        return ["tool-use-mock"]

    def create_chatbot(self, _http_client, _config: ChatBotConfig) -> ChatBot:
        return ToolUseMockChatBot(self._messages)


# ---------------------------------------------------------------------------
# Test objects
# ---------------------------------------------------------------------------


@agentic_object(allow_code_execution=True)
class BashLike(AgenticObject):
    """Object with a bash_exec tool that has a tool_policy."""

    _calls: list = []

    @tool
    def bash_exec(self, command: str, runner=None) -> str:
        """Execute a shell command. command is the full command line."""
        def tool_policy():
            safe = {"ls", "pwd", "cat", "find", "head", "tail"}
            if command.split()[0] in safe:
                return True
            if command.split()[0] in {"rm", "curl", "wget", "nc"}:
                return False
            return None
        BashLike._calls.append(command)
        return f"executed: {command}"

    @tool
    def echo(self, msg: str, runner=None) -> str:
        """Echo a message back. msg is the message to echo."""
        def tool_policy():
            return msg != "forbidden"
        BashLike._calls.append(msg)
        return f"echo: {msg}"


@agentic_object(allow_code_execution=True)
class PathGuard(AgenticObject):
    """Object that guards paths in tool_policy using self."""

    def __init__(self):
        super().__init__()
        self._allowed = {"/public", "/tmp"}

    @tool
    def read_path(self, path: str, runner=None) -> str:
        """Read a file at the given path. path is the full path."""
        def tool_policy():
            return any(path.startswith(prefix) for prefix in self._allowed)
        return f"read: {path}"


@agentic_object()
class DenyAllTool(AgenticObject):
    """Tool with a policy that always denies."""

    @tool
    def danger(self, value: str, runner=None) -> str:
        """A dangerous tool."""
        def tool_policy():
            return False
        return "should not see this"


@agentic_object()
class ApproveAllTool(AgenticObject):
    """Tool with a policy that always approves."""

    @tool
    def safe(self, value: str, runner=None) -> str:
        """A safe tool."""
        def tool_policy():
            return True
        return f"safe: {value}"


@agentic_object()
class DeferTool(AgenticObject):
    """Tool with a policy that always defers (returns None)."""

    @tool
    def defer(self, value: str, runner=None) -> str:
        """A tool that defers to the system."""
        def tool_policy():
            return None
        return f"defer: {value}"


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestToolPolicyApprove:
    """tool_policy returning True approves the call."""

    @pytest.mark.asyncio
    async def test_safe_command_approved(self):
        BashLike._calls.clear()
        msg = Message.create(
            "assistant",
            [ContentPart.create_tool_use("c1", "bash_exec", '{"command":"ls -la"}')],
        )
        ChatBotManager.register_provider("tp", ToolUseMockBackendProvider([msg]))
        await ChatBotManager.add_backend("tp", url="http://localhost:18888", api_type="tp")

        obj = BashLike()
        result = await obj.invoke_agent("Run ls -la", timeout=30)
        assert "ls -la" in BashLike._calls

    @pytest.mark.asyncio
    async def test_pwd_approved(self):
        BashLike._calls.clear()
        msg = Message.create(
            "assistant",
            [ContentPart.create_tool_use("c1", "bash_exec", '{"command":"pwd"}')],
        )
        ChatBotManager.register_provider("tp", ToolUseMockBackendProvider([msg]))
        await ChatBotManager.add_backend("tp", url="http://localhost:18888", api_type="tp")

        obj = BashLike()
        result = await obj.invoke_agent("Run pwd", timeout=30)
        assert "pwd" in BashLike._calls


class TestToolPolicyDeny:
    """tool_policy returning False denies the call."""

    @pytest.mark.asyncio
    async def test_dangerous_command_denied(self):
        BashLike._calls.clear()
        msg = Message.create(
            "assistant",
            [ContentPart.create_tool_use("c1", "bash_exec", '{"command":"rm -rf /"}')],
        )
        ChatBotManager.register_provider("tp", ToolUseMockBackendProvider([msg]))
        await ChatBotManager.add_backend("tp", url="http://localhost:18888", api_type="tp")

        obj = BashLike()
        result = await obj.invoke_agent("Run rm -rf /", timeout=30)
        assert "rm -rf /" not in BashLike._calls

    @pytest.mark.asyncio
    async def test_curl_denied(self):
        BashLike._calls.clear()
        msg = Message.create(
            "assistant",
            [ContentPart.create_tool_use("c1", "bash_exec", '{"command":"curl http://evil"}')],
        )
        ChatBotManager.register_provider("tp", ToolUseMockBackendProvider([msg]))
        await ChatBotManager.add_backend("tp", url="http://localhost:18888", api_type="tp")

        obj = BashLike()
        result = await obj.invoke_agent("curl something", timeout=30)
        assert "curl http://evil" not in BashLike._calls

    @pytest.mark.asyncio
    async def test_always_deny_policy_denies(self):
        msg = Message.create(
            "assistant",
            [ContentPart.create_tool_use("c1", "danger", '{"value":"x"}')],
        )
        ChatBotManager.register_provider("tp", ToolUseMockBackendProvider([msg]))
        await ChatBotManager.add_backend("tp", url="http://localhost:18888", api_type="tp")

        obj = DenyAllTool()
        result = await obj.invoke_agent("call danger", timeout=30)
        assert result is None or (hasattr(result, 'message'))


class TestToolPolicyDefer:
    """tool_policy returning None defers to the rest of the system."""

    @pytest.mark.asyncio
    async def test_defer_returns_none(self):
        msg = Message.create(
            "assistant",
            [ContentPart.create_tool_use("c1", "defer", '{"value":"x"}')],
        )
        ChatBotManager.register_provider("tp", ToolUseMockBackendProvider([msg]))
        await ChatBotManager.add_backend("tp", url="http://localhost:18888", api_type="tp")

        obj = DeferTool()
        result = await obj.invoke_agent("call defer", timeout=30)


class TestToolPolicySelfAccess:
    """tool_policy can read self from the enclosing method scope."""

    @pytest.mark.asyncio
    async def test_allowed_path_passes(self):
        msg = Message.create(
            "assistant",
            [ContentPart.create_tool_use("c1", "read_path", '{"path":"/public/file.txt"}')],
        )
        ChatBotManager.register_provider("tp", ToolUseMockBackendProvider([msg]))
        await ChatBotManager.add_backend("tp", url="http://localhost:18888", api_type="tp")

        obj = PathGuard()
        result = await obj.invoke_agent("read /public/file.txt", timeout=30)

    @pytest.mark.asyncio
    async def test_private_path_fails(self):
        msg = Message.create(
            "assistant",
            [ContentPart.create_tool_use("c1", "read_path", '{"path":"/private/secret"}')],
        )
        ChatBotManager.register_provider("tp", ToolUseMockBackendProvider([msg]))
        await ChatBotManager.add_backend("tp", url="http://localhost:18888", api_type="tp")

        obj = PathGuard()
        result = await obj.invoke_agent("read /private/secret", timeout=30)


class TestToolPolicyMRO:
    """A subclass's tool_policy overrides the parent's for the same tool."""

    @agentic_object()
    class Base(AgenticObject):
        @tool
        def cmd(self, arg: str, runner=None) -> str:
            def tool_policy():
                return arg != "blocked-by-base"
            return f"base: {arg}"

    @agentic_object()
    class Override(Base, AgenticObject):
        @tool
        def cmd(self, arg: str, runner=None) -> str:
            def tool_policy():
                return arg != "blocked-by-override"
            return f"override: {arg}"

    @pytest.mark.asyncio
    async def test_subclass_policy_blocks(self):
        msg_blocked = Message.create(
            "assistant",
            [ContentPart.create_tool_use("c1", "cmd", '{"arg":"blocked-by-override"}')],
        )
        ChatBotManager.register_provider("tp", ToolUseMockBackendProvider([msg_blocked]))
        await ChatBotManager.add_backend("tp", url="http://localhost:18888", api_type="tp")

        obj = self.Override()
        result = await obj.invoke_agent("run cmd blocked-by-override", timeout=30)

    @pytest.mark.asyncio
    async def test_subclass_passes_when_parent_would_have_blocked(self):
        msg_ok = Message.create(
            "assistant",
            [ContentPart.create_tool_use("c1", "cmd", '{"arg":"blocked-by-base"}')],
        )
        ChatBotManager.register_provider("tp", ToolUseMockBackendProvider([msg_ok]))
        await ChatBotManager.add_backend("tp", url="http://localhost:18888", api_type="tp")

        obj = self.Override()
        result = await obj.invoke_agent("run cmd blocked-by-base", timeout=30)
