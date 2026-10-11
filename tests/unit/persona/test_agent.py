"""Unit tests for the persona Agent class."""

import pytest

from peteos.config import ConfigManager
from peteos.persona.agent import Agent
from peteos.persona.role import Role
from peteos.persona.toolmanager import Tool, ToolManager


def _reset_config_state():
    """Reset ConfigManager state for clean test isolation."""
    ConfigManager._config_file = None
    ConfigManager._config_dir = None
    ConfigManager._variables.clear()
    ConfigManager._on_reloaded_hooks.clear()
    ConfigManager._file_observer = None
    ConfigManager._file_observer_thread = None


@pytest.fixture(autouse=True)
def config_for_agent_tests(tmp_path):
    """Set up minimum ConfigManager state for session-dependent tests."""
    _reset_config_state()
    ConfigManager._variables["CONFIG_DIR"] = str(tmp_path / "peteos")
    ConfigManager.set_variable("SESSIONS_DIR", "{CONFIG_DIR}/sessions")
    yield
    _reset_config_state()


@pytest.fixture
def agent():
    """Create a fresh Agent for each test."""
    role = Role(name="test", description="Test role", model=".*")
    tool_manager = ToolManager()
    return Agent(role, tool_manager)


@pytest.fixture
def agent_with_assistant_role():
    role = Role(name="assistant", description="Assistant role", model=".*")
    tool_manager = ToolManager()
    return Agent(role, tool_manager)


@pytest.fixture
def agent_with_auto_approve():
    role = Role(
        name="autobot",
        description="Auto-approves some tools",
        model=".*",
        auto_approve_tools=["web_fetch"]
    )
    tool_manager = ToolManager()
    return Agent(role, tool_manager)


class TestAgentInit:
    """Test Agent.__init__."""

    def test_agent_creation(self, agent):
        assert agent._role.name == "test"

    def test_role_property(self, agent):
        assert agent.role.name == "test"
        assert agent.role.description == "Test role"


class TestAgentChannelRegistry:
    """Test Agent channel management — removed in refactor, tests updated to match."""

    def test_channels_removed_from_agent(self, agent):
        """Channels are no longer managed on the Agent (moved to Session level)."""
        assert not hasattr(agent, "_channels")

    def test_channel_methods_removed_from_agent(self, agent):
        """Channel registry methods removed from Agent."""
        assert not hasattr(agent, "register_channel")
        assert not hasattr(agent, "deregister_channel")
        assert not hasattr(agent, "get_channel")
        assert not hasattr(agent, "list_channels")


class TestAgentHooks:
    """Test Agent hook registration and behavior."""

    @pytest.mark.asyncio
    async def test_session_has_hooks_registered(self, agent):
        """create_session registers hooks on the session's tool_definitions_message."""
        session = await agent.create_session()
        assert len(session._message_hooks) > 0

    @pytest.mark.asyncio
    async def test_on_before_notification_publish_not_implemented(self, agent):
        """The persona Agent does not yet have _on_before_notification_publish.

        The code has a TODO in create_session about registering these hooks
        on the environment. This test documents the current state.
        """
        assert not hasattr(agent, "_on_before_notification_publish")


class TestAgentToolHooks:
    """Test Agent tool list and tool filter hooks."""

    def test_tool_list_hook_returns_json(self):
        """_tool_list_hook returns a JSON list of tool definitions."""
        tool_manager = ToolManager()
        tool_manager.register_tool(
            Tool(name="calc", description="calculator", func=lambda: None)
        )
        role = Role(name="test", description="Test role", model=".*")
        agent = Agent(role, tool_manager)
        result = agent._tool_list_hook()
        parsed = pytest.importorskip("json").loads(result)
        assert len(parsed) == 1
        assert parsed[0]["name"] == "calc"
        assert "description" in parsed[0]
        assert "parameters" in parsed[0]

    def test_tool_list_hook_empty_when_no_tools(self, agent):
        """_tool_list_hook returns an empty list when no tools are registered."""
        result = agent._tool_list_hook()
        parsed = pytest.importorskip("json").loads(result)
        assert parsed == []

    def test_tool_filter_hook_returns_empty_list(self, agent):
        """_tool_filter_hook returns an empty list when no role filter is set."""
        result = agent._tool_filter_hook()
        parsed = pytest.importorskip("json").loads(result)
        assert parsed == []

    @pytest.mark.asyncio
    async def test_tool_filter_hook_returns_role_filter(self):
        """_tool_filter_hook returns the role's tool filter."""
        role = Role(name="filtered", description="F", tool_filter=["calc.*", "read.*"])
        tool_manager = ToolManager()
        agent = Agent(role, tool_manager)
        result = agent._tool_filter_hook()
        parsed = pytest.importorskip("json").loads(result)
        assert parsed == ["calc.*", "read.*"]
