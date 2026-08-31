"""Tests for system prompt hooks in AgenticObject."""

from __future__ import annotations

import pytest

from peteos.conversation.system_prompt_message import SystemPromptMessage
from peteos.conversation.session import Session
from peteos.oap import AgenticObject, agentic_object


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@agentic_object()
class PlainObj(AgenticObject):
    """Object without code execution."""


@agentic_object(allow_code_execution=True)
class CodeExecObj(AgenticObject):
    """Object with code execution enabled."""


@agentic_object(allow_code_execution=True, imports=["numpy"])
class CodeExecWithImportsObj(AgenticObject):
    """Object with code execution and imports."""


# ---------------------------------------------------------------------------
# Test hooks are stored in the dict
# ---------------------------------------------------------------------------


class TestHookStorage:
    """Verify hooks are stored in _oap_system_prompt_hooks."""

    def test_no_code_exec_no_python_exec_hook(self):
        """Without allow_code_execution, no python_exec hook is stored."""
        obj = PlainObj()
        assert "python_exec" not in obj._oap_system_prompt_hooks

    def test_code_exec_has_python_exec_hook(self):
        """With allow_code_execution, python_exec hook is stored."""
        obj = CodeExecObj()
        assert "python_exec" in obj._oap_system_prompt_hooks

    def test_code_exec_with_imports_has_python_exec_hook(self):
        """Imports don't change hook registration."""
        obj = CodeExecWithImportsObj()
        assert "python_exec" in obj._oap_system_prompt_hooks

    def test_output_schema_hook_always_registered(self):
        """Output schema hook is always registered."""
        obj = PlainObj()
        assert "output_schema" in obj._oap_system_prompt_hooks


# ---------------------------------------------------------------------------
# Test hook callbacks return correct content
# ---------------------------------------------------------------------------


class TestHookContent:
    """Verify hook callbacks return the expected text."""

    def test_python_exec_hook_content(self):
        """python_exec hook returns code execution instructions."""
        obj = CodeExecObj()
        hook = obj._oap_system_prompt_hooks["python_exec"]
        text = hook()
        assert "# Code Execution" in text
        assert "python_exec" in text
        assert "Forbidden" in text

    def test_python_exec_hook_includes_imports(self):
        """python_exec hook includes available modules when imports are set."""
        obj = CodeExecWithImportsObj()
        hook = obj._oap_system_prompt_hooks["python_exec"]
        text = hook()
        assert "numpy" in text

    def test_python_exec_hook_no_imports(self):
        """python_exec hook has no modules list when imports are empty."""
        obj = CodeExecObj()
        hook = obj._oap_system_prompt_hooks["python_exec"]
        text = hook()
        assert "Available modules:" not in text

    def test_output_schema_hook_content(self):
        """output_schema hook returns default schema when no output_schema set."""
        obj = PlainObj()
        hook = obj._oap_system_prompt_hooks["output_schema"]
        text = hook()
        assert "# any" in text


# ---------------------------------------------------------------------------
# Test hooks on system prompt message
# ---------------------------------------------------------------------------


class TestHooksOnSystemPromptMessage:
    """Verify hooks are registered on the SystemPromptMessage."""

    def _make_session_with_hooks(self, obj: AgenticObject) -> Session:
        """Create a session and register the object's system prompt hooks."""
        spm = SystemPromptMessage.create("static prompt")
        session = Session.create(obj._oap_agent.agent_dir, spm, None)
        session.session_dir.mkdir(exist_ok=True)
        for name, callback in obj._oap_system_prompt_hooks.items():
            session.register_hook(spm, name, callback)
        return session

    def test_system_prompt_message_has_hook_ids(self):
        """After registration, system prompt message has hook_ids."""
        obj = CodeExecObj()
        session = self._make_session_with_hooks(obj)
        spm = session.active_context.system_prompt_message
        assert len(spm.hook_ids) == 2

    def test_materialize_resolves_hook_content(self):
        """materialize() resolves hook outputs into content."""
        obj = CodeExecObj()
        session = self._make_session_with_hooks(obj)

        session.materialize()
        spm = session.active_context.system_prompt_message
        content = spm.content
        assert len(content) > 0
        text_parts = [p.text for p in content if p.type == "text"]
        combined = "\n".join(text_parts)
        assert "static prompt" in combined
        assert "# Code Execution" in combined

    def test_materialize_includes_imports_in_content(self):
        """materialize() resolves import info into system prompt."""
        obj = CodeExecWithImportsObj()
        session = self._make_session_with_hooks(obj)

        session.materialize()
        spm = session.active_context.system_prompt_message
        content = spm.content
        text_parts = [p.text for p in content if p.type == "text"]
        combined = "\n".join(text_parts)
        assert "numpy" in combined

    def test_no_hooks_on_plain_object(self):
        """Plain object only has output_schema hook."""
        obj = PlainObj()
        session = self._make_session_with_hooks(obj)
        assert len(session._message_hooks) == 1
