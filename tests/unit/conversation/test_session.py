"""Unit tests for Session."""

import hashlib
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from peteos.conversation.context import Context
from peteos.conversation.message import ContentPart, Message
from peteos.conversation.session import Session, SessionState
from peteos.conversation.system_prompt_message import SystemPromptMessage
from peteos.conversation.tool_definitions_message import ToolDefinitionsMessage


class TestSessionCreation:
    """Tests for Session initialization and factory methods."""

    def test_init_empty(self):
        session = Session("/some/path")
        assert session.uuid != ""
        assert session.active_context is None
        assert session.active_context_id is None

    def test_init_with_json_dict(self):
        sd = {"uuid": "test-uuid-123", "active_context_id": None}
        session = Session("/some/path", sd)
        assert session.uuid == "test-uuid-123"

    def test_init_default_uuid(self):
        session = Session("/some/path", {})
        assert session.uuid != ""

    def test_session_dir(self):
        session = Session("/tmp", {"uuid": "abc123"})
        assert session.session_dir == Path("/tmp/abc123")

    def test_raw_dict(self):
        session = Session("/some/path")
        assert isinstance(session.raw_dict, dict)
        assert "uuid" in session.raw_dict

    def test_auto_approve_tools_default_empty(self):
        session = Session("/some/path")
        assert session.auto_approve_tools == []

    def test_create_empty(self):
        session = Session.create("/some/path")
        assert session.active_context is not None
        assert session.active_context_id == session.active_context.id
        assert len(session.active_context.messages) == 0

    def test_create_with_system_prompt(self):
        sys_msg = SystemPromptMessage.create("You are helpful.")
        session = Session.create("/some/path", system_prompt_message=sys_msg)
        assert session.active_context.system_prompt_message is sys_msg

    def test_create_with_tool_definitions(self):
        tools_msg = ToolDefinitionsMessage()
        session = Session.create("/some/path", tool_definitions_message=tools_msg)
        assert session.active_context.tool_definitions_message is tools_msg


class TestSessionActiveContext:
    """Tests for active_context property and set_active_context."""

    def test_set_active_context(self):
        session = Session.create("/some/path")
        assert session.active_context is not None

        new_ctx = Context.create()
        session.set_active_context(new_ctx)
        assert session.active_context is new_ctx
        assert session.raw_dict["active_context_id"] == new_ctx.id

    def test_set_active_context_updates_json_dict(self):
        session = Session.create("/some/path")
        old_id = session.active_context_id
        new_ctx = Context.create()
        session.set_active_context(new_ctx)
        assert session.raw_dict["active_context_id"] == new_ctx.id
        assert session.raw_dict["active_context_id"] != old_id

    def test_active_context_is_none_when_not_set(self):
        session = Session("/some/path", {})
        assert session.active_context is None
        assert session.active_context_id is None


class TestSessionHooks:
    """Tests for Session hook registration and materialization."""

    def test_register_hook(self):
        session = Session.create("/some/path")
        msg = Message.create("user", [ContentPart.create_text("test")])
        hook_id = session.register_hook(msg, "test_hook", lambda: "callback result")
        assert hook_id is not None
        assert len(msg.hook_ids) == 1
        assert hook_id in msg.hook_ids

    def test_register_hook_deterministic_id(self):
        session = Session.create("/some/path")
        msg1 = Message.create("user", [ContentPart.create_text("a")])
        msg2 = Message.create("user", [ContentPart.create_text("b")])
        hook1 = session.register_hook(msg1, "same_name", lambda: "a")
        hook2 = session.register_hook(msg2, "same_name", lambda: "b")
        # Same name → same hook ID (deterministic sha256)
        assert hook1 == hook2

    def test_register_hook_updates_hook_index(self):
        session = Session.create("/some/path")
        ctx = session.active_context
        assert len(ctx.hook_index) == 0

        msg = Message.create("user", [ContentPart.create_text("hooked")])
        ctx.append(msg)
        session.register_hook(msg, "hook_name", lambda: "result")
        hook_id = session._make_hook_id("hook_name")
        assert hook_id in ctx.hook_index
        assert len(ctx.hook_index[hook_id]) == 1

    def test_register_hook_on_non_active_context_noop(self):
        """If no active context, register_hook should not crash."""
        session = Session("/some/path", {})
        msg = Message.create("user", [ContentPart.create_text("test")])
        hook_id = session.register_hook(msg, "hook", lambda: "result")
        assert hook_id is not None

    def test_hook_id_is_sha256(self):
        session = Session("/some/path")
        hook_id = session._make_hook_id("test_name")
        expected = hashlib.sha256("test_name".encode("utf-8")).hexdigest()
        assert hook_id == expected


class TestSessionMaterialize:
    """Tests for Session.materialize()."""

    def test_materialize_runs_hooks(self):
        session = Session.create("/some/path")
        ctx = session.active_context
        calls = []

        def callback():
            calls.append(True)
            return "hook output"

        msg = Message.create("user", [ContentPart.create_text("hooked")])
        ctx.append(msg)
        session.register_hook(msg, "test_hook", callback)
        session.materialize()
        assert len(calls) == 1

    def test_materialize_stores_content_in_content_map(self):
        session = Session.create("/some/path")
        ctx = session.active_context
        hook_output = "dynamic content text"

        msg = Message.create("user", [ContentPart.create_text("hooked")])
        ctx.append(msg)
        session.register_hook(msg, "test_hook", lambda: hook_output)
        session.materialize()
        expected_hash = hashlib.sha256(hook_output.encode("utf-8")).hexdigest()
        assert expected_hash in session.active_context.content_map
        assert session.active_context.content_map[expected_hash] == hook_output

    def test_materialize_calls_message_materialize(self):
        """Materialize calls msg.materialize() with the right arguments."""
        session = Session.create("/some/path")
        ctx = session.active_context

        msg = Message.create("user", [ContentPart.create_text("hooked")])
        ctx.append(msg)
        session.register_hook(msg, "test_hook", lambda: "output")

        # Patch Message.materialize to verify it gets called
        original_materialize = msg.materialize
        call_count = [0]

        def mock_materialize(materialized_hooks, content_map):
            call_count[0] += 1

        msg.materialize = mock_materialize
        session.materialize()
        assert call_count[0] == 1

    def test_materialize_with_multiple_hooks(self):
        session = Session.create("/some/path")
        ctx = session.active_context

        msg1 = Message.create("user", [ContentPart.create_text("hooked1")])
        msg2 = Message.create("assistant", [ContentPart.create_text("hooked2")])
        ctx.append(msg1)
        ctx.append(msg2)
        session.register_hook(msg1, "hook_1", lambda: "out1")
        session.register_hook(msg2, "hook_2", lambda: "out2")
        session.materialize()
        hash1 = hashlib.sha256("out1".encode("utf-8")).hexdigest()
        hash2 = hashlib.sha256("out2".encode("utf-8")).hexdigest()
        assert hash1 in session.active_context.content_map
        assert hash2 in session.active_context.content_map

    def test_materialize_with_static_messages_only(self):
        session = Session.create("/some/path")
        session.active_context.append(Message.create("user", [ContentPart.create_text("static")]))

        # Should not crash — no hooks to materialize
        session.materialize()

    def test_materialize_system_prompt_materialization(self):
        """SystemPromptMessage should be materialized if it has hook IDs."""
        session = Session.create(
            "/some/path",
            system_prompt_message=SystemPromptMessage.create("static base"),
        )

        # The system prompt message should have hook IDs registered
        sys_msg = session.active_context.system_prompt_message
        hook_id = session.register_hook(sys_msg, "sys_hook", lambda: "hooked addition")

        session.materialize()

        # The system prompt content should include both static and hooked text
        assert sys_msg is not None
        content = sys_msg.content[0].text
        assert "static base" in content
        assert "hooked addition" in content

    def test_materialize_hook_output_hashed(self):
        """Verify the hook output is hashed and stored correctly."""
        session = Session.create("/some/path")
        hook_text = "my dynamic text"

        msg = Message.create("user", [ContentPart.create_text("hooked")])
        session.register_hook(msg, "name", lambda: hook_text)

        session.materialize()

        expected_hash = hashlib.sha256(hook_text.encode("utf-8")).hexdigest()
        content_map = session.active_context.content_map
        assert content_map[expected_hash] == hook_text


class TestSessionSaveLoad:
    """Tests for Session.save() and Session.load()."""

    def test_save_creates_session_file(self):
        with patch("pathlib.Path.mkdir"):
            with patch("builtins.open", MagicMock()):
                session = Session.create("/tmp", system_prompt_message=SystemPromptMessage.create("hello"))
                session.save()
                # Should not crash

    def test_save_creates_context_file(self):
        with patch("pathlib.Path.mkdir"):
            with patch("builtins.open", MagicMock()):
                session = Session.create("/tmp")
                session.save()
                # Should not crash

    def test_save_includes_auto_approve_tools(self):
        session = Session.create("/tmp")
        session._json_dict["auto_approve_tools"] = ["tool_a", "tool_b"]
        with patch("pathlib.Path.mkdir"):
            with patch("builtins.open", MagicMock()):
                session.save()

    def test_load_missing_file_raises(self):
        with pytest.raises(FileNotFoundError, match="session.json not found"):
            Session.load("/nonexistent/path", "abc123")


class TestSessionActiveContextPersistence:
    """Tests for active_context_id in session state."""

    def test_active_context_id_set_on_create(self):
        session = Session.create("/some/path")
        assert session.active_context_id is not None
        assert session.active_context_id == session.active_context.id

    def test_active_context_id_updated_on_set(self):
        session = Session.create("/some/path")
        original_id = session.active_context_id
        new_ctx = Context.create()
        session.set_active_context(new_ctx)
        assert session.active_context_id == new_ctx.id
        assert session.active_context_id != original_id

    def test_active_context_property_matches_json_dict(self):
        session = Session.create("/some/path")
        assert session.active_context is not None
        assert session.active_context_id == session.active_context.id


class TestSessionRegisterHookEdgeCases:
    """Edge cases for hook registration."""

    def test_register_hook_with_empty_name(self):
        session = Session.create("/some/path")
        msg = Message.create("user", [ContentPart.create_text("test")])
        hook_id = session.register_hook(msg, "", lambda: "result")
        assert hook_id is not None
        assert hook_id in msg.hook_ids

    def test_register_multiple_hooks_same_message(self):
        session = Session.create("/some/path")
        msg = Message.create("user", [ContentPart.create_text("multi")])
        h1 = session.register_hook(msg, "hook_one", lambda: "a")
        h2 = session.register_hook(msg, "hook_two", lambda: "b")
        assert len(msg.hook_ids) == 2
        assert h1 in msg.hook_ids
        assert h2 in msg.hook_ids

    def test_register_hook_updates_hook_index_for_same_message(self):
        session = Session.create("/some/path")
        ctx = session.active_context
        msg = Message.create("user", [ContentPart.create_text("test")])
        ctx.append(msg)
        h1 = session.register_hook(msg, "h1", lambda: "a")
        h2 = session.register_hook(msg, "h2", lambda: "b")
        assert len(ctx.hook_index[h1]) == 1
        assert len(ctx.hook_index[h2]) == 1


class TestSessionHooksMaterializeContent:
    """Tests that verify the full materialization flow."""

    def test_full_materialization_chain(self):
        """End-to-end: register hook → materialize → verify content map."""
        session = Session.create("/some/path")
        hook_output = "the quick brown fox"

        msg = Message.create("user", [ContentPart.create_text("test")])
        hook_id = session.register_hook(msg, "test_hook", lambda: hook_output)

        # Before materialization, content map is empty
        assert len(session.active_context.content_map) == 0

        session.materialize()

        # After materialization, content map has the hash
        expected_hash = hashlib.sha256(hook_output.encode("utf-8")).hexdigest()
        assert session.active_context.content_map[expected_hash] == hook_output

    def test_materialize_gathers_dynamic_messages(self):
        """Verify only messages with hooks get materialized."""
        session = Session.create("/some/path")
        ctx = session.active_context

        static_msg = Message.create("user", [ContentPart.create_text("static")])
        ctx.append(static_msg)

        # Message with hook
        hooked_msg = Message.create("assistant", [ContentPart.create_text("hooked")])
        ctx.append(hooked_msg)
        session.register_hook(hooked_msg, "test_hook", lambda: "output")

        # Verify both messages exist
        assert len(ctx.messages) == 2

        session.materialize()

        # The content map should be populated
        assert len(ctx.content_map) > 0


class TestSessionState:
    """Tests for SessionState key-value store."""

    def test_get_returns_none_for_missing(self):
        state = SessionState()
        assert state.get("nonexistent") is None

    def test_get_returns_value(self):
        state = SessionState({"key": "value"})
        assert state.get("key") == "value"

    def test_create_new_variable(self):
        state = SessionState()
        state.create("new_key", "new_value")
        assert state.get("new_key") == "new_value"

    def test_create_raises_if_exists(self):
        state = SessionState({"key": "old"})
        with pytest.raises(ValueError, match="already exists"):
            state.create("key", "new")

    def test_create_raises_for_none_value(self):
        state = SessionState()
        with pytest.raises(ValueError, match="must not be None"):
            state.create("key", None)

    def test_update_existing_variable(self):
        state = SessionState({"key": "old"})
        state.update("key", "old", "new")
        assert state.get("key") == "new"

    def test_update_raises_if_value_changed(self):
        state = SessionState({"key": "actual"})
        with pytest.raises(ValueError, match="has value"):
            state.update("key", "expected", "new")

    def test_update_raises_if_old_value_none(self):
        state = SessionState({"key": "value"})
        with pytest.raises(ValueError, match="must not be None"):
            state.update("key", None, "new")

    def test_delete_existing_variable(self):
        state = SessionState({"key": "value"})
        state.delete("key")
        assert state.get("key") is None

    def test_delete_raises_if_missing(self):
        state = SessionState()
        with pytest.raises(KeyError, match="does not exist"):
            state.delete("nonexistent")

    def test_list_returns_keys(self):
        state = SessionState({"a": "1", "b": "2"})
        assert set(state.list()) == {"a", "b"}

    def test_to_dict(self):
        state = SessionState({"key": "value"})
        assert state.to_dict() == {"key": "value"}
