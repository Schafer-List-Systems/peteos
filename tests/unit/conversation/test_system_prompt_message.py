"""Unit tests for SystemPromptMessage."""

from peteos.conversation.message import ContentPart
from peteos.conversation.system_prompt_message import SystemPromptMessage


class TestSystemPromptMessage:
    """Tests for SystemPromptMessage."""

    def test_create_without_static_text(self):
        msg = SystemPromptMessage.create()
        assert msg.role == ""
        assert msg.raw_dict.get("_static_text") is None
        assert msg.content == []

    def test_create_with_static_text(self):
        msg = SystemPromptMessage.create("You are a helpful assistant.")
        assert len(msg.content) == 1
        assert msg.content[0].type == "text"
        assert msg.content[0].text == "You are a helpful assistant."

    def test_create_sets_static_text_in_raw_dict(self):
        msg = SystemPromptMessage.create("static content")
        assert msg.raw_dict["_static_text"] == "static content"

    def test_materialize_no_hooks(self):
        msg = SystemPromptMessage.create("base prompt")
        # Materialize with None should be a no-op
        msg.materialize(materialized_hooks=None, content_map=None)
        assert len(msg.content) == 1
        assert msg.content[0].text == "base prompt"

    def test_materialize_static_only(self):
        msg = SystemPromptMessage.create("static prompt")
        msg.materialize(
            materialized_hooks={"hook_1": "hash_1"},
            content_map={"hash_1": "hooked text"},
        )
        assert msg.content[0].text == "static prompt"

    def test_materialize_static_and_hooked(self):
        msg = SystemPromptMessage.create("static")
        msg.raw_dict["_hook_ids"].append("hook_1")
        msg.materialize(
            materialized_hooks={"hook_1": "hash_1"},
            content_map={"hash_1": "hooked"},
        )
        assert msg.content[0].text == "static\n\nhooked"

    def test_materialize_missing_hash_ignores(self):
        msg = SystemPromptMessage.create("base")
        msg.raw_dict["_hook_ids"].append("missing_hook")
        # Missing hook ID → error logged but no crash, hook skipped
        msg.materialize(
            materialized_hooks={},
            content_map={},
        )
        assert msg.content[0].text == "base"

    def test_materialize_missing_content_ignores(self):
        msg = SystemPromptMessage.create("base")
        msg.raw_dict["_hook_ids"].append("hook_1")
        # Hash not in content_map → hook skipped
        msg.materialize(
            materialized_hooks={"hook_1": "missing_hash"},
            content_map={},
        )
        assert msg.content[0].text == "base"

    def test_materialize_multiple_hooks(self):
        msg = SystemPromptMessage.create("start")
        msg.raw_dict["_hook_ids"].extend(["hook_a", "hook_b"])
        msg.materialize(
            materialized_hooks={
                "hook_a": "hash_a",
                "hook_b": "hash_b",
            },
            content_map={
                "hash_a": "middle",
                "hash_b": "end",
            },
        )
        assert msg.content[0].text == "start\n\nmiddle\n\nend"

    def test_materialize_no_static_multiple_hooks(self):
        msg = SystemPromptMessage.create()
        msg.raw_dict["_hook_ids"].extend(["hook_a", "hook_b"])
        msg.materialize(
            materialized_hooks={
                "hook_a": "hash_a",
                "hook_b": "hash_b",
            },
            content_map={
                "hash_a": "part A",
                "hash_b": "part B",
            },
        )
        assert msg.content[0].text == "part A\n\npart B"

    def test_materialize_no_hook_ids_no_change(self):
        msg = SystemPromptMessage.create("unchanged")
        msg.materialize(
            materialized_hooks={},
            content_map={},
        )
        assert msg.content[0].text == "unchanged"
