"""Unit tests for Context."""

import pytest

from peteos.conversation.context import Context
from peteos.conversation.message import ContentPart, Message
from peteos.conversation.system_prompt_message import SystemPromptMessage
from peteos.conversation.tool_definitions_message import ToolDefinitionsMessage


class TestContextCreation:
    """Tests for Context creation and factory methods."""

    def test_create_empty(self):
        ctx = Context.create()
        assert ctx.message_count == 0
        assert ctx.messages == []
        assert ctx.id != ""
        assert isinstance(ctx.content_map, dict)

    def test_create_with_system_prompt(self):
        sys_msg = SystemPromptMessage.create("You are helpful.")
        ctx = Context.create(system_prompt_message=sys_msg)
        assert ctx.system_prompt_message is not None
        assert ctx.system_prompt_message.content[0].text == "You are helpful."
        assert len(ctx.messages) == 1

    def test_create_with_tool_definitions(self):
        tools_msg = ToolDefinitionsMessage()
        ctx = Context.create(tool_definitions_message=tools_msg)
        assert ctx.tool_definitions_message is not None
        assert ctx.tool_definitions_message is tools_msg
        assert len(ctx.messages) == 1

    def test_create_with_both_special_messages(self):
        sys_msg = SystemPromptMessage.create("Be helpful.")
        tools_msg = ToolDefinitionsMessage()
        ctx = Context.create(
            system_prompt_message=sys_msg,
            tool_definitions_message=tools_msg,
        )
        assert ctx.system_prompt_message is sys_msg
        assert ctx.tool_definitions_message is tools_msg

    def test_create_inherits_content_map(self):
        parent = Context.create()
        parent.content_map["hash_1"] = "content_1"
        child = Context.create(parent_context=parent)
        assert "hash_1" in child.content_map
        assert child.content_map["hash_1"] == "content_1"
        assert child._json_dict["origin_context_id"] == parent.id

    def test_create_no_content_map_inheritance_when_no_parent(self):
        ctx = Context.create()
        assert ctx._json_dict.get("origin_context_id") is None

    def test_create_independent_content_maps(self):
        parent = Context.create()
        parent.content_map["hash_1"] = "content_1"
        child = Context.create(parent_context=parent)
        child.content_map["hash_2"] = "content_2"
        assert "hash_2" not in parent.content_map

    def test_load_from_dict(self):
        ctx = Context.create()
        ctx.append(Message.create("user", [ContentPart.create_text("Hi")]))
        raw = ctx.raw_dict
        # Override id for determinism
        raw["id"] = "ctx-123"
        loaded = Context.load_from_dict(raw)
        assert loaded.id == "ctx-123"
        assert len(loaded.messages) == 1
        assert loaded.messages[0].role == "user"
        assert loaded.messages[0].content[0].text == "Hi"

    def test_load_from_dict_default_id(self):
        ctx = Context.load_from_dict({})
        assert ctx.id != ""

    def test_load_from_dict_default_messages(self):
        ctx = Context.load_from_dict({"id": "x"})
        assert ctx.messages == []

    def test_load_from_dict_preserves_hooks(self):
        ctx = Context.create()
        msg = Message.create("user", [ContentPart.create_text("Hi")])
        msg.raw_dict["_hook_ids"].append("hook_1")
        msg.raw_dict["_hook_ids"].append("hook_2")
        ctx.append(msg)
        raw = ctx.raw_dict
        raw["id"] = "ctx-2"
        loaded = Context.load_from_dict(raw)
        assert "hook_1" in loaded.hook_index
        assert "hook_2" in loaded.hook_index
        assert len(loaded.hook_index["hook_1"]) == 1

    def test_raw_dict(self):
        ctx = Context.create()
        assert isinstance(ctx.raw_dict, dict)
        assert "messages" in ctx.raw_dict

    def test_messages_are_message_instances(self):
        ctx = Context.create()
        msg = Message.create("user", [ContentPart.create_text("test")])
        ctx.append(msg)
        assert isinstance(ctx.messages[0], Message)
        assert ctx.messages[0].role == "user"

    def test_message_count(self):
        ctx = Context.create()
        assert ctx.message_count == 0
        ctx.append(Message.create("user", [ContentPart.create_text("a")]))
        assert ctx.message_count == 1


class TestContextAppend:
    """Tests for Context.append()."""

    def test_append_single_message(self):
        ctx = Context.create()
        msg = Message.create("user", [ContentPart.create_text("Hello")])
        ctx.append(msg)
        assert ctx.message_count == 1
        assert len(ctx.messages) == 1

    def test_append_multiple_messages(self):
        ctx = Context.create()
        for i in range(3):
            text = f"msg {i}"
            ctx.append(Message.create("user", [ContentPart.create_text(text)]))
        assert ctx.message_count == 3
        assert ctx.messages[0].content[0].text == "msg 0"

    def test_append_updates_raw_dict(self):
        ctx = Context.create()
        msg = Message.create("user", [ContentPart.create_text("raw")])
        ctx.append(msg)
        assert len(ctx.raw_dict["messages"]) == 1

    def test_append_increments_sequence_number(self):
        ctx = Context.create()
        ctx.append(Message.create("user", [ContentPart.create_text("a")]))
        ctx.append(Message.create("user", [ContentPart.create_text("b")]))
        assert ctx._json_dict["message_sequence"] == 2

    def test_sequence_number_in_raw_dict(self):
        ctx = Context.create()
        msg = Message.create("user", [ContentPart.create_text("seq")])
        ctx.append(msg)
        assert "_sequence_number" in msg.raw_dict
        # mutation_counter is 3 after 3 default anchor additions (system_prompt, tools, messages),
        # so the first appended message gets sequence 3
        assert msg.raw_dict["_sequence_number"] == 3

    def test_append_at_messages_anchor(self):
        ctx = Context.create()
        msg1 = Message.create("user", [ContentPart.create_text("first")])
        ctx.append(msg1, anchor_point="messages")
        assert ctx.messages[0] is msg1

    def test_append_with_named_anchor_inserts_at_anchor(self):
        ctx = Context.create()
        msg1 = Message.create("user", [ContentPart.create_text("first")])
        ctx.append(msg1, anchor_point="messages")
        ctx.add_anchor("insertion_point", 0)
        msg2 = Message.create("assistant", [ContentPart.create_text("inserted")])
        ctx.append(msg2, anchor_point="insertion_point")
        assert ctx.messages[0] is msg2
        assert ctx.messages[1] is msg1

    def test_append_indexed_anchor_shifts_anchors(self):
        ctx = Context.create()
        msg1 = Message.create("user", [ContentPart.create_text("first")])
        ctx.append(msg1, anchor_point="messages")
        msg2 = Message.create("assistant", [ContentPart.create_text("second")])
        ctx.append(msg2, anchor_point="messages")
        ctx.add_anchor("before_first", 0)
        msg3 = Message.create("user", [ContentPart.create_text("before")])
        ctx.append(msg3, anchor_point="before_first")
        assert ctx.messages[0] is msg3
        assert ctx.messages[1] is msg1
        assert ctx.messages[2] is msg2

    def test_append_with_indexed_anchor(self):
        ctx = Context.create()
        msg1 = Message.create("user", [ContentPart.create_text("first")])
        ctx.append(msg1, anchor_point="messages")
        msg2 = Message.create("assistant", [ContentPart.create_text("second")])
        ctx.append(msg2, anchor_point="messages")
        ctx.add_anchor("before_last", 1)
        msg3 = Message.create("user", [ContentPart.create_text("inserted")])
        ctx.append(msg3, anchor_point="before_last")
        assert ctx.messages[0] is msg1
        assert ctx.messages[1] is msg3
        assert ctx.messages[2] is msg2


class TestContextAnchors:
    """Tests for anchor points."""

    def test_anchor_points_has_messages_default(self):
        ctx = Context.create()
        # Three default anchors all at 0 for fresh context
        assert ("system_prompt", 0) in ctx.anchor_points
        assert ("tools", 0) in ctx.anchor_points
        assert ("messages", 0) in ctx.anchor_points

    def test_add_anchor(self):
        ctx = Context.create()
        ctx.add_anchor("start", 0)
        assert ("start", 0) in ctx.anchor_points

    def test_add_anchor_with_message_index(self):
        ctx = Context.create()
        ctx.append(Message.create("user", [ContentPart.create_text("first")]))
        ctx.append(Message.create("user", [ContentPart.create_text("second")]))
        ctx.add_anchor("after_first", 1)
        assert ("after_first", 1) in ctx.anchor_points

    def test_add_anchor_duplicate_raises(self):
        ctx = Context.create()
        ctx.add_anchor("my_anchor", 0)
        with pytest.raises(ValueError, match="Anchor point already exists"):
            ctx.add_anchor("my_anchor", 0)

    def test_anchor_positive_index(self):
        ctx = Context.create()
        ctx.append(Message.create("user", [ContentPart.create_text("first")]))
        ctx.append(Message.create("user", [ContentPart.create_text("second")]))
        ctx.add_anchor("before_last", 1)
        assert ("before_last", 1) in ctx.anchor_points

    def test_anchor_zero_index(self):
        ctx = Context.create()
        ctx.add_anchor("start", 0)
        assert ("start", 0) in ctx.anchor_points

    def test_get_anchor_index(self):
        ctx = Context.create()
        ctx.append(Message.create("user", [ContentPart.create_text("a")]))
        ctx.append(Message.create("user", [ContentPart.create_text("b")]))
        ctx.add_anchor("alpha", 1)
        ctx.add_anchor("beta", 2)
        # defaults at 0, alpha at 1 goes before messages(2), beta at 2 after messages
        assert ctx.get_anchor_index("alpha") == 2
        assert ctx.get_anchor_index("beta") == 4

    def test_anchor_insertion_sorting(self):
        ctx = Context.create()
        ctx.append(Message.create("user", [ContentPart.create_text("first")]))
        ctx.append(Message.create("user", [ContentPart.create_text("second")]))
        ctx.add_anchor("middle", 1)
        # defaults at 0, messages shifted to 2 by 2 appends, middle at 1 goes before messages
        assert ctx.anchor_points == [("system_prompt", 0), ("tools", 0), ("middle", 1), ("messages", 2)]

    def test_anchor_insertion_at_end(self):
        ctx = Context.create()
        ctx.append(Message.create("user", [ContentPart.create_text("first")]))
        ctx.add_anchor("after", 1)
        assert ("after", 1) in ctx.anchor_points

    def test_anchor_insertion_at_beginning(self):
        ctx = Context.create()
        ctx.add_anchor("before_all", 0)
        assert ("before_all", 0) in ctx.anchor_points

    def test_anchor_insertion_middle_of_existing(self):
        ctx = Context.create()
        ctx.append(Message.create("user", [ContentPart.create_text("a")]))
        ctx.append(Message.create("user", [ContentPart.create_text("b")]))
        ctx.append(Message.create("user", [ContentPart.create_text("c")]))
        ctx.append(Message.create("user", [ContentPart.create_text("d")]))
        ctx.append(Message.create("user", [ContentPart.create_text("e")]))
        ctx.add_anchor("first", 0)
        ctx.add_anchor("last", 5)
        ctx.add_anchor("middle", 3)
        indices = [idx for _, idx in ctx.anchor_points]
        assert indices == sorted(indices)


class TestContextHooks:
    """Tests for hook indexing in Context."""

    def test_hook_index_empty(self):
        ctx = Context.create()
        assert ctx.hook_index == {}

    def test_hook_index_with_hooks(self):
        ctx = Context.create()
        msg = Message.create("user", [ContentPart.create_text("hooked")])
        msg.raw_dict["_hook_ids"].append("hook_1")
        ctx.append(msg)
        assert "hook_1" in ctx.hook_index

    def test_hook_index_same_hook_on_multiple_messages(self):
        ctx = Context.create()
        msg1 = Message.create("user", [ContentPart.create_text("first")])
        msg1.raw_dict["_hook_ids"].append("shared_hook")
        msg2 = Message.create("assistant", [ContentPart.create_text("second")])
        msg2.raw_dict["_hook_ids"].append("shared_hook")
        ctx.append(msg1)
        ctx.append(msg2)
        assert len(ctx.hook_index["shared_hook"]) == 2


class TestContextFork:
    """Tests for Context.fork_insert_sequence()."""

    def test_fork_basic(self):
        ctx = Context.create()
        ctx.append(Message.create("user", [ContentPart.create_text("hello")]))
        fork = ctx.fork_insert_sequence()
        assert fork.id != ctx.id
        assert len(fork.messages) == 1
        assert fork.messages[0].content[0].text == "hello"

    def test_fork_inherits_content_map(self):
        parent = Context.create()
        parent.content_map["hash_1"] = "val_1"
        fork = parent.fork_insert_sequence()
        assert "hash_1" in fork.content_map

    def test_fork_inherits_anchor_points(self):
        ctx = Context.create()
        ctx.append(Message.create("user", [ContentPart.create_text("msg")]))
        ctx.add_anchor("test", 1)
        fork = ctx.fork_insert_sequence()
        assert ("test", 1) in fork.anchor_points

    def test_fork_appending_does_not_affect_parent(self):
        ctx = Context.create()
        ctx.append(Message.create("user", [ContentPart.create_text("parent msg")]))
        fork = ctx.fork_insert_sequence()
        # fork already has 1 copied message
        assert len(fork.messages) == 1
        assert fork.messages[0].content[0].text == "parent msg"
        fork.append(Message.create("assistant", [ContentPart.create_text("fork msg")]))
        # Parent unchanged, fork now has 2
        assert len(ctx.messages) == 1
        assert len(fork.messages) == 2
        assert fork.messages[1].content[0].text == "fork msg"

    def test_fork_with_count(self):
        ctx = Context.create()
        for i in range(4):
            text = f"msg {i}"
            ctx.append(Message.create("user", [ContentPart.create_text(text)]))
        fork = ctx.fork_insert_sequence(start=-2)
        assert len(fork.messages) == 2
        assert fork.messages[0].content[0].text == "msg 2"
        assert fork.messages[1].content[0].text == "msg 3"

    def test_fork_with_ids(self):
        ctx = Context.create()
        ids = []
        for i in range(3):
            text = f"msg {i}"
            msg = Message.create("user", [ContentPart.create_text(text)])
            ids.append(msg.id)
            ctx.append(msg)
        # After 3 default anchors (mutation_counter=3), messages have values 3,4,5
        # fork(start=4, end=5) captures the message with mutation_counter 4 (the 2nd message)
        fork = ctx.fork_insert_sequence(start=4, end=5)
        assert len(fork.messages) == 1
        assert fork.messages[0].id == ids[1]

    def test_fork_excludes_special_messages(self):
        sys_msg = SystemPromptMessage.create("system prompt")
        tools_msg = ToolDefinitionsMessage()
        ctx = Context.create(
            system_prompt_message=sys_msg,
            tool_definitions_message=tools_msg,
        )
        ctx.append(Message.create("user", [ContentPart.create_text("user msg")]))
        fork = ctx.fork_insert_sequence()
        non_special = [
            m for m in fork.messages
            if not isinstance(m, (SystemPromptMessage, ToolDefinitionsMessage))
        ]
        assert len(non_special) == 1

    def test_fork_with_new_system_prompt(self):
        old_sys = SystemPromptMessage.create("old prompt")
        new_sys = SystemPromptMessage.create("new prompt")
        ctx = Context.create(system_prompt_message=old_sys)
        fork = ctx.fork_insert_sequence(system_prompt_message=new_sys)
        assert fork.system_prompt_message.content[0].text == "new prompt"

    def test_fork_with_new_tool_definitions(self):
        old_tools = ToolDefinitionsMessage()
        new_tools = ToolDefinitionsMessage()
        ctx = Context.create(tool_definitions_message=old_tools)
        fork = ctx.fork_insert_sequence(tool_definitions_message=new_tools)
        assert fork.tool_definitions_message is not None
        assert fork.tool_definitions_message.id == new_tools.id

    def test_fork_message_independence_from_raw_dict(self):
        ctx = Context.create()
        msg = Message.create("user", [ContentPart.create_text("original")])
        ctx.append(msg)
        fork = ctx.fork_insert_sequence()
        fork.messages[0].raw_dict["content"] = [{"type": "text", "text": "modified"}]
        assert fork.messages[0].content[0].text == "modified"

    def test_fork_preserves_origin_context_id(self):
        ctx = Context.create()
        fork = ctx.fork_insert_sequence()
        assert fork._json_dict["origin_context_id"] == ctx.id

    def test_fork_with_empty_context(self):
        ctx = Context.create()
        fork = ctx.fork_insert_sequence()
        assert len(fork.messages) == 0


class TestContextForkPrivate:
    """Tests for Context._fork() — the mechanical fork implementation."""

    def _make_messages(self, ctx: Context, count: int) -> list[Message]:
        """Helper to create and append N user messages, returning them."""
        msgs = []
        for i in range(count):
            msg = Message.create("user", [ContentPart.create_text(f"msg {i}")])
            msgs.append(msg)
            ctx.append(msg)
        return msgs

    def test_fork_selects_only_selected_messages(self):
        ctx = Context.create()
        msgs = self._make_messages(ctx, 5)
        # Select only 2 messages (indices 1 and 3)
        selected = {msgs[1], msgs[3]}
        child = ctx._fork(selected)
        child_ids = {m.id for m in child.messages}
        assert {m.id for m in selected} == child_ids

    def test_fork_preserves_parent_anchor_order(self):
        """Anchors must be added in the same order as in the parent context."""
        ctx = Context.create()
        msgs = self._make_messages(ctx, 4)
        ctx.add_anchor("after_msg_1", 2)
        ctx.add_anchor("after_msg_3", 4)
        child = ctx._fork(set(msgs))
        names = [name for name, _ in child.anchor_points]
        # Parent anchors: (system_prompt,0), (tools,0), (after_msg_1,2), (messages,4), (after_msg_3,4)
        # Home anchors: msg0/1 → after_msg_1, msg2/3 → messages
        # after_msg_3 is excluded — no selected message has it as home anchor
        assert names == [
            "system_prompt", "tools", "after_msg_1", "messages",
        ]

    def test_fork_adds_missing_home_anchors(self):
        """Home anchors for selected messages are automatically added."""
        ctx = Context.create()
        msgs = self._make_messages(ctx, 4)
        ctx.add_anchor("custom", 4)
        # Select only the first 2 messages (home anchor should be "messages")
        selected = set(msgs[:2])
        child = ctx._fork(selected)
        assert "messages" in {name for name, _ in child.anchor_points}
        # custom anchor should NOT be added since no message is in that partition
        assert "custom" not in {name for name, _ in child.anchor_points}

    def test_fork_home_anchor_assignment(self):
        """Each message is appended to its home anchor (first anchor with position > msg index)."""
        ctx = Context.create()
        msgs = self._make_messages(ctx, 4)
        ctx.add_anchor("mid", 2)
        selected = set(msgs)
        child = ctx._fork(selected)
        # Messages 0 and 1 have home anchor "messages" (position 2 > them)
        # Messages 2 and 3 have home anchor "mid" (position 4 > them)
        # The fork should have correct anchor assignments
        assert len(child.messages) == 4

    def test_fork_replaces_old_system_prompt(self):
        """When a new system prompt replaces the old, old is removed and new is added."""
        ctx = Context.create()
        old_sys = SystemPromptMessage.create("old")
        new_sys = SystemPromptMessage.create("new")
        self._make_messages(ctx, 2)
        selected = set(ctx.messages)
        child = ctx._fork(selected, system_prompt_message=new_sys)
        assert child.system_prompt_message.id == new_sys.id
        assert child.system_prompt_message.content[0].text == "new"
        # Old parent message must NOT be in child
        assert old_sys.id not in {m.id for m in child.messages}

    def test_fork_replaces_old_tool_definitions(self):
        """When a new tool definitions replaces the old, old is removed and new is added."""
        ctx = Context.create()
        old_tools = ToolDefinitionsMessage()
        new_tools = ToolDefinitionsMessage()
        self._make_messages(ctx, 2)
        selected = set(ctx.messages)
        child = ctx._fork(selected, tool_definitions_message=new_tools)
        assert child.tool_definitions_message is not None
        assert child.tool_definitions_message.id == new_tools.id
        assert old_tools.id not in {m.id for m in child.messages}

    def test_fork_defaults_to_parent_special_messages(self):
        """If no special messages are selected or replaced, parent's are inherited."""
        ctx = Context.create(
            system_prompt_message=SystemPromptMessage.create("sys"),
            tool_definitions_message=ToolDefinitionsMessage(),
        )
        self._make_messages(ctx, 2)
        selected = set(ctx.messages[0:2])  # no special messages
        child = ctx._fork(selected)
        assert child.system_prompt_message is not None
        assert child.system_prompt_message.content[0].text == "sys"
        assert child.tool_definitions_message is not None

    def test_fork_falls_back_to_parent_special_messages_when_none(self):
        """Passing None for special messages falls back to parent's, not excludes them."""
        ctx = Context.create(
            system_prompt_message=SystemPromptMessage.create("sys"),
            tool_definitions_message=ToolDefinitionsMessage(),
        )
        self._make_messages(ctx, 2)
        selected = set(ctx.messages[0:2])
        child = ctx._fork(selected, system_prompt_message=None, tool_definitions_message=None)
        # or semantics: None falls back to parent
        assert child.system_prompt_message is not None
        assert child.system_prompt_message.content[0].text == "sys"
        assert child.tool_definitions_message is not None

    def test_fork_registers_hooks_on_child(self):
        """Messages with _hook_ids should be registered in the child's hook index."""
        ctx = Context.create()
        json_dict = {"role": "user", "content": [{"type": "text", "text": "hooked"}]}
        json_dict["_hook_ids"] = ["hook_1", "hook_2"]
        hooked_msg = Message.from_dict(json_dict)
        ctx.append(hooked_msg)
        assert set(ctx.hook_index.keys()) == {"hook_1", "hook_2"}
        child = ctx._fork({hooked_msg})
        assert set(child.hook_index.keys()) == {"hook_1", "hook_2"}

    def test_fork_child_registered_on_parent(self):
        """The child's ID is added to the parent's children list."""
        ctx = Context.create()
        msgs = self._make_messages(ctx, 2)
        child = ctx._fork(set(msgs))
        assert child.id in ctx._json_dict["children"]

    def test_fork_child_inherits_content_map(self):
        """The child's content_map is a copy of the parent's."""
        ctx = Context.create()
        ctx.content_map["a"] = "1"
        ctx.content_map["b"] = "2"
        msgs = self._make_messages(ctx, 2)
        child = ctx._fork(set(msgs))
        assert child.content_map == {"a": "1", "b": "2"}
        # Mutating child's content_map doesn't affect parent
        child.content_map["c"] = "3"
        assert "c" not in ctx.content_map

    def test_fork_anchor_completeness_always_includes_basic_anchors(self):
        """system_prompt, tools, and messages anchors are always present."""
        ctx = Context.create()
        msgs = self._make_messages(ctx, 2)
        # Fork with no anchors specified — only selected messages
        child = ctx._fork(set(msgs))
        names = {name for name, _ in child.anchor_points}
        assert "system_prompt" in names
        assert "tools" in names
        assert "messages" in names

    def test_fork_anchor_names_parameter_ignored_completeness(self):
        """Passing anchor_point_names adds to (not replaces) default anchors."""
        ctx = Context.create()
        self._make_messages(ctx, 2)
        # Passing a non-empty set should still include system_prompt, tools, messages
        child = ctx._fork(set(ctx.messages), anchor_point_names={"custom"})
        names = {name for name, _ in child.anchor_points}
        assert "system_prompt" in names
        assert "tools" in names
        assert "messages" in names


class TestContextCompaction:
    """Tests for Context.compaction methods (rolling_sequence_window, strip_thinking)."""

    def test_rolling_sequence_window_full_copy(self):
        ctx = Context.create()
        for i in range(5):
            ctx.append(Message.create("user", [ContentPart.create_text(f"msg {i}")]))
        window = ctx.rolling_sequence_window(5)
        # count >= non_special_count -> no fork needed, returns None
        assert window is None

    def test_rolling_sequence_window_partial(self):
        ctx = Context.create()
        for i in range(5):
            ctx.append(Message.create("user", [ContentPart.create_text(f"msg {i}")]))
        window = ctx.rolling_sequence_window(2)
        assert len(window.messages) == 2
        assert window.messages[0].content[0].text == "msg 3"
        assert window.messages[1].content[0].text == "msg 4"

    def test_rolling_sequence_window_with_thinking_messages(self):
        sys_msg = SystemPromptMessage.create("system prompt")
        ctx = Context.create(system_prompt_message=sys_msg)
        ctx.append(Message.create("user", [ContentPart.create_text("msg1")]))
        ctx.append(Message.create("assistant", [ContentPart.create_thinking("thinking")]))
        ctx.append(Message.create("user", [ContentPart.create_text("msg2")]))
        ctx.append(Message.create("assistant", [ContentPart.create_thinking("more")]))
        ctx.append(Message.create("user", [ContentPart.create_text("msg3")]))
        # Non-special msgs by seq: msg1(1), thinking(2), msg2(3), more(4), msg3(5)
        # lo = entries[-2][0] = 4, hi = 6 (full sequence space)
        window = ctx.rolling_sequence_window(2)
        assert len(window.messages) == 3
        assert window.messages[0].content[0].text == "system prompt"
        assert window.messages[1].content[0].type == "thinking"
        assert window.messages[2].content[0].text == "msg3"

    def test_rolling_sequence_window_anchored_messages(self):
        ctx = Context.create()
        ctx.append(Message.create("user", [ContentPart.create_text("msg1")]))
        ctx.add_anchor("insertion_point", 0)
        ctx.append(Message.create("assistant", [ContentPart.create_text("before_all")]))
        ctx.append(Message.create("user", [ContentPart.create_text("msg2")]))
        ctx.append(Message.create("user", [ContentPart.create_text("msg3")]))
        # Sorted by sequence: msg1(0), before_all(1), msg2(2), msg3(3)
        window = ctx.rolling_sequence_window(2)
        assert len(window.messages) == 2
        assert window.messages[0].content[0].text == "msg2"
        assert window.messages[1].content[0].text == "msg3"

    def test_rolling_sequence_window_empty_context(self):
        ctx = Context.create()
        window = ctx.rolling_sequence_window(1)
        # 0 non-special messages, 1 >= 0 -> returns None
        assert window is None

    def test_strip_thinking_removes_thinking_parts(self):
        ctx = Context.create()
        ctx.append(Message.create("user", [ContentPart.create_text("hello")]))
        ctx.append(Message.create("assistant", [ContentPart.create_thinking("thinking")]))
        ctx.append(Message.create("assistant", [
            ContentPart.create_thinking("more thinking"),
            ContentPart.create_text("answer"),
        ]))
        result = ctx.strip_thinking()
        assert len(result.messages) == 2
        assert result.messages[0].content[0].text == "hello"
        assert result.messages[1].content[0].text == "answer"

    def test_strip_thinking_skips_empty_messages(self):
        ctx = Context.create()
        ctx.append(Message.create("user", [ContentPart.create_text("hello")]))
        ctx.append(Message.create("assistant", [ContentPart.create_thinking("only thinking")]))
        result = ctx.strip_thinking()
        assert len(result.messages) == 1
        assert result.messages[0].content[0].text == "hello"

    def test_strip_thinking_removes_all_non_special(self):
        ctx = Context.create()
        ctx.append(Message.create("assistant", [ContentPart.create_thinking("only thinking")]))
        result = ctx.strip_thinking()
        assert len(result.messages) == 0

    def test_strip_thinking_preserves_parent(self):
        ctx = Context.create()
        ctx.append(Message.create("user", [ContentPart.create_text("hello")]))
        ctx.append(Message.create("assistant", [ContentPart.create_thinking("thinking")]))
        result = ctx.strip_thinking()
        assert len(ctx.messages) == 2
        assert len(result.messages) == 1

    def test_strip_thinking_copies_by_value(self):
        ctx = Context.create()
        msg = Message.create("user", [ContentPart.create_text("hello")])
        ctx.append(msg)
        result = ctx.strip_thinking()
        result.messages[0].raw_dict["content"] = [{"type": "text", "text": "modified"}]
        assert result.messages[0].content[0].text == "modified"
        # Parent unchanged
        assert ctx.messages[0].content[0].text == "hello"


class TestContextGetSpecialMessages:
    """Tests for system_prompt_message and tool_definitions_message properties."""

    def test_no_system_prompt_returns_none(self):
        ctx = Context.create()
        assert ctx.system_prompt_message is None

    def test_no_tool_definitions_returns_none(self):
        ctx = Context.create()
        assert ctx.tool_definitions_message is None

    def test_system_prompt_with_message_id_mismatch(self):
        sys_msg = SystemPromptMessage.create("prompt")
        ctx = Context.create(system_prompt_message=sys_msg)
        ctx._json_dict["system_prompt_message_id"] = "wrong_id"
        assert ctx.system_prompt_message is None

    def test_system_prompt_with_wrong_type(self):
        sys_msg = SystemPromptMessage.create("prompt")
        tools_msg = ToolDefinitionsMessage()
        ctx = Context.create(
            system_prompt_message=sys_msg,
            tool_definitions_message=tools_msg,
        )
        ctx._messages = []
        ctx._json_dict["messages"] = []
        ctx._json_dict["system_prompt_message_id"] = "nonexistent"
        assert ctx.system_prompt_message is None

    def test_tool_definitions_at_index_0(self):
        ctx = Context.create(tool_definitions_message=ToolDefinitionsMessage())
        assert ctx.tool_definitions_message is not None


class TestContextGatherDynamicMessages:
    """Tests for Context._gather_dynamic_messages()."""

    def test_gather_no_hooks(self):
        ctx = Context.create()
        ctx.append(Message.create("user", [ContentPart.create_text("no hooks")]))
        assert ctx._gather_dynamic_messages() == []

    def test_gather_with_hooks(self):
        ctx = Context.create()
        msg1 = Message.create("user", [ContentPart.create_text("hooked")])
        msg1.raw_dict["_hook_ids"].append("hook_1")
        ctx.append(msg1)
        dynamic = ctx._gather_dynamic_messages()
        assert len(dynamic) == 1
        assert dynamic[0] is msg1

    def test_gather_unique_messages(self):
        ctx = Context.create()
        msg1 = Message.create("user", [ContentPart.create_text("one")])
        msg1.raw_dict["_hook_ids"].append("shared_hook")
        msg2 = Message.create("assistant", [ContentPart.create_text("two")])
        msg2.raw_dict["_hook_ids"].append("shared_hook")
        ctx.append(msg1)
        ctx.append(msg2)
        gathered = ctx._gather_dynamic_messages()
        assert len(gathered) == 2

    def test_gather_does_not_duplicate_same_message(self):
        ctx = Context.create()
        msg = Message.create("user", [ContentPart.create_text("shared hook")])
        msg.raw_dict["_hook_ids"].append("shared")
        ctx.append(msg)
        gathered = ctx._gather_dynamic_messages()
        assert len(gathered) == 1
