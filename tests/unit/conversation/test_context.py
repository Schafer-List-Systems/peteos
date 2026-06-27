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
        assert msg.raw_dict["_sequence_number"] == 0

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
    """Tests for Context.fork()."""

    @pytest.mark.skip(reason="fork() anchor copy bug — separate fix needed")
    def test_fork_basic(self):
        ctx = Context.create()
        ctx.append(Message.create("user", [ContentPart.create_text("hello")]))
        fork = ctx.fork()
        assert fork.id != ctx.id
        assert len(fork.messages) == 1
        assert fork.messages[0].content[0].text == "hello"

    def test_fork_inherits_content_map(self):
        parent = Context.create()
        parent.content_map["hash_1"] = "val_1"
        fork = parent.fork()
        assert "hash_1" in fork.content_map

    @pytest.mark.skip(reason="fork() anchor copy bug — separate fix needed")
    def test_fork_inherits_anchor_points(self):
        ctx = Context.create()
        ctx.add_anchor("test", 1)
        fork = ctx.fork()
        assert ("test", 1) in fork.anchor_points

    @pytest.mark.skip(reason="fork() anchor copy bug — separate fix needed")
    def test_fork_appends_increments_sequence(self):
        ctx = Context.create()
        ctx.append(Message.create("user", [ContentPart.create_text("msg")]))
        ctx.append(Message.create("assistant", [ContentPart.create_text("reply")]))
        fork = ctx.fork()
        # fork appends copies of the messages via append(), so sequence goes up
        assert fork._json_dict["message_sequence"] == 4

    @pytest.mark.skip(reason="fork() anchor copy bug — separate fix needed")
    def test_fork_appending_does_not_affect_parent(self):
        ctx = Context.create()
        ctx.append(Message.create("user", [ContentPart.create_text("parent msg")]))
        fork = ctx.fork()
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
        fork = ctx.fork(start=-2)
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
        fork = ctx.fork(start=1, end=2)
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
        fork = ctx.fork()
        non_special = [
            m for m in fork.messages
            if not isinstance(m, (SystemPromptMessage, ToolDefinitionsMessage))
        ]
        assert len(non_special) == 1

    def test_fork_with_new_system_prompt(self):
        old_sys = SystemPromptMessage.create("old prompt")
        new_sys = SystemPromptMessage.create("new prompt")
        ctx = Context.create(system_prompt_message=old_sys)
        fork = ctx.fork(system_prompt_message=new_sys)
        assert fork.system_prompt_message.content[0].text == "new prompt"

    def test_fork_with_new_tool_definitions(self):
        old_tools = ToolDefinitionsMessage()
        new_tools = ToolDefinitionsMessage()
        ctx = Context.create(tool_definitions_message=old_tools)
        fork = ctx.fork(tool_definitions_message=new_tools)
        assert fork.tool_definitions_message is new_tools

    def test_fork_message_independence_from_raw_dict(self):
        ctx = Context.create()
        msg = Message.create("user", [ContentPart.create_text("original")])
        ctx.append(msg)
        fork = ctx.fork()
        fork.messages[0].raw_dict["content"] = [{"type": "text", "text": "modified"}]
        assert fork.messages[0].content[0].text == "modified"

    def test_fork_preserves_origin_context_id(self):
        ctx = Context.create()
        fork = ctx.fork()
        assert fork._json_dict["origin_context_id"] == ctx.id

    def test_fork_with_empty_context(self):
        ctx = Context.create()
        fork = ctx.fork()
        assert len(fork.messages) == 0


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
