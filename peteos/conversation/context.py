from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

from .message import Message
from peteos.utils import get_logger

from .system_prompt_message import SystemPromptMessage
from .tool_definitions_message import ToolDefinitionsMessage

_logger = get_logger(__name__)


class Context:
    """Wrapper around a serialized context dict.

    The dict represents a single step in a session: a set of messages that form
    one turn. Follows the agentic_process pattern  -  the dict is the source of truth.

    The internal _messages list is kept in sync with json_dict["messages"].

    Supports forked contexts with inherited content maps,
    and dynamic message resolution via content hashes.

    Anchor points partition the messages array. Each anchor stores a ``(name,
    index)`` pair where *index* is a non-negative absolute position acting as
    an **end iterator**  -  it points to the first element *after* the partition.
    Appending at an anchor inserts at that position and shifts all subsequent
    messages and anchor indices by one.

    System prompt messages and tool definitions messages are only handled in
    factory methods (``create`` / ``fork``), never in ``__init__`` which is
    used for deserialization. System prompt is always at index 0; tool
    definitions message is always at index 1 (or index 0 if no system prompt).

    **Invariants  -  guaranteed by the append/add_anchor methods and assumed by
    fork(), rolling_sequence_window(), and rolling_token_window():**

    1. **Anchor list is sorted ascending by position.** ``add_anchor`` always
       inserts at the correct sorted position; ties are broken by insertion
       order.
    2. **The last anchor always has position equal to message_count.** This
       is the end-of-all-messages sentinel. ``add_anchor`` bounds-checks
       ``msg_index <= len(self._messages)`` and ``append`` increments
       ``message_sequence`` in lockstep.
    3. **Every message at index *I* has an anchor with position > *I*.**
       The message's home anchor is the first anchor in the list with
       position >= *I* + 1. The last sentinel (invariant 2) guarantees
       such an anchor always exists.
    4. **Messages within a partition maintain their insertion/display
       order, which matches their sequence ordering.** ``append`` only
       inserts  -  it never reorders existing messages.

    Example:
        >>> ctx = Context({"messages": [
        ...     {"role": "user", "id": "1", "content": [{"type": "text", "text": "Hi"}]}
        ... ]})
        >>> ctx.raw_dict
        {'messages': [{'role': 'user', 'id': '1', 'content': [{'type': 'text', 'text': 'Hi'}]}]}
    """

    # ------------------------------------------------------------------ #
    # Factory methods (where special messages are injected)
    # ------------------------------------------------------------------ #

    @classmethod
    def create(
        cls,
        system_prompt_message: SystemPromptMessage | None = None,
        tool_definitions_message: ToolDefinitionsMessage | None = None,
        parent_context: Context | None = None,
    ) -> "Context":
        """Create a new empty context.

        The system prompt message and the tool definitions message are
        placed in the first slots in that order (if present).
        If a parent context is provided, the content map is inherited from it.

        Args:
            system_prompt_message: Optional system prompt for this context.
            tool_definitions_message: Optional tool definitions for this context.
            parent_context: Optional parent to inherit the content map from.

        Returns:
            A new Context instance.
        """
        ctx = cls({}, parent_context=parent_context)
        # Add default anchor points via append (not in __init__ for clean deserialization)
        ctx.add_anchor("system_prompt", 0)
        ctx.add_anchor("tools", 0)
        ctx.add_anchor("messages", 0)
        if system_prompt_message is not None:
            ctx.append(system_prompt_message, anchor_point="system_prompt")
            ctx.raw_dict["system_prompt_message_id"] = system_prompt_message.id
        if tool_definitions_message is not None:
            ctx.append(tool_definitions_message, anchor_point="tools")
            ctx.raw_dict["tool_definitions_message_id"] = tool_definitions_message.id
        return ctx

    @staticmethod
    def load(path: str) -> "Context":
        """Load a context from a file."""
        with open(path, "r") as f:
            json_dict = json.load(f)
        return Context.load_from_dict(json_dict)

    @staticmethod
    def load_from_dict(json_dict: dict) -> "Context":
        """Load a context from a dictionary.

        The context is built exactly as it was serialized  -  no system
        prompt message is injected or modified.
        """
        return Context(json_dict)

    def save(self, session_dir: str | Path) -> None:
        """Save the context to a JSON file in the session directory.

        The file is named ``{context_id}.json``.

        Args:
            session_dir: Path to the session directory on disk.
        """
        path = Path(session_dir) / f"{self.id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self._json_dict, f, indent=2)

    def total_token_count(self, encoding: str = "cl100k_base") -> int:
        """Sum of token counts across all messages.

        Special messages (system prompt, tool definitions) are already
        included in ``self.messages``.

        Args:
            encoding: Tiktoken encoding name.

        Returns:
            Total token count as an integer.
        """
        return sum(msg.count_tokens(encoding) for msg in self.messages)

    # ------------------------------------------------------------------ #
    # Internal construction (used by all factory methods)
    # ------------------------------------------------------------------ #

    def __init__(
        self,
        json_dict: dict,
        parent_context: Context | None = None,
    ) -> None:
        """
        Context objects shall only be created using the factory functions!

        Args:
            json_dict: The raw serialized context dict.
            parent_context: If provided, inherit the parent's content map.
        """
        self._json_dict = json_dict
        self._json_dict.setdefault("messages", [])
        self._json_dict.setdefault("id", str(uuid.uuid4()))
        self._json_dict.setdefault("content_map", {})
        self._json_dict.setdefault("anchor_points", [])
        self._json_dict.setdefault("message_sequence", 0)
        self._json_dict.setdefault("_mutation_counter", 0)
        self._json_dict.setdefault("children", [])

        # Inherit content map from parent if provided, otherwise start empty
        if parent_context is not None:
            self._json_dict["content_map"] = dict(parent_context._json_dict["content_map"])
            self._json_dict["origin_context_id"] = parent_context.id
            self._json_dict["parent_mutation_counter"] = self._json_dict["_mutation_counter"]

        # __init__ is purely a deserializer — no side effects, no default anchors.
        # Default anchors are added by Context.create() for fresh contexts.
        self._messages: list[Message] = [
            Message.from_dict(msg) for msg in self._json_dict.get("messages", [])
        ]
        self._anchor_sequences: list[int] = list(
            self._json_dict.get("_anchor_sequences", [])
        )

        self._hook_index: dict[str, list[Message]] = {}
        self._update_hook_index()

    @property
    def id(self) -> str:
        """Return the context ID."""
        return self._json_dict.get("id", "")

    @property
    def raw_dict(self) -> dict:
        """Return the wrapped serialized dict."""
        return self._json_dict

    @property
    def messages(self) -> list[Message]:
        """Return the list of messages in this context."""
        return self._messages

    @property
    def message_count(self) -> int:
        """Return the number of messages in this context (same as the sequence counter)."""
        return self._json_dict["message_sequence"]

    @property
    def content_map(self) -> dict[str, str]:
        """Return the content hash → string map for this context."""
        return self._json_dict["content_map"]

    @property
    def hook_index(self) -> dict[str, list[Message]]:
        """Return the hook ID → messages index."""
        return self._hook_index

    @property
    def system_prompt_message(self) -> SystemPromptMessage | None:
        """Return the system prompt message if present."""
        message_id = self._json_dict.get("system_prompt_message_id")
        if not message_id:
            return None
        if not self._messages:
            _logger.error("system_prompt_message_id set but no messages found")
            return None
        msg = self._messages[0]
        if msg.id != message_id:
            _logger.error(f"Expected SystemPromptMessage with id {message_id}, got {msg.id}")
            return None
        if not isinstance(msg, SystemPromptMessage):
            _logger.error(f"Expected SystemPromptMessage with id {message_id}, got {msg.__class__.__name__}")
            return None
        return msg

    @property
    def tool_definitions_message(self) -> ToolDefinitionsMessage | None:
        """Return the tool definitions message if present."""
        message_id = self._json_dict.get("tool_definitions_message_id")
        if not message_id:
            return None

        msg = None
        for candidate in self.messages[:2]:
            if candidate.id == message_id:
                msg = candidate
                break
        if msg is None:
            _logger.error(f"Expected ToolDefinitionsMessage with id {message_id}, not found in first 2 messages")
            return None
        if not isinstance(msg, ToolDefinitionsMessage):
            _logger.error(f"Expected ToolDefinitionsMessage with id {message_id}, got {msg.__class__.__name__}")
            return None
        return msg

    def _add_message_to_hook_index(self, message: Message) -> None:
        """Add a single message's hook IDs to the hook index.

        Args:
            message: The message whose hooks to index.
        """
        for hook_id in message.hook_ids:
            if hook_id not in self._hook_index:
                self._hook_index[hook_id] = []
            self._hook_index[hook_id].append(message)

    def _update_hook_index(self) -> None:
        """Rebuild the hook index from all messages' hook IDs."""
        self._hook_index.clear()
        for msg in self._messages:
            self._add_message_to_hook_index(msg)

    @property
    def anchor_points(self) -> list[tuple[str, int]]:
        """Return the ordered list of (name, index) anchor points.

        Each index is a non-negative absolute position into the messages
        array, acting as an **end iterator**  -  it points to the first
        element *after* the partition.
        """
        return list(self._json_dict["anchor_points"])

    def get_anchor_index(self, name: str) -> int:
        """Return the position of *name* in the anchor point list."""
        return next(i for i, (n, _) in enumerate(self._json_dict["anchor_points"]) if n == name)

    def get_anchor_msg_index(self, name: str) -> int:
        """Return the message index stored for anchor *name*."""
        return self._json_dict["anchor_points"][self.get_anchor_index(name)][1]

    def add_anchor(self, name: str, msg_index: int, after_existing: bool = True) -> None:
        """Register a new anchor point.

        The *msg_index* is always a non-negative absolute position into the
        messages array (0 = before all messages, len(messages) = after all
        messages). Each anchor records the current sequence counter so it can
        be selected during forking.

        Args:
            name: The anchor point name.
            msg_index: Absolute message index (0 to len(messages)).
            after_existing: Ordering when another anchor already occupies this
                position.

        Raises:
            ValueError: If *name* already exists.
        """
        if any(n == name for n, _ in self._json_dict["anchor_points"]):
            raise ValueError(f"Anchor point already exists: {name}")

        assert 0 <= msg_index <= len(self._messages), (
            f"Anchor '{name}' index {msg_index} is out of bounds "
            f"[0, {len(self._messages)}]"
        )

        # Record the current mutation count for this anchor
        self._anchor_sequences.append(self._json_dict["_mutation_counter"])
        self._json_dict["_mutation_counter"] += 1

        for pos, (_, existing_idx) in enumerate(self._json_dict["anchor_points"]):
            if after_existing and existing_idx > msg_index:
                self._json_dict["anchor_points"].insert(pos, (name, msg_index))
                return
            if not after_existing and existing_idx >= msg_index:
                self._json_dict["anchor_points"].insert(pos, (name, msg_index))
                return
        self._json_dict["anchor_points"].append((name, msg_index))

    def _shift_anchors_after_insert(self, anchor_list_index: int) -> None:
        """Increment stored indices for all anchors at or after *anchor_list_index* in the list."""
        for i in range(anchor_list_index, len(self._json_dict["anchor_points"])):
            n, idx = self._json_dict["anchor_points"][i]
            new_idx = idx + 1
            self._json_dict["anchor_points"][i] = (n, new_idx)
            assert 0 <= new_idx <= len(self._messages), (
                f"Anchor '{n}' index {new_idx} is out of bounds "
                f"[0, {len(self._messages)}]"
            )

    def append(self, message: Message, anchor_point: str = "messages") -> None:
        """Insert a Message at the anchor point's end-iterator position.

        All messages from that position onward are shifted one slot to the
        right. Every anchor that appears at or after the target anchor in the
        list and whose stored index is at or past the insertion point is
        incremented by 1.

        Args:
            message: The Message to append.
            anchor_point: Name of the anchor point. Defaults to "messages".
        """
        anchor_list_index = self.get_anchor_index(anchor_point)
        msg_index = self.raw_dict["anchor_points"][anchor_list_index][1]

        self._messages.insert(msg_index, message)
        self.raw_dict["messages"].insert(msg_index, message.raw_dict)
        self._json_dict["message_sequence"] += 1
        # Assign sequence number from the mutation counter (0-based, no remapping)
        message.raw_dict["_sequence_number"] = self._json_dict["_mutation_counter"]
        self._json_dict["_mutation_counter"] += 1
        self._shift_anchors_after_insert(anchor_list_index)
        self._add_message_to_hook_index(message)

    def _fork(
        self,
        selected_messages: set[Message],
        anchor_point_names: set[str] | None = None,
        *,
        system_prompt_message: SystemPromptMessage | None = None,
        tool_definitions_message: ToolDefinitionsMessage | None = None,
    ) -> Context:
        """Mechanical forking given selected Message objects and anchor point names.

        The mutation counter is managed automatically by the child's append()
        and add_anchor() calls  -  no manual manipulation.

        Args:
            selected_messages: Message objects to copy into the fork.
            anchor_point_names: Set of anchor point names to copy.
            system_prompt_message: Optional new system prompt for the fork.
            tool_definitions_message: Optional new tool definitions message.

        Returns:
            A new Context with the selected messages and anchors.
        """
        # Ensure completeness: consistent special messages in selected_messages.
        # If an arg replaces the parent's special message, remove the old one first.
        system_prompt_message = system_prompt_message or self.system_prompt_message
        if self.system_prompt_message in selected_messages:
            selected_messages.discard(self.system_prompt_message)
        if system_prompt_message is not None:
            selected_messages.add(system_prompt_message)
        system_prompt_id = system_prompt_message.id if system_prompt_message else None

        tool_definitions_message = tool_definitions_message or self.tool_definitions_message
        if self.tool_definitions_message in selected_messages:
            selected_messages.discard(self.tool_definitions_message)
        if tool_definitions_message is not None:
            selected_messages.add(tool_definitions_message)
        tool_definitions_id = tool_definitions_message.id if tool_definitions_message else None

        # Ensure completeness: start with anchor_point_names or empty set
        if anchor_point_names is None:
            anchor_point_names_set: set[str] = set()
        else:
            anchor_point_names_set = set(anchor_point_names)
        anchor_point_names_set.add("system_prompt")
        anchor_point_names_set.add("tools")
        anchor_point_names_set.add("messages")

        # Find each message's home anchor: first anchor with position > msg position
        for msg_index, msg in enumerate(self.messages):
            if msg in selected_messages:
                for anchor_name, anchor_msg_index in self.anchor_points:
                    if anchor_msg_index > msg_index:
                        anchor_point_names_set.add(anchor_name)
                        break

        # Build ordered list preserving parent anchor order
        anchor_point_names_ordered: list[str] = [
            name for name, _ in self.anchor_points
            if name in anchor_point_names_set
        ]

        # Create child — build from scratch via __init__
        child = Context(json_dict={}, parent_context=self)
        child._json_dict["system_prompt_message_id"] = system_prompt_id
        child._json_dict["tool_definitions_message_id"] = tool_definitions_id

        # Add anchors in parent order
        for name in anchor_point_names_ordered:
            child.add_anchor(name, 0)

        # Append selected messages from parent in display order via their home anchor
        for msg_index, msg in enumerate(self.messages):
            if msg in selected_messages:
                home = None
                for anchor_name, anchor_msg_index in self.anchor_points:
                    if anchor_msg_index > msg_index:
                        home = anchor_name
                        break
                child.append(Message.from_dict(dict(msg.raw_dict)), anchor_point=home)

        # Append replacement special messages not present in parent
        if system_prompt_message and system_prompt_message not in self.messages:
            child.append(Message.from_dict(dict(system_prompt_message.raw_dict)), anchor_point="system_prompt")
        if tool_definitions_message and tool_definitions_message not in self.messages:
            child.append(Message.from_dict(dict(tool_definitions_message.raw_dict)), anchor_point="tools")

        # Register child on parent
        self._json_dict["children"].append(child.id)

        return child

    def fork_insert_sequence(
        self,
        *,
        system_prompt_message: SystemPromptMessage | None = None,
        tool_definitions_message: ToolDefinitionsMessage | None = None,
        start: int | None = None,
        end: int | None = None,
    ) -> Context:
        """Fork this context with Python-style slice semantics on mutation counter values.

        Args:
            system_prompt_message: Optional new system prompt for the fork.
            tool_definitions_message: Optional new tool definitions message.
            start: Start of mutation counter range (inclusive, supports negative indices).
                None means from the first mutation.
            end: End of mutation counter range (exclusive, supports negative indices).
                None means up to the last mutation.
                Both ``start`` and ``end`` being None copies the entire context.

        Returns:
            A new Context with the selected messages.
        """
        max_mutation = self._json_dict["_mutation_counter"]
        lo = 0 if start is None else (max_mutation + start if start < 0 else start)
        hi = max_mutation if end is None else (max_mutation + end if end < 0 else end)

        # Select messages within range
        selected: set[Message] = set()
        for msg in self.messages:
            seq = msg.raw_dict.get("_sequence_number")
            if seq is not None and lo <= seq < hi:
                selected.add(msg)

        # Select anchors whose mutation counter falls within range
        anchor_names: set[str] = set()
        for idx, (name, _) in enumerate(self.anchor_points):
            if idx < len(self._anchor_sequences):
                if lo <= self._anchor_sequences[idx] < hi:
                    anchor_names.add(name)

        # Ensure completeness: add home anchors for selected messages
        home_anchors: set[str] = set()
        for msg in selected:
            parent_idx = self.messages.index(msg)
            for anc_name, anc_pos in self.anchor_points:
                if anc_pos > parent_idx:
                    home_anchors.add(anc_name)
                    break
        anchor_names = anchor_names | home_anchors

        return self._fork(
            selected,
            anchor_names,
            system_prompt_message=system_prompt_message,
            tool_definitions_message=tool_definitions_message,
        )

    def _gather_dynamic_messages(self) -> list[Message]:
        """Gather all unique messages that have at least one hook from the hook index."""
        dynamic_messages: list[Message] = []
        seen_ids: set[str] = set()
        for hook_messages in self._hook_index.values():
            for msg in hook_messages:
                if msg.id not in seen_ids:
                    seen_ids.add(msg.id)
                    dynamic_messages.append(msg)
        return dynamic_messages

    def rolling_sequence_window(self, count: int) -> "Context | None":
        """Fork including the last *count* non-special messages by sequence number.

        Special messages are always included if they fall within the sequence
        range. All messages between the selected messages are also included.

        Returns:
            A new Context with the selected messages, or None if the window
            covers all non-special messages (no fork needed).
        """
        # 1. Count non-special messages
        non_special_count = sum(
            1 for msg in self.messages
            if msg not in (self.system_prompt_message, self.tool_definitions_message)
        )
        if count >= non_special_count:
            return None

        # 2. Collect non-special messages with sequence numbers
        entries: list[tuple[int, Message]] = []
        for msg in self.messages:
            if msg not in (self.system_prompt_message, self.tool_definitions_message):
                seq = msg.raw_dict.get("_sequence_number")
                if seq is not None:
                    entries.append((seq, msg))

        # 3. Sort by sequence number ascending
        entries.sort(key=lambda e: e[0])

        # 4. Determine sequence range on the full entity sequence space
        lo = entries[-count][0]
        hi = self._json_dict["_mutation_counter"]

        # 5. Select messages
        selected: set[Message] = set()
        for seq, msg in entries:
            if lo <= seq < hi:
                selected.add(msg)

        # 6. Fork via private method
        return self._fork(selected)

    def rolling_token_window(
        self,
        max_tokens: int,
        encoding: str = "cl100k_base",
    ) -> "Context | None":
        """Fork keeping messages from the end until total tokens <= *max_tokens*.

        Messages are selected by sequence number (same ordering as
        ``rolling_sequence_window``). The total token count includes both
        special messages (system prompt, tool definitions) and non-special
        messages.

        Returns:
            A new Context whose total tokens are <= max_tokens, or None if
            no messages need to be dropped.
        """
        non_special = [
            (msg.raw_dict.get("_sequence_number"), msg)
            for msg in self.messages
            if msg not in (self.system_prompt_message, self.tool_definitions_message)
            and msg.raw_dict.get("_sequence_number") is not None
        ]
        non_special.sort(key=lambda e: e[0])
        if not non_special:
            return None

        # Special messages total
        special_tokens = sum(
            msg.count_tokens(encoding)
            for msg in (self.system_prompt_message, self.tool_definitions_message)
            if msg is not None
        )

        # Token counts for non-special messages
        msg_tokens: list[tuple[int, Message, int]] = []
        for seq, msg in non_special:
            tc = msg.count_tokens(encoding)
            msg_tokens.append((seq, msg, tc))

        total = special_tokens + sum(tc for _, _, tc in msg_tokens)
        if total <= max_tokens:
            return None

        # Start with special messages' tokens, accumulate from the end
        # backwards until we'd exceed max_tokens
        running = special_tokens
        kept = 0
        for i in range(len(msg_tokens) - 1, -1, -1):
            tc = msg_tokens[i][2]
            if running + tc > max_tokens:
                break
            running += tc
            kept += 1

        # Determine sequence range from the kept messages (last `kept`)
        # When kept == 0, no non-special messages fit  -  fork with only special messages.
        hi = self._json_dict["_mutation_counter"]
        if kept == 0:
            start_seq = hi  # empty range [hi, hi) → no non-special messages
        else:
            start_seq = msg_tokens[len(msg_tokens) - kept][0]

        # Select messages
        selected: set[Message] = set()
        for seq, msg in non_special:
            if start_seq <= seq < hi:
                selected.add(msg)

        # Fork via private method
        return self._fork(selected)

    def strip_thinking(self) -> "Context":
        """Fork removing thinking content parts from all messages.

        Empty messages (no non-thinking content parts) are skipped.
        Messages are copied by value, not shared.
        """
        child = self.fork_insert_sequence()
        stripped: list[Message] = []
        for msg in child.messages:
            raw = dict(msg.raw_dict)
            raw_content = [
                part.raw_dict for part in msg.content
                if part.type != "thinking"
            ]
            if raw_content:
                raw["content"] = raw_content
                stripped.append(Message.from_dict(raw))
        child._messages = stripped
        child._json_dict["messages"] = [m.raw_dict for m in stripped]
        return child
