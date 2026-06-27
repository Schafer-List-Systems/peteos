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
    one turn. Follows the agentic_process pattern — the dict is the source of truth.

    The internal _messages list is kept in sync with json_dict["messages"].

    Supports forked contexts with inherited content maps,
    and dynamic message resolution via content hashes.

    Anchor points partition the messages array. Each anchor stores a ``(name,
    index)`` pair where *index* is a non-negative absolute position acting as
    an **end iterator** — it points to the first element *after* the partition.
    Appending at an anchor inserts at that position and shifts all subsequent
    messages and anchor indices by one.

    System prompt messages and tool definitions messages are only handled in
    factory methods (``create`` / ``fork``), never in ``__init__`` which is
    used for deserialization. System prompt is always at index 0; tool
    definitions message is always at index 1 (or index 0 if no system prompt).

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

        The context is built exactly as it was serialized — no system
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
        self._json_dict.setdefault("_message_sequence_counter", 0)
        self._json_dict.setdefault("_anchor_sequence_counter", 0)
        self._json_dict.setdefault("children", [])

        # Inherit content map from parent if provided, otherwise start empty
        if parent_context is not None:
            self._json_dict["content_map"] = dict(parent_context._json_dict["content_map"])
            self._json_dict["origin_context_id"] = parent_context.id

        # Add default anchor points only for fresh contexts (empty anchor list = no serialization)
        if not self._json_dict["anchor_points"]:
            assert(not self._json_dict["messages"])
            self._messages: list[Message] = []
            self._anchor_sequences: list[int] = []
            self.add_anchor("system_prompt", 0)
            self.add_anchor("tools", 0)
            self.add_anchor("messages", 0)
        else:
            self._anchor_sequences: list[int] = list(
                self._json_dict.get("_anchor_sequences", [])
            )
            self._messages: list[Message] = [
                Message.from_dict(msg) for msg in self._json_dict["messages"]
            ]

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
        array, acting as an **end iterator** — it points to the first
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

        # Record the current sequence number for this anchor
        self._anchor_sequences.append(self._json_dict["_anchor_sequence_counter"])
        self._json_dict["_anchor_sequence_counter"] += 1

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
        # Assign sequence number from the message counter (0-based, no remapping)
        message.raw_dict["_sequence_number"] = self._json_dict["_message_sequence_counter"]
        self._json_dict["_message_sequence_counter"] += 1
        self._shift_anchors_after_insert(anchor_list_index)
        self._add_message_to_hook_index(message)

    def fork(
        self,
        *,
        system_prompt_message: SystemPromptMessage | None = None,
        tool_definitions_message: ToolDefinitionsMessage | None = None,
        start: int | None = None,
        end: int | None = None,
    ) -> Context:
        """Fork this context with Python-style slice semantics on sequence numbers.

        Forking uses a sequence counter system: every entity (message or anchor
        point) stores the sequence number it was assigned when inserted. The
        context's ``_sequence_counter`` tracks the next value to assign.

        **Fork algorithm (sequence range slicing):**

        1. Resolve slice semantics → absolute ``(lo, hi)`` bounds:

           ``resolve_slice(start, end, max_seq)``:
               if start is None:      start = 0
               if end is None:         end = max_seq + 1        # exclusive past last
               if start < 0:           start = max_seq + 1 + start   # -1 → last seq
               if end < 0:             end = max_seq + 1 + end
               return (start, end)

           Example: parent has seq 0..9, ``(-3, None)`` → ``(7, 10)``

        2. Collect selected entities from parent, keyed by sequence number:

           ``selected_msgs = {seq: msg for (seq, msg) in parent._messages_by_seq
                              if lo <= seq < hi}``

           ``selected_anchors = {seq: anchor for (seq, anchor) in parent._anchors_by_seq
                                 if lo <= seq < hi}``

        3. Create child with fresh special messages, continue parent's sequence
           numbering:

           ``child = Context.create(system_prompt_message, tool_definitions_message, parent)``
           ``child._sequence_counter = lo``  # no remapping needed

        4. Re-insert selected entities in **display order** (not sequence order):

           ``interleaved = []``
           ``for idx, entity in enumerate(parent._ordered_entities):``
           ``    seq = entity.sequence``
           ``    if entity.type == "message"  and seq in selected_msgs:``
           ``        interleaved.append((idx, "msg", seq))``
           ``    if entity.type == "anchor"   and seq in selected_anchors:``
           ``        interleaved.append((idx, "anchor", seq))``
           ``interleaved.sort(by=idx)``  # preserve display order

        5. Insert into child using append/add_anchor in display order:

           ``for (display_idx, etype, seq) in interleaved:``
           ``    if etype == "msg":``
           ``        child.append(Message.from_dict(selected_msgs[seq].raw_dict),``
           ``                     anchor_point="messages")``
           ``    else:``
           ``        child.add_anchor(selected_anchors[seq].name,``
           ``                         child.get_anchor_msg_index(selected_anchors[seq].name))``

        The forked context's sequence counter starts at ``lo`` so no sequence
        remapping is necessary — entities retain their original sequence values.

        Args:
            system_prompt_message: Optional new system prompt for the fork.
            tool_definitions_message: Optional new tool definitions message.
            start: Start of sequence range (inclusive, supports negative indices).
                None means from the first sequence.
            end: End of sequence range (exclusive, supports negative indices).
                None means up to the last sequence.
                Both ``start`` and ``end`` being None copies the entire context.

        Returns:
            A new Context with the selected messages.
        """
        # ── Resolve range to absolute (lo, hi) ────────────────────
        max_seq = self._json_dict["_message_sequence_counter"]
        lo = 0 if start is None else (max_seq + start if start < 0 else start)
        hi = max_seq if end is None else (max_seq + end if end < 0 else end)

        # ── Create child with fresh special messages ──────────────
        child = Context.create(
            system_prompt_message or self.system_prompt_message,
            tool_definitions_message or self.tool_definitions_message,
            parent_context=self,
        )
        # Prevent counter conflict with messages added by Context.create()
        child._json_dict["_message_sequence_counter"] = max(
            lo, child._json_dict["_message_sequence_counter"]
        )
        child._anchor_sequences = []  # fresh state

        # ── Build seq → Message lookup (non-special messages) ─────
        msg_by_seq: dict[int, Message] = {}
        for msg in self.messages:
            if msg not in (self.system_prompt_message, self.tool_definitions_message):
                seq = msg.raw_dict.get("_sequence_number")
                if seq is not None:
                    msg_by_seq[seq] = msg

        # ── Append selected messages in sequence order ────────────
        # Counter auto-assigns correct sequence numbers — no remapping.
        for seq in sorted(msg_by_seq.keys()):
            if lo <= seq < hi:
                child.append(
                    Message.from_dict(dict(msg_by_seq[seq].raw_dict)),
                    anchor_point="messages",
                )

        # ── Set child's message sequence counter so new messages continue
        #    from the end of the forked range (no remapping needed) ──
        child._json_dict["_message_sequence_counter"] = hi

        # ── Register child on parent ──────────────────────────────
        self._json_dict["children"].append(child.id)

        return child

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

    def rolling_sequence_window(self, count: int) -> "Context":
        """Fork including the last *count* non-special messages by sequence number.

        Special messages are always included if they fall within the sequence
        range. All messages between the selected messages are also included.

        Args:
            count: Number of non-special messages to keep from the end.

        Returns:
            A new Context with the selected messages.
        """
        # 1. Early return: count covers all non-special messages
        non_special_count = sum(
            1 for msg in self.messages
            if msg not in (self.system_prompt_message, self.tool_definitions_message)
        )
        if count >= non_special_count:
            return self.fork()

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
        hi = self._json_dict["_message_sequence_counter"]

        # 5. Fork
        return self.fork(start=lo, end=hi)

    def strip_thinking(self) -> "Context":
        """Fork removing thinking content parts from all messages.

        Empty messages (no non-thinking content parts) are skipped.
        Messages are copied by value, not shared.
        """
        child = self.fork()
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
