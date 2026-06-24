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

        # Inherit content map from parent if provided, otherwise start empty
        if parent_context is not None:
            self._json_dict["content_map"] = dict(parent_context._json_dict["content_map"])
            self._json_dict["origin_context_id"] = parent_context.id

        # Add default anchor points only for fresh contexts (empty anchor list = no serialization)
        if not self._json_dict["anchor_points"]:
            assert(not self._json_dict["messages"])
            self._messages: list[Message] = []
            self.add_anchor("system_prompt", 0)
            self.add_anchor("tools", 0)
            self.add_anchor("messages", 0)
        else:
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
        messages).

        When multiple anchors share the same message index, *after_existing*
        controls the ordering in the anchor list.  With ``after_existing=True``
        (default), the new anchor is placed after existing anchors at the same
        position (it appears later in the list).  With ``after_existing=False``,
        it is placed before them.

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
        message.raw_dict["_sequence_number"] = self._json_dict["message_sequence"]
        self._shift_anchors_after_insert(anchor_list_index)
        self._add_message_to_hook_index(message)

    def fork(
        self,
        *,
        system_prompt_message: SystemPromptMessage | None = None,
        tool_definitions_message: ToolDefinitionsMessage | None = None,
        count: int | None = None,
        ids: list[str] | None = None,
    ) -> Context:
        """Fork this context, creating a new one that shares Message objects
        and inherits the content map.

        The fork has its own _json_dict and _messages list, so appending
        to one does not affect the other. However, the Message objects
        themselves are shared — mutating a message in the fork mutates it
        in the origin too.

        Args:
            system_prompt_message: Optional new system prompt for the fork.
            tool_definitions_message: Optional new tool definitions message.
            count: If set, fork the last n messages from the origin.
            ids: If set, fork only the messages whose IDs are in this list.
            Both omitted to copy all messages.

        Returns:
            A new Context with the selected messages.

        Raises:
            ValueError: If both count and ids are provided.
        """
        if count is not None and ids is not None:
            raise ValueError("Provide only count or ids, not both")

        # Select messages to fork
        if count is not None:
            selected = self.messages[-count:]
        elif ids is not None:
            selected = [m for m in self.messages if m.id in ids]
        else:
            selected = list(self.messages)

        ctx = Context.create(
            system_prompt_message=system_prompt_message or self.system_prompt_message,
            tool_definitions_message=tool_definitions_message or self.tool_definitions_message,
            parent_context=self,
        )
        ctx._json_dict["anchor_points"] = list(self._json_dict["anchor_points"])
        ctx._json_dict["message_sequence"] = self._json_dict["message_sequence"]
        # Remove special messages from selected to avoid duplicating them
        selected = [m for m in selected if m != self.system_prompt_message and m != self.tool_definitions_message]
        # Recreate new Message objects from raw dicts so the fork has independent copies
        selected_dicts = [msg.raw_dict for msg in selected]
        for raw_dict in selected_dicts:
            ctx.append(Message.from_dict(dict(raw_dict)))
        return ctx

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
