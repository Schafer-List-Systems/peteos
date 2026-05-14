"""ChatHistory class for managing chat messages and request fields."""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from .message import Message
from .contentpart import ContentPart


class ChatHistory:
    """Container for chat messages and request configuration.

    ChatHistory represents the complete prompt being sent to the chatbot.
    All data is stored as messages with roles. Special content types
    (tools, system prompts, etc.) are represented as messages with
    specific roles. Additional request fields (generation config) are stored
    in a single config dictionary.

    Messages may be unanchored (default conversation history) or anchored
    (persistent across compaction). The ``messages`` property returns a
    merged view: front anchors, then unanchored, then back anchors.

    Example:
        >>> history = ChatHistory()
        >>> history.append_message(Message(...))
        >>> history.append_message(Message(...), anchor="front")

    Attributes:
        messages: Merged view of all messages in order.
        generation_config: Dictionary of model generation parameters including
            tool_choice, temperature, max_tokens, etc.
    """

    def __init__(
        self,
        messages: Optional[List[Message]] = None,
        generation_config: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Initialize ChatHistory.

        Args:
            messages: List of Message instances. If provided without anchors,
                all messages go into the unanchored list.
            generation_config: Dictionary of model generation parameters including
                tool_choice, temperature, max_tokens, etc.
        """
        if messages is not None:
            self._unanchored: List[Message] = list(messages)
        else:
            self._unanchored = []
        self._anchor_groups: Dict[str, List[Message]] = {
            "front": [],
            "back": [],
        }
        self.generation_config = generation_config if generation_config is not None else {}

    @property
    def messages(self) -> List[Message]:
        """All messages in order: front anchors, unanchored, back anchors.

        This is a computed property that returns a new list each time.
        Do not mutate the returned list — use append_message() instead.
        """
        # Front first (ordered), then unanchored, then remaining anchors, then back
        front = self._anchor_groups.get("front", [])
        back = self._anchor_groups.get("back", [])
        others = [
            group for name, group in self._anchor_groups.items()
            if name not in ("front", "back")
        ]
        return front + self._unanchored + others + back

    def append_message(self, message: Message, anchor: Optional[str] = None) -> None:
        """
        Append a message to the chat history.

        Args:
            message: The message to append.
            anchor: Optional anchor name ("front", "back", or custom).
                Messages with anchors persist across compaction.
        """
        if anchor is not None:
            if anchor not in self._anchor_groups:
                self._anchor_groups[anchor] = []
            self._anchor_groups[anchor].append(message)
        else:
            self._unanchored.append(message)

    def set_generation_config(self, key: str, value: Any) -> None:
        """
        Set a generation configuration parameter.

        Args:
            key: Parameter name (e.g., "max_tokens", "temperature", "tool_choice").
            value: Parameter value.
        """
        self.generation_config[key] = value

    def get_generation_config(self, key: str, default: Any = None) -> Any:
        """
        Get a generation configuration parameter.

        Args:
            key: Parameter name.
            default: Default value if key not found.

        Returns:
            Parameter value or default.
        """
        return self.generation_config.get(key, default)

    def __len__(self) -> int:
        """Return the number of messages in the history."""
        return len(self.messages)

    def __iter__(self):
        """Iterate over messages."""
        return iter(self.messages)

    def __repr__(self) -> str:
        return f"ChatHistory(messages={len(self.messages)}, config={self.generation_config})"

    def rolling_window_discard(
        self, trigger_threshold: int, target_tokens: int | None = None
    ) -> tuple[int, int]:
        """Discard oldest unanchored messages when token count exceeds threshold.

        Only discards when total tokens exceed trigger_threshold. When that
        happens, discards down to target_tokens (default: half of threshold).
        This keeps the context size within a stable range and avoids frequent
        KV cache rebuilds.

        Counts tokens for all messages (anchored + unanchored) but only
        discards from the unanchored list. Anchored messages are protected.

        Computes all message token counts once, then subtracts oldest from
        the running total until it fits within the target level.

        Args:
            trigger_threshold: Token count above which discarding is triggered.
            target_tokens: Token count to discard down to. Defaults to
                trigger_threshold // 2.

        Returns:
            A tuple of (messages_removed, remaining_total_tokens).
        """
        if target_tokens is None:
            target_tokens = trigger_threshold // 2

        # Compute token counts once for all messages
        token_counts = {id(msg): msg.count_tokens() for msg in self.messages}
        total_tokens = sum(token_counts.values())

        if total_tokens <= trigger_threshold:
            return (0, total_tokens)

        # Subtract oldest unanchored messages from running total
        removed = 0
        while total_tokens > target_tokens and self._unanchored:
            msg = self._unanchored.pop(0)
            total_tokens -= token_counts[id(msg)]
            removed += 1

        return (removed, total_tokens)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for persistence.

        All messages (anchored and unanchored) are serialized in their
        respective groups. Each message is converted via to_dict().

        Returns:
            Dictionary with 'unanchored', 'anchors', and 'generation_config'.
        """
        return {
            "unanchored": [msg.to_dict() for msg in self._unanchored],
            "anchors": {
                name: [msg.to_dict() for msg in group]
                for name, group in self._anchor_groups.items()
            },
            "generation_config": self.generation_config,
        }

    @classmethod
    def from_dict(cls, data: Union[Dict[str, Any], List[Dict[str, Any]]]) -> "ChatHistory":
        """Reconstruct ChatHistory from dictionary.

        Supports two formats:
        - New format: dict with 'unanchored', 'anchors', 'generation_config'
        - Legacy format: plain list of message dicts (all go to unanchored)

        Args:
            data: Serialized chat history data.

        Returns:
            Reconstructed ChatHistory instance.
        """
        if isinstance(data, list):
            # Legacy format: plain list of messages, all unanchored
            msg_list = []
            for msg_data in data:
                creation_ts = None
                ts = msg_data.get("creation_timestamp")
                if ts:
                    creation_ts = datetime.fromisoformat(ts)
                msg_list.append(
                    Message.from_dict(
                        {"role": msg_data.get("role", "user"), "content": msg_data.get("content", [])},
                        creation_timestamp=creation_ts,
                        message_id=msg_data.get("id"),
                    )
                )
            return cls(messages=msg_list, generation_config={})

        # New format with anchor groups
        generation_config = data.get("generation_config", {})
        chat_history = cls(messages=[], generation_config=generation_config)

        for msg_data in data.get("unanchored", []):
            chat_history._unanchored.append(Message.from_dict(msg_data))

        for anchor_name, msg_list in data.get("anchors", {}).items():
            for msg_data in msg_list:
                chat_history._anchor_groups[anchor_name].append(
                    Message.from_dict(msg_data)
                )

        return chat_history
