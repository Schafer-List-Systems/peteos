"""Message class for chat history."""

import uuid
import json
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from .contentpart import ContentPart

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from peteos.toolmanager import ToolManager


class Message:
    """A single message in chat history.

    Message represents a complete chat message with role, content, and metadata.
    Content is always a list of ContentPart objects, supporting both simple text
    and multi-modal content (text + images + videos + etc.).

    Example:
        >>> # Simple text message
        >>> msg = Message(role="user", content=[ContentPart(type="text", text="Hello")])
        >>>
        >>> # Multi-part message
        >>> msg = Message(
        ...     role="user",
        ...     content=[
        ...         ContentPart(type="text", text="What's in this image?"),
        ...         ContentPart(type="image", source={"type": "base64", "data": "..."})
        ...     ]
        ... )

    Attributes:
        role: The role of the message sender ("user", "assistant", "system").
        content: List of ContentPart objects representing the message content.
        metadata: Optional dictionary with additional metadata (timestamps, sources, etc.).
        creation_timestamp: When the message was created (defaults to now).
        id: UUID identifying this message (auto-generated if not provided).
    """

    def __init__(
        self,
        role: str,
        content: List[ContentPart],
        metadata: Optional[Dict[str, Any]] = None,
        creation_timestamp: Optional[datetime] = None,
        message_id: Optional[str] = None
    ) -> None:
        """
        Initialize a Message.

        Args:
            role: The role of the message sender.
            content: List of ContentPart objects.
            metadata: Optional metadata dictionary.
            creation_timestamp: Timestamp of message creation (defaults to now).
            message_id: UUID identifying this message (auto-generated if not provided).
        """
        self._role = role
        self._content = content
        self.metadata = metadata or {}
        self.creation_timestamp = creation_timestamp or datetime.now()
        self._id = message_id or str(uuid.uuid4())

    @property
    def content(self) -> List[ContentPart]:
        """Return the list of content parts."""
        return self._content

    def get_role(self) -> str:
        """Return the message role.

        Can be overridden in derived classes to return a computed or modified role.
        """
        return self._role

    def set_role(self, role: str) -> None:
        """Set the message role."""
        self._role = role

    def get_id(self) -> str:
        """Return the message ID."""
        return self._id

    @property
    def text(self) -> str:
        """Get all text content concatenated.

        Returns:
            Concatenated text from all text ContentParts.
        """
        texts = [part.text for part in self.content if part.type == "text" and part.text]
        if not texts:
            return ""
        return " ".join(texts).replace("  ", " ")

    def printable(self) -> str:
        """Return a string suitable for display to a human.

        Subclasses can override to provide custom formatting (e.g.
        multi-modal messages with image/video references).
        """
        return self.text

    def serialize_content(self) -> List[Dict[str, Any]]:
        """Serialize content parts as API-agnostic dicts for message bodies.

        Returns a list of dicts, one per content part, suitable for
        inclusion in chatbot request bodies.  The base implementation
        mirrors ``part.to_dict()`` (``{"type": ..., **data}``).

        Subclasses can override to customise per-message-type formatting
        (e.g. Anthropic may need API-specific field names).
        """
        return [part.to_dict() for part in self.content]

    def _compute_token_count(self, encoding: str = "cl100k_base") -> int:
        """Compute token count from serialized content without caching.

        Subclasses that override serialize_content() automatically get the
        correct count here.

        Args:
            encoding: Tiktoken encoding to use.

        Returns:
            Token count as an integer.
        """
        from peteos.utils.tiktoken import count_tiktoken

        serialized = self.serialize_content()
        full_text = f"{self.get_role()}: {json.dumps(serialized, ensure_ascii=False) if serialized else ''}"
        return count_tiktoken(full_text, encoding)

    def count_tokens(self, encoding: str = "cl100k_base") -> int:
        """Count tokens in this message's content.

        Uses cached token count from metadata if available, otherwise
        computes and caches the result.

        Args:
            encoding: Tiktoken encoding to use.

        Returns:
            Token count as an integer.
        """
        if "token_count" in self.metadata:
            return self.metadata["token_count"]
        token_count = self._compute_token_count(encoding)
        self.metadata["token_count"] = token_count
        return token_count

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation.

        Returns:
            Dictionary with role, content (as list of dicts), and metadata.
        """
        return {
            "role": self.get_role(),
            "id": self.get_id(),
            "content": [part.to_dict() for part in self.content],
            **self.metadata
        }

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        creation_timestamp: Optional[datetime] = None,
        message_id: Optional[str] = None
    ) -> "Message":
        """Create Message from dictionary.

        Args:
            data: Dictionary with role and content fields.
            creation_timestamp: Timestamp of message creation.
            message_id: UUID for the message.

        Returns:
            Message instance.

        Raises:
            ValueError: If role is not provided in data.
        """
        if "role" not in data:
            raise ValueError("Message dictionary must contain a 'role' field")
        role = data["role"]
        content_data = data.get("content", [])

        # Handle both list and string content for flexibility
        if isinstance(content_data, str):
            content = [ContentPart(part_type="text", text=content_data)]
        else:
            content = [ContentPart.from_dict(part) for part in content_data]

        msg = cls(
            role=role,
            content=content,
            creation_timestamp=creation_timestamp,
            message_id=message_id,
            **{k: v for k, v in data.items() if k not in ("role", "id", "content", "metadata")}
        )
        if "id" in data and data["id"] and not message_id:
            msg._id = data["id"]
        if "metadata" in data and data["metadata"]:
            msg.metadata.update(data["metadata"])
        return msg

    def __repr__(self) -> str:
        return f"Message(role={self.get_role()!r}, content_count={len(self.content)}, id={self.get_id()!r})"


class SystemPromptMessage(Message):
    """A system message whose content is dynamically assembled from hooks.

    Each hook is a callable returning a text string.  When
    ``serialize_content()`` is called the hooks are invoked in order,
    their return values are joined, and a single text content block is
    returned.

    Subclasses can add/remove hooks between serialisations to produce
    dynamic content (e.g. an awake-status fragment).

    Example:
        msg = SystemPromptMessage()
        msg.add_hook(lambda: "You are helpful.")
        msg.add_hook(lambda: f"Status: {'awake' if check_awake() else 'asleep'}")
        msg.serialize_content()  # -> [{"type": "text", "text": "You are helpful.\\nStatus: asleep"}]
    """

    def __init__(
        self,
        hooks: Optional[List[Callable[[], str]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        creation_timestamp: Optional[datetime] = None,
        message_id: Optional[str] = None,
    ) -> None:
        # content must be non-empty for the parent constructor;
        # the actual text comes from hooks at serialisation time.
        super().__init__(
            role="system",
            content=[ContentPart(part_type="text", text="")],
            metadata=metadata,
            creation_timestamp=creation_timestamp,
            message_id=message_id,
        )
        self._hooks: List[Callable[[], str]] = hooks if hooks is not None else []

    def add_hook(self, hook: Callable[[], str]) -> None:
        """Add a hook that returns a text string for the system prompt."""
        self._hooks.append(hook)

    def serialize_content(self) -> List[Dict[str, Any]]:
        prompt = "\n".join(hook() for hook in self._hooks)
        return [{"type": "text", "text": prompt}]

    def count_tokens(self, encoding: str = "cl100k_base") -> int:
        """Count tokens, always recomputing since content is dynamic."""
        return self._compute_token_count(encoding)


class ToolDefinitionsMessage(Message):
    """A message whose content is dynamically generated from a ToolManager.

    Like ``SystemPromptMessage``, the content is assembled at serialization
    time rather than stored statically.  This allows tools to be registered
    or deregistered on the ``ToolManager`` between calls and have the chat
    history reflect the current set automatically.

    Attributes:
        tool_manager: The ToolManager whose current tool list is queried
            each time ``content`` is accessed or ``serialize_content()``
            is called.
    """

    def __init__(
        self,
        tool_manager: "ToolManager",
        metadata: Optional[Dict[str, Any]] = None,
        creation_timestamp: Optional[datetime] = None,
        message_id: Optional[str] = None,
    ) -> None:
        # Set _content directly to bypass the property in the parent __init__
        object.__setattr__(self, "_content", [])
        super().__init__(
            role="tool",
            content=[],
            metadata=metadata,
            creation_timestamp=creation_timestamp,
            message_id=message_id,
        )
        self._tool_manager = tool_manager

    def get_role(self) -> str:
        """Always returns 'tool'."""
        return "tool"

    @property
    def content(self) -> List[ContentPart]:
        """Return ContentParts for every tool currently registered."""
        parts: List[ContentPart] = []
        for tool in self._tool_manager.get_tool_list():
            parts.append(ContentPart(
                part_type="tool",
                name=tool.name,
                description=tool.description,
                parameters=tool.parameters,
            ))
        return parts

    def serialize_content(self) -> List[Dict[str, Any]]:
        """Serialize tool definitions for API request bodies."""
        return [part.to_dict() for part in self.content]

    def count_tokens(self, encoding: str = "cl100k_base") -> int:
        """Count tokens, always recomputing since content is dynamic."""
        return self._compute_token_count(encoding)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary.  Content is kept empty because it is
        dynamically regenerated at load time from the ToolManager."""
        d = {
            "role": self.get_role(),
            "id": self.get_id(),
            "content": [],
            **self.metadata
        }
        d["type"] = "tool_definitions"
        return d
