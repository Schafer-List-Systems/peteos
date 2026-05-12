"""Message class for chat history."""

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from .contentpart import ContentPart


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
        self.content = content
        self.metadata = metadata or {}
        self.creation_timestamp = creation_timestamp or datetime.now()
        self.id = message_id or str(uuid.uuid4())

    def get_role(self) -> str:
        """Return the message role.

        Can be overridden in derived classes to return a computed or modified role.
        """
        return self._role

    def set_role(self, role: str) -> None:
        """Set the message role."""
        self._role = role


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

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation.

        Returns:
            Dictionary with role, content (as list of dicts), and metadata.
        """
        return {
            "role": self.get_role(),
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
            **{k: v for k, v in data.items() if k not in ("role", "content", "metadata")}
        )
        if "metadata" in data and data["metadata"]:
            msg.metadata.update(data["metadata"])
        return msg

    def __repr__(self) -> str:
        return f"Message(role={self.get_role()!r}, content_count={len(self.content)}, id={self.id!r})"
