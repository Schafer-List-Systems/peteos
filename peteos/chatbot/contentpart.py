"""ContentPart class for multi-modal message content."""

from typing import Any, Dict


class ContentPart:
    """A single part of message content.

    ContentPart represents one element in a message's content array.
    Types include "text", "image", "video", "audio", etc.

    Example:
        >>> text_part = ContentPart(type="text", text="Hello world")
        >>> image_part = ContentPart(type="image", source={"type": "base64", "data": "..."})

    Attributes:
        type: The type of content part (e.g., "text", "image", "video").
        data: Additional data specific to the part type.
    """

    def __init__(self, part_type: str, **data: Any) -> None:
        """
        Initialize a ContentPart.

        Args:
            part_type: The type of content part.
            **data: Additional data for this part type.
        """
        self.type = part_type
        self.data = data

    @property
    def text(self) -> str | None:
        """Get the text content if this is a text part."""
        return self.data.get("text")

    @property
    def source(self) -> Dict[str, Any] | None:
        """Get the source if this is a media part."""
        return self.data.get("source")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation.

        Returns:
            Dictionary with type and data fields.
        """
        return {"type": self.type, **self.data}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ContentPart":
        """Create ContentPart from dictionary.

        Args:
            data: Dictionary with type and data fields.

        Returns:
            ContentPart instance.
        """
        part_type = data.get("type", "text")
        return cls(part_type=part_type, **data)

    def __repr__(self) -> str:
        return f"ContentPart(type={self.type!r}, data={self.data!r})"
