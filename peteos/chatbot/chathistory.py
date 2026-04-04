"""ChatHistory class for managing chat messages and request fields."""

from typing import Any, Dict, List, Optional

from .message import Message
from .contentpart import ContentPart


class ChatHistory:
    """Container for chat messages and request configuration.

    ChatHistory represents the complete prompt being sent to the chatbot.
    All data is stored as messages with roles. Special content types
    (tools, system prompts, etc.) are represented as messages with
    specific roles. Additional request fields (generation config) are stored
    in a single config dictionary.

    Example:
        >>> history = ChatHistory(
        ...     messages=[
        ...         Message(role="system", content=[ContentPart(type="text", text="You are helpful")]),
        ...         Message(role="tool", content=[ContentPart(type="tool", name="get_weather", ...)]),
        ...         Message(role="user", content=[ContentPart(type="text", text="Hello")])
        ...     ],
        ...     generation_config={
        ...         "max_tokens": 4096,
        ...         "tool_choice": {"type": "auto"}
        ...     }
        ... )

    Attributes:
        messages: List of Message instances with various roles.
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
            messages: List of Message instances. System messages should be first.
            generation_config: Dictionary of model generation parameters including
                tool_choice, temperature, max_tokens, etc.
        """
        self.messages = messages if messages is not None else []
        self.generation_config = generation_config if generation_config is not None else {}

    def append_message(self, message: Message) -> None:
        """
        Append a message to the chat history.

        Args:
            message: The message to append.
        """
        self.messages.append(message)

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
