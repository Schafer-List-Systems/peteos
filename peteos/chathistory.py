from typing import List

from peteos.message import Message


class ChatHistory:
    """Container for chat messages."""

    def __init__(self):
        """Initialize empty chat history."""
        self.messages: List[Message] = []

    def get_content(self) -> List[dict]:
        """
        Get all message contents.

        Returns:
            List of message content dictionaries.
        """
        return [msg.content for msg in self.messages]

    def append_message(self, message: Message) -> None:
        """
        Append a message to the chat history.

        Args:
            message: The message to append.
        """
        self.messages.append(message)
