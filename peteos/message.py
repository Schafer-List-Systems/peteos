from datetime import datetime


class Message:
    """A single (immutable) message in chat history."""

    def __init__(self, content: dict, creation_timestamp: datetime = None):
        """
        Initialize a Message.

        Args:
            content: Dictionary containing the message content.
                Example: {"role": "user", "content": "Hello"}
                Or for multi-part: {"role": "assistant", "content": [{"type": "text", "text": "..."}]}
            creation_timestamp: Timestamp of message creation (defaults to now).
        """
        self.content = content
        self.creation_timestamp = creation_timestamp or datetime.now()
