from typing import Optional

from peteos.chathistory import ChatHistory
from peteos.message import Message


class ChatBot:
    """A chatbot that communicates with an LLM."""

    def __init__(self, url: str, model: str):
        """
        Initialize ChatBot.

        Args:
            url: The URL of the LLM endpoint.
            model: The model identifier to use.
        """
        self._url = url
        self._model = model

    async def send_message(self, chat_history: ChatHistory) -> Message:
        """
        Send a chat history to the LLM and receive a response message.

        Args:
            chat_history: The ChatHistory to send to the LLM.

        Returns:
            A Message containing the LLM's response.
        """
        pass

    def list_available_models(self) -> list[str]:
        """
        List all available models from the LLM endpoint.

        Returns:
            A list of model identifiers.
        """
        pass

    @property
    def model(self) -> str:
        """Get the current model identifier."""
        return self._model

    @model.setter
    def model(self, value: str) -> None:
        """Set a new model identifier."""
        self._model = value
