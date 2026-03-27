from typing import Dict

from peteos.chatbot import ChatBot
from peteos.session import Session


class Agent:
    """Manages concurrent sessions.

    Each session runs in its own thread. The agent also manages the channels
    to the sessions.
    """

    def __init__(self, chatbot: ChatBot):
        """
        Initialize Agent.

        Args:
            chatbot: The ChatBot instance to use (obligatory).
        """
        self._chatbot = chatbot
        self._sessions: Dict[str, Session] = {}
        self._channels: Dict[str, object] = {}
