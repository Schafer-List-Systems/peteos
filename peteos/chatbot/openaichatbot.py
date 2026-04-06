"""OpenAI-compatible ChatBot implementation."""

from abc import ABC, abstractmethod
import json
from typing import Dict, Any, List, Optional, AsyncGenerator

from peteos.logger import get_logger
from .httpclient import HTTPClient
from .chatbotresponse import ChatBotResponse, GenericChatBotResponse
from .chathistory import ChatHistory
from .message import Message

_logger = get_logger(__name__)


class OpenAIChatBot(GenericChatBot):
    """ChatBot implementation for OpenAI-compatible API."""

    # Default translation configuration for OpenAI API
    RESPONSE_TRANSLATIONS = {
        # Streaming mode (delta events)
        "choices[*].delta.role": "role",
        "choices[*].delta.content": "text",
        "choices[*].delta.reasoning": "reasoning",
        "choices[*].delta.thinking": "reasoning",
        "choices[*].delta.tool_calls": "tool_calls",
        # Non-streaming mode (message object)
        "choices[*].message.role": "role",
        "choices[*].message.content": "text",
        "choices[*].message.reasoning": "reasoning",
        "choices[*].message.thinking": "reasoning",
        "choices[*].message.tool_calls": "tool_calls",
    }

    REQUEST_TRANSLATIONS = {
        "text": "content",
        "reasoning": "reasoning",
        "tool_calls": "tool_calls",
    }

    def __init__(self, http_client: HTTPClient, model: str, base_url: str):
        """
        Initialize OpenAIChatBot.

        Args:
            http_client: HTTP client for making API requests.
            model: The OpenAI model identifier (e.g., "gpt-4").
            base_url: The OpenAI API base URL.
        """
        super().__init__(
            http_client=http_client,
            model=model,
            base_url=base_url,
            chat_endpoint="/v1/chat/completions",
            models_endpoint="/v1/models",
            response_translations=self.RESPONSE_TRANSLATIONS,
            request_translations=self.REQUEST_TRANSLATIONS,
        )

    def _build_body(self, chat_history: ChatHistory, streaming: bool) -> Dict[str, Any]:
        """Build OpenAI-specific request body.

        OpenAI format:
        - system messages included in messages[] array with role="system"
        - tools from tool messages
        - tool_choice from generation_config
        """
        body = {}
        body["model"] = self._model
        body["stream"] = streaming

        messages = []
        tools = []

        for msg in chat_history.messages:
            role = msg.role

            if role == "tool":
                # Extract tool definitions
                for part in msg.content:
                    if part.type == "tool":
                        tools.append(part.data)
            elif role in ("user", "assistant", "system"):
                # Conversation messages - build message dict from ContentPart fields
                # Use translation table to map uniform keys to API-specific keys
                msg_dict = {"role": role}
                for part in msg.content:
                    # part.data contains fields like "text", "reasoning", etc.
                    # Use translation to map to API-specific key
                    for key, value in part.data.items():
                        if key in self._request_translations:
                            api_key = self._request_translations[key]
                        else:
                            api_key = key
                        msg_dict[api_key] = value
                messages.append(msg_dict)

        body["messages"] = messages

        if tools:
            body["tools"] = [{"type": "function", "function": t} for t in tools]

        # Copy generation config (includes tool_choice)
        for key, value in chat_history.generation_config.items():
            if key not in body:
                body[key] = value

        _logger.debug("OpenAI request body: %s", json.dumps(body, indent=2))
        return body
