"""OpenAI-compatible ChatBot implementation."""

from abc import ABC, abstractmethod
import json
from typing import Dict, Any, List, Optional, AsyncGenerator

from peteos.logger import get_logger
from .chatbot import GenericChatBot
from .chatbotresponse import ChatBotResponse, GenericChatBotResponse
from .httpclient import HTTPClient
from .chathistory import ChatHistory
from .message import Message

_logger = get_logger(__name__)


class OpenAIChatBot(GenericChatBot):
    """ChatBot implementation for OpenAI-compatible API."""

    # Default translation configuration for OpenAI API
    # Translates OpenAI SSE events to uniform delta format
    # All index fields are preserved for merge_delta_into_target to use
    RESPONSE_TRANSLATIONS = {
        # Streaming mode (delta events)
        "choices[*].delta.role": "role",
        "choices[*].delta.reasoning": "reasoning",
        "choices[*].delta.content": "content",
        "choices[*].delta.finish_reason": "stop_reason",
        # Tool calls: translate individual fields, preserving index
        # OpenAI returns one tool call per event, so we use [0] for extraction
        "choices[*].delta.tool_calls[0].index": "tool_calls[0].index",
        "choices[*].delta.tool_calls[0].type": "tool_calls[0].type",
        "choices[*].delta.tool_calls[0].id": "tool_calls[0].id",
        "choices[*].delta.tool_calls[0].function.name": "tool_calls[0].name",
        "choices[*].delta.tool_calls[0].function.arguments": "tool_calls[0].arguments",
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

    async def send_message(
        self,
        chat_history: ChatHistory,
        streaming: bool = True
    ) -> ChatBotResponse:
        """Send a chat history to the LLM and receive a response."""
        body = self._build_body(chat_history, streaming)

        if streaming:
            stream = self._http_client.stream_post(f"{self._base_url}{self._chat_endpoint}", body)
            return OpenAIChatBotResponse(stream, self._translations)
        else:
            response_data = await self._http_client.post(f"{self._base_url}{self._chat_endpoint}", body)
            return OpenAIChatBotResponse.from_json(response_data, self._translations)

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
            elif role == "tool_result":
                # Tool result messages - build tool_result dict for API
                for part in msg.content:
                    if part.type == "tool_result":
                        tool_name = part.data.get("name", "unknown")
                        tool_content = part.data.get("content", "")
                        success = part.data.get("success", False)
                        msg_dict = {
                            "role": "tool",
                            "name": tool_name,
                            "content": tool_content
                        }
                        messages.append(msg_dict)
                        break
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


class OpenAIChatBotResponse(GenericChatBotResponse):
    """ChatBotResponse for OpenAI-compatible API.

    Uses generic delta translation and merging from parent class.
    No special-case handling needed - index-based merging handles tool_calls automatically.
    """

    pass
