"""Anthropic-compatible ChatBot implementation."""

import json
from typing import Any, Dict, AsyncGenerator

from peteos.logger import get_logger
from .chatbot import GenericChatBot
from .chatbotresponse import ChatBotResponse, GenericChatBotResponse
from .chathistory import ChatHistory
from peteos.utils.delta_merge import merge_delta_into_target as _merge_delta_into_target
from .httpclient import HTTPClient
from .message import Message

_logger = get_logger(__name__)


class AnthropicChatBot(GenericChatBot):
    """ChatBot implementation for Anthropic-compatible API."""

    # Default translation configuration for Anthropic API
    # Based on Qwen's Anthropic-compatible endpoint structure:
    # - content_block_start: {"type":"content_block_start","content_block":{"type":"thinking"},"index":0}
    # - content_block_delta: {"type":"content_block_delta","delta":{"type":"thinking_delta","thinking":"..."},"index":0}
    # - index is top-level metadata (not in arrays), so it's ignored during translation
    RESPONSE_TRANSLATIONS = {
        # stream=True entries (streaming mode)
        "message_start.message.role": "role",              # role in message_start event
        "delta.thinking": "reasoning",  # thinking chunks (type discriminator used)
        "delta.text": "text",  # text chunks (type discriminator used)
        "delta.partial_json": "tool_arguments",  # tool JSON args
        "content_block_start.content_block.type": "tool_type",  # tool block start
        "content_block_start.content_block.text": "text",  # text block start
        "content_block_start.content_block.reasoning": "reasoning",  # thinking block start (reasoning field)
        "content_block_start.content_block.thinking": "reasoning",  # thinking block start (thinking field)
        # stream=False entries (non-streaming mode)
        "role": "role",                                    # role at top level
        "content[*].text": "text",                         # content array at top level
        "content[*].thinking": "reasoning",                # content array with thinking
        # Message delta (final stop reason)
        "message_delta.delta.stop_reason": "stop_reason",
    }

    REQUEST_TRANSLATIONS = {
        "text": "content",
        "reasoning": "reasoning",
        "tool_calls": "tool_calls",
    }

    def __init__(self, http_client: HTTPClient, model: str, base_url: str, max_tokens: int = 4096):
        """
        Initialize AnthropicChatBot.

        Args:
            http_client: HTTP client for making API requests.
            model: The Anthropic model identifier (e.g., "claude-3-opus-20240229").
            base_url: The Anthropic API base URL.
            max_tokens: Maximum tokens to generate (default: 4096).
        """
        super().__init__(
            http_client=http_client,
            model=model,
            base_url=base_url,
            chat_endpoint="/v1/messages",
            models_endpoint="/v1/models",
            response_translations=self.RESPONSE_TRANSLATIONS,
            request_translations=self.REQUEST_TRANSLATIONS,
        )
        self._max_tokens = max_tokens

    async def send_message(
        self,
        chat_history: ChatHistory,
        streaming: bool = True,
        **kwargs
    ) -> ChatBotResponse:
        """Send a chat history to Anthropic-compatible API."""
        body = self._build_body(chat_history, streaming)
        body.update(kwargs)

        if streaming:
            stream = self._http_client.stream_post(f"{self._base_url}{self._chat_endpoint}", body)
            return AnthropicChatBotResponse(stream, self._translations)
        else:
            response_data = await self._http_client.post(f"{self._base_url}{self._chat_endpoint}", body)
            return AnthropicChatBotResponse.from_json(response_data, self._translations)

    def _build_body(self, chat_history: ChatHistory, streaming: bool) -> Dict[str, Any]:
        """Build Anthropic-specific request body.

        Anthropic format:
        - system goes to separate 'system' field
        - messages[] only contains user/assistant
        - tools use 'input_schema' instead of 'parameters'
        - tool_choice in Anthropic format
        """
        body = {}
        body["model"] = self._model
        body["stream"] = streaming
        body["max_tokens"] = self._max_tokens

        messages = []
        system_parts = []
        tools = []

        for msg in chat_history.messages:
            role = msg.role

            if role == "system":
                # Collect system parts
                system_parts.extend(msg.content)
            elif role == "tool":
                # Extract tool definitions, convert to input_schema
                for part in msg.content:
                    if part.type == "tool":
                        tool_def = dict(part.data)
                        # Convert parameters to input_schema
                        if "parameters" in tool_def:
                            tool_def["input_schema"] = tool_def.pop("parameters")
                        tools.append(tool_def)
            elif role in ("user", "assistant"):
                # Conversation messages
                content = [part.to_dict() for part in msg.content]
                messages.append({
                    "role": role,
                    "content": content
                })

        body["messages"] = messages

        if system_parts:
            # Convert system parts to string
            system_text = " ".join(
                part.text for part in system_parts
                if part.type == "text" and part.text
            )
            body["system"] = system_text

        if tools:
            body["tools"] = tools

        # Copy generation config (excluding tool_choice for now)
        for key, value in chat_history.generation_config.items():
            if key not in body and key != "tool_choice":
                body[key] = value

        # Add tool_choice if present
        tool_choice = chat_history.generation_config.get("tool_choice")
        if tool_choice:
            body["tool_choice"] = tool_choice

        return body


class AnthropicChatBotResponse(GenericChatBotResponse):
    """ChatBotResponse for Anthropic-compatible API.

    Uses generic delta translation and merging from parent class.
    Anthropic's index field is top-level metadata (not in arrays), so it's
    simply ignored during translation.
    """

    def _process_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process an Anthropic event, defaulting role to 'assistant' if missing from message_start.

        The Anthropic API includes role in message_start. This fallback handles
        non-compliant backends that omit the role field entirely.
        """
        if event.get("type") == "message_start" and "role" not in event.get("message", {}):
            _merge_delta_into_target(self._data, {"role": "assistant"})

        return self._translate_event(event)
