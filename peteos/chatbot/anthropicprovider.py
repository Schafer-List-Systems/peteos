"""Anthropic backend provider."""

from typing import Optional

import httpx

from .backendprovider import BackendProvider
from .httpclient import HTTPClient
from .chatbotconfig import ChatBotConfig
from .chatbot import ChatBot
from .anthropicchatbot import AnthropicChatBot


class AnthropicChatBotProvider(BackendProvider):

    async def list_models(self, url: str, api_key: Optional[str] = None) -> list[str]:
        headers: dict = {}
        if api_key:
            headers["x-api-key"] = api_key
            headers["anthropic-version"] = "2023-06-01"
        async with httpx.AsyncClient(headers=headers) as client:
            resp = await client.get(f"{url}/v1/models")
            resp.raise_for_status()
            data = resp.json()
            return [m["id"] for m in data.get("data", []) if "id" in m]

    def create_chatbot(self, http_client: HTTPClient, config: ChatBotConfig) -> ChatBot:
        return AnthropicChatBot(http_client, config)
