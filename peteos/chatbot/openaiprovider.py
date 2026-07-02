"""OpenAI backend provider."""

import asyncio
from typing import Optional

import httpx

from .backendprovider import BackendProvider
from .httpclient import HTTPClient
from .chatbotconfig import ChatBotConfig
from .chatbot import ChatBot
from .openaichatbot import OpenAIChatBot


class OpenAIChatBotProvider(BackendProvider):

    async def list_models(self, url: str, api_key: Optional[str] = None) -> list[str]:
        models_url = f"{url}/v1/models"
        if api_key:
            models_url += f"?key={api_key}"
        async with httpx.AsyncClient() as client:
            resp = await client.get(models_url)
            resp.raise_for_status()
            data = resp.json()
            return [m["id"] for m in data.get("data", []) if "id" in m]

    def create_chatbot(self, http_client: HTTPClient, config: ChatBotConfig) -> ChatBot:
        return OpenAIChatBot(http_client, config)
