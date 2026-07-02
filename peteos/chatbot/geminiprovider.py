"""Google Gemini backend provider."""

from typing import Optional

import httpx

from .backendprovider import BackendProvider
from .httpclient import HTTPClient
from .chatbotconfig import ChatBotConfig
from .chatbot import ChatBot
from .geminichatbot import GeminiChatBot


# Deprecated/retired models that should not be returned by list_models
_GEMINI_BLACKLIST = {
    "antigravity-preview-05-2026",
    "deep-research-max-preview-04-2026",
    "deep-research-preview-04-2026",
    "deep-research-pro-preview-12-2025",
    "gemini-2.5-computer-use-preview-10-2025",
    "gemini-2.0-flash",
    "gemini-2.0-flash-001",
    "gemini-2.0-flash-lite-001",
    "gemini-2.0-flash-lite",
}


class GeminiChatBotProvider(BackendProvider):

    async def list_models(self, url: str, api_key: Optional[str] = None) -> list[str]:
        models_url = f"{url}/v1beta/models?key={api_key}"
        async with httpx.AsyncClient() as client:
            resp = await client.get(models_url)
            resp.raise_for_status()
            data = resp.json()
            # Gemini uses "name" (e.g. "models/gemini-2.5-flash"), not "id"
            # Only return models that support generateContent (what our chatbot needs)
            models = [
                m["name"].removeprefix("models/")
                for m in data.get("models", [])
                if "name" in m
                and "generateContent" in m.get("supportedGenerationMethods", [])
            ]
            models = [m for m in models if m not in _GEMINI_BLACKLIST]
            return models

    def create_chatbot(self, http_client: HTTPClient, config: ChatBotConfig) -> ChatBot:
        return GeminiChatBot(http_client, config)
