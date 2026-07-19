"""Simple mock chatbot for unit testing."""

from typing import AsyncGenerator, Dict, Any, List

from .backendprovider import BackendProvider
from .chatbot import ChatBot
from .chatbotconfig import ChatBotConfig
from .chatbotresponse import ChatBotResponse, GenericChatBotResponse
from peteos.conversation.message import Message


class SimpleMockChatBotResponse(GenericChatBotResponse):
    """ChatBotResponse for the SimpleMockChatBot.

    Wraps a pre-built Message and returns it directly from _build_message().
    """

    def __init__(self, message: Message) -> None:
        self._message = message

        async def _noop_stream() -> AsyncGenerator[str, None]:
            yield "[DONE]"

        super().__init__(_noop_stream(), translations={})

        self._data = {
            "role": message.role,
            "content": [
                _normalize_content_part(part.raw_dict)
                for part in message.content
            ],
        }
        self._message_cache = message
        self._has_text_part = any(part.type == "text" for part in message.content)

    def _build_message(self) -> Message:
        return self._message


def _normalize_content_part(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize ContentPart raw dict to the internal chatbot data format.

    Text blocks use "content" key (not "text") to match the accumulated data
    format used by real chatbot implementations.
    """
    norm = dict(raw)
    if norm.get("type") == "text":
        norm["content"] = norm.pop("text")
    return norm


class SimpleMockChatBot(ChatBot):
    """Mock chatbot that returns a predefined list of messages one by one.

    Used for unit testing. Takes Message objects in the constructor and
    returns them sequentially on each send_context() call. When exhausted,
    returns a response with an "error" key in data.

    No streaming support. model returns "simple-mock".
    """

    def __init__(self, responses: List[Message]) -> None:
        super().__init__(ChatBotConfig(name="simple-mock", url="http://localhost", model="simple-mock"))
        self._responses = list(responses)

    async def send_context(
        self,
        _context: object,
        _generation_config: Dict[str, Any] | None = None,
        _streaming: bool | None = None,
    ) -> ChatBotResponse:
        if self._responses:
            return SimpleMockChatBotResponse(self._responses.pop(0))

        # Exhausted — simulate HTTP 503 Service Unavailable.
        # Real chatbots raise RuntimeError on 5xx errors, so do the same.
        raise RuntimeError("HTTP 503 from http://localhost: unexpected request — no more mock responses")

    def list_available_models(self) -> List[str]:
        return ["simple-mock"]


class SimpleMockBackendProvider(BackendProvider):
    """BackendProvider that creates SimpleMockChatBot instances.

    Constructor takes the list of messages. Each create_chatbot() call
    returns a SimpleMockChatBot pre-loaded with those messages.
    """

    def __init__(self, responses: List[Message]) -> None:
        self._responses = responses

    async def list_models(self, url: str, api_key: str | None = None) -> List[str]:
        return ["simple-mock"]

    def create_chatbot(
        self,
        _http_client: object | None,
        _config: ChatBotConfig,
    ) -> ChatBot:
        return SimpleMockChatBot(self._responses)
