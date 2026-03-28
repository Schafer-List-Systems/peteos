# ChatBot Architecture

## Overview

The ChatBot classes provide a unified interface for communicating with different LLM APIs. The architecture separates:

1. **ChatBot** - Sends messages to the API
2. **ChatBotResponse** - Streams and normalizes API responses

This design supports multiple API protocols (OpenAI-compatible, Anthropic-compatible) while presenting a common interface to the rest of the framework.

## Key Design Decisions

### API Protocol vs. Model

The class names refer to the **API protocol**, not the LLM model:

- `OpenAIChatBot` - OpenAI-compatible API (uses `reasoning` key for thinking content)
- `AnthropicChatBot` - Anthropic-compatible API (uses `thinking` or `reasoning` keys)
- `GenericChatBot` - Configurable bot for any API protocol

The model (e.g., `qwen3.5-35b`, `gpt-4`, `claude-3`) is a parameter passed to the API. Any model can run on any API protocol as long as it speaks that protocol.

### Streaming-First Design

All responses stream via Server-Sent Events (SSE). Non-streaming responses are converted to a single-event stream for consistent handling.

### Configurable Translation

The `GenericChatBot` and `GenericChatBotResponse` classes use path-based translation configuration:

```python
GenericChatBot(
    http_client,
    model="my-model",
    chat_endpoint="/v1/chat/completions",
    models_endpoint="/v1/models",
    response_translations={
        "choices[*].delta.content": "text_content",
        "choices[*].delta.reasoning": "thinking_content"
    }
)
```

Path notation supports:
- Simple keys: `"field"` -> `data["field"]`
- Array indexing: `"choices[0]"` -> `data["choices"][0]`
- Wildcards: `"choices[*].delta.content"` -> iterate all choices
- Type discriminators: `"content_block_start.content_block.text"` -> matches `type` field, then navigates nested structure

## Components

### ChatBot (Abstract Base Class)

```python
class ChatBot(ABC):
    def __init__(self, http_client: HTTPClient, model: str)
    async def send_message(chat_history: ChatHistory, streaming: bool=True) -> ChatBotResponse
    def list_available_models() -> List[str]
```

**Responsibilities:**
- Manage HTTP connection to API
- Build API-specific request bodies
- Return streaming responses wrapped in `ChatBotResponse`

### GenericChatBot

Configurable base class that handles request building and response wrapping:

**Constructor Parameters:**
- `http_client` - HTTP client for API requests
- `model` - Model identifier to use
- `base_url` - API base URL (e.g., `"http://localhost:8000"`)
- `chat_endpoint` - Chat endpoint (default: `"/v1/chat/completions"`)
- `models_endpoint` - Models listing endpoint (default: `"/v1/models"`)
- `response_translations` - Dict mapping source path -> target field
- `**defaults` - Additional request body parameters (e.g., `max_tokens=4096`)

**Request Building:**
- Builds messages array from `ChatHistory`
- Extracts system messages separately (for APIs that require it)
- Merges with `defaults` and streaming flag

### OpenAIChatBot

**API Endpoint:** `/v1/chat/completions`

**Request Format:**
```json
{
    "model": "qwen3.5-35b",
    "messages": [{"role": "user", "content": "..."}],
    "stream": true
}
```

**Response Format:**
```json
{"choices": [{"delta": {"content": "...", "reasoning": "..."}}]}
```

**Translation Config:**
```python
{
    "choices[*].delta.content": "text_content",
    "choices[*].delta.reasoning": "thinking_content",
    "choices[*].delta.thinking": "thinking_content"
}
```

**Key Behavior:**
- Maps `reasoning` key to `thinking_content` property
- Supports incremental reasoning streaming

### AnthropicChatBot

**API Endpoint:** `/v1/messages`

**Request Format:**
```json
{
    "model": "claude-3-opus",
    "messages": [{"role": "user", "content": "..."}],
    "system": "optional system prompt",
    "stream": true,
    "max_tokens": 4096
}
```

**Response Format:**
```json
{"type": "content_block_delta", "delta": {"type": "text_delta", "text": "..."}}
{"type": "content_block_delta", "delta": {"type": "thinking_delta", "thinking": "..."}}
```

**Translation Config:**
```python
{
    "content_block_delta.delta.text": "text_content",
    "content_block_delta.delta.reasoning": "thinking_content",
    "content_block_delta.delta.thinking": "thinking_content",
    "content_block_start.content_block.text": "text_content",
    "content_block_start.content_block.reasoning": "thinking_content",
    "message_start.message.content[*].text": "text_content"
}
```

**Key Behavior:**
- Separates system messages from user messages
- Supports `thinking_delta` and `reasoning_delta` keys
- Handles `message_start`, `content_block_start`, `content_block_delta`, `content_block_stop` events

### ChatBotResponse (Abstract Base Class)

```python
class ChatBotResponse:
    def __aiter__() -> self
    async def __anext__() -> str  # accumulated text chunk
    @property
    def text_content() -> str  # final accumulated response
    @property
    def thinking_content() -> str  # final accumulated reasoning
```

**Responsibilities:**
- Parse SSE stream line-by-line
- Translate API-specific events to common schema
- Accumulate content in separate fields based on key naming
- Yield accumulated text on each iteration

### GenericChatBotResponse

Configurable response class that translates events using path-based lookups:

**Constructor:**
```python
GenericChatBotResponse(stream, translations)
```

Where `translations` is a Dict mapping source path -> target field. Target fields ending in `_thinking` are accumulated into `thinking_content`, all others into `text_content`.

### OpenAIChatBotResponse

Uses standard OpenAI translation config (inherits from `GenericChatBotResponse`).

### AnthropicChatBotResponse

Uses standard Anthropic translation config (inherits from `GenericChatBotResponse`).

## Streaming Behavior

### How It Works

1. `ChatBot.send_message()` calls `HTTPClient.stream_post()` which yields raw SSE lines
2. `ChatBotResponse.__anext__()` processes each SSE line:
   - Parses `data: {...}` JSON
   - Calls `_translate_event()` to normalize to common schema
   - Accumulates content into appropriate fields
   - Returns accumulated text (includes all previous chunks)

### Yield Behavior

Each `__anext__()` call returns the **accumulated** text, not just the new chunk:

```python
async for chunk in response:
    # chunk contains all text accumulated so far
    print(chunk)  # "Hello", "Hello World", "Hello World!"
```

### Non-Streaming Mode

When `streaming=False`:
1. `ChatBot` makes non-streaming HTTP request
2. Response is wrapped in a generator that yields a single SSE event
3. Response class behaves identically to streaming mode

## Reasoning/Thinking Support

Different APIs expose reasoning differently:

| API Type | Key Name | Streaming Behavior |
|----------|----------|-------------------|
| OpenAI-compatible | `reasoning` | Incremental (like content) |
| Anthropic-compatible | `thinking` | Incremental (`thinking_delta`) |
| Anthropic-compatible | `reasoning` | Incremental (`reasoning_delta`) |

All are normalized to `response.thinking_content`.

## Testing

### Mock Server

Tests use `tests/http/mock_server.py` which provides:
- `create_openai_mock_server(reasoning, response, port)`
- `create_anthropic_mock_server(thinking, response, port)`

### Test Coverage

- **Unit tests** (`tests/test_chatbot_response.py`): Response parsing logic
- **E2E tests** (`tests/e2e/test_chatbot_integration.py`): Full integration with mock servers
- Both streaming and non-streaming modes tested

## Usage

### Environment Configuration

Copy `.env.example` to `.env` and configure your API endpoints:

```bash
cp .env.example .env
```

Edit `.env`:
```
OPENAI_COMPATIBLE_BASE_URL=http://192.168.255.10:8123
OPENAI_COMPATIBLE_MODEL=qwen/qwen3.5-35b-a3b
```

### Running Examples

```bash
# OpenAI-compatible API (streaming)
python examples/openai_chatbot.py

# Anthropic-compatible API (streaming)
python examples/anthropic_chatbot.py
```

Or with explicit environment variables:

```bash
OPENAI_COMPATIBLE_BASE_URL=http://192.168.255.10:8123 \
OPENAI_COMPATIBLE_MODEL=qwen/qwen3.5-35b-a3b \
python examples/openai_chatbot.py
```

### Streaming Mode

By default, examples run in streaming mode. To disable streaming:

```bash
# Non-streaming mode
USE_STREAMING=false python examples/openai_chatbot.py

# Non-streaming with Anthropic API
USE_STREAMING=false python examples/anthropic_chatbot.py
```

The `USE_STREAMING` environment variable controls the `streaming` parameter passed to `send_message()`.

### Programmatic Usage

```python
from peteos.chatbot import OpenAIChatBot, AnthropicChatBot, GenericChatBot
from peteos.httpclient import HTTPClient
from peteos.chathistory import ChatHistory
from peteos.message import Message

# OpenAI-compatible
async def main():
    http_client = HTTPClient(timeout=60.0)
    chatbot = OpenAIChatBot(
        http_client=http_client,
        model="qwen3.5-35b",
        base_url="http://192.168.255.10:8123"
    )

    history = ChatHistory()
    history.append_message(Message(content={"role": "user", "content": "Hello!"}))

    response = await chatbot.send_message(history, streaming=True)

    async for chunk in response:
        print(chunk, end="", flush=True)

    print("\nThinking:", response.thinking_content)
```

See `examples/openai_chatbot.py` and `examples/anthropic_chatbot.py` for complete working examples.