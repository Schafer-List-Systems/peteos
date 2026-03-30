# ChatBot Architecture

## Overview

The ChatBot classes provide a unified interface for communicating with different LLM APIs. The architecture separates:

1. **ChatBot** - Sends messages to the API
2. **ChatBotResponse** - Streams and normalizes API responses

This design supports multiple API protocols (OpenAI-compatible, Anthropic-compatible) while presenting a common interface to the rest of the framework.

## Key Design Decisions

### Generic Approach

The architecture uses a single configurable `GenericChatBot` base class with configurable endpoints and response translations. `OpenAIChatBot` and `AnthropicChatBot` are thin wrappers that pre-configure this generic class for their respective APIs.

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
        "choices[*].delta.content": "text",
        "choices[*].delta.reasoning": "reasoning"
    },
    max_tokens=4096  # Additional request parameters
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
- `request_translations` - Dict mapping uniform keys -> API-specific keys
- `**defaults` - Additional request body parameters (e.g., `max_tokens=4096`)

**Request Building:**
- Builds messages array from `ChatHistory`
- Extracts system messages separately (for APIs that require it)
- Merges with `defaults` and streaming flag
- Translates message content keys using `request_translations`

**Message Translation:**
- Keys in `request_translations` are translated (e.g., `"text"` -> `"content"`)
- Keys not in the table are forwarded as-is

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
    "choices[*].delta.role": "role",
    "choices[*].message.content": "text",
    "choices[*].message.reasoning": "reasoning",
    "choices[*].message.thinking": "reasoning",
    "choices[*].delta.content": "text",
    "choices[*].delta.reasoning": "reasoning",
    "choices[*].delta.thinking": "reasoning",
    "choices[*].delta.tool_calls": "tool_calls",
    "choices[*].message.tool_calls": "tool_calls",
}
```

**Request Translation:**
```python
{
    "text": "content",
    "reasoning": "reasoning",
    "tool_calls": "tool_calls",
}
```

**Key Behavior:**
- Maps `reasoning`/`thinking` key to `reasoning` field
- Supports incremental reasoning streaming
- Supports tool_calls in delta and message formats

### AnthropicChatBot

**API Endpoint:** `/v1/messages`

**Constructor Parameters:**
- `http_client` - HTTP client for API requests
- `model` - Model identifier to use
- `base_url` - API base URL
- `max_tokens` - Maximum tokens to generate (default: 4096)
- Additional parameters via `**kwargs` in `send_message()`

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

**Response Format (Streaming):**
```json
{"type": "message_start", "message": {"role": "assistant", ...}}
{"type": "content_block_start", "content_block": {"type": "text", "text": "..."}}
{"type": "content_block_delta", "delta": {"type": "text_delta", "text": "..."}}
{"type": "content_block_stop"}
```

**Response Format (Non-Streaming):**
```json
{"role": "assistant", "content": [{"type": "text", "text": "..."}], ...}
```

**Translation Config (Streaming):**
```python
{
    "message_start.message.role": "role",              # role in message_start event
    "content_block_delta.delta.text": "text",          # text chunks
    "content_block_delta.delta.thinking": "reasoning", # reasoning chunks
    "content_block_delta.delta.reasoning": "reasoning",
    "content_block_start.content_block.text": "text",  # text block start
    "content_block_start.content_block.thinking": "reasoning",  # thinking block start
    "content_block_start.content_block.reasoning": "reasoning",  # reasoning block start
}
```

**Translation Config (Non-Streaming):**
```python
{
    "role": "role",                                    # role at top level
    "content[*].text": "text",                         # content array at top level
    "content[*].thinking": "reasoning",                # content array with thinking
}
```

**Request Translation:**
```python
{
    "text": "content",
    "reasoning": "reasoning",
    "tool_calls": "tool_calls",
}
```

**Key Behavior:**
- Separates system messages from user messages
- Supports `thinking_delta` and `reasoning_delta` keys
- Handles `message_start`, `content_block_start`, `content_block_delta`, `content_block_stop` events
- Adds `max_tokens` to every request
- Accepts additional `**kwargs` in `send_message()` that merge into request body
- Defaults role to 'assistant' if missing from `message_start` event (handles non-compliant backends)

### ChatBotResponse (Generic Class)

```python
class ChatBotResponse:
    def __aiter__() -> AsyncIterator[tuple[str, Any]]
    async def __anext__() -> tuple[str, Any]  # (key, chunk) pair
    @property
    def data() -> Dict[str, Any]  # all accumulated fields

class GenericChatBotResponse(ChatBotResponse):
    def __init__(stream, translations: Dict[str, str])
    async def _translate_event(event) -> Dict[str, Any]
    def _accumulate_event(event) -> None
    @classmethod
    def from_json(data: dict, translations: Dict[str, str]) -> "GenericChatBotResponse"
```

**Responsibilities:**
- Parse SSE stream line-by-line
- Translate API-specific events to common schema using path-based translations
- Accumulate all fields into `response.data` dict
- Yield **(key, chunk)** tuples for each field update
  - `("text", "Hello")`, `("reasoning", "Thinking...")`
- Supports multiple fields per event (e.g., `message_start` with both text and reasoning)

### GenericChatBotResponse

Configurable response class that translates events using path-based lookups:

**Constructor:**
```python
GenericChatBotResponse(stream, translations: Dict[str, str])
```

Where `translations` maps source path -> target field. All accumulated fields are accessible via `response.data`:
- `response.data["text"]`
- `response.data["reasoning"]`
- `response.data["tool_calls"]`
- `response.data["role"]`

**Async Iteration:**
```python
async for key, chunk in response:
    # key: "text", "reasoning", etc.
    # chunk: actual delta (not accumulated)
    print(f"{key}: {chunk}")
```

**Example:**
```python
response = OpenAIChatBotResponse(stream)
async for key, chunk in response:
    if key == "text":
        yield chunk  # stream text to user
# After iteration:
response.data["text"]  # accumulated full response
response.data["reasoning"]  # accumulated reasoning
```

**Non-Streaming Mode:**
```python
# Wrap JSON response as SSE stream
response = GenericChatBotResponse.from_json(
    {"choices": [{"delta": {"content": "Hello"}}]},
    translations
)
# Yields all fields in a single SSE event
# Same iteration behavior as streaming mode
```

### AnthropicChatBotResponse

Uses standard Anthropic translation config (inherits from `GenericChatBotResponse`).

**Override:** `_process_event()` defaults role to 'assistant' if missing from `message_start` event, handling non-compliant backends that omit the role field.

## Streaming Behavior

### How It Works

1. `ChatBot.send_message()` calls `HTTPClient.stream_post()` which yields raw SSE lines
2. `GenericChatBotResponse._event_generator()` processes each SSE line:
   - Parses `data: {...}` JSON
   - Calls `_translate_event()` to normalize to common schema
   - Accumulates all fields into `response.data` dict
   - Yields **(key, chunk)** tuple for each field update

### Yield Behavior

Each iteration yields a **(key, chunk)** pair:

```python
# Simple streaming
async for key, chunk in response:
    if key == "text":
        print(chunk, end="")  # "H", "e", "l", "l", "o"

# Multi-field event
data: {"type": "message_start", "message": {"content": [...], "reasoning": "R"}}
# Yields: ("text", "..."), ("reasoning", "R")
```

**Key Points:**
- `chunk` is the **actual delta**, not accumulated
- `response.data[key]` contains the **accumulated** value
- Multiple fields from same event are yielded sequentially

### Non-Streaming Mode

When `streaming=False`:
1. `ChatBot` makes non-streaming HTTP request
2. Response is wrapped via `GenericChatBotResponse.from_json()`
3. Yields all fields in a single SSE event
4. Same iteration behavior as streaming mode

## Reasoning/Thinking Support

Different APIs expose reasoning differently:

| API Type | Key Name | Streaming Behavior |
|----------|----------|-------------------|
| OpenAI-compatible | `reasoning` | Incremental (like content) |
| OpenAI-compatible | `thinking` | Incremental (like content) |
| Anthropic-compatible | `thinking` | Incremental (`thinking_delta`) |
| Anthropic-compatible | `reasoning` | Incremental (`reasoning_delta`) |

All are normalized to `response.data["reasoning"]`.

## Testing

### Mock Server

Tests use `tests/http/mock_server.py` which provides:
- `create_openai_mock_server(reasoning, response, port)`
- `create_anthropic_mock_server(thinking, response, port)`

### Test Coverage

- **Unit tests** (`tests/test_chatbot_response.py`): Response parsing logic, path translation, wildcard support
- **E2E tests** (`tests/e2e/test_chatbot_integration.py`): Full integration with mock servers
- Both streaming and non-streaming modes tested

Run tests:
```bash
# Unit tests
pytest tests/test_chatbot_response.py -v

# E2E tests
pytest tests/e2e/test_chatbot_integration.py -v
```

## Usage

### Environment Configuration

Copy `.env.example` to `.env` and configure your API endpoints:

```bash
cp .env.example .env
```

**Note:** The `.env.example` file defines `OPENAI_COMPATIBLE_*` and `ANTHROPIC_COMPATIBLE_*` variables, but actual examples use `BASE_URL`, `MODEL`, `CHAT_PROTOCOL`, and `USE_STREAMING`. Update your `.env` accordingly:

```
BASE_URL=http://localhost:8000
MODEL=qwen/qwen3.5-35b-a3b
CHAT_PROTOCOL=anthropic  # or "openai"
USE_STREAMING=true       # or "false"
```

### Running Examples

```bash
# OpenAI-compatible API (streaming)
BASE_URL=http://localhost:8000 \
MODEL=qwen/qwen3.5-35b-a3b \
CHAT_PROTOCOL=openai \
python examples/chatbot.py

# Anthropic-compatible API (streaming)
BASE_URL=http://localhost:8000 \
MODEL=qwen/qwen3.5-35b-a3b \
CHAT_PROTOCOL=anthropic \
python examples/chatbot.py
```

### Streaming Mode

By default, examples run in streaming mode. To disable streaming:

```bash
# Non-streaming mode
BASE_URL=http://localhost:8000 \
MODEL=qwen/qwen3.5-35b-a3b \
CHAT_PROTOCOL=anthropic \
USE_STREAMING=false \
python examples/chatbot.py
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
        base_url="http://localhost:8000"
    )

    history = ChatHistory()
    history.append_message(Message(content={"role": "user", "content": "Hello!"}))

    response = await chatbot.send_message(history, streaming=True)

    # Stream text content to user
    async for key, chunk in response:
        if key == "text":
            print(chunk, end="", flush=True)

    # Access accumulated values after iteration
    print("\nReasoning:", response.data.get("reasoning", ""))
    print("Full text:", response.data.get("text", ""))

# Anthropic-compatible
async def main():
    http_client = HTTPClient(timeout=60.0)
    chatbot = AnthropicChatBot(
        http_client=http_client,
        model="claude-3-opus",
        base_url="http://localhost:8000",
        max_tokens=4096
    )

    history = ChatHistory()
    history.append_message(Message(content={"role": "user", "content": "Hello!"}))

    response = await chatbot.send_message(history, streaming=True)

    # Stream text content to user
    async for key, chunk in response:
        if key == "text":
            print(chunk, end="", flush=True)

    # Access accumulated values after iteration
    print("\nReasoning:", response.data.get("reasoning", ""))
    print("Full text:", response.data.get("text", ""))
```

See `examples/chatbot.py` and `examples/chatbot_repl.py` for complete working examples.

## ChatBotManager

The `ChatBotManager` provides a centralized registry for managing multiple LLM backend providers. It auto-detects API types and discovers models at startup.

### Features

- **Auto-detection**: Probes `/v1/models` endpoint to detect OpenAI vs Anthropic format
- **Model discovery**: Lists all available models per backend and creates ChatBot instances
- **Regex filtering**: Query ChatBots by model name pattern
- **Configuration persistence**: Load/restore backend configs from JSON

### Usage

```python
import asyncio
from peteos.chatbotmanager import ChatBotManager

async def main():
    manager = ChatBotManager()

    # Add a backend - auto-detects API type and models
    backend = await manager.add_backend("local-llm", "http://localhost:8000")
    print(f"Detected API: {backend.api_type}")  # "openai" or "anthropic"
    print(f"Models: {list(backend.models.keys())}")

    # List all ChatBots matching a pattern
    for model_id, chatbot in manager.list_chatbots("qwen.*"):
        print(f"  {model_id} -> {chatbot.__class__.__name__}")

    # Remove a backend
    manager.remove_backend("local-llm")

    # Load from JSON (API type and models auto-detected)
    await manager.load_from_json({
        "backends": [
            {"name": "backend1", "url": "http://backend1:8000"},
            {"name": "backend2", "url": "http://backend2:8000"}
        ]
    })

    # Load from file (async)
    await manager.load_from_file("backends.json")

asyncio.run(main())
```

### JSON Configuration

Save backends to a JSON file for later restoration:

```json
{
  "backends": [
    {
      "name": "local-llm",
      "url": "http://localhost:8000"
    },
    {
      "name": "remote-llm",
      "url": "https://api.example.com"
    }
  ]
}
```

Load from file:

```python
manager.load_from_file("backends.json")
```

**Note**: The JSON only contains `name` and `url`. The `api_type` and `models` are auto-detected at runtime by probing each backend's `/v1/models` endpoint.

### API Reference

```python
class ChatBotManager:
    def __init__()

    async def add_backend(name: str, url: str) -> BackendInfo
        """Add backend, detect API type, discover models."""

    def remove_backend(name: str) -> bool
        """Remove backend by name."""

    def list_chatbots(model_regex: str) -> List[Tuple[str, ChatBot]]
        """List all ChatBots matching regex pattern."""

    async def load_from_json(json_obj: dict) -> None
        """Load backends from JSON object (API auto-detected). Clears current state first."""

    async def load_from_file(filepath: str) -> None
        """Load backends from JSON file (async)."""
```

### BackendInfo

```python
@dataclass
class BackendInfo:
    name: str
    url: str
    api_type: str  # "openai" or "anthropic"
    models: Dict[str, ChatBot]
```

### Testing

```bash
# Unit tests
pytest tests/test_chatbot_manager.py -v
```
