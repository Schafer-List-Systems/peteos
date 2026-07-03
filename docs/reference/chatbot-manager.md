# ChatBotManager

```python
from peteos.chatbot import ChatBotManager
```

A singleton manager for configuring LLM backends. Use this to register the chatbot provider(s) your agents will use.

## Adding a Backend

```python
await ChatBotManager.add_backend("my-backend", "http://localhost:8000")
```

Adds a new backend by listing models from the provider and creating a `ChatBot` instance for every model it finds.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | — | Unique identifier for the backend. |
| `url` | `str` | — | Base URL of the API (e.g., `http://localhost:8000`). |
| `api_type` | `str \| None` | `None` | API type override (`"openai"`, `"anthropic"`, or `"gemini"`). If not provided, auto-detected. |
| `**kwargs` | — | — | See below for the full list. |

**Supported keyword arguments:**

| Keyword | Default | Description |
|---|---|---|
| `api_key` | `None` | API authentication key. |
| `chat_endpoint` | API-specific default | Custom chat endpoint path. |
| `models_endpoint` | API-specific default | Custom models listing endpoint path. |
| `streaming` | `False` | Use streaming mode by default. |
| `max_tokens` | `4096` | Maximum tokens to generate. |
| `retry_delays` | `None` | List of delay intervals (in seconds) for retry attempts. |

**Returns:** A `BackendInfo` dataclass with detected API type and discovered models.

**Raises:** `ValueError` if a backend with the same name already exists.

## Loading from JSON

Backends can be configured programmatically or loaded from a JSON file:

```python
await ChatBotManager.load_from_file("config.json")
```

The JSON format requires `name` and `url` for each backend. All other fields are optional. The JSON keys correspond one-to-one with the keyword arguments of [`add_backend`](#add_backend).

Minimal example:

```json
{
    "backends": [
        {
            "name": "my-backend",
            "url": "http://localhost:8000"
        }
    ]
}
```

Full example:

```json
{
    "backends": [
        {
            "name": "my-backend",
            "url": "http://localhost:8000",
            "api_type": "openai",
            "api_key": "sk-...",
            "chat_endpoint": "/v1/chat/completions",
            "models_endpoint": "/v1/models",
            "streaming": false,
            "max_tokens": 4096,
            "retry_delays": [0.5, 1.0, 2.0]
        }
    ]
}
```

## Removing and Resetting

```python
# Remove a single backend
ChatBotManager.remove_backend("my-backend")

# Clear all backends
ChatBotManager.reset()
```

## Listing Chatbots

```python
bots = ChatBotManager.list_chatbots("gpt-4.*")
# Returns list of (model_id, ChatBot) tuples
```

| Parameter | Type | Description |
|---|---|---|
| `model_regex` | `str` | Regex pattern to filter model IDs. |

**Returns:** Sorted list of `(model_id, ChatBot)` tuples matching the pattern.

## `BackendInfo`

```python
@dataclass
class BackendInfo:
    name: str
    url: str
    api_type: str  # "openai" or "anthropic"
    models: dict[str, Any]  # model_id -> ChatBot instance
```

Returned by `add_backend()` and `load_from_file()` to show which models were discovered.
