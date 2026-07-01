# ChatBotManager

```python
from peteos.chatbot import ChatBotManager
```

A singleton manager for configuring LLM backends. Use this to register the chatbot provider(s) your agents will use.

## Adding a Backend

```python
await ChatBotManager.add_backend("local", "http://localhost:8000")
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | — | Unique identifier for the backend. |
| `url` | `str` | — | Base URL of the API (e.g., `http://localhost:8000`). |
| `api_type` | `str \| None` | `None` | API type override (`"openai"` or `"anthropic"`). If not provided, auto-detected. |
| `**kwargs` | — | — | Additional configuration (streaming, max_tokens, etc.). |

**Returns:** A `BackendInfo` dataclass with detected API type and discovered models.

**Raises:** `ValueError` if a backend with the same name already exists.

## Loading from JSON

Backends can be configured programmatically or loaded from a JSON file:

```python
await ChatBotManager.load_from_file("config.json")
```

The JSON format is:

```json
{
    "backends": [
        {
            "name": "my-backend",
            "url": "http://localhost:8000",
            "api_type": "openai",
            "streaming": false
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
