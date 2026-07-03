# Backends

The [ChatBotManager](../reference/chatbot-manager.md) configures which LLM APIs the agent can use. It is a class-method-only singleton — there is exactly one manager shared across the entire application.

## Registering a Backend

Register a backend by name and URL. The manager auto-detects the API type (OpenAI, Anthropic, or Gemini) and discovers available models:

```python
from peteos.chatbot.manager import ChatBotManager

await ChatBotManager.add_backend("local", "http://localhost:8000")
```

You can also specify the API type explicitly to skip auto-detection:

```python
await ChatBotManager.add_backend("anthropic-api", "https://api.anthropic.com", api_type="anthropic")
```

## Configuration from JSON

For production deployments, backends are typically loaded from a JSON file:

```json
{
    "backends": [
        {
            "name": "production",
            "url": "https://api.openai.com/v1",
            "api_type": "openai",
            "streaming": true,
            "max_tokens": 4096
        },
        {
            "name": "local",
            "url": "http://localhost:8000",
            "api_type": "openai"
        }
    ]
}
```

```python
await ChatBotManager.load_from_file("config/chatbot_config.json")
```

`load_from_file` clears all previously registered backends before loading — it replaces state entirely.
