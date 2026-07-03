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

## Auto-Configuration

When you import `peteos`, the manager automatically discovers and loads a `peteos.json` configuration file from standard locations. The first file found wins — no merging, no explicit calls needed:

1. **`PETEOS_CONFIG`** environment variable (full path to `peteos.json`)
2. **`$XDG_CONFIG_HOME/peteos/peteos.json`** (defaults to `~/.config/peteos/peteos.json`)
3. **`/etc/peteos/peteos.json`** (system-wide)
4. **`./peteos.json`** (current working directory)

The configuration format is identical to the JSON file format described above. If no config file is found, a warning is logged and backends remain empty — no error is raised.

```json
{
    "backends": [
        {
            "name": "local",
            "url": "http://localhost:3001"
        },
        {
            "name": "gemini",
            "url": "https://generativelanguage.googleapis.com",
            "api_type": "gemini",
            "api_key": "YOUR_GEMINI_API_KEY"
        },
        {
            "name": "anthropic",
            "url": "https://api.anthropic.com",
            "api_type": "anthropic",
            "api_key": "YOUR_ANTHROPIC_API_KEY"
        }
    ]
}
```

```python
import peteos  # auto-loads peteos.json on import
```

## Manual Loading

If you need to load a configuration from a non-standard location, use `load_from_json()` or `load_from_file()`:

```python
from peteos.chatbot import ChatBotManager
import asyncio

asyncio.run(ChatBotManager.load_from_file("/custom/path/config.json"))
```

Or load programmatically with `add_backend()`.
