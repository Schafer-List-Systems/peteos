# Session

```python
from peteos.conversation.session import Session
```

A session represents a single interaction thread. It wraps a JSON dict as the source of truth and holds the active `Context`.

## Factory Methods

```python
Session.create(parent_dir: str, system_prompt_message: SystemPromptMessage | None = None, tool_definitions_message: ToolDefinitionsMessage | None = None) -> Session
```

Create a new session with a freshly created context.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `parent_dir` | `str` | — | The agent directory on disk. |
| `system_prompt_message` | `SystemPromptMessage \| None` | `None` | Optional system prompt for this session. |
| `tool_definitions_message` | `ToolDefinitionsMessage \| None` | `None` | Optional tool definitions for this session. |

```python
Session.load(parent_dir: str, session_uuid: str) -> Session
```

Load a session from `parent_dir / session_uuid / session.json`.

| Parameter | Type | Description |
|---|---|---|
| `parent_dir` | `str` | The agent directory. |
| `session_uuid` | `str` | The session UUID (also the subdirectory name). |

**Raises:** `FileNotFoundError` if `session.json` does not exist.

## Instance Properties

| Property | Type | Description |
|---|---|---|
| `uuid` | `str` | The session UUID. |
| `session_dir` | `Path` | The session directory path (`parent_dir / uuid`). |
| `active_context` | `Context \| None` | The currently active context Python object. |
| `active_context_id` | `str \| None` | The active context ID from the JSON dict. |
| `auto_approve_tools` | `list[str]` | List of tool names auto-approved by this session. |
| `raw_dict` | `dict` | The wrapped serialized dict. |

## Instance Methods

```python
set_active_context(context: Context) -> None
```

Set the active context, synchronizing both the JSON dict ID and the Python object reference.

```python
save() -> None
```

Save the session and its active context to disk. Creates the session directory if needed.

```python
rolling_sequence_window(count: int) -> Context | None
```

Apply a rolling sequence window to the active context, keeping the last `count` non-special messages.

| Parameter | Type | Description |
|---|---|---|
| `count` | `int` | Number of messages to keep from the end. |

**Returns:** The new active Context, or None if no messages were dropped.

```python
rolling_token_window(max_tokens: int, encoding: str = "cl100k_base") -> Context | None
```

Apply a rolling token window to the active context, keeping messages from the end until total tokens ≤ `max_tokens`.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `max_tokens` | `int` | — | Maximum token count. |
| `encoding` | `str` | `"cl100k_base"` | Tiktoken encoding name. |

**Returns:** The new active Context, or None if no messages were dropped.
