# Context

```python
from peteos.conversation.context import Context
```

A context represents the chat history for a step in a session, wrapped as a mutable JSON dict. The dict is the source of truth. Messages can be appended, anchors can be added, and dynamic messages are resolved via a content map — making the context a live, evolving structure rather than a static snapshot.

## Factory Methods

```python
Context.create(system_prompt_message: SystemPromptMessage | None = None, tool_definitions_message: ToolDefinitionsMessage | None = None, parent_context: Context | None = None) -> Context
```

Create a new empty context. The system prompt and tool definitions messages are placed in the first slots. If `parent_context` is provided, the content map is inherited.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `system_prompt_message` | `SystemPromptMessage \| None` | `None` | Optional system prompt. |
| `tool_definitions_message` | `ToolDefinitionsMessage \| None` | `None` | Optional tool definitions. |
| `parent_context` | `Context \| None` | `None` | Parent to inherit the content map from (for forks). |

```python
Context.load(path: str) -> Context
```

Load a context from a JSON file.

```python
Context.load_from_dict(json_dict: dict) -> Context
```

Load a context from a dictionary (used during deserialization).

## Instance Properties

| Property | Type | Description |
|---|---|---|
| `id` | `str` | The context ID. |
| `messages` | `list[Message]` | List of messages in this context. |
| `message_count` | `int` | Number of messages (same as the sequence counter). |
| `content_map` | `dict[str, str]` | Hash → string map for dynamic message resolution. |
| `anchor_points` | `list[tuple[str, int]]` | Ordered list of `(name, index)` anchor points. |
| `hook_index` | `dict[str, list[Message]]` | Hook ID → messages index. |
| `system_prompt_message` | `SystemPromptMessage \| None` | The system prompt message, or None. |
| `tool_definitions_message` | `ToolDefinitionsMessage \| None` | The tool definitions message, or None. |
| `raw_dict` | `dict` | The wrapped serialized dict. |

## Instance Methods

```python
append(message: Message, anchor_point: str = "messages") -> None
```

Insert a message at the anchor point's end-iterator position. Shifts subsequent messages and increments all anchors at or after the insertion point.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `message` | `Message` | — | The message to append. |
| `anchor_point` | `str` | `"messages"` | Name of the anchor point. |

```python
add_anchor(name: str, msg_index: int, after_existing: bool = True) -> None
```

Register a new anchor point. The `msg_index` is an absolute position into the messages array (0 = before all messages, len(messages) = after all messages).

| Parameter | Type | Description |
|---|---|---|
| `name` | `str` | Anchor point name. |
| `msg_index` | `int` | Absolute message index. |
| `after_existing` | `bool` | Ordering when another anchor occupies the same position. |

**Raises:** `ValueError` if the anchor name already exists.

```python
fork_insert_sequence(system_prompt_message: SystemPromptMessage | None = None, tool_definitions_message: ToolDefinitionsMessage | None = None, start: int | None = None, end: int | None = None) -> Context
```

Fork this context with Python-style slice semantics on mutation counter values. Only messages created within the `[start, end)` range are copied into the child.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `system_prompt_message` | `SystemPromptMessage \| None` | `None` | Optional new system prompt for the fork. |
| `tool_definitions_message` | `ToolDefinitionsMessage \| None` | `None` | Optional new tool definitions message. |
| `start` | `int \| None` | `None` | Start of mutation counter range (inclusive). `None` = from the first. |
| `end` | `int \| None` | `None` | End of mutation counter range (exclusive). `None` = to the last. |

**Returns:** A new Context with the selected messages.

```python
rolling_sequence_window(count: int) -> Context | None
```

Fork including the last `count` non-special messages by sequence number. Returns None if the window already covers all messages.

```python
rolling_token_window(max_tokens: int, encoding: str = "cl100k_base") -> Context | None
```

Fork keeping messages from the end until total tokens ≤ `max_tokens`. Includes special messages (system prompt, tool definitions). Returns None if no messages need to be dropped.

```python
strip_thinking() -> Context
```

Fork removing all `thinking` content parts from every message. Empty messages are skipped.

```python
total_token_count(encoding: str = "cl100k_base") -> int
```

Sum of token counts across all messages in this context.

```python
save(session_dir: str | Path) -> None
```

Save the context to `{context_id}.json` in the given session directory.
