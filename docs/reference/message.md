# Message

```python
from peteos.conversation import Message, ContentPart
```

Used for steering the agent — pushing new messages into the runner's event queue from inside tool methods. See **[Steering the Agent](../concepts/steering.md)** for when and how to use this.

## ContentPart Factory Methods

```python
ContentPart.create_text(text: str) -> ContentPart
ContentPart.create_thinking(text: str) -> ContentPart
```

| Factory | Description |
|---|---|
| `create_text` | Plain text content. |
| `create_thinking` | Model's internal reasoning content. |

```python
ContentPart.create_image(source: dict) -> ContentPart
ContentPart.create_video(source: dict) -> ContentPart
ContentPart.create_pdf(source: dict) -> ContentPart
```

| Factory | `source` format |
|---|---|
| All three | `{"type": "base64"|"url", "data": str, "media_type": str}` or `{"type": "url", "url": str}` |

```python
ContentPart.create_tool_use(call_id: str, name: str, arguments: str) -> ContentPart
ContentPart.create_tool_result(call_id: str, content: str) -> ContentPart
```

- `create_tool_use`: `arguments` is a JSON string of tool arguments.
- `create_tool_result`: `call_id` must match the tool use it responds to.

## Message Factory Methods

```python
Message.create(role: str, content_parts: list[ContentPart], metadata: dict | None = None) -> Message
```

Create a message from content parts. `role` is one of `"user"`, `"assistant"`, `"system"`, `"tool"`.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `role` | `str` | — | Message role. |
| `content_parts` | `list[ContentPart]` | — | Content parts to include. |
| `metadata` | `dict \| None` | `None` | Arbitrary metadata. |

## Instance Properties

| Property | Type | Description |
|---|---|---|
| `role` | `str` | The message role. |
| `id` | `str` | The message UUID. |
| `content` | `list[ContentPart]` | List of content parts. |
| `creation_timestamp` | `datetime` | When the message was created. |
| `metadata` | `dict` | Arbitrary metadata dict. |
