# Anthropic Delta Translation

This document specifies how Anthropic-compatible API streaming responses are translated into the uniform delta format.

## Anthropic SSE Event Format

### Event Types

```
message_start     → Initial message metadata (role)
content_block_start → Start of a content block (type, id, name for tools)
content_block_delta → Incremental content (thinking, text, tool arguments)
content_block_stop → End of a content block
message_delta     → Final message metadata (stop_reason)
message_stop      → Stream completion
```

### Raw SSE Stream

Each event line contains:
```
event: <event_type>
data: <JSON object>
```

### 1. Message Start Event

```json
{
  "type": "message_start",
  "message": {
    "id": "msg_xxx",
    "role": "assistant",
    "model": "qwen/qwen3.5-35b-a3b",
    "stop_reason": null,
    "content": [],
    "usage": {"input_tokens": 294, "output_tokens": 0}
  }
}
```

### 2. Content Block Start Event

**Thinking block:**
```json
{
  "type": "content_block_start",
  "index": 0,
  "content_block": {
    "type": "thinking",
    "thinking": ""
  }
}
```

**Text block:**
```json
{
  "type": "content_block_start",
  "index": 0,
  "content_block": {
    "type": "text",
    "text": ""
  }
}
```

**Tool use block:**
```json
{
  "type": "content_block_start",
  "index": 1,
  "content_block": {
    "type": "tool_use",
    "id": "chatcmpl-tool-bf20b0ee27af2ff0",
    "name": "calculate",
    "input": {}
  }
}
```

### 3. Content Block Delta Events

**Thinking delta:**
```json
{
  "type": "content_block_delta",
  "index": 0,
  "delta": {
    "type": "thinking_delta",
    "thinking": "The user wants me to calculate:"
  }
}
```

**Text delta:**
```json
{
  "type": "content_block_delta",
  "index": 0,
  "delta": {
    "type": "text_delta",
    "text": "final response"
  }
}
```

**Tool arguments delta:**
```json
{
  "type": "content_block_delta",
  "index": 1,
  "delta": {
    "type": "input_json_delta",
    "partial_json": "{\"expression\":\"2**16 + 32 * 15 - 100\"}"
  }
}
```

**Signature delta (not used in uniform format):**
```json
{
  "type": "content_block_delta",
  "index": 0,
  "delta": {
    "type": "signature_delta",
    "signature": "82ba0132aa424fa2938b11af8f7ed003"
  }
}
```

### 4. Message Delta Event

```json
{
  "type": "message_delta",
  "delta": {
    "stop_reason": "tool_use",
    "stop_sequence": null
  },
  "message": {
    "stop_reason": "tool_use"
  }
}
```

### 5. Content Block Stop / Message Stop

```json
{
  "type": "content_block_stop",
  "index": 1
}

{
  "type": "message_stop"
}
```

## Index Field Semantics

Anthropic places `index` at the **top-level of content block events**, not inside arrays:

```json
{
  "type": "content_block_delta",
  "index": 0,        # <-- Points to content[0] in uniform format
  "delta": {...}
}
```

This top-level `index` indicates which content array item the event applies to. During translation:
1. The `index` is extracted from the event
2. It is propagated into the uniform delta's array items
3. The `index` is stripped from the final accumulated result

## Translation to Uniform Delta Format

### Translation Rules

| Anthropic Source Path | Uniform Target | Notes |
|-----------------------|----------------|-------|
| `message_start.message.role` | `role` | Only in message_start event |
| `message_delta.delta.stop_reason` | `stop_reason` | Only in message_delta event |
| `content_block_start.content_block.type` | `content[0].type` | "thinking", "text", or "tool_use" |
| `content_block_start.content_block.thinking` | `content[0].content` | Thinking block text |
| `content_block_start.content_block.text` | `content[0].content` | Text block content |
| `content_block_start.content_block.id` | `content[0].id` | Tool use ID |
| `content_block_start.content_block.name` | `content[0].name` | Tool use name |
| `content_block_delta.delta.thinking` | `content[0].content` | Thinking fragment |
| `content_block_delta.delta.text` | `content[0].content` | Text fragment |
| `content_block_delta.delta.partial_json` | `content[0].arguments` | Tool arguments fragment |

### Key Translation Strategy

1. **content_block_start**: Extract block metadata (type, id, name) and initial content
2. **content_block_delta**: Append content fragments or arguments to the block at position `index`
3. **message_start**: Extract `role` field
4. **message_delta**: Extract `stop_reason` field

## Examples

### Example 1: Thinking Block Accumulation

**SSE Events:**
```json
{"type":"content_block_start","content_block":{"type":"thinking","thinking":""},"index":0}
{"type":"content_block_delta","delta":{"type":"thinking_delta","thinking":"The"},"index":0}
{"type":"content_block_delta","delta":{"type":"thinking_delta","thinking":" user"},"index":0}
```

**Translated Uniform Deltas:**
```python
{"content": [{"index": 0, "type": "thinking", "content": ""}]}
{"content": [{"index": 0, "content": "The"}]}
{"content": [{"index": 0, "content": " user"}]}
```

**After Accumulation:**
```python
{
    "content": [{
        "type": "thinking",
        "content": "The user"
    }]
}
```

### Example 2: Tool Use Block

**SSE Events:**
```json
{"type":"content_block_start","content_block":{"type":"tool_use","id":"toolu_xxx","name":"calculate","input":{}},"index":1}
{"type":"content_block_delta","delta":{"type":"input_json_delta","partial_json":"{"},"index":1}
{"type":"content_block_delta","delta":{"type":"input_json_delta","partial_json":"\"expression\":\"2+2\"}"},"index":1}
```

**Translated Uniform Deltas:**
```python
{"content": [{"index": 1, "type": "tool_use", "id": "toolu_xxx", "name": "calculate", "arguments": {}}]}
{"content": [{"index": 1, "arguments": "{"}]}
{"content": [{"index": 1, "arguments": "\"expression\":\"2+2\"}"}]}
```

**After Accumulation:**
```python
{
    "content": [{
        "type": "tool_use",
        "id": "toolu_xxx",
        "name": "calculate",
        "arguments": '{"expression":"2+2"}'
    }]
}
```

### Example 3: Full Response with Multiple Blocks

**SSE Events (abbreviated):**
```json
{"type":"message_start","message":{"role":"assistant"}}
{"type":"content_block_start","content_block":{"type":"thinking","thinking":"Let"},"index":0}
{"type":"content_block_delta","delta":{"type":"thinking_delta","thinking":" me"},"index":0}
{"type":"content_block_start","content_block":{"type":"tool_use","id":"t1","name":"add"},"index":1}
{"type":"content_block_delta","delta":{"type":"input_json_delta","partial_json":"{"},"index":1}
{"type":"content_block_stop","index":1}
{"type":"message_delta","delta":{"stop_reason":"tool_use"}}
```

**Translation Result (after accumulation):**
```python
{
    "role": "assistant",
    "content": [
        {
            "type": "thinking",
            "content": "Let me"
        },
        {
            "type": "tool_use",
            "id": "t1",
            "name": "add",
            "arguments": "{"
        }
    ],
    "stop_reason": "tool_use"
}
```

## Key Differences from OpenAI

| Aspect | OpenAI | Anthropic |
|--------|--------|-----------|
| Index location | In arrays (`choices[0]`, `tool_calls[0]`) | **Top-level event** (`index: 0`) |
| Thinking field | `delta.reasoning` | `delta.thinking` (with `thinking_delta` type) |
| Text field | `delta.content` | `delta.text` (with `text_delta` type) |
| Tool call ID | `tool_calls[0].id` | `content_block_start.content_block.id` |
| Tool arguments | `function.arguments` | `partial_json` |
| Block start event | No explicit start | **Required**: `content_block_start` |
| Stop reason | `finish_reason` in delta | **Separate event**: `message_delta` |

## Implementation Notes

1. **Sequential Block Processing**: Anthropic returns each content block as a separate event with its own `index`. The translation must:
   - Create new content array items when `index` exceeds current length
   - Concatenate string fragments for the same `index`
   - Merge object fields (tool_use.id, tool_use.name) into same array item

2. **Event Filtering**: Only process relevant events:
   - `message_start`: Extract role
   - `content_block_start`: Initialize new content block
   - `content_block_delta`: Append content/arguments
   - `message_delta`: Extract stop_reason
   - Ignore `content_block_stop` and `message_stop` (informational only)

3. **Delta Type Handling**: For `content_block_delta`, check `delta.type` to determine which field contains the content:
   - `thinking_delta`: Use `delta.thinking`
   - `text_delta`: Use `delta.text`
   - `input_json_delta`: Use `delta.partial_json`