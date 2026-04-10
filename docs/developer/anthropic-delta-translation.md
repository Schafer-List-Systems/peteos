# Anthropic Delta Translation

This document specifies how Anthropic-compatible API streaming responses are translated into the uniform delta format.

## Anthropic SSE Event Format

### Basic Stream Structure

```json
{
  "type": "message_start",
  "message": {
    "id": "msg_xxx",
    "type": "message",
    "role": "assistant",
    "content": [],
    "model": "model-name",
    "stop_reason": null,
    "stop_sequence": null
  }
}
```

### Content Block Events

```json
{
  "type": "content_block_start",
  "index": 0,
  "content_block": {
    "type": "thinking",
    "thinking": "thought content"
  }
}
```

```json
{
  "type": "content_block_delta",
  "index": 0,
  "delta": {
    "type": "thinking_delta",
    "thinking": "more thought content"
  }
}
```

```json
{
  "type": "content_block_stop",
  "index": 0
}
```

### Tool Call Format

```json
{
  "type": "content_block_start",
  "index": 0,
  "content_block": {
    "type": "tool_use",
    "id": "toolu_xxx",
    "name": "function_name"
  }
}
```

```json
{
  "type": "content_block_delta",
  "index": 0,
  "delta": {
    "type": "input_json_delta",
    "partial_json": '{"key": "value'
  }
}
```

### Message Delta (Stop Reason)

```json
{
  "type": "message_delta",
  "delta": {
    "stop_reason": "end_turn",
    "stop_sequence": null
  },
  "message": {
    "id": "msg_xxx",
    "model": "model-name",
    "stop_reason": "end_turn",
    "stop_sequence": null
  }
}
```

## Translation to Uniform Delta Format

### Translation Rules

#### 1. Content Blocks

Anthropic uses `content_block_start`, `content_block_delta`, and `content_block_stop` events with a top-level `index` field.

```
source_path                                    target_path
---------------------------------------------  -------------------------
message_start.message.role                     "role"
message_delta.delta.stop_reason                "stop_reason"

# For thinking blocks
content_block_start.content_block.type         "content[0].type"      (value: "thinking")
content_block_start.content_block.thinking     "content[0].content"
content_block_delta.delta.thinking             "content[0].content"   (concatenated)

# For text blocks
content_block_start.content_block.type         "content[0].type"      (value: "text")
content_block_start.content_block.text         "content[0].content"
content_block_delta.delta.text                 "content[0].content"   (concatenated)

# For tool_use blocks
content_block_start.content_block.type         "content[0].type"      (value: "tool_use")
content_block_start.content_block.id           "content[0].id"
content_block_start.content_block.name         "content[0].name"
content_block_delta.delta.partial_json         "content[0].arguments" (concatenated)
```

#### 2. Index Field Handling

Anthropic places `index` at the **top level of events**, not inside arrays:

```json
{
  "type": "content_block_delta",
  "index": 0,           # <-- Top-level, points to content[0]
  "delta": {...}
}
```

During translation, this top-level `index` is used to:
1. Determine which content array item to update
2. Include `index` in the translated delta for merge control
3. Strip `index` from the final accumulated result

## Translation Implementation

### Event Type Mapping

```python
EVENT_TYPE_MAP = {
    "content_block_start": "start",
    "content_block_delta": "delta",
    "content_block_stop": "stop",
    "message_start": "message_start",
    "message_delta": "message_delta"
}
```

### Translation Function Algorithm

```python
def translate_anthropic_event(event: Dict) -> Dict:
    """
    Translate Anthropic SSE event to uniform delta format.
    """
    result = {}
    
    # Handle message_start - extract role
    if event.get("type") == "message_start":
        message = event.get("message", {})
        if "role" in message:
            result["role"] = message["role"]
        return result
    
    # Handle message_delta - extract stop_reason
    if event.get("type") == "message_delta":
        delta = event.get("delta", {})
        if "stop_reason" in delta:
            result["stop_reason"] = delta["stop_reason"]
        return result
    
    # Extract top-level index (position pointer)
    index = event.get("index")
    
    # Handle content_block_start
    if event.get("type") == "content_block_start":
        cb = event.get("content_block", {})
        block_type = cb.get("type")
        
        content_block = {
            "index": index,
            "type": block_type,
        }
        
        if block_type == "thinking":
            content_block["content"] = cb.get("thinking", "")
        elif block_type == "text":
            content_block["content"] = cb.get("text", "")
        elif block_type == "tool_use":
            content_block["id"] = cb.get("id")
            content_block["name"] = cb.get("name")
            content_block["arguments"] = ""
        
        result["content"] = [content_block]
        return result
    
    # Handle content_block_delta
    if event.get("type") == "content_block_delta":
        delta = event.get("delta", {})
        delta_type = delta.get("type")
        
        content_block = {
            "index": index,
        }
        
        if delta_type == "thinking_delta":
            content_block["type"] = "thinking"
            content_block["content"] = delta.get("thinking", "")
        elif delta_type == "text_delta":
            content_block["type"] = "text"
            content_block["content"] = delta.get("text", "")
        elif delta_type == "input_json_delta":
            content_block["type"] = "tool_use"
            content_block["arguments"] = delta.get("partial_json", "")
        
        result["content"] = [content_block]
        return result
    
    return result
```

### Handling Multiple Content Blocks

Anthropic returns each content block as a separate event with its own `index`. The translation must:

1. Create new content array items when `index` exceeds current length
2. Concatenate string fragments for the same `index`
3. Merge object fields (tool_use.id, tool_use.name) into same array item

**Example: Multi-block response**

```json
// Event 1: thinking block
{
  "type": "content_block_start",
  "index": 0,
  "content_block": {"type": "thinking", "thinking": "Let me"}
}
// Translates to: {"content": [{"index": 0, "type": "thinking", "content": "Let me"}]}

// Event 2: more thinking
{
  "type": "content_block_delta",
  "index": 0,
  "delta": {"type": "thinking_delta", "thinking": " calculate"}
}
// Merges into content[0]: {"content": [{"index": 0, "type": "thinking", "content": "Let me calculate"}]}

// Event 3: tool_use block
{
  "type": "content_block_start",
  "index": 1,
  "content_block": {"type": "tool_use", "id": "toolu_xxx", "name": "add"}
}
// Creates content[1]: {"content": [..., {"index": 1, "type": "tool_use", "id": "toolu_xxx", "name": "add"}]}
```

## Key Differences from OpenAI

| Aspect | OpenAI | Anthropic |
|--------|--------|-----------|
| Role location | `choices[0].delta.role` | `message_start.message.role` |
| Reasoning field | `delta.reasoning` | `delta.thinking` (with `thinking_delta` type) |
| Text field | `delta.content` | `delta.text` (with `text_delta` type) |
| Tool call ID | `tool_calls[0].id` | `content_block_start.content_block.id` |
| Index location | **In arrays** (`choices[0]`, `tool_calls[0]`) | **Top-level event** (`index: 0`) |
| Tool arguments | `function.arguments` | `partial_json` |
| Block start event | No explicit start | `content_block_start` required |
| Stop reason | `finish_reason` in delta | `message_delta` event |

## Notes

- Anthropic requires `content_block_start` before `content_block_delta` for each block
- The top-level `index` field is a position pointer to the content array
- Tool call `arguments` (via `partial_json`) arrives in multiple fragments
- `content_block_stop` signals end of a block (not strictly needed for translation)
- Different block types (`thinking`, `text`, `tool_use`) are identified by `content_block.type`