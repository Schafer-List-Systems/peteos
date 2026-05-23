# OpenAI Delta Translation

This document specifies how OpenAI-compatible API streaming responses translate to the uniform delta format.

## OpenAI SSE Format

Each SSE line is a Python dict representing a choice delta:

```python
{
  'id': 'chatcmpl-xxx',
  'object': 'chat.completion.chunk',
  'created': 1775470645,
  'model': 'qwen/qwen3.5-35b-a3b',
  'choices': [{
    'index': 0,
    'delta': {
      'role': 'assistant',      # Only in first event
      'reasoning': 'fragment',  # Reasoning text fragments
      'content': 'fragment',    # Final text fragments
      'tool_calls': [...]       # Tool call deltas (when present)
    },
    'logprobs': None,
    'finish_reason': 'stop' | 'tool_calls'  # Only in final event
  }]
}
```

### Reasoning Events

Reasoning arrives as sequential fragments in `delta.reasoning`:

```python
{'choices': [{'delta': {'reasoning': 'The'}}]}
{'choices': [{'delta': {'reasoning': ' user'}}]}
{'choices': [{'delta': {'reasoning': ' wants'}}]}
```

### Text Events

Text arrives as sequential fragments in `delta.content`:

```python
{'choices': [{'delta': {'content': 'Hello'}}]}
{'choices': [{'delta': {'content': ' world'}}]}
```

### Tool Call Events

Tool calls arrive with `index` at multiple levels:

```python
{
  'choices': [{
    'delta': {
      'tool_calls': [{
        'index': 0,
        'id': 'call_xxx',
        'type': 'function',
        'function': {
          'name': 'calculate',
          'arguments': '{"expr":'
        }
      }]
    }
  }]
}
```

Arguments arrive as fragments in `tool_calls[0].function.arguments`.

## Translation Rules

### Post-Processing

After path-based translation, `_set_text_types()` and `_set_tool_call_types()` inject type discriminators:

1. `_set_tool_call_types`: Sets `type: "tool_use"` on items with `name` + `arguments`/`id`
2. `_set_text_types`: Sets `type: "text"` on items without a `type`

### Examples

#### Example 1: Reasoning Fragment

**OpenAI SSE:**
```json
{"choices": [{"delta": {"reasoning": "Hello"}}]}
```

**Translates to uniform delta:**
```python
{"_reasoning": "Hello"}
```

**After post-processing → thinking block in content array:**
```python
{
    "content": [{
        "index": 0,
        "type": "thinking",
        "content": "Hello"
    }]
}
```

#### Example 2: Text Fragment

**OpenAI SSE:**
```json
{"choices": [{"delta": {"content": "Hello"}}]}
```

**Translates to uniform delta:**
```python
{"content": [{"index": 0, "content": "Hello"}]}
```

**After `_set_text_types` post-processing:**
```python
{"content": [{"index": 0, "type": "text", "content": "Hello"}]}
```

#### Example 3: Tool Call

**OpenAI SSE:**
```json
{
  "choices": [{
    "delta": {
      "tool_calls": [{
        "index": 0,
        "id": "call_abc",
        "type": "function",
        "function": {"name": "add", "arguments": "{}"}
      }]
    }
  }]
}
```

**Translates to uniform delta:**
```python
{
    "content": [{
        "index": 0,
        "id": "call_abc",
        "name": "add",
        "arguments": "{}"
    }]
}
```

**After `_set_tool_call_types` post-processing:**
```python
{
    "content": [{
        "index": 0,
        "type": "tool_use",
        "id": "call_abc",
        "name": "add",
        "arguments": "{}"
    }]
}
```

### Mapping to Uniform Format

| OpenAI Field | Uniform Target | Notes |
|--------------|----------------|-------|
| `choices[0].delta.role` | `role` | Only in first event |
| `choices[0].delta.reasoning` | `_reasoning` (→ `content[N].type="thinking"` after post-processing) | Reasoning fragments |
| `choices[0].delta.content` | `content[0].content` | Text fragments |
| `choices[0].finish_reason` | `stop_reason` | Only in final event |
| `choices[0].delta.tool_calls[0].index` | `content[0].index` | Array position preserved |
| `choices[0].delta.tool_calls[0].id` | `content[0].id` | Tool call ID |
| `choices[0].delta.tool_calls[0].function.name` | `content[0].name` | Function name |
| `choices[0].delta.tool_calls[0].function.arguments` | `content[0].arguments` | JSON argument fragments |
