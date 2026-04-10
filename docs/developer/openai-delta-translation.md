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

### Examples

#### Example 1: String Dictionary (No Array)

**OpenAI SSE:**
```json
{"choices": [{"index": 0, "delta": {"reasoning": "Hello"}}]}
```

**Translates to uniform delta:**
```python
{"reasoning": "Hello"}
```

#### Example 2: Array with Index (tool call at position 0)

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
    "tool_calls": [{
        "index": 0,
        "type": "tool_use",
        "id": "call_abc",
        "name": "add",
        "arguments": "{}"
    }]
}
```

#### Example 3: Array with Index (tool call at position 1)

**OpenAI SSE:**
```json
{
  "choices": [{
    "delta": {
      "tool_calls": [{
        "index": 1,
        "id": "call_def",
        "type": "function",
        "function": {"name": "sub", "arguments": "{}"}
      }]
    }
  }]
}
```

**Translates to uniform delta:**
```python
{
    "tool_calls": [{
        "index": 1,
        "type": "tool_use",
        "id": "call_def",
        "name": "sub",
        "arguments": "{}"
    }]
}
```

### Mapping to Uniform Format

| OpenAI Field | Uniform Target | Notes |
|--------------|----------------|-------|
| `choices[0].delta.role` | `role` | Only in first event |
| `choices[0].delta.reasoning` | `reasoning` | Reasoning fragments (no array) |
| `choices[0].delta.content` | `content` | Text fragments (no array) |
| `choices[0].finish_reason` | `stop_reason` | Only in final event |
| `choices[0].delta.tool_calls[0].index` | `tool_calls[0].index` | Array position preserved |
| `choices[0].delta.tool_calls[0].type` | `tool_calls[0].type` | Always `"tool_use"` |
| `choices[0].delta.tool_calls[0].id` | `tool_calls[0].id` | Tool call ID |
| `choices[0].delta.tool_calls[0].function.name` | `tool_calls[0].name` | Function name |
| `choices[0].delta.tool_calls[0].function.arguments` | `tool_calls[0].arguments` | JSON argument fragments |
