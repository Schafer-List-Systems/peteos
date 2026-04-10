# OpenAI-Compatible Chat API

This document shows how to use the OpenAI-compatible API for multi-turn conversations with reasoning support.

Consult the OpenAI developers reference for details:
`https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create`

## Format Comparison

### Uniform Internal Format

In the internal system, messages use a content array with `ContentPart` objects:

```json
{
  "role": "assistant",
  "content": [
    {"type": "text", "text": "Hello!"},
    {"type": "reasoning", "reasoning": "The user said hello, so I should respond with a greeting."}
  ]
}
```

### OpenAI Backend Format

The backend expects messages with `content` as a string and `reasoning` as a separate field:

```json
{
  "role": "assistant",
  "content": "Hello!",
  "reasoning": "The user said hello, so I should respond with a greeting."
}
```

### Translation Mapping

| Uniform Format (ContentPart) | OpenAI Backend Format |
|------------------------------|----------------------|
| `{"type": "text", "text": "..."}` | `"content": "..."` |
| `{"type": "reasoning", "reasoning": "..."}` | `"reasoning": "..."` |
| `{"type": "tool_calls", "tool_calls": [...]}` | `"tool_calls": [...]` |

## Multi-turn Example

### Request 1: Simple greeting

```bash
curl -s http://192.168.12.20:3001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen/qwen3.5-35b-a3b",
    "messages": [
      {"role": "user", "content": "hello"}
    ],
    "stream": false
  }' | python3 -m json.tool
```

**Response:**
```json
{
  "id": "chatcmpl-9536c527ca0850f5",
  "object": "chat.completion",
  "created": 1775427156,
  "model": "qwen/qwen3.5-35b-a3b",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "\n\nHello! \ud83d\udc4b How can I help you today?",
        "reasoning": "Thinking Process:\n\n1.  **Analyze the Input:**\n    *   Input: \"hello\"\n    *   Intent: Greeting.\n    ...\n"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 11,
    "total_tokens": 929,
    "completion_tokens": 918
  }
}
```

### Request 2: Continuation with reasoning from previous turn

```bash
curl -s http://192.168.12.20:3001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen/qwen3.5-35b-a3b",
    "messages": [
      {"role": "user", "content": "hello"},
      {
        "role": "assistant",
        "content": "Hello! How can I help you today?",
        "reasoning": "This is a simple greeting, no need for tools."
      },
      {"role": "user", "content": "what is the sine of 44 degrees"}
    ],
    "stream": false
  }' | python3 -m json.tool
```

**Response:**
```json
{
  "id": "chatcmpl-945f37bc78026193",
  "object": "chat.completion",
  "created": 1775427567,
  "model": "qwen/qwen3.5-35b-a3b",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "\n\nThe sine of 44 degrees is approximately **0.6947**.\n\nHere is the value to more decimal places:\n$$ \\sin(44^\\circ) \\approx 0.69465837 $$",
        "reasoning": "Thinking Process:\n\n1.  **Analyze the Request:** The user is asking for the sine of 44 degrees ($\\sin(44^\\circ)$).\n\n2.  **Determine the Nature of the Answer:** This is a mathematical calculation. Since $44^\\circ$ is not a special angle (like 30, 45, 60 degrees), the result will be an irrational number...\n\n7.  **Final Decision:** Provide the value rounded to four decimal places, as that's common for general queries.\n\n    *   $\\sin(44^\\circ) \\approx 0.6947$\n\n8.  **Output Generation:** (Matches the drafted response)\n    \"The sine of 44 degrees is approximately 0.6947.\" (Or similar).\n"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 39,
    "total_tokens": 719,
    "completion_tokens": 680
  }
}
```

## Concrete Example: Multi-turn Conversation

### Uniform Format (Internal)

This is how the conversation is stored internally:

```json
{
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "hello"}
      ]
    },
    {
      "role": "assistant",
      "content": [
        {"type": "text", "text": "Hello! How can I help you today?"},
        {"type": "reasoning", "reasoning": "This is a simple greeting, no need for tools."}
      ]
    },
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "what is the sine of 44 degrees"}
      ]
    }
  ]
}
```

### Translated to OpenAI Format

This is how the same conversation is sent to the backend:

```json
{
  "model": "qwen/qwen3.5-35b-a3b",
  "stream": false,
  "messages": [
    {
      "role": "user",
      "content": "hello"
    },
    {
      "role": "assistant",
      "content": "Hello! How can I help you today?",
      "reasoning": "This is a simple greeting, no need for tools."
    },
    {
      "role": "user",
      "content": "what is the sine of 44 degrees"
    }
  ]
}
```

### Translation Process

1. **Iterate over each message** in the conversation
2. **For each ContentPart in the message:**
   - If `type == "text"`, extract the `text` value and map to `content`
   - If `type == "reasoning"`, extract the `reasoning` value and map to `reasoning`
   - If `type == "tool_calls"`, extract and map to `tool_calls`
3. **Build the message dict** with API-specific keys

## Key Points

1. **Reasoning as separate field**: The `reasoning` field must be at the message level, not inside the `content` array.

2. **Content as string**: The `content` field should be a plain string, not an array of parts.

3. **Full history with reasoning**: You can include previous messages with both `content` and `reasoning` fields to provide context for the model's thinking.

4. **Tool calls**: Tools should be sent separately in the `tools` array, not as content parts.

5. **Translation table**: Use `REQUEST_TRANSLATIONS` to map uniform keys to API-specific keys:
   - `"text"` → `"content"`
   - `"reasoning"` → `"reasoning"`
   - `"tool_calls"` → `"tool_calls"`
