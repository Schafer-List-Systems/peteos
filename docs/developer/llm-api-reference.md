# LLM API Reference

This document describes the request/response formats of major LLM APIs to understand the variations that the `ChatBot` translation layer must handle.

## Major LLM API Formats

### 1. OpenAI Chat Completions API

**Endpoint:** `/v1/chat/completions`

**Request Format:**
```json
{
  "model": "gpt-4o",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant"},
    {"role": "user", "content": "What's the weather?"},
    {"role": "assistant", "content": "Let me check that for you"}
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get weather for a city",
        "parameters": {
          "type": "object",
          "properties": {
            "city": {"type": "string"}
          },
          "required": ["city"]
        }
      }
    }
  ],
  "tool_choice": "auto",
  "max_tokens": 4096,
  "stream": true
}
```

**Response Format (streaming):**
```
data: {"choices": [{"delta": {"role": "assistant", "content": "Sure"}}]}
data: {"choices": [{"delta": {"content": " I can help"}}]}
data: {"choices": [{"delta": {"content": " with that."}}]}
data: [DONE]
```

---

### 2. Anthropic Messages API

**Endpoint:** `/v1/messages`

**Request Format:**
```json
{
  "model": "claude-3-opus",
  "messages": [
    {"role": "user", "content": "What's the weather?"},
    {"role": "assistant", "content": "Let me check"}
  ],
  "system": "You are a helpful assistant\n\nAvailable tools:\n- get_weather: Get weather for a city",
  "tools": [
    {
      "name": "get_weather",
      "description": "Get weather for a city",
      "input_schema": {
        "type": "object",
        "properties": {
          "city": {"type": "string"}
        },
        "required": ["city"]
      }
    }
  ],
  "max_tokens": 4096,
  "stream": true
}
```

**Response Format (streaming):**
```
{"type": "message_start", "message": {"role": "assistant"}}
{"type": "content_block_delta", "delta": {"type": "text_delta", "text": "Sure"}}
{"type": "content_block_delta", "delta": {"type": "text_delta", "text": " I can help"}}
{"type": "message_delta", "delta": {"stop_reason": "end_turn"}}
{"type": "message_stop"}
```

---

### 3. Anthropic Complete API (Legacy)

**Endpoint:** `/v1/complete`

**Request Format:**
```json
{
  "model": "claude-2",
  "prompt": "\n\nHuman: What's the weather?\n\nAssistant:",
  "max_tokens_to_sample": 4096,
  "system": "You are a helpful assistant",
  "stream": true
}
```

**Note:** This API does not support structured tools or modern message formats.

---

### 4. Google Vertex AI / Gemini API

**Endpoint:** `/generateContent`

**Request Format:**
```json
{
  "contents": [
    {
      "role": "user",
      "parts": [
        {"text": "What's the weather?"}
      ]
    },
    {
      "role": "model", 
      "parts": [
        {"text": "Let me check"}
      ]
    }
  ],
  "systemInstruction": {
    "parts": [{"text": "You are a helpful assistant"}]
  },
  "tools": [
    {
      "functionDeclarations": [
        {
          "name": "get_weather",
          "description": "Get weather for a city",
          "parameters": {
            "type": "OBJECT",
            "properties": {
              "city": {"type": "STRING"}
            }
          }
        }
      ]
    }
  ],
  "generationConfig": {
    "maxOutputTokens": 4096
  }
}
```

---

### 5. Cohere Chat API

**Endpoint:** `/v1/chat`

**Request Format:**
```json
{
  "model": "command-r",
  "message": "What's the weather?",
  "chat_history": [
    {"role": "USER", "message": "Hello"},
    {"role": "CHATBOT", "message": "Hi there"}
  ],
  "tools": [
    {
      "name": "get_weather",
      "description": "Get weather for a city",
      "parameter_definitions": {
        "city": {
          "type": "string",
          "description": "City name"
        }
      }
    }
  ],
  "search_queries_only": false,
  "temperature": 0.7
}
```

---

### 6. Mistral / Pixtral API

**Endpoint:** `/v1/chat/completions`

**Request Format:**
```json
{
  "model": "mistral-large",
  "messages": [
    {"role": "system", "content": "You are helpful"},
    {"role": "user", "content": [{"type": "text", "text": "What's this image?"}, {"type": "image", "url": "..."}]}
  ],
  "tools": [...],
  "max_tokens": 4096,
  "stream": true
}
```

**Multi-modal Support:** Content can be an array of objects with `type: "text"` or `type: "image_url"`.

---

### 7. Hugging Face Inference API

**Endpoint:** `/v1/chat/completions`

**Request Format:**
```json
{
  "model": "mistralai/Mixtral-8x7B",
  "messages": [
    {"role": "system", "content": "You are helpful"},
    {"role": "user", "content": "Hello"}
  ],
  "stream": true
}
```

**Note:** Follows OpenAI-compatible format.

---

## Comparison Summary

| API | System Prompt | Messages | Tools | Images/Media |
|-----|---------------|----------|-------|--------------|
| **OpenAI** | `messages[].content` (role: system) | `messages[]` array | `tools[]` array | `messages[].content[]` with `type: image_url` |
| **Anthropic Messages** | `system` field | `messages[]` array | `tools[]` array | `messages[].content[]` with `type: image` |
| **Anthropic Complete** | `system` field | `prompt` string (legacy) | No native support | No |
| **Gemini** | `systemInstruction.parts` | `contents[]` | `tools[].functionDeclarations` | `contents[].parts[]` with `file_data` |
| **Cohere** | No direct field | `chat_history[]` | `tools[]` | No (business API) |
| **Mistral** | `messages[].content` (role: system) | `messages[]` | `tools[]` | `messages[].content[]` with `type: image_url` |
| **Hugging Face** | `messages[].content` (role: system) | `messages[]` | `tools[]` | API-dependent |

---

## Key Architectural Differences

### 1. System Prompt Location

- **Inside messages array:** OpenAI, Mistral, Hugging Face (uses `role: "system"`)
- **Separate field:** Anthropic (`system` field)
- **Special structure:** Gemini (`systemInstruction.parts`)

### 2. Messages Structure

- **Standard array:** OpenAI, Anthropic, Mistral (`messages[]` with `role` and `content`)
- **Role-pairs:** Gemini (`contents[]` with `role` and `parts[]`)
- **Split format:** Cohere (separate `message` + `chat_history[]`)

### 3. Tools Structure

All support `tools[]` but schema varies:

- **OpenAI/Mistral/HF:** `parameters` object with JSON Schema
- **Anthropic:** `input_schema` with JSON Schema
- **Gemini:** `parameters` with Google-specific type system (`OBJECT`, `STRING`, etc.)
- **Cohere:** `parameter_definitions` with custom format

### 4. Multi-Modal Content

- **OpenAI/Mistral:** `content` is array: `[{type: "text", text: "..."}, {type: "image_url", image_url: {...}}]`
- **Anthropic:** `content` is array: `[{type: "text", text: "..."}, {type: "image", source: {...}}]`
- **Gemini:** `parts` array: `[{text: "..."}, {file_data: {mime_type: "image/jpeg", data: "..."}}]`
- **Cohere:** No native multi-modal support in base API

---

## Design Implications

The `ChatBot` architecture must handle:

1. **Translation of fields:** `text` → `content`, `system` extraction, tools array merging
2. **Message format differences:** Array of messages vs. parts-based content
3. **System prompt placement:** Whether to extract from messages or set as separate field
4. **Tools schema variation:** Different parameter schema formats across APIs
5. **Streaming format differences:** SSE-like vs. event-based streaming

**Solution:** ChatHistory should contain the complete prompt representation (messages, system, tools, extra fields), and ChatBot implementations translate between this uniform representation and API-specific formats.
