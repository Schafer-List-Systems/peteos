# Chat History Design

**Status**: Decided - Ready for implementation

## Core Structure

### ChatHistory Class

```python
class ChatHistory:
    def __init__(
        self,
        messages: Optional[List[Message]] = None,
        generation_config: Optional[Dict[str, Any]] = None
    ):
        # All data stored as messages with specific roles
        self.messages = messages if messages else []
        
        # Generation config (model parameters, excludes backend-managed fields)
        self.generation_config = generation_config if generation_config else {}
```

### Message Class

```python
class Message:
    def __init__(
        self,
        role: str,  # "user", "assistant", "system", "tool", etc.
        content: List[ContentPart],  # Always a list
        metadata: Optional[Dict] = None
    ):
        self.role = role
        self.content = content
        self.metadata = metadata or {}
```

### ContentPart Class

```python
class ContentPart:
    def __init__(self, part_type: str, **data):
        self.type = part_type  # "text", "tool", "image", etc.
        self.data = data  # text: "hello", or tool: {...}, image: {...}
```

### Tool as Special Message

Tools are stored as messages with role="tool":

```python
tool_message = Message(
    role="tool",
    content=[ContentPart(
        type="tool",
        name="get_weather",
        description="Get weather for a city",
        parameters={"type": "object", "properties": {...}}
    )]
)
```

---

## Design Decisions

### 1. MessageContent is Always a List
- Makes code understanding easier
- Consistent regardless of content complexity
- Enables future multi-modal support

### 2. Tools as Special Messages (role="tool")
- Tools are messages with `role: "tool"`
- ContentPart stores tool data: `name`, `description`, `parameters` (JSON Schema)
- All content types are ContentParts with different types
- Translation layer extracts tools by filtering messages with role="tool"

### 3. System Prompt as Message (Option B)
- System prompt is a message with `role: "system"`
- Placed at position 0 in messages array
- Allows multiple system messages (future: fixed + sliding context)

### 4. ContentPart is Generic
- ContentPart accepts any data via `**kwargs`
- New content types just add new part types (image, video, audio, etc.)
- No special classes needed for new types

### 5. Response Format Constraints (Reserved)
- OpenAI: `response_format: {"type": "json_object"}`
- Gemini: `responseSchema` for JSON Schema output constraints
- Can add to `generation_config` later as `response_format`
- Keep in mind for future implementation

### 6. Excluded Fields (Backend-managed)
- `frequency_penalty`, `presence_penalty`
- These are handled by the LLM backend, not part of prompt

---

## API Translation Rules

### OpenAI
- **Messages**: Pass through as-is
- **System**: Extract `role: "system"` messages from array
- **Tools**: Extract messages with `role: "tool"`, flatten ContentPart to tool definition
- **Config**: `max_tokens`, `temperature`, `stop_sequences`, `response_format`
- **Tool Choice**: Pass through

### Anthropic Messages
- **Messages**: Pass through as-is (users/assistants)
- **System**: Concatenate all `role: "system"` messages into `system` field
- **Tools**: Extract `role: "tool"` messages, rename `parameters` → `input_schema`
- **Config**: `max_tokens`, `temperature`, `stop_sequences`

### Gemini
- **Messages**: Transform `messages[]` → `contents[]`, rename roles (assistant→model)
- **System**: Flatten into `systemInstruction.parts`
- **Content arrays**: Flatten into `parts[]`
- **Tools**: Extract `role: "tool"`, transform to `functionDeclarations`, convert JSON Schema → Google types
- **Config**: `maxOutputTokens`, `temperature`, `topP`

### Cohere
- **Messages**: Split into `message` (current) + `chat_history[]`
- **System**: Include in messages (no separate field)
- **Tools**: Extract `role: "tool"`, rename `parameters` → `parameter_definitions`
- **Config**: `temperature`, `p`, `stop_sequences`

---

## Future Considerations

1. **Multi-modal content**: Image/video/audio parts - just new ContentPart types
2. **Multiple system messages**: Fixed system + context-dependent system
3. **Response format constraints**: JSON mode, enum constraints
4. **Custom metadata**: Timestamps, sources, annotations
