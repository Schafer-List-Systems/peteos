# Execution Environment

## Overview

The Execution Environment provides the runtime context for agent interactions. It encapsulates the agentic loop that sends chat history to the LLM, processes responses, handles tool calls, and maintains conversation state.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Agent                                │
│              (manages concurrent sessions)                   │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                        Session                              │
│  ┌──────────────┐  ┌──────────────────┐  ┌──────────────┐ │
│  │     Role     │  │  ChatHistory     │  │    UUID      │ │
│  └──────────────┘  └──────────────────┘  └──────────────┘ │
│                           │                                 │
│              ┌────────────▼────────────┐                   │
│              │  ExecutionEnvironment  │                   │
│              │  ┌──────────────────┐  │                   │
│              │  │    ChatBot       │  │                   │
│              │  ├──────────────────┤  │                   │
│              │  │  ToolManager     │  │                   │
│              │  ├──────────────────┤  │                   │
│              │  │  MessageQueue    │  │                   │
│              │  │  InterruptFlag   │  │                   │
│              │  └──────────────────┘  │                   │
│              └─────────────────────────┘                   │
└─────────────────────────────────────────────────────────────┘
```

## Components

### ChatBot

Communicates with the LLM API. Handles:
- Building request bodies from chat history
- Sending requests (streaming or non-streaming)
- Returning `ChatBotResponse` objects

**Key Methods:**
- `send_message(chat_history, streaming=True)` - sends request, returns response
- `list_available_models()` - lists available models

### ChatBotResponse

Streams LLM responses with support for:
- Server-Sent Events (SSE) parsing
- Accumulating all response fields into `response.data` dict
- Path-based translation of API-specific schemas
- Async iteration yielding `(key, chunk)` tuples

**Key Properties:**
- `response.data["text"]` - accumulated response text
- `response.data["reasoning"]` - accumulated reasoning/thinking content
- `response.data["tool_calls"]` - list of tool call dicts (pre-parsed)

### ToolManager

Registers and executes tools (functions) available to the agent:
- `register_tool(tool)` - register a `Tool` instance or callable
- `get_tool(name)` - retrieve tool by name
- `execute(**args)` - invoke tool with arguments

### ChatHistory

Container for conversation messages:
- `append_message(message)` - add message
- `get_content()` - extract message contents for API

## REPLExecutionEnvironment

The `REPLExecutionEnvironment` implements the core agentic loop. It's a "Read-Eval-Print Loop" for agent interactions with tool use support.

### Design Principles

1. **Streaming-First**: Processes LLM responses as they stream in
2. **Tool-Use Support**: Automatically detects and executes tool calls
3. **Stateful**: Maintains conversation history throughout the loop
4. **Interruptible**: Can be terminated gracefully from other threads

### The REPL Loop

```python
async def run(self) -> None:
    """Main agentic loop."""
    while not self._interrupt:
        # 1. Send chat history to LLM
        response = await self.chatbot.send_message(
            self.chat_history,
            streaming=True
        )

        # 2. Collect accumulated response
        # Yields (key, chunk) tuples: ("text", "Hello"), ("reasoning", "Thinking...")
        async for key, chunk in response:
            pass  # Streaming iteration

        # 3. Check for interrupt
        if self._interrupt:
            break

        # 4. Extract content from response.data dict
        response_data = response.data
        reasoning = response_data.get("reasoning", "")
        text = response_data.get("text", "")
        tool_calls = response_data.get("tool_calls")  # Pre-parsed list

        # 5. Append reasoning as separate field (if present)
        if reasoning:
            self.chat_history.append_message(
                Message(content={
                    "role": "assistant",
                    "reasoning": reasoning
                })
            )

        # 6. Execute tool calls (already parsed from API response)
        if tool_calls:
            # 7a. Execute each tool call
            for tool_call in tool_calls:
                tool = self.tool_manager.get_tool(tool_call["name"])
                result = tool.execute(**tool_call["arguments"])
                self.chat_history.append_message(
                    Message(content={
                        "role": "tool",
                        "name": tool_call["name"],
                        "content": str(result)
                    })
                )
            # Loop continues with updated history
        else:
            # 7b. Final answer - append and exit
            self.chat_history.append_message(
                Message(content={
                    "role": "assistant",
                    "content": text
                })
            )
            break
```

### Tool Call Format

Tool calls come pre-parsed in `response.data["tool_calls"]` as a list of dicts:

```python
tool_calls = response.data.get("tool_calls")
# [
#   {"name": "get_weather", "arguments": {"city": "London"}},
#   {"name": "search", "arguments": {"query": "weather"}}
# ]
```

The API is responsible for returning tool calls in this structured format. The response parser extracts tool_calls from the JSON response and makes them available directly.

### Interrupt Control

Thread-safe interrupt mechanism for graceful shutdown:

```python
# Request termination from any thread
env.set_interrupt()

# Check and clear
env.clear_interrupt()
```

**Usage Pattern:**
```python
async def run_with_timeout(env, timeout_seconds):
    import asyncio

    # Set timeout interrupt
    async def timeout_handler():
        await asyncio.sleep(timeout_seconds)
        env.set_interrupt()

    await asyncio.gather(
        env.run(),
        timeout_handler()
    )
```

## Message Flow

```
User Input
    │
    ▼
┌─────────────────┐
│  ChatHistory    │ (user message appended)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   ChatBot       │ (send_message)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ ChatBotResponse │ (streaming)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ REPLExecution   │
│  Environment    │
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
Thinking    Tool Call?
Content     │
    │       ▼
    │    Yes ────┐    No ────┐
    │             │           │
    │             ▼           ▼
    │      ┌──────────┐  ┌──────────┐
    │      │ Execute  │  │Append to │
    │      │ Tool     │  │History   │
    │      │ Result   │  │& Exit    │
    │      └────┬─────┘  └──────────┘
    │           │
    └───────────┼──────────┐
                ▼          │
         ┌──────────┐      │
         │Append to │      │
         │History   │      │
         └────┬─────┘      │
              │            │
              └──── Loop ──┘
```

## Usage Example

```python
from peteos.chatbot import OpenAIChatBot
from peteos.replexecutionenvironment import REPLExecutionEnvironment
from peteos.toolmanager import ToolManager
from peteos.chathistory import ChatHistory
from peteos.httpclient import HTTPClient
from peteos.role import Role

# Setup
http_client = HTTPClient(timeout=60.0)
chatbot = OpenAIChatBot(
    http_client=http_client,
    model="qwen3.5-35b",
    base_url="http://localhost:8000"
)
tool_manager = ToolManager()
chat_history = ChatHistory()

# Register tools
@tool_manager.register_tool
def get_weather(city: str) -> str:
    """Get weather for a city."""
    return f"Sunny in {city}"

# Create execution environment
env = REPLExecutionEnvironment(
    chatbot=chatbot,
    chat_history=chat_history,
    tool_manager=tool_manager
)

# Run the agentic loop
async def main():
    await env.run()

asyncio.run(main())
```

## Testing

Unit tests for `REPLExecutionEnvironment` cover:
- Basic conversation flow
- Tool call detection and execution
- Loop continuation on tool calls
- Loop termination on final answer
- Interrupt handling
- Thinking content accumulation

Run tests:
```bash
pytest tests/unit/test_replexecutionenvironment.py -v
```

## Design Rationale

### Why Stream Processing?

Streaming allows:
- Real-time feedback to users
- Early response parsing
- Lower latency perception
- Ability to interrupt mid-stream

### Why Pre-Parsed Tool Calls?

Tool calls come pre-parsed in `response.data["tool_calls"]`:
- API response parsing handles JSON extraction
- REPL doesn't need to parse text for embedded JSON
- Consistent format across all responses
- Eliminates need for `_parse_tool_calls_from_text()`

### Why Accumulate Response First?

Accumulating `ChatBotResponse` before processing:
- All fields available in `response.data` dict
- Supports multi-field events (reasoning + tool_calls)
- Clean separation between streaming and processing
- Works with both streaming and non-streaming modes

### Why Separate Thinking Content?

Separating thinking from response text:
- Enables transparent reasoning display
- Useful for debugging/auditing
- Allows different handling of reasoning vs. answers
- Supports models with explicit thinking phases
