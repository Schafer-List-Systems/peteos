# Agent and Channels Architecture

## Overview

The Agent serves as a central hub that connects multiple channels (interactive shell, REST API, etc.) to sessions. Channels provide different interfaces for users to interact with agents, while sessions manage the agentic loop with the underlying LLM.

## Architecture

```
                    ┌─────────────────────────────────────────────────────────┐
                    │                         Agent                          │
                    │  ┌───────────────┐  ┌───────────────┐  ┌──────────────┐│
                    │  │  _sessions    │  │   _channels  │  │_session_     ││
                    │  │  UUID→Session │  │  str→Channel │  │   channels   ││
                    │  └───────────────┘  └───────────────┘  └──────────────┘│
                    └──────────────────────────┬────────────────────────────┘
                                               │
                          ┌────────────────────┴────────────────────┐
                          │              hook system                │
                          │ (before_tool_execution, etc.)           │
                          └────────────────────┬────────────────────┘
                                               │
           ┌───────────────────────────────────┼───────────────────────────────────┐
           │                                   │                                   │
           ▼                                   ▼                                   ▼
    ┌──────────────┐                    ┌──────────────┐                    ┌──────────────┐
    │Channel 1     │                    │Channel 2     │                    │Channel N     │
    │(Shell)       │                    │(REST API)    │                    │(Future...)   │
    │              │                    │              │                    │              │
    │active: S1    │                    │active: S1    │                    │active: S2    │
    └──────────────┘                    └──────────────┘                    └──────────────┘
           │                                   │                                   │
           └───────────────────────────────────┴───────────────────────────────────┘
                                           │
                                           ▼
                                   ┌──────────────┐
                                   │   Session 1  │
                                   │   UUID: S1   │
                                   │              │
                                   │  role: test  │
                                   │ execution:   │
                                   │   REPL       │
                                   └──────────────┘
```

## Channel Types

### InteractiveShellChannel

A synchronous, blocking REPL-style interface for command-line interaction.

**Features:**
- Command parsing with `/` prefix
- Blocking `receive()` waits for user input
- Non-command lines forwarded as messages to active session
- Simple text-based output

**Commands:**
| Command | Description |
|---------|-------------|
| `/new <role>` | Create a new session with the specified role |
| `/list` | List all sessions with their UUIDs |
| `/select <uuid>` | Select a session as the active session for this channel |
| `/messages` | Show the last 10 messages from the active session |
| `/quit` | Exit the shell |

### RESTApiChannel

An asynchronous HTTP server providing REST endpoints and WebSocket for real-time updates.

**Endpoints:**

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/chat` | Send message to active session |
| `POST` | `/sessions` | Create new session |
| `GET` | `/sessions` | List all sessions |
| `POST` | `/sessions/{uuid}/select` | Select active session |
| `GET` | `/sessions/{uuid}/messages` | Get session message history |
| `WS` | `/ws/{session_uuid}` | WebSocket for real-time streaming |

**WebSocket Messages:**

Incoming:
```json
{"type": "chat", "content": "Hello"}
{"type": "subscribe", "session_uuid": "..."}
```

Outgoing:
```json
{"type": "message", "content": "[Agent] Tool called: get_weather with args: {...}"}
{"type": "message", "content": "[Agent] The weather in London is sunny."}
```

## Message Flow

### Incoming Messages (Channel → Agent → Session)

1. User types a message or command through a channel
2. For commands (`/` prefix), the channel handles them directly
3. For non-command input, the channel:
   - Retrieves the channel's active session
   - Creates a `Message` object with user content
   - Calls `session.queue_message(message)`
4. The session's `queue_message()` method adds the message to the execution environment

### Outgoing Messages (Session → Agent → Channels)

The Agent uses the ExecutionEnvironment hook system to intercept and forward messages:

1. **`before_tool_execution`** - Fired before each tool call
   - Agent logs: "Tool called: {name} with args: {args}"
   - Returns `(True, "")` to allow execution

2. **`after_tool_execution`** - Fired after tool completes
   - Agent logs: "Tool '{name}' {status}: {result}"

3. **`before_loop_continue`** - Fired when loop continues
   - Agent extracts delta messages (tool results)
   - Forwards tool result messages to all subscribed channels

4. **`before_loop_exit`** - Fired when loop exits
   - Agent extracts the final answer from ChatHistory
   - Forwards the response to all subscribed channels

## Hook System Integration

The Agent registers hook callbacks on each session's ExecutionEnvironment when the session is created:

```python
def create_session(self, role_name: str) -> Session:
    session = Session(...)
    self._sessions[session.uuid] = session
    self._session_channels[session.uuid] = set()

    # Register hooks
    env = session.execution_environment
    env.register_hook("before_tool_execution",
                      self._on_before_tool_execution, session.uuid)
    env.register_hook("after_tool_execution",
                      self._on_after_tool_execution, session.uuid)
    env.register_hook("before_loop_continue",
                      self._on_before_loop_continue, session.uuid)
    env.register_hook("before_loop_exit",
                      self._on_before_loop_exit, session.uuid)

    return session
```

### Hook Callback Signatures

```python
# before_tool_execution: returns (allow: bool, message: str)
def _on_before_tool_execution(self, session_uuid: UUID, tool_call: dict) -> tuple:

# after_tool_execution: no return value
def _on_after_tool_execution(self, session_uuid: UUID, tool_call: dict, 
                              result: str, success: bool) -> None:

# before_loop_continue: returns (should_exit: bool, reason: str)
def _on_before_loop_continue(self, session_uuid: UUID, delta_messages: List[Message]) -> None:

# before_loop_exit: returns None
def _on_before_loop_exit(self, session_uuid: UUID, reason: str) -> None:
```

## Channel Subscriptions

When a session is created, the Agent initializes an empty set of subscribed channels:

```python
self._session_channels[session.uuid] = set()
```

Channels subscribe to sessions when they select a session as active:

```python
def select_session(self, session_uuid: UUID) -> None:
    self._active_session_uuid = session_uuid
    # The Agent maintains session_channels mapping
```

The `_notify_channels` method forwards messages to all subscribed channels:

```python
def _notify_channels(self, session_uuid: UUID, message: str) -> None:
    channels = self._session_channels.get(session_uuid, set())
    for channel in channels:
        channel.send(message)
```

## Multi-Session Support

Multiple channels can have different active sessions:

```
Channel A (Shell)   -> Session 1
Channel B (REST)    -> Session 2
Channel C (REST)    -> Session 1
```

In this case:
- Session 1 sends messages to Channel A and Channel C
- Session 2 sends messages to Channel B
- Both channels maintain independent state

## Example Usage

### Shell Channel

```python
from peteos.agent import Agent
from peteos.rolemanager import RoleManager
from peteos.chatbot.manager import ChatBotManager
from peteos.toolmanager import ToolManager
from peteos.channels import InteractiveShellChannel

# Setup
role_manager = RoleManager()
chatbot_manager = ChatBotManager()
tool_manager = ToolManager()

agent = Agent(role_manager, chatbot_manager, tool_manager)
shell = InteractiveShellChannel("shell", agent)

# Start shell
shell.run()
```

### REST API Channel

```python
from peteos.channels import RESTApiChannel

rest = RESTApiChannel("rest", agent, host="0.0.0.0", port=8080)
server_url = await rest.start()
print(f"REST API available at: {server_url}")

# ... use the API ...

await rest.stop()
```

## Testing

Run the existing test suite:

```bash
pytest tests/ -v
```

The channel implementation relies on the existing hook tests in `tests/unit/test_hooks_simple.py` to verify the hook mechanism works correctly.
