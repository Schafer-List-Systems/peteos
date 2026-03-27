# Peteos Architecture

## Initial Architecture (2026-03-27)

### Overview
Project now includes Message, ChatHistory, ToolManager, ExecutionEnvironment, Session, Role, Agent, ChatBot, and REPLExecutionEnvironment classes.

### Structure
```
peteos/
├── __init__.py
├── agent.py
├── chatbot.py
├── message.py
├── chathistory.py
├── toolmanager.py
├── executionenvironment.py
├── replexecutionenvironment.py
├── session.py
├── role.py
```

### Components
- **Message**: Represents a single message with content and timestamp
- **ChatHistory**: Container for a list of Message objects

  - Methods: `get_content()` - returns list of message content dictionaries
  - Methods: `append_message(Message)` - appends message to history
- **Tool**: Represents a tool with properties

  - Attributes: `name` (str), `description` (str), `func` (Callable), `parameters` (dict)
  - Class Methods: `from_callable(Callable)` - creates Tool from callable, extracts name/description/parameters
  - Methods: `__call__(**kwargs)` - invokes the tool

- **ToolManager**: Manages tools for the execution environment

  - Contains: dictionary of registered tools
  - Methods: `register_tool(...)` - registers tools (accepts Tool instance or Callable with metadata)
  - Methods: `get_tool(name: str)` - retrieves tool by name
- **ChatBot**: Communicates with an LLM

  - Attributes: `url` (str), `model` (str)
  - Methods: `send_message(ChatHistory)` (async) - sends history, returns Message
  - Methods: `list_available_models()` - lists available models
  - Property: `model` - getter and setter for model identifier

- **ExecutionEnvironment**: Abstract base class for agent execution environments

  - Contains: `ChatHistory`, `ToolManager`, `ChatBot`, message queue
  - Methods: `queue_message(Message)` - adds message to queue, `get_chat_history()` - returns ChatHistory
  - Abstract Methods: `run()` - runs agentic loop until LLM responds with final answer or interrupted (must be overridden)

- **REPLExecutionEnvironment**: Concrete implementation of ExecutionEnvironment for REPL

  - Inherits from: `ExecutionEnvironment`
  - Methods: `run()` - REPL loop that reads input, processes via chatbot, appends output to ChatHistory

- **Session**: Container for execution environment

  - Contains: `Role`, `ExecutionEnvironment`, `ChatHistory`, `uuid` (UUID)
  - Creates `ChatHistory` locally if None provided in constructor
  - Generates `uuid` locally if None provided in constructor
  - Passes `ChatHistory` and `ChatBot` to `ExecutionEnvironment` constructor
  - Static Methods: `load_from_string(str)`, `load_from_file(str)` - UUID set from loaded data

- **Role**: Represents a role with identity and context

  - Attributes: `name` (str), `description` (str), `system_prompt` (Optional[str])
  - Static Methods: `load_from_dict(dict)` - loads from dict with name, description, system_prompt keys
  - Static Methods: `load_from_path(str)` - loads from directory with description.md and optional system_prompt.md

- **Agent**: Manages concurrent sessions

  - Contains: `ChatBot`, session dict, channel dict
  - Each session runs in its own thread
  - Manages channels to sessions
  - Passes `ChatBot` to sessions/executions

### Dependencies
- Python >= 3.10
- No runtime dependencies

### Class Diagram

```mermaid
classDiagram
    class Message {
        +dict content
        +datetime creation_timestamp
        +__init__(content: dict, creation_timestamp: datetime=None)
    }

    class ChatHistory {
        +List[Message] messages
        +__init__()
        +get_content() List[dict]
        +append_message(message: Message)
    }

    class Tool {
        +str name
        +str description
        +Callable func
        +dict parameters
        +from_callable(func: Callable) Tool
        +__call__(**kwargs) Any
    }

    class ToolManager {
        +Dict[str, Tool] _tools
        +__init__()
        +register_tool(tool: Tool=None, name: str=None, description: str=None, parameters: dict=None, func: Callable=None)
        +get_tool(name: str) Tool
    }

    class ChatBot {
        +str _url
        +str _model
        +__init__(url: str, model: str)
        +send_message(chat_history: ChatHistory) Message
        +list_available_models() list[str]
        +model str {get; set}
    }

    class ExecutionEnvironment {
        +ToolManager tool_manager
        +ChatHistory chat_history
        +ChatBot chatbot
        +deque _message_queue
        +__init__(chatbot: ChatBot, chat_history: ChatHistory, tool_manager: ToolManager)
        +queue_message(message: Message)
        +get_chat_history() ChatHistory
        +run() #abstract
    }

    class REPLExecutionEnvironment {
        +__init__(chatbot: ChatBot, chat_history: ChatHistory, tool_manager: ToolManager)
        +run()
    }

    class Session {
        +Role role
        +ExecutionEnvironment execution_environment
        +ChatHistory chat_history
        +uuid UUID
        +__init__(role: Role, tool_manager: ToolManager, chatbot: ChatBot, chat_history: ChatHistory=None, session_uuid: UUID=None)
        +load_from_string(data: str) Session
        +load_from_file(file_path: str) Session
    }

    class Role {
        +str name
        +str description
        +str system_prompt
        +__init__(name: str, description: str, system_prompt: str=None)
        +load_from_dict(data: dict) Role
        +load_from_path(path: str) Role
    }

    class Agent {
        +ChatBot _chatbot
        +Dict[str, Session] _sessions
        +Dict[str, object] _channels
        +__init__(chatbot: ChatBot)
    }

    ExecutionEnvironment <|-- REPLExecutionEnvironment
    ChatHistory --> Message : contains
    ExecutionEnvironment --> ChatHistory : has
    ExecutionEnvironment --> ToolManager : has
    ExecutionEnvironment --> ChatBot : has
    Session --> ExecutionEnvironment : has
    Session --> ChatHistory : has
    Session --> Role : has
    Session --> ChatBot : has
    Agent --> Session : manages
    Agent --> ChatBot : has
    ChatBot --> Message : returns
    ChatBot --> ChatHistory : accepts
```

### Architecture Diagram

```mermaid
graph TD
    A[peteos Package] --> B[__init__.py]
    A --> C[agent.py]
    A --> D[chatbot.py]
    A --> E[message.py]
    A --> F[chathistory.py]
    A --> G[toolmanager.py]
    A --> H[executionenvironment.py]
    A --> I[replexecutionenvironment.py]
    A --> J[session.py]
    A --> K[role.py]
    F -->|imports| E
    H -->|imports| F
    H -->|imports| G
    H -->|imports| D
    I -->|imports| H
    I -->|imports| F
    I -->|imports| G
    I -->|imports| D
    J -->|imports| H
    J -->|imports| G
    J -->|imports| D
    K -->|imports| H
    C -->|imports| D
    D -->|imports| F
    D -->|imports| E
    B -->|imports| C
    B -->|imports| D
    B -->|imports| E
    B -->|imports| F
    B -->|imports| G
    B -->|imports| H
    B -->|imports| I
    B -->|imports| J
    B -->|imports| K
```