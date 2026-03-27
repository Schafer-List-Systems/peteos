# Architecture Changelog

## [2026-03-27] - REPLExecutionEnvironment Class
- Added `REPLExecutionEnvironment` class: concrete implementation of ExecutionEnvironment
- Implements `run()` method with REPL logic (empty placeholder)
- Appends LLM output to ChatHistory instead of printing
- Updated architecture documentation

## [2026-03-27] - ChatBot Integration
- Made `ChatBot` obligatory for `ExecutionEnvironment`, `Session`, and `Agent`
- `ChatBot` is first parameter in constructors
- Updated architecture documentation

## [2026-03-27] - ChatBot Class
- Added `ChatBot` class: communicates with an LLM via URL and model
- Async `send_message(ChatHistory)` method: sends history, returns Message
- `list_available_models()` method: lists available models
- `model` property with getter and setter
- Updated architecture documentation

## [2026-03-27] - ChatHistory Methods
- Added `get_content()`: returns list of all message content dictionaries
- Added `append_message(Message)`: appends message to history
- Updated architecture documentation

## [2026-03-27] - Session UUID
- Added `uuid` attribute to Session: generated during construction
- `load_from_string` and `load_from_file` set UUID from loaded data
- Updated architecture documentation

## [2026-03-27] - Agent Class
- Added `Agent` class: manages concurrent sessions with per-session threads and channels
- Updated architecture documentation

## [2026-03-27] - Session.obligatory Role
- Made `Role` an obligatory constructor parameter for Session
- Updated architecture documentation

## [2026-03-27] - Role Implementation
- Implemented `Role.load_from_dict(dict)`: loads from dict with name, description, system_prompt keys
- Implemented `Role.load_from_path(str)`: loads from directory with description.md and optional system_prompt.md
- Name derived from path suffix
- Updated architecture documentation

## [2026-03-27] - Role Class
- Added `Role` class: represents a role with name, description, and optional system prompt
- Added static methods `load_from_string()` and `load_from_file()`
- Updated architecture documentation

## [2026-03-27] - Session Class
- Added `Session` class: container for ExecutionEnvironment
- Session has obligatory `ToolManager` and optional `ChatHistory`
- Creates `ChatHistory` locally if None provided
- Passes `ChatHistory` to `ExecutionEnvironment` constructor
- Updated architecture documentation with class and component diagrams

## [2026-03-27] - ExecutionEnvironment Constructor
- Made `ChatHistory` obligatory constructor parameter (no longer auto-created)
- Message queue created locally in constructor
- Updated architecture documentation

## [2026-03-27] - ToolManager Implementation
- Added `Tool` class: represents tools with name, description, func, parameters
- Implemented `Tool.from_callable()`: extracts name, description (from docstring), parameters from callable
- Implemented `ToolManager.register_tool()`: registers tools (accepts Tool instance or Callable)
- Implemented `ToolManager.get_tool()`: retrieves tool by name
- Updated architecture documentation

## [2026-03-27] - ExecutionEnvironment.run() Abstract Method
- Added `ToolManager` class: placeholder for tool management
- Added `ExecutionEnvironment` class: contains obligatory ToolManager and creates ChatHistory in constructor
- Updated architecture documentation with component diagram

## [2026-03-27] - Message and ChatHistory Classes
- Added `Message` class: holds content (dict) and creation_timestamp
- Added `ChatHistory` class: container for list of Messages
- Updated architecture documentation with component diagram

## [2026-03-27] - Initial Empty Project
- Created bare project structure
- No framework components implemented yet
- Documentation structure established

## [2026-03-27] - New Session:
- session id: 810fb120-e4e5-4e32-9718-88bbcaf7641a
- model: qwen3.5-35b-a3b
