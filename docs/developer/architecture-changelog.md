## [2026-03-30] - Session ChatBotManager Integration
- Added `model` field to `Role`: regex pattern for ChatBot selection (default: `".*"`)
- Updated `ExecutionEnvironment` to accept `ChatBotManager` and `Role` instead of `ChatBot`
- Added `_select_chatbot()` method: selects ChatBot using `role.model` regex pattern
- Updated `REPLExecutionEnvironment` constructor to accept `chatbot_manager` and `role`
- Updated `Session` to store `ChatBotManager` and pass it with `role` to execution environment
- Changed mock model name from `"qwen"` to `"mock_model"` in tests for security
- Updated `architecture.md`: Role, ExecutionEnvironment, and Session documentation

## [2026-03-30] - Documentation Updates
- Updated `execution-environment.md`: Fixed REPL loop behavior to match current code (stores full `response.data` dict, not separate fields)
- Updated `chatbot.md`: Fixed Anthropic API behavior, added request/response format differences, removed sensitive IP addresses
- Updated `architecture.md`: Added all class documentation, utility functions, and implementation details
- Created `tests.md`: New documentation for testing infrastructure, test coverage, and writing new tests
- Added "Key Implementation Details" section to architecture.md: streaming behavior, error handling, thread safety, message format evolution

## [2026-03-29] - ChatBotResponse Refactoring
- Converted `GenericChatBotResponse` from async iterator to async generator
- `__anext__` now yields all **(key, chunk)** pairs per SSE event (not just first field)
- Supports multi-field events (e.g., `message_start` with both text and reasoning)
- `response.data` contains all accumulated fields: `text`, `reasoning`, `tool_calls`
- `_build_body()` now passes full message dict (supports multi-part content)
- Updated chatbot.md with async generator behavior and yield examples
- Renamed response keys: `text_content` -> `text`, `thinking_content` -> `reasoning`
- All 51 tests pass

## [2026-03-27] - GenericChatBot Refactoring
- Added `GenericChatBot` class: configurable base with endpoints and response translation
- Added `GenericChatBotResponse` class: path-based event translation
- Added `get_value_at_path()` utility: supports wildcards, indexing, and type discriminators
- `OpenAIChatBot` and `AnthropicChatBot` now inherit from `GenericChatBot`
- Response translations configurable via path notation (e.g., `"choices[*].delta.content"`)
- Type discriminator support: paths like `"content_block_start.content_block.text"` match `type` field then navigate
- Updated chatbot.md documentation with new architecture
- Added `.env.example` and updated examples usage documentation
- All 32 tests pass

## [2026-03-27] - ChatBot Documentation and Examples
- Created dedicated ChatBot documentation in `docs/developer/chatbot.md`
- Fixed `AnthropicChatBot` non-streaming bug (stream was hardcoded to True)
- Added `test_anthropic_chatbot_non_streaming_with_mock` E2E test
- Removed commented-out real endpoint tests
- Created examples directory with `openai_chatbot.py` and `anthropic_chatbot.py`
- Clarified API protocol vs. model distinction in documentation

## [2026-03-27] - Anthropic API Support
- Updated `AnthropicChatBotResponse` to support Anthropic-compatible format
- Handles `thinking`/`thinking_delta` keys (Anthropic-compatible) in addition to `reasoning`/`reasoning_delta`
- Response translation: `thinking` → `reasoning`, `text` → `text`
- Both OpenAI and Anthropic endpoints now supported on the same host/port
- Added test: `test_anthropic_compatible_thinking`

## [2026-03-27] - ChatBot Refactoring with Response Wrappers
- Made `ChatBot` an abstract base class
- Created `HTTPClient` class for async HTTP communication with SSE streaming
- Created `ChatBotResponse` abstract base class for streaming responses
- Created `OpenAIChatBotResponse` and `AnthropicChatBotResponse` implementations
- Response classes translate API-specific schemas to common format:
  - OpenAI/Qwen: `reasoning`/`thinking` → `reasoning`, `content` → `text`
  - Anthropic: `reasoning`/`reasoning_delta` → `reasoning`, `thinking`/`thinking_delta` → `reasoning`, `text`/`text_delta` → `text`
- `send_message()` supports streaming mode (default) and non-streaming mode via `streaming: bool = True`
- `ChatBotResponse` is async iterable, yields accumulated text as it arrives
- HTTPClient can be mocked for testing
- Updated architecture documentation with class and component diagrams

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
